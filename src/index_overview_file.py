"""
Index Overview File Generator
============================

Creates comprehensive overview files summarizing market indices and key metrics.
Analyzes market breadth, sector performance, and index relationships.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from .overview_price_mas import generate_overview_template_part1, get_price_ma_summary
from .tornado_chart import generate_tornado_charts

logger = logging.getLogger(__name__)


def _extract_data_date_from_dataframe(df: pd.DataFrame) -> str:
    """
    Extract data date from DataFrame.
    
    Args:
        df: DataFrame with date column
        
    Returns:
        Date string in YYYYMMDD format from the data, not file generation
    """
    try:
        if df is not None and not df.empty and 'date' in df.columns:
            # Get the date from the DataFrame data
            data_date = df['date'].iloc[0]
            if isinstance(data_date, str):
                # Handle string dates like '2025-09-05'
                return data_date.replace('-', '')
            else:
                # Handle pandas Timestamp
                return data_date.strftime('%Y%m%d')
        
        # Fallback to current date if no data found
        logger.warning("No date column found in index overview DataFrame, using file generation date as fallback")
        return datetime.now().strftime('%Y%m%d')
        
    except Exception as e:
        logger.error(f"Error extracting data date from index overview DataFrame: {e}")
        return datetime.now().strftime('%Y%m%d')


def generate_tornado_chart_from_analysis(percentage_file, output_path, timeframe='daily'):
    """
    Generate tornado chart directly from index percentage analysis CSV file.
    
    Args:
        percentage_file: Path to percentage analysis CSV file
        output_path: Directory to save the tornado chart
        
    Returns:
        str: Path to generated tornado chart file
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for server environment
        import matplotlib.pyplot as plt
        
        # Load percentage analysis data
        df = pd.read_csv(percentage_file)
        
        # Identify available indexes
        available_indexes = []
        for col in df.columns:
            if col.endswith('_Above%'):
                index_name = col.replace('_Above%', '')
                if f'{index_name}_Below%' in df.columns:
                    available_indexes.append(index_name)
        
        if not available_indexes:
            logger.warning("No valid index data found for tornado chart")
            return None
        
        # Filter out rows with all zero values
        mask = pd.Series([False] * len(df))
        for index_name in available_indexes:
            above_col = f'{index_name}_Above%'
            below_col = f'{index_name}_Below%'
            mask |= (df[above_col] > 0) | (df[below_col] > 0)
        
        chart_df = df[mask].copy()
        
        if chart_df.empty:
            logger.warning("No non-zero data for tornado chart")
            return None
        
        # Preserve original table order - do not sort
        # Original code sorted by total activity but this changes the order from the source table
        # Commenting out the sort to maintain the original order from the CSV file
        
        metrics = chart_df['Metric'].tolist()
        y = np.arange(len(metrics))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(metrics) * 0.6)))
        
        # Colors: colorblind-safe palette
        colors = {
            'SP500': {'above': "#377eb8", 'below': "#e41a1c"},      # Blue/Red
            'NASDAQ100': {'above': "#4daf4a", 'below': "#ff7f00"}, # Green/Orange
            'Russell1000': {'above': "#984ea3", 'below': "#ffff33"}, # Purple/Yellow
            'Russell2000': {'above': "#a65628", 'below': "#f781bf"}  # Brown/Pink
        }
        
        # Calculate bar positioning
        bar_height = 0.35 / len(available_indexes)
        offsets = np.linspace(-0.15, 0.15, len(available_indexes))
        
        # Create bars for each index
        for i, index_name in enumerate(available_indexes):
            above_col = f'{index_name}_Above%'
            below_col = f'{index_name}_Below%'
            
            above_color = colors.get(index_name, {'above': '#1f77b4'})['above']
            below_color = colors.get(index_name, {'below': '#d62728'})['below']
            
            # Plot bars
            ax.barh(y + offsets[i], chart_df[above_col], height=bar_height, 
                   color=above_color, label=f'{index_name} Above%', alpha=0.85, 
                   edgecolor='black', linewidth=0.3)
            ax.barh(y + offsets[i], -chart_df[below_col], height=bar_height, 
                   color=below_color, label=f'{index_name} Below%', alpha=0.85,
                   edgecolor='black', linewidth=0.3)
            
            # Add value labels for significant values
            for j in range(len(metrics)):
                above_val = chart_df[above_col].iloc[j]
                below_val = chart_df[below_col].iloc[j]
                
                if above_val > 3:  # Only show labels for values > 3%
                    ax.text(above_val + 1, y[j] + offsets[i], f"{above_val:.0f}%", 
                           va='center', ha='left', fontsize=8, color=above_color, fontweight='bold')
                if below_val > 3:
                    ax.text(-below_val - 1, y[j] + offsets[i], f"{below_val:.0f}%", 
                           va='center', ha='right', fontsize=8, color=below_color, fontweight='bold')
        
        # Format chart
        ax.set_yticks(y)
        ax.set_yticklabels(metrics, fontsize=10)
        ax.axvline(0, color='black', linewidth=1.2)
        ax.legend(loc='lower right', fontsize=9, frameon=True, ncol=len(available_indexes))
        ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
        
        # Dynamic title
        title = f"{' vs '.join(available_indexes)}: Market Breadth Analysis (Tornado Chart)"
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Add grid and formatting
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        ax.invert_yaxis()  # Most significant metrics at top
        
        plt.tight_layout()
        
        # Use source filename pattern: original_name_tornado.png
        source_path = Path(percentage_file)
        base_name = source_path.stem  # Gets filename without extension
        chart_file = output_path / f"{base_name}_tornado.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()  # Close figure to free memory
        
        logger.info(f"Tornado chart saved: {chart_file}")
        return str(chart_file)
        
    except Exception as e:
        logger.error(f"Error creating tornado chart from analysis: {e}")
        return None


def identify_index_tickers(batch_data, user_config=None, config=None):
    """Identify which tickers are market indices or specific classifications based on configuration."""
    
    # If we have SP500 configuration and basic calculation results available, use them
    if (user_config and hasattr(user_config, 'index_overview_indexes') and 
        user_config.index_overview_indexes.strip().upper() == 'SP500' and config):
        
        return get_sp500_tickers_from_calculations(batch_data, config, user_config)
    
    # Fall back to pattern-based identification (original logic)
    # Default index patterns (fallback)
    default_patterns = [
        '^', 'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO',
        'XL', 'IY', 'VGT', 'VHT', 'VFH', 'VNQ', 'VAW', 'VDC',
        'VIS', 'VOX', 'VPU', 'ARKK', 'SOXX', 'SMH', 'IBB', 'XBI'
    ]
    
    # Build comprehensive ticker lists from config
    index_patterns = []
    etf_patterns = []
    stock_patterns = []
    
    if user_config:
        # Add configured indexes
        if hasattr(user_config, 'index_overview_indexes') and user_config.index_overview_indexes.strip():
            config_indexes = [idx.strip() for idx in user_config.index_overview_indexes.split(',') if idx.strip()]
            index_patterns.extend(config_indexes)
            
        # Add configured sectors (treated as index patterns for broad categories)
        if hasattr(user_config, 'index_overview_sectors') and user_config.index_overview_sectors.strip():
            config_sectors = [sec.strip() for sec in user_config.index_overview_sectors.split(',') if sec.strip()]
            index_patterns.extend(config_sectors)
            
        # Add configured ETFs
        if hasattr(user_config, 'index_overview_etfs') and user_config.index_overview_etfs.strip():
            config_etfs = [etf.strip() for etf in user_config.index_overview_etfs.split(',') if etf.strip()]
            etf_patterns.extend(config_etfs)
            
        # Add configured stocks
        if hasattr(user_config, 'index_overview_stocks') and user_config.index_overview_stocks.strip():
            config_stocks = [stock.strip() for stock in user_config.index_overview_stocks.split(',') if stock.strip()]
            stock_patterns.extend(config_stocks)
    
    # Use defaults if no config provided
    if not index_patterns:
        index_patterns = default_patterns
    else:
        index_patterns.append('^')  # Always include ^ patterns for market indices
    
    indices = {}
    stocks = {}
    
    for ticker, df in batch_data.items():
        # Check exact match first, then pattern matching
        is_index = (
            ticker in index_patterns or 
            ticker in etf_patterns or
            any(ticker.startswith(pattern) or pattern in ticker for pattern in index_patterns)
        )
        
        # Force specific stocks to be categorized as stocks even if they match patterns
        if ticker in stock_patterns:
            is_index = False
        
        if is_index:
            indices[ticker] = df
        else:
            stocks[ticker] = df
            
    return indices, stocks


def get_sp500_tickers_from_calculations(batch_data, config, user_config):
    """
    Get S&P 500 tickers using boolean classifications from basic calculation results.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        config: Config object with directory paths
        user_config: User configuration object
        
    Returns:
        tuple: (indices, sp500_stocks) where indices is empty and sp500_stocks contains SP500 members
    """
    indices = {}  # No traditional indices in this mode
    sp500_stocks = {}
    
    try:
        # Load basic calculation results that contain boolean classifications
        from .basic_calculations import find_latest_basic_calculation_file
        calc_file = find_latest_basic_calculation_file(config, 'daily', user_config.ticker_choice)
        
        if not calc_file:
            logger.warning(f"Basic calculation file not found for daily timeframe")
            logger.warning("Falling back to pattern-based identification")
            return identify_index_tickers(batch_data, user_config, None)  # Fall back without config to use patterns
            
        # Read basic calculations with boolean classifications
        calc_df = pd.read_csv(calc_file)
        
        if 'SP500' not in calc_df.columns:
            logger.warning("SP500 boolean column not found in basic calculations")
            logger.warning("Falling back to pattern-based identification")
            return identify_index_tickers(batch_data, user_config, None)  # Fall back
            
        # Debug info
        logger.info(f"Basic calculation file has {len(calc_df)} rows and columns: {list(calc_df.columns)[:10]}...")
        sp500_column_sample = calc_df['SP500'].dropna().head(10)
        logger.info(f"SP500 column sample values: {sp500_column_sample.tolist()}")
        logger.info(f"SP500 column data type: {calc_df['SP500'].dtype}")
            
        # Filter for S&P 500 members (handle various boolean formats)
        # SP500 column might be boolean True/False or string "True"/"False"
        sp500_mask = (calc_df['SP500'] == True) | (calc_df['SP500'] == 'True') | (calc_df['SP500'] == 1)
        sp500_tickers = calc_df[sp500_mask]['ticker'].tolist()
        
        # More debug info
        logger.info(f"SP500 mask found {sp500_mask.sum()} matching records out of {len(calc_df)} total")
        
        # Build sp500_stocks dictionary with available data
        # For SP500 mode, we primarily need the calculation data, not necessarily all market data
        for ticker in sp500_tickers:
            if ticker in batch_data:
                sp500_stocks[ticker] = batch_data[ticker]
            else:
                # Even if market data isn't available, we still want to include this ticker
                # Create a minimal DataFrame to represent the ticker
                import pandas as pd
                sp500_stocks[ticker] = pd.DataFrame()
                
        logger.info(f"Found {len(sp500_tickers)} S&P 500 tickers in calculations, {len(sp500_stocks)} included ({len([k for k, v in sp500_stocks.items() if not v.empty])} with market data)")
        
    except Exception as e:
        logger.error(f"Error loading S&P 500 data from calculations: {e}")
        logger.warning("Falling back to pattern-based identification")
        return identify_index_tickers(batch_data, user_config, None)  # Fall back
        
    return indices, sp500_stocks


def generate_index_percentage_analysis(calc_df, output_path, timeframe, user_config, config=None):
    """
    Generate comprehensive index percentage analysis table with transposed format.
    Variables become rows, indexes become columns (Above/Below for each).
    
    Args:
        calc_df: DataFrame with basic calculation results
        output_path: Path to save the table
        timeframe: Data timeframe  
        user_config: User configuration
        
    Returns:
        str: Path to generated percentage analysis table
    """
    try:
        # Parse configured indexes from user settings
        indexes_str = user_config.index_overview_indexes.strip() if user_config.index_overview_indexes else 'SP500'
        index_names = [idx.strip().upper() for idx in indexes_str.split(',') if idx.strip()]
        
        # Define metrics with Above/Below logic (4-column format)
        metrics_definitions = [
            # Change metrics - each metric has Above% and Below%
            ('Day Change (%)', '1-day % change: (Close_t - Close_t-1)/Close_t-1 * 100', 'daily_pct_change', lambda x: x > 0, lambda x: x < 0),
            ('Month Change (%)', '1-month: (Close_t - Close_t-22)/Close_t-22 * 100', 'daily_monthly_22d_pct_change', lambda x: x > 0, lambda x: x < 0),
            ('Quarter Change (%)', '1-quarter: (Close_t - Close_t-66)/Close_t-66 * 100', 'daily_monthly_66d_pct_change', lambda x: x > 0, lambda x: x < 0),
            ('Half Year Change (%)', '6-month: (Close_t - Close_t-120)/Close_t-120 * 100', 'daily_half_year_pct_change', lambda x: x > 0, lambda x: x < 0),
            
            # Price vs MA metrics
            ('Price vs SMA 10', 'Price compared to 10-day Simple Moving Average', 'priceabovesma10', lambda x: x == True, lambda x: x == False),
            ('Price vs EMA 20', 'Price compared to 20-day Exponential Moving Average', 'priceaboveema20', lambda x: x == True, lambda x: x == False),
            ('Price vs SMA 50', 'Price compared to 50-day Simple Moving Average', 'priceabovesma50', lambda x: x == True, lambda x: x == False),
            
            # MA Comparison metrics
            ('SMA 10 vs SMA 20', 'Short-term MA compared to medium-term MA', 'sma10vssma20', lambda x: x == True, lambda x: x == False),
            ('SMA 20 vs SMA 50', 'Medium-term MA compared to long-term MA', 'sma20vssma50', lambda x: x == True, lambda x: x == False),
            ('SMA 50 vs SMA 200', 'Long-term MA compared to very long-term MA', 'sma50vssma200', lambda x: x == True, lambda x: x == False),
            
            # Trend Analysis
            ('Trend Strength', 'Percentage of up days in last 10 trading days', 'trend_days_10_pct', lambda x: x > 60, lambda x: x < 40),
            ('Perfect Bull Alignment', 'SMA 20 > SMA 50 > SMA 200 alignment', 'perfectbullishalignment', lambda x: x == True, lambda x: x == False),
            
            # High/Low Analysis
            ('20-Day Position', 'Current price vs 20-day high/low range', 'at_20day_high', lambda x: x == True, None),
            ('20-Day Low Position', 'Current price vs 20-day low', 'at_20day_low', lambda x: x == True, None),
            
            # RSI Analysis
            ('RSI Momentum', 'RSI overbought vs oversold conditions', 'rsi_14', lambda x: x > 70, lambda x: x < 30),
        ]
        
        # Create analysis table with multiple index columns
        analysis_rows = []
        
        for metric_name, calculation_logic, column_name, above_condition, below_condition in metrics_definitions:
            row_data = {
                'Metric': metric_name,
                'Calculation Logic': calculation_logic
            }
            
            # Calculate for each configured index
            for index_name in index_names:
                if index_name in calc_df.columns:
                    # Filter to this index members
                    index_mask = (calc_df[index_name] == True) | (calc_df[index_name] == 'True') | (calc_df[index_name] == 1)
                    index_df = calc_df[index_mask].copy()
                    total_members = len(index_df)
                    
                    if total_members > 0 and column_name in index_df.columns:
                        # Calculate Above%
                        above_count = index_df[column_name].apply(above_condition).sum() if above_condition else 0
                        above_pct = (above_count / total_members) * 100
                        
                        # Calculate Below%
                        below_count = index_df[column_name].apply(below_condition).sum() if below_condition else 0
                        below_pct = (below_count / total_members) * 100
                        
                        row_data[f'{index_name}_Above%'] = above_pct
                        row_data[f'{index_name}_Below%'] = below_pct
                    else:
                        row_data[f'{index_name}_Above%'] = 0.0
                        row_data[f'{index_name}_Below%'] = 0.0
                        
            analysis_rows.append(row_data)
        
        # Create DataFrame
        analysis_df = pd.DataFrame(analysis_rows)
        
        # Generate dynamic filename based on configured indexes  
        indexes_filename = '_'.join(index_names)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f'{indexes_filename}_Percentage_Analysis_{timeframe}_{timestamp}.csv'
        
        analysis_df.to_csv(output_file, index=False, float_format='%.2f')
        
        logger.info(f"Index percentage analysis generated: {output_file}")
        print(f"📊 Multi-Index Percentage Analysis Generated:")
        print(f"  • Output file: {output_file.name}")
        print(f"  • Indexes: {', '.join(index_names)}")
        print(f"  • Metrics analyzed: {len(analysis_rows)}")
        print(f"  • Format: Metric | Logic | {index_names[0]}_Above% | {index_names[0]}_Below% | ...")
        
        # Display member counts for each index
        for index_name in index_names:
            if index_name in calc_df.columns:
                index_mask = (calc_df[index_name] == True) | (calc_df[index_name] == 'True') | (calc_df[index_name] == 1)
                count = index_mask.sum()
                print(f"  • {index_name}: {count} members")
        
        # Database sync removed - using CSV files directly
        
        # Generate PDF report if enabled
        if config and getattr(user_config, 'pdf_reports_enable', True):
            try:
                from .pdf_report_generator import generate_pdf_report_from_csv
                pdf_path = generate_pdf_report_from_csv(str(output_file), config, user_config)
                if pdf_path:
                    print(f"  📄 PDF report generated: {Path(pdf_path).name}")
                else:
                    print(f"  ⚠️  PDF report generation failed")
            except Exception as e:
                logger.warning(f"PDF report generation failed: {e}")
                print(f"  ⚠️  PDF report generation failed: {e}")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error generating S&P 500 percentage analysis: {e}")
        return None


def generate_index_calculation_table(stocks_data, output_path, timeframe, user_config, config):
    """
    Generate INDEX_CALCULATION table using metrics from basic_calculation results.
    
    Args:
        stocks_data: Dictionary of {ticker: DataFrame} for stocks (e.g., S&P 500 members)
        output_path: Path to save the table
        timeframe: Data timeframe
        user_config: User configuration
        config: Config object with directory paths
        
    Returns:
        str: Path to generated INDEX_CALCULATION table
    """
    try:
        # Load basic calculation results (try date-stamped version first)
        from .basic_calculations import find_latest_basic_calculation_file
        calc_file = find_latest_basic_calculation_file(config, timeframe, user_config.ticker_choice)
        
        if not calc_file:
            logger.warning(f"Basic calculation file not found for {timeframe}")
            return None
            
        calc_df = pd.read_csv(calc_file)
        
        # Filter to only the stocks we're analyzing (e.g., S&P 500 members)
        stock_tickers = list(stocks_data.keys())
        calc_filtered = calc_df[calc_df['ticker'].isin(stock_tickers)].copy()
        
        if calc_filtered.empty:
            logger.warning("No calculation data found for selected stocks")
            return None
            
        # Create comprehensive INDEX_CALCULATION table with all PDF-specified metrics
        index_calc_columns = [
            # Basic identification and pricing
            'ticker', 'date', 'current_price', 
            
            # Day/Period Changes (from PDF spec)
            'daily_pct_change',  # Day change
            'monthly_pct_change', 'quarter_pct_change', 'yearly_pct_change',  # Period changes
            
            # Moving Averages and Price Relationships
            'sma_10', 'price_to_sma_10_pct',  # Price to SMA 10
            'sma_20', 'price_to_sma_20_pct',  # Price to SMA 20  
            'sma_50', 'price_to_sma_50_pct',  # Price to SMA 50
            'sma_200', 'ema_10', 'ema_20',
            
            # Index Overview Metrics (from new function)
            'trend_days_10_pct',  # Trend Days %
            'sma10_vs_sma20', 'sma20_vs_sma50', 'sma50_vs_sma200',  # MA Comparisons
            'perfect_bullish_alignment',  # Perfect Bullish Alignment
            'price_above_sma10', 'price_above_ema20', 'price_above_sma50',  # Price vs MAs
            'at_20day_high', 'at_20day_low',  # 20-day High/Low
            '5day_low_vs_30day_high',  # Strong Momentum Signal
            
            # Technical Indicators
            'rsi_14', 'macd', 'macd_signal',
            
            # Volume and Volatility
            'avg_volume_20', 'volume_trend',
            'atr', 'atr_pct',
            
            # Position Analysis
            'price_position_52w',
            
            # Classification
            'SP500'  # Boolean classification
        ]
        
        # Select available columns
        available_columns = [col for col in index_calc_columns if col in calc_filtered.columns]
        index_calc_table = calc_filtered[available_columns].copy()
        
        # Sort by ticker
        index_calc_table = index_calc_table.sort_values('ticker')
        
        # Add derived metrics
        if 'current_price' in index_calc_table.columns and 'sma_10' in index_calc_table.columns:
            # Calculate additional relative metrics if needed
            pass
        
        # Format numerical columns
        numeric_cols = ['current_price', 'daily_pct_change', 'price_to_sma_10_pct', 'price_to_sma_20_pct', 'price_to_sma_50_pct', 'rsi_14', 'price_position_52w']
        for col in numeric_cols:
            if col in index_calc_table.columns:
                index_calc_table[col] = pd.to_numeric(index_calc_table[col], errors='coerce').round(2)
        
        # Generate output file for individual stock metrics table using data date
        date_stamp = _extract_data_date_from_dataframe(index_calc_table)
        output_file = output_path / f'index_stocks_{timeframe}_{date_stamp}.csv'
        
        index_calc_table.to_csv(output_file, index=False, float_format='%.2f')
        
        logger.info(f"INDEX_CALCULATION table generated: {output_file}")
        print(f"📊 INDEX_CALCULATION Table Generated:")
        print(f"  • Output file: {output_file.name}")
        print(f"  • Stocks analyzed: {len(index_calc_table)}")
        print(f"  • Metrics included: {len(index_calc_table.columns)}")
        
        # Show key summary statistics
        if 'daily_pct_change' in index_calc_table.columns:
            avg_change = index_calc_table['daily_pct_change'].mean()
            print(f"  • Average daily change: {avg_change:.2f}%")
            
        if 'price_to_sma_10_pct' in index_calc_table.columns:
            avg_sma_dist = index_calc_table['price_to_sma_10_pct'].mean()
            print(f"  • Average distance to SMA 10: {avg_sma_dist:.2f}%")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error generating INDEX_CALCULATION table: {e}")
        return None


def calculate_index_metrics(df, ticker):
    """Calculate key metrics for an index."""
    if df is None or df.empty or 'Close' not in df.columns:
        return {}
        
    close = df['Close']
    metrics = {
        'ticker': ticker,
        'current_price': close.iloc[-1],
        'data_points': len(df),
        'start_date': df.index.min().strftime('%Y-%m-%d'),
        'end_date': df.index.max().strftime('%Y-%m-%d')
    }
    
    # Performance metrics
    if len(df) > 1:
        # Returns over different periods
        periods = {
            '1d': 1,
            '5d': 5, 
            '1m': 22,
            '3m': 66,
            '6m': 132,
            '1y': 252,
            'ytd': None,  # Will calculate separately
            'total': len(df) - 1
        }
        
        for period_name, days in periods.items():
            if period_name == 'ytd':
                # Year-to-date calculation
                current_year = datetime.now().year
                try:
                    ytd_start_idx = df.index >= f'{current_year}-01-01'
                    if ytd_start_idx.any():
                        ytd_start_price = df[ytd_start_idx]['Close'].iloc[0]
                        metrics[f'return_{period_name}'] = (close.iloc[-1] / ytd_start_price) - 1
                    else:
                        metrics[f'return_{period_name}'] = None
                except:
                    metrics[f'return_{period_name}'] = None
            elif days and days < len(df):
                metrics[f'return_{period_name}'] = (close.iloc[-1] / close.iloc[-days-1]) - 1
            elif period_name == 'total':
                metrics[f'return_{period_name}'] = (close.iloc[-1] / close.iloc[0]) - 1
            else:
                metrics[f'return_{period_name}'] = None
                
        # Volatility
        daily_returns = close.pct_change().dropna()
        if len(daily_returns) > 10:
            metrics['volatility_daily'] = daily_returns.std()
            metrics['volatility_annualized'] = daily_returns.std() * np.sqrt(252)
            
        # Drawdown analysis
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        metrics['current_drawdown'] = drawdown.iloc[-1]
        
        # 52-week high/low analysis
        if len(df) >= 252:
            high_52w = close.rolling(252).max().iloc[-1]
            low_52w = close.rolling(252).min().iloc[-1]
            metrics['high_52w'] = high_52w
            metrics['low_52w'] = low_52w
            metrics['distance_from_high'] = (close.iloc[-1] / high_52w) - 1
            metrics['distance_from_low'] = (close.iloc[-1] / low_52w) - 1
        else:
            metrics['high_52w'] = close.max()
            metrics['low_52w'] = close.min()
            metrics['distance_from_high'] = (close.iloc[-1] / close.max()) - 1 if close.max() != 0 else 0
            metrics['distance_from_low'] = (close.iloc[-1] / close.min()) - 1 if close.min() != 0 else 0
            
        # Trend analysis
        if len(df) >= 50:
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            metrics['above_sma_20'] = close.iloc[-1] > sma_20.iloc[-1]
            metrics['above_sma_50'] = close.iloc[-1] > sma_50.iloc[-1]
            metrics['sma_20_slope'] = (sma_20.iloc[-1] - sma_20.iloc[-6]) / 5 if len(df) >= 25 else 0
            metrics['sma_50_slope'] = (sma_50.iloc[-1] - sma_50.iloc[-11]) / 10 if len(df) >= 60 else 0
            
    # Volume analysis (if available)
    if 'Volume' in df.columns:
        volume = df['Volume']
        metrics['avg_volume'] = volume.mean()
        if len(df) >= 20:
            metrics['volume_trend'] = (volume.tail(10).mean() / volume.tail(20).head(10).mean()) - 1
            
    return metrics


def analyze_market_breadth(indices, stocks):
    """Analyze overall market breadth and health."""
    breadth_analysis = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_indices': len(indices),
        'total_stocks': len(stocks)
    }
    
    if not stocks:
        return breadth_analysis
        
    # Analyze stock performance
    stock_returns = {}
    advancing_declining = {'advancing': 0, 'declining': 0, 'unchanged': 0}
    
    for ticker, df in stocks.items():
        if df is not None and not df.empty and 'Close' in df.columns and len(df) > 1:
            daily_return = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
            stock_returns[ticker] = daily_return
            
            if daily_return > 0.001:  # More than 0.1% gain
                advancing_declining['advancing'] += 1
            elif daily_return < -0.001:  # More than 0.1% loss
                advancing_declining['declining'] += 1
            else:
                advancing_declining['unchanged'] += 1
                
    if stock_returns:
        breadth_analysis.update({
            'advance_decline_ratio': advancing_declining['advancing'] / max(advancing_declining['declining'], 1),
            'pct_advancing': advancing_declining['advancing'] / len(stock_returns),
            'pct_declining': advancing_declining['declining'] / len(stock_returns),
            'avg_stock_return': np.mean(list(stock_returns.values())),
            'median_stock_return': np.median(list(stock_returns.values())),
            'stocks_above_zero': sum(1 for r in stock_returns.values() if r > 0) / len(stock_returns)
        })
        
        # New highs/lows analysis (simplified)
        new_highs = sum(1 for ticker, df in stocks.items() 
                       if df is not None and not df.empty and 'Close' in df.columns and len(df) >= 20
                       and df['Close'].iloc[-1] == df['Close'].tail(20).max())
        new_lows = sum(1 for ticker, df in stocks.items() 
                      if df is not None and not df.empty and 'Close' in df.columns and len(df) >= 20
                      and df['Close'].iloc[-1] == df['Close'].tail(20).min())
        
        breadth_analysis.update({
            'new_highs_20d': new_highs,
            'new_lows_20d': new_lows,
            'new_high_low_ratio': new_highs / max(new_lows, 1)
        })
    
    return breadth_analysis


def create_index_overview(batch_data, output_path, timeframe, user_config=None, config=None, data_date=None):
    """
    Create comprehensive overview of market indices and breadth analysis.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        output_path: Path to save overview files
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration with index overview settings
        config: Config object with directory paths (optional)
        data_date: Data date from centralized date management (e.g., '20250829')
    """
    logger.info(f"Creating index overview for {len(batch_data)} tickers ({timeframe})")
    
    # Special handling for index mode when batch_data might be empty (SP500, NASDAQ100, etc.)
    if (not batch_data and user_config and hasattr(user_config, 'index_overview_indexes') and 
        user_config.index_overview_indexes.strip() and config):
        print(f"📋 Index Overview ({timeframe}) - Index mode using basic calculation data:")
        
        # Generate INDEX_CALCULATION table directly from basic calculation results
        try:
            # Use centralized data date if provided, otherwise fall back to file discovery
            if data_date:
                # Construct filename directly using centralized data date
                from src.config import get_file_safe_user_choice
                safe_user_choice = get_file_safe_user_choice(user_config.ticker_choice, preserve_hyphens=True)
                calc_file = config.directories['BASIC_CALCULATION_DIR'] / f'basic_calculation_{safe_user_choice}_{timeframe}_{data_date}.csv'
                print(f"  • Using centralized data date: {data_date}")
            else:
                # Fall back to file discovery (legacy mode)
                from .basic_calculations import find_latest_basic_calculation_file
                calc_file = find_latest_basic_calculation_file(config, timeframe, user_config.ticker_choice)
                print(f"  • Using file discovery fallback")
            
            if calc_file and calc_file.exists():
                calc_df = pd.read_csv(calc_file)
                sp500_mask = (calc_df['SP500'] == True) | (calc_df['SP500'] == 'True') | (calc_df['SP500'] == 1)
                sp500_count = sp500_mask.sum()
                total_count = len(calc_df)
                
                print(f"  • S&P 500 members found: {sp500_count}")
                print(f"  • Total tickers in calculation: {total_count}")
                
                # Generate comprehensive index percentage analysis (transposed format)
                if sp500_count > 0:
                    from .index_overview import run_index_counts_analysis, run_index_pctchg_analysis, run_index_rs_analysis
                    
                    # 1. Generate counts analysis (SP500/NASDAQ100 percentage analysis)
                    percentage_file = run_index_counts_analysis(output_path, timeframe, user_config, config, data_date)
                    if percentage_file:
                        print(f"  • Index counts analysis: {Path(percentage_file).name}")
                        
                        # Generate tornado chart from percentage analysis data
                        if user_config and getattr(user_config, 'index_overview_tornado_chart', False):
                            try:
                                print("📊 Generating tornado chart from index analysis...")
                                tornado_file = generate_tornado_chart_from_analysis(percentage_file, output_path, timeframe)
                                if tornado_file:
                                    print(f"  • Tornado chart: {Path(tornado_file).name}")
                            except Exception as e:
                                logger.error(f"Error generating tornado chart: {e}")
                                print(f"  ❌ Tornado chart generation failed: {e}")
                    
                    # 2. Generate percentage change analysis for sectors
                    pctchg_sectors_file = run_index_pctchg_analysis(output_path, timeframe, user_config, config, data_date, 'sectors')
                    if pctchg_sectors_file:
                        print(f"  • Index pctChg sectors analysis: {Path(pctchg_sectors_file).name}")
                    
                    # 3. Generate percentage change analysis for indexes
                    pctchg_indexes_file = run_index_pctchg_analysis(output_path, timeframe, user_config, config, data_date, 'indexes')
                    if pctchg_indexes_file:
                        print(f"  • Index pctChg indexes analysis: {Path(pctchg_indexes_file).name}")
                    
                    # 4. Generate relative strength analysis for sectors
                    rs_sectors_file = run_index_rs_analysis(output_path, timeframe, user_config, config, data_date, 'sectors')
                    if rs_sectors_file:
                        print(f"  • Index RS sectors analysis: {Path(rs_sectors_file).name}")
                    
                    # 5. Generate relative strength analysis for indexes
                    rs_indexes_file = run_index_rs_analysis(output_path, timeframe, user_config, config, data_date, 'indexes')
                    if rs_indexes_file:
                        print(f"  • Index RS indexes analysis: {Path(rs_indexes_file).name}")
                
                # Create a simple mock batch_data for the SP500 tickers
                sp500_tickers = calc_df[sp500_mask]['ticker'].tolist()
                mock_batch_data = {ticker: pd.DataFrame() for ticker in sp500_tickers}
                
                # Generate detailed INDEX_CALCULATION table
                index_table_file = generate_index_calculation_table(mock_batch_data, output_path, timeframe, user_config, config)
                if index_table_file:
                    print(f"  • INDEX_CALCULATION table: {Path(index_table_file).name}")
                
                return sp500_count
            else:
                print(f"  ❌ Basic calculation file not found: {calc_file}")
                return 0
                
        except Exception as e:
            logger.error(f"Error in SP500 mode index overview: {e}")
            print(f"  ❌ SP500 mode failed: {e}")
            return 0
    
    # Standard mode with market data
    # Separate indices from individual stocks
    indices, stocks = identify_index_tickers(batch_data, user_config, config)
    
    print(f"📋 Index Overview ({timeframe}):")
    print(f"  • Identified {len(indices)} index tickers")
    print(f"  • Identified {len(stocks)} stock tickers")
    
    # Show configured categories if available
    if user_config:
        if hasattr(user_config, 'index_overview_indexes') and user_config.index_overview_indexes.strip():
            config_indexes = [idx.strip() for idx in user_config.index_overview_indexes.split(',') if idx.strip()]
            print(f"  • Configured indexes: {', '.join(config_indexes)}")
            
        if hasattr(user_config, 'index_overview_sectors') and user_config.index_overview_sectors.strip():
            config_sectors = [sec.strip() for sec in user_config.index_overview_sectors.split(',') if sec.strip()]
            print(f"  • Configured sectors: {', '.join(config_sectors)}")
            
        if hasattr(user_config, 'index_overview_etfs') and user_config.index_overview_etfs.strip():
            config_etfs = [etf.strip() for etf in user_config.index_overview_etfs.split(',') if etf.strip()]
            print(f"  • Configured ETFs: {', '.join(config_etfs)}")
            
        if hasattr(user_config, 'index_overview_stocks') and user_config.index_overview_stocks.strip():
            config_stocks = [stock.strip() for stock in user_config.index_overview_stocks.split(',') if stock.strip()]
            print(f"  • Configured stocks: {', '.join(config_stocks)}")
    
    # Analyze each index
    index_results = []
    for ticker, df in indices.items():
        try:
            metrics = calculate_index_metrics(df, ticker)
            if metrics:
                index_results.append(metrics)
        except Exception as e:
            logger.error(f"Error analyzing index {ticker}: {e}")
            
    # Create breadth analysis
    breadth_analysis = analyze_market_breadth(indices, stocks)
    
    # Save index overview
    if index_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Index metrics file
        index_df = pd.DataFrame(index_results)
        from src.config import get_file_safe_user_choice
        safe_user_choice = get_file_safe_user_choice(user_config.ticker_choice, preserve_hyphens=True) if user_config else 'unknown'
        index_file = output_path / f'index_metrics_{safe_user_choice}_{timeframe}_{timestamp}.csv'
        index_df.to_csv(index_file, index=False)
        
        # Breadth analysis file
        breadth_df = pd.DataFrame([breadth_analysis])
        breadth_file = output_path / f'market_breadth_{timeframe}_{timestamp}.csv'
        breadth_df.to_csv(breadth_file, index=False)
        
        # Combined overview file
        overview_data = {
            'overview_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframe': timeframe,
            'total_tickers_analyzed': len(batch_data),
            'index_tickers': len(indices),
            'stock_tickers': len(stocks),
            **breadth_analysis
        }
        
        overview_df = pd.DataFrame([overview_data])
        overview_file = output_path / f'indexes_overview_{timeframe}_{timestamp}.csv'
        overview_df.to_csv(overview_file, index=False)
        
        logger.info(f"Index overview completed - {len(index_results)} indices analyzed")
        logger.info(f"Files created: {index_file.name}, {breadth_file.name}, {overview_file.name}")
        
        # Print summary
        if index_results:
            returns_1d = [r.get('return_1d', 0) for r in index_results if r.get('return_1d') is not None]
            if returns_1d:
                print(f"  • Average 1-day return: {np.mean(returns_1d):.2%}")
                
        if breadth_analysis.get('advance_decline_ratio'):
            print(f"  • Advance/Decline Ratio: {breadth_analysis['advance_decline_ratio']:.2f}")
            print(f"  • Stocks Advancing: {breadth_analysis.get('pct_advancing', 0):.1%}")
        
        print(f"  • Overview files saved to: {output_path}")
        
        # Generate enhanced INDEX_CALCULATION table with basic calculation metrics
        if config and user_config and hasattr(user_config, 'index_overview_indexes') and user_config.index_overview_indexes.strip().upper() == 'SP500':
            print("📊 Generating S&P 500 INDEX_CALCULATION table...")
            try:
                index_table_file = generate_index_calculation_table(stocks, output_path, timeframe, user_config, config)
                if index_table_file:
                    print(f"  • INDEX_CALCULATION table: {Path(index_table_file).name}")
            except Exception as e:
                logger.error(f"Error generating INDEX_CALCULATION table: {e}")
                print(f"  ❌ INDEX_CALCULATION table generation failed: {e}")
        
        # Generate overview template part 1 (price + MAs)
        print("📊 Generating overview template part 1...")
        try:
            template_file = generate_overview_template_part1(batch_data, output_path, user_config)
            if template_file:
                print(f"  • Template file: overview_template_part1.csv")
        except Exception as e:
            logger.error(f"Error generating overview template: {e}")
            print(f"  ❌ Template generation failed: {e}")
        
        # Generate tornado charts if enabled
        if user_config and getattr(user_config, 'index_overview_tornado_chart', False):
            print("📊 Generating tornado charts...")
            try:
                # Create combined dataset for tornado chart
                all_metrics = []
                for ticker, df in batch_data.items():
                    try:
                        from .overview_price_mas import calculate_price_metrics
                        metrics = calculate_price_metrics(df, ticker)
                        all_metrics.append(metrics)
                    except Exception:
                        continue
                
                if all_metrics:
                    tornado_data = pd.DataFrame(all_metrics)
                    chart_files = generate_tornado_charts(tornado_data, user_config, output_path)
                    if chart_files:
                        for chart_file in chart_files:
                            print(f"  • Chart: {Path(chart_file).name}")
                else:
                    print("  ⚠️  No data available for tornado charts")
                    
            except Exception as e:
                logger.error(f"Error generating tornado charts: {e}")
                print(f"  ❌ Tornado chart generation failed: {e}")
        
        # Display price/MA summary
        try:
            summary = get_price_ma_summary(batch_data)
            if summary:
                print(f"📈 Price & MA Summary:")
                print(f"  • Average 1-month return: {summary.get('avg_return_1m', 0)*100:.1f}%")
                print(f"  • Average volatility: {summary.get('avg_volatility', 0)*100:.1f}%")
                print(f"  • Average RSI: {summary.get('avg_rsi', 50):.1f}")
        except Exception as e:
            logger.debug(f"Error generating price/MA summary: {e}")
        
    else:
        logger.warning("No index data to analyze")
        print(f"  ⚠️  No valid index data found in batch")
        
    return len(index_results)