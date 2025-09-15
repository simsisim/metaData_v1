"""
Indexes Overview Relative Strength Display
==========================================

Displays Relative Strength (RS) metrics for specified indexes across different timeframes.
Focuses on RS rankings, strength comparisons, and momentum indicators.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_configurable_rs_metrics(timeframe, user_config, calc_df):
    """
    Get configurable RS metrics based on timeframe and user configuration.
    
    Args:
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration with RS period settings
        calc_df: DataFrame to check available columns
        
    Returns:
        list: List of tuples (metric_name, column_name, description)
    """
    rs_metrics = []
    
    # Always include IBD RS Rating if available
    if 'rs_ibd' in calc_df.columns:
        rs_metrics.append(('IBD RS Rating', 'rs_ibd', 'IBD-style Relative Strength Rating (1-99 scale)'))
    
    if timeframe == 'daily':
        # Daily RS periods configuration - map to systematic RS column names  
        if hasattr(user_config, 'daily_daily_periods') and user_config.daily_daily_periods:
            daily_rs_periods = [int(p.strip()) for p in user_config.daily_daily_periods.split(';') if p.strip().isdigit()]
            for period in daily_rs_periods:
                col_name = f'daily_daily_daily_{period}d_RS'  # Systematic naming
                if col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}-Day', col_name, f'{period}-day relative strength vs market'))
                # Also check for percentile column
                per_col_name = f'daily_daily_daily_{period}d_rs_per'
                if per_col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Percentile', per_col_name, f'{period}-day RS percentile ranking'))
        
        # Weekly RS periods configuration - map to systematic RS column names
        if hasattr(user_config, 'daily_weekly_periods') and user_config.daily_weekly_periods:
            weekly_rs_periods = [int(p.strip()) for p in user_config.daily_weekly_periods.split(';') if p.strip().isdigit()]
            for period in weekly_rs_periods:
                col_name = f'daily_daily_weekly_{period}d_RS'  # Systematic naming
                if col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Weekly', col_name, f'{period}-day weekly relative strength vs market'))
                # Also check for percentile column
                per_col_name = f'daily_daily_weekly_{period}d_rs_per'
                if per_col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Weekly Percentile', per_col_name, f'{period}-day weekly RS percentile ranking'))
        
        # Monthly RS periods configuration - map to systematic RS column names
        if hasattr(user_config, 'daily_monthly_periods') and user_config.daily_monthly_periods:
            monthly_rs_periods = [int(p.strip()) for p in user_config.daily_monthly_periods.split(';') if p.strip().isdigit()]
            for period in monthly_rs_periods:
                col_name = f'daily_daily_monthly_{period}d_RS'  # Systematic naming
                if col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Monthly', col_name, f'{period}-day monthly relative strength vs market'))
                # Also check for percentile column
                per_col_name = f'daily_daily_monthly_{period}d_rs_per'
                if per_col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Monthly Percentile', per_col_name, f'{period}-day monthly RS percentile ranking'))
        
        # Quarterly RS periods configuration - map to systematic RS column names
        if hasattr(user_config, 'daily_quarterly_periods') and user_config.daily_quarterly_periods:
            quarterly_rs_periods = [int(p.strip()) for p in user_config.daily_quarterly_periods.split(';') if p.strip().isdigit()]
            for period in quarterly_rs_periods:
                col_name = f'daily_daily_quarterly_{period}d_RS'  # Systematic naming
                if col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Quarterly', col_name, f'{period}-day quarterly relative strength vs market'))
                # Also check for percentile column
                per_col_name = f'daily_daily_quarterly_{period}d_rs_per'
                if per_col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Quarterly Percentile', per_col_name, f'{period}-day quarterly RS percentile ranking'))
        
        # Yearly RS periods configuration - map to systematic RS column names
        if hasattr(user_config, 'daily_yearly_periods') and user_config.daily_yearly_periods:
            yearly_rs_periods = [int(p.strip()) for p in user_config.daily_yearly_periods.split(';') if p.strip().isdigit()]
            for period in yearly_rs_periods:
                col_name = f'daily_daily_yearly_{period}d_RS'  # Systematic naming
                if col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Yearly', col_name, f'{period}-day yearly relative strength vs market'))
                # Also check for percentile column
                per_col_name = f'daily_daily_yearly_{period}d_rs_per'
                if per_col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}D Yearly Percentile', per_col_name, f'{period}-day yearly RS percentile ranking'))
    
    elif timeframe == 'weekly':
        # Weekly RS periods configuration - systematic naming
        if hasattr(user_config, 'weekly_weekly_periods') and user_config.weekly_weekly_periods:
            weekly_rs_periods = [int(p.strip()) for p in user_config.weekly_weekly_periods.split(';') if p.strip().isdigit()]
            for period in weekly_rs_periods:
                col_name = f'weekly_weekly_weekly_{period}w_RS'  # Systematic naming
                if col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}-Week', col_name, f'{period}-week relative strength vs market'))
                # Also check for percentile column
                per_col_name = f'weekly_weekly_weekly_{period}w_rs_per'
                if per_col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}W Percentile', per_col_name, f'{period}-week RS percentile ranking'))
        
        # Weekly monthly RS periods configuration - systematic naming
        if hasattr(user_config, 'weekly_monthly_periods') and user_config.weekly_monthly_periods:
            weekly_monthly_periods = [int(p.strip()) for p in user_config.weekly_monthly_periods.split(';') if p.strip().isdigit()]
            for period in weekly_monthly_periods:
                col_name = f'weekly_weekly_monthly_{period}w_RS'  # Systematic naming
                if col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}W Monthly', col_name, f'{period}-week monthly relative strength vs market'))
                # Also check for percentile column
                per_col_name = f'weekly_weekly_monthly_{period}w_rs_per'
                if per_col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}W Monthly Percentile', per_col_name, f'{period}-week monthly RS percentile ranking'))
    
    elif timeframe == 'monthly':
        # Monthly RS periods configuration - systematic naming
        if hasattr(user_config, 'RS_monthly_periods') and user_config.RS_monthly_periods:
            monthly_rs_periods = [int(p.strip()) for p in user_config.RS_monthly_periods.split(';') if p.strip().isdigit()]
            for period in monthly_rs_periods:
                col_name = f'monthly_monthly_monthly_{period}m_RS'  # Systematic naming
                if col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}-Month', col_name, f'{period}-month relative strength vs market'))
                # Also check for percentile column
                per_col_name = f'monthly_monthly_monthly_{period}m_rs_per'
                if per_col_name in calc_df.columns:
                    rs_metrics.append((f'RS {period}M Percentile', per_col_name, f'{period}-month RS percentile ranking'))
    
    # Add additional systematic RS columns if available (any remaining RS metrics not caught above)
    # Check for systematic RS columns that might not have been processed
    rs_columns = [col for col in calc_df.columns if '_RS' in col and col not in [metric[1] for metric in rs_metrics]]
    for col in rs_columns:
        # Extract meaningful display name from systematic column name
        if '_rs_per' in col:
            display_name = f'Additional RS Percentile ({col.replace("_rs_per", "")})'
            description = f'Additional RS percentile ranking from {col}'
        elif '_RS' in col:
            display_name = f'Additional RS ({col.replace("_RS", "")})'
            description = f'Additional relative strength from {col}'
        else:
            continue
        
        rs_metrics.append((display_name, col, description))
    
    return rs_metrics


def generate_index_rs_display(calc_df, output_path, timeframe, user_config, data_date, config=None, analysis_type='tickers'):
    """
    Generate index Relative Strength display for specified indexes.
    Uses provided calc_df data directly from CSV files.
    
    Args:
        calc_df: DataFrame with basic calculation results
        output_path: Path to save the RS display file
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration with index settings
        data_date: Data date for filename generation (e.g., '20250829')
        config: Config object (optional)
        analysis_type: Type of analysis - 'sectors', 'indexes', or 'tickers' (legacy)
        
    Returns:
        str: Path to generated RS display file
    """
    try:
        # Parse configured tickers based on analysis type
        if analysis_type == 'sectors':
            pctchgrs_tickers_str = user_config.index_overview_pctchgrs_sectors.strip() if user_config.index_overview_pctchgrs_sectors else 'XLF,XLE,XLK,XLV,XLI,XLB,XLRE,XLP,XLY,XLU,XLC'
        elif analysis_type == 'indexes':
            pctchgrs_tickers_str = user_config.index_overview_pctchgrs_indexes.strip() if user_config.index_overview_pctchgrs_indexes else 'SPY,QQQ'
        else:  # Legacy 'tickers' mode
            pctchgrs_tickers_str = user_config.index_overview_pctchgrs_tickers.strip() if user_config.index_overview_pctchgrs_tickers else 'SPY,QQQ'
        
        ticker_list = [ticker.strip().upper() for ticker in pctchgrs_tickers_str.split(',') if ticker.strip()]
        
        # Using CSV data directly - database loading removed
        
        if calc_df is None or calc_df.empty:
            logger.warning("No calculation data available for RS analysis")
            return None
        
        # Get configurable RS metrics based on timeframe
        rs_metrics = get_configurable_rs_metrics(timeframe, user_config, calc_df)
        
        # Create RS display table with individual ticker data
        display_rows = []
        
        for metric_name, column_name, description in rs_metrics:
            row_data = {
                'RS_Metric': metric_name,
                'Description': description
            }
            
            # Calculate RS values for each configured ticker (individual ticker analysis)
            for ticker in ticker_list:
                ticker_data = calc_df[calc_df['ticker'] == ticker]
                
                if not ticker_data.empty and column_name in ticker_data.columns:
                    # Get the latest RS value for this ticker
                    latest_rs = ticker_data[column_name].iloc[-1] if not pd.isna(ticker_data[column_name].iloc[-1]) else 0.0
                    row_data[f'{ticker}'] = latest_rs
                else:
                    # No RS data found for this ticker
                    row_data[f'{ticker}'] = np.nan
                        
            display_rows.append(row_data)
        
        # Create DataFrame
        display_df = pd.DataFrame(display_rows)
        
        # Use the caller's data date for consistent filename generation
        safe_user_choice = str(user_config.ticker_choice).replace('-', '_')
        output_file = output_path / f'pctChgRS_rs_{analysis_type}_{safe_user_choice}_{timeframe}_{data_date}.csv'
        
        display_df.to_csv(output_file, index=False, float_format='%.2f')
        
        # File logging removed - using CSV files directly
        
        logger.info(f"Index RS display generated: {output_file}")
        print(f"📊 Index Relative Strength Display Generated:")
        print(f"  • Output file: {output_file.name}")
        print(f"  • Tickers: {', '.join(ticker_list)}")
        print(f"  • Data date: {data_date}")
        print(f"  • RS metrics analyzed: {len(display_rows)}")
        print(f"  • Format: RS_Metric | Description | {ticker_list[0]} | {ticker_list[1]} | ...")
        
        # Display sample RS data for verification
        for ticker in ticker_list:
            ticker_count = (calc_df['ticker'] == ticker).sum()
            print(f"  • {ticker}: {ticker_count} records")
            
            # Show current RS summary if available
            if ticker_count > 0:
                ticker_data = calc_df[calc_df['ticker'] == ticker]
                if 'rs_ibd' in calc_df.columns and not ticker_data.empty:
                    current_rs_ibd = ticker_data['rs_ibd'].iloc[-1] if not pd.isna(ticker_data['rs_ibd'].iloc[-1]) else 0
                    print(f"    Current IBD RS: {current_rs_ibd:.1f}")
        
        # Note: PDF generation happens at the end of the program via reportlab under /results/reports/
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error generating index RS display: {e}")
        return None


# PDF generation removed - handled by reportlab at end of pipeline under /results/reports/


def calculate_rs_strength_distribution(calc_df, index_names):
    """
    Calculate RS strength distribution for index members.
    
    Args:
        calc_df: DataFrame with calculation results
        index_names: List of index names to analyze
        
    Returns:
        dict: RS distribution statistics by index
    """
    rs_distributions = {}
    
    for index_name in index_names:
        if index_name in calc_df.columns and 'rs_ibd' in calc_df.columns:
            if index_name.startswith('^'):
                index_data = calc_df[calc_df['ticker'] == index_name]
            else:
                index_mask = (calc_df[index_name] == True) | (calc_df[index_name] == 'True') | (calc_df[index_name] == 1)
                index_data = calc_df[index_mask]
            
            if not index_data.empty:
                rs_values = index_data['rs_ibd'].dropna()
                if len(rs_values) > 0:
                    rs_distributions[index_name] = {
                        'very_strong_rs': (rs_values >= 90).sum(),  # RS >= 90
                        'strong_rs': ((rs_values >= 80) & (rs_values < 90)).sum(),  # RS 80-89
                        'above_average_rs': ((rs_values >= 60) & (rs_values < 80)).sum(),  # RS 60-79
                        'average_rs': ((rs_values >= 40) & (rs_values < 60)).sum(),  # RS 40-59
                        'below_average_rs': ((rs_values >= 20) & (rs_values < 40)).sum(),  # RS 20-39
                        'weak_rs': (rs_values < 20).sum(),  # RS < 20
                        'total_count': len(rs_values),
                        'avg_rs': rs_values.mean(),
                        'median_rs': rs_values.median()
                    }
    
    return rs_distributions


def generate_rs_summary_table(calc_df, index_names, output_path, timeframe):
    """
    Generate a summary table of RS strength distribution.
    
    Args:
        calc_df: DataFrame with calculation results
        index_names: List of index names
        output_path: Output directory
        timeframe: Data timeframe
        
    Returns:
        str: Path to RS summary file
    """
    try:
        rs_dist = calculate_rs_strength_distribution(calc_df, index_names)
        
        if not rs_dist:
            return None
            
        # Create summary rows
        summary_rows = []
        for index_name, dist in rs_dist.items():
            summary_rows.append({
                'Index': index_name,
                'Total_Members': dist['total_count'],
                'Very_Strong_rs_(90+)': dist['very_strong_rs'],
                'Strong_rs_(80-89)': dist['strong_rs'],
                'Above_Avg_rs_(60-79)': dist['above_average_rs'],
                'Average_rs_(40-59)': dist['average_rs'],
                'Below_Avg_rs_(20-39)': dist['below_average_rs'],
                'Weak_rs_(<20)': dist['weak_rs'],
                'Average_RS': dist['avg_rs'],
                'Median_RS': dist['median_rs'],
                'Strong_rs_Percentage': ((dist['very_strong_rs'] + dist['strong_rs']) / dist['total_count']) * 100
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save RS summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        indexes_filename = '_'.join(index_names)
        summary_file = output_path / f'{indexes_filename}_rs_summary_{timeframe}_{timestamp}.csv'
        
        summary_df.to_csv(summary_file, index=False, float_format='%.2f')
        
        logger.info(f"RS summary table generated: {summary_file}")
        print(f"  • RS Summary: {summary_file.name}")
        
        return str(summary_file)
        
    except Exception as e:
        logger.error(f"Error generating RS summary table: {e}")
        return None


def find_latest_rs_summary_file(config, timeframe, ticker_choice):
    """
    Find the latest RS summary file for the given timeframe and ticker choice.
    
    Args:
        config: Config object
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        ticker_choice: User ticker choice number
        
    Returns:
        Path to latest RS summary file or None if not found
    """
    rs_dir = config.directories['RESULTS_DIR'] / 'rs'
    if not rs_dir.exists():
        return None
    
    # Pattern: rs_ibd_stocks_{timeframe}_{choice}_{date}.csv (correct naming convention)
    import re
    pattern = re.compile(rf'rs_ibd_stocks_{timeframe}_{ticker_choice}_(\d+)\.csv')
    
    summary_files = []
    for file_path in rs_dir.glob(f'rs_ibd_stocks_{timeframe}_{ticker_choice}_*.csv'):
        match = pattern.match(file_path.name)
        if match:
            date_str = match.group(1)
            summary_files.append((date_str, file_path))
    
    if summary_files:
        # Sort by date and return the latest
        summary_files.sort(key=lambda x: x[0], reverse=True)
        return summary_files[0][1]
    
    return None


def run_index_rs_analysis(output_path, timeframe, user_config, config, data_date=None, analysis_type='tickers'):
    """
    Main entry point for index RS analysis.
    
    Args:
        output_path: Directory to save analysis files
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration
        config: Config object
        data_date: Centralized data date (e.g., '20250829'), if None will use file discovery
        analysis_type: Type of analysis - 'sectors', 'indexes', or 'tickers' (legacy)
        
    Returns:
        str: Path to generated analysis file or None if failed
    """
    try:
        # Use centralized data date if provided, otherwise fall back to file discovery
        if data_date:
            # For RS analysis, we still need to find the RS summary file
            # But we can be more targeted in the search using the data date context
            rs_summary_file = find_latest_rs_summary_file(config, timeframe, user_config.ticker_choice)
            logger.info(f"Using centralized data context for {timeframe} RS analysis: data_date={data_date}")
        else:
            # Fall back to file discovery (legacy mode)  
            rs_summary_file = find_latest_rs_summary_file(config, timeframe, user_config.ticker_choice)
            logger.warning(f"Using file discovery fallback for {timeframe} RS analysis")
        
        if not rs_summary_file:
            logger.warning(f"RS summary file not found for {timeframe}, ticker choice {user_config.ticker_choice}")
            return None
            
        calc_df = pd.read_csv(rs_summary_file)
        
        if calc_df.empty:
            logger.warning("RS summary data is empty")
            return None
            
        # Generate the RS display
        rs_display_file = generate_index_rs_display(calc_df, output_path, timeframe, user_config, data_date, config, analysis_type)
        
        # Also generate RS summary table
        indexes_str = user_config.index_overview_indexes.strip() if user_config.index_overview_indexes else 'SPY,QQQ'
        index_names = [idx.strip().upper() for idx in indexes_str.split(',') if idx.strip()]
        
        rs_summary_file = generate_rs_summary_table(calc_df, index_names, output_path, timeframe)
        
        return rs_display_file
        
    except Exception as e:
        logger.error(f"Error in index RS analysis: {e}")
        return None


def get_index_rs_leaders(calc_df, index_names, top_n=10):
    """
    Get top RS performers for each index.
    
    Args:
        calc_df: DataFrame with calculation results
        index_names: List of index names to analyze
        top_n: Number of top performers to return
        
    Returns:
        dict: Top RS performers by index
    """
    rs_leaders = {}
    
    for index_name in index_names:
        if index_name in calc_df.columns and 'rs_ibd' in calc_df.columns:
            if index_name.startswith('^'):
                index_data = calc_df[calc_df['ticker'] == index_name]
            else:
                index_mask = (calc_df[index_name] == True) | (calc_df[index_name] == 'True') | (calc_df[index_name] == 1)
                index_data = calc_df[index_mask]
            
            if not index_data.empty and len(index_data) >= top_n:
                # Sort by RS and get top performers
                top_rs = index_data.nlargest(top_n, 'rs_ibd')
                rs_leaders[index_name] = top_rs[['ticker', 'rs_ibd', 'daily_pct_change']].to_dict('records')
    
    return rs_leaders


def calculate_index_rs_momentum(calc_df, index_names):
    """
    Calculate overall RS momentum for indexes.
    
    Args:
        calc_df: DataFrame with calculation results
        index_names: List of index names
        
    Returns:
        dict: RS momentum metrics by index
    """
    rs_momentum = {}
    
    for index_name in index_names:
        if index_name in calc_df.columns:
            if index_name.startswith('^'):
                index_data = calc_df[calc_df['ticker'] == index_name]
            else:
                index_mask = (calc_df[index_name] == True) | (calc_df[index_name] == 'True') | (calc_df[index_name] == 1)
                index_data = calc_df[index_mask]
            
            if not index_data.empty:
                momentum_metrics = {}
                
                # Calculate various RS-based momentum indicators
                if 'rs_ibd' in index_data.columns:
                    rs_values = index_data['rs_ibd'].dropna()
                    if len(rs_values) > 0:
                        momentum_metrics['avg_rs_rating'] = rs_values.mean()
                        momentum_metrics['median_rs_rating'] = rs_values.median()
                        momentum_metrics['rs_leaders_pct'] = (rs_values >= 80).sum() / len(rs_values) * 100
                
                # Price momentum that supports RS
                if 'daily_pct_change' in index_data.columns:
                    daily_changes = index_data['daily_pct_change'].dropna()
                    if len(daily_changes) > 0:
                        momentum_metrics['avg_daily_change'] = daily_changes.mean()
                        momentum_metrics['positive_momentum_pct'] = (daily_changes > 0).sum() / len(daily_changes) * 100
                
                # Volume confirmation for RS
                if 'volume_trend' in index_data.columns:
                    volume_trends = index_data['volume_trend'].dropna()
                    if len(volume_trends) > 0:
                        momentum_metrics['avg_volume_trend'] = volume_trends.mean()
                        momentum_metrics['increasing_volume_pct'] = (volume_trends > 0).sum() / len(volume_trends) * 100
                
                rs_momentum[index_name] = momentum_metrics
    
    return rs_momentum