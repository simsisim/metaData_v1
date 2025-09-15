"""
Index Overview Percentage Change Display
=======================================

Displays percentage change metrics for specified indexes across different timeframes.
Focuses on change-based metrics including daily, monthly, quarterly, and yearly changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_configurable_pctchg_metrics(timeframe, user_config, calc_df):
    """
    Get configurable percentage change metrics based on timeframe and user configuration.
    
    Args:
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration with period settings
        calc_df: DataFrame to check available columns
        
    Returns:
        list: List of tuples (metric_name, column_name, description)
    """
    pctchg_metrics = []
    
    if timeframe == 'daily':
        # Basic daily change (always included)
        if 'daily_pct_change' in calc_df.columns:
            pctchg_metrics.append(('Daily Change (%)', 'daily_pct_change', 'Current day percentage change'))
        
        # Daily periods configuration - using systematic naming
        if hasattr(user_config, 'daily_daily_periods') and user_config.daily_daily_periods:
            daily_periods = [int(p.strip()) for p in user_config.daily_daily_periods.split(';') if p.strip().isdigit()]
            for period in daily_periods:
                col_name = f'daily_daily_daily_{period}d_pct_change'
                if col_name in calc_df.columns:
                    pctchg_metrics.append((f'{period}-Day Change (%)', col_name, f'{period}-day percentage change'))
        
        # Weekly periods (in days for daily data) - using systematic naming
        if hasattr(user_config, 'daily_weekly_periods') and user_config.daily_weekly_periods:
            weekly_periods = [int(p.strip()) for p in user_config.daily_weekly_periods.split(';') if p.strip().isdigit()]
            for period in weekly_periods:
                col_name = f'daily_daily_weekly_{period}d_pct_change'
                if col_name in calc_df.columns:
                    pctchg_metrics.append((f'{period//7}W Change (%)', col_name, f'{period}-day ({period//7} week) percentage change'))
        
        # Monthly periods - using systematic naming
        if hasattr(user_config, 'daily_monthly_periods') and user_config.daily_monthly_periods:
            monthly_periods = [int(p.strip()) for p in user_config.daily_monthly_periods.split(';') if p.strip().isdigit()]
            for period in monthly_periods:
                col_name = f'daily_daily_monthly_{period}d_pct_change'
                if col_name in calc_df.columns:
                    pctchg_metrics.append((f'{period//22}M Change (%)', col_name, f'{period}-day ({period//22} month) percentage change'))
        
        # Quarterly periods - using systematic naming
        if hasattr(user_config, 'daily_quarterly_periods') and user_config.daily_quarterly_periods:
            quarterly_periods = [int(p.strip()) for p in user_config.daily_quarterly_periods.split(';') if p.strip().isdigit()]
            for period in quarterly_periods:
                col_name = f'daily_daily_quarterly_{period}d_pct_change'
                if col_name in calc_df.columns:
                    pctchg_metrics.append((f'{period//66}Q Change (%)', col_name, f'{period}-day ({period//66} quarter) percentage change'))
        
        # Yearly periods - using systematic naming
        if hasattr(user_config, 'daily_yearly_periods') and user_config.daily_yearly_periods:
            yearly_periods = [int(p.strip()) for p in user_config.daily_yearly_periods.split(';') if p.strip().isdigit()]
            for period in yearly_periods:
                col_name = f'daily_daily_yearly_{period}d_pct_change'
                if col_name in calc_df.columns:
                    pctchg_metrics.append((f'{period//252}Y Change (%)', col_name, f'{period}-day (1 year) percentage change'))
    
    elif timeframe == 'weekly':
        # Weekly periods configuration - using systematic naming
        if hasattr(user_config, 'weekly_weekly_periods') and user_config.weekly_weekly_periods:
            weekly_periods = [int(p.strip()) for p in user_config.weekly_weekly_periods.split(';') if p.strip().isdigit()]
            for period in weekly_periods:
                col_name = f'weekly_weekly_weekly_{period}w_pct_change'
                if col_name in calc_df.columns:
                    pctchg_metrics.append((f'{period}-Week Change (%)', col_name, f'{period}-week percentage change'))
        
        # Monthly periods (in weeks for weekly data) - using systematic naming
        if hasattr(user_config, 'weekly_monthly_periods') and user_config.weekly_monthly_periods:
            monthly_periods = [int(p.strip()) for p in user_config.weekly_monthly_periods.split(';') if p.strip().isdigit()]
            for period in monthly_periods:
                col_name = f'weekly_weekly_monthly_{period}w_pct_change'
                if col_name in calc_df.columns:
                    pctchg_metrics.append((f'{period//4}M Change (%)', col_name, f'{period}-week ({period//4} month) percentage change'))
    
    elif timeframe == 'monthly':
        # Monthly periods configuration - using systematic naming
        if hasattr(user_config, 'RS_monthly_periods') and user_config.RS_monthly_periods:
            monthly_periods = [int(p.strip()) for p in user_config.RS_monthly_periods.split(';') if p.strip().isdigit()]
            for period in monthly_periods:
                col_name = f'monthly_monthly_monthly_{period}m_pct_change'
                if col_name in calc_df.columns:
                    pctchg_metrics.append((f'{period}-Month Change (%)', col_name, f'{period}-month percentage change'))
    
    # Add standard position and indicator metrics (always included)
    # Using timeframe-prefixed MA naming for consistency
    standard_metrics = [
        ('Distance from SMA 10 (%)', f'{timeframe}_price2_sma10pct', 'Current price distance from 10-day SMA'),
        ('Distance from SMA 20 (%)', f'{timeframe}_price2_sma20pct', 'Current price distance from 20-day SMA'),
        ('Distance from SMA 50 (%)', f'{timeframe}_price2_sma50pct', 'Current price distance from 50-day SMA'),
        ('Distance from SMA 200 (%)', f'{timeframe}_price2_sma200pct', 'Current price distance from 200-day SMA'),
        ('52-Week Position (%)', 'price_position_52w', 'Current price position within 52-week range'),
        ('20-Day Position (%)', 'price_position_20d', 'Current price position within 20-day range'),
        ('RSI Value', 'rsi_14', 'Relative Strength Index (14-period)'),
        ('Trend Days %', 'trend_days_10_pct', 'Percentage of up days in last 10 trading days'),
    ]
    
    # Add standard metrics if they exist in the data
    for metric_name, column_name, description in standard_metrics:
        if column_name in calc_df.columns:
            pctchg_metrics.append((metric_name, column_name, description))
    
    return pctchg_metrics


def generate_index_pctchg_display(calc_df, output_path, timeframe, user_config, data_date, config=None, analysis_type='tickers'):
    """
    Generate index percentage change display for specified indexes.
    Uses provided calc_df data directly from CSV files.
    
    Args:
        calc_df: DataFrame with basic calculation results
        output_path: Path to save the display file
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration with index settings
        data_date: Data date for filename generation (e.g., '20250829')
        config: Config object (optional)
        analysis_type: Type of analysis - 'sectors', 'indexes', or 'tickers' (legacy)
        
    Returns:
        str: Path to generated percentage change display file
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
            logger.warning("No calculation data available for pctChg analysis")
            return None
        
        # Get configurable periods based on timeframe
        pctchg_metrics = get_configurable_pctchg_metrics(timeframe, user_config, calc_df)
        
        # Create display table for percentage changes with individual ticker data
        display_rows = []
        
        for metric_name, column_name, description in pctchg_metrics:
            row_data = {
                'Metric': metric_name,
                'Description': description
            }
            
            # Calculate values for each configured ticker (individual ticker analysis)
            for ticker in ticker_list:
                ticker_data = calc_df[calc_df['ticker'] == ticker]
                
                if not ticker_data.empty and column_name in ticker_data.columns:
                    # Get the latest value for this ticker
                    latest_value = ticker_data[column_name].iloc[-1] if not pd.isna(ticker_data[column_name].iloc[-1]) else 0.0
                    row_data[f'{ticker}'] = latest_value
                else:
                    # No data found for this ticker
                    row_data[f'{ticker}'] = np.nan
                        
            display_rows.append(row_data)
        
        # Add stage analysis columns if available
        try:
            # Load stage analysis file based on timeframe and date
            stage_analysis_file = config.directories['RESULTS_DIR'] / 'stage_analysis' / f'stage_analysis_0_{timeframe}_{data_date}.csv'
            
            if stage_analysis_file.exists():
                stage_df = pd.read_csv(stage_analysis_file)
                
                # Define stage analysis metrics to add
                stage_metrics = [
                    ('Stage Code', 'daily_stage_code', 'Market stage classification code'),
                    ('Stage Name', 'daily_stage_name', 'Market stage classification name'),
                    ('MA Alignment', 'daily_ma_alignment', 'Moving average alignment trend')
                ]
                
                # Add stage analysis rows
                for metric_name, column_name, description in stage_metrics:
                    if column_name in stage_df.columns:
                        row_data = {
                            'Metric': metric_name,
                            'Description': description
                        }
                        
                        # Get stage analysis data for each ticker
                        for ticker in ticker_list:
                            ticker_stage_data = stage_df[stage_df['ticker'] == ticker]
                            
                            if not ticker_stage_data.empty and column_name in ticker_stage_data.columns:
                                # Get the latest value for this ticker
                                latest_value = ticker_stage_data[column_name].iloc[-1] if not pd.isna(ticker_stage_data[column_name].iloc[-1]) else 'N/A'
                                row_data[f'{ticker}'] = latest_value
                            else:
                                # No data found for this ticker
                                row_data[f'{ticker}'] = 'N/A'
                        
                        display_rows.append(row_data)
                        
                logger.info(f"Stage analysis columns added from: {stage_analysis_file}")
            else:
                logger.info(f"Stage analysis file not found: {stage_analysis_file}")
                
        except Exception as e:
            logger.warning(f"Could not load stage analysis data: {e}")
        
        # Create DataFrame
        display_df = pd.DataFrame(display_rows)
        
        # Use the caller's data date for consistent filename generation
        safe_user_choice = str(user_config.ticker_choice).replace('-', '_')
        output_file = output_path / f'pctChgRS_pctChg_{analysis_type}_{safe_user_choice}_{timeframe}_{data_date}.csv'
        
        display_df.to_csv(output_file, index=False, float_format='%.4f')
        
        # File logging removed - using CSV files directly
        
        logger.info(f"Index percentage change display generated: {output_file}")
        print(f"📊 Index Percentage Change Display Generated:")
        print(f"  • Output file: {output_file.name}")
        print(f"  • Tickers: {', '.join(ticker_list)}")
        print(f"  • Data date: {data_date}")
        print(f"  • Metrics analyzed: {len(display_rows)}")
        print(f"  • Format: Metric | Description | {ticker_list[0]} | {ticker_list[1]} | ...")
        
        # Display sample data for verification
        for ticker in ticker_list:
            ticker_count = (calc_df['ticker'] == ticker).sum()
            print(f"  • {ticker}: {ticker_count} records")
        
        # Note: PDF generation happens at the end of the program via reportlab under /results/reports/
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error generating index percentage change display: {e}")
        return None


# PDF generation removed - handled by reportlab at end of pipeline under /results/reports/


def get_index_current_changes(calc_df, index_names):
    """
    Get current percentage changes for specified indexes.
    
    Args:
        calc_df: DataFrame with calculation results
        index_names: List of index names to analyze
        
    Returns:
        dict: Current percentage changes by index
    """
    current_changes = {}
    
    for index_name in index_names:
        if index_name in calc_df.columns:
            if index_name.startswith('^'):
                # Handle index symbols directly
                index_data = calc_df[calc_df['ticker'] == index_name]
            else:
                # Handle boolean index membership columns
                index_mask = (calc_df[index_name] == True) | (calc_df[index_name] == 'True') | (calc_df[index_name] == 1)
                index_data = calc_df[index_mask]
            
            if not index_data.empty and 'daily_pct_change' in index_data.columns:
                avg_change = index_data['daily_pct_change'].mean()
                current_changes[index_name] = {
                    'daily_change_avg': avg_change,
                    'daily_change_median': index_data['daily_pct_change'].median(),
                    'members_count': len(index_data)
                }
    
    return current_changes


def run_index_pctchg_analysis(output_path, timeframe, user_config, config, data_date=None, analysis_type='tickers'):
    """
    Main entry point for index percentage change analysis.
    
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
            # Construct filename directly using centralized data date
            safe_user_choice = str(user_config.ticker_choice).replace('-', '_')
            calc_file = config.directories['BASIC_CALCULATION_DIR'] / f'basic_calculation_{safe_user_choice}_{timeframe}_{data_date}.csv'
            logger.info(f"Using centralized data date for {timeframe} pctChg analysis: {data_date}")
        else:
            # Fall back to file discovery (legacy mode)
            from ..basic_calculations import find_latest_basic_calculation_file
            calc_file = find_latest_basic_calculation_file(config, timeframe, user_config.ticker_choice)
            logger.warning(f"Using file discovery fallback for {timeframe} pctChg analysis")
        
        if not calc_file or not calc_file.exists():
            logger.warning(f"Basic calculation file not found for {timeframe}: {calc_file}")
            return None
            
        calc_df = pd.read_csv(calc_file)
        
        if calc_df.empty:
            logger.warning("Basic calculation data is empty")
            return None
            
        # Generate the percentage change display
        return generate_index_pctchg_display(calc_df, output_path, timeframe, user_config, data_date, config, analysis_type)
        
    except Exception as e:
        logger.error(f"Error in index percentage change analysis: {e}")
        return None