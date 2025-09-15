"""
Index Overview Counts Generator
==============================

Generates index counts analysis files with percentage analysis for configured universes.
Uses the UniverseFilter system to ensure consistency with PER/RS analysis by filtering
data using the same ticker_universe files.

This module creates ***_counts_daily/weekly/monthly format files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from ..universe_filter import UniverseFilter, filter_dataframe_by_universe

logger = logging.getLogger(__name__)


def generate_index_counts_analysis(calc_df, output_path, timeframe, user_config, data_date, config=None):
    """
    Generate index counts analysis table (percentage analysis with rename).
    
    Args:
        calc_df: DataFrame with basic calculation results
        output_path: Path to save the analysis file
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration with index settings
        data_date: Data date for filename generation (e.g., '20250829')
        config: Config object (optional)
        
    Returns:
        str: Path to generated counts analysis file
    """
    try:
        # Parse configured universes from user settings (NEW: use universe filter configuration)
        # Try new universe_filter setting first, fallback to old indexes_counts for compatibility
        if hasattr(user_config, 'index_overview_universe_filter') and user_config.index_overview_universe_filter:
            universes_str = user_config.index_overview_universe_filter.strip()
            logger.info("Using new universe filtering configuration")
        else:
            # Fallback to old configuration for compatibility
            universes_str = user_config.index_overview_indexes_counts.strip() if user_config.index_overview_indexes_counts else 'SP500'
            logger.warning("Using deprecated indexes_counts configuration, consider migrating to universe_filter")
            
        # Handle both comma and semicolon delimiters
        if ';' in universes_str:
            universe_names = [idx.strip().upper() for idx in universes_str.split(';') if idx.strip()]
        else:
            universe_names = [idx.strip().upper() for idx in universes_str.split(',') if idx.strip()]
        
        # Initialize universe filter
        universe_filter = UniverseFilter(config) if config else None
        
        if not universe_filter:
            logger.error("Config object required for universe filtering")
            return None
        
        # Define metrics with Above/Below logic for counts analysis
        # Dynamic column names based on timeframe (daily_, weekly_, monthly_)
        prefix = f"{timeframe}_"
        
        # Timeframe-specific column patterns
        if timeframe == 'daily':
            short_change_col = f'{timeframe}_{timeframe}_{timeframe}_1d_pct_change'
            month_change_col = f'{timeframe}_{timeframe}_monthly_22d_pct_change'
            quarter_change_col = f'{timeframe}_{timeframe}_quarterly_66d_pct_change'
        elif timeframe == 'weekly':
            short_change_col = f'{timeframe}_{timeframe}_{timeframe}_1w_pct_change'
            month_change_col = f'{timeframe}_{timeframe}_{timeframe}_5w_pct_change'  # ~1 month
            quarter_change_col = f'{timeframe}_{timeframe}_{timeframe}_15w_pct_change'  # ~3 months
        elif timeframe == 'monthly':
            short_change_col = f'{timeframe}_{timeframe}_{timeframe}_1m_pct_change'
            month_change_col = f'{timeframe}_{timeframe}_{timeframe}_3m_pct_change'
            quarter_change_col = f'{timeframe}_{timeframe}_{timeframe}_6m_pct_change'
        
        metrics_definitions = [
            # Change metrics - each metric has Above% and Below%
            ('Day Change (%)', '1-day % change: (Close_t - Close_t-1)/Close_t-1 * 100', short_change_col, lambda x: x > 0, lambda x: x < 0),
            ('Month Change (%)', '1-month: (Close_t - Close_t-22)/Close_t-22 * 100', month_change_col, lambda x: x > 0, lambda x: x < 0),
            ('Quarter Change (%)', '1-quarter: (Close_t - Close_t-66)/Close_t-66 * 100', quarter_change_col, lambda x: x > 0, lambda x: x < 0),
            ('Half Year Change (%)', '6-month: (Close_t - Close_t-120)/Close_t-120 * 100', f'{prefix}half_year_pct_change', lambda x: x > 0, lambda x: x < 0),
            
            # Price vs MA metrics
            ('Price vs SMA 10', 'Price compared to 10-day Simple Moving Average', f'{prefix}priceabovesma10', lambda x: x == True, lambda x: x == False),
            ('Price vs EMA 20', 'Price compared to 20-day Exponential Moving Average', f'{prefix}priceaboveema20', lambda x: x == True, lambda x: x == False),
            ('Price vs SMA 50', 'Price compared to 50-day Simple Moving Average', f'{prefix}priceabovesma50', lambda x: x == True, lambda x: x == False),
            
            # MA Comparison metrics
            ('SMA 10 vs SMA 20', 'Short-term MA compared to medium-term MA', f'{prefix}sma10vssma20', lambda x: x == True, lambda x: x == False),
            ('SMA 20 vs SMA 50', 'Medium-term MA compared to long-term MA', f'{prefix}sma20vssma50', lambda x: x == True, lambda x: x == False),
            ('SMA 50 vs SMA 200', 'Long-term MA compared to very long-term MA', f'{prefix}sma50vssma200', lambda x: x == True, lambda x: x == False),
            
            # Trend Analysis
            ('Trend Strength', 'Percentage of up days in last 10 trading days', 'trend_days_10_pct', lambda x: x > 60, lambda x: x < 40),
            ('Perfect Bull Alignment', 'SMA 20 > SMA 50 > SMA 200 alignment', f'{prefix}perfectbullishalignment', lambda x: x == True, lambda x: x == False),
            
            # High/Low Analysis
            ('20-Day Position', 'Current price vs 20-day high/low range', f'{prefix}at_20day_high', lambda x: x == True, None),
            ('20-Day Low Position', 'Current price vs 20-day low', f'{prefix}at_20day_low', lambda x: x == True, None),
            
            # RSI Analysis
            ('RSI Momentum', 'RSI overbought vs oversold conditions', f'{prefix}rsi_14', lambda x: x > 70, lambda x: x < 30),
        ]
        
        # Create analysis table with multiple universe columns
        analysis_rows = []
        
        for metric_name, calculation_logic, column_name, above_condition, below_condition in metrics_definitions:
            row_data = {
                'Metric': metric_name,
                'Description': calculation_logic
            }
            
            # Calculate for each configured universe using UniverseFilter
            for universe_name in universe_names:
                try:
                    # NEW: Use UniverseFilter to get universe members (consistent with PER/RS)
                    universe_df = universe_filter.filter_dataframe_by_universe(
                        calc_df, universe_name, ticker_column='ticker'
                    )
                    total_members = len(universe_df)
                    
                    if total_members > 0 and column_name in universe_df.columns:
                        # Calculate Above%
                        above_count = universe_df[column_name].apply(above_condition).sum() if above_condition else 0
                        above_pct = (above_count / total_members) * 100
                        
                        # Calculate Below%
                        below_count = universe_df[column_name].apply(below_condition).sum() if below_condition else 0
                        below_pct = (below_count / total_members) * 100
                        
                        row_data[f'{universe_name}_Above%'] = above_pct
                        row_data[f'{universe_name}_Below%'] = below_pct
                        
                        logger.debug(f"Universe {universe_name}: {total_members} members, {above_count} above, {below_count} below for {metric_name}")
                    else:
                        row_data[f'{universe_name}_Above%'] = 0.0
                        row_data[f'{universe_name}_Below%'] = 0.0
                        logger.warning(f"No data for universe {universe_name} and metric {metric_name}")
                        
                except Exception as e:
                    logger.error(f"Error processing universe {universe_name} for metric {metric_name}: {e}")
                    row_data[f'{universe_name}_Above%'] = 0.0
                    row_data[f'{universe_name}_Below%'] = 0.0
                        
            analysis_rows.append(row_data)
        
        # Create DataFrame
        analysis_df = pd.DataFrame(analysis_rows)
        
        # Use the caller's data date for consistent filename generation
        safe_user_choice = str(user_config.ticker_choice).replace('-', '_')
        output_file = output_path / f'index_counts_{safe_user_choice}_{timeframe}_{data_date}.csv'
        
        analysis_df.to_csv(output_file, index=False, float_format='%.2f')
        
        # File logging removed - using CSV files directly
        
        logger.info(f"Index counts analysis generated: {output_file}")
        print(f"📊 Index Counts Analysis Generated (Universe Filtered):")
        print(f"  • Output file: {output_file.name}")
        print(f"  • Universes: {', '.join(universe_names)}")
        print(f"  • Data date: {data_date}")
        print(f"  • Metrics analyzed: {len(analysis_rows)}")
        print(f"  • Format: Metric | Logic | {universe_names[0]}_Above% | {universe_names[0]}_Below% | ...")
        print(f"  • Filtering method: ticker_universe files (consistent with PER/RS)")
        
        # Display member counts for each universe using UniverseFilter
        for universe_name in universe_names:
            try:
                universe_members = universe_filter.get_universe_members(universe_name)
                # Filter calc_df to see how many of those members are actually in our dataset
                universe_df = universe_filter.filter_dataframe_by_universe(calc_df, universe_name, 'ticker')
                print(f"  • {universe_name}: {len(universe_df)} members in dataset (total: {len(universe_members)} in universe)")
            except Exception as e:
                print(f"  • {universe_name}: Error getting member count - {e}")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error generating index counts analysis: {e}")
        return None


def run_index_counts_analysis(output_path, timeframe, user_config, config, data_date=None):
    """
    Main entry point for index counts analysis generation.
    
    Args:
        output_path: Directory to save analysis files
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration
        config: Config object
        data_date: Centralized data date (e.g., '20250829'), if None will use file discovery
        
    Returns:
        str: Path to generated analysis file or None if failed
    """
    try:
        # Use centralized data date if provided, otherwise fall back to file discovery
        if data_date:
            # Construct filename directly using centralized data date
            safe_user_choice = str(user_config.ticker_choice).replace('-', '_')
            calc_file = config.directories['BASIC_CALCULATION_DIR'] / f'basic_calculation_{safe_user_choice}_{timeframe}_{data_date}.csv'
            logger.info(f"Using centralized data date for {timeframe} counts analysis: {data_date}")
        else:
            # Fall back to file discovery (legacy mode)
            from ..basic_calculations import find_latest_basic_calculation_file
            calc_file = find_latest_basic_calculation_file(config, timeframe, user_config.ticker_choice)
            logger.warning(f"Using file discovery fallback for {timeframe} counts analysis")
        
        if not calc_file or not calc_file.exists():
            logger.warning(f"Basic calculation file not found for {timeframe}: {calc_file}")
            return None
            
        calc_df = pd.read_csv(calc_file)
        
        if calc_df.empty:
            logger.warning("Basic calculation data is empty")
            return None
            
        # Generate the counts analysis
        return generate_index_counts_analysis(calc_df, output_path, timeframe, user_config, data_date, config)
        
    except Exception as e:
        logger.error(f"Error in index counts analysis: {e}")
        return None