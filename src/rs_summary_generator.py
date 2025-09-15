"""
RS Summary File Generator
========================

Creates wide-format summary files by horizontally joining multiple RS period files by ticker.
Each ticker gets one row with all period attributes as columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RSSummaryGenerator:
    """
    Generates wide-format summary files by horizontally joining RS period files.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize RS summary generator.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object
        """
        self.config = config
        self.user_config = user_config
        self.rs_dir = config.directories['RESULTS_DIR'] / 'rs'
        
    def create_wide_format_summaries(self, ticker_choice=0):
        """
        Create wide-format summary files for all categories.
        
        Args:
            ticker_choice: User ticker choice number
            
        Returns:
            Dictionary with paths to generated summary files
        """
        logger.info("Creating wide-format RS summary files...")
        
        summary_files = {}
        
        if not self.rs_dir.exists():
            logger.warning("RS directory not found")
            return summary_files
            
        # Find and group RS period files
        rs_files = self._find_period_files(ticker_choice)
        
        if not rs_files:
            logger.warning("No RS period files found for summary generation")
            return summary_files
            
        # Create summary for each category
        for category, files in rs_files.items():
            if files:
                try:
                    summary_file = self._create_category_summary(category, files, ticker_choice)
                    if summary_file:
                        summary_files[category] = summary_file
                        logger.info(f"Created wide-format summary for {category}: {summary_file}")
                except Exception as e:
                    logger.error(f"Error creating summary for {category}: {e}")
        
        logger.info(f"Generated {len(summary_files)} wide-format summary files")
        return summary_files
    
    def _find_period_files(self, ticker_choice):
        """
        Find and categorize all RS period files.
        
        Args:
            ticker_choice: User ticker choice number
            
        Returns:
            Dictionary of categorized file lists
        """
        rs_files = {
            'stocks_daily': [],
            'stocks_weekly': [],
            'sectors_daily': [],
            'sectors_weekly': [],
            'industries_daily': [],
            'industries_weekly': []
        }
        
        # Pattern to match RS period files: rs_ibd_{level}_{timeframe}_{choice}_{date}_{period}period.csv
        pattern = re.compile(r'rs_ibd_(\\w+)_(\\w+)_(\\d+)_(\\d+)_(\\d+)period\\.csv')
        
        for file_path in self.rs_dir.glob('rs_ibd_*period.csv'):
            match = pattern.match(file_path.name)
            if match:
                level, timeframe, choice, date, period = match.groups()
                
                # Only include files for the specified ticker choice
                if int(choice) == ticker_choice:
                    category = f"{level}_{timeframe}"
                    if category in rs_files:
                        rs_files[category].append({
                            'path': file_path,
                            'level': level,
                            'timeframe': timeframe,
                            'period': int(period),
                            'date': date
                        })
        
        # Sort files by period for consistent processing
        for category in rs_files:
            rs_files[category].sort(key=lambda x: x['period'])
        
        return rs_files
    
    def _create_category_summary(self, category, files, ticker_choice):
        """
        Create wide-format summary for a specific category.
        
        Args:
            category: Category name (e.g., 'stocks_daily')
            files: List of file information dictionaries
            ticker_choice: User ticker choice number
            
        Returns:
            Path to summary file or None if failed
        """
        logger.info(f"Creating wide-format summary for {category} with {len(files)} period files...")
        
        level, timeframe = category.split('_')
        frequency_suffix = 'd' if timeframe == 'daily' else 'w'
        
        # Start with the first period file as base
        if not files:
            return None
            
        try:
            # Read the first period file
            base_df = pd.read_csv(files[0]['path'], index_col=0)
            if base_df.empty:
                logger.warning(f"Base file is empty for {category}")
                return None
                
            # Initialize the summary dataframe with first period
            summary_df = self._format_period_data(base_df, files[0]['period'], frequency_suffix)
            
            # Add data from remaining period files
            for file_info in files[1:]:
                period_df = pd.read_csv(file_info['path'], index_col=0)
                if not period_df.empty:
                    formatted_period_df = self._format_period_data(
                        period_df, file_info['period'], frequency_suffix
                    )
                    
                    # Merge horizontally by ticker (index)
                    summary_df = summary_df.join(
                        formatted_period_df[self._get_period_columns(file_info['period'], frequency_suffix)],
                        how='outer'
                    )
            
            # Add metadata
            summary_df = self._add_summary_metadata(summary_df, files, frequency_suffix)
            
            # Generate output filename
            date_str = datetime.now().strftime('%Y%m%d')
            output_filename = f"rs_wide_summary_{level}_{ticker_choice}_{timeframe}_{date_str}.csv"
            output_path = self.rs_dir / output_filename
            
            # Save summary file
            summary_df.to_csv(output_path, index=False, float_format='%.4f')
            
            logger.info(f"Saved wide-format summary: {len(summary_df)} tickers, {len(summary_df.columns)} columns")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating summary for {category}: {e}")
            return None
    
    def _format_period_data(self, df, period, frequency_suffix):
        """
        Format period data with appropriate column names.
        
        Args:
            df: Period DataFrame
            period: Period number
            frequency_suffix: Frequency suffix ('d' or 'w')
            
        Returns:
            Formatted DataFrame
        """
        formatted_df = df.copy()
        
        # Column mapping for this period
        column_mapping = {}
        
        for col in df.columns:
            if col == 'ticker':
                continue  # Skip ticker as it's the index
            elif col.startswith('rs_percentile_'):
                # rs_percentile_5 -> rs_percentile_5d
                parts = col.split('_')
                if len(parts) >= 3:
                    period_from_col = parts[2]
                    column_mapping[col] = f'rs_percentile_{period_from_col}{frequency_suffix}'
            elif col.startswith('rs_'):
                # rs_5 -> rs_5d
                parts = col.split('_')
                if len(parts) >= 2:
                    period_from_col = parts[1]
                    column_mapping[col] = f'rs_{period_from_col}{frequency_suffix}'
            elif col.startswith('stock_return_'):
                # stock_return_5 -> stock_return_5d
                parts = col.split('_')
                if len(parts) >= 3:
                    period_from_col = parts[2]
                    column_mapping[col] = f'stock_return_{period_from_col}{frequency_suffix}'
            elif col == 'benchmark_return':
                # benchmark_return -> benchmark_return_d (shared across periods)
                column_mapping[col] = f'benchmark_return_{frequency_suffix}'
            elif col.endswith('_benchmark_return'):
                # daily_benchmark_return -> daily_benchmark_return_d (timeframe-prefixed)
                column_mapping[col] = f'{col}_{frequency_suffix}'
            elif col == 'calculation_date':
                # calculation_date -> calculation_date_d (shared across periods)
                column_mapping[col] = f'calculation_date_{frequency_suffix}'
            else:
                # Other columns get period-specific naming
                column_mapping[col] = f'{col}_{period}{frequency_suffix}'
        
        # Apply column renaming
        formatted_df = formatted_df.rename(columns=column_mapping)
        
        return formatted_df
    
    def _get_period_columns(self, period, frequency_suffix):
        """
        Get list of columns specific to this period (excludes shared columns).
        
        Args:
            period: Period number
            frequency_suffix: Frequency suffix ('d' or 'w')
            
        Returns:
            List of column names for this period
        """
        # These are the period-specific columns we want to merge
        return [
            f'rs_{period}{frequency_suffix}',
            f'stock_return_{period}{frequency_suffix}',
            f'rs_percentile_{period}{frequency_suffix}'
        ]
    
    def _add_summary_metadata(self, summary_df, files, frequency_suffix):
        """
        Add metadata columns to the summary.
        
        Args:
            summary_df: Summary DataFrame
            files: List of source file information
            frequency_suffix: Frequency suffix ('d' or 'w')
            
        Returns:
            DataFrame with metadata columns added
        """
        # Add timeframe identifier
        timeframe_name = 'daily' if frequency_suffix == 'd' else 'weekly'
        summary_df[f'timeframe'] = timeframe_name
        
        # Add periods covered
        periods = [f['period'] for f in files]
        summary_df[f'periods_covered'] = ','.join(map(str, sorted(periods)))
        
        # Add summary creation date
        summary_df[f'summary_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return summary_df


def generate_wide_format_summaries(config, user_config, ticker_choice=0):
    """
    Standalone function to generate wide-format RS summary files.
    
    Args:
        config: Config object
        user_config: User configuration object  
        ticker_choice: User ticker choice number
        
    Returns:
        Dictionary with summary file paths
    """
    logger.info("Starting wide-format RS summary generation...")
    
    generator = RSSummaryGenerator(config, user_config)
    summary_files = generator.create_wide_format_summaries(ticker_choice)
    
    logger.info(f"Generated {len(summary_files)} wide-format summary files")
    return summary_files