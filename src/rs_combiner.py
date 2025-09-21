"""
Relative Strength File Combiner
===============================

Combines multiple RS CSV files with different periods into single unified datasets.
Merges files by ticker and applies proper naming conventions with frequency suffixes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class RSFileCombiner:
    """
    Combines multiple RS period files into unified datasets with proper naming conventions.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize RS file combiner.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object
        """
        self.config = config
        self.user_config = user_config
        self.rs_dir = config.directories['RESULTS_DIR'] / 'rs'
        
    def combine_rs_files(self, ticker_choice=0):
        """
        Combine all RS files into unified datasets.
        
        Args:
            ticker_choice: User ticker choice number
            
        Returns:
            Dictionary with paths to combined files
        """
        logger.info("Starting RS file combination process...")
        
        combined_files = {}
        
        if not self.rs_dir.exists():
            logger.warning("RS directory not found")
            return combined_files
        
        # Find and group RS files
        rs_files = self._find_rs_files(ticker_choice)
        
        if not rs_files:
            logger.warning("No RS files found to combine")
            return combined_files
        
        logger.info(f"Found {sum(len(files) for files in rs_files.values())} RS files to combine")
        
        # Combine files for each category
        for category, files in rs_files.items():
            if files:
                try:
                    combined_file = self._combine_category_files(category, files, ticker_choice)
                    if combined_file:
                        combined_files[category] = combined_file
                        logger.info(f"Combined {category}: {combined_file}")
                except Exception as e:
                    logger.error(f"Error combining {category} files: {e}")
        
        logger.info(f"RS file combination completed. Created {len(combined_files)} combined files")
        return combined_files
    
    def _find_rs_files(self, ticker_choice):
        """
        Find and categorize all RS files.
        
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
        
        # Pattern to match RS files: rs_{benchmark}_{method}_{level}_{timeframe}_{choice}_{date}.csv
        # Updated pattern for new naming convention with benchmark ticker
        new_pattern = re.compile(r'rs_(\w+)_(\w+)_(\w+)_(\w+)_(\d+)_(\d+)\.csv')

        # Legacy pattern for old naming convention: rs_ibd_{level}_{timeframe}_{choice}_{date}_{period}period.csv
        legacy_pattern = re.compile(r'rs_ibd_(\w+)_(\w+)_(\d+)_(\d+)_(\d+)period\.csv')

        # Check both new and legacy patterns
        for file_path in self.rs_dir.glob('rs_*.csv'):
            # Try new pattern first: rs_{benchmark}_{method}_{level}_{timeframe}_{choice}_{date}.csv
            new_match = new_pattern.match(file_path.name)
            if new_match:
                benchmark, method, level, timeframe, choice, date = new_match.groups()

                # Only include files for the specified ticker choice
                if int(choice) == ticker_choice:
                    category = f"{level}_{timeframe}"
                    if category in rs_files:
                        rs_files[category].append({
                            'path': file_path,
                            'level': level,
                            'timeframe': timeframe,
                            'benchmark': benchmark,
                            'method': method,
                            'date': date
                        })
                continue

            # Try legacy pattern: rs_ibd_{level}_{timeframe}_{choice}_{date}_{period}period.csv
            legacy_match = legacy_pattern.match(file_path.name)
            if legacy_match:
                level, timeframe, choice, date, period = legacy_match.groups()

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
    
    def _combine_category_files(self, category, files, ticker_choice):
        """
        Combine files for a specific category (e.g., stocks_daily).
        
        Args:
            category: Category name (e.g., 'stocks_daily')
            files: List of file information dictionaries
            ticker_choice: User ticker choice number
            
        Returns:
            Path to combined file or None if failed
        """
        logger.info(f"Combining {len(files)} files for {category}...")
        
        level, timeframe = category.split('_')
        frequency_suffix = 'd' if timeframe == 'daily' else 'w'
        
        combined_data = None
        
        for file_info in files:
            try:
                # Read the file
                df = pd.read_csv(file_info['path'], index_col=0)
                
                if df.empty:
                    logger.debug(f"Skipping empty file: {file_info['path'].name}")
                    continue
                
                # Rename columns with appropriate suffixes
                renamed_df = self._rename_columns_with_suffix(
                    df, file_info['period'], frequency_suffix
                )
                
                # Merge with existing data
                if combined_data is None:
                    combined_data = renamed_df.copy()
                else:
                    # Handle overlapping columns by using suffixes and then merging duplicates
                    combined_data = combined_data.join(
                        renamed_df, how='outer', rsuffix=f'_dup_{file_info["period"]}'
                    )
                
                logger.debug(f"Added {file_info['period']}-period data: {len(renamed_df)} tickers")
                
            except Exception as e:
                logger.error(f"Error processing file {file_info['path'].name}: {e}")
                continue
        
        if combined_data is None or combined_data.empty:
            logger.warning(f"No data to combine for {category}")
            return None
        
        # Clean up duplicate columns
        combined_data = self._cleanup_duplicate_columns(combined_data)
        
        # Add metadata columns
        combined_data = self._add_metadata_columns(combined_data, files, frequency_suffix)
        
        # Generate output filename
        date_str = datetime.now().strftime('%Y%m%d')
        output_filename = f"rs_combined_{level}_{ticker_choice}_{timeframe}_{date_str}.csv"
        output_path = self.rs_dir / output_filename
        
        # Save combined file
        combined_data.to_csv(output_path, float_format='%.4f')
        
        logger.info(f"Saved combined {category}: {len(combined_data)} tickers, {len(combined_data.columns)} columns")
        return output_path
    
    def _rename_columns_with_suffix(self, df, period, frequency_suffix):
        """
        Rename columns with appropriate period and frequency suffixes.
        
        Args:
            df: DataFrame to rename
            period: Period number
            frequency_suffix: Frequency suffix ('d' or 'w')
            
        Returns:
            DataFrame with renamed columns
        """
        renamed_df = df.copy()
        
        # Define column renaming rules
        column_mapping = {}
        
        for col in df.columns:
            if col == 'ticker':
                # Keep ticker as is (it's the index anyway)
                continue
            elif col == 'calculation_date':
                # Add frequency suffix to distinguish daily vs weekly calculation dates
                column_mapping[col] = f'calculation_date_{frequency_suffix}'
            elif col == 'benchmark_return':
                # Benchmark return is the same for all periods in a timeframe
                column_mapping[col] = f'benchmark_return_{frequency_suffix}'
            elif col.endswith('_benchmark_return'):
                # daily_benchmark_return -> daily_benchmark_return_d (timeframe-prefixed)
                column_mapping[col] = f'{col}_{frequency_suffix}'
            elif col.startswith('rs_percentile_'):
                # Percentile columns: rs_percentile_5 -> rs_percentile_5d (handle first to avoid rs_ match)
                parts = col.split('_')
                if len(parts) >= 3:
                    period_from_col = parts[2]
                    column_mapping[col] = f'rs_percentile_{period_from_col}{frequency_suffix}'
                else:
                    column_mapping[col] = f'{col}{frequency_suffix}'
            elif col.startswith('rs_'):
                # RS columns: rs_5 -> rs_5d, rs_10 -> rs_10d
                period_from_col = col.split('_')[1]
                column_mapping[col] = f'rs_{period_from_col}{frequency_suffix}'
            elif col.startswith('stock_return_'):
                # Stock return columns: stock_return_5 -> stock_return_5d
                period_from_col = col.split('_')[2]
                column_mapping[col] = f'stock_return_{period_from_col}{frequency_suffix}'
            else:
                # Other columns: add frequency suffix
                column_mapping[col] = f'{col}_{frequency_suffix}'
        
        # Apply renaming
        renamed_df = renamed_df.rename(columns=column_mapping)
        
        return renamed_df
    
    def _cleanup_duplicate_columns(self, combined_data):
        """
        Clean up duplicate columns created during joining process.
        
        Args:
            combined_data: DataFrame with potential duplicate columns
            
        Returns:
            DataFrame with duplicate columns cleaned up
        """
        # Find columns that have duplicates (ending with _dup_X)
        duplicate_pattern = re.compile(r'(.+)_dup_\d+$')
        
        columns_to_remove = []
        
        for col in combined_data.columns:
            match = duplicate_pattern.match(col)
            if match:
                original_col = match.group(1)
                
                if original_col in combined_data.columns:
                    # The original column exists, so we can remove the duplicate
                    # But first, fill any NaN values in original with duplicate values
                    original_series = combined_data[original_col]
                    duplicate_series = combined_data[col]
                    
                    # Combine the series, preferring original values
                    combined_series = original_series.fillna(duplicate_series)
                    combined_data[original_col] = combined_series
                    
                    # Mark duplicate column for removal
                    columns_to_remove.append(col)
        
        # Remove duplicate columns
        if columns_to_remove:
            combined_data = combined_data.drop(columns=columns_to_remove)
            logger.debug(f"Removed {len(columns_to_remove)} duplicate columns")
        
        return combined_data
    
    def _add_metadata_columns(self, combined_data, files, frequency_suffix):
        """
        Add metadata columns to the combined dataset.
        
        Args:
            combined_data: Combined DataFrame
            files: List of source file information
            frequency_suffix: Frequency suffix ('d' or 'w')
            
        Returns:
            DataFrame with metadata columns added
        """
        # Add timeframe identifier
        timeframe_name = 'daily' if frequency_suffix == 'd' else 'weekly'
        combined_data[f'timeframe'] = timeframe_name
        
        # Add periods analyzed
        periods = [f['period'] for f in files]
        combined_data[f'periods_analyzed_{frequency_suffix}'] = ','.join(map(str, sorted(periods)))
        
        # Add file count
        combined_data[f'source_files_{frequency_suffix}'] = len(files)
        
        # Add combination date
        combined_data[f'combined_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return combined_data
    
    def get_combination_summary(self, combined_files):
        """
        Generate a summary of the combination process.
        
        Args:
            combined_files: Dictionary of combined file paths
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_combined_files': len(combined_files),
            'categories_processed': list(combined_files.keys()),
            'file_details': {}
        }
        
        for category, file_path in combined_files.items():
            try:
                # Read first few rows to get basic info
                df = pd.read_csv(file_path, nrows=5)
                
                summary['file_details'][category] = {
                    'path': str(file_path),
                    'columns': len(df.columns),
                    'sample_columns': list(df.columns[:10]),  # First 10 columns
                    'file_size_mb': round(file_path.stat().st_size / (1024*1024), 2)
                }
            except Exception as e:
                summary['file_details'][category] = {
                    'path': str(file_path),
                    'error': str(e)
                }
        
        return summary
    
    def create_rs_master_file(self, combined_files, ticker_choice=0):
        """
        Create a master file combining both daily and weekly data where possible.
        
        Args:
            combined_files: Dictionary of combined file paths
            ticker_choice: User ticker choice number
            
        Returns:
            Path to master file or None
        """
        logger.info("Creating RS master file with both daily and weekly data...")
        
        daily_files = {k: v for k, v in combined_files.items() if 'daily' in k}
        weekly_files = {k: v for k, v in combined_files.items() if 'weekly' in k}
        
        if not daily_files or not weekly_files:
            logger.info("Cannot create master file - need both daily and weekly data")
            return None
        
        master_data = {}
        
        # Process each level (stocks, sectors, industries)
        for level in ['stocks', 'sectors', 'industries']:
            daily_key = f"{level}_daily"
            weekly_key = f"{level}_weekly"
            
            if daily_key in combined_files and weekly_key in combined_files:
                try:
                    # Read both files
                    daily_df = pd.read_csv(combined_files[daily_key], index_col=0)
                    weekly_df = pd.read_csv(combined_files[weekly_key], index_col=0)
                    
                    # Merge on ticker
                    master_df = daily_df.join(weekly_df, how='outer', rsuffix='_weekly_dup')
                    
                    # Remove duplicate columns
                    master_df = master_df.loc[:, ~master_df.columns.str.endswith('_weekly_dup')]
                    
                    master_data[level] = master_df
                    
                    logger.info(f"Master {level}: {len(master_df)} tickers, {len(master_df.columns)} columns")
                    
                except Exception as e:
                    logger.error(f"Error creating master file for {level}: {e}")
        
        if not master_data:
            logger.warning("No master data created")
            return None
        
        # Save master files
        date_str = datetime.now().strftime('%Y%m%d')
        master_files = {}
        
        for level, df in master_data.items():
            output_filename = f"rs_master_{level}_{ticker_choice}_multi_{date_str}.csv"
            output_path = self.rs_dir / output_filename
            df.to_csv(output_path, float_format='%.4f')
            master_files[level] = output_path
            
            logger.info(f"Saved master {level} file: {output_path}")
        
        return master_files


def combine_rs_files(config, user_config, ticker_choice=0):
    """
    Standalone function to combine RS files.
    
    Args:
        config: Config object
        user_config: User configuration object
        ticker_choice: User ticker choice number
        
    Returns:
        Dictionary with combination results
    """
    logger.info("Starting RS file combination...")
    
    combiner = RSFileCombiner(config, user_config)
    
    # Combine files by category
    combined_files = combiner.combine_rs_files(ticker_choice)
    
    # Create master files if possible
    master_files = None
    if len(combined_files) >= 2:  # Need at least daily and weekly
        master_files = combiner.create_rs_master_file(combined_files, ticker_choice)
    
    # Generate summary
    summary = combiner.get_combination_summary(combined_files)
    
    results = {
        'combined_files': combined_files,
        'master_files': master_files or {},
        'summary': summary
    }
    
    logger.info("RS file combination completed")
    return results