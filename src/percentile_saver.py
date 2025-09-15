"""
Percentile File Saver
====================

Specialized file saver for RS percentile rankings with systematic naming.
Creates organized output files in the percentiles/ directory structure.

Features:
- Pure percentile ranking storage (separate from RS values)
- Multi-benchmark percentile support
- Systematic file naming and column organization
- IBD-style 1-99 percentile rankings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class PercentileSaver:
    """
    Specialized saver for RS percentile rankings with systematic naming.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize percentile saver.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object
        """
        self.config = config
        self.user_config = user_config
        
        # Create percentiles directory structure
        self.percentiles_dir = Path(config.directories['RESULTS_DIR']) / 'rs' / 'percentiles'
        self.percentiles_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Percentile Saver initialized: {self.percentiles_dir}")
    
    def save_multi_benchmark_percentiles(self, benchmark_percentiles: Dict[str, Dict[int, pd.DataFrame]], 
                                       level: str, choice: str, timeframe: str, 
                                       date_str: Optional[str] = None) -> List[Path]:
        """
        Save multi-benchmark percentile results to files.
        
        Args:
            benchmark_percentiles: Percentile results from SystematicPercentileCalculator
            level: Analysis level ('stocks', 'sectors', 'industries')
            choice: User ticker choice
            timeframe: Data timeframe
            date_str: Optional date string
            
        Returns:
            List of saved file paths
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')
        
        logger.info(f"Saving multi-benchmark percentiles for {level} (choice {choice}, {timeframe})")
        
        saved_files = []
        
        # Get all periods across all benchmarks
        all_periods = set()
        for benchmark_data in benchmark_percentiles.values():
            all_periods.update(benchmark_data.keys())
        
        if not all_periods:
            logger.warning("No periods found in percentile results")
            return saved_files
        
        # Only generate combined file with all periods
        combined_file = self._save_combined_percentiles_file(
            benchmark_percentiles, level, choice, timeframe, date_str
        )
        if combined_file:
            saved_files.append(combined_file)
        else:
            logger.warning("No combined percentile file generated")
        
        logger.info(f"Multi-benchmark percentile saving completed: {len(saved_files)} files created")
        return saved_files
    
    def _combine_period_percentiles(self, benchmark_percentiles: Dict[str, Dict[int, pd.DataFrame]], 
                                  period: int) -> Optional[pd.DataFrame]:
        """
        Combine percentile data from all benchmarks for a specific period.
        
        Args:
            benchmark_percentiles: Multi-benchmark percentile results
            period: Period to combine
            
        Returns:
            Combined DataFrame for the period or None
        """
        period_dfs = []
        
        for benchmark, benchmark_data in benchmark_percentiles.items():
            if period in benchmark_data:
                df = benchmark_data[period].copy()
                period_dfs.append(df)
        
        if not period_dfs:
            return None
        
        # Start with first benchmark as base
        combined_df = period_dfs[0].copy()
        
        # Merge additional benchmarks
        for df in period_dfs[1:]:
            # Merge on ticker column
            combined_df = pd.merge(combined_df, df, on='ticker', how='outer', suffixes=('', '_dup'))
            
            # Handle duplicate columns (keep first calculation_date)
            duplicate_cols = [col for col in combined_df.columns if col.endswith('_dup')]
            for dup_col in duplicate_cols:
                original_col = dup_col.replace('_dup', '')
                if original_col == 'calculation_date' and original_col in combined_df.columns:
                    # Keep the first calculation_date, drop duplicate
                    combined_df = combined_df.drop(columns=[dup_col])
                else:
                    # For other duplicates, log warning and drop
                    logger.warning(f"Unexpected duplicate column: {dup_col}")
                    combined_df = combined_df.drop(columns=[dup_col])
        
        return combined_df
    
    def _save_combined_percentiles_file(self, benchmark_percentiles: Dict[str, Dict[int, pd.DataFrame]], 
                                      level: str, choice: str, timeframe: str, 
                                      date_str: str) -> Optional[Path]:
        """
        Save a combined file with all periods and benchmarks for percentiles.
        
        Args:
            benchmark_percentiles: Multi-benchmark percentile results
            level: Analysis level
            choice: User ticker choice
            timeframe: Data timeframe
            date_str: Date string
            
        Returns:
            Path to saved combined file or None
        """
        logger.info("Creating combined periods percentile file")
        
        # Get all unique tickers across all results
        all_tickers = set()
        for benchmark_data in benchmark_percentiles.values():
            for period_data in benchmark_data.values():
                all_tickers.update(period_data['ticker'].tolist())
        
        if not all_tickers:
            logger.warning("No tickers found for combined percentile file")
            return None
        
        # Create base DataFrame with tickers
        combined_df = pd.DataFrame({'ticker': sorted(all_tickers)})
        
        # Add columns for each benchmark/period combination
        for benchmark, benchmark_data in benchmark_percentiles.items():
            for period, period_df in benchmark_data.items():
                # Find percentile columns for this benchmark/period
                percentile_columns = [col for col in period_df.columns if '_rs_per' in col]
                
                if percentile_columns:
                    # Add both RS values and percentiles
                    for per_col in percentile_columns:
                        period_data = period_df[['ticker', per_col]].copy()
                        
                        # Merge into combined DataFrame
                        combined_df = pd.merge(combined_df, period_data, on='ticker', how='left')
                        logger.info(f"Added percentile {per_col}: {len(period_data)} values")
                    
                    # Also add corresponding RS values for reference
                    rs_columns = [col for col in period_df.columns if '_rs_vs_' in col and benchmark in col]
                    for rs_col in rs_columns:
                        if rs_col in period_df.columns:
                            period_data = period_df[['ticker', rs_col]].copy()
                            combined_df = pd.merge(combined_df, period_data, on='ticker', how='left')
                            logger.info(f"Added RS reference {rs_col}: {len(period_data)} values")
        
        # Add metadata columns
        if benchmark_percentiles:
            first_benchmark = list(benchmark_percentiles.keys())[0]
            first_period = list(benchmark_percentiles[first_benchmark].keys())[0]
            sample_df = benchmark_percentiles[first_benchmark][first_period]
            
            if 'calculation_date' in sample_df.columns:
                combined_df['calculation_date'] = sample_df['calculation_date'].iloc[0]
        
        # Organize columns systematically before saving
        combined_df = self._organize_percentile_columns_systematically(combined_df)
        
        # Save combined file
        filename = f"percentiles_{level}_{choice}_{timeframe}_combined_{date_str}.csv"
        output_file = self.percentiles_dir / filename
        
        combined_df.to_csv(output_file, index=False, float_format='%.6f')
        
        logger.info(f"Saved combined percentile file: {filename} ({len(combined_df)} tickers, {len(combined_df.columns)} columns)")
        
        return output_file
    
    def save_single_benchmark_percentiles(self, percentile_results: Dict[int, pd.DataFrame], 
                                        benchmark: str, level: str, choice: str, 
                                        timeframe: str, date_str: Optional[str] = None) -> List[Path]:
        """
        Save single benchmark percentile results (backward compatibility).
        
        Args:
            percentile_results: Single benchmark percentile results
            benchmark: Benchmark name
            level: Analysis level
            choice: User ticker choice
            timeframe: Data timeframe
            date_str: Optional date string
            
        Returns:
            List of saved file paths
        """
        # Convert to multi-benchmark format and save
        benchmark_percentiles = {benchmark: percentile_results}
        return self.save_multi_benchmark_percentiles(
            benchmark_percentiles, level, choice, timeframe, date_str
        )
    
    def _organize_percentile_columns_systematically(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Organize DataFrame columns following systematic naming convention for percentiles.
        
        Order: ticker, timeframe, RS columns, percentile columns, calculation_date
        
        Args:
            df: DataFrame to organize
            
        Returns:
            DataFrame with systematically ordered columns
        """
        if df.empty:
            return df
            
        # Start with essential columns
        ordered_cols = []
        remaining_cols = list(df.columns)
        
        # 1. Ticker column (always first)
        if 'ticker' in remaining_cols:
            ordered_cols.append('ticker')
            remaining_cols.remove('ticker')
        
        # 2. Timeframe column (if present)
        if 'timeframe' in remaining_cols:
            ordered_cols.append('timeframe')
            remaining_cols.remove('timeframe')
        
        # 3. Systematic RS columns (reference values)
        rs_cols = [col for col in remaining_cols if '_rs_vs_' in col or (col.endswith('_rs') and '_rs_per' not in col)]
        rs_cols_sorted = self._sort_systematic_columns(rs_cols)
        ordered_cols.extend(rs_cols_sorted)
        for col in rs_cols_sorted:
            if col in remaining_cols:
                remaining_cols.remove(col)
        
        # 4. Systematic percentile columns
        percentile_cols = [col for col in remaining_cols if '_rs_per' in col]
        percentile_cols_sorted = self._sort_systematic_columns(percentile_cols)
        ordered_cols.extend(percentile_cols_sorted)
        for col in percentile_cols_sorted:
            if col in remaining_cols:
                remaining_cols.remove(col)
        
        # 5. Benchmark return columns
        benchmark_cols = [col for col in remaining_cols if col.startswith('benchmark_return_')]
        benchmark_cols.sort()
        ordered_cols.extend(benchmark_cols)
        for col in benchmark_cols:
            if col in remaining_cols:
                remaining_cols.remove(col)
        
        # 6. Calculation date (always last)
        if 'calculation_date' in remaining_cols:
            remaining_cols.remove('calculation_date')
            ordered_cols.append('calculation_date')
        
        # 7. Any remaining columns
        remaining_cols.sort()
        ordered_cols.extend(remaining_cols)
        
        # Filter out any columns that don't actually exist in the DataFrame
        final_cols = [col for col in ordered_cols if col in df.columns]
        
        return df[final_cols]
    
    def _sort_systematic_columns(self, columns: List[str]) -> List[str]:
        """
        Sort columns following systematic naming convention.
        
        Args:
            columns: List of column names to sort
            
        Returns:
            Sorted list of column names
        """
        def extract_sort_key(col_name):
            """Extract sort key from systematic column name."""
            try:
                # Handle systematic naming: daily_daily_daily_5d_rs_per or daily_daily_daily_5d_rs_vs_SPY
                parts = col_name.split('_')
                if len(parts) >= 5:
                    data_timeframe = parts[0]     # daily
                    calc_timeframe = parts[1]     # daily  
                    period_type = parts[2]        # daily
                    period_str = parts[3]         # 5d
                    
                    # Extract period number for sorting
                    period_num = int(period_str.replace('d', '').replace('w', '').replace('m', '').replace('y', ''))
                    
                    # For benchmark-specific columns
                    if '_vs_' in col_name:
                        benchmark = parts[-1]      # SPY
                        return (data_timeframe, calc_timeframe, period_type, period_num, benchmark, 'rs')
                    # For percentile columns
                    elif '_rs_per' in col_name:
                        return (data_timeframe, calc_timeframe, period_type, period_num, '', 'per')
                    # For simple RS columns
                    else:
                        return (data_timeframe, calc_timeframe, period_type, period_num, '', 'rs')
                
                # Fallback for unknown formats
                return ('zzz', 'zzz', 'zzz', 999, col_name, 'zzz')
                
            except (ValueError, IndexError):
                # Fallback for malformed column names
                return ('zzz', 'zzz', 'zzz', 999, col_name, 'zzz')
        
        # Sort using the extracted keys
        return sorted(columns, key=extract_sort_key)
    
    def get_percentile_summary(self, level: str, choice: str, timeframe: str) -> Dict:
        """
        Get summary of saved percentile files.
        
        Args:
            level: Analysis level
            choice: User ticker choice  
            timeframe: Data timeframe
            
        Returns:
            Dictionary with file summary information
        """
        pattern = f"percentiles_{level}_{choice}_{timeframe}_combined_*.csv"
        files = list(self.percentiles_dir.glob(pattern))
        
        summary = {
            'directory': str(self.percentiles_dir),
            'pattern': pattern,
            'total_files': len(files),
            'files': []
        }
        
        for file_path in sorted(files):
            try:
                df = pd.read_csv(file_path)
                file_info = {
                    'filename': file_path.name,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'ticker_count': len(df),
                    'column_count': len(df.columns),
                    'rs_columns': [col for col in df.columns if '_rs_vs_' in col or (col.endswith('_rs') and '_rs_per' not in col)],
                    'percentile_columns': [col for col in df.columns if '_rs_per' in col],
                    'benchmark_columns': [col for col in df.columns if col.startswith('benchmark_return_')],
                    'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                summary['files'].append(file_info)
            except Exception as e:
                logger.error(f"Error reading percentile file {file_path.name}: {e}")
        
        return summary
    
    def validate_percentile_files(self, level: str, choice: str, timeframe: str) -> Dict[str, List[str]]:
        """
        Validate percentile files for systematic naming compliance.
        
        Args:
            level: Analysis level
            choice: User ticker choice
            timeframe: Data timeframe
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            'valid_files': [],
            'invalid_files': [],
            'missing_percentiles': [],
            'systematic_naming_issues': []
        }
        
        pattern = f"percentiles_{level}_{choice}_{timeframe}_combined_*.csv"
        files = list(self.percentiles_dir.glob(pattern))
        
        import re
        rs_pattern = r'^[a-z]+_[a-z]+_[a-z]+_\d+d_rs(_vs_[A-Z]+)?$'
        per_pattern = r'^[a-z]+_[a-z]+_[a-z]+_\d+d_rs_per$'
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                file_issues = []
                
                # Check for systematic naming compliance
                for col in df.columns:
                    if '_rs_vs_' in col or col.endswith('_rs'):
                        if not re.match(rs_pattern, col):
                            file_issues.append(f"Invalid RS column: {col}")
                    elif '_rs_per' in col:
                        if not re.match(per_pattern, col):
                            file_issues.append(f"Invalid percentile column: {col}")
                
                # Check that every RS column has a corresponding percentile
                rs_cols = [col for col in df.columns if '_rs_vs_' in col or (col.endswith('_rs') and '_rs_per' not in col)]
                for rs_col in rs_cols:
                    expected_per_col = rs_col.replace('_rs_vs_', '_rs_per').replace('_rs', '_rs_per')
                    if expected_per_col not in df.columns:
                        file_issues.append(f"Missing percentile for RS column: {rs_col} -> {expected_per_col}")
                
                if file_issues:
                    validation_results['invalid_files'].append(file_path.name)
                    validation_results['systematic_naming_issues'].extend([f"{file_path.name}: {issue}" for issue in file_issues])
                else:
                    validation_results['valid_files'].append(file_path.name)
                    
            except Exception as e:
                validation_results['invalid_files'].append(file_path.name)
                validation_results['systematic_naming_issues'].append(f"{file_path.name}: Error reading file - {e}")
        
        return validation_results