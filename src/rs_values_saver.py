"""
RS Values File Saver
====================

Specialized file saver for pure RS values (without percentiles).
Creates organized output files in the rs_values/ directory structure.

Features:
- Pure RS value storage (no percentile mixing)
- Multi-benchmark column organization
- Systematic file naming
- Metadata tracking
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RSValuesSaver:
    """
    Specialized saver for RS values without percentile data.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize RS values saver.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object
        """
        self.config = config
        self.user_config = user_config
        
        # Create rs_values directory structure
        self.rs_values_dir = Path(config.directories['RESULTS_DIR']) / 'rs' / 'rs_values'
        self.rs_values_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RS Values Saver initialized: {self.rs_values_dir}")
    
    def save_multi_benchmark_rs_results(self, benchmark_results: Dict[str, Dict[int, pd.DataFrame]], 
                                      level: str, choice: str, timeframe: str, 
                                      date_str: Optional[str] = None) -> List[Path]:
        """
        Save multi-benchmark RS results to rs_values files.
        
        Args:
            benchmark_results: Results from MultiBenchmarkRSCalculator
            level: Analysis level ('stocks', 'sectors', 'industries')
            choice: User ticker choice
            timeframe: Data timeframe
            date_str: Optional date string
            
        Returns:
            List of saved file paths
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')
        
        logger.info(f"Saving multi-benchmark RS results for {level} (choice {choice}, {timeframe})")
        
        saved_files = []
        
        # Get all periods across all benchmarks
        all_periods = set()
        for benchmark_data in benchmark_results.values():
            all_periods.update(benchmark_data.keys())
        
        if not all_periods:
            logger.warning("No periods found in benchmark results")
            return saved_files
        
        # Only generate combined file with all periods
        combined_file = self._save_combined_periods_file(
            benchmark_results, level, choice, timeframe, date_str
        )
        if combined_file:
            saved_files.append(combined_file)
        else:
            logger.warning("No combined RS file generated")
        
        logger.info(f"Multi-benchmark RS saving completed: {len(saved_files)} files created")
        return saved_files
    
    def _combine_period_data(self, benchmark_results: Dict[str, Dict[int, pd.DataFrame]], 
                           period: int) -> Optional[pd.DataFrame]:
        """
        Combine data from all benchmarks for a specific period.
        
        Args:
            benchmark_results: Multi-benchmark results
            period: Period to combine
            
        Returns:
            Combined DataFrame for the period or None
        """
        period_dfs = []
        
        for benchmark, benchmark_data in benchmark_results.items():
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
                    # For other duplicates, rename appropriately
                    logger.warning(f"Unexpected duplicate column: {dup_col}")
                    combined_df = combined_df.drop(columns=[dup_col])
        
        return combined_df
    
    def _save_combined_periods_file(self, benchmark_results: Dict[str, Dict[int, pd.DataFrame]], 
                                  level: str, choice: str, timeframe: str, 
                                  date_str: str) -> Optional[Path]:
        """
        Save a combined file with all periods and benchmarks.
        
        Args:
            benchmark_results: Multi-benchmark results
            level: Analysis level
            choice: User ticker choice
            timeframe: Data timeframe
            date_str: Date string
            
        Returns:
            Path to saved combined file or None
        """
        logger.info("Creating combined periods RS file")
        
        # Get all unique tickers across all results
        all_tickers = set()
        for benchmark_data in benchmark_results.values():
            for period_data in benchmark_data.values():
                all_tickers.update(period_data['ticker'].tolist())
        
        if not all_tickers:
            logger.warning("No tickers found for combined file")
            return None
        
        # Create base DataFrame with tickers
        combined_df = pd.DataFrame({'ticker': sorted(all_tickers)})
        
        # Add columns for each benchmark/period combination
        for benchmark, benchmark_data in benchmark_results.items():
            for period, period_df in benchmark_data.items():
                # Find RS column for this benchmark/period using systematic naming
                rs_columns = [col for col in period_df.columns if '_rs_vs_' in col and benchmark in col]
                
                if rs_columns:
                    rs_column = rs_columns[0]
                    period_data = period_df[['ticker', rs_column]].copy()
                    
                    # Merge into combined DataFrame
                    combined_df = pd.merge(combined_df, period_data, on='ticker', how='left')
                    
                    logger.info(f"Added {rs_column}: {len(period_data)} values")
        
        # Add metadata columns
        if benchmark_results:
            first_benchmark = list(benchmark_results.keys())[0]
            first_period = list(benchmark_results[first_benchmark].keys())[0]
            sample_df = benchmark_results[first_benchmark][first_period]
            
            if 'calculation_date' in sample_df.columns:
                combined_df['calculation_date'] = sample_df['calculation_date'].iloc[0]
        
        # Add benchmark return columns
        for benchmark, benchmark_data in benchmark_results.items():
            for period, period_df in benchmark_data.items():
                benchmark_return_col = f'benchmark_return_{benchmark}'
                if benchmark_return_col in period_df.columns:
                    combined_df[f'{benchmark_return_col}_{period}d'] = period_df[benchmark_return_col].iloc[0]
                    break  # Only add once per benchmark
        
        # Organize columns systematically before saving
        combined_df = self._organize_columns_systematically(combined_df)
        
        # Save combined file
        filename = f"rs_{level}_{choice}_{timeframe}_combined_{date_str}.csv"
        output_file = self.rs_values_dir / filename
        
        combined_df.to_csv(output_file, index=False, float_format='%.6f')
        
        logger.info(f"Saved combined RS file: {filename} ({len(combined_df)} tickers, {len(combined_df.columns)} columns)")
        
        return output_file
    
    def save_single_benchmark_rs_results(self, rs_results: Dict[int, pd.DataFrame], 
                                       benchmark: str, level: str, choice: str, 
                                       timeframe: str, date_str: Optional[str] = None) -> List[Path]:
        """
        Save single benchmark RS results (backward compatibility).
        
        Args:
            rs_results: Single benchmark results
            benchmark: Benchmark name
            level: Analysis level
            choice: User ticker choice
            timeframe: Data timeframe
            date_str: Optional date string
            
        Returns:
            List of saved file paths
        """
        # Convert to multi-benchmark format and save
        benchmark_results = {benchmark: rs_results}
        return self.save_multi_benchmark_rs_results(
            benchmark_results, level, choice, timeframe, date_str
        )
    
    def get_rs_values_summary(self, level: str, choice: str, timeframe: str) -> Dict:
        """
        Get summary of saved RS values files.
        
        Args:
            level: Analysis level
            choice: User ticker choice  
            timeframe: Data timeframe
            
        Returns:
            Dictionary with file summary information
        """
        pattern = f"rs_{level}_{choice}_{timeframe}_combined_*.csv"
        files = list(self.rs_values_dir.glob(pattern))
        
        summary = {
            'directory': str(self.rs_values_dir),
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
                    'rs_columns': [col for col in df.columns if '_rs_vs_' in col or col.endswith('_rs')],
                    'benchmark_columns': [col for col in df.columns if col.startswith('benchmark_return_')],
                    'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                summary['files'].append(file_info)
            except Exception as e:
                logger.error(f"Error reading RS file {file_path.name}: {e}")
        
        return summary
    
    def _organize_columns_systematically(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Organize DataFrame columns following systematic naming convention.
        
        Order: ticker, timeframe, RS columns (by period), benchmark returns, calculation_date
        
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
        
        # 3. Systematic RS columns, sorted by benchmark and period
        rs_cols = [col for col in remaining_cols if '_rs_vs_' in col or col.endswith('_rs')]
        rs_cols_sorted = self._sort_systematic_rs_columns(rs_cols)
        ordered_cols.extend(rs_cols_sorted)
        for col in rs_cols_sorted:
            if col in remaining_cols:
                remaining_cols.remove(col)
        
        # 4. Benchmark return columns
        benchmark_cols = [col for col in remaining_cols if col.startswith('benchmark_return_')]
        benchmark_cols.sort()  # Simple alphabetical sort for benchmark returns
        ordered_cols.extend(benchmark_cols)
        for col in benchmark_cols:
            if col in remaining_cols:
                remaining_cols.remove(col)
        
        # 5. Calculation date (always last)
        if 'calculation_date' in remaining_cols:
            remaining_cols.remove('calculation_date')
            ordered_cols.append('calculation_date')
        
        # 6. Any remaining columns
        remaining_cols.sort()  # Alphabetical sort for any other columns
        ordered_cols.extend(remaining_cols)
        
        # Filter out any columns that don't actually exist in the DataFrame
        final_cols = [col for col in ordered_cols if col in df.columns]
        
        return df[final_cols]
    
    def _sort_systematic_rs_columns(self, rs_columns: List[str]) -> List[str]:
        """
        Sort RS columns following systematic naming convention.
        
        Sorts by: data_timeframe, calculation_timeframe, period_type, period, benchmark
        Example: daily_daily_daily_1d_rs_vs_SPY < daily_daily_daily_3d_rs_vs_SPY
        
        Args:
            rs_columns: List of RS column names
            
        Returns:
            Sorted list of RS column names
        """
        def extract_sort_key(col_name):
            """Extract sort key from systematic column name."""
            try:
                # Handle systematic naming: daily_daily_daily_5d_rs_vs_SPY
                if '_rs_vs_' in col_name:
                    parts = col_name.split('_')
                    if len(parts) >= 6:
                        data_timeframe = parts[0]  # daily
                        calc_timeframe = parts[1]  # daily  
                        period_type = parts[2]     # daily
                        period_str = parts[3]      # 5d
                        benchmark = parts[-1]      # SPY
                        
                        # Extract period number for sorting
                        period_num = int(period_str.replace('d', '').replace('w', '').replace('m', '').replace('y', ''))
                        
                        return (data_timeframe, calc_timeframe, period_type, period_num, benchmark)
                
                # Handle simple RS columns
                elif col_name.endswith('_rs'):
                    # Extract period if possible
                    import re
                    period_match = re.search(r'(\d+)d', col_name)
                    period_num = int(period_match.group(1)) if period_match else 999
                    return ('zz', 'zz', 'zz', period_num, col_name)  # Sort after systematic columns
                
                # Fallback for unknown formats
                return ('zzz', 'zzz', 'zzz', 999, col_name)
                
            except (ValueError, IndexError):
                # Fallback for malformed column names
                return ('zzz', 'zzz', 'zzz', 999, col_name)
        
        # Sort using the extracted keys
        return sorted(rs_columns, key=extract_sort_key)