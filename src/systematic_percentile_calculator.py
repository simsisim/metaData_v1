"""
Systematic Percentile Calculator
==============================

Standardized percentile calculation module that follows systematic naming conventions.
Separates percentile ranking from RS calculation for maximum flexibility.

Features:
- Systematic column naming: daily_daily_daily_5d_rs_per
- IBD-style 1-99 percentile rankings
- Multi-benchmark percentile support
- Configurable percentile methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
# Date normalization removed - using simple date operations

logger = logging.getLogger(__name__)


class SystematicPercentileCalculator:
    """
    Systematic percentile calculator following established naming conventions.
    Focuses purely on percentile ranking without RS calculation.
    """
    
    def __init__(self, config=None, user_config=None):
        """
        Initialize systematic percentile calculator.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object
        """
        self.config = config
        self.user_config = user_config
        self.method_name = 'systematic_percentile'
        
        logger.info("Initialized SystematicPercentileCalculator")
    
    def calculate_percentiles(self, rs_data: Dict[str, Dict[int, pd.DataFrame]], 
                            method: str = 'ibd', universe_type: str = 'all', 
                            level: str = 'stocks') -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Calculate systematic percentiles for RS data from multiple benchmarks.
        
        Args:
            rs_data: Dict of {benchmark: {period: DataFrame}} with RS values
            method: Percentile calculation method ('ibd', 'standard')
            universe_type: Universe scope ('all', 'sector', 'industry', 'ticker_choice')
            level: Analysis level ('stocks', 'sectors', 'industries')
            
        Returns:
            Dict of {benchmark: {period: DataFrame}} with added percentile columns
        """
        logger.info(f"Calculating systematic percentiles using {method} method")
        logger.info(f"Benchmarks: {list(rs_data.keys())}")
        
        percentile_results = {}
        
        for benchmark, benchmark_data in rs_data.items():
            logger.info(f"Processing percentiles for benchmark: {benchmark}")
            
            benchmark_percentiles = {}
            for period, rs_df in benchmark_data.items():
                if rs_df.empty:
                    logger.warning(f"Empty DataFrame for {benchmark} {period}d")
                    continue
                    
                # Calculate percentiles for this period
                period_df = self._calculate_period_percentiles(rs_df, period, benchmark, method)
                benchmark_percentiles[period] = period_df
                
                logger.info(f"Calculated percentiles for {benchmark} {period}d: {len(period_df)} tickers")
            
            if benchmark_percentiles:
                percentile_results[benchmark] = benchmark_percentiles
                
        logger.info(f"Systematic percentile calculation completed: {len(percentile_results)} benchmarks")
        return percentile_results
    
    def _calculate_period_percentiles(self, rs_df: pd.DataFrame, period: int, 
                                    benchmark: str, method: str) -> pd.DataFrame:
        """
        Calculate percentiles for a specific period and benchmark.
        
        Args:
            rs_df: DataFrame with RS values
            period: Calculation period
            benchmark: Benchmark name
            method: Percentile method
            
        Returns:
            DataFrame with RS values and percentile columns
        """
        result_df = rs_df.copy()
        
        # Find RS columns for this benchmark and period
        rs_columns = [col for col in result_df.columns if '_rs_vs_' in col and benchmark in col]
        
        for rs_col in rs_columns:
            if rs_col in result_df.columns:
                rs_values = result_df[rs_col]
                
                # Calculate percentiles
                if method == 'ibd':
                    percentiles = self._calculate_ibd_percentiles(rs_values)
                else:
                    percentiles = self._calculate_standard_percentiles(rs_values)
                
                # Generate systematic percentile column name
                percentile_col = self._generate_percentile_column_name(rs_col)
                result_df[percentile_col] = percentiles
                
                logger.debug(f"Added percentile column: {percentile_col}")
        
        return result_df
    
    def _generate_percentile_column_name(self, rs_column_name: str) -> str:
        """
        Generate systematic percentile column name from RS column name.
        
        Converts: daily_daily_daily_5d_rs_vs_SPY 
        To:       daily_daily_daily_5d_rs_per
        
        Args:
            rs_column_name: RS column name following systematic convention
            
        Returns:
            Systematic percentile column name
        """
        if '_rs_vs_' in rs_column_name:
            # Replace _rs_vs_BENCHMARK with _rs_per
            base_name = rs_column_name.split('_rs_vs_')[0]
            return f"{base_name}_rs_per"
        elif rs_column_name.endswith('_rs'):
            # Simple replacement if no benchmark suffix
            return rs_column_name.replace('_rs', '_rs_per')
        else:
            # Fallback - add _per suffix
            return f"{rs_column_name}_per"
    
    def _calculate_ibd_percentiles(self, rs_values: pd.Series) -> pd.Series:
        """
        Calculate IBD-style percentile rankings (1-99 scale).
        
        Args:
            rs_values: Series with RS values
            
        Returns:
            Series with IBD percentile rankings
        """
        try:
            # Calculate percentile ranks (0-1 scale)
            percentile_ranks = rs_values.rank(pct=True, na_option='keep')
            
            # Convert to IBD-style 1-99 scale
            # 0-1 scale -> multiply by 98 and add 1 -> 1-99 scale
            ibd_percentiles = (percentile_ranks * 98) + 1
            
            # Round to nearest integer
            ibd_percentiles = ibd_percentiles.round().astype('Int64')  # Use nullable integer type
            
            return ibd_percentiles
            
        except Exception as e:
            logger.error(f"Error calculating IBD percentiles: {e}")
            return pd.Series(index=rs_values.index, dtype='Int64')
    
    def _calculate_standard_percentiles(self, rs_values: pd.Series) -> pd.Series:
        """
        Calculate standard percentile rankings (0-100 scale).
        
        Args:
            rs_values: Series with RS values
            
        Returns:
            Series with standard percentile rankings
        """
        try:
            # Calculate percentile ranks (0-1 scale)
            percentile_ranks = rs_values.rank(pct=True, na_option='keep')
            
            # Convert to 0-100 scale
            standard_percentiles = percentile_ranks * 100
            
            # Round to nearest integer
            standard_percentiles = standard_percentiles.round().astype('Int64')
            
            return standard_percentiles
            
        except Exception as e:
            logger.error(f"Error calculating standard percentiles: {e}")
            return pd.Series(index=rs_values.index, dtype='Int64')
    
    def combine_multi_benchmark_percentiles(self, 
                                          benchmark_percentiles: Dict[str, Dict[int, pd.DataFrame]]) -> Dict[int, pd.DataFrame]:
        """
        Combine percentile results from multiple benchmarks into unified DataFrames by period.
        
        Args:
            benchmark_percentiles: Percentile results from multiple benchmarks
            
        Returns:
            Dict of {period: DataFrame} with all benchmark percentile columns
        """
        logger.info("Combining multi-benchmark percentile results")
        
        combined_results = {}
        
        # Get all periods across all benchmarks
        all_periods = set()
        for benchmark_data in benchmark_percentiles.values():
            all_periods.update(benchmark_data.keys())
        
        for period in sorted(all_periods):
            # Collect DataFrames for this period from all benchmarks
            period_dfs = []
            
            for benchmark, benchmark_data in benchmark_percentiles.items():
                if period in benchmark_data:
                    df = benchmark_data[period].copy()
                    # Set ticker as index for merging
                    if 'ticker' in df.columns:
                        df = df.set_index('ticker')
                    period_dfs.append(df)
            
            if period_dfs:
                # Merge all benchmark results for this period
                combined_df = period_dfs[0]
                for df in period_dfs[1:]:
                    combined_df = combined_df.join(df, how='outer', rsuffix='_dup')
                
                # Reset index to make ticker a column again
                combined_df = combined_df.reset_index()
                
                # Ensure normalized calculation_date is present
                if 'calculation_date' not in combined_df.columns and period_dfs:
                    if 'calculation_date' in period_dfs[0].columns:
                        first_calc_date = period_dfs[0]['calculation_date'].iloc[0]
                        combined_df['calculation_date'] = first_calc_date
                    else:
                        # Fallback to current date if no calculation date found
                        combined_df['calculation_date'] = datetime.now()
                
                combined_results[period] = combined_df
                
                logger.info(f"Combined {period}d percentile results: {len(combined_df)} tickers, {len(combined_df.columns)} columns")
        
        logger.info(f"Multi-benchmark percentile combination completed: {len(combined_results)} periods")
        return combined_results
    
    def calculate_cross_benchmark_percentiles(self, 
                                            combined_rs_data: Dict[int, pd.DataFrame],
                                            method: str = 'ibd') -> Dict[int, pd.DataFrame]:
        """
        Calculate percentiles across all benchmarks combined (cross-benchmark ranking).
        
        Args:
            combined_rs_data: Dict of {period: DataFrame} with RS values from all benchmarks
            method: Percentile calculation method
            
        Returns:
            Dict of {period: DataFrame} with cross-benchmark percentile rankings
        """
        logger.info(f"Calculating cross-benchmark percentiles using {method} method")
        
        cross_benchmark_results = {}
        
        for period, combined_df in combined_rs_data.items():
            if combined_df.empty:
                continue
                
            result_df = combined_df.copy()
            
            # Find all RS columns (excluding benchmark returns and calculation dates)
            rs_columns = [col for col in result_df.columns 
                         if '_rs_vs_' in col and not col.startswith('benchmark_return')]
            
            # Calculate composite RS score (average or max across benchmarks)
            if len(rs_columns) > 1:
                # Calculate average RS across benchmarks
                rs_values_df = result_df[rs_columns]
                composite_rs = rs_values_df.mean(axis=1, skipna=True)
                
                # Calculate percentiles for composite RS
                if method == 'ibd':
                    composite_percentiles = self._calculate_ibd_percentiles(composite_rs)
                else:
                    composite_percentiles = self._calculate_standard_percentiles(composite_rs)
                
                # Add composite columns
                result_df[f'composite_rs_{period}d'] = composite_rs
                result_df[f'composite_rs_per_{period}d'] = composite_percentiles
                
                logger.info(f"Added cross-benchmark percentiles for period {period}d: {len(result_df)} tickers")
            
            cross_benchmark_results[period] = result_df
        
        logger.info(f"Cross-benchmark percentile calculation completed: {len(cross_benchmark_results)} periods")
        return cross_benchmark_results
    
    def validate_systematic_naming(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate that DataFrame columns follow systematic naming convention.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            'valid_rs_columns': [],
            'valid_percentile_columns': [],
            'invalid_columns': [],
            'missing_percentiles': []
        }
        
        rs_pattern = r'^[a-z]+_[a-z]+_[a-z]+_\d+d_rs(_vs_[A-Z]+)?$'
        per_pattern = r'^[a-z]+_[a-z]+_[a-z]+_\d+d_rs_per$'
        
        import re
        
        for col in df.columns:
            if '_rs_vs_' in col or col.endswith('_rs'):
                if re.match(rs_pattern, col):
                    validation_results['valid_rs_columns'].append(col)
                    
                    # Check if corresponding percentile exists
                    expected_per_col = self._generate_percentile_column_name(col)
                    if expected_per_col not in df.columns:
                        validation_results['missing_percentiles'].append(expected_per_col)
                else:
                    validation_results['invalid_columns'].append(col)
                    
            elif col.endswith('_rs_per'):
                if re.match(per_pattern, col):
                    validation_results['valid_percentile_columns'].append(col)
                else:
                    validation_results['invalid_columns'].append(col)
        
        return validation_results