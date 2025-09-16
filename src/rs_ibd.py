"""
IBD-Style Relative Strength Calculator
=====================================

Implements Investor's Business Daily (IBD) style Relative Strength calculations
using the formula: RS = (1 + Stock Return) / (1 + Benchmark Return)

This approach measures how a stock performs relative to a benchmark over
various time periods, with results converted to percentile rankings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from .rs_base import RSCalculatorBase, RSResults

logger = logging.getLogger(__name__)


class IBDRelativeStrengthCalculator(RSCalculatorBase):
    """
    IBD-style Relative Strength calculator implementing batch processing
    for efficient calculation across large stock universes.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize IBD RS calculator.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object with RS settings
        """
        super().__init__(config, user_config)
        self.method_name = 'ibd'
        
    def calculate_stock_rs(self, price_data, benchmark_data, periods, benchmark_name='SPY'):
        """
        Calculate IBD-style RS values for individual stocks using vectorized operations.
        
        Args:
            price_data: DataFrame with stock price data (dates x tickers)
            benchmark_data: Series with benchmark price data (indexed by date)
            periods: List of tuples [(period_value, period_category, column_suffix), ...]
            benchmark_name: Name of benchmark for column naming
            
        Returns:
            Dictionary of {column_suffix: DataFrame} with RS values
        """
        logger.info(f"Calculating IBD RS for {len(price_data.columns)} stocks vs {benchmark_name} across {len(periods)} periods")
        
        rs_results = {}
        
        # Align price data and benchmark data by date
        aligned_data = price_data.align(benchmark_data, axis=0, join='inner')[0]
        aligned_benchmark = price_data.align(benchmark_data, axis=0, join='inner')[1]
        
        if aligned_data.empty:
            logger.warning("No overlapping dates between price data and benchmark")
            return rs_results
            
        logger.info(f"Aligned data: {len(aligned_data)} dates, {len(aligned_data.columns)} tickers")
        
        for period_value, period_category, column_suffix in periods:
            logger.info(f"Calculating RS for {period_value}-period ({period_category})...")
            
            # Skip if insufficient data
            if len(aligned_data) <= period_value:
                logger.warning(f"Insufficient data for {period_value}-period calculation")
                continue
                
            try:
                # Calculate stock returns: (Current Price / Price N periods ago) - 1
                stock_returns = aligned_data.iloc[-1] / aligned_data.iloc[-period_value-1] - 1
                
                # Calculate benchmark return over same period
                benchmark_return = aligned_benchmark.iloc[-1] / aligned_benchmark.iloc[-period_value-1] - 1
                
                # Calculate IBD-style RS: (1 + Stock Return) / (1 + Benchmark Return)
                rs_values = (1 + stock_returns) / (1 + benchmark_return)
                
                # Handle any division by zero or infinite values
                rs_values = rs_values.replace([np.inf, -np.inf], np.nan)
                
                # Create result DataFrame with new naming convention
                rs_column_name = f'{column_suffix}_rs_vs_{benchmark_name}'
                return_column_name = f'{column_suffix}_stock_return'
                benchmark_return_column_name = f'{column_suffix}_benchmark_return_{benchmark_name}'
                
                rs_df = pd.DataFrame({
                    'ticker': rs_values.index,
                    rs_column_name: rs_values.values,
                    return_column_name: stock_returns.values,
                    benchmark_return_column_name: benchmark_return,
                    'date': aligned_data.index[-1].strftime('%Y-%m-%d')  # Format as YYYY-MM-DD
                })
                
                rs_df = rs_df.set_index('ticker')
                
                # Remove any rows with NaN RS values
                rs_df = rs_df.dropna(subset=[rs_column_name])
                
                rs_results[column_suffix] = rs_df
                
                logger.info(f"RS calculated for {len(rs_df)} tickers ({period_value}-period {period_category} vs {benchmark_name})")
                
            except Exception as e:
                logger.error(f"Error calculating RS for {period_value}-period {period_category}: {e}")
                continue
                
        return rs_results
    
    def calculate_composite_rs(self, composite_indices, benchmark_data, periods, benchmark_name='SPY'):
        """
        Calculate IBD-style RS values for composite indices (sectors/industries).
        
        Args:
            composite_indices: Dictionary of {group_name: Series} with composite price data
            benchmark_data: Series with benchmark price data
            periods: List of tuples [(period_value, period_category, column_suffix), ...]
            benchmark_name: Name of benchmark for column naming
            
        Returns:
            Dictionary of {column_suffix: DataFrame} with RS values for composites
        """
        logger.info(f"Calculating composite RS for {len(composite_indices)} groups vs {benchmark_name}")
        
        # Convert composite indices to DataFrame for easier processing
        composite_df = pd.DataFrame(composite_indices)
        
        # Use the same logic as stock RS calculation
        return self.calculate_stock_rs(composite_df, benchmark_data, periods, benchmark_name)
    
    def process_universe(self, ticker_list, timeframe='daily', benchmark_ticker='SPY', batch_size=100):
        """
        Process entire stock universe for RS calculations using true batched approach.

        Args:
            ticker_list: List of ticker symbols to analyze
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            benchmark_ticker: Benchmark ticker symbol
            batch_size: Number of tickers to process per batch

        Returns:
            RSResults object with calculated data
        """
        logger.info(f"Processing {len(ticker_list)} tickers for IBD RS calculation vs {benchmark_ticker}")

        # Initialize results container
        results = RSResults(self.method_name, 'stocks', timeframe)
        results.metadata['benchmark_ticker'] = benchmark_ticker
        results.metadata['universe_size'] = len(ticker_list)

        # Load benchmark data once
        benchmark_data = self.load_benchmark_data(benchmark_ticker, timeframe)
        if benchmark_data is None:
            logger.error(f"Could not load benchmark data for {benchmark_ticker}")
            return results

        # Get analysis periods
        periods = self.get_analysis_periods(timeframe)
        logger.info(f"Calculating RS for periods: {periods}")

        # Process tickers in batches for memory efficiency
        import math
        total_batches = math.ceil(len(ticker_list) / batch_size)
        processed_count = 0

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(ticker_list))
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            logger.info(f"🔄 Processing RS batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Load price data for this batch only
            batch_price_data = self._load_price_data(batch_tickers, timeframe)
            if batch_price_data.empty:
                logger.warning(f"No price data loaded for batch {batch_count}")
                continue

            # Calculate RS values for this batch
            batch_rs_results = self.calculate_stock_rs(batch_price_data, benchmark_data, periods, benchmark_ticker)

            # Merge batch results into main results
            for column_suffix, period_df in batch_rs_results.items():
                if column_suffix not in results.rs_values:
                    results.rs_values[column_suffix] = period_df
                else:
                    # Concatenate with existing results
                    results.rs_values[column_suffix] = pd.concat([
                        results.rs_values[column_suffix],
                        period_df
                    ], ignore_index=False)

            processed_count += len(batch_tickers)
            logger.info(f"✅ RS batch {batch_count} completed: {len(batch_tickers)} tickers processed")

            # Memory cleanup
            del batch_price_data
            del batch_rs_results

            # Force garbage collection for large batches
            if len(batch_tickers) > 50:
                import gc
                gc.collect()

        logger.info(f"✅ RS calculation completed: {processed_count} tickers processed in {total_batches} batches")

        # Set calculation date in metadata
        if benchmark_data is not None and len(benchmark_data) > 0:
            results.metadata['calculation_date'] = str(benchmark_data.index[-1])

        return results
    
    def _calculate_percentile_rankings(self, rs_series):
        """
        Calculate percentile rankings for RS values using IBD-style 1-99 scale.
        
        Args:
            rs_series: Pandas Series with RS values
            
        Returns:
            Pandas Series with percentile rankings (1-99 scale)
        """
        try:
            # Calculate percentile ranks (0-1 scale)
            percentile_ranks = rs_series.rank(pct=True, na_option='keep')
            
            # Convert to IBD-style 1-99 scale
            # 0-1 scale -> multiply by 98 and add 1 -> 1-99 scale
            ibd_percentiles = (percentile_ranks * 98) + 1
            
            # Round to nearest integer
            ibd_percentiles = ibd_percentiles.round().astype('Int64')  # Use nullable integer type
            
            return ibd_percentiles
            
        except Exception as e:
            logger.error(f"Error calculating percentile rankings: {e}")
            # Return series filled with NaN if calculation fails
            return pd.Series(index=rs_series.index, dtype='Int64')
    
    def _load_price_data(self, ticker_list, timeframe):
        """
        Load historical price data for a list of tickers.

        Args:
            ticker_list: List of ticker symbols
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')

        Returns:
            DataFrame with price data (dates x tickers)
        """
        data_dir = self.config.get_market_data_dir(timeframe)
        price_data = {}
        
        logger.info(f"Loading {timeframe} price data for {len(ticker_list)} tickers")
        
        loaded_count = 0
        for ticker in ticker_list:
            try:
                ticker_file = data_dir / f"{ticker}.csv"
                if ticker_file.exists():
                    df = pd.read_csv(ticker_file, index_col=0, parse_dates=True)
                    if 'Close' in df.columns and len(df) > 0:
                        price_data[ticker] = df['Close']
                        loaded_count += 1
            except Exception as e:
                logger.debug(f"Could not load data for {ticker}: {e}")
                continue
                
        logger.info(f"Successfully loaded price data for {loaded_count} tickers")
        
        if price_data:
            # Convert to DataFrame and forward fill missing values
            df = pd.DataFrame(price_data)
            df = df.ffill().dropna(how='all')
            return df
        else:
            return pd.DataFrame()
    
    def save_rs_results(self, results, level='stocks', choice=0):
        """
        Save RS results to CSV file.
        
        Args:
            results: RSResults object with calculated data
            level: Analysis level ('stocks', 'sectors', 'industries')
            choice: User ticker choice number
            
        Returns:
            List containing path to summary file
        """
        saved_files = []
        # Extract data date from the summary DataFrame instead of metadata
        date_str = None
        
        # Create summary DataFrame with all periods
        if results.rs_values:
            summary_df = self._create_summary_dataframe(results)
            if not summary_df.empty:
                summary_file = self.save_results(
                    summary_df, level, self.method_name,
                    results.timeframe, choice, None
                )
                saved_files.append(summary_file)
        
        return saved_files
    
    def _create_summary_dataframe(self, results):
        """
        Create a wide-format summary DataFrame by horizontally joining all periods by ticker.
        
        Args:
            results: RSResults object with calculated data
            
        Returns:
            DataFrame with wide-format RS data (one row per ticker)
        """
        if not results.rs_values:
            return pd.DataFrame()
            
        # Sort column suffixes for consistent column ordering
        sorted_suffixes = sorted(results.rs_values.keys())
        
        # Start with the first period as base
        first_suffix = sorted_suffixes[0]
        summary_df = results.rs_values[first_suffix].copy()
        
        # Join remaining periods horizontally by ticker (index)
        for suffix in sorted_suffixes[1:]:
            period_df = results.rs_values[suffix].copy()
            
            # Select only the period-specific columns for joining
            # Exclude common columns like 'date' from joining
            common_cols = ['date']
            period_specific_cols = [col for col in period_df.columns 
                                  if col not in common_cols and col != period_df.index.name]
            
            if period_specific_cols:
                summary_df = summary_df.join(period_df[period_specific_cols], how='outer')
        
        # Add metadata
        summary_df['timeframe'] = results.timeframe
        # Keep the date column as is (already formatted as YYYY-MM-DD)
        
        return summary_df