"""
Moving Average Relative Strength Calculator
==========================================

Implements Moving Average-style Relative Strength calculations using the formula:
RS = (current_price - moving_average) / moving_average * 100

This approach measures how far a stock's current price is from its own moving average,
providing trend-based relative strength analysis complementary to IBD methodology.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import gc
from .rs_base import RSCalculatorBase, RSResults

logger = logging.getLogger(__name__)


class MARelativeStrengthCalculator(RSCalculatorBase):
    """
    Moving Average-style Relative Strength calculator implementing batch processing
    for efficient calculation across large stock universes.
    """

    def __init__(self, config, user_config):
        """
        Initialize MA RS calculator.

        Args:
            config: Config object with directory paths
            user_config: User configuration object with RS settings
        """
        super().__init__(config, user_config)
        self.method_name = 'ma'

        # Parse MA periods from configuration
        self.ma_periods = self._parse_ma_periods()
        logger.info(f"MA RS Calculator initialized with periods: {self.ma_periods}")

    def _parse_ma_periods(self):
        """
        Parse moving average periods from user configuration.

        Returns:
            List of integers representing MA periods
        """
        ma_method = getattr(self.user_config, 'rs_ma_method', '20;50')
        try:
            periods = [int(p.strip()) for p in ma_method.split(';') if p.strip()]
            if not periods:
                logger.warning("No MA periods found in rs_ma_method, using default [20, 50]")
                return [20, 50]
            return periods
        except Exception as e:
            logger.error(f"Error parsing rs_ma_method '{ma_method}': {e}. Using default [20, 50]")
            return [20, 50]

    def calculate_stock_rs(self, price_data, benchmark_data, periods, benchmark_name='SPY'):
        """
        Calculate Moving Average-style RS values for individual stocks using vectorized operations.

        Args:
            price_data: DataFrame with stock price data (dates x tickers)
            benchmark_data: Series with benchmark price data (indexed by date) - not used for MA method
            periods: List of tuples [(period_value, period_category, column_suffix), ...]
            benchmark_name: Name of benchmark for column naming

        Returns:
            Dictionary of {column_suffix: DataFrame} with RS values
        """
        logger.info(f"Calculating MA RS for {len(price_data.columns)} stocks using MA periods: {self.ma_periods}")

        rs_results = {}

        # Calculate MA RS for each configured MA period
        for ma_period in self.ma_periods:
            logger.info(f"Processing {ma_period}-day moving average RS calculations")

            # Calculate moving averages for all stocks
            ma_data = price_data.rolling(window=ma_period, min_periods=ma_period).mean()

            # Calculate MA RS: (price - ma) / ma * 100
            ma_rs_data = ((price_data - ma_data) / ma_data * 100).round(4)

            # Create column name following existing convention
            column_suffix = f"{ma_period}MA_{benchmark_name}"

            # Store results
            rs_results[column_suffix] = ma_rs_data

            logger.info(f"Completed {ma_period}MA RS calculation - {len(ma_rs_data.columns)} stocks processed")

        logger.info(f"MA RS calculation complete - {len(rs_results)} MA periods processed")
        return rs_results

    def calculate_sector_rs(self, sector_composites, benchmark_data, periods, benchmark_name='SPY'):
        """
        Calculate Moving Average RS for sector composites.

        Args:
            sector_composites: DataFrame with sector composite price data
            benchmark_data: Series with benchmark price data - not used for MA method
            periods: List of tuples [(period_value, period_category, column_suffix), ...]
            benchmark_name: Name of benchmark for column naming

        Returns:
            Dictionary of {column_suffix: DataFrame} with sector RS values
        """
        logger.info(f"Calculating MA RS for {len(sector_composites.columns)} sectors")

        # Use same logic as stocks - sectors are just composite price series
        return self.calculate_stock_rs(sector_composites, benchmark_data, periods, benchmark_name)

    def calculate_industry_rs(self, industry_composites, benchmark_data, periods, benchmark_name='SPY'):
        """
        Calculate Moving Average RS for industry composites.

        Args:
            industry_composites: DataFrame with industry composite price data
            benchmark_data: Series with benchmark price data - not used for MA method
            periods: List of tuples [(period_value, period_category, column_suffix), ...]
            benchmark_name: Name of benchmark for column naming

        Returns:
            Dictionary of {column_suffix: DataFrame} with industry RS values
        """
        logger.info(f"Calculating MA RS for {len(industry_composites.columns)} industries")

        # Use same logic as stocks - industries are just composite price series
        return self.calculate_stock_rs(industry_composites, benchmark_data, periods, benchmark_name)

    def _create_ma_periods_for_timeframe(self, timeframe):
        """
        Create period configurations for MA calculations based on timeframe.

        Args:
            timeframe: String timeframe ('daily', 'weekly', 'monthly')

        Returns:
            List of tuples [(period_value, period_category, column_suffix), ...]
        """
        # For MA method, we use the configured MA periods regardless of timeframe
        # This is different from IBD which uses different period categories
        periods = []

        for ma_period in self.ma_periods:
            # Create period tuple similar to IBD format
            period_tuple = (ma_period, 'ma', f"{ma_period}MA")
            periods.append(period_tuple)

        logger.info(f"Created MA periods for {timeframe}: {periods}")
        return periods

    def process_stocks(self, ticker_list, timeframe, ticker_choice, benchmark_tickers):
        """
        Process Moving Average RS calculations for stocks.

        Args:
            ticker_list: List of ticker symbols
            timeframe: Timeframe to process ('daily', 'weekly', 'monthly')
            ticker_choice: User ticker choice number
            benchmark_tickers: List of benchmark ticker symbols

        Returns:
            RSResults object with calculation results
        """
        logger.info(f"Processing MA RS for {len(ticker_list)} stocks - {timeframe} timeframe")

        price_data = None
        benchmark_data = None
        rs_results = None
        combined_rs_data = None

        try:
            # Load price data using inherited method
            price_data = self._load_price_data(ticker_list, timeframe)
            if price_data is None or price_data.empty:
                logger.warning(f"No price data available for {timeframe} MA RS calculation")
                error_result = RSResults('ma', 'stocks', timeframe)
                error_result.metadata['error'] = "No price data available"
                return error_result

            # Load benchmark data (not used in MA calculations but needed for interface consistency)
            benchmark_data = self._load_benchmark_data(benchmark_tickers[0], timeframe)

            # Create MA-specific periods
            periods = self._create_ma_periods_for_timeframe(timeframe)

            # Calculate MA RS for each benchmark (maintain multi-benchmark support)
            files_created = []

            for benchmark_ticker in benchmark_tickers:
                logger.info(f"Processing MA RS vs {benchmark_ticker} benchmark")

                # Calculate MA RS values
                rs_results = self.calculate_stock_rs(price_data, benchmark_data, periods, benchmark_ticker)

                # Combine all MA periods into single DataFrame
                combined_rs_data = self._combine_ma_rs_periods(rs_results, price_data.index)

                # Add universe data and metadata (inherited from base class)
                final_data = self._prepare_final_dataset(combined_rs_data, 'stocks')

                # Save results with MA-specific naming
                output_file = self._save_ma_results(final_data, 'stocks', timeframe, ticker_choice, benchmark_ticker)
                if output_file:
                    files_created.append(output_file)

                # Clean up intermediate results after processing each benchmark
                del rs_results, combined_rs_data, final_data

            # Create lightweight result object without storing large DataFrames
            result = RSResults('ma', 'stocks', timeframe)
            result.metadata['universe_size'] = len(ticker_list)
            result.metadata['files_created'] = files_created
            # Don't store rs_values to save memory - files are saved already

            return result

        except Exception as e:
            logger.error(f"Error in MA RS stocks processing: {e}")
            error_result = RSResults('ma', 'stocks', timeframe)
            error_result.metadata['error'] = str(e)
            return error_result
        finally:
            # Explicit memory cleanup
            try:
                del price_data, benchmark_data
                if 'rs_results' in locals():
                    del rs_results
                if 'combined_rs_data' in locals():
                    del combined_rs_data
                import gc
                gc.collect()
            except:
                pass

    def process_sectors(self, sector_composites, timeframe, ticker_choice, benchmark_tickers):
        """
        Process Moving Average RS calculations for sectors.

        Args:
            sector_composites: DataFrame with sector composite data (pre-built)
            timeframe: Timeframe to process
            ticker_choice: User ticker choice number
            benchmark_tickers: List of benchmark ticker symbols

        Returns:
            RSResults object with calculation results
        """
        logger.info(f"Processing MA RS for sectors - {timeframe} timeframe")

        benchmark_data = None
        rs_results = None
        combined_rs_data = None

        try:
            # Use pre-built sector composites
            if sector_composites is None or sector_composites.empty:
                logger.warning(f"No sector composite data available for {timeframe}")
                error_result = RSResults('ma', 'sectors', timeframe)
                error_result.metadata['error'] = "No sector composite data available"
                return error_result

            # Load benchmark data
            benchmark_data = self._load_benchmark_data(benchmark_tickers[0], timeframe)

            # Create MA-specific periods
            periods = self._create_ma_periods_for_timeframe(timeframe)

            # Calculate and save results
            files_created = []

            for benchmark_ticker in benchmark_tickers:
                rs_results = self.calculate_sector_rs(sector_composites, benchmark_data, periods, benchmark_ticker)
                combined_rs_data = self._combine_ma_rs_periods(rs_results, sector_composites.index)
                final_data = self._prepare_final_dataset(combined_rs_data, 'sectors')

                output_file = self._save_ma_results(final_data, 'sectors', timeframe, ticker_choice, benchmark_ticker)
                if output_file:
                    files_created.append(output_file)

                # Clean up intermediate results after processing each benchmark
                del rs_results, combined_rs_data, final_data

            result = RSResults('ma', 'sectors', timeframe)
            result.metadata['universe_size'] = len(sector_composites.columns) if sector_composites is not None else 0
            result.metadata['files_created'] = files_created
            # Don't store rs_values to save memory - files are saved already
            return result

        except Exception as e:
            logger.error(f"Error in MA RS sectors processing: {e}")
            error_result = RSResults('ma', 'sectors', timeframe)
            error_result.metadata['error'] = str(e)
            return error_result
        finally:
            # Explicit memory cleanup
            try:
                del benchmark_data
                if 'rs_results' in locals():
                    del rs_results
                if 'combined_rs_data' in locals():
                    del combined_rs_data
                import gc
                gc.collect()
            except:
                pass

    def process_industries(self, industry_composites, timeframe, ticker_choice, benchmark_tickers):
        """
        Process Moving Average RS calculations for industries.

        Args:
            industry_composites: DataFrame with industry composite data (pre-built)
            timeframe: Timeframe to process
            ticker_choice: User ticker choice number
            benchmark_tickers: List of benchmark ticker symbols

        Returns:
            RSResults object with calculation results
        """
        logger.info(f"Processing MA RS for industries - {timeframe} timeframe")

        benchmark_data = None
        rs_results = None
        combined_rs_data = None

        try:
            # Use pre-built industry composites
            if industry_composites is None or industry_composites.empty:
                logger.warning(f"No industry composite data available for {timeframe}")
                error_result = RSResults('ma', 'industries', timeframe)
                error_result.metadata['error'] = "No industry composite data available"
                return error_result

            # Load benchmark data
            benchmark_data = self._load_benchmark_data(benchmark_tickers[0], timeframe)

            # Create MA-specific periods
            periods = self._create_ma_periods_for_timeframe(timeframe)

            # Calculate and save results
            files_created = []

            for benchmark_ticker in benchmark_tickers:
                rs_results = self.calculate_industry_rs(industry_composites, benchmark_data, periods, benchmark_ticker)
                combined_rs_data = self._combine_ma_rs_periods(rs_results, industry_composites.index)
                final_data = self._prepare_final_dataset(combined_rs_data, 'industries')

                output_file = self._save_ma_results(final_data, 'industries', timeframe, ticker_choice, benchmark_ticker)
                if output_file:
                    files_created.append(output_file)

                # Clean up intermediate results after processing each benchmark
                del rs_results, combined_rs_data, final_data

            result = RSResults('ma', 'industries', timeframe)
            result.metadata['universe_size'] = len(industry_composites.columns) if industry_composites is not None else 0
            result.metadata['files_created'] = files_created
            # Don't store rs_values to save memory - files are saved already
            return result

        except Exception as e:
            logger.error(f"Error in MA RS industries processing: {e}")
            error_result = RSResults('ma', 'industries', timeframe)
            error_result.metadata['error'] = str(e)
            return error_result
        finally:
            # Explicit memory cleanup
            try:
                del benchmark_data
                if 'rs_results' in locals():
                    del rs_results
                if 'combined_rs_data' in locals():
                    del combined_rs_data
                import gc
                gc.collect()
            except:
                pass

    def _combine_ma_rs_periods(self, rs_results, date_index):
        """
        Combine MA RS results from multiple periods into single DataFrame with tickers as index.
        Memory-optimized version using pandas operations.

        Args:
            rs_results: Dictionary of {column_suffix: DataFrame} with RS values (dates x tickers)
            date_index: Date index for the data

        Returns:
            Combined DataFrame with tickers as index and RS periods as columns
        """
        if not rs_results:
            return pd.DataFrame()

        # Get the latest date for each period
        latest_date = date_index[-1] if len(date_index) > 0 else None
        if latest_date is None:
            return pd.DataFrame()

        # Memory-efficient approach using pandas concat
        data_frames = []
        for period_name, period_df in rs_results.items():
            if period_df is not None and not period_df.empty and latest_date in period_df.index:
                # Extract latest row and transpose to get tickers as index
                latest_row = period_df.loc[[latest_date]].T
                latest_row.columns = [period_name]
                data_frames.append(latest_row)

        if data_frames:
            # Combine all periods efficiently
            result_df = pd.concat(data_frames, axis=1)
            result_df['date'] = latest_date
            result_df.index.name = 'ticker'

            # Clean up intermediate data frames
            del data_frames
            return result_df
        else:
            return pd.DataFrame()

    def calculate_composite_rs(self, composite_data, benchmark_data, periods, benchmark_name='SPY'):
        """
        Calculate Moving Average RS for composite data (sectors/industries).

        Args:
            composite_data: DataFrame with composite price data
            benchmark_data: Series with benchmark price data - not used for MA method
            periods: List of tuples [(period_value, period_category, column_suffix), ...]
            benchmark_name: Name of benchmark for column naming

        Returns:
            Dictionary of {column_suffix: DataFrame} with RS values
        """
        # Use same logic as stock RS calculation
        return self.calculate_stock_rs(composite_data, benchmark_data, periods, benchmark_name)

    def _save_ma_results(self, data, entity_type, timeframe, ticker_choice, benchmark_ticker):
        """
        Save MA RS results with MA-specific file naming.

        Args:
            data: DataFrame with RS results
            entity_type: Type of entity ('stocks', 'sectors', 'industries')
            timeframe: Timeframe processed
            ticker_choice: User ticker choice
            benchmark_ticker: Benchmark ticker name

        Returns:
            Path to saved file or None if failed
        """
        try:
            # Extract date from data
            data_date = self._extract_data_date_from_dataframe(data)

            # Create MA-specific filename with benchmark ticker
            filename = f"rs_{benchmark_ticker}_ma_{entity_type}_{timeframe}_{ticker_choice}_{data_date}.csv"
            output_path = Path(self.config.directories['RS_DIR']) / filename

            # Save using inherited method
            success = self._save_results_to_file(data, output_path)

            if success:
                logger.info(f"Saved MA RS {entity_type} results: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Failed to save MA RS {entity_type} results: {output_path}")
                return None

        except Exception as e:
            logger.error(f"Error saving MA RS results: {e}")
            return None