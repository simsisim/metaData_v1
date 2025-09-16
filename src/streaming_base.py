"""
Streaming Calculation Base Class
===============================

Memory-efficient base class for calculations that can be processed in batches
with immediate file writes instead of memory accumulation.

Provides:
- Streaming file writes (write after each batch)
- Memory cleanup and garbage collection
- Optimized data types for memory efficiency
- File management utilities
- Memory monitoring capabilities
"""

import pandas as pd
import numpy as np
import gc
import psutil
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StreamingCalculationBase(ABC):
    """
    Base class for memory-efficient streaming calculations.

    Implements the pattern:
    Process Batch → Write to File → Clear Memory → Repeat

    Instead of:
    Process All Batches → Accumulate in Memory → Write File
    """

    def __init__(self, config, user_config):
        """
        Initialize streaming processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with processing settings
        """
        self.config = config
        self.user_config = user_config

        # Setup efficient data types for memory optimization
        self.dtype_mapping = {
            # Price data (float32 = 4 bytes vs float64 = 8 bytes)
            'Open': 'float32',
            'High': 'float32',
            'Low': 'float32',
            'Close': 'float32',
            'Adj Close': 'float32',
            'price': 'float32',
            'sma': 'float32',
            'returns': 'float32',

            # Volume data (int32 = 4 bytes vs int64 = 8 bytes)
            'Volume': 'int32',
            'volume': 'int32',
            'avg_volume': 'int32',

            # Calculated metrics
            'rsi': 'float32',
            'volatility': 'float32',
            'rs_score': 'float32',
            'score': 'float32',
            'change_pct': 'float32',

            # Rankings and counts (int16 = 2 bytes, int8 = 1 byte)
            'rank': 'int16',
            'percentile': 'int16',
            'stage': 'int8',
            'signal': 'int8',
            'trend': 'int8',
            'count': 'int16',

            # Boolean flags (bool = 1 byte)
            'breakout': 'bool',
            'above_ma': 'bool',
            'volume_spike': 'bool'
        }

        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()

        logger.info(f"Streaming processor initialized. Initial memory: {self.initial_memory:.1f} MB")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def log_memory_usage(self, context: str = ""):
        """Log current memory usage with context."""
        current_memory = self.get_memory_usage()
        memory_change = current_memory - self.initial_memory
        logger.debug(f"Memory usage {context}: {current_memory:.1f} MB (change: {memory_change:+.1f} MB)")

    def optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame to memory-efficient data types.

        Args:
            df: DataFrame to optimize

        Returns:
            DataFrame with optimized dtypes
        """
        for col in df.columns:
            if col in self.dtype_mapping:
                try:
                    df[col] = df[col].astype(self.dtype_mapping[col])
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not convert {col} to {self.dtype_mapping[col]}: {e}")
        return df

    def cleanup_memory(self, *objects):
        """
        Explicit memory cleanup with garbage collection.

        Args:
            *objects: Objects to delete
        """
        for obj in objects:
            if obj is not None:
                del obj
        gc.collect()

    def extract_date_from_batch_data(self, batch_data: Dict[str, pd.DataFrame]) -> str:
        """
        Extract the latest date from batch data for filename generation.

        Args:
            batch_data: Dictionary of ticker -> DataFrame

        Returns:
            Date string in YYYYMMDD format
        """
        latest_date = None

        for ticker, df in batch_data.items():
            if df.empty:
                continue

            # Try different date column names
            date_cols = ['Date', 'date', 'timestamp']
            df_date_col = None

            for col in date_cols:
                if col in df.columns:
                    df_date_col = col
                    break

            if df_date_col is None and df.index.name in ['Date', 'date']:
                # Date is in index
                ticker_latest = df.index.max()
            elif df_date_col:
                # Date is in column
                ticker_latest = pd.to_datetime(df[df_date_col]).max()
            else:
                continue

            if latest_date is None or ticker_latest > latest_date:
                latest_date = ticker_latest

        if latest_date is None:
            # Fallback to current date
            latest_date = datetime.now()
            logger.warning("Could not extract date from batch data, using current date")

        # Convert to YYYYMMDD format
        if isinstance(latest_date, str):
            latest_date = pd.to_datetime(latest_date)

        return latest_date.strftime('%Y%m%d')

    def get_output_filename(self, batch_data: Dict[str, pd.DataFrame],
                          timeframe: str, ticker_choice: int) -> str:
        """
        Generate output filename based on batch data and parameters.

        Args:
            batch_data: First batch data to extract date
            timeframe: Processing timeframe (daily, weekly, monthly)
            ticker_choice: User ticker choice number

        Returns:
            Complete output file path
        """
        date_str = self.extract_date_from_batch_data(batch_data)
        filename = f"{self.get_calculation_name()}_{ticker_choice}_{timeframe}_{date_str}.csv"
        return str(self.get_output_directory() / filename)

    def write_csv_header(self, filepath: str, columns: List[str]):
        """
        Write CSV header to file.

        Args:
            filepath: Output file path
            columns: Column names for header
        """
        header_df = pd.DataFrame(columns=columns)
        header_df.to_csv(filepath, index=False)
        logger.debug(f"CSV header written to {filepath}")

    def append_results_to_csv(self, filepath: str, results: List[Dict[str, Any]]):
        """
        Append results to CSV file.

        Args:
            filepath: Output file path
            results: List of result dictionaries to append
        """
        if not results:
            return

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Optimize data types
        results_df = self.optimize_dataframe_dtypes(results_df)

        # Append to file
        results_df.to_csv(filepath, mode='a', header=False, index=False, float_format='%.4f')

        logger.debug(f"Appended {len(results)} results to {filepath}")

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str, ticker_choice: int,
                              output_file: str) -> int:
        """
        Process a single batch with streaming writes and memory cleanup.

        Args:
            batch_data: Dictionary of ticker -> DataFrame
            timeframe: Processing timeframe
            ticker_choice: User ticker choice
            output_file: Output file path

        Returns:
            Number of tickers processed
        """
        if not batch_data:
            logger.warning("Empty batch data provided")
            return 0

        # Process batch with memory optimization
        batch_results = []
        processed_count = 0

        for ticker, df in batch_data.items():
            try:
                # Optimize DataFrame memory usage
                df = self.optimize_dataframe_dtypes(df)

                # Calculate single ticker result (implemented by subclass)
                result = self.calculate_single_ticker(df, ticker, timeframe)

                if result:
                    batch_results.append(result)
                    processed_count += 1

                # Clean up individual ticker data immediately
                del df

            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                continue

        # Write results to file immediately
        if batch_results:
            self.append_results_to_csv(output_file, batch_results)

        # Comprehensive memory cleanup
        self.cleanup_memory(batch_results, batch_data)

        logger.debug(f"Processed batch: {processed_count} tickers, memory cleaned")
        return processed_count

    def process_timeframe_streaming(self, batches: List[Dict[str, pd.DataFrame]],
                                  timeframe: str, ticker_choice: int) -> Dict[str, Any]:
        """
        Process all batches for a timeframe using streaming approach.

        Args:
            batches: List of batch data dictionaries
            timeframe: Processing timeframe
            ticker_choice: User ticker choice

        Returns:
            Processing results summary
        """
        if not batches:
            logger.warning(f"No batches provided for {timeframe} streaming processing")
            return {}

        # Initialize output file using first batch
        output_file = self.get_output_filename(batches[0], timeframe, ticker_choice)

        # Process first batch to establish file structure
        first_batch_results = self.process_first_batch(batches[0], timeframe, ticker_choice)

        if not first_batch_results:
            logger.warning(f"No results from first batch for {timeframe}")
            return {}

        # Write header and first batch results
        columns = list(first_batch_results[0].keys())
        self.write_csv_header(output_file, columns)
        self.append_results_to_csv(output_file, first_batch_results)

        total_processed = len(first_batch_results)

        # Process remaining batches with streaming
        for batch_num, batch_data in enumerate(batches[1:], 1):
            batch_count = self.process_batch_streaming(
                batch_data, timeframe, ticker_choice, output_file
            )
            total_processed += batch_count

            # Memory monitoring (every 5 batches)
            if batch_num % 5 == 0:
                self.log_memory_usage(f"after batch {batch_num + 1}")

        # Final memory cleanup and logging
        final_memory = self.get_memory_usage()
        memory_saved = self.initial_memory - final_memory

        logger.info(f"Streaming processing completed for {timeframe}: "
                   f"{total_processed} tickers, {len(batches)} batches")
        logger.info(f"Final memory: {final_memory:.1f} MB (saved: {memory_saved:+.1f} MB)")

        return {
            'timeframe': timeframe,
            'output_file': output_file,
            'tickers_processed': total_processed,
            'batches_processed': len(batches),
            'memory_saved_mb': memory_saved
        }

    def process_first_batch(self, batch_data: Dict[str, pd.DataFrame],
                          timeframe: str, ticker_choice: int) -> List[Dict[str, Any]]:
        """
        Process first batch to establish file structure and extract date.

        Args:
            batch_data: First batch data
            timeframe: Processing timeframe
            ticker_choice: User ticker choice

        Returns:
            List of result dictionaries
        """
        batch_results = []

        # Check if this processor uses batch processing (like screeners)
        # If it has process_batch_streaming method, it's a batch processor
        if hasattr(self, 'process_batch_streaming') and callable(getattr(self, 'process_batch_streaming')):
            # For batch processors (screeners), call the batch processing method directly
            # But we need to capture the results instead of writing to file

            # Temporarily override the append_results_to_csv method to capture results
            original_append = self.append_results_to_csv
            captured_results = []

            def capture_results(file_path, results):
                captured_results.extend(results)

            self.append_results_to_csv = capture_results

            try:
                # Call the batch processing method
                result_count = self.process_batch_streaming(batch_data, timeframe, ticker_choice, 'dummy_file.csv')
                batch_results = captured_results
            finally:
                # Restore original method
                self.append_results_to_csv = original_append
        else:
            # For single ticker processors (basic calculations, stage analysis)
            for ticker, df in batch_data.items():
                try:
                    df = self.optimize_dataframe_dtypes(df)
                    result = self.calculate_single_ticker(df, ticker, timeframe)
                    if result:
                        batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error in first batch processing for {ticker}: {e}")
                    continue

        return batch_results

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str,
                              timeframe: str) -> Dict[str, Any]:
        """
        Calculate result for a single ticker.

        Args:
            df: Ticker price data
            ticker: Ticker symbol
            timeframe: Processing timeframe

        Returns:
            Dictionary with calculation results
        """
        pass

    @abstractmethod
    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        pass

    @abstractmethod
    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        pass