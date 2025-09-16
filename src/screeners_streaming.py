"""
Screeners Streaming Processors
=============================

Memory-efficient streaming versions of all screener processors that process
batches and write results immediately instead of accumulating in memory.

Includes:
- PVB Screener Streaming
- ATR1 Screener Streaming
- Future screeners (Minervini, Giusti, etc.)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from src.streaming_base import StreamingCalculationBase
from src.screeners.pvb_screener import pvb_screener
from src.screeners.atr1_screener import atr1_screener
from src.screeners.drwish_screener import drwish_screener
from src.user_defined_data import get_pvb_params_for_timeframe, get_atr1_params_for_timeframe, get_drwish_params_for_timeframe

logger = logging.getLogger(__name__)


class PVBScreenerStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for PVB (Price Volume Breakout) screener.
    """

    def __init__(self, config, user_config):
        """
        Initialize PVB screener streaming processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with PVB screener settings
        """
        super().__init__(config, user_config)

        # Create PVB screener output directory
        self.pvb_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'pvb'
        self.pvb_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PVB screener streaming processor initialized, output dir: {self.pvb_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "pvb_screener"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.pvb_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        This method won't be used directly since PVB screener processes batches.
        Keeping for interface compatibility.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str, ticker_choice: int,
                              output_file: str) -> int:
        """
        Process a single batch with PVB screener and streaming writes.

        Args:
            batch_data: Dictionary of ticker -> DataFrame
            timeframe: Processing timeframe
            ticker_choice: User ticker choice
            output_file: Output file path

        Returns:
            Number of results processed
        """
        if not batch_data:
            logger.warning("Empty batch data provided for PVB screener")
            return 0

        try:
            # Get PVB parameters for this timeframe
            pvb_params = get_pvb_params_for_timeframe(self.user_config, timeframe)
            if not pvb_params:
                logger.warning(f"No PVB parameters configured for {timeframe}")
                return 0

            # Don't add ticker_choice to pvb_params - it breaks the screener
            # pvb_params['ticker_choice'] = ticker_choice

            # Process batch using existing PVB screener logic
            batch_results = pvb_screener(batch_data, pvb_params)

            if not batch_results:
                logger.warning(f"No PVB results from batch with {len(batch_data)} tickers")
                logger.warning(f"PVB params: {pvb_params}")
                return 0

            logger.info(f"PVB screener returned {len(batch_results)} results")

            # Convert results to consistent format (using actual PVB screener output fields)
            formatted_results = []
            for result in batch_results:
                try:
                    result_row = {
                        'ticker': result.get('ticker', ''),
                        'timeframe': timeframe,
                        'signal_date': result.get('signal_date', ''),
                        'signal_type': result.get('signal_type', ''),
                        'current_price': result.get('current_price', 0.0),
                        'signal_price': result.get('signal_price', 0.0),
                        'sma': result.get('sma', 0.0),
                        'volume': result.get('volume', 0),
                        'volume_highest': result.get('volume_highest', 0),
                        'days_since_signal': result.get('days_since_signal', 0),
                        'score': result.get('score', 0.0),  # PVB uses 'score' not 'pvb_score'
                        'screen_type': 'pvb'
                    }

                    # Add any additional fields from the result
                    for key, value in result.items():
                        if key not in result_row:
                            result_row[key] = value

                    formatted_results.append(result_row)

                except Exception as e:
                    logger.warning(f"Failed to format PVB result: {e}")
                    continue

            # Write results to file immediately
            if formatted_results:
                self.append_results_to_csv(output_file, formatted_results)

            # Comprehensive memory cleanup
            self.cleanup_memory(batch_results, formatted_results, batch_data)

            logger.debug(f"PVB batch processed: {len(formatted_results)} results")
            return len(formatted_results)

        except Exception as e:
            logger.error(f"PVB batch processing failed for {timeframe}: {e}")
            return 0


class ATR1ScreenerStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for ATR1 screener.
    """

    def __init__(self, config, user_config):
        """
        Initialize ATR1 screener streaming processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with ATR1 screener settings
        """
        super().__init__(config, user_config)

        # Create ATR1 screener output directory
        self.atr1_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'atr1'
        self.atr1_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ATR1 screener streaming processor initialized, output dir: {self.atr1_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "atr1_screener"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.atr1_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        This method won't be used directly since ATR1 screener processes batches.
        Keeping for interface compatibility.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str, ticker_choice: int,
                              output_file: str) -> int:
        """
        Process a single batch with ATR1 screener and streaming writes.

        Args:
            batch_data: Dictionary of ticker -> DataFrame
            timeframe: Processing timeframe
            ticker_choice: User ticker choice
            output_file: Output file path

        Returns:
            Number of results processed
        """
        if not batch_data:
            logger.warning("Empty batch data provided for ATR1 screener")
            return 0

        try:
            # Get ATR1 parameters for this timeframe
            atr1_params = get_atr1_params_for_timeframe(self.user_config, timeframe)
            if not atr1_params:
                logger.warning(f"No ATR1 parameters configured for {timeframe}")
                return 0

            # Don't add ticker_choice to atr1_params - it breaks the screener
            # atr1_params['ticker_choice'] = ticker_choice

            # Process batch using existing ATR1 screener logic
            batch_results = atr1_screener(batch_data, atr1_params)

            if not batch_results:
                logger.debug("No ATR1 results from batch")
                return 0

            # Convert results to consistent format (using actual ATR1 screener output fields)
            formatted_results = []
            for result in batch_results:
                try:
                    result_row = {
                        'ticker': result.get('ticker', ''),
                        'timeframe': timeframe,
                        'signal_date': result.get('signal_date', ''),
                        'signal_type': result.get('signal_type', ''),
                        'current_price': result.get('current_price', 0.0),
                        'signal_close_price': result.get('signal_close_price', 0.0),
                        'price_change_pct': result.get('price_change_pct', 0.0),
                        'days_since_signal': result.get('days_since_signal', 0),
                        'vstop': result.get('vstop', 0.0),
                        'vstop2': result.get('vstop2', 0.0),
                        'vstopseries': result.get('vstopseries', 0.0),
                        'uptrend': result.get('uptrend', False),
                        'score': result.get('score', 0.0),  # ATR1 uses 'score' not 'atr1_score'
                        'screen_type': 'atr1'
                    }

                    # Add any additional fields from the result
                    for key, value in result.items():
                        if key not in result_row:
                            result_row[key] = value

                    formatted_results.append(result_row)

                except Exception as e:
                    logger.warning(f"Failed to format ATR1 result: {e}")
                    continue

            # Write results to file immediately
            if formatted_results:
                self.append_results_to_csv(output_file, formatted_results)

            # Comprehensive memory cleanup
            self.cleanup_memory(batch_results, formatted_results, batch_data)

            logger.debug(f"ATR1 batch processed: {len(formatted_results)} results")
            return len(formatted_results)

        except Exception as e:
            logger.error(f"ATR1 batch processing failed for {timeframe}: {e}")
            return 0


class DRWISHScreenerStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for DRWISH screener (GLB, Blue Dot, Black Dot).
    """

    def __init__(self, config, user_config):
        """
        Initialize DRWISH screener streaming processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with DRWISH screener settings
        """
        super().__init__(config, user_config)

        # Create DRWISH screener output directory
        self.drwish_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'drwish'
        self.drwish_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DRWISH screener streaming processor initialized, output dir: {self.drwish_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "drwish_screener"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.drwish_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        This method won't be used directly since DRWISH screener processes batches.
        Keeping for interface compatibility.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str, ticker_choice: int,
                              output_file: str) -> int:
        """
        Process a single batch with DRWISH screener and streaming writes.

        Args:
            batch_data: Dictionary of ticker -> DataFrame
            timeframe: Processing timeframe
            ticker_choice: User ticker choice
            output_file: Output file path

        Returns:
            Number of results processed
        """
        if not batch_data:
            logger.warning("Empty batch data provided for DRWISH screener")
            return 0

        try:
            # Get DRWISH parameters for this timeframe
            drwish_params = get_drwish_params_for_timeframe(self.user_config, timeframe)
            if not drwish_params:
                logger.warning(f"No DRWISH parameters configured for {timeframe}")
                return 0

            # Don't add ticker_choice to drwish_params - it breaks the screener
            # drwish_params['ticker_choice'] = ticker_choice

            # Process batch using existing DRWISH screener logic
            batch_results = drwish_screener(batch_data, drwish_params)

            if not batch_results:
                logger.debug("No DRWISH results from batch")
                return 0

            logger.info(f"DRWISH screener returned {len(batch_results)} results")

            # Convert results to consistent format (using actual DRWISH screener output fields)
            formatted_results = []
            for result in batch_results:
                try:
                    result_row = {
                        'ticker': result.get('ticker', ''),
                        'timeframe': timeframe,
                        'strategy': result.get('strategy', ''),
                        'signal_type': result.get('signal_type', ''),
                        'signal_date': result.get('signal_date', ''),
                        'current_price': result.get('current_price', 0.0),
                        'signal_price': result.get('signal_price', 0.0),
                        'score': result.get('score', 0.0),
                        'screen_type': 'drwish'
                    }

                    # Add any additional fields from the result
                    for key, value in result.items():
                        if key not in result_row:
                            result_row[key] = value

                    formatted_results.append(result_row)

                except Exception as e:
                    logger.warning(f"Failed to format DRWISH result: {e}")
                    continue

            # Write results to file immediately
            if formatted_results:
                self.append_results_to_csv(output_file, formatted_results)

            # Comprehensive memory cleanup
            self.cleanup_memory(batch_results, formatted_results, batch_data)

            logger.debug(f"DRWISH batch processed: {len(formatted_results)} results")
            return len(formatted_results)

        except Exception as e:
            logger.error(f"DRWISH batch processing failed for {timeframe}: {e}")
            return 0


def run_pvb_screener_streaming(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                             weekly_data: Dict[str, pd.DataFrame] = None,
                             monthly_data: Dict[str, pd.DataFrame] = None,
                             ticker_choice: int = 0) -> Dict[str, str]:
    """
    Run PVB screener using streaming approach for memory efficiency.

    Args:
        config: Configuration object
        user_config: User configuration object
        daily_data: Daily market data batches
        weekly_data: Weekly market data batches
        monthly_data: Monthly market data batches
        ticker_choice: User ticker choice number

    Returns:
        Dictionary of timeframe -> output file path mappings
    """
    # Check if PVB screener is enabled
    if not getattr(user_config, 'pvb_enable', False):
        logger.info("PVB screener disabled")
        return {}

    processor = PVBScreenerStreamingProcessor(config, user_config)
    results = {}

    # Process each timeframe with available data
    timeframe_data = {
        'daily': daily_data,
        'weekly': weekly_data,
        'monthly': monthly_data
    }

    for timeframe, data in timeframe_data.items():
        # Check if this timeframe is enabled
        timeframe_enabled = getattr(user_config, f'pvb_{timeframe}_enable', True)
        if not timeframe_enabled:
            logger.info(f"PVB screener disabled for {timeframe} timeframe")
            continue

        if data:
            logger.info(f"Processing PVB screener for {timeframe} timeframe with streaming...")

            # Convert data to batch format if needed
            if isinstance(data, dict):
                batches = [data]  # Single batch
            else:
                batches = data  # Already in batch format

            # Process with streaming
            result = processor.process_timeframe_streaming(batches, timeframe, ticker_choice)

            if result and 'output_file' in result:
                results[timeframe] = result['output_file']
                logger.info(f"PVB screener completed for {timeframe}: "
                           f"{result['tickers_processed']} results processed, "
                           f"saved to {result['output_file']}")

                # Log memory savings
                if 'memory_saved_mb' in result:
                    logger.info(f"Memory saved: {result['memory_saved_mb']:.1f} MB")

    return results


def run_drwish_screener_streaming(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                                weekly_data: Dict[str, pd.DataFrame] = None,
                                monthly_data: Dict[str, pd.DataFrame] = None,
                                ticker_choice: int = 0) -> Dict[str, str]:
    """
    Run DRWISH screener using streaming approach for memory efficiency.

    Args:
        config: Configuration object
        user_config: User configuration object
        daily_data: Daily market data batches
        weekly_data: Weekly market data batches
        monthly_data: Monthly market data batches
        ticker_choice: User ticker choice number

    Returns:
        Dictionary of timeframe -> output file path mappings
    """
    # Check if DRWISH screener is enabled
    if not getattr(user_config, 'drwish_enable', False):
        logger.info("DRWISH screener disabled")
        return {}

    processor = DRWISHScreenerStreamingProcessor(config, user_config)
    results = {}

    # Process each timeframe with available data
    timeframe_data = {
        'daily': daily_data,
        'weekly': weekly_data,
        'monthly': monthly_data
    }

    for timeframe, data in timeframe_data.items():
        # Check if this timeframe is enabled
        timeframe_enabled = getattr(user_config, f'drwish_{timeframe}_enable', True)
        if not timeframe_enabled:
            logger.info(f"DRWISH screener disabled for {timeframe} timeframe")
            continue

        if data:
            logger.info(f"Processing DRWISH screener for {timeframe} timeframe with streaming...")

            # Convert data to batch format if needed
            if isinstance(data, dict):
                batches = [data]  # Single batch
            else:
                batches = data  # Already in batch format

            # Process with streaming
            result = processor.process_timeframe_streaming(batches, timeframe, ticker_choice)

            if result and 'output_file' in result:
                results[timeframe] = result['output_file']
                logger.info(f"DRWISH screener completed for {timeframe}: "
                           f"{result['tickers_processed']} results processed, "
                           f"saved to {result['output_file']}")

                # Log memory savings
                if 'memory_saved_mb' in result:
                    logger.info(f"Memory saved: {result['memory_saved_mb']:.1f} MB")

    return results


def run_atr1_screener_streaming(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                              weekly_data: Dict[str, pd.DataFrame] = None,
                              monthly_data: Dict[str, pd.DataFrame] = None,
                              ticker_choice: int = 0) -> Dict[str, str]:
    """
    Run ATR1 screener using streaming approach for memory efficiency.

    Args:
        config: Configuration object
        user_config: User configuration object
        daily_data: Daily market data batches
        weekly_data: Weekly market data batches
        monthly_data: Monthly market data batches
        ticker_choice: User ticker choice number

    Returns:
        Dictionary of timeframe -> output file path mappings
    """
    # Check if ATR1 screener is enabled
    if not getattr(user_config, 'atr1_enable', False):
        logger.info("ATR1 screener disabled")
        return {}

    processor = ATR1ScreenerStreamingProcessor(config, user_config)
    results = {}

    # Process each timeframe with available data
    timeframe_data = {
        'daily': daily_data,
        'weekly': weekly_data,
        'monthly': monthly_data
    }

    for timeframe, data in timeframe_data.items():
        # Check if this timeframe is enabled
        timeframe_enabled = getattr(user_config, f'atr1_{timeframe}_enable', True)
        if not timeframe_enabled:
            logger.info(f"ATR1 screener disabled for {timeframe} timeframe")
            continue

        if data:
            logger.info(f"Processing ATR1 screener for {timeframe} timeframe with streaming...")

            # Convert data to batch format if needed
            if isinstance(data, dict):
                batches = [data]  # Single batch
            else:
                batches = data  # Already in batch format

            # Process with streaming
            result = processor.process_timeframe_streaming(batches, timeframe, ticker_choice)

            if result and 'output_file' in result:
                results[timeframe] = result['output_file']
                logger.info(f"ATR1 screener completed for {timeframe}: "
                           f"{result['tickers_processed']} results processed, "
                           f"saved to {result['output_file']}")

                # Log memory savings
                if 'memory_saved_mb' in result:
                    logger.info(f"Memory saved: {result['memory_saved_mb']:.1f} MB")

    return results
