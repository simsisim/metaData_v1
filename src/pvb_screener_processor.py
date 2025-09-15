"""
PVB Screener Processor
=====================

Processes PVB (Price Volume Breakout) screener across different timeframes and generates
separate output files following the same accumulation pattern as basic_calculations.

Creates output files:
- pvb_screener_{choice}_daily_{date}.csv
- pvb_screener_{choice}_weekly_{date}.csv
- pvb_screener_{choice}_monthly_{date}.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from src.screeners.pvb_screener import pvb_screener
from src.user_defined_data import get_pvb_params_for_timeframe

logger = logging.getLogger(__name__)


class PVBScreenerProcessor:
    """
    Processes PVB screener for multiple timeframes and generates separate output files.
    """

    def __init__(self, config, user_config):
        """
        Initialize PVB screener processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with PVB screener settings
        """
        self.config = config
        self.user_config = user_config

        # Create PVB screener output directory
        self.pvb_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'pvb'
        self.pvb_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PVB screener processor initialized, output dir: {self.pvb_dir}")

    def process_pvb_batch(self, batch_data: Dict[str, pd.DataFrame], timeframe: str,
                         ticker_choice: int = 0) -> bool:
        """
        Process PVB screener for a specific batch and timeframe.
        Uses batch accumulation pattern like basic_calculations.

        Args:
            batch_data: Dictionary with ticker -> DataFrame mappings
            timeframe: 'daily', 'weekly', or 'monthly'
            ticker_choice: User ticker choice number

        Returns:
            True if processing succeeded, False otherwise
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} PVB screening")
            return False

        logger.debug(f"Processing PVB batch for {timeframe}: {len(batch_data)} tickers")

        # Initialize global accumulation storage (same pattern as stage_analysis)
        if not hasattr(self, 'all_results'):
            self.all_results = {}
        if timeframe not in self.all_results:
            self.all_results[timeframe] = {}

        # Get PVB parameters for this timeframe
        try:
            pvb_params = get_pvb_params_for_timeframe(self.user_config, timeframe)
            if pvb_params:
                pvb_params['ticker_choice'] = ticker_choice  # Add for filename generation
        except Exception as e:
            logger.error(f"Failed to get PVB parameters for {timeframe}: {e}")
            return False

        # Process batch using existing PVB screener logic
        processed_count = 0
        error_count = 0

        try:
            # Call existing PVB screener function
            batch_results = pvb_screener(batch_data, pvb_params)

            # Accumulate results by ticker (same pattern as stage_analysis)
            for result in batch_results:
                try:
                    ticker = result.get('ticker')
                    if not ticker:
                        logger.warning("PVB result missing ticker, skipping")
                        continue

                    # Convert result to consistent format for accumulation
                    result_row = {
                        'ticker': ticker,
                        'timeframe': timeframe,
                        'signal_date': result.get('signal_date', ''),
                        'signal_type': result.get('signal_type', ''),
                        'current_price': result.get('current_price', 0.0),
                        'sma': result.get('sma', 0.0),
                        'price_breakout': result.get('price_breakout', False),
                        'volume_breakout': result.get('volume_breakout', False),
                        'trend_aligned': result.get('trend_aligned', False),
                        'pvb_score': result.get('pvb_score', 0.0)
                    }

                    # Add any additional fields from the result
                    for key, value in result.items():
                        if key not in result_row:
                            result_row[key] = value

                    # Store result in global accumulation (same pattern as stage_analysis)
                    self.all_results[timeframe][ticker] = result_row
                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process PVB result for ticker: {e}")
                    error_count += 1
                    continue

        except Exception as e:
            logger.error(f"PVB batch processing failed for {timeframe}: {e}")
            return False

        logger.info(f"PVB batch summary ({timeframe}): {processed_count} processed, {error_count} errors")
        return True

    def save_pvb_matrix(self, ticker_choice: int = 0) -> Dict[str, str]:
        """
        Save accumulated PVB screener results to files for all timeframes.
        Similar to save_stage_analysis_matrix.

        Args:
            ticker_choice: User ticker choice number

        Returns:
            Dictionary mapping timeframe -> output file path
        """
        if not hasattr(self, 'all_results'):
            logger.warning("No accumulated PVB screener results to save")
            return {}

        results = {}

        for timeframe in self.all_results:
            if not self.all_results[timeframe]:
                logger.warning(f"No valid {timeframe} PVB screener results to save")
                continue

            # Convert accumulated results to DataFrame
            pvb_results = []
            for ticker, result in self.all_results[timeframe].items():
                if 'error' not in result:  # Skip error entries
                    pvb_results.append(result)

            if not pvb_results:
                logger.warning(f"No valid {timeframe} PVB screener results after filtering")
                continue

            # Create output DataFrame
            results_df = pd.DataFrame(pvb_results)

            # Sort by PVB score (descending) then by ticker
            if 'pvb_score' in results_df.columns:
                results_df = results_df.sort_values(['pvb_score', 'ticker'], ascending=[False, True])
            else:
                results_df = results_df.sort_values('ticker')

            # Generate output filename using consistent date format
            data_date = None
            if not results_df.empty and 'signal_date' in results_df.columns:
                # Use the latest signal date from the results
                try:
                    # Handle both string and datetime signal_date
                    latest_signal = results_df['signal_date'].iloc[0]
                    if isinstance(latest_signal, str):
                        data_date = latest_signal.replace('-', '')  # Convert 2025-08-29 to 20250829
                    else:
                        data_date = latest_signal.strftime('%Y%m%d')  # Handle datetime
                except:
                    pass

            # Fallback to current date if no signal date found
            if not data_date:
                data_date = datetime.now().strftime('%Y%m%d')
                logger.warning(f"Using current date as fallback for {timeframe} PVB screener: {data_date}")
            else:
                logger.info(f"Using signal date for {timeframe} PVB screener filename: {data_date}")

            output_filename = f"pvb_screener_{ticker_choice}_{timeframe}_{data_date}.csv"
            output_path = self.pvb_dir / output_filename

            # Save results
            results_df.to_csv(output_path, index=False, float_format='%.4f')

            logger.info(f"Saved {timeframe} PVB screener: {len(results_df)} results to {output_path}")
            results[timeframe] = str(output_path)

        logger.info(f"PVB screener matrix save completed: {len(results)} files generated")
        return results

    def process_all_timeframes(self, daily_data: Dict[str, pd.DataFrame] = None,
                              weekly_data: Dict[str, pd.DataFrame] = None,
                              monthly_data: Dict[str, pd.DataFrame] = None,
                              ticker_choice: int = 0) -> Dict[str, bool]:
        """
        Process PVB screener for all enabled timeframes.
        Now just accumulates results, actual file saving done by save_pvb_matrix.

        Args:
            daily_data: Daily market data
            weekly_data: Weekly market data
            monthly_data: Monthly market data
            ticker_choice: User ticker choice number

        Returns:
            Dictionary mapping timeframe -> success status (True/False)
        """
        results = {}

        # Process daily data
        if daily_data and getattr(self.user_config, 'pvb_daily_enable', True):
            daily_success = self.process_pvb_batch(daily_data, 'daily', ticker_choice)
            if daily_success:
                results['daily'] = True

        # Process weekly data
        if weekly_data and getattr(self.user_config, 'pvb_weekly_enable', True):
            weekly_success = self.process_pvb_batch(weekly_data, 'weekly', ticker_choice)
            if weekly_success:
                results['weekly'] = True

        # Process monthly data
        if monthly_data and getattr(self.user_config, 'pvb_monthly_enable', True):
            monthly_success = self.process_pvb_batch(monthly_data, 'monthly', ticker_choice)
            if monthly_success:
                results['monthly'] = True

        logger.info(f"PVB screener processing completed: {len(results)} timeframes processed")
        return results

    def get_processing_summary(self, results: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate a summary of the PVB screener processing.

        Args:
            results: Dictionary of timeframe -> file path mappings

        Returns:
            Summary dictionary
        """
        summary = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframes_processed': list(results.keys()),
            'total_files_generated': len(results),
            'output_directory': str(self.pvb_dir),
            'file_details': {}
        }

        for timeframe, file_path in results.items():
            try:
                # Read file to get basic stats
                df = pd.read_csv(file_path)

                # Get signal type distribution
                signal_counts = {}
                if 'signal_type' in df.columns:
                    signal_counts = df['signal_type'].value_counts().to_dict()

                summary['file_details'][timeframe] = {
                    'file_path': file_path,
                    'ticker_count': len(df),
                    'signal_distribution': signal_counts,
                    'file_size_mb': round(Path(file_path).stat().st_size / (1024*1024), 2)
                }
            except Exception as e:
                summary['file_details'][timeframe] = {
                    'file_path': file_path,
                    'error': str(e)
                }

        return summary


def run_pvb_screener_processing(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                               weekly_data: Dict[str, pd.DataFrame] = None,
                               monthly_data: Dict[str, pd.DataFrame] = None,
                               ticker_choice: int = 0) -> Dict[str, str]:
    """
    Convenience function to run PVB screener processing.
    Now uses accumulation pattern like stage_analysis.

    Args:
        config: Configuration object
        user_config: User configuration object
        daily_data: Daily market data
        weekly_data: Weekly market data
        monthly_data: Monthly market data
        ticker_choice: User ticker choice number

    Returns:
        Dictionary of timeframe -> output file path mappings
    """
    # Check if PVB screener is enabled
    if not getattr(user_config, 'pvb_enable', False):
        logger.info("PVB screener disabled")
        return {}

    processor = PVBScreenerProcessor(config, user_config)

    # Process all timeframes (accumulate results)
    processor.process_all_timeframes(daily_data, weekly_data, monthly_data, ticker_choice)

    # Save accumulated results to files
    results = processor.save_pvb_matrix(ticker_choice)

    # Generate and log summary
    summary = processor.get_processing_summary(results)
    logger.info(f"PVB Screener Summary: {summary['total_files_generated']} files, "
               f"{summary['timeframes_processed']} timeframes")

    return results