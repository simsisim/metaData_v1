"""
DRWISH Screener Processor
=========================

Processes DRWISH (Dr. Wish Suite) screener across different timeframes and multiple
parameter sets, generating separate output files following the accumulation pattern
used by other processors.

Creates output files:
- drwish_screener_{choice}_set1_daily_{date}.csv
- drwish_screener_{choice}_set2_daily_{date}.csv
- drwish_screener_{choice}_set1_weekly_{date}.csv
- drwish_screener_{choice}_set2_weekly_{date}.csv
- etc.
"""

import pandas as pd
import numpy as np
import gc
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from src.screeners.drwish_screener import drwish_screener
from src.user_defined_data import get_drwish_params_for_timeframe

logger = logging.getLogger(__name__)


class DRWISHScreenerProcessor:
    """
    Processes DRWISH screener for multiple timeframes and parameter sets,
    generating separate output files using accumulation pattern.
    """

    def __init__(self, config, user_config):
        """
        Initialize DRWISH screener processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with DRWISH screener settings
        """
        self.config = config
        self.user_config = user_config

        # Create DRWISH screener output directory
        self.drwish_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'drwish'
        self.drwish_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DRWISH screener processor initialized, output dir: {self.drwish_dir}")

    def process_drwish_batch(self, batch_data: Dict[str, pd.DataFrame], timeframe: str,
                           ticker_choice: int = 0) -> bool:
        """
        Process DRWISH screener for a specific batch and timeframe with multiple parameter sets.
        Uses nested accumulation pattern for multiple parameter sets.

        Args:
            batch_data: Dictionary with ticker -> DataFrame mappings
            timeframe: 'daily', 'weekly', or 'monthly'
            ticker_choice: User ticker choice number

        Returns:
            True if processing succeeded, False otherwise
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} DRWISH screening")
            return False

        logger.debug(f"Processing DRWISH batch for {timeframe}: {len(batch_data)} tickers")

        # Initialize nested global accumulation storage for parameter sets
        if not hasattr(self, 'all_results'):
            self.all_results = {}
        if timeframe not in self.all_results:
            self.all_results[timeframe] = {}

        # Get DRWISH parameter sets for this timeframe
        try:
            drwish_param_sets = get_drwish_params_for_timeframe(self.user_config, timeframe)
            if not drwish_param_sets:
                logger.warning(f"No DRWISH parameter sets found for {timeframe}")
                return False
        except Exception as e:
            logger.error(f"Failed to get DRWISH parameters for {timeframe}: {e}")
            return False

        # Process each parameter set
        total_processed = 0
        total_errors = 0

        for param_set in drwish_param_sets:
            set_name = param_set.get('parameter_set_name', 'set1')

            # Initialize accumulation for this parameter set
            if set_name not in self.all_results[timeframe]:
                self.all_results[timeframe][set_name] = {}

            logger.info(f"Processing DRWISH {set_name} for {timeframe} (lookback: {param_set['lookback_period']}, historical: {param_set['calculate_historical_GLB']})")

            processed_count = 0
            error_count = 0

            try:
                # Call existing DRWISH screener function for this parameter set
                batch_results = drwish_screener(batch_data, param_set)

                # Accumulate results by ticker for this parameter set
                for result in batch_results:
                    try:
                        ticker = result.get('ticker')
                        if not ticker:
                            logger.warning("DRWISH result missing ticker, skipping")
                            continue

                        # Convert result to consistent format for accumulation
                        result_row = {
                            'ticker': ticker,
                            'timeframe': timeframe,
                            'parameter_set': set_name,
                            'strategy': result.get('strategy', ''),
                            'signal_type': result.get('signal_type', ''),
                            'signal_date': result.get('signal_date', ''),
                            'current_price': result.get('current_price', 0.0),
                            'signal_price': result.get('signal_price', 0.0),
                            'score': result.get('score', 0.0),
                            'lookback_period': param_set['lookback_period'],
                            'historical_glb_period': param_set['calculate_historical_GLB'],
                            'confirmation_period': param_set['confirmation_period']
                        }

                        # Add any additional fields from the result
                        for key, value in result.items():
                            if key not in result_row:
                                result_row[key] = value

                        # Store result in nested accumulation (timeframe -> parameter_set -> ticker)
                        self.all_results[timeframe][set_name][ticker] = result_row
                        processed_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to process DRWISH {set_name} result for ticker: {e}")
                        error_count += 1
                        continue

            except Exception as e:
                logger.error(f"DRWISH {set_name} batch processing failed for {timeframe}: {e}")
                error_count += len(batch_data)

            logger.info(f"DRWISH {set_name} batch summary ({timeframe}): {processed_count} processed, {error_count} errors")
            total_processed += processed_count
            total_errors += error_count

        logger.info(f"DRWISH batch total summary ({timeframe}): {total_processed} processed, {total_errors} errors across {len(drwish_param_sets)} parameter sets")
        return True

    def save_drwish_matrix(self, ticker_choice: int = 0) -> Dict[str, str]:
        """
        Save accumulated DRWISH screener results to files for all timeframes and parameter sets.
        Only saves results for timeframes that have been processed.

        Args:
            ticker_choice: User ticker choice number

        Returns:
            Dictionary mapping timeframe_parameterset -> output file path
        """
        if not hasattr(self, 'all_results'):
            logger.warning("No accumulated DRWISH screener results to save")
            return {}

        results = {}
        timeframes_to_save = []

        # Only save results for the current timeframe being processed
        for timeframe in list(self.all_results.keys()):
            if self.all_results[timeframe]:  # Only if timeframe has results
                timeframes_to_save.append(timeframe)

        for timeframe in timeframes_to_save:
            # Process each parameter set for this timeframe
            for set_name in list(self.all_results[timeframe].keys()):
                if not self.all_results[timeframe][set_name]:
                    logger.warning(f"No valid {timeframe} {set_name} DRWISH screener results to save")
                    continue

                # Convert accumulated results to DataFrame
                drwish_results = []
                for ticker, result in self.all_results[timeframe][set_name].items():
                    if 'error' not in result:  # Skip error entries
                        drwish_results.append(result)

                if not drwish_results:
                    logger.warning(f"No valid {timeframe} {set_name} DRWISH screener results after filtering")
                    continue

                # Create output DataFrame
                results_df = pd.DataFrame(drwish_results)

                # Sort by score (descending) then by ticker
                if 'score' in results_df.columns:
                    results_df = results_df.sort_values(['score', 'ticker'], ascending=[False, True])
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
                    logger.warning(f"Using current date as fallback for {timeframe} {set_name} DRWISH screener: {data_date}")
                else:
                    logger.info(f"Using signal date for {timeframe} {set_name} DRWISH screener filename: {data_date}")

                output_filename = f"drwish_screener_{ticker_choice}_{set_name}_{timeframe}_{data_date}.csv"
                output_path = self.drwish_dir / output_filename

                # Save results
                results_df.to_csv(output_path, index=False, float_format='%.4f')

                # Memory cleanup after CSV write - free accumulated results data
                del results_df
                del self.all_results[timeframe][set_name]
                gc.collect()
                logger.debug(f"Memory cleaned up after {timeframe} {set_name} DRWISH screener CSV write and results cleared")

                logger.info(f"Saved {timeframe} {set_name} DRWISH screener: {len(drwish_results)} results to {output_path}")
                results[f"{timeframe}_{set_name}"] = str(output_path)

        # Clear the processed timeframes from all_results
        for timeframe in timeframes_to_save:
            if timeframe in self.all_results:
                del self.all_results[timeframe]

        logger.info(f"DRWISH screener matrix save completed: {len(results)} files generated")
        return results

    def process_all_timeframes(self, daily_data: Dict[str, pd.DataFrame] = None,
                              weekly_data: Dict[str, pd.DataFrame] = None,
                              monthly_data: Dict[str, pd.DataFrame] = None,
                              ticker_choice: int = 0) -> Dict[str, bool]:
        """
        Process DRWISH screener for all enabled timeframes.
        Now just accumulates results, actual file saving done by save_drwish_matrix.

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
        if daily_data and getattr(self.user_config, 'yf_daily_data', True):
            daily_success = self.process_drwish_batch(daily_data, 'daily', ticker_choice)
            if daily_success:
                results['daily'] = True

        # Process weekly data
        if weekly_data and getattr(self.user_config, 'yf_weekly_data', True):
            weekly_success = self.process_drwish_batch(weekly_data, 'weekly', ticker_choice)
            if weekly_success:
                results['weekly'] = True

        # Process monthly data
        if monthly_data and getattr(self.user_config, 'yf_monthly_data', True):
            monthly_success = self.process_drwish_batch(monthly_data, 'monthly', ticker_choice)
            if monthly_success:
                results['monthly'] = True

        logger.info(f"DRWISH screener processing completed: {len(results)} timeframes processed")
        return results

    def get_processing_summary(self, results: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate a summary of the DRWISH screener processing.

        Args:
            results: Dictionary of timeframe_parameterset -> file path mappings

        Returns:
            Summary dictionary
        """
        summary = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframes_processed': list(set([key.split('_')[0] for key in results.keys()])),
            'parameter_sets_processed': list(set([key.split('_')[1] for key in results.keys()])),
            'total_files_generated': len(results),
            'output_directory': str(self.drwish_dir),
            'file_details': {}
        }

        for timeframe_set, file_path in results.items():
            try:
                # Read file to get basic stats
                df = pd.read_csv(file_path)

                # Get strategy distribution
                strategy_counts = {}
                if 'strategy' in df.columns:
                    strategy_counts = df['strategy'].value_counts().to_dict()

                # Get signal type distribution
                signal_counts = {}
                if 'signal_type' in df.columns:
                    signal_counts = df['signal_type'].value_counts().to_dict()

                summary['file_details'][timeframe_set] = {
                    'file_path': file_path,
                    'ticker_count': len(df),
                    'strategy_distribution': strategy_counts,
                    'signal_distribution': signal_counts,
                    'file_size_mb': round(Path(file_path).stat().st_size / (1024*1024), 2)
                }
            except Exception as e:
                summary['file_details'][timeframe_set] = {
                    'file_path': file_path,
                    'error': str(e)
                }

        return summary


def run_drwish_screener_processing(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                                 weekly_data: Dict[str, pd.DataFrame] = None,
                                 monthly_data: Dict[str, pd.DataFrame] = None,
                                 ticker_choice: int = 0) -> Dict[str, str]:
    """
    Convenience function to run DRWISH screener processing.
    Uses accumulation pattern with multiple parameter sets support.

    Args:
        config: Configuration object
        user_config: User configuration object
        daily_data: Daily market data
        weekly_data: Weekly market data
        monthly_data: Monthly market data
        ticker_choice: User ticker choice number

    Returns:
        Dictionary of timeframe_parameterset -> output file path mappings
    """
    # Check if DRWISH screener is enabled
    if not getattr(user_config, 'drwish_enable', False):
        logger.info("DRWISH screener disabled")
        return {}

    processor = DRWISHScreenerProcessor(config, user_config)

    # Process all timeframes (accumulate results)
    processor.process_all_timeframes(daily_data, weekly_data, monthly_data, ticker_choice)

    # Save accumulated results to files
    results = processor.save_drwish_matrix(ticker_choice)

    # Generate and log summary
    summary = processor.get_processing_summary(results)
    logger.info(f"DRWISH Screener Summary: {summary['total_files_generated']} files, "
               f"{summary['timeframes_processed']} timeframes, {summary['parameter_sets_processed']} parameter sets")

    return results