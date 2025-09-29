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
import gc
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from src.streaming_base import StreamingCalculationBase
from src.screeners.pvb_screener import pvb_screener
from src.screeners.atr1_screener import atr1_screener
from src.screeners.drwish_screener import drwish_screener
from src.screeners.guppy_screener import GuppyScreener
from src.user_defined_data import get_pvb_params_for_timeframe, get_atr1_params_for_timeframe, get_drwish_params_for_timeframe, get_volume_suite_params_for_timeframe, get_guppy_screener_params_for_timeframe

# Import original Volume Suite components (sophisticated algorithms)
from src.screeners.volume_suite_components.HVAbsoluteETC import run_HVAbsoluteStrategy_Enhanced
from src.screeners.volume_suite_components.HVStdv import run_HVStdvStrategy
from src.screeners.volume_suite_components.enhanced_volume_anomaly import run_enhanced_volume_anomaly_detection
from src.screeners.volume_suite_components.volume_indicators import run_volume_indicators_analysis

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

        # Create PVB screener output directory using user-configurable path
        self.pvb_dir = config.directories['PVB_SCREENER_DIR']
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

        # Create ATR1 screener output directory using user-configurable path
        self.atr1_dir = config.directories['ATR1_SCREENER_DIR']
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

        # Create DRWISH screener output directory using user-configurable path
        self.drwish_dir = config.directories['DRWISH_SCREENER_DIR']
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
            # Get DRWISH parameters for this timeframe (now returns list of parameter sets)
            drwish_param_sets = get_drwish_params_for_timeframe(self.user_config, timeframe)
            if not drwish_param_sets:
                logger.warning(f"No DRWISH parameters configured for {timeframe}")
                return 0

            total_processed = 0

            # Process each parameter set
            for param_set in drwish_param_sets:
                set_name = param_set.get('parameter_set_name', 'set1')
                logger.info(f"Processing DRWISH {set_name} for {timeframe} (lookback: {param_set['lookback_period']}, historical: {param_set['calculate_historical_GLB']})")

                # Process batch using existing DRWISH screener logic
                batch_results = drwish_screener(batch_data, param_set)

                if not batch_results:
                    logger.debug(f"No DRWISH {set_name} results from batch")
                    continue

                logger.info(f"DRWISH {set_name} screener returned {len(batch_results)} results")

                # Convert results to consistent format (using actual DRWISH screener output fields)
                formatted_results = []
                for result in batch_results:
                    try:
                        result_row = {
                            'ticker': result.get('ticker', ''),
                            'timeframe': timeframe,
                            'parameter_set': set_name,  # Add parameter set identifier
                            'strategy': result.get('strategy', ''),
                            'signal_type': result.get('signal_type', ''),
                            'signal_date': result.get('signal_date', ''),
                            'current_price': result.get('current_price', 0.0),
                            'signal_price': result.get('signal_price', 0.0),
                            'score': result.get('score', 0.0),
                            'screen_type': 'drwish',
                            'lookback_period': param_set['lookback_period'],
                            'historical_glb_period': param_set['calculate_historical_GLB']
                        }

                        # Add any additional fields from the result
                        for key, value in result.items():
                            if key not in result_row:
                                result_row[key] = value

                        formatted_results.append(result_row)

                    except Exception as e:
                        logger.warning(f"Failed to format DRWISH {set_name} result: {e}")
                        continue

                # Generate parameter set specific output file
                set_output_file = output_file.replace('.csv', f'_{set_name}.csv')

                # Write results to parameter set specific file
                if formatted_results:
                    self.append_results_to_csv(set_output_file, formatted_results)
                    total_processed += len(formatted_results)

                # Comprehensive memory cleanup for this parameter set
                self.cleanup_memory(batch_results, formatted_results, None)

            logger.debug(f"DRWISH batch processed: {total_processed} total results across {len(drwish_param_sets)} parameter sets")
            return total_processed

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
    if not getattr(user_config, 'pvb_TWmodel_enable', False):
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
        timeframe_enabled = getattr(user_config, f'pvb_TWmodel_{timeframe}_enable', True)
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


class VolumeSuiteStreamingProcessor(StreamingCalculationBase):
    """
    Volume Suite streaming processor using original sophisticated algorithms.
    Follows PVB TW streaming pattern with batch-by-batch processing and immediate writes.
    """

    def __init__(self, config, user_config):
        super().__init__(config, user_config)

        # Create Volume Suite output directory
        self.volume_suite_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'volume_suite'
        self.volume_suite_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Volume Suite streaming processor initialized, output dir: {self.volume_suite_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "volume_suite_screener"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.volume_suite_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        This method won't be used directly since Volume Suite screener processes batches.
        Keeping for interface compatibility.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str, ticker_choice: int = 0) -> Dict[str, Any]:
        """
        Process Volume Suite batch using original sophisticated algorithms.
        Following PVB TW streaming pattern with immediate writes per component.
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} Volume Suite streaming")
            return {}

        logger.debug(f"Processing Volume Suite batch for {timeframe}: {len(batch_data)} tickers")

        # Get Volume Suite parameters for this timeframe (like PVB TW pattern)
        try:
            volume_suite_params = get_volume_suite_params_for_timeframe(self.user_config, timeframe)
            if not volume_suite_params:
                logger.warning(f"No Volume Suite parameters found for {timeframe}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get Volume Suite parameters for {timeframe}: {e}")
            return {}

        # Get component flags from nested structure
        vs_config = volume_suite_params.get('volume_suite', {})
        current_date = datetime.now().strftime('%Y%m%d')

        component_files = []
        total_processed = 0
        total_errors = 0

        # Process each component using ORIGINAL sophisticated algorithms
        try:
            # 1. HV Absolute Analysis (Original Algorithm)
            if vs_config.get('enable_hv_absolute', False):
                hv_absolute_results, hv_processed = self._process_hv_absolute_component(
                    batch_data, vs_config, timeframe, current_date
                )
                if hv_absolute_results:
                    component_files.append(hv_absolute_results)
                    total_processed += hv_processed

            # 2. HV StdDev Analysis (Original Algorithm)
            if vs_config.get('enable_hv_stdv', False):
                hv_stdv_results, stdv_processed = self._process_hv_stdv_component(
                    batch_data, vs_config, timeframe, current_date
                )
                if hv_stdv_results:
                    component_files.append(hv_stdv_results)
                    total_processed += stdv_processed

            # 3. Enhanced Volume Anomaly (Original Algorithm)
            if vs_config.get('enable_enhanced_anomaly', False):
                enhanced_results, enhanced_processed = self._process_enhanced_anomaly_component(
                    batch_data, vs_config, timeframe, current_date
                )
                if enhanced_results:
                    component_files.append(enhanced_results)
                    total_processed += enhanced_processed

            # 4. Volume Indicators (Original Algorithm)
            if vs_config.get('enable_volume_indicators', False):
                vol_indicators_results, vol_processed = self._process_volume_indicators_component(
                    batch_data, vs_config, timeframe, current_date
                )
                if vol_indicators_results:
                    component_files.append(vol_indicators_results)
                    total_processed += vol_processed

            # 5. PVB ClModel (Separate from PVB TW)
            if vs_config.get('enable_pvb_clmodel_integration', False):
                pvb_clmodel_results, pvb_processed = self._process_pvb_clmodel_component(
                    batch_data, vs_config, timeframe, current_date
                )
                if pvb_clmodel_results:
                    component_files.append(pvb_clmodel_results)
                    total_processed += pvb_processed

        except Exception as e:
            logger.error(f"Error in Volume Suite batch processing: {e}")
            total_errors += 1

        # Memory cleanup (like PVB TW)
        gc.collect()

        logger.info(f"Volume Suite batch summary ({timeframe}): {total_processed} total results across {len(component_files)} component files, {total_errors} errors")

        return {
            'output_files': component_files,
            'tickers_processed': total_processed,
            'errors': total_errors,
            'timeframe': timeframe,
            'components_processed': len(component_files)
        }

    def _process_hv_absolute_component(self, batch_data: Dict[str, pd.DataFrame],
                                     vs_config: Dict, timeframe: str, current_date: str) -> Tuple[str, int]:
        """Process HV Absolute component using original sophisticated algorithm"""
        try:
            # Get HV Absolute parameters (original format)
            hv_absolute_params = {
                'month_cuttoff': vs_config.get('hv_month_cutoff', 15),
                'day_cuttoff': vs_config.get('hv_day_cutoff', 3),
                'std_cuttoff': vs_config.get('hv_std_cutoff', 10),  # 10σ threshold
                'min_stock_volume': vs_config.get('hv_min_volume', 100000),
                'min_price': vs_config.get('hv_min_price', 20),
                'use_enhanced_filtering': True
            }

            # Call ORIGINAL sophisticated function
            hv_results = run_HVAbsoluteStrategy_Enhanced(batch_data, hv_absolute_params)

            if not hv_results:
                return None, 0

            # Use ORIGINAL file naming convention
            output_filename = f"volume_suite_hv_absolute_{timeframe}_{current_date}.csv"
            output_file = self.volume_suite_dir / output_filename

            # Write results with ORIGINAL field structure (not my basic fields)
            self._write_component_results(output_file, hv_results)

            logger.info(f"HV Absolute: {len(hv_results)} results saved to {output_file}")
            return str(output_file), len(hv_results)

        except Exception as e:
            logger.error(f"Error processing HV Absolute component: {e}")
            return None, 0

    def _process_hv_stdv_component(self, batch_data: Dict[str, pd.DataFrame],
                                 vs_config: Dict, timeframe: str, current_date: str) -> Tuple[str, int]:
        """Process HV StdDev component using original sophisticated algorithm"""
        try:
            # Get HV StdDev parameters (original format)
            hv_stdv_params = {
                'std_cuttoff': vs_config.get('stdv_cutoff', 12),  # 12σ threshold
                'min_stock_volume': vs_config.get('stdv_min_volume', 10000)
            }

            # Call ORIGINAL sophisticated function
            stdv_results = run_HVStdvStrategy(batch_data, hv_stdv_params)

            if not stdv_results:
                return None, 0

            # Use ORIGINAL file naming convention
            output_filename = f"volume_suite_hv_stdv_{timeframe}_{current_date}.csv"
            output_file = self.volume_suite_dir / output_filename

            # Write results with ORIGINAL field structure
            self._write_component_results(output_file, stdv_results)

            logger.info(f"HV StdDev: {len(stdv_results)} results saved to {output_file}")
            return str(output_file), len(stdv_results)

        except Exception as e:
            logger.error(f"Error processing HV StdDev component: {e}")
            return None, 0

    def _process_enhanced_anomaly_component(self, batch_data: Dict[str, pd.DataFrame],
                                          vs_config: Dict, timeframe: str, current_date: str) -> Tuple[str, int]:
        """Process Enhanced Anomaly component using original sophisticated algorithm"""
        try:
            # Get Enhanced Anomaly parameters (original format)
            enhanced_params = vs_config  # Use all vs_config parameters

            # Call ORIGINAL sophisticated function
            enhanced_results = run_enhanced_volume_anomaly_detection(batch_data, enhanced_params)

            if not enhanced_results:
                return None, 0

            # Use ORIGINAL file naming convention
            output_filename = f"volume_suite_enhanced_anomaly_{timeframe}_{current_date}.csv"
            output_file = self.volume_suite_dir / output_filename

            # Write results with ORIGINAL field structure
            self._write_component_results(output_file, enhanced_results)

            logger.info(f"Enhanced Anomaly: {len(enhanced_results)} results saved to {output_file}")
            return str(output_file), len(enhanced_results)

        except Exception as e:
            logger.error(f"Error processing Enhanced Anomaly component: {e}")
            return None, 0

    def _process_volume_indicators_component(self, batch_data: Dict[str, pd.DataFrame],
                                           vs_config: Dict, timeframe: str, current_date: str) -> Tuple[str, int]:
        """Process Volume Indicators component using original sophisticated algorithm"""
        try:
            # Get Volume Indicators parameters (original format)
            vol_indicators_params = {
                'vroc_threshold': vs_config.get('vroc_threshold', 100),
                'vroc_period': vs_config.get('vroc_period', 25),
                'rvol_threshold': vs_config.get('rvol_threshold', 2),
                'rvol_period': vs_config.get('rvol_period', 20),
                'rvol_extreme_threshold': vs_config.get('rvol_extreme_threshold', 5),
                'mfi_period': vs_config.get('mfi_period', 14),
                'mfi_overbought': vs_config.get('mfi_overbought', 80),
                'mfi_oversold': vs_config.get('mfi_oversold', 20),
                'vpt_threshold': vs_config.get('vpt_threshold', 0.05),
                'adtv_3m_threshold': vs_config.get('adtv_3m_threshold', 0.2),
                'adtv_6m_threshold': vs_config.get('adtv_6m_threshold', 0.5),
                'adtv_1y_threshold': vs_config.get('adtv_1y_threshold', 2),
                'adtv_min_volume': vs_config.get('adtv_min_volume', 100000)
            }

            # Call ORIGINAL sophisticated function
            vol_indicators_analysis = run_volume_indicators_analysis(batch_data, vol_indicators_params)

            if not vol_indicators_analysis or not vol_indicators_analysis.get('signals'):
                return None, 0

            # Extract signals and format like original output:
            # ticker,signal_date,signal_type,screen_type,price,volume,strength,raw_data
            formatted_results = []
            signals = vol_indicators_analysis['signals']

            # Process VROC anomalies (like original file format)
            for vroc_signal in signals.get('vroc_anomalies', []):
                raw_data_dict = {
                    'VROC': vroc_signal.get('VROC', 0),
                    'VROC_Zscore': vroc_signal.get('VROC_Zscore', 0),
                    'Signal_Strength': vroc_signal.get('Signal_Strength', 'Moderate')
                }
                formatted_results.append({
                    'ticker': vroc_signal.get('Ticker', ''),
                    'signal_date': vroc_signal.get('Date', ''),
                    'signal_type': 'vroc_anomalies',
                    'screen_type': 'volume_indicators',
                    'price': vroc_signal.get('Close', 0),
                    'volume': vroc_signal.get('Volume', 0),
                    'strength': vroc_signal.get('Signal_Strength', 'Moderate'),
                    'raw_data': str(raw_data_dict)
                })

            # Process RVOL anomalies
            for rvol_signal in signals.get('rvol_anomalies', []):
                raw_data_dict = {
                    'RVOL': rvol_signal.get('RVOL', 0),
                    'Volume_vs_Average': rvol_signal.get('Volume_vs_Average', ''),
                    'Signal_Strength': rvol_signal.get('Signal_Strength', 'Moderate')
                }
                formatted_results.append({
                    'ticker': rvol_signal.get('Ticker', ''),
                    'signal_date': rvol_signal.get('Date', ''),
                    'signal_type': 'rvol_anomalies',
                    'screen_type': 'volume_indicators',
                    'price': rvol_signal.get('Close', 0),
                    'volume': rvol_signal.get('Volume', 0),
                    'strength': rvol_signal.get('Signal_Strength', 'Moderate'),
                    'raw_data': str(raw_data_dict)
                })

            # Process other signal types (ADTV, MFI, VPT) following same pattern
            for adtv_signal in signals.get('adtv_trend_anomalies', []):
                raw_data_dict = {
                    'ADTV_Ratio': adtv_signal.get('ADTV_Ratio', 0),
                    'Timeframe': adtv_signal.get('Timeframe', ''),
                    'Signal_Strength': adtv_signal.get('Signal_Strength', 'Moderate')
                }
                formatted_results.append({
                    'ticker': adtv_signal.get('Ticker', ''),
                    'signal_date': adtv_signal.get('Date', ''),
                    'signal_type': 'adtv_trend_anomalies',
                    'screen_type': 'volume_indicators',
                    'price': adtv_signal.get('Close', 0),
                    'volume': adtv_signal.get('Volume', 0),
                    'strength': adtv_signal.get('Signal_Strength', 'Moderate'),
                    'raw_data': str(raw_data_dict)
                })

            for mfi_signal in signals.get('mfi_signals', []):
                raw_data_dict = {
                    'MFI': mfi_signal.get('MFI', 0),
                    'Signal_Type': mfi_signal.get('Signal_Type', ''),
                    'Signal_Strength': mfi_signal.get('Signal_Strength', 'Moderate')
                }
                formatted_results.append({
                    'ticker': mfi_signal.get('Ticker', ''),
                    'signal_date': mfi_signal.get('Date', ''),
                    'signal_type': 'mfi_signals',
                    'screen_type': 'volume_indicators',
                    'price': mfi_signal.get('Close', 0),
                    'volume': mfi_signal.get('Volume', 0),
                    'strength': mfi_signal.get('Signal_Strength', 'Moderate'),
                    'raw_data': str(raw_data_dict)
                })

            for vpt_signal in signals.get('vpt_signals', []):
                raw_data_dict = {
                    'VPT': vpt_signal.get('VPT', 0),
                    'VPT_Change': vpt_signal.get('VPT_Change', 0),
                    'Signal_Direction': vpt_signal.get('Signal_Direction', ''),
                    'Signal_Strength': vpt_signal.get('Signal_Strength', 'Moderate')
                }
                formatted_results.append({
                    'ticker': vpt_signal.get('Ticker', ''),
                    'signal_date': vpt_signal.get('Date', ''),
                    'signal_type': 'vpt_signals',
                    'screen_type': 'volume_indicators',
                    'price': vpt_signal.get('Close', 0),
                    'volume': vpt_signal.get('Volume', 0),
                    'strength': vpt_signal.get('Signal_Strength', 'Moderate'),
                    'raw_data': str(raw_data_dict)
                })

            if not formatted_results:
                return None, 0

            # Use ORIGINAL file naming convention
            output_filename = f"volume_suite_volume_indicators_{timeframe}_{current_date}.csv"
            output_file = self.volume_suite_dir / output_filename

            # Write results with ORIGINAL field structure matching old format
            self._write_component_results(output_file, formatted_results)

            logger.info(f"Volume Indicators: {len(formatted_results)} results saved to {output_file}")
            return str(output_file), len(formatted_results)

        except Exception as e:
            logger.error(f"Error processing Volume Indicators component: {e}")
            return None, 0

    def _process_pvb_clmodel_component(self, batch_data: Dict[str, pd.DataFrame],
                                     vs_config: Dict, timeframe: str, current_date: str) -> Tuple[str, int]:
        """Process PVB ClModel component (separate from PVB TW)"""
        try:
            # Get PVB ClModel parameters (separate from PVB TW)
            pvb_clmodel_params = {
                'pvb_clmodel_price_period': vs_config.get('pvb_clmodel_price_period', 15),
                'pvb_clmodel_volume_period': vs_config.get('pvb_clmodel_volume_period', 15),
                'pvb_clmodel_trend_length': vs_config.get('pvb_clmodel_trend_length', 50),
                'pvb_clmodel_volume_multiplier': vs_config.get('pvb_clmodel_volume_multiplier', 1.5),
                'pvb_clmodel_direction': vs_config.get('pvb_clmodel_direction', 'Long')
            }

            # Call PVB ClModel function (separate implementation)
            pvb_clmodel_results = self._run_pvb_clmodel_analysis(batch_data, pvb_clmodel_params)

            if not pvb_clmodel_results:
                return None, 0

            # Use ORIGINAL file naming convention (changed from 'pvb' to 'pvb_clmodel')
            output_filename = f"volume_suite_pvb_clmodel_{timeframe}_{current_date}.csv"
            output_file = self.volume_suite_dir / output_filename

            # Write results
            self._write_component_results(output_file, pvb_clmodel_results)

            logger.info(f"PVB ClModel: {len(pvb_clmodel_results)} results saved to {output_file}")
            return str(output_file), len(pvb_clmodel_results)

        except Exception as e:
            logger.error(f"Error processing PVB ClModel component: {e}")
            return None, 0

    def _run_pvb_clmodel_analysis(self, batch_data: Dict[str, pd.DataFrame], params: Dict) -> List[Dict]:
        """PVB ClModel analysis (separate from PVB TW screener)"""
        results = []

        for ticker, ticker_data in batch_data.items():
            try:
                if ticker_data.empty or 'Close' not in ticker_data.columns or 'Volume' not in ticker_data.columns:
                    continue

                close_prices = ticker_data['Close']
                volume = ticker_data['Volume']

                # Get parameters
                price_period = params.get('pvb_clmodel_price_period', 15)
                volume_period = params.get('pvb_clmodel_volume_period', 15)
                trend_length = params.get('pvb_clmodel_trend_length', 50)
                volume_multiplier = params.get('pvb_clmodel_volume_multiplier', 1.5)
                direction = params.get('pvb_clmodel_direction', 'Long')

                # Price momentum
                if len(close_prices) > price_period:
                    price_momentum = (close_prices.iloc[-1] - close_prices.iloc[-price_period]) / close_prices.iloc[-price_period]
                else:
                    price_momentum = 0

                # Volume analysis
                vol_avg = volume.rolling(window=volume_period).mean()
                if len(vol_avg) > 0 and vol_avg.iloc[-1] > 0:
                    vol_ratio = volume.iloc[-1] / vol_avg.iloc[-1]
                else:
                    vol_ratio = 1

                # Trend analysis
                if len(close_prices) >= trend_length:
                    trend_start = close_prices.iloc[-trend_length]
                    trend_end = close_prices.iloc[-1]
                    trend_score = (trend_end - trend_start) / trend_start
                else:
                    trend_score = 0

                # Combined score
                combined_score = (price_momentum * 0.4) + ((vol_ratio - 1) * 0.4) + (trend_score * 0.2)

                # Signal generation
                if direction.lower() == 'long':
                    if vol_ratio > volume_multiplier and price_momentum > 0:
                        signal = "BUY"
                    elif vol_ratio < (1/volume_multiplier) and price_momentum < 0:
                        signal = "SELL"
                    else:
                        signal = "HOLD"
                else:
                    if vol_ratio > volume_multiplier and price_momentum < 0:
                        signal = "SELL"
                    elif vol_ratio < (1/volume_multiplier) and price_momentum > 0:
                        signal = "BUY"
                    else:
                        signal = "HOLD"

                result = {
                    'Ticker': ticker,
                    'Date': ticker_data.index[-1] if len(ticker_data) > 0 else None,
                    'Price_Momentum': round(price_momentum * 100, 2),
                    'Volume_Ratio': round(vol_ratio, 2),
                    'Trend_Score': round(trend_score * 100, 2),
                    'Combined_Score': round(combined_score * 100, 2),
                    'Signal': signal,
                    'Close': close_prices.iloc[-1] if len(close_prices) > 0 else None
                }

                results.append(result)

            except Exception as e:
                logger.warning(f"Error processing PVB ClModel for {ticker}: {e}")
                continue

        return results

    def _write_component_results(self, output_file: Path, results: List[Dict]):
        """Write component results to CSV preserving original format"""
        if not results:
            return

        # Convert to DataFrame and save (preserving original field names)
        df = pd.DataFrame(results)

        # Write to CSV (append if file exists for streaming)
        if output_file.exists():
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, index=False)

        # Memory cleanup
        del df
        gc.collect()



def run_all_volume_suite_streaming(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                                 weekly_data: Dict[str, pd.DataFrame] = None,
                                 monthly_data: Dict[str, pd.DataFrame] = None,
                                 ticker_choice: int = 0) -> Dict[str, str]:
    """
    Run Volume Suite screener using ORIGINAL sophisticated algorithms with streaming.
    Following PVB TW streaming pattern with immediate writes per component.
    """
    # Check if Volume Suite screener is enabled
    if not getattr(user_config, "volume_suite_enable", False):
        logger.info("Volume Suite screener disabled")
        return {}

    logger.info("Starting Volume Suite screener with streaming pattern...")

    processor = VolumeSuiteStreamingProcessor(config, user_config)
    results = {}

    # Process each timeframe with available data
    timeframe_data = {
        "daily": daily_data,
        "weekly": weekly_data,
        "monthly": monthly_data
    }

    for timeframe, data in timeframe_data.items():
        # Check if this timeframe is enabled
        timeframe_enabled = getattr(user_config, f"volume_suite_{timeframe}_enable", True)
        if not timeframe_enabled:
            logger.info(f"Volume Suite screener disabled for {timeframe} timeframe")
            continue

        if data:
            logger.info(f"Processing Volume Suite screener for {timeframe} timeframe with streaming...")

            # Convert data to batch format if needed
            if isinstance(data, dict):
                batches = [data]  # Single batch
            else:
                batches = data  # Already in batch format

            # Process with streaming (using original sophisticated algorithms)
            result = processor.process_batch_streaming(batches[0], timeframe, ticker_choice)

            if result and "tickers_processed" in result:
                results[timeframe] = result["tickers_processed"]
                tickers_processed = result["tickers_processed"]
                logger.info(f"Volume Suite screener completed for {timeframe}: {tickers_processed} results processed")

                # Show component files created
                if "output_files" in result:
                    output_files = result["output_files"]
                    logger.info(f"Component files created: {len(output_files)}")
                    for file_path in output_files:
                        logger.info(f"  - {file_path}")

    return results


class GuppyScreenerStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for GUPPY GMMA screener following PVB TW pattern.
    Memory-efficient batch processing with immediate writes.
    """

    def __init__(self, config, user_config):
        """
        Initialize GUPPY screener streaming processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with GUPPY screener settings
        """
        super().__init__(config, user_config)

        # Create GUPPY screener output directory
        self.guppy_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'guppy'
        self.guppy_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GUPPY screener instance with configuration
        guppy_config = {
            'timeframe': 'daily',  # Will be overridden per timeframe
            'guppy_screener': {
                'ma_type': getattr(user_config, 'guppy_screener_ma_type', 'EMA'),
                'short_term_emas': getattr(user_config, 'guppy_screener_short_term_emas', [3, 5, 8, 10, 12, 15]),
                'long_term_emas': getattr(user_config, 'guppy_screener_long_term_emas', [30, 35, 40, 45, 50, 60])
            }
        }
        self.guppy_screener = GuppyScreener(guppy_config)

        logger.info(f"GUPPY screener streaming processor initialized, output dir: {self.guppy_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "guppy_screener"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.guppy_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        This method won't be used directly since GUPPY screener processes batches.
        Keeping for interface compatibility.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str, ticker_choice: int = 0) -> Dict[str, Any]:
        """
        Process GUPPY screener batch using memory-efficient streaming pattern.
        Following PVB TW streaming pattern with immediate writes and memory cleanup.

        Args:
            batch_data: Dictionary of ticker -> DataFrame
            timeframe: Processing timeframe
            ticker_choice: User ticker choice

        Returns:
            Dictionary with processing results and output files
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} GUPPY streaming")
            return {}

        logger.debug(f"Processing GUPPY batch for {timeframe}: {len(batch_data)} tickers")

        # Get GUPPY parameters for this timeframe (includes hierarchical flag checking)
        try:
            guppy_params = get_guppy_screener_params_for_timeframe(self.user_config, timeframe)
            if not guppy_params or not guppy_params.get('enable_guppy_screener'):
                logger.debug(f"GUPPY screener disabled for {timeframe}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get GUPPY parameters for {timeframe}: {e}")
            return {}

        current_date = datetime.now().strftime('%Y%m%d')
        all_results = []
        component_results = {
            'bullish_alignment': [],
            'bearish_alignment': [],
            'compression_breakout': [],
            'expansion_signal': [],
            'bullish_crossover': [],
            'bearish_crossover': []
        }

        processed_tickers = 0
        total_errors = 0

        try:
            # Run GUPPY screening for entire batch
            batch_results = self.guppy_screener.run_guppy_screening(batch_data)

            if batch_results:
                all_results.extend(batch_results)

                # Sort results by signal type for individual files
                for result in batch_results:
                    signal_type = result.get('signal_type', 'unknown')
                    if signal_type in component_results:
                        component_results[signal_type].append(result)

                processed_tickers = len(batch_results)

            # Memory cleanup after batch processing
            gc.collect()

            # Write consolidated results immediately
            output_files = []
            if all_results:
                consolidated_filename = f"guppy_screener_{timeframe}_{current_date}.csv"
                consolidated_file = self.guppy_dir / consolidated_filename
                self._write_results_to_csv(consolidated_file, all_results)
                output_files.append(str(consolidated_file))
                logger.info(f"GUPPY consolidated: {len(all_results)} results saved to {consolidated_file}")

            # Write individual component files if enabled
            if guppy_params['guppy_screener'].get('save_individual_files', True):
                for component_name, component_data in component_results.items():
                    if component_data:
                        component_filename = f"guppy_{component_name}_{timeframe}_{current_date}.csv"
                        component_file = self.guppy_dir / component_filename
                        self._write_results_to_csv(component_file, component_data)
                        output_files.append(str(component_file))

            # Memory cleanup
            self.cleanup_memory(all_results, component_results, batch_data)

        except Exception as e:
            logger.error(f"Error in GUPPY batch processing: {e}")
            total_errors += 1

        logger.info(f"GUPPY batch summary ({timeframe}): {processed_tickers} tickers processed, "
                   f"{len(all_results)} total results, {total_errors} errors")

        return {
            'output_files': output_files,
            'tickers_processed': len(all_results),
            'processed_tickers': processed_tickers,
            'errors': total_errors,
            'timeframe': timeframe
        }

    def _write_results_to_csv(self, output_file: Path, results: List[Dict]):
        """Write GUPPY results to CSV with memory optimization."""
        if not results:
            return

        try:
            # Convert to DataFrame with optimized dtypes
            df = pd.DataFrame(results)
            df = self.optimize_dataframe_dtypes(df)

            # Write to CSV
            df.to_csv(output_file, index=False)

            # Memory cleanup
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"Error writing GUPPY results to {output_file}: {e}")


def run_all_guppy_screener_streaming(config, user_config, timeframes: List[str], clean_file_path: str) -> Dict[str, int]:
    """
    Run GUPPY GMMA screener using streaming processing with hierarchical flag validation.
    Following PVB TW pattern with DataReader for data loading.

    Args:
        config: Configuration object
        user_config: User configuration with GUPPY screener settings
        timeframes: List of timeframes to process
        clean_file_path: Path to clean tickers file

    Returns:
        Dictionary with processing results per timeframe
    """
    # Check master flag first
    if not getattr(user_config, "guppy_screener_enable", False):
        print(f"\n⏭️  GUPPY screener disabled - skipping GUPPY processing")
        logger.info("GUPPY screener disabled (master flag)")
        return {}

    # Check if any timeframe is enabled
    enabled_timeframes = []
    for timeframe in timeframes:
        if getattr(user_config, f"guppy_screener_{timeframe}_enable", True):
            enabled_timeframes.append(timeframe)

    if not enabled_timeframes:
        print(f"\n⚠️  GUPPY master enabled but all timeframes disabled - skipping GUPPY processing")
        logger.warning("GUPPY master enabled but all timeframes disabled - skipping")
        return {}

    print(f"\n🔍 GUPPY SCREENER - Processing timeframes: {', '.join(enabled_timeframes)}")
    logger.info(f"GUPPY screener enabled for: {', '.join(enabled_timeframes)}")

    # Initialize processor
    processor = GuppyScreenerStreamingProcessor(config, user_config)
    results = {}

    # Process each enabled timeframe using PVB TW pattern
    for timeframe in enabled_timeframes:
        guppy_enabled = getattr(user_config, f'guppy_screener_{timeframe}_enable', True)
        if not guppy_enabled:
            print(f"⏭️  GUPPY screener disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing GUPPY {timeframe.upper()} timeframe...")
        logger.info(f"Starting GUPPY screener for {timeframe} timeframe...")

        # Initialize DataReader for this timeframe (following PVB TW pattern)
        batch_size = getattr(user_config, 'batch_size', 100)
        from src.data_reader import DataReader
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers from file
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        import math
        total_batches = math.ceil(total_tickers / batch_size)

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        total_results = 0
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]

            logger.info(f"Processing GUPPY batch {batch_num + 1}/{total_batches}: tickers {start_idx + 1}-{end_idx}")

            # Get batch data using DataReader
            batch_data = data_reader.read_batch_data(batch_tickers)

            if batch_data:
                # Process batch using GUPPY screener
                batch_result = processor.process_batch_streaming(batch_data, timeframe)
                if batch_result and "tickers_processed" in batch_result:
                    total_results += batch_result["tickers_processed"]
                    logger.info(f"GUPPY batch {batch_num + 1} completed: {batch_result['tickers_processed']} results")

        results[timeframe] = total_results
        logger.info(f"GUPPY screener completed for {timeframe}: {total_results} total results processed")

    if results:
        print(f"✅ GUPPY screener processing completed")
    else:
        print(f"⚠️  GUPPY screener completed with no results")

    return results

