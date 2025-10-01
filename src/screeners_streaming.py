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
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from src.streaming_base import StreamingCalculationBase
from src.screeners.pvb_screener import pvb_screener
from src.screeners.atr1_screener import atr1_screener
from src.screeners.drwish_screener import drwish_screener
from src.screeners.guppy_screener import GuppyScreener
from src.screeners.rti_screener import RTIScreener
from src.user_defined_data import get_pvb_params_for_timeframe, get_atr1_params_for_timeframe, get_drwish_params_for_timeframe, get_volume_suite_params_for_timeframe, get_guppy_screener_params_for_timeframe, get_rti_screener_params_for_timeframe

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

            # Add ticker_info for exchange enrichment (if available)
            pvb_params['ticker_info'] = getattr(self, 'ticker_info', None)

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
                        'exchange': result.get('exchange', 'N/A'),  # Add exchange field
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
        """Write GUPPY results to CSV with memory optimization and append mode for batches"""
        if not results:
            return

        try:
            # Convert to DataFrame with optimized dtypes
            df = pd.DataFrame(results)
            df = self.optimize_dataframe_dtypes(df)

            # Write to CSV (append if file exists for streaming batches)
            if output_file.exists():
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
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


class GoldLaunchPadStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for Gold Launch Pad following GUPPY pattern.
    Memory-efficient batch processing with immediate writes and component separation.
    """

    def __init__(self, config, user_config):
        """
        Initialize Gold Launch Pad streaming processor.
        """
        super().__init__(config, user_config)

        # Create Gold Launch Pad output directory
        self.glp_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'gold_launch_pad'
        self.glp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Gold Launch Pad screener instance with configuration
        glp_config = {
            'timeframe': 'daily',  # Will be overridden per timeframe
            'gold_launch_pad': {
                'ma_periods': getattr(user_config, 'gold_launch_pad_ma_periods', [10, 20, 50]),
                'ma_type': getattr(user_config, 'gold_launch_pad_ma_type', 'EMA'),
                'zscore_window': getattr(user_config, 'gold_launch_pad_zscore_window', 50),
                'max_spread_threshold': getattr(user_config, 'gold_launch_pad_max_spread_threshold', 1.0),
                'slope_lookback_pct': getattr(user_config, 'gold_launch_pad_slope_lookback_pct', 0.3),
                'min_slope_threshold': getattr(user_config, 'gold_launch_pad_min_slope_threshold', 0.0001),
                'price_proximity_stdv': getattr(user_config, 'gold_launch_pad_price_proximity_stdv', 2.0),
                'proximity_window': getattr(user_config, 'gold_launch_pad_proximity_window', 20),
                'min_price': getattr(user_config, 'gold_launch_pad_min_price', 5.0),
                'min_volume': getattr(user_config, 'gold_launch_pad_min_volume', 100000),
                'save_individual_files': getattr(user_config, 'gold_launch_pad_save_individual_files', True)
            }
        }
        from src.screeners.gold_launch_pad import GoldLaunchPadScreener
        self.glp_screener = GoldLaunchPadScreener(glp_config)

        logger.info(f"Gold Launch Pad streaming processor initialized, output dir: {self.glp_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "gold_launch_pad"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.glp_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        This method won't be used directly since Gold Launch Pad screener processes batches.
        Keeping for interface compatibility with StreamingCalculationBase abstract class.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str) -> Dict[str, Any]:
        """
        Process Gold Launch Pad batch using memory-efficient streaming pattern.
        Following GUPPY pattern with component separation and immediate writes.
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} Gold Launch Pad streaming")
            return {}

        logger.debug(f"Processing Gold Launch Pad batch for {timeframe}: {len(batch_data)} tickers")

        # Get Gold Launch Pad parameters for this timeframe
        try:
            from src.user_defined_data import get_gold_launch_pad_params_for_timeframe
            glp_params = get_gold_launch_pad_params_for_timeframe(self.user_config, timeframe)
            if not glp_params or not glp_params.get('enable_gold_launch_pad'):
                logger.debug(f"Gold Launch Pad disabled for {timeframe}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get Gold Launch Pad parameters for {timeframe}: {e}")
            return {}

        # Initialize result containers (component separation like GUPPY)
        all_results = []
        component_results = {
            'tight_grouping': [],
            'bullish_stacking': [],
            'positive_slope': [],
            'price_above_ma': [],
            'price_proximity': []
        }
        current_date = self.extract_date_from_batch_data(batch_data)
        processed_tickers = 0

        try:
            # Run Gold Launch Pad screening for entire batch
            batch_results = self.glp_screener.run_gold_launch_pad_screening(batch_data)

            if batch_results:
                all_results.extend(batch_results)

                # Sort results by signal type for individual files (GUPPY pattern)
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
                consolidated_filename = f"gold_launch_pad_{timeframe}_{current_date}.csv"
                consolidated_file = self.glp_dir / consolidated_filename
                self._write_results_to_csv(consolidated_file, all_results)
                output_files.append(str(consolidated_file))
                logger.info(f"Gold Launch Pad consolidated: {len(all_results)} results saved to {consolidated_file}")

            # Write individual component files if enabled (GUPPY pattern)
            if glp_params['gold_launch_pad'].get('save_individual_files', True):
                for component_name, component_data in component_results.items():
                    if component_data:
                        component_filename = f"gold_launch_pad_{component_name}_{timeframe}_{current_date}.csv"
                        component_file = self.glp_dir / component_filename
                        self._write_results_to_csv(component_file, component_data)
                        output_files.append(str(component_file))

            # Memory cleanup
            self.cleanup_memory(all_results, component_results, batch_data)

        except Exception as e:
            logger.error(f"Error in Gold Launch Pad batch processing: {e}")

        logger.info(f"Gold Launch Pad batch summary ({timeframe}): {processed_tickers} tickers processed, "
                   f"{len(all_results)} total results")

        return {
            "tickers_processed": processed_tickers,
            "total_results": len(all_results),
            "output_files": output_files
        }

    def _write_results_to_csv(self, output_file: Path, results: List[Dict]):
        """Write Gold Launch Pad results to CSV with memory optimization and append mode for batches"""
        if not results:
            return

        try:
            # Convert to DataFrame with optimized dtypes
            df = pd.DataFrame(results)
            df = self.optimize_dataframe_dtypes(df)

            # Write to CSV (append if file exists for streaming batches)
            if output_file.exists():
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)

            # Memory cleanup
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"Error writing Gold Launch Pad results to {output_file}: {e}")


def run_all_gold_launch_pad_streaming(config, user_config, timeframes: List[str], clean_file_path: str) -> Dict[str, int]:
    """
    Run Gold Launch Pad screener using streaming processing with hierarchical flag validation.
    Following GUPPY/PVB TW pattern with DataReader for data loading.
    """
    # Check master flag first
    if not getattr(user_config, "gold_launch_pad_enable", False):
        print(f"\n⏭️  Gold Launch Pad screener disabled - skipping processing")
        logger.info("Gold Launch Pad screener disabled (master flag)")
        return {}

    # Check if any timeframe is enabled
    enabled_timeframes = []
    for timeframe in timeframes:
        if getattr(user_config, f"gold_launch_pad_{timeframe}_enable", True):
            enabled_timeframes.append(timeframe)

    if not enabled_timeframes:
        print(f"\n⚠️  Gold Launch Pad master enabled but all timeframes disabled - skipping processing")
        logger.warning("Gold Launch Pad master enabled but all timeframes disabled")
        return {}

    print(f"\n🚀 GOLD LAUNCH PAD SCREENER - Processing timeframes: {', '.join(enabled_timeframes)}")
    logger.info(f"Gold Launch Pad screener enabled for: {', '.join(enabled_timeframes)}")

    # Initialize processor
    processor = GoldLaunchPadStreamingProcessor(config, user_config)
    results = {}

    # Process each enabled timeframe using GUPPY/PVB TW pattern
    for timeframe in enabled_timeframes:
        glp_enabled = getattr(user_config, f'gold_launch_pad_{timeframe}_enable', True)
        if not glp_enabled:
            print(f"⏭️  Gold Launch Pad disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing Gold Launch Pad {timeframe.upper()} timeframe...")
        logger.info(f"Starting Gold Launch Pad for {timeframe} timeframe...")

        # Initialize DataReader for this timeframe (following GUPPY/PVB TW pattern)
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

            logger.info(f"Processing Gold Launch Pad batch {batch_num + 1}/{total_batches}: tickers {start_idx + 1}-{end_idx}")

            # Get batch data using DataReader
            batch_data = data_reader.read_batch_data(batch_tickers)

            if batch_data:
                # Process batch using Gold Launch Pad screener
                batch_result = processor.process_batch_streaming(batch_data, timeframe)
                if batch_result and "tickers_processed" in batch_result:
                    total_results += batch_result["tickers_processed"]
                    logger.info(f"Gold Launch Pad batch {batch_num + 1} completed: {batch_result['tickers_processed']} results")

        results[timeframe] = total_results
        logger.info(f"Gold Launch Pad screener completed for {timeframe}: {total_results} total results processed")

    if results:
        print(f"✅ Gold Launch Pad screener processing completed")
    else:
        print(f"⚠️  Gold Launch Pad screener completed with no results")

    return results

# ====================================================================
# RTI SCREENER STREAMING PROCESSOR
# ====================================================================

class RTIStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for RTI (Range Tightening Indicator) screener.
    Identifies low-volatility consolidations preceding breakouts.

    Signal Types:
    - Zone1: Extremely tight volatility (0-5%)
    - Zone2: Low volatility (5-10%)
    - Zone3: Moderate low volatility (10-15%)
    - Orange_Dot: Extended consolidation signal
    - Range_Expansion: Breakout from consolidation
    """

    def __init__(self, config, user_config):
        """Initialize RTI streaming processor"""
        super().__init__(config, user_config)

        # Create output directory
        self.rti_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'rti'
        self.rti_dir.mkdir(parents=True, exist_ok=True)

        # Initialize RTI screener with configuration
        rti_config = {
            'timeframe': 'daily',  # Will be overridden per timeframe
            'rti_screener': {
                'rti_period': getattr(user_config, 'rti_period', 50),
                'rti_short_period': getattr(user_config, 'rti_short_period', 5),
                'rti_swing_period': getattr(user_config, 'rti_swing_period', 15),
                'zone1_threshold': getattr(user_config, 'rti_zone1_threshold', 5.0),
                'zone2_threshold': getattr(user_config, 'rti_zone2_threshold', 10.0),
                'zone3_threshold': getattr(user_config, 'rti_zone3_threshold', 15.0),
                'low_volatility_threshold': getattr(user_config, 'rti_low_volatility_threshold', 20.0),
                'expansion_multiplier': getattr(user_config, 'rti_expansion_multiplier', 2.0),
                'consecutive_low_vol_bars': getattr(user_config, 'rti_consecutive_low_vol_bars', 2),
                'min_consolidation_period': getattr(user_config, 'rti_min_consolidation_period', 3),
                'breakout_confirmation_period': getattr(user_config, 'rti_breakout_confirmation_period', 2),
                'min_price': getattr(user_config, 'rti_min_price', 5.0),
                'min_volume': getattr(user_config, 'rti_min_volume', 100000),
                'save_individual_files': getattr(user_config, 'rti_save_individual_files', True),
            }
        }
        self.rti_screener = RTIScreener(rti_config)

        logger.info(f"RTI streaming processor initialized, output dir: {self.rti_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming"""
        return "rti"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation"""
        return self.rti_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        Required by StreamingCalculationBase abstract class.
        Not used since RTI processes batches.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str, ticker_info: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Process RTI batch using memory-efficient streaming pattern.
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} RTI screening")
            return {}

        logger.debug(f"Processing RTI batch for {timeframe}: {len(batch_data)} tickers")

        # Get RTI parameters for this timeframe
        try:
            rti_params = get_rti_screener_params_for_timeframe(self.user_config, timeframe)
            if not rti_params or not rti_params.get('enable_rti'):
                logger.debug(f"RTI disabled for {timeframe}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get RTI parameters for {timeframe}: {e}")
            return {}

        # Initialize result containers
        all_results = []
        component_results = {
            'zone1': [],
            'zone2': [],
            'zone3': [],
            'orange_dot': [],
            'range_expansion': []
        }
        current_date = self.extract_date_from_batch_data(batch_data)
        processed_tickers = 0

        try:
            # Update screener configuration for this timeframe
            self.rti_screener.config['timeframe'] = timeframe
            self.rti_screener.timeframe = timeframe

            # Run RTI screening for entire batch
            batch_results = self.rti_screener.run_rti_screening(
                batch_data,
                ticker_info=ticker_info,
                batch_info={'timeframe': timeframe}
            )

            if batch_results:
                all_results.extend(batch_results)

                # Sort results by signal type for individual files
                for result in batch_results:
                    signal_type = result.get('signal_type', '').lower().replace(' ', '_')
                    if signal_type in component_results:
                        component_results[signal_type].append(result)

                processed_tickers = len(set(r['ticker'] for r in batch_results))

            # Memory cleanup after batch processing
            gc.collect()

            # Write consolidated results immediately
            output_files = []
            if all_results:
                consolidated_filename = f"rti_consolidated_{timeframe}_{current_date}.csv"
                consolidated_file = self.rti_dir / consolidated_filename
                self._write_results_to_csv(consolidated_file, all_results)
                output_files.append(str(consolidated_file))
                logger.info(f"RTI consolidated: {len(all_results)} results saved to {consolidated_file}")

            # Write individual component files if enabled
            if self.rti_screener.save_individual_files:
                for component_name, component_data in component_results.items():
                    if component_data:
                        component_filename = f"rti_{component_name}_{timeframe}_{current_date}.csv"
                        component_file = self.rti_dir / component_filename
                        self._write_results_to_csv(component_file, component_data)
                        output_files.append(str(component_file))
                        logger.info(f"RTI {component_name}: {len(component_data)} results")

            # Memory cleanup
            self.cleanup_memory(all_results, component_results, batch_data)

        except Exception as e:
            logger.error(f"Error in RTI batch processing: {e}")

        logger.info(f"RTI batch summary ({timeframe}): {processed_tickers} tickers, "
                   f"{len(all_results)} signals")

        return {
            "tickers_processed": processed_tickers,
            "total_signals": len(all_results),
            "output_files": output_files
        }

    def _write_results_to_csv(self, output_file: Path, results: List[Dict]):
        """Write RTI results to CSV with memory optimization and append mode for batches"""
        if not results:
            return

        try:
            # Convert to DataFrame with optimized dtypes
            df = pd.DataFrame(results)
            df = self.optimize_dataframe_dtypes(df)

            # Write to CSV (append if file exists for streaming batches)
            if output_file.exists():
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)

            # Memory cleanup
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"Error writing RTI results to {output_file}: {e}")

def run_all_rti_streaming(config, user_config, timeframes: List[str], clean_file_path: str) -> Dict[str, int]:
    """
    Run RTI screener using streaming processing with hierarchical flag validation.

    Args:
        config: System configuration
        user_config: User configuration
        timeframes: List of timeframes to process
        clean_file_path: Path to ticker list file

    Returns:
        Dictionary with timeframe results
    """
    # Check master flag first
    if not getattr(user_config, "rti_enable", False):
        print(f"\n⏭️  RTI Screener disabled - skipping processing")
        logger.info("RTI Screener disabled (master flag)")
        return {}

    # Check if any timeframe is enabled
    enabled_timeframes = []
    for timeframe in timeframes:
        if getattr(user_config, f"rti_{timeframe}_enable", False):
            enabled_timeframes.append(timeframe)

    if not enabled_timeframes:
        print(f"\n⚠️  RTI master enabled but all timeframes disabled - skipping processing")
        logger.warning("RTI master enabled but all timeframes disabled")
        return {}

    print(f"\n🔍 RTI SCREENER - Processing timeframes: {', '.join(enabled_timeframes)}")
    logger.info(f"RTI enabled for: {', '.join(enabled_timeframes)}")

    # Initialize processor
    processor = RTIStreamingProcessor(config, user_config)
    results = {}

    # Process each enabled timeframe
    for timeframe in enabled_timeframes:
        rti_enabled = getattr(user_config, f'rti_{timeframe}_enable', False)
        if not rti_enabled:
            print(f"⏭️  RTI disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing RTI {timeframe.upper()} timeframe...")
        logger.info(f"Starting RTI for {timeframe} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        from src.data_reader import DataReader
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers from file
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Load ticker_info for exchange data (optional but recommended)
        ticker_universe_all_path = config.base_dir / 'results' / 'ticker_universes' / 'ticker_universe_all.csv'
        ticker_info = None
        if ticker_universe_all_path.exists():
            try:
                ticker_info = pd.read_csv(ticker_universe_all_path, usecols=['ticker', 'exchange'])
                logger.info(f"Loaded exchange data for {len(ticker_info)} tickers")
            except Exception as e:
                logger.warning(f"Could not load ticker_universe_all.csv: {e}")

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        import math
        total_batches = math.ceil(total_tickers / batch_size)

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        total_signals = 0
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]

            print(f"🔄 Loading batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers) - {((batch_num+1)/total_batches)*100:.1f}%")

            # Get batch data using DataReader
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if batch_data:
                print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_num + 1}")

                # Process batch using RTI screener
                batch_result = processor.process_batch_streaming(batch_data, timeframe, ticker_info)
                if batch_result and "total_signals" in batch_result:
                    total_signals += batch_result["total_signals"]
                    logger.info(f"RTI batch {batch_num + 1} completed: {batch_result['total_signals']} signals")
            else:
                print(f"⚠️  No valid data in batch {batch_num + 1}")

        results[timeframe] = total_signals
        print(f"✅ RTI completed for {timeframe}: {total_signals} signals")
        logger.info(f"RTI completed for {timeframe}: {total_signals} signals")

    if results:
        print(f"✅ RTI SCREENER COMPLETED!")
        print(f"📊 Total signals: {sum(results.values())}")
        print(f"🕒 Timeframes processed: {', '.join(results.keys())}")
    else:
        print(f"⚠️  RTI completed with no results")

    return results








# ====================================================================
# ADL ENHANCED SCREENER STREAMING PROCESSOR
# ====================================================================

class ADLEnhancedStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for ADL Enhanced screener.
    Memory-efficient batch processing with immediate writes.
    
    Integrates all 5 analysis modules:
    - Basic ADL calculation (existing)
    - Month-over-month accumulation
    - Short-term momentum
    - Moving average analysis
    - Composite scoring
    """

    def __init__(self, config, user_config):
        """
        Initialize ADL Enhanced screener streaming processor.
        
        Args:
            config: Configuration object with directories
            user_config: User configuration with ADL screener settings
        """
        super().__init__(config, user_config)

        # Create ADL screener output directory
        self.adl_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'adl'
        self.adl_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ADL Enhanced screener instance
        from src.screeners.ad_line.adl_screener_enhanced import create_enhanced_screener
        
        adl_config = {
            'timeframe': 'daily',  # Will be overridden per timeframe
            'adl_output_dir': str(self.adl_dir)
        }
        
        self.adl_screener = create_enhanced_screener(adl_config, user_config)

        logger.info(f"ADL Enhanced screener streaming processor initialized, output dir: {self.adl_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "adl_enhanced"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.adl_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        This method won't be used directly since ADL screener processes batches.
        Keeping for interface compatibility.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str, ticker_choice: int = 0) -> Dict[str, Any]:
        """
        Process ADL Enhanced screener batch using memory-efficient streaming pattern.
        Following GUPPY/PVB TW streaming pattern with immediate writes and memory cleanup.
        
        Args:
            batch_data: Dictionary of ticker -> DataFrame
            timeframe: Processing timeframe
            ticker_choice: User ticker choice
            
        Returns:
            Dictionary with processing results and output files
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} ADL Enhanced streaming")
            return {}

        logger.debug(f"Processing ADL Enhanced batch for {timeframe}: {len(batch_data)} tickers")

        current_date = datetime.now().strftime('%Y%m%d')
        output_files = []

        try:
            # Update screener timeframe
            self.adl_screener.timeframe = timeframe

            # Run ADL Enhanced screening for entire batch
            all_results = self.adl_screener.run_enhanced_screening(batch_data)

            if not all_results:
                logger.debug(f"No ADL results for {timeframe} batch")
                return {}

            # Write composite results (main output)
            composite_results = all_results.get('composite_results', [])
            if composite_results:
                composite_file = self.adl_dir / f"adl_composite_ranked_{timeframe}_{current_date}.csv"
                self._write_results_to_csv(composite_file, composite_results)
                output_files.append(str(composite_file))
                logger.info(f"ADL composite: {len(composite_results)} results saved to {composite_file}")

            # Write top candidates file
            top_candidates = all_results.get('top_candidates', [])
            if top_candidates:
                top_file = self.adl_dir / f"adl_top_candidates_{timeframe}_{current_date}.csv"
                self._write_results_to_csv(top_file, top_candidates)
                output_files.append(str(top_file))
                logger.info(f"ADL top candidates: {len(top_candidates)} saved to {top_file}")

            # Write individual component files if enabled
            if self.user_config.adl_screener_output_separate_signals:
                # MoM results
                mom_results = all_results.get('mom_results', [])
                if mom_results:
                    mom_file = self.adl_dir / f"adl_mom_accumulation_{timeframe}_{current_date}.csv"
                    self._write_results_to_csv(mom_file, mom_results)
                    output_files.append(str(mom_file))

                # Short-term momentum results
                short_term_results = all_results.get('short_term_results', [])
                if short_term_results:
                    st_file = self.adl_dir / f"adl_short_term_momentum_{timeframe}_{current_date}.csv"
                    self._write_results_to_csv(st_file, short_term_results)
                    output_files.append(str(st_file))

                # MA alignment results
                ma_results = all_results.get('ma_results', [])
                if ma_results:
                    ma_file = self.adl_dir / f"adl_ma_alignment_{timeframe}_{current_date}.csv"
                    self._write_results_to_csv(ma_file, ma_results)
                    output_files.append(str(ma_file))

                # Divergence results (existing functionality)
                divergence_results = all_results.get('divergence_results', [])
                if divergence_results:
                    div_file = self.adl_dir / f"adl_divergence_{timeframe}_{current_date}.csv"
                    self._write_results_to_csv(div_file, divergence_results)
                    output_files.append(str(div_file))

                # Breakout results (existing functionality)
                breakout_results = all_results.get('breakout_results', [])
                if breakout_results:
                    bo_file = self.adl_dir / f"adl_breakout_{timeframe}_{current_date}.csv"
                    self._write_results_to_csv(bo_file, breakout_results)
                    output_files.append(str(bo_file))

            # Write summary statistics if enabled
            if self.user_config.adl_screener_output_summary_stats:
                summary = all_results.get('summary', {})
                if summary:
                    summary_file = self.adl_dir / f"adl_summary_{timeframe}_{current_date}.txt"
                    self._write_summary_to_file(summary_file, summary)
                    output_files.append(str(summary_file))

            # Memory cleanup
            self.cleanup_memory(all_results, batch_data)
            gc.collect()

            return {
                "tickers_processed": len(batch_data),
                "composite_count": len(composite_results),
                "top_candidates_count": len(top_candidates),
                "output_files": output_files
            }

        except Exception as e:
            logger.error(f"Error in ADL Enhanced batch processing: {e}")
            return {}

    def _write_results_to_csv(self, output_file: Path, results: List[Dict]):
        """Write ADL results to CSV with memory optimization and append mode for batches"""
        if not results:
            return

        try:
            # Convert to DataFrame with optimized dtypes
            df = pd.DataFrame(results)
            df = self.optimize_dataframe_dtypes(df)

            # Write to CSV (append if file exists for streaming batches)
            if output_file.exists():
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)

            # Memory cleanup
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"Error writing ADL results to {output_file}: {e}")

    def _write_summary_to_file(self, output_file: Path, summary: Dict):
        """Write summary statistics to text file."""
        try:
            with open(output_file, 'w') as f:
                f.write("ADL Enhanced Screener Summary\n")
                f.write("=" * 50 + "\n\n")
                
                for key, value in summary.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                f.write("\n" + "=" * 50 + "\n")
        except Exception as e:
            logger.error(f"Error writing summary to {output_file}: {e}")


class StockbeeStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for Stockbee Suite screener.

    Implements Pradeep Bonde's 4 momentum screening strategies:
    1. 9M MOVERS - High volume institutional activity (100% functional)
    2. 20% Weekly Movers - Strong weekly momentum (100% functional)
    3. 4% Daily Gainers - Daily momentum leaders (100% functional)
    4. Industry Leaders - Sector rotation analysis (~60% functional without RS data)

    PHASE A IMPLEMENTATION:
    - 3 of 4 screeners fully functional (no external RS data needed)
    - Industry Leaders works with fallback logic (limited without RS)
    - Clear placeholders for future RS enhancement (Phase B)
    """

    def __init__(self, config, user_config):
        """Initialize Stockbee streaming processor"""
        super().__init__(config, user_config)

        # Create output directory
        self.stockbee_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'stockbee'
        self.stockbee_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Stockbee screener with configuration
        stockbee_config = {
            'timeframe': 'daily',  # Will be overridden per timeframe
            'enable_stockbee_suite': True,
            'stockbee_suite': {
                # Component enables
                'enable_9m_movers': getattr(user_config, 'stockbee_suite_9m_movers', True),
                'enable_weekly_movers': getattr(user_config, 'stockbee_suite_weekly_movers', True),
                'enable_daily_gainers': getattr(user_config, 'stockbee_suite_daily_gainers', True),
                'enable_industry_leaders': getattr(user_config, 'stockbee_suite_industry_leaders', True),

                # 9M Movers parameters
                '9m_volume_threshold': getattr(user_config, 'stockbee_suite_9m_volume_threshold', 9_000_000),
                '9m_rel_vol_threshold': getattr(user_config, 'stockbee_suite_9m_rel_vol_threshold', 1.25),

                # Weekly Movers parameters
                'weekly_gain_threshold': getattr(user_config, 'stockbee_suite_weekly_gain_threshold', 20.0),
                'weekly_rel_vol_threshold': getattr(user_config, 'stockbee_suite_weekly_rel_vol_threshold', 1.25),
                'weekly_min_avg_volume': getattr(user_config, 'stockbee_suite_weekly_min_avg_volume', 100_000),

                # Daily Gainers parameters
                'daily_gain_threshold': getattr(user_config, 'stockbee_suite_daily_gain_threshold', 4.0),
                'daily_rel_vol_threshold': getattr(user_config, 'stockbee_suite_daily_rel_vol_threshold', 1.5),
                'daily_min_volume': getattr(user_config, 'stockbee_suite_daily_min_volume', 100_000),

                # Industry Leaders parameters
                'industry_top_pct': getattr(user_config, 'stockbee_suite_industry_top_pct', 20.0),
                'industry_top_stocks': getattr(user_config, 'stockbee_suite_industry_top_stocks', 4),
                'industry_min_size': getattr(user_config, 'stockbee_suite_industry_min_size', 3),

                # General filters
                'min_market_cap': getattr(user_config, 'stockbee_suite_min_market_cap', 1_000_000_000),
                'min_price': getattr(user_config, 'stockbee_suite_min_price', 5.0),
                'exclude_funds': getattr(user_config, 'stockbee_suite_exclude_funds', True),
                'save_individual_files': getattr(user_config, 'stockbee_suite_save_individual_files', True),
            },
            'stockbee_output_dir': str(self.stockbee_dir)
        }

        from src.screeners.stockbee.stockbee_screener import StockbeeScreener
        self.stockbee_screener = StockbeeScreener(stockbee_config)

        logger.info(f"Stockbee streaming processor initialized, output dir: {self.stockbee_dir}")
        logger.info("⚠️ PHASE A: Running with limited RS data (Industry Leaders at ~60% accuracy)")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming"""
        return "stockbee"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation"""
        return self.stockbee_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        Required by StreamingCalculationBase abstract class.
        Not used since Stockbee processes batches.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str,
                              ticker_info: Optional[pd.DataFrame] = None,
                              rs_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process Stockbee batch using memory-efficient streaming pattern.

        PHASE A Implementation:
        - 9M Movers: 100% functional (no RS needed)
        - Weekly Movers: 100% functional (no RS needed)
        - Daily Gainers: 100% functional (no RS needed)
        - Industry Leaders: ~60% functional (works without RS, enhanced with RS in Phase B)

        Args:
            batch_data: Dictionary of ticker -> DataFrame
            timeframe: Processing timeframe
            ticker_info: DataFrame with market cap, exchange, industry (REQUIRED)
            rs_data: Optional RS data for enhanced Industry Leaders (PHASE B)

        Returns:
            Dictionary with processing results
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} Stockbee streaming")
            return {}

        logger.debug(f"Processing Stockbee batch for {timeframe}: {len(batch_data)} tickers")

        # Get Stockbee parameters for this timeframe
        try:
            from src.user_defined_data import get_stockbee_suite_params_for_timeframe
            stockbee_params = get_stockbee_suite_params_for_timeframe(self.user_config, timeframe)
            if not stockbee_params or not stockbee_params.get('enable_stockbee_suite'):
                logger.debug(f"Stockbee disabled for {timeframe}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get Stockbee parameters for {timeframe}: {e}")
            return {}

        # Initialize result containers
        all_results = []
        component_results = {
            '9m_movers': [],
            'weekly_movers': [],
            'daily_gainers': [],
            'industry_leaders': []
        }
        current_date = self.extract_date_from_batch_data(batch_data)
        processed_tickers = 0

        try:
            # Update screener configuration for this timeframe
            self.stockbee_screener.config['timeframe'] = timeframe
            self.stockbee_screener.timeframe = timeframe

            # Warn if data not provided
            if ticker_info is None:
                logger.error(f"❌ No ticker_info provided - market cap and industry filters DISABLED")
            if rs_data is None:
                logger.warning(f"⚠️ No RS data provided - Industry Leaders will use fallback logic (~60% accuracy)")

            # TODO PHASE B: Load RS data for enhanced Industry Leaders accuracy
            # rs_data = self._load_rs_data(timeframe, current_date)
            # When RS data available:
            # - Industry Leaders will rank industries by composite RS
            # - Stock rankings within industries will use RS scores
            # - Expected improvement: 60% → 100% accuracy

            # Run Stockbee screening for entire batch
            batch_results = self.stockbee_screener.run_stockbee_screening(
                batch_data,
                ticker_info=ticker_info,
                rs_data=rs_data,  # None in Phase A
                batch_info={'timeframe': timeframe}
            )

            if batch_results:
                all_results.extend(batch_results)

                # Sort results by screener type for individual files
                for result in batch_results:
                    screen_type = result.get('screen_type', 'unknown')

                    # Map screen types to component names
                    if '9m_movers' in screen_type or '9m' in screen_type.lower():
                        component_results['9m_movers'].append(result)
                    elif 'weekly' in screen_type.lower():
                        component_results['weekly_movers'].append(result)
                    elif 'daily' in screen_type.lower():
                        component_results['daily_gainers'].append(result)
                    elif 'industry' in screen_type.lower():
                        component_results['industry_leaders'].append(result)

                processed_tickers = len(set(r['ticker'] for r in batch_results))

            # Memory cleanup after batch processing
            gc.collect()

            # Write consolidated results immediately (with append mode!)
            output_files = []
            if all_results:
                consolidated_filename = f"stockbee_consolidated_{timeframe}_{current_date}.csv"
                consolidated_file = self.stockbee_dir / consolidated_filename
                self._write_results_to_csv(consolidated_file, all_results)
                output_files.append(str(consolidated_file))
                logger.info(f"Stockbee consolidated: {len(all_results)} results saved to {consolidated_file}")

            # Write individual component files if enabled
            if stockbee_params['stockbee_suite'].get('save_individual_files', True):
                for component_name, component_data in component_results.items():
                    if component_data:
                        component_filename = f"stockbee_{component_name}_{timeframe}_{current_date}.csv"
                        component_file = self.stockbee_dir / component_filename
                        self._write_results_to_csv(component_file, component_data)
                        output_files.append(str(component_file))
                        logger.info(f"Stockbee {component_name}: {len(component_data)} results")

            # Memory cleanup
            self.cleanup_memory(all_results, component_results, batch_data)

        except Exception as e:
            logger.error(f"Error in Stockbee batch processing: {e}")

        logger.info(f"Stockbee batch summary ({timeframe}): {processed_tickers} tickers processed, "
                   f"{len(all_results)} total signals")

        return {
            "tickers_processed": processed_tickers,
            "total_signals": len(all_results),
            "output_files": output_files,
            "component_counts": {k: len(v) for k, v in component_results.items()}
        }

    def _write_results_to_csv(self, output_file: Path, results: List[Dict]):
        """Write Stockbee results to CSV with memory optimization and append mode for batches"""
        if not results:
            return

        try:
            # Convert to DataFrame with optimized dtypes
            df = pd.DataFrame(results)
            df = self.optimize_dataframe_dtypes(df)

            # CRITICAL: Write to CSV with append mode for streaming batches
            if output_file.exists():
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)

            # Memory cleanup
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"Error writing Stockbee results to {output_file}: {e}")


def run_all_stockbee_streaming(config, user_config, timeframes: List[str], clean_file_path: str) -> Dict[str, int]:
    """
    Run Stockbee Suite screener using streaming processing with hierarchical flag validation.

    PHASE A Implementation:
    - 9M Movers, Weekly Movers, Daily Gainers: 100% functional
    - Industry Leaders: ~60% functional (fallback logic without RS data)
    - RS data: NOT YET IMPLEMENTED (Industry Leaders uses simplified ranking)

    Args:
        config: System configuration
        user_config: User configuration
        timeframes: List of timeframes to process
        clean_file_path: Path to ticker list file

    Returns:
        Dictionary with timeframe results
    """
    # Check master flag first
    if not getattr(user_config, "stockbee_suite_enable", False):
        print(f"\n⏭️  Stockbee Suite Screener disabled - skipping processing")
        logger.info("Stockbee Suite Screener disabled (master flag)")
        return {}

    # Check if any timeframe is enabled
    enabled_timeframes = []
    for timeframe in timeframes:
        if getattr(user_config, f"stockbee_suite_{timeframe}_enable", False):
            enabled_timeframes.append(timeframe)

    if not enabled_timeframes:
        print(f"\n⚠️  Stockbee master enabled but all timeframes disabled - skipping processing")
        logger.warning("Stockbee master enabled but all timeframes disabled")
        return {}

    print(f"\n📊 STOCKBEE SUITE SCREENER - Processing timeframes: {', '.join(enabled_timeframes)}")
    print(f"⚠️  PHASE A: 3 of 4 screeners at 100%, Industry Leaders at ~60% (limited RS data)")
    logger.info(f"Stockbee enabled for: {', '.join(enabled_timeframes)}")

    # Initialize processor
    processor = StockbeeStreamingProcessor(config, user_config)
    results = {}

    # Load ticker_info once for all timeframes (market cap + exchange + industry)
    ticker_universe_all_path = config.base_dir / 'results' / 'ticker_universes' / 'ticker_universe_all.csv'
    ticker_info = None
    if ticker_universe_all_path.exists():
        try:
            ticker_info = pd.read_csv(ticker_universe_all_path,
                                     usecols=['ticker', 'exchange', 'market_cap', 'sector', 'industry'])
            logger.info(f"Loaded ticker info for {len(ticker_info)} tickers (including industry data)")
        except Exception as e:
            logger.error(f"Could not load ticker_universe_all.csv: {e}")
            logger.warning("⚠️ Market cap and industry filters will be DISABLED")
    else:
        logger.warning(f"⚠️ ticker_universe_all.csv not found - market cap and industry filters DISABLED")

    # PHASE A: RS data NOT YET IMPLEMENTED
    rs_data = None
    # TODO PHASE B: Load RS data for enhanced Industry Leaders
    # rs_data = processor._load_rs_data(timeframe, current_date)

    # Process each enabled timeframe
    for timeframe in enabled_timeframes:
        stockbee_enabled = getattr(user_config, f'stockbee_suite_{timeframe}_enable', False)
        if not stockbee_enabled:
            print(f"⏭️  Stockbee disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing Stockbee {timeframe.upper()} timeframe...")
        logger.info(f"Starting Stockbee for {timeframe} timeframe...")

        # Initialize DataReader for this timeframe
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

        total_signals = 0
        component_totals = {'9m_movers': 0, 'weekly_movers': 0, 'daily_gainers': 0, 'industry_leaders': 0}

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]

            print(f"🔄 Loading batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers) - {((batch_num+1)/total_batches)*100:.1f}%")

            # Get batch data using DataReader
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if batch_data:
                print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_num + 1}")

                # Process batch using Stockbee screener
                batch_result = processor.process_batch_streaming(
                    batch_data,
                    timeframe,
                    ticker_info=ticker_info,
                    rs_data=rs_data  # None in Phase A
                )

                if batch_result and "total_signals" in batch_result:
                    total_signals += batch_result["total_signals"]

                    # Accumulate component counts
                    if "component_counts" in batch_result:
                        for component, count in batch_result["component_counts"].items():
                            component_totals[component] = component_totals.get(component, 0) + count

                    logger.info(f"Stockbee batch {batch_num + 1} completed: {batch_result['total_signals']} signals")
            else:
                print(f"⚠️  No valid data in batch {batch_num + 1}")

        results[timeframe] = total_signals

        # Display component breakdown
        print(f"\n✅ Stockbee completed for {timeframe}: {total_signals} total signals")
        print(f"   📈 9M Movers: {component_totals['9m_movers']}")
        print(f"   📈 Weekly Movers: {component_totals['weekly_movers']}")
        print(f"   📈 Daily Gainers: {component_totals['daily_gainers']}")
        print(f"   📈 Industry Leaders: {component_totals['industry_leaders']} (⚠️ limited RS data)")

        logger.info(f"Stockbee completed for {timeframe}: {total_signals} signals, component breakdown: {component_totals}")

    if results:
        print(f"\n✅ STOCKBEE SUITE SCREENER COMPLETED!")
        print(f"📊 Total signals: {sum(results.values())}")
        print(f"🕒 Timeframes processed: {', '.join(results.keys())}")
        print(f"⚠️  NOTE: PHASE A results - Industry Leaders at ~60% accuracy (full accuracy requires Phase B)")
    else:
        print(f"⚠️  Stockbee completed with no results")

    return results


def run_all_adl_enhanced_streaming(config, user_config, timeframes: List[str], clean_file_path: str) -> Dict[str, int]:
    """
    Run ADL Enhanced screener using streaming processing with hierarchical flag validation.
    Following GUPPY/PVB TW pattern with DataReader for data loading.
    
    Args:
        config: Configuration object
        user_config: User configuration
        timeframes: List of timeframes to process
        clean_file_path: Path to ticker list file
        
    Returns:
        Dictionary of {timeframe: result_count}
    """
    # Check master flag first
    if not getattr(user_config, "adl_screener_enable", False):
        print(f"\n⏭️  ADL Enhanced Screener disabled - skipping processing")
        logger.info("ADL Enhanced Screener disabled (master flag)")
        return {}

    # Check if any timeframe is enabled
    enabled_timeframes = []
    for timeframe in timeframes:
        if getattr(user_config, f"adl_screener_{timeframe}_enable", False):
            enabled_timeframes.append(timeframe)

    if not enabled_timeframes:
        print(f"\n⚠️  ADL Enhanced Screener master enabled but all timeframes disabled - skipping processing")
        logger.warning("ADL Enhanced Screener master enabled but all timeframes disabled")
        return {}

    print(f"\n🔍 ADL ENHANCED SCREENER - Processing timeframes: {', '.join(enabled_timeframes)}")
    logger.info(f"ADL Enhanced Screener enabled for: {', '.join(enabled_timeframes)}")

    # Initialize processor
    processor = ADLEnhancedStreamingProcessor(config, user_config)
    results = {}

    # Process each enabled timeframe using GUPPY/PVB TW pattern
    for timeframe in enabled_timeframes:
        adl_enabled = getattr(user_config, f'adl_screener_{timeframe}_enable', False)
        if not adl_enabled:
            print(f"⏭️  ADL Enhanced Screener disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing ADL Enhanced {timeframe.upper()} timeframe...")
        logger.info(f"Starting ADL Enhanced Screener for {timeframe} timeframe...")

        # Initialize DataReader for this timeframe
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

        # Accumulators for all batches
        all_composite_results = []
        all_top_candidates = []
        all_mom_results = []
        all_short_term_results = []
        all_ma_results = []
        all_divergence_results = []
        all_breakout_results = []

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]

            logger.info(f"Processing ADL Enhanced batch {batch_num + 1}/{total_batches}: tickers {start_idx + 1}-{end_idx}")

            # Get batch data using DataReader
            batch_data = data_reader.read_batch_data(batch_tickers)

            if batch_data:
                # Process batch using ADL Enhanced screener WITHOUT writing files
                all_results = processor.adl_screener.run_enhanced_screening(batch_data)

                if all_results:
                    # Accumulate results from this batch
                    all_composite_results.extend(all_results.get('composite_results', []))
                    all_mom_results.extend(all_results.get('mom_results', []))
                    all_short_term_results.extend(all_results.get('short_term_results', []))
                    all_ma_results.extend(all_results.get('ma_results', []))
                    all_divergence_results.extend(all_results.get('divergence_results', []))
                    all_breakout_results.extend(all_results.get('breakout_results', []))

                    logger.info(f"ADL Enhanced batch {batch_num + 1} completed: {len(all_results.get('composite_results', []))} composite results")

                # Memory cleanup
                del batch_data
                del all_results
                gc.collect()

        # After ALL batches processed, rank and write files
        current_date = datetime.now().strftime('%Y%m%d')

        if all_composite_results:
            # Re-rank all accumulated composite results
            ranked_composite = processor.adl_screener.composite_scorer.rank_stocks(all_composite_results)

            # Generate top candidates from ALL results
            top_candidates = processor.adl_screener.composite_scorer.generate_top_candidates(ranked_composite)

            # Write composite results
            composite_file = processor.adl_dir / f"adl_composite_ranked_{timeframe}_{current_date}.csv"
            processor._write_results_to_csv(composite_file, ranked_composite)
            logger.info(f"ADL composite: {len(ranked_composite)} total results saved to {composite_file}")

            # Write top candidates
            if top_candidates:
                top_file = processor.adl_dir / f"adl_top_candidates_{timeframe}_{current_date}.csv"
                processor._write_results_to_csv(top_file, top_candidates)
                logger.info(f"ADL top candidates: {len(top_candidates)} saved to {top_file}")

            # Write component files if enabled
            if user_config.adl_screener_output_separate_signals:
                if all_mom_results:
                    mom_file = processor.adl_dir / f"adl_mom_accumulation_{timeframe}_{current_date}.csv"
                    processor._write_results_to_csv(mom_file, all_mom_results)

                if all_short_term_results:
                    st_file = processor.adl_dir / f"adl_short_term_momentum_{timeframe}_{current_date}.csv"
                    processor._write_results_to_csv(st_file, all_short_term_results)

                if all_ma_results:
                    ma_file = processor.adl_dir / f"adl_ma_alignment_{timeframe}_{current_date}.csv"
                    processor._write_results_to_csv(ma_file, all_ma_results)

                if all_divergence_results:
                    div_file = processor.adl_dir / f"adl_divergence_{timeframe}_{current_date}.csv"
                    processor._write_results_to_csv(div_file, all_divergence_results)

                if all_breakout_results:
                    bo_file = processor.adl_dir / f"adl_breakout_{timeframe}_{current_date}.csv"
                    processor._write_results_to_csv(bo_file, all_breakout_results)

            # Write summary if enabled
            if user_config.adl_screener_output_summary_stats:
                # Create comprehensive summary with all result types
                complete_summary = {
                    'total_tickers_analyzed': len(all_composite_results),
                    'composite_count': len(all_composite_results),
                    'top_candidates_count': len(top_candidates) if top_candidates else 0,
                    'mom_count': len(all_mom_results),
                    'short_term_count': len(all_short_term_results),
                    'ma_count': len(all_ma_results),
                    'divergence_count': len(all_divergence_results),
                    'breakout_count': len(all_breakout_results),
                }

                # Add composite score statistics
                if ranked_composite:
                    composite_stats = processor.adl_screener.composite_scorer.create_summary_statistics(ranked_composite)
                    complete_summary.update(composite_stats)

                summary_file = processor.adl_dir / f"adl_summary_{timeframe}_{current_date}.txt"
                processor._write_summary_to_file(summary_file, complete_summary)

        results[timeframe] = len(all_composite_results)
        logger.info(f"ADL Enhanced Screener completed for {timeframe}: {len(all_composite_results)} total composite results")

    if results:
        print(f"✅ ADL Enhanced Screener processing completed")
    else:
        print(f"⚠️  ADL Enhanced Screener completed with no results")

    return results
