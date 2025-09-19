"""
Basic Calculations Streaming Processor
=====================================

Memory-efficient streaming version of basic calculations that processes
batches and writes results immediately instead of accumulating in memory.

This replaces the accumulation pattern in basic_calculations.py with
streaming writes for improved memory efficiency.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from src.streaming_base import StreamingCalculationBase
from src.basic_calculations import (
    calculate_technical_indicators,
    calculate_atr_and_atrext,
    calculate_percentage_moves,
    calculate_volume_metrics,
    calculate_ath_atl,
    calculate_candle_strength,
    load_index_boolean_data
)

logger = logging.getLogger(__name__)


class BasicCalculationsStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for basic calculations.

    Processes tickers in batches and writes results immediately to file
    instead of accumulating in memory like the original basic_calculations.
    """

    def __init__(self, config, user_config):
        """
        Initialize basic calculations streaming processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with calculation settings
        """
        super().__init__(config, user_config)

        # Create basic calculations output directory
        self.basic_calc_dir = config.directories['BASIC_CALCULATION_DIR']
        self.basic_calc_dir.mkdir(parents=True, exist_ok=True)

        # Load boolean classification data once
        self.boolean_classifications = load_index_boolean_data(config)

        logger.info(f"Basic calculations streaming processor initialized, output dir: {self.basic_calc_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "basic_calculation"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.basic_calc_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        Calculate basic calculations for a single ticker.

        Args:
            df: Ticker OHLCV data
            ticker: Ticker symbol
            timeframe: Processing timeframe

        Returns:
            Dictionary with calculation results
        """
        try:
            if df is None or df.empty:
                logger.debug(f"Skipping {ticker}: No data")
                return None

            # Initialize ticker result dictionary
            calculation_date = df.index.max() if len(df) > 0 else None
            ticker_result = {
                'ticker': ticker,
                'date': calculation_date,
                'timeframe': timeframe,
                'current_price': df['Close'].iloc[-1] if 'Close' in df.columns and len(df) > 0 else None,
                'data_points': len(df)
            }

            # Calculate technical indicators with user configuration
            indicators = calculate_technical_indicators(df, timeframe, self.user_config)
            ticker_result.update(indicators)

            # Calculate ATR and ATRext metrics (if enabled)
            if self.user_config.enable_atr_calculation:
                atr_metrics = calculate_atr_and_atrext(
                    df,
                    atr_period=self.user_config.atr_period,
                    sma_period=self.user_config.atr_sma_period,
                    enable_percentile=self.user_config.enable_atr_percentile,
                    percentile_period=self.user_config.atr_percentile_period
                )
                ticker_result.update(atr_metrics)

            # Calculate percentage moves
            move_metrics = calculate_percentage_moves(df, timeframe, self.user_config)
            ticker_result.update(move_metrics)

            # Calculate volume metrics
            volume_metrics = calculate_volume_metrics(df, timeframe)
            ticker_result.update(volume_metrics)

            # Calculate ATH/ATL metrics
            ath_atl_metrics = calculate_ath_atl(df, timeframe)
            ticker_result.update(ath_atl_metrics)

            # Calculate candle strength metrics
            candle_strength_metrics = calculate_candle_strength(df, timeframe)
            ticker_result.update(candle_strength_metrics)

            # NOTE: index_overview_metrics calculation removed (obsolete module)

            # Calculate advanced indicators if enabled
            if hasattr(self.user_config, 'indicators_enable') and self.user_config.indicators_enable:
                try:
                    from src.indicators.indicators_calculation import calculate_all_indicators

                    # Build config from user settings
                    indicators_config = {
                        'kurutoga': {
                            'enabled': getattr(self.user_config, 'indicators_kurutoga_enable', True),
                            'length': getattr(self.user_config, 'indicators_kurutoga_length', 14),
                        },
                        'divergence': {
                            'enabled': getattr(self.user_config, 'indicators_divergence_enable', True),
                            'length': getattr(self.user_config, 'indicators_divergence_length', 14),
                        },
                        'kama': {
                            'enabled': getattr(self.user_config, 'indicators_kama_enable', True),
                            'length': getattr(self.user_config, 'indicators_kama_length', 14),
                            'fast_sc': getattr(self.user_config, 'indicators_kama_fast_sc', 2),
                            'slow_sc': getattr(self.user_config, 'indicators_kama_slow_sc', 30),
                        }
                    }

                    advanced_indicators = calculate_all_indicators(df, indicators_config)
                    ticker_result.update(advanced_indicators)

                except ImportError:
                    logger.debug("Advanced indicators module not available")
                except Exception as e:
                    logger.warning(f"Error calculating advanced indicators for {ticker}: {e}")

            # Add boolean classifications if available
            if ticker in self.boolean_classifications:
                ticker_result.update(self.boolean_classifications[ticker])

            return ticker_result

        except Exception as e:
            logger.error(f"Error calculating basic calculations for {ticker}: {e}")
            return None


def run_basic_calculations_streaming(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                                   weekly_data: Dict[str, pd.DataFrame] = None,
                                   monthly_data: Dict[str, pd.DataFrame] = None,
                                   ticker_choice: int = 0) -> Dict[str, str]:
    """
    Run basic calculations using streaming approach for memory efficiency.

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
    # Check if basic calculations are enabled
    if not getattr(user_config, 'enable_basic_calculations', True):
        logger.info("Basic calculations disabled")
        return {}

    processor = BasicCalculationsStreamingProcessor(config, user_config)
    results = {}

    # Process each timeframe with available data
    timeframe_data = {
        'daily': daily_data,
        'weekly': weekly_data,
        'monthly': monthly_data
    }

    for timeframe, data in timeframe_data.items():
        if data:
            logger.info(f"Processing basic calculations for {timeframe} timeframe with streaming...")

            # Convert data to batch format if needed
            if isinstance(data, dict):
                batches = [data]  # Single batch
            else:
                batches = data  # Already in batch format

            # Process with streaming
            result = processor.process_timeframe_streaming(batches, timeframe, ticker_choice)

            if result and 'output_file' in result:
                results[timeframe] = result['output_file']
                logger.info(f"Basic calculations completed for {timeframe}: "
                           f"{result['tickers_processed']} tickers processed, "
                           f"saved to {result['output_file']}")

                # Log memory savings
                if 'memory_saved_mb' in result:
                    logger.info(f"Memory saved: {result['memory_saved_mb']:.1f} MB")

    return results