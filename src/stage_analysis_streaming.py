"""
Stage Analysis Streaming Processor
==================================

Memory-efficient streaming version of stage analysis that processes
batches and writes results immediately instead of accumulating in memory.

This replaces the accumulation pattern in stage_analysis_processor.py with
streaming writes for improved memory efficiency.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from src.streaming_base import StreamingCalculationBase
from src.stage_analysis import StageAnalyzer
from src.basic_calculations import calculate_technical_indicators

logger = logging.getLogger(__name__)


class StageAnalysisStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for stage analysis.

    Processes tickers in batches and writes results immediately to file
    instead of accumulating in memory like the original stage_analysis_processor.
    """

    def __init__(self, config, user_config):
        """
        Initialize stage analysis streaming processor.

        Args:
            config: Configuration object with directories
            user_config: User configuration with stage analysis settings
        """
        super().__init__(config, user_config)

        # Create stage analysis output directory
        self.stage_dir = config.directories['RESULTS_DIR'] / 'stage_analysis'
        self.stage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analyzers for each timeframe
        self.analyzers = {}
        self._initialize_analyzers()

        logger.info(f"Stage analysis streaming processor initialized, output dir: {self.stage_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "stage_analysis"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.stage_dir

    def _initialize_analyzers(self):
        """Initialize stage analyzers for each enabled timeframe."""
        # Daily analyzer
        if getattr(self.user_config, 'stage_analysis_daily_enabled', True):
            daily_config = self._create_timeframe_config('daily')
            self.analyzers['daily'] = StageAnalyzer(daily_config)

        # Weekly analyzer
        if getattr(self.user_config, 'stage_analysis_weekly_enabled', True):
            weekly_config = self._create_timeframe_config('weekly')
            self.analyzers['weekly'] = StageAnalyzer(weekly_config)

        # Monthly analyzer
        if getattr(self.user_config, 'stage_analysis_monthly_enabled', True):
            monthly_config = self._create_timeframe_config('monthly')
            self.analyzers['monthly'] = StageAnalyzer(monthly_config)

        logger.debug(f"Initialized stage analyzers for: {list(self.analyzers.keys())}")

    def _create_timeframe_config(self, timeframe: str):
        """
        Create a configuration object for a specific timeframe.

        Args:
            timeframe: 'daily', 'weekly', or 'monthly'

        Returns:
            Configuration object with timeframe-specific parameters
        """
        class TimeframeConfig:
            pass

        config = TimeframeConfig()

        # Set parameters based on timeframe, fallback to legacy if new ones don't exist
        if timeframe == 'daily':
            config.stage_ema_fast_period = getattr(self.user_config, 'stage_daily_ema_fast_period',
                                                 getattr(self.user_config, 'stage_ema_fast_period', 10))
            config.stage_sma_medium_period = getattr(self.user_config, 'stage_daily_sma_medium_period',
                                                   getattr(self.user_config, 'stage_sma_medium_period', 20))
            config.stage_sma_slow_period = getattr(self.user_config, 'stage_daily_sma_slow_period',
                                                 getattr(self.user_config, 'stage_sma_slow_period', 50))
            config.stage_atr_period = getattr(self.user_config, 'stage_daily_atr_period',
                                            getattr(self.user_config, 'stage_atr_period', 14))
            config.stage_atr_threshold_low = getattr(self.user_config, 'stage_daily_atr_threshold_low',
                                                   getattr(self.user_config, 'stage_atr_threshold_low', 4.0))
            config.stage_atr_threshold_high = getattr(self.user_config, 'stage_daily_atr_threshold_high',
                                                    getattr(self.user_config, 'stage_atr_threshold_high', 7.0))
        elif timeframe == 'weekly':
            config.stage_ema_fast_period = getattr(self.user_config, 'stage_weekly_ema_fast_period',
                                                 getattr(self.user_config, 'stage_ema_fast_period', 10))
            config.stage_sma_medium_period = getattr(self.user_config, 'stage_weekly_sma_medium_period',
                                                   getattr(self.user_config, 'stage_sma_medium_period', 20))
            config.stage_sma_slow_period = getattr(self.user_config, 'stage_weekly_sma_slow_period',
                                                 getattr(self.user_config, 'stage_sma_slow_period', 50))
            config.stage_atr_period = getattr(self.user_config, 'stage_weekly_atr_period',
                                            getattr(self.user_config, 'stage_atr_period', 14))
            config.stage_atr_threshold_low = getattr(self.user_config, 'stage_weekly_atr_threshold_low',
                                                   getattr(self.user_config, 'stage_atr_threshold_low', 4.0))
            config.stage_atr_threshold_high = getattr(self.user_config, 'stage_weekly_atr_threshold_high',
                                                    getattr(self.user_config, 'stage_atr_threshold_high', 7.0))
        elif timeframe == 'monthly':
            config.stage_ema_fast_period = getattr(self.user_config, 'stage_monthly_ema_fast_period',
                                                 getattr(self.user_config, 'stage_ema_fast_period', 10))
            config.stage_sma_medium_period = getattr(self.user_config, 'stage_monthly_sma_medium_period',
                                                   getattr(self.user_config, 'stage_sma_medium_period', 20))
            config.stage_sma_slow_period = getattr(self.user_config, 'stage_monthly_sma_slow_period',
                                                 getattr(self.user_config, 'stage_sma_slow_period', 50))
            config.stage_atr_period = getattr(self.user_config, 'stage_monthly_atr_period',
                                            getattr(self.user_config, 'stage_atr_period', 14))
            config.stage_atr_threshold_low = getattr(self.user_config, 'stage_monthly_atr_threshold_low',
                                                   getattr(self.user_config, 'stage_atr_threshold_low', 4.0))
            config.stage_atr_threshold_high = getattr(self.user_config, 'stage_monthly_atr_threshold_high',
                                                    getattr(self.user_config, 'stage_atr_threshold_high', 7.0))

        # Common settings
        config.stage_ma_convergence_threshold = getattr(self.user_config, 'stage_ma_convergence_threshold', 1.0)
        config.stage_volume_spike_threshold = getattr(self.user_config, 'stage_volume_spike_threshold', 2.0)
        config.stage_breakout_threshold = getattr(self.user_config, 'stage_breakout_threshold', 1.02)

        return config

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        Calculate stage analysis for a single ticker.

        Args:
            df: Ticker OHLCV data
            ticker: Ticker symbol
            timeframe: Processing timeframe

        Returns:
            Dictionary with stage analysis results
        """
        try:
            if df is None or df.empty:
                logger.debug(f"Skipping {ticker}: No data")
                return None

            # Get analyzer for this timeframe
            if timeframe not in self.analyzers:
                logger.warning(f"No analyzer available for {timeframe} timeframe")
                return None

            analyzer = self.analyzers[timeframe]

            # Calculate stage analysis
            stage_result = analyzer.analyze_ticker(ticker, df, timeframe)

            if stage_result is None:
                logger.debug(f"No stage analysis result for {ticker}")
                return None

            # Get additional technical indicators needed for stage analysis
            indicators = calculate_technical_indicators(df, timeframe, self.user_config)

            # Build comprehensive result using actual stage_result structure
            ticker_result = {
                'ticker': ticker,
                'timeframe': timeframe,
                'date': df.index.max() if len(df) > 0 else None,
                'current_price': df['Close'].iloc[-1] if 'Close' in df.columns and len(df) > 0 else None,
            }

            # Add the complete stage analysis results
            ticker_result.update(stage_result)

            # Add technical indicators
            ticker_result.update(indicators)

            return ticker_result

        except Exception as e:
            logger.error(f"Error calculating stage analysis for {ticker}: {e}")
            return None


def run_stage_analysis_streaming(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                               weekly_data: Dict[str, pd.DataFrame] = None,
                               monthly_data: Dict[str, pd.DataFrame] = None,
                               ticker_choice: int = 0) -> Dict[str, str]:
    """
    Run stage analysis using streaming approach for memory efficiency.

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
    # Check if stage analysis is enabled
    if not getattr(user_config, 'enable_stage_analysis', True):
        logger.info("Stage analysis disabled")
        return {}

    processor = StageAnalysisStreamingProcessor(config, user_config)
    results = {}

    # Process each timeframe with available data
    timeframe_data = {
        'daily': daily_data,
        'weekly': weekly_data,
        'monthly': monthly_data
    }

    for timeframe, data in timeframe_data.items():
        # Check if this timeframe is enabled
        timeframe_enabled = getattr(user_config, f'stage_analysis_{timeframe}_enabled', True)
        if not timeframe_enabled:
            logger.info(f"Stage analysis disabled for {timeframe} timeframe")
            continue

        if data and timeframe in processor.analyzers:
            logger.info(f"Processing stage analysis for {timeframe} timeframe with streaming...")

            # Convert data to batch format if needed
            if isinstance(data, dict):
                batches = [data]  # Single batch
            else:
                batches = data  # Already in batch format

            # Process with streaming
            result = processor.process_timeframe_streaming(batches, timeframe, ticker_choice)

            if result and 'output_file' in result:
                results[timeframe] = result['output_file']
                logger.info(f"Stage analysis completed for {timeframe}: "
                           f"{result['tickers_processed']} tickers processed, "
                           f"saved to {result['output_file']}")

                # Log memory savings
                if 'memory_saved_mb' in result:
                    logger.info(f"Memory saved: {result['memory_saved_mb']:.1f} MB")

    return results