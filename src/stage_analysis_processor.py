"""
Stage Analysis Processor
========================

Processes stage analysis for stocks across different timeframes and generates
separate output files similar to the RS processor structure.

Creates output files:
- stage_analysis_{choice}_daily_{date}.csv
- stage_analysis_{choice}_weekly_{date}.csv  
- stage_analysis_{choice}_monthly_{date}.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from src.stage_analysis import StageAnalyzer
from src.basic_calculations import calculate_technical_indicators

logger = logging.getLogger(__name__)


class StageAnalysisProcessor:
    """
    Processes stage analysis for multiple timeframes and generates separate output files.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize stage analysis processor.
        
        Args:
            config: Configuration object with directories
            user_config: User configuration with stage analysis settings
        """
        self.config = config
        self.user_config = user_config
        
        # Create stage analysis output directory
        self.stage_dir = config.directories['RESULTS_DIR'] / 'stage_analysis'
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers for each timeframe
        self.analyzers = {}
        self._initialize_analyzers()
        
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
            
        logger.info(f"Initialized stage analyzers for: {list(self.analyzers.keys())}")
    
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
            config.stage_ma_convergence_threshold = getattr(self.user_config, 'stage_daily_ma_convergence_threshold',
                                                          getattr(self.user_config, 'stage_ma_convergence_threshold', 1.0))
        elif timeframe == 'weekly':
            config.stage_ema_fast_period = getattr(self.user_config, 'stage_weekly_ema_fast_period', 10)
            config.stage_sma_medium_period = getattr(self.user_config, 'stage_weekly_sma_medium_period', 20)
            config.stage_sma_slow_period = getattr(self.user_config, 'stage_weekly_sma_slow_period', 50)
            config.stage_atr_period = getattr(self.user_config, 'stage_weekly_atr_period', 14)
            config.stage_atr_threshold_low = getattr(self.user_config, 'stage_weekly_atr_threshold_low', 4.0)
            config.stage_atr_threshold_high = getattr(self.user_config, 'stage_weekly_atr_threshold_high', 7.0)
            config.stage_ma_convergence_threshold = getattr(self.user_config, 'stage_weekly_ma_convergence_threshold', 1.0)
        elif timeframe == 'monthly':
            config.stage_ema_fast_period = getattr(self.user_config, 'stage_monthly_ema_fast_period', 10)
            config.stage_sma_medium_period = getattr(self.user_config, 'stage_monthly_sma_medium_period', 20)
            config.stage_sma_slow_period = getattr(self.user_config, 'stage_monthly_sma_slow_period', 50)
            config.stage_atr_period = getattr(self.user_config, 'stage_monthly_atr_period', 14)
            config.stage_atr_threshold_low = getattr(self.user_config, 'stage_monthly_atr_threshold_low', 4.0)
            config.stage_atr_threshold_high = getattr(self.user_config, 'stage_monthly_atr_threshold_high', 7.0)
            config.stage_ma_convergence_threshold = getattr(self.user_config, 'stage_monthly_ma_convergence_threshold', 1.0)
        
        return config
    
    def process_timeframe(self, market_data: Dict[str, pd.DataFrame], timeframe: str,
                         ticker_choice: int = 0) -> Optional[str]:
        """
        Process stage analysis for a specific timeframe.
        Uses batch accumulation pattern like basic_calculations.

        Args:
            market_data: Dictionary with ticker -> DataFrame mappings
            timeframe: 'daily', 'weekly', or 'monthly'
            ticker_choice: User ticker choice number

        Returns:
            Path to output file or None if processing failed
        """
        if timeframe not in self.analyzers:
            logger.warning(f"Stage analysis not enabled for {timeframe} timeframe")
            return None

        if not market_data:
            logger.warning(f"No market data provided for {timeframe} stage analysis")
            return None

        logger.info(f"Processing {timeframe} stage analysis for {len(market_data)} tickers...")

        # Initialize global accumulation storage (same pattern as basic_calculations)
        if not hasattr(self, 'all_results'):
            self.all_results = {}
        if timeframe not in self.all_results:
            self.all_results[timeframe] = {}

        analyzer = self.analyzers[timeframe]
        
        # Process each ticker
        processed_count = 0
        error_count = 0
        
        for ticker, df in market_data.items():
            try:
                if df is None or df.empty or len(df) < 50:
                    continue

                # Calculate basic indicators needed for stage analysis
                indicators = calculate_technical_indicators(df, timeframe, self.user_config)

                # Run stage analysis
                stage_result = analyzer.analyze_ticker(ticker, df, timeframe)

                if stage_result:
                    # Combine ticker info with stage result using YYYY-MM-DD date format
                    calculation_date = df.index[-1].strftime('%Y-%m-%d')
                    result_row = {
                        'ticker': ticker,
                        'date': calculation_date,
                        'timeframe': timeframe,
                        f'{timeframe}_price': df['Close'].iloc[-1],
                        **stage_result
                    }

                    # Add relevant indicators
                    if indicators:
                        # Add key moving averages used in stage analysis (using consistent timeframe-prefixed names)
                        ma_keys = [
                            f'{timeframe}_ema_{analyzer.ema_fast_period}',
                            f'{timeframe}_sma_{analyzer.sma_medium_period}',
                            f'{timeframe}_sma_{analyzer.sma_slow_period}'
                        ]

                        for ma_key in ma_keys:
                            if ma_key in indicators:
                                result_row[ma_key] = indicators[ma_key]

                        # Add ATR information
                        if 'atr' in indicators:
                            result_row['atr'] = indicators['atr']
                        if 'atr_pct' in indicators:
                            result_row['atr_pct'] = indicators['atr_pct']
                        if 'atr_ratio' in indicators:
                            result_row[f'{timeframe}_atr_ratio_sma_50'] = indicators['atr_ratio']

                    # Store result in global accumulation (same pattern as basic_calculations)
                    self.all_results[timeframe][ticker] = result_row
                    processed_count += 1

            except Exception as e:
                logger.warning(f"Stage analysis failed for {ticker} ({timeframe}): {e}")
                error_count += 1
                # Store error for failed ticker (same pattern as basic_calculations)
                self.all_results[timeframe][ticker] = {'error': str(e)}
                continue

        logger.info(f"Stage analysis summary ({timeframe}): {processed_count} processed, {error_count} errors")

        # Return success - actual file creation will be done later by save_stage_analysis_matrix
        return True
    
    def save_stage_analysis_matrix(self, ticker_choice: int = 0) -> Dict[str, str]:
        """
        Save accumulated stage analysis results to files for all timeframes.
        Similar to save_basic_calculations_matrix in basic_calculations.

        Args:
            ticker_choice: User ticker choice number

        Returns:
            Dictionary mapping timeframe -> output file path
        """
        if not hasattr(self, 'all_results'):
            logger.warning("No accumulated stage analysis results to save")
            return {}

        results = {}

        for timeframe in self.all_results:
            if not self.all_results[timeframe]:
                continue

            # Convert accumulated results to DataFrame
            stage_results = []
            for ticker, result in self.all_results[timeframe].items():
                if 'error' not in result:  # Skip error entries
                    stage_results.append(result)

            if not stage_results:
                logger.warning(f"No valid {timeframe} stage analysis results to save")
                continue

            # Create output DataFrame
            results_df = pd.DataFrame(stage_results)

            # Sort by ticker
            results_df = results_df.sort_values('ticker')

            # Generate output filename using YYYY-MM-DD date format
            data_date = None
            if not results_df.empty and 'date' in results_df.columns:
                # Use the latest data date from the results (YYYY-MM-DD format)
                data_date = results_df['date'].iloc[0].replace('-', '')  # Convert 2025-08-29 to 20250829

            # Fallback to current date if no data date found
            if not data_date:
                data_date = datetime.now().strftime('%Y%m%d')
                logger.warning(f"Using current date as fallback for {timeframe} stage analysis: {data_date}")
            else:
                logger.info(f"Using data date for {timeframe} stage analysis filename: {data_date}")

            output_filename = f"stage_analysis_{ticker_choice}_{timeframe}_{data_date}.csv"
            output_path = self.stage_dir / output_filename

            # Save results
            results_df.to_csv(output_path, index=False, float_format='%.4f')

            logger.info(f"Saved {timeframe} stage analysis: {len(results_df)} results to {output_path}")
            results[timeframe] = str(output_path)

        logger.info(f"Stage analysis matrix save completed: {len(results)} files generated")
        return results

    def process_all_timeframes(self, daily_data: Dict[str, pd.DataFrame] = None,
                              weekly_data: Dict[str, pd.DataFrame] = None,
                              monthly_data: Dict[str, pd.DataFrame] = None,
                              ticker_choice: int = 0) -> Dict[str, str]:
        """
        Process stage analysis for all enabled timeframes.
        Now just accumulates results, actual file saving done by save_stage_analysis_matrix.

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
        if daily_data and 'daily' in self.analyzers:
            daily_success = self.process_timeframe(daily_data, 'daily', ticker_choice)
            if daily_success:
                results['daily'] = True

        # Process weekly data
        if weekly_data and 'weekly' in self.analyzers:
            weekly_success = self.process_timeframe(weekly_data, 'weekly', ticker_choice)
            if weekly_success:
                results['weekly'] = True

        # Process monthly data
        if monthly_data and 'monthly' in self.analyzers:
            monthly_success = self.process_timeframe(monthly_data, 'monthly', ticker_choice)
            if monthly_success:
                results['monthly'] = True

        logger.info(f"Stage analysis processing completed: {len(results)} timeframes processed")
        return results
    
    def get_processing_summary(self, results: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate a summary of the stage analysis processing.
        
        Args:
            results: Dictionary of timeframe -> file path mappings
            
        Returns:
            Summary dictionary
        """
        summary = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframes_processed': list(results.keys()),
            'total_files_generated': len(results),
            'output_directory': str(self.stage_dir),
            'file_details': {}
        }
        
        for timeframe, file_path in results.items():
            try:
                # Read file to get basic stats
                df = pd.read_csv(file_path)
                
                # Get stage distribution (look for sa_name column)
                sa_name_col = None
                for col in df.columns:
                    if '_sa_name' in col:
                        sa_name_col = col
                        break
                stage_counts = df[sa_name_col].value_counts().to_dict() if sa_name_col else {}
                
                summary['file_details'][timeframe] = {
                    'file_path': file_path,
                    'ticker_count': len(df),
                    'stage_distribution': stage_counts,
                    'file_size_mb': round(Path(file_path).stat().st_size / (1024*1024), 2)
                }
            except Exception as e:
                summary['file_details'][timeframe] = {
                    'file_path': file_path,
                    'error': str(e)
                }
        
        return summary


def run_stage_analysis_processing(config, user_config, daily_data: Dict[str, pd.DataFrame] = None,
                                weekly_data: Dict[str, pd.DataFrame] = None,
                                monthly_data: Dict[str, pd.DataFrame] = None,
                                ticker_choice: int = 0) -> Dict[str, str]:
    """
    Convenience function to run stage analysis processing.
    Now uses accumulation pattern like basic_calculations.

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
    # Check if any stage analysis is enabled
    daily_enabled = getattr(user_config, 'stage_analysis_daily_enabled', True)
    weekly_enabled = getattr(user_config, 'stage_analysis_weekly_enabled', True)
    monthly_enabled = getattr(user_config, 'stage_analysis_monthly_enabled', True)

    if not (daily_enabled or weekly_enabled or monthly_enabled):
        logger.info("Stage analysis disabled for all timeframes")
        return {}

    processor = StageAnalysisProcessor(config, user_config)

    # Process all timeframes (accumulate results)
    processor.process_all_timeframes(daily_data, weekly_data, monthly_data, ticker_choice)

    # Save accumulated results to files
    results = processor.save_stage_analysis_matrix(ticker_choice)

    # Generate and log summary
    summary = processor.get_processing_summary(results)
    logger.info(f"Stage Analysis Summary: {summary['total_files_generated']} files, "
               f"{summary['timeframes_processed']} timeframes")

    return results