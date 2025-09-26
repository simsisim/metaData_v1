"""
MMM Gap Calculation Module
==========================

Calculate daily price gaps and market manipulation indicators for configured tickers.

Key Calculations:
- gap = Open[i] - Close[i-1]
- AdjustClose_woGap = Close[i] - (Close[i-1] - Open[i])
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class MmmGapsProcessor:
    """
    Process MMM gap calculations for configured tickers.
    """

    def __init__(self, config, user_config, timeframe):
        """
        Initialize MMM gaps processor.

        Args:
            config: System configuration
            user_config: User configuration
            timeframe: Processing timeframe ('daily', 'weekly', 'monthly')
        """
        self.config = config
        self.user_config = user_config
        self.timeframe = timeframe

    def run_gap_analysis(self):
        """
        Run complete gap analysis for all configured tickers.

        Returns:
            Dictionary with gap analysis results and file paths
        """
        try:
            logger.info(f"Starting MMM gap analysis for {self.timeframe} timeframe...")

            # Get tickers from configuration
            tickers = self._parse_tickers_config()
            if not tickers:
                logger.warning("No tickers configured for MMM gap analysis")
                return {'error': 'No tickers configured'}

            logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")

            # Get input and output directories
            input_dir = self._get_input_directory()
            output_dir = self._get_output_directory()

            if not input_dir or not output_dir:
                return {'error': 'Input/output directories not configured'}

            # Process each ticker
            gap_results = {
                'tickers_processed': [],
                'csv_files': [],
                'failed_tickers': [],
                'total_gaps_calculated': 0
            }

            for ticker in tickers:
                try:
                    result = self._process_ticker_gaps(ticker, input_dir, output_dir)
                    if result and 'error' not in result:
                        gap_results['tickers_processed'].append(ticker)
                        gap_results['csv_files'].append(result['csv_file'])
                        gap_results['total_gaps_calculated'] += result.get('gap_count', 0)
                        logger.info(f"✅ {ticker}: {result.get('gap_count', 0)} gaps calculated")
                    else:
                        gap_results['failed_tickers'].append(ticker)
                        logger.warning(f"❌ {ticker}: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    gap_results['failed_tickers'].append(ticker)
                    logger.error(f"❌ {ticker}: Error processing gaps: {e}")

            logger.info(f"MMM gap analysis completed: {len(gap_results['tickers_processed'])} successful, "
                       f"{len(gap_results['failed_tickers'])} failed")

            return gap_results

        except Exception as e:
            logger.error(f"Error in MMM gap analysis: {e}")
            return {'error': str(e)}

    def _parse_tickers_config(self) -> List[str]:
        """
        Parse tickers from user configuration.

        Returns:
            List of ticker symbols
        """
        try:
            tickers_config = getattr(self.user_config, 'sr_mmm_gaps_tickers', '')
            if not tickers_config:
                return []

            # Handle semicolon-separated format: XLY;XLC
            if ';' in tickers_config:
                tickers = [ticker.strip() for ticker in tickers_config.split(';')]
            else:
                # Handle single ticker or comma-separated
                tickers = [ticker.strip() for ticker in tickers_config.replace(',', ';').split(';')]

            # Filter out empty strings
            tickers = [ticker for ticker in tickers if ticker]

            logger.debug(f"Parsed tickers: {tickers}")
            return tickers

        except Exception as e:
            logger.error(f"Error parsing tickers configuration: {e}")
            return []

    def _get_input_directory(self) -> Optional[Path]:
        """
        Get input directory based on timeframe.

        Returns:
            Path to input directory or None if not configured
        """
        try:
            if self.timeframe == 'daily':
                input_dir = getattr(self.user_config, 'sr_mmm_gaps_values_input_folder_daily', '')
            elif self.timeframe == 'weekly':
                input_dir = getattr(self.user_config, 'sr_mmm_gaps_values_input_folder_weekly', '')
            elif self.timeframe == 'monthly':
                input_dir = getattr(self.user_config, 'sr_mmm_gaps_values_input_folder_monthly', '')
            else:
                # Fallback to general input folder
                input_dir = getattr(self.user_config, 'sr_mmm_gaps_values_input_folder', '')

            if input_dir:
                path = Path(input_dir)
                if path.exists():
                    logger.debug(f"Using input directory: {path}")
                    return path
                else:
                    logger.warning(f"Input directory does not exist: {path}")

            return None

        except Exception as e:
            logger.error(f"Error getting input directory: {e}")
            return None

    def _get_output_directory(self) -> Optional[Path]:
        """
        Get output directory for gap files based on timeframe configuration.

        Returns:
            Path to output directory
        """
        try:
            # Use timeframe-specific output directory from configuration
            if self.timeframe == 'daily':
                output_dir = getattr(self.user_config, 'sr_mmm_gaps_values_output_folder_daily',
                                   '../downloadData_v1/data/market_data/daily/')
            elif self.timeframe == 'weekly':
                output_dir = getattr(self.user_config, 'sr_mmm_gaps_values_output_folder_weekly',
                                   '../downloadData_v1/data/market_data/weekly/')
            elif self.timeframe == 'monthly':
                output_dir = getattr(self.user_config, 'sr_mmm_gaps_values_output_folder_monthly',
                                   '../downloadData_v1/data/market_data/monthly/')
            else:
                # Fallback to MMM output directory
                logger.warning(f"Unknown timeframe '{self.timeframe}', using fallback directory")
                mmm_output_dir = getattr(self.user_config, 'sr_mmm_output_dir', 'results/sustainability_ratios/mmm')
                output_dir = str(Path(mmm_output_dir) / 'gaps')

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Using {self.timeframe} output directory: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            return None

    def _process_ticker_gaps(self, ticker: str, input_dir: Path, output_dir: Path) -> Dict:
        """
        Process gap calculations for a single ticker.

        Args:
            ticker: Ticker symbol
            input_dir: Input directory path
            output_dir: Output directory path

        Returns:
            Dictionary with processing results
        """
        try:
            # Load ticker data
            input_file = input_dir / f"{ticker}.csv"
            if not input_file.exists():
                return {'error': f'Input file not found: {input_file}'}

            logger.debug(f"Loading data from: {input_file}")
            df = pd.read_csv(input_file, index_col=0, parse_dates=True)

            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {'error': f'Missing columns: {missing_columns}'}

            # Sort by date to ensure proper gap calculation
            df = df.sort_index()

            # Calculate gaps
            gap_df = self._calculate_gaps(df)

            # Add filename suffix
            filename_suffix = getattr(self.user_config, 'sr_mmm_gaps_values_filename_suffix', '_gap')
            output_file = output_dir / f"{ticker}{filename_suffix}.csv"

            # Save gap data
            gap_df.to_csv(output_file, index=True)

            logger.debug(f"Gap data saved to: {output_file}")

            return {
                'csv_file': str(output_file),
                'gap_count': len(gap_df.dropna(subset=['gap'])),
                'date_range': f"{gap_df.index.min()} to {gap_df.index.max()}",
                'success': True
            }

        except Exception as e:
            logger.error(f"Error processing {ticker} gaps: {e}")
            return {'error': str(e)}

    def _calculate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate gap metrics for OHLCV data and prepare for gap analysis display.

        Calculates gap metrics and renames columns for proper gap analysis visualization:
        1. gap = Open[i] - Close[i-1]
        2. AdjustClose_woGap = Close[i] - (Close[i-1] - Open[i])
        3. Replaces Close column with AdjustClose_woGap for gap-focused analysis
        4. Preserves original Close as Close_original

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with gap analysis columns and Close replaced with gap-adjusted values
        """
        try:
            # Create copy to avoid modifying original data
            result_df = df.copy()

            # Ensure data is sorted by date
            result_df = result_df.sort_index()

            # Get previous close by shifting 1 day
            prev_close = result_df['Close'].shift(1)
            current_open = result_df['Open']
            current_close = result_df['Close']

            # Calculate gap: Open[i] - Close[i-1]
            gap = current_open - prev_close

            # Calculate AdjustClose_woGap: Close[i] - (Close[i-1] - Open[i])
            adjust_close_wo_gap = current_close - (prev_close - current_open)

            # COLUMN RENAMING FOR GAP ANALYSIS:
            # Preserve original Close for reference
            result_df['Close_original'] = result_df['Close']

            # Replace Close column with gap-adjusted values for analysis
            result_df['Close'] = adjust_close_wo_gap

            # Add gap column for reference
            result_df['gap'] = gap

            logger.debug(f"Gap calculations completed: {len(result_df)} rows processed")
            logger.debug("Applied column renaming: Close -> Close_original, AdjustClose_woGap -> Close")

            return result_df

        except Exception as e:
            logger.error(f"Error in gap calculations: {e}")
            raise e

