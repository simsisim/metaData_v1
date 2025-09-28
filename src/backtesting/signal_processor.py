"""
Signal Processor - Load and Process Screener Outputs
====================================================

Handles loading and processing of screener CSV files for backtesting.
Converts screener outputs into standardized signal format for analysis.

Responsibilities:
- Find and load screener CSV files
- Standardize signal format across different screeners
- Validate signal data quality
- Filter signals by date range and criteria
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Processes screener CSV outputs into standardized signals for backtesting.
    """

    def __init__(self, config):
        """
        Initialize SignalProcessor.

        Args:
            config: System configuration object
        """
        self.config = config
        self.results_dir = Path(config.directories.get('RESULTS_DIR', 'results'))
        self.screener_dir = self.results_dir / 'screeners'

        # Standard signal columns expected across all screeners
        self.required_columns = [
            'ticker', 'signal_date', 'signal_type', 'signal_price'
        ]

        # Optional columns that enhance backtesting
        self.optional_columns = [
            'current_price', 'performance_since_signal', 'days_since_signal',
            'score', 'volume', 'screen_type', 'timeframe'
        ]

    def find_screener_files(self, timeframes: List[str] = None) -> Dict[str, List[Path]]:
        """
        Find all screener CSV output files.

        Args:
            timeframes: List of timeframes to search for (default: all)

        Returns:
            Dict mapping screener types to list of file paths
        """
        if timeframes is None:
            timeframes = ['daily', 'weekly', 'monthly']

        screener_files = {}

        try:
            if not self.screener_dir.exists():
                logger.warning(f"Screener directory not found: {self.screener_dir}")
                return {}

            # Known screener types and their subdirectories
            screener_types = [
                'pvb', 'atr1', 'atr2', 'giusti', 'minervini', 'drwish',
                'volume_suite', 'stockbee', 'qullamaggie', 'adl',
                'guppy', 'gold_launch_pad', 'rti'
            ]

            for screener_type in screener_types:
                screener_path = self.screener_dir / screener_type
                if screener_path.exists():
                    files = []
                    for timeframe in timeframes:
                        # Look for files matching pattern: {screener}*{timeframe}*.csv
                        pattern = f"*{timeframe}*.csv"
                        matching_files = list(screener_path.glob(pattern))
                        files.extend(matching_files)

                    if files:
                        screener_files[screener_type] = files

            logger.info(f"Found screener files for {len(screener_files)} screener types")
            return screener_files

        except Exception as e:
            logger.error(f"Error finding screener files: {e}")
            return {}

    def load_all_signals(self, screener_files: Dict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
        """
        Load and standardize signals from all screener files.

        Args:
            screener_files: Dict mapping screener types to file paths

        Returns:
            Dict mapping screener types to standardized signal DataFrames
        """
        all_signals = {}

        for screener_type, file_paths in screener_files.items():
            try:
                signals = self._load_screener_signals(screener_type, file_paths)
                if not signals.empty:
                    all_signals[screener_type] = signals
                    logger.info(f"Loaded {len(signals)} signals from {screener_type}")

            except Exception as e:
                logger.error(f"Error loading signals for {screener_type}: {e}")
                continue

        return all_signals

    def _load_screener_signals(self, screener_type: str, file_paths: List[Path]) -> pd.DataFrame:
        """
        Load and standardize signals from a specific screener type.

        Args:
            screener_type: Type of screener (e.g., 'pvb', 'atr1')
            file_paths: List of CSV file paths for this screener

        Returns:
            Standardized DataFrame with signals
        """
        all_signals = []

        for file_path in file_paths:
            try:
                # Load CSV file
                df = pd.read_csv(file_path)

                if df.empty:
                    logger.debug(f"Empty file: {file_path}")
                    continue

                # Standardize the DataFrame
                standardized_df = self._standardize_signal_format(df, screener_type, file_path)

                if not standardized_df.empty:
                    all_signals.append(standardized_df)

            except Exception as e:
                logger.warning(f"Error loading file {file_path}: {e}")
                continue

        # Combine all signals for this screener type
        if all_signals:
            combined_signals = pd.concat(all_signals, ignore_index=True)
            return self._post_process_signals(combined_signals, screener_type)
        else:
            return pd.DataFrame()

    def _standardize_signal_format(
        self,
        df: pd.DataFrame,
        screener_type: str,
        file_path: Path
    ) -> pd.DataFrame:
        """
        Standardize signal format for a specific screener DataFrame.

        Args:
            df: Raw screener DataFrame
            screener_type: Type of screener
            file_path: Source file path

        Returns:
            Standardized DataFrame
        """
        try:
            # Check for required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns in {file_path}: {missing_cols}")
                return pd.DataFrame()

            # Create standardized DataFrame
            standardized = df.copy()

            # Ensure signal_date is datetime
            if 'signal_date' in standardized.columns:
                standardized['signal_date'] = pd.to_datetime(standardized['signal_date'])

            # Add metadata columns
            standardized['screener_type'] = screener_type
            standardized['source_file'] = str(file_path.name)

            # Extract timeframe from filename if not present
            if 'timeframe' not in standardized.columns:
                timeframe = self._extract_timeframe_from_filename(file_path.name)
                standardized['timeframe'] = timeframe

            return standardized

        except Exception as e:
            logger.error(f"Error standardizing signals from {file_path}: {e}")
            return pd.DataFrame()

    def _post_process_signals(self, df: pd.DataFrame, screener_type: str) -> pd.DataFrame:
        """
        Post-process signals after combining all files for a screener type.

        Args:
            df: Combined signals DataFrame
            screener_type: Type of screener

        Returns:
            Post-processed DataFrame
        """
        try:
            # Remove duplicates
            original_count = len(df)
            df = df.drop_duplicates(subset=['ticker', 'signal_date', 'signal_type'])
            if len(df) < original_count:
                logger.info(f"Removed {original_count - len(df)} duplicate signals for {screener_type}")

            # Sort by signal date
            df = df.sort_values('signal_date').reset_index(drop=True)

            # Validate signal types
            valid_signal_types = ['Buy', 'Sell', 'Close Buy', 'Close Sell']
            invalid_signals = df[~df['signal_type'].isin(valid_signal_types)]
            if not invalid_signals.empty:
                logger.warning(f"Found {len(invalid_signals)} signals with invalid signal types")

            # Filter out future signals (signals after current date)
            current_date = datetime.now().date()
            future_signals = df[df['signal_date'].dt.date > current_date]
            if not future_signals.empty:
                logger.warning(f"Filtering out {len(future_signals)} future signals")
                df = df[df['signal_date'].dt.date <= current_date]

            return df

        except Exception as e:
            logger.error(f"Error post-processing signals for {screener_type}: {e}")
            return df

    def _extract_timeframe_from_filename(self, filename: str) -> str:
        """Extract timeframe from filename."""
        filename_lower = filename.lower()
        if 'daily' in filename_lower:
            return 'daily'
        elif 'weekly' in filename_lower:
            return 'weekly'
        elif 'monthly' in filename_lower:
            return 'monthly'
        else:
            return 'daily'  # Default assumption

    def filter_signals_by_date_range(
        self,
        signals: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Filter signals by date range.

        Args:
            signals: Signals DataFrame
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)

        Returns:
            Filtered DataFrame
        """
        try:
            filtered_signals = signals.copy()

            if start_date:
                filtered_signals = filtered_signals[
                    filtered_signals['signal_date'] >= start_date
                ]

            if end_date:
                filtered_signals = filtered_signals[
                    filtered_signals['signal_date'] <= end_date
                ]

            logger.info(f"Filtered signals: {len(signals)} -> {len(filtered_signals)}")
            return filtered_signals

        except Exception as e:
            logger.error(f"Error filtering signals by date range: {e}")
            return signals

    def get_signal_summary(self, signals_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Get summary statistics for loaded signals.

        Args:
            signals_data: Dict mapping screener types to signal DataFrames

        Returns:
            Summary statistics dictionary
        """
        try:
            summary = {
                'total_screeners': len(signals_data),
                'total_signals': sum(len(df) for df in signals_data.values()),
                'screener_breakdown': {},
                'signal_type_breakdown': {},
                'date_range': {}
            }

            all_signals = []
            for screener_type, df in signals_data.items():
                summary['screener_breakdown'][screener_type] = len(df)
                all_signals.append(df)

            if all_signals:
                combined_signals = pd.concat(all_signals, ignore_index=True)

                # Signal type breakdown
                signal_type_counts = combined_signals['signal_type'].value_counts()
                summary['signal_type_breakdown'] = signal_type_counts.to_dict()

                # Date range
                summary['date_range'] = {
                    'earliest': combined_signals['signal_date'].min().strftime('%Y-%m-%d'),
                    'latest': combined_signals['signal_date'].max().strftime('%Y-%m-%d')
                }

            return summary

        except Exception as e:
            logger.error(f"Error creating signal summary: {e}")
            return {}