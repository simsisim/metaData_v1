"""
RS/PER Data Loader Module
========================

Loads and validates RS (Relative Strength) and PER (Percentile) data files
for comprehensive multi-timeframe market analysis report generation.

This module handles:
- Loading RS files (IBD and MA methods) for stocks, sectors, industries
- Loading PER files with NASDAQ100 percentile rankings
- Data validation and merging across different calculation methods
- Dynamic date-based file discovery
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RSPERDataLoader:
    """
    Main data loader for RS and PER analysis files.
    Handles loading, validation, and merging of multi-timeframe data.
    """

    def __init__(self, config):
        """Initialize the data loader with configuration."""
        self.config = config
        self.rs_dir = config.directories['RESULTS_DIR'] / 'rs'
        self.per_dir = config.directories['RESULTS_DIR'] / 'per'

        # Define 9 timeframes for analysis
        self.timeframes = {
            '3d': 'daily_daily_daily_3d_rs_vs_QQQ',
            '5d': 'daily_daily_daily_5d_rs_vs_QQQ',
            '7d': 'daily_daily_weekly_7d_rs_vs_QQQ',
            '14d': 'daily_daily_weekly_14d_rs_vs_QQQ',
            '22d': 'daily_daily_monthly_22d_rs_vs_QQQ',
            '44d': 'daily_daily_monthly_44d_rs_vs_QQQ',
            '66d': 'daily_daily_quarterly_66d_rs_vs_QQQ',
            '132d': 'daily_daily_quarterly_132d_rs_vs_QQQ',
            '252d': 'daily_daily_yearly_252d_rs_vs_QQQ'
        }

        # Percentile column mapping
        self.percentile_cols = {
            '3d': 'daily_daily_daily_3d_rs_vs_QQQ_per_NASDAQ100',
            '5d': 'daily_daily_daily_5d_rs_vs_QQQ_per_NASDAQ100',
            '7d': 'daily_daily_weekly_7d_rs_vs_QQQ_per_NASDAQ100',
            '14d': 'daily_daily_weekly_14d_rs_vs_QQQ_per_NASDAQ100',
            '22d': 'daily_daily_monthly_22d_rs_vs_QQQ_per_NASDAQ100',
            '44d': 'daily_daily_monthly_44d_rs_vs_QQQ_per_NASDAQ100',
            '66d': 'daily_daily_quarterly_66d_rs_vs_QQQ_per_NASDAQ100',
            '132d': 'daily_daily_quarterly_132d_rs_vs_QQQ_per_NASDAQ100',
            '252d': 'daily_daily_yearly_252d_rs_vs_QQQ_per_NASDAQ100'
        }

    def load_daily_data(self, date_str: Optional[str] = None) -> Dict:
        """
        Load all RS and PER data for a specific date.

        Args:
            date_str: Date string in YYYYMMDD format. If None, uses most recent files.

        Returns:
            Dictionary containing all loaded and validated data
        """
        if date_str is None:
            date_str = self._get_latest_date()

        logger.info(f"Loading RS/PER data for date: {date_str}")

        try:
            # Load RS files
            rs_data = self._load_rs_files(date_str)

            # Load PER files
            per_data = self._load_per_files(date_str)

            # Validate data structure
            self._validate_data_structure(rs_data, per_data)

            # Merge and deduplicate data
            merged_data = self._merge_and_deduplicate(rs_data, per_data)

            logger.info(f"Successfully loaded RS/PER data for {date_str}")
            return {
                'date': date_str,
                'rs_data': rs_data,
                'per_data': per_data,
                'merged_data': merged_data,
                'timeframes': list(self.timeframes.keys()),
                'validation_status': 'success'
            }

        except Exception as e:
            logger.error(f"Failed to load RS/PER data for {date_str}: {e}")
            raise

    def _load_rs_files(self, date_str: str) -> Dict[str, pd.DataFrame]:
        """Load all RS files for the given date."""
        rs_data = {}

        # RS file patterns
        rs_patterns = {
            'rs_ibd_stocks': f'rs_QQQ_ibd_stocks_daily_2-5_{date_str}.csv',
            'rs_ma_stocks': f'rs_QQQ_ma_stocks_daily_2-5_{date_str}.csv',
            'rs_ibd_sectors': f'rs_QQQ_ibd_sectors_daily_2-5_{date_str}.csv',
            'rs_ibd_industries': f'rs_QQQ_ibd_industries_daily_2-5_{date_str}.csv'
        }

        for key, filename in rs_patterns.items():
            file_path = self.rs_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    rs_data[key] = df
                    logger.debug(f"Loaded {key}: {len(df)} records")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.warning(f"RS file not found: {filename}")

        return rs_data

    def _load_per_files(self, date_str: str) -> Dict[str, pd.DataFrame]:
        """Load all PER files for the given date."""
        per_data = {}

        # PER file patterns
        per_patterns = {
            'per_stocks': f'per_QQQ_NASDAQ100_ibd_stocks_daily_2-5_{date_str}.csv',
            'per_sectors': f'per_QQQ_NASDAQ100_ibd_sectors_daily_2-5_{date_str}.csv',
            'per_industries': f'per_QQQ_NASDAQ100_ibd_industries_daily_2-5_{date_str}.csv'
        }

        for key, filename in per_patterns.items():
            file_path = self.per_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    per_data[key] = df
                    logger.debug(f"Loaded {key}: {len(df)} records")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.warning(f"PER file not found: {filename}")

        return per_data

    def _validate_data_structure(self, rs_data: Dict, per_data: Dict) -> None:
        """Validate that required columns exist for all timeframes."""
        validation_errors = []

        # Check RS data for required timeframe columns
        for dataset_name, df in rs_data.items():
            if df is not None:
                missing_cols = []
                for tf, col_name in self.timeframes.items():
                    if col_name not in df.columns:
                        missing_cols.append(f"{tf}({col_name})")

                if missing_cols:
                    validation_errors.append(f"{dataset_name} missing timeframes: {', '.join(missing_cols)}")

        # Check PER data for required percentile columns
        for dataset_name, df in per_data.items():
            if df is not None:
                missing_cols = []
                for tf, col_name in self.percentile_cols.items():
                    if col_name not in df.columns:
                        missing_cols.append(f"{tf}({col_name})")

                if missing_cols:
                    validation_errors.append(f"{dataset_name} missing percentile columns: {', '.join(missing_cols)}")

        if validation_errors:
            error_msg = "Data validation failed:\n" + "\n".join(validation_errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Data structure validation passed")

    def _merge_and_deduplicate(self, rs_data: Dict, per_data: Dict) -> Dict:
        """Merge RS and PER data, handling multiple calculation methods."""
        merged_data = {}

        # Merge stocks data (combine IBD and MA methods)
        if 'rs_ibd_stocks' in rs_data and 'per_stocks' in per_data:
            stocks_merged = self._merge_stocks_data(
                rs_data.get('rs_ibd_stocks'),
                rs_data.get('rs_ma_stocks'),
                per_data['per_stocks']
            )
            merged_data['stocks'] = stocks_merged

        # Merge sectors data
        if 'rs_ibd_sectors' in rs_data and 'per_sectors' in per_data:
            sectors_merged = self._merge_entity_data(
                rs_data['rs_ibd_sectors'],
                per_data['per_sectors'],
                'sectors'
            )
            merged_data['sectors'] = sectors_merged

        # Merge industries data
        if 'rs_ibd_industries' in rs_data and 'per_industries' in per_data:
            industries_merged = self._merge_entity_data(
                rs_data['rs_ibd_industries'],
                per_data['per_industries'],
                'industries'
            )
            merged_data['industries'] = industries_merged

        return merged_data

    def _merge_stocks_data(self, rs_ibd_df: pd.DataFrame, rs_ma_df: Optional[pd.DataFrame],
                          per_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stock data from IBD RS, MA RS, and PER sources."""
        # Start with PER data as base (has percentile rankings)
        merged_df = per_df.copy()

        # Add IBD RS data
        if rs_ibd_df is not None:
            # Select relevant RS columns
            rs_cols = ['ticker'] + list(self.timeframes.values())
            rs_subset = rs_ibd_df[rs_cols].copy()

            # Rename columns to avoid conflicts
            rs_rename = {col: f"{col}_ibd" for col in self.timeframes.values()}
            rs_subset = rs_subset.rename(columns=rs_rename)

            # Merge with PER data
            merged_df = merged_df.merge(rs_subset, on='ticker', how='left', suffixes=('', '_ibd'))

        # Add MA RS data if available
        if rs_ma_df is not None:
            rs_cols = ['ticker'] + list(self.timeframes.values())
            rs_subset = rs_ma_df[rs_cols].copy()

            # Rename columns to avoid conflicts
            rs_rename = {col: f"{col}_ma" for col in self.timeframes.values()}
            rs_subset = rs_subset.rename(columns=rs_rename)

            # Merge with existing data
            merged_df = merged_df.merge(rs_subset, on='ticker', how='left', suffixes=('', '_ma'))

        # Calculate composite RS values (prefer IBD, fallback to MA)
        for tf, col_name in self.timeframes.items():
            ibd_col = f"{col_name}_ibd"
            ma_col = f"{col_name}_ma"
            composite_col = f"rs_{tf}"

            if ibd_col in merged_df.columns:
                merged_df[composite_col] = merged_df[ibd_col]
                if ma_col in merged_df.columns:
                    # Fill missing IBD values with MA values
                    merged_df[composite_col] = merged_df[composite_col].fillna(merged_df[ma_col])
            elif ma_col in merged_df.columns:
                merged_df[composite_col] = merged_df[ma_col]

        return merged_df

    def _merge_entity_data(self, rs_df: pd.DataFrame, per_df: pd.DataFrame,
                          entity_type: str) -> pd.DataFrame:
        """Merge RS and PER data for sectors or industries."""
        # Start with PER data as base
        merged_df = per_df.copy()

        # Select relevant RS columns
        rs_cols = ['ticker'] + list(self.timeframes.values())
        rs_subset = rs_df[rs_cols].copy()

        # Merge with PER data
        merged_df = merged_df.merge(rs_subset, on='ticker', how='left', suffixes=('', '_rs'))

        # Add simplified RS column names
        for tf, col_name in self.timeframes.items():
            if col_name in merged_df.columns:
                merged_df[f"rs_{tf}"] = merged_df[col_name]

        return merged_df

    def _get_latest_date(self) -> str:
        """Find the most recent date with available RS/PER files."""
        latest_date = None

        # Check RS directory for recent files
        if self.rs_dir.exists():
            rs_files = list(self.rs_dir.glob('rs_QQQ_*_daily_*.csv'))
            for file in rs_files:
                # Extract date from filename (last 8 digits before .csv)
                filename = file.stem
                date_match = filename.split('_')[-1]
                if len(date_match) == 8 and date_match.isdigit():
                    if latest_date is None or date_match > latest_date:
                        latest_date = date_match

        # Check PER directory for recent files
        if self.per_dir.exists():
            per_files = list(self.per_dir.glob('per_QQQ_*_daily_*.csv'))
            for file in per_files:
                filename = file.stem
                date_match = filename.split('_')[-1]
                if len(date_match) == 8 and date_match.isdigit():
                    if latest_date is None or date_match > latest_date:
                        latest_date = date_match

        if latest_date is None:
            # Fallback to current date
            latest_date = datetime.now().strftime('%Y%m%d')
            logger.warning(f"No data files found, using current date: {latest_date}")

        return latest_date

    def get_available_dates(self, lookback_days: int = 30) -> List[str]:
        """Get list of available dates with RS/PER data."""
        available_dates = set()

        # Check RS files
        if self.rs_dir.exists():
            rs_files = list(self.rs_dir.glob('rs_QQQ_*_daily_*.csv'))
            for file in rs_files:
                filename = file.stem
                date_match = filename.split('_')[-1]
                if len(date_match) == 8 and date_match.isdigit():
                    available_dates.add(date_match)

        # Check PER files
        if self.per_dir.exists():
            per_files = list(self.per_dir.glob('per_QQQ_*_daily_*.csv'))
            for file in per_files:
                filename = file.stem
                date_match = filename.split('_')[-1]
                if len(date_match) == 8 and date_match.isdigit():
                    available_dates.add(date_match)

        # Sort and return recent dates
        sorted_dates = sorted(list(available_dates), reverse=True)
        return sorted_dates[:lookback_days]

    def validate_timeframe_data(self, df: pd.DataFrame, required_timeframes: List[str] = None) -> Dict:
        """Validate data completeness for specified timeframes."""
        if required_timeframes is None:
            required_timeframes = list(self.timeframes.keys())

        validation_results = {
            'total_records': len(df),
            'timeframe_coverage': {},
            'missing_data_summary': {},
            'quality_score': 0.0
        }

        total_possible = len(df) * len(required_timeframes)
        total_valid = 0

        for tf in required_timeframes:
            rs_col = f"rs_{tf}"
            per_col = self.percentile_cols.get(tf)

            if rs_col in df.columns:
                valid_rs = df[rs_col].notna().sum()
                validation_results['timeframe_coverage'][f"{tf}_rs"] = {
                    'valid_count': valid_rs,
                    'missing_count': len(df) - valid_rs,
                    'coverage_pct': (valid_rs / len(df)) * 100
                }
                total_valid += valid_rs

            if per_col and per_col in df.columns:
                valid_per = df[per_col].notna().sum()
                validation_results['timeframe_coverage'][f"{tf}_per"] = {
                    'valid_count': valid_per,
                    'missing_count': len(df) - valid_per,
                    'coverage_pct': (valid_per / len(df)) * 100
                }

        # Calculate overall quality score
        if total_possible > 0:
            validation_results['quality_score'] = (total_valid / total_possible) * 100

        return validation_results