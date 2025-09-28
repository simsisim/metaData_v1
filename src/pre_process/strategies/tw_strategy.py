"""
TradingView Transformation Strategy
===================================

Handles TradingView CSV format transformation.
Extracts logic from existing TickerExtractor and ColumnStandardizer for TW files.
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
import logging
from .base_strategy import TransformationStrategy

logger = logging.getLogger(__name__)


class TWTransformationStrategy(TransformationStrategy):
    """
    Transformation strategy for TradingView files.

    Expected format:
    - Filename: "TICKER, FREQUENCY.csv" (e.g., "INDEX_CPCS, 1D.csv")
    - Columns: time,open,high,low,close
    - Date format: YYYY-MM-DD
    """

    def __init__(self):
        super().__init__()

        # Regex patterns for ticker extraction
        self.ticker_patterns = [
            # Standard TradingView pattern: "TICKER, FREQUENCY.csv"
            r'^([A-Z0-9_\-\.]+),\s*[0-9]+[DWMY]\.csv$',

            # Alternative pattern: "TICKER_FREQUENCY.csv"
            r'^([A-Z0-9_\-\.]+)_[0-9]+[DWMY]\.csv$',

            # Fallback pattern: "TICKER.csv" (no frequency)
            r'^([A-Z0-9_\-\.]+)\.csv$'
        ]

        # Column mapping: TradingView → Standard
        self.column_mapping = {
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        }

        # Required columns in TradingView format
        self.required_tv_columns = ['time', 'open', 'high', 'low', 'close']

    def extract_ticker(self, filename: str) -> Optional[str]:
        """
        Extract ticker from TradingView filename patterns.

        Args:
            filename: TradingView filename

        Returns:
            Extracted ticker or None
        """
        self.log_transformation("Extracting ticker", f"filename: {filename}")

        try:
            for pattern in self.ticker_patterns:
                match = re.match(pattern, filename, re.IGNORECASE)
                if match:
                    ticker = match.group(1).upper()
                    self.log_transformation("Ticker extracted", f"{filename} -> {ticker}")
                    return ticker

            logger.warning(f"[TW] No ticker pattern matched for filename: {filename}")
            return None

        except Exception as e:
            logger.error(f"[TW] Error extracting ticker from {filename}: {e}")
            return None

    def validate_format(self, df: pd.DataFrame) -> Dict:
        """
        Validate TradingView format requirements.

        Args:
            df: Source DataFrame

        Returns:
            Validation result
        """
        result = {'is_valid': True, 'errors': [], 'warnings': []}

        try:
            # Check for required columns
            df_columns = [col.lower() for col in df.columns]
            missing_columns = []

            for required_col in self.required_tv_columns:
                if required_col not in df_columns:
                    missing_columns.append(required_col)

            if missing_columns:
                result['is_valid'] = False
                result['errors'].append(f"Missing required TradingView columns: {missing_columns}")

            # Check for empty DataFrame
            if len(df) == 0:
                result['is_valid'] = False
                result['errors'].append("DataFrame is empty")

            # Check for valid data types
            if result['is_valid']:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_columns:
                        try:
                            pd.to_numeric(df[col], errors='raise')
                        except:
                            result['warnings'].append(f"Column '{col}' contains non-numeric values")

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Validation error: {e}")

        return result

    def standardize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize TradingView DataFrame to standard format.

        Args:
            df: TradingView DataFrame

        Returns:
            Tuple of (standardized_df, processing_info)
        """
        processing_info = {
            'original_columns': list(df.columns),
            'original_rows': len(df),
            'transformations': [],
            'warnings': [],
            'errors': []
        }

        try:
            standardized_df = df.copy()

            # Validate input format
            validation_result = self.validate_format(standardized_df)
            if not validation_result['is_valid']:
                processing_info['errors'].extend(validation_result['errors'])
                return df, processing_info

            # Rename columns
            rename_result = self._rename_columns(standardized_df)
            standardized_df = rename_result['dataframe']
            processing_info['transformations'].extend(rename_result['transformations'])

            # Validate and fix data types
            validation_result = self._validate_and_fix_data_types(standardized_df)
            standardized_df = validation_result['dataframe']
            processing_info['transformations'].extend(validation_result['transformations'])
            processing_info['warnings'].extend(validation_result['warnings'])

            # Add Volume column if missing (set to 0)
            if 'Volume' not in standardized_df.columns:
                standardized_df['Volume'] = 0
                processing_info['transformations'].append("Added Volume column (set to 0)")

            # Final validation
            final_validation = self._validate_standard_format(standardized_df)
            if not final_validation['is_valid']:
                processing_info['errors'].extend(final_validation['errors'])

            processing_info['final_columns'] = list(standardized_df.columns)
            processing_info['final_rows'] = len(standardized_df)

            return standardized_df, processing_info

        except Exception as e:
            processing_info['errors'].append(f"Standardization error: {e}")
            return df, processing_info

    def _rename_columns(self, df: pd.DataFrame) -> Dict:
        """Rename columns from TradingView to standard format."""
        result = {'dataframe': df.copy(), 'transformations': []}

        try:
            columns_to_rename = {}

            for tv_col, std_col in self.column_mapping.items():
                if tv_col in df.columns:
                    columns_to_rename[tv_col] = std_col
                    result['transformations'].append(f"Renamed '{tv_col}' → '{std_col}'")

            if columns_to_rename:
                result['dataframe'] = result['dataframe'].rename(columns=columns_to_rename)

        except Exception as e:
            logger.error(f"[TW] Error renaming columns: {e}")

        return result

    def _validate_and_fix_data_types(self, df: pd.DataFrame) -> Dict:
        """Validate and fix data types."""
        result = {'dataframe': df.copy(), 'transformations': [], 'warnings': []}

        try:
            # Convert numeric columns
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    try:
                        result['dataframe'][col] = pd.to_numeric(result['dataframe'][col], errors='coerce')
                        result['transformations'].append(f"Converted '{col}' to numeric")
                    except Exception as e:
                        result['warnings'].append(f"Failed to convert '{col}' to numeric: {e}")

            # Handle Date column
            if 'Date' in df.columns:
                try:
                    result['dataframe']['Date'] = pd.to_datetime(result['dataframe']['Date']).dt.strftime('%Y-%m-%d')
                    result['transformations'].append("Standardized Date format to YYYY-MM-DD")
                except Exception as e:
                    result['warnings'].append(f"Failed to standardize Date format: {e}")

        except Exception as e:
            result['warnings'].append(f"Data type validation error: {e}")

        return result

    def _validate_standard_format(self, df: pd.DataFrame) -> Dict:
        """Validate the final standardized format."""
        result = {'is_valid': True, 'errors': []}

        try:
            expected_columns = self.get_standard_columns()
            missing_columns = []

            for col in expected_columns:
                if col not in df.columns:
                    missing_columns.append(col)

            if missing_columns:
                result['is_valid'] = False
                result['errors'].append(f"Missing standard columns: {missing_columns}")

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Standard format validation error: {e}")

        return result

    def get_expected_columns(self) -> List[str]:
        """Get list of expected TradingView columns."""
        return self.required_tv_columns.copy()