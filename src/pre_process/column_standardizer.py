"""
Column Standardizer
==================

Converts TradingView CSV format to standardized format.
Handles column renaming, data validation, and format consistency.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ColumnStandardizer:
    """
    Standardizes TradingView CSV columns to match expected format.

    Transformations:
    - time → Date
    - open → Open
    - high → High
    - low → Low
    - close → Close
    """

    def __init__(self):
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

        # Expected columns in standard format
        self.standard_columns = ['Date', 'Open', 'High', 'Low', 'Close']

    def standardize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize TradingView DataFrame to standard format.

        Args:
            df: TradingView DataFrame

        Returns:
            Tuple[pd.DataFrame, Dict]: (standardized DataFrame, processing info)
        """
        processing_info = {
            'original_columns': list(df.columns),
            'original_rows': len(df),
            'transformations': [],
            'warnings': [],
            'errors': []
        }

        try:
            # Create a copy to avoid modifying original
            standardized_df = df.copy()

            # Validate input format
            validation_result = self._validate_tradingview_format(standardized_df)
            if not validation_result['is_valid']:
                processing_info['errors'].extend(validation_result['errors'])
                return standardized_df, processing_info

            # Rename columns
            rename_result = self._rename_columns(standardized_df)
            standardized_df = rename_result['dataframe']
            processing_info['transformations'].extend(rename_result['transformations'])

            # Validate data types and formats
            validation_result = self._validate_and_fix_data_types(standardized_df)
            standardized_df = validation_result['dataframe']
            processing_info['warnings'].extend(validation_result['warnings'])

            # Ensure column order
            standardized_df = self._ensure_column_order(standardized_df)

            # Final validation
            final_validation = self._validate_standard_format(standardized_df)
            if not final_validation['is_valid']:
                processing_info['errors'].extend(final_validation['errors'])

            processing_info['final_columns'] = list(standardized_df.columns)
            processing_info['final_rows'] = len(standardized_df)

            logger.debug(f"Standardized DataFrame: {processing_info['original_rows']} → "
                        f"{processing_info['final_rows']} rows")

            return standardized_df, processing_info

        except Exception as e:
            error_msg = f"Error standardizing DataFrame: {e}"
            logger.error(error_msg)
            processing_info['errors'].append(error_msg)
            return df, processing_info

    def _validate_tradingview_format(self, df: pd.DataFrame) -> Dict:
        """Validate TradingView format requirements."""
        result = {'is_valid': True, 'errors': [], 'warnings': []}

        try:
            # Check for required columns
            missing_columns = []
            for col in self.required_tv_columns:
                if col not in df.columns:
                    missing_columns.append(col)

            if missing_columns:
                result['is_valid'] = False
                result['errors'].append(f"Missing required TradingView columns: {missing_columns}")

            # Check for empty DataFrame
            if df.empty:
                result['is_valid'] = False
                result['errors'].append("DataFrame is empty")

            # Check for extra columns (warn but don't fail)
            extra_columns = [col for col in df.columns if col not in self.required_tv_columns]
            if extra_columns:
                result['warnings'].append(f"Extra columns will be ignored: {extra_columns}")

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Validation error: {e}")

        return result

    def _rename_columns(self, df: pd.DataFrame) -> Dict:
        """Rename columns from TradingView to standard format."""
        result = {'dataframe': df.copy(), 'transformations': []}

        try:
            # Apply column mapping
            columns_to_rename = {}
            for tv_col, std_col in self.column_mapping.items():
                if tv_col in df.columns:
                    columns_to_rename[tv_col] = std_col
                    result['transformations'].append(f"Renamed '{tv_col}' → '{std_col}'")

            if columns_to_rename:
                result['dataframe'] = result['dataframe'].rename(columns=columns_to_rename)

        except Exception as e:
            logger.error(f"Error renaming columns: {e}")

        return result

    def _validate_and_fix_data_types(self, df: pd.DataFrame) -> Dict:
        """Validate and fix data types in standardized DataFrame."""
        result = {'dataframe': df.copy(), 'warnings': []}

        try:
            # Convert Date column to datetime if needed
            if 'Date' in df.columns:
                try:
                    result['dataframe']['Date'] = pd.to_datetime(result['dataframe']['Date'])
                except Exception as e:
                    result['warnings'].append(f"Could not convert Date column to datetime: {e}")

            # Convert OHLC columns to numeric
            ohlc_columns = ['Open', 'High', 'Low', 'Close']
            for col in ohlc_columns:
                if col in df.columns:
                    try:
                        result['dataframe'][col] = pd.to_numeric(result['dataframe'][col], errors='coerce')

                        # Check for NaN values introduced by conversion
                        nan_count = result['dataframe'][col].isna().sum()
                        if nan_count > 0:
                            result['warnings'].append(f"Column '{col}' has {nan_count} NaN values after conversion")

                    except Exception as e:
                        result['warnings'].append(f"Could not convert column '{col}' to numeric: {e}")

        except Exception as e:
            result['warnings'].append(f"Error validating data types: {e}")

        return result

    def _ensure_column_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns are in the expected order."""
        try:
            # Get available standard columns in order
            available_columns = [col for col in self.standard_columns if col in df.columns]

            # Add any extra columns at the end
            extra_columns = [col for col in df.columns if col not in self.standard_columns]
            ordered_columns = available_columns + extra_columns

            return df[ordered_columns]

        except Exception as e:
            logger.error(f"Error ordering columns: {e}")
            return df

    def _validate_standard_format(self, df: pd.DataFrame) -> Dict:
        """Validate the final standardized format."""
        result = {'is_valid': True, 'errors': []}

        try:
            # Check required standard columns
            missing_std_columns = []
            for col in self.standard_columns:
                if col not in df.columns:
                    missing_std_columns.append(col)

            if missing_std_columns:
                result['is_valid'] = False
                result['errors'].append(f"Missing required standard columns: {missing_std_columns}")

            # Check data integrity
            if 'Date' in df.columns and 'Close' in df.columns:
                if df['Date'].isna().any():
                    result['errors'].append("Date column contains NaN values")

                if df['Close'].isna().any():
                    result['errors'].append("Close column contains NaN values")

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Standard format validation error: {e}")

        return result

    def get_column_mapping(self) -> Dict[str, str]:
        """Get the column mapping dictionary."""
        return self.column_mapping.copy()

    def get_required_columns(self) -> List[str]:
        """Get list of required TradingView columns."""
        return self.required_tv_columns.copy()

    def get_standard_columns(self) -> List[str]:
        """Get list of expected standard columns."""
        return self.standard_columns.copy()

    def is_tradingview_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in TradingView format."""
        return all(col in df.columns for col in self.required_tv_columns)

    def is_standard_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in standard format."""
        return all(col in df.columns for col in self.standard_columns)

    def get_transformation_summary(self, processing_info: Dict) -> str:
        """Get a summary of transformations performed."""
        summary = []

        if processing_info.get('transformations'):
            summary.append(f"Transformations: {len(processing_info['transformations'])}")

        if processing_info.get('warnings'):
            summary.append(f"Warnings: {len(processing_info['warnings'])}")

        if processing_info.get('errors'):
            summary.append(f"Errors: {len(processing_info['errors'])}")

        return " | ".join(summary) if summary else "No issues"