"""
Google Finance Transformation Strategy
======================================

Handles Google Finance CSV format transformation.
Processes GF files with complex column structure and timestamp formats.
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from dateutil import parser
import logging
from .base_strategy import TransformationStrategy

logger = logging.getLogger(__name__)


class GFTransformationStrategy(TransformationStrategy):
    """
    Transformation strategy for Google Finance files.

    Expected format:
    - Filename: "^TICKER_daily.csv" (e.g., "^DJUSFB_daily.csv")
    - Columns: Date,Open,,Date,High,,Date,Low,,Date,Close,,Date,Volume
    - Column positions: [0,1,4,7,10,13] for Date,Open,High,Low,Close,Volume
    - Date format: "Tue Jan 02 2024 11:00:00 GMT-0500 (Eastern Standard Time)"
    """

    def __init__(self):
        super().__init__()

        # Regex patterns for ticker extraction
        self.ticker_patterns = [
            # Standard GF pattern: "^TICKER_daily.csv"
            r'^([A-Z0-9\^_\-\.]+)_daily\.csv$',

            # Alternative patterns
            r'^([A-Z0-9\^_\-\.]+)_weekly\.csv$',
            r'^([A-Z0-9\^_\-\.]+)_monthly\.csv$',

            # Fallback pattern: "TICKER.csv"
            r'^([A-Z0-9\^_\-\.]+)\.csv$'
        ]

        # Column positions for data extraction
        self.column_positions = {
            'Date': 0,     # First Date column
            'Open': 1,     # Open value
            'High': 4,     # High value
            'Low': 7,      # Low value
            'Close': 10,   # Close value
            'Volume': 13   # Volume value
        }

        # Expected header pattern (for validation)
        self.expected_header_pattern = r'Date,Open,.*,Date,High,.*,Date,Low,.*,Date,Close,.*,Date,Volume'

    def extract_ticker(self, filename: str) -> Optional[str]:
        """
        Extract ticker from Google Finance filename patterns.

        Args:
            filename: GF filename (e.g., "^DJUSFB_daily.csv")

        Returns:
            Extracted ticker (e.g., "^DJUSFB") or None
        """
        self.log_transformation("Extracting ticker", f"filename: {filename}")

        try:
            for pattern in self.ticker_patterns:
                match = re.match(pattern, filename, re.IGNORECASE)
                if match:
                    ticker = match.group(1).upper()
                    self.log_transformation("Ticker extracted", f"{filename} -> {ticker}")
                    return ticker

            logger.warning(f"[GF] No ticker pattern matched for filename: {filename}")
            return None

        except Exception as e:
            logger.error(f"[GF] Error extracting ticker from {filename}: {e}")
            return None

    def validate_format(self, df: pd.DataFrame) -> Dict:
        """
        Validate Google Finance format requirements.

        Args:
            df: Source DataFrame

        Returns:
            Validation result
        """
        result = {'is_valid': True, 'errors': [], 'warnings': []}

        try:
            # Check minimum column count
            if len(df.columns) < 14:
                result['is_valid'] = False
                result['errors'].append(f"Expected at least 14 columns, got {len(df.columns)}")
                return result

            # Check for empty DataFrame
            if len(df) == 0:
                result['is_valid'] = False
                result['errors'].append("DataFrame is empty")
                return result

            # Validate column positions exist
            for col_name, position in self.column_positions.items():
                if position >= len(df.columns):
                    result['is_valid'] = False
                    result['errors'].append(f"Missing column at position {position} for {col_name}")

            # Check if Date column contains valid timestamps
            if result['is_valid']:
                try:
                    date_sample = df.iloc[0, self.column_positions['Date']]
                    if pd.isna(date_sample):
                        result['warnings'].append("Date column contains NaN values")
                    else:
                        # Try to parse a sample date
                        self._parse_gf_date(str(date_sample))
                except Exception as e:
                    result['warnings'].append(f"Date parsing issues detected: {e}")

            # Check numeric columns
            for col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                position = self.column_positions[col_name]
                try:
                    sample_value = df.iloc[0, position]
                    if not pd.isna(sample_value):
                        float(sample_value)
                except Exception as e:
                    result['warnings'].append(f"Non-numeric value in {col_name} column: {e}")

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Validation error: {e}")

        return result

    def standardize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize Google Finance DataFrame to standard format.

        Args:
            df: GF DataFrame with complex column structure

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
            # Validate input format
            validation_result = self.validate_format(df)
            if not validation_result['is_valid']:
                processing_info['errors'].extend(validation_result['errors'])
                return df, processing_info

            processing_info['warnings'].extend(validation_result['warnings'])

            # Extract relevant columns
            extraction_result = self._extract_gf_columns(df)
            if not extraction_result['success']:
                processing_info['errors'].extend(extraction_result['errors'])
                return df, processing_info

            standardized_df = extraction_result['dataframe']
            processing_info['transformations'].extend(extraction_result['transformations'])

            # Process dates
            date_result = self._process_dates(standardized_df)
            standardized_df = date_result['dataframe']
            processing_info['transformations'].extend(date_result['transformations'])
            processing_info['warnings'].extend(date_result['warnings'])

            # Validate numeric columns
            numeric_result = self._validate_numeric_columns(standardized_df)
            standardized_df = numeric_result['dataframe']
            processing_info['transformations'].extend(numeric_result['transformations'])
            processing_info['warnings'].extend(numeric_result['warnings'])

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

    def _extract_gf_columns(self, df: pd.DataFrame) -> Dict:
        """Extract relevant columns from GF format."""
        result = {'success': True, 'dataframe': None, 'transformations': [], 'errors': []}

        try:
            # Extract columns by position
            extracted_data = {}

            for col_name, position in self.column_positions.items():
                if position < len(df.columns):
                    extracted_data[col_name] = df.iloc[:, position]
                    result['transformations'].append(f"Extracted {col_name} from column {position}")
                else:
                    result['errors'].append(f"Column position {position} not available for {col_name}")
                    result['success'] = False

            if result['success']:
                result['dataframe'] = pd.DataFrame(extracted_data)
                result['transformations'].append("Created standardized DataFrame from extracted columns")

        except Exception as e:
            result['success'] = False
            result['errors'].append(f"Column extraction error: {e}")

        return result

    def _parse_gf_date(self, date_str: str) -> str:
        """
        Parse Google Finance date format to standard YYYY-MM-DD.

        Args:
            date_str: GF date string

        Returns:
            Standardized date string
        """
        try:
            # Remove timezone info and extra text that confuses the parser
            # From: "Tue Jan 02 2024 11:00:00 GMT-0500 (Eastern Standard Time)"
            # To: "Tue Jan 02 2024"

            # Extract just the date part before time
            date_parts = date_str.split(' ')
            if len(date_parts) >= 4:
                # Take first 4 parts: "Tue Jan 02 2024"
                clean_date = ' '.join(date_parts[:4])
                parsed_date = parser.parse(clean_date)
                return parsed_date.strftime('%Y-%m-%d')

            # If that fails, try with dateutil's fuzzy parsing
            parsed_date = parser.parse(date_str, fuzzy=True)
            return parsed_date.strftime('%Y-%m-%d')

        except Exception as e:
            logger.warning(f"[GF] Failed to parse date '{date_str}': {e}")
            # Final fallback: try regex extraction
            date_match = re.search(r'(\w{3} \w{3} \d{1,2} \d{4})', date_str)
            if date_match:
                try:
                    fallback_date = parser.parse(date_match.group(1))
                    return fallback_date.strftime('%Y-%m-%d')
                except:
                    pass

            return date_str  # Return original if all parsing fails

    def _process_dates(self, df: pd.DataFrame) -> Dict:
        """Process and standardize date column."""
        result = {'dataframe': df.copy(), 'transformations': [], 'warnings': []}

        try:
            if 'Date' in df.columns:
                processed_dates = []
                failed_parses = 0

                for date_value in df['Date']:
                    if pd.isna(date_value):
                        processed_dates.append(None)
                        failed_parses += 1
                    else:
                        parsed_date = self._parse_gf_date(str(date_value))
                        processed_dates.append(parsed_date)
                        if parsed_date == str(date_value):  # Parsing failed
                            failed_parses += 1

                result['dataframe']['Date'] = processed_dates
                result['transformations'].append("Processed Date column from GF timestamp format")

                if failed_parses > 0:
                    result['warnings'].append(f"Failed to parse {failed_parses} date entries")

        except Exception as e:
            result['warnings'].append(f"Date processing error: {e}")

        return result

    def _validate_numeric_columns(self, df: pd.DataFrame) -> Dict:
        """Validate and clean numeric columns."""
        result = {'dataframe': df.copy(), 'transformations': [], 'warnings': []}

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        try:
            for col in numeric_columns:
                if col in df.columns:
                    try:
                        # Convert to numeric, coercing errors to NaN
                        original_count = len(result['dataframe'][col])
                        result['dataframe'][col] = pd.to_numeric(result['dataframe'][col], errors='coerce')

                        # Count NaN values
                        nan_count = result['dataframe'][col].isna().sum()
                        if nan_count > 0:
                            result['warnings'].append(f"Column '{col}': {nan_count}/{original_count} values converted to NaN")

                        result['transformations'].append(f"Converted '{col}' to numeric type")

                    except Exception as e:
                        result['warnings'].append(f"Failed to process numeric column '{col}': {e}")

        except Exception as e:
            result['warnings'].append(f"Numeric validation error: {e}")

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

            # Check for data presence
            if len(df) == 0:
                result['is_valid'] = False
                result['errors'].append("Final DataFrame is empty")

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Standard format validation error: {e}")

        return result

    def get_expected_columns(self) -> List[str]:
        """Get list of expected GF column positions."""
        return [f"Position {pos}" for pos in self.column_positions.values()]