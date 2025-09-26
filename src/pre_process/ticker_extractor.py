"""
Ticker Extractor
================

Extracts ticker symbols from TradingView filename patterns.
Handles various filename formats and validates extracted tickers.
"""

import re
import os
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TickerExtractor:
    """
    Extracts ticker symbols from TradingView filename patterns.

    Expected patterns:
    - 'INDEX_CPCS, 1D.csv' -> 'INDEX_CPCS'
    - 'FRED_UMCSENT, 1M.csv' -> 'FRED_UMCSENT'
    - 'AAPL, 1W.csv' -> 'AAPL'
    """

    def __init__(self):
        # Regex patterns for ticker extraction
        self.patterns = [
            # Standard TradingView pattern: "TICKER, FREQUENCY.csv"
            r'^([A-Z0-9_\-\.]+),\s*[0-9]+[DWMY]\.csv$',

            # Alternative pattern: "TICKER_FREQUENCY.csv"
            r'^([A-Z0-9_\-\.]+)_[0-9]+[DWMY]\.csv$',

            # Fallback pattern: "TICKER.csv" (no frequency)
            r'^([A-Z0-9_\-\.]+)\.csv$'
        ]

    def extract_ticker_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract ticker from filename using pattern matching.

        Args:
            filename: The filename to extract ticker from

        Returns:
            Optional[str]: Extracted ticker or None if extraction fails
        """
        try:
            # Clean filename (remove path if present)
            clean_filename = os.path.basename(filename)

            # Try each pattern
            for pattern in self.patterns:
                match = re.match(pattern, clean_filename, re.IGNORECASE)
                if match:
                    ticker = match.group(1).upper()

                    # Validate extracted ticker
                    if self._validate_ticker(ticker):
                        logger.debug(f"Extracted ticker '{ticker}' from filename '{clean_filename}'")
                        return ticker
                    else:
                        logger.warning(f"Invalid ticker '{ticker}' extracted from '{clean_filename}'")

            logger.warning(f"Could not extract ticker from filename: {clean_filename}")
            return None

        except Exception as e:
            logger.error(f"Error extracting ticker from filename '{filename}': {e}")
            return None

    def extract_ticker_and_frequency(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract both ticker and frequency from filename.

        Args:
            filename: The filename to extract from

        Returns:
            Tuple[Optional[str], Optional[str]]: (ticker, frequency) or (None, None)
        """
        try:
            clean_filename = os.path.basename(filename)

            # Pattern for ticker and frequency: "TICKER, FREQUENCY.csv"
            match = re.match(r'^([A-Z0-9_\-\.]+),\s*([0-9]+[DWMY])\.csv$',
                           clean_filename, re.IGNORECASE)

            if match:
                ticker = match.group(1).upper()
                frequency = match.group(2).upper()

                if self._validate_ticker(ticker) and self._validate_frequency(frequency):
                    logger.debug(f"Extracted ticker '{ticker}' and frequency '{frequency}' from '{clean_filename}'")
                    return ticker, frequency

            # Try ticker extraction only
            ticker = self.extract_ticker_from_filename(filename)
            if ticker:
                return ticker, None

            return None, None

        except Exception as e:
            logger.error(f"Error extracting ticker and frequency from '{filename}': {e}")
            return None, None

    def _validate_ticker(self, ticker: str) -> bool:
        """
        Validate extracted ticker symbol.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            bool: True if ticker is valid
        """
        try:
            # Basic validation rules
            if not ticker:
                return False

            # Length check (1-20 characters)
            if len(ticker) < 1 or len(ticker) > 20:
                return False

            # Character validation (alphanumeric, underscore, dash, dot)
            if not re.match(r'^[A-Z0-9_\-\.]+$', ticker):
                return False

            # Cannot start with number
            if ticker[0].isdigit():
                return False

            return True

        except Exception:
            return False

    def _validate_frequency(self, frequency: str) -> bool:
        """
        Validate extracted frequency.

        Args:
            frequency: Frequency to validate (e.g., '1D', '1W', '1M')

        Returns:
            bool: True if frequency is valid
        """
        try:
            # Pattern: number + letter (D/W/M/Y)
            return bool(re.match(r'^[0-9]+[DWMY]$', frequency, re.IGNORECASE))
        except Exception:
            return False

    def extract_frequency_from_pattern(self, pattern: str) -> Optional[str]:
        """
        Extract frequency information from file pattern.

        Args:
            pattern: Pattern like '*1D*', '*1W*'

        Returns:
            Optional[str]: Extracted frequency or None
        """
        try:
            # Extract frequency from pattern like '*1D*', '*1W*'
            match = re.search(r'([0-9]+[DWMY])', pattern, re.IGNORECASE)
            if match:
                frequency = match.group(1).upper()
                return frequency if self._validate_frequency(frequency) else None
            return None

        except Exception as e:
            logger.error(f"Error extracting frequency from pattern '{pattern}': {e}")
            return None

    def get_target_filename(self, ticker: str, template: str = "{ticker}.csv") -> str:
        """
        Generate target filename using ticker and template.

        Args:
            ticker: Ticker symbol
            template: Filename template

        Returns:
            str: Generated filename
        """
        try:
            # Validate ticker
            if not self._validate_ticker(ticker):
                raise ValueError(f"Invalid ticker: {ticker}")

            # Replace template variables
            filename = template.replace('{ticker}', ticker)

            logger.debug(f"Generated target filename '{filename}' for ticker '{ticker}'")
            return filename

        except Exception as e:
            logger.error(f"Error generating target filename for ticker '{ticker}': {e}")
            return f"{ticker}.csv"  # Fallback

    def batch_extract_tickers(self, filenames: list) -> dict:
        """
        Extract tickers from a batch of filenames.

        Args:
            filenames: List of filenames

        Returns:
            dict: Mapping of filename to extracted ticker
        """
        results = {}
        for filename in filenames:
            ticker = self.extract_ticker_from_filename(filename)
            results[filename] = ticker

        logger.info(f"Batch extracted tickers from {len(filenames)} files, "
                   f"successful: {sum(1 for v in results.values() if v is not None)}")

        return results