"""
Base Transformation Strategy
============================

Abstract base class defining the interface for origin-specific transformation strategies.
All strategies must implement ticker extraction, column standardization, and validation.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TransformationStrategy(ABC):
    """
    Abstract base class for origin-specific transformation strategies.

    Each data origin (TW, GF, etc.) implements this interface to handle
    their specific file formats, column structures, and naming conventions.
    """

    def __init__(self):
        self.origin = self.__class__.__name__.replace('TransformationStrategy', '').replace('Strategy', '')
        self.processing_stats = {
            'files_processed': 0,
            'successful': 0,
            'failed': 0,
            'warnings': [],
            'errors': []
        }

    @abstractmethod
    def extract_ticker(self, filename: str) -> Optional[str]:
        """
        Extract ticker symbol from filename.

        Args:
            filename: Source filename

        Returns:
            Extracted ticker symbol or None if extraction fails
        """
        pass

    @abstractmethod
    def standardize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize DataFrame columns to standard format.

        Args:
            df: Source DataFrame

        Returns:
            Tuple of (standardized_df, processing_info)
        """
        pass

    @abstractmethod
    def validate_format(self, df: pd.DataFrame) -> Dict:
        """
        Validate source DataFrame format.

        Args:
            df: Source DataFrame

        Returns:
            Validation result with is_valid, errors, warnings
        """
        pass

    @abstractmethod
    def get_expected_columns(self) -> List[str]:
        """
        Get list of expected columns in source format.

        Returns:
            List of expected column names
        """
        pass

    def get_standard_columns(self) -> List[str]:
        """
        Get list of standard output columns.

        Returns:
            List of standard column names
        """
        return ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    def update_stats(self, success: bool, warning: str = None, error: str = None):
        """Update processing statistics."""
        self.processing_stats['files_processed'] += 1
        if success:
            self.processing_stats['successful'] += 1
        else:
            self.processing_stats['failed'] += 1

        if warning:
            self.processing_stats['warnings'].append(warning)
        if error:
            self.processing_stats['errors'].append(error)

    def get_processing_stats(self) -> Dict:
        """Get current processing statistics."""
        return self.processing_stats.copy()

    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'files_processed': 0,
            'successful': 0,
            'failed': 0,
            'warnings': [],
            'errors': []
        }

    def get_origin(self) -> str:
        """Get the origin identifier for this strategy."""
        return self.origin

    def log_transformation(self, action: str, details: str = ""):
        """Log transformation action."""
        logger.debug(f"[{self.origin}] {action}: {details}")