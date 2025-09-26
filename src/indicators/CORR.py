"""
Correlation (CORR) Indicator Module
==================================

Chart-focused correlation implementation for multi-panel chart generation.
Calculates rolling correlation between two assets for relative movement analysis.

Usage:
    from src.indicators.CORR import calculate_corr_for_chart, parse_corr_params

    params = parse_corr_params("CORR(20)")
    result = calculate_corr_for_chart(data1, data2, **params)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Union, Optional

logger = logging.getLogger(__name__)


def parse_corr_params(param_string: str) -> Dict[str, Union[int, float]]:
    """
    Parse CORR parameters from CSV string format.

    Args:
        param_string: String like "CORR(20)" or "CORR()" or "CORR(60)"

    Returns:
        Dict with 'period'

    Examples:
        >>> parse_corr_params("CORR(20)")
        {'period': 20}
        >>> parse_corr_params("CORR()")
        {'period': 30}
        >>> parse_corr_params("CORR(60)")
        {'period': 60}
    """
    try:
        # Remove spaces and extract parameters
        clean_string = param_string.replace(" ", "")

        # Handle empty parameters (use defaults)
        if clean_string == "CORR()":
            return {'period': 30}

        # Handle single parameter: CORR(period)
        match = re.match(r'CORR\((\d+)\)', clean_string)
        if match:
            period = int(match.group(1))
            # Validate period range
            if period < 2:
                logger.warning(f"CORR period {period} too small, using minimum of 2")
                period = 2
            elif period > 252:
                logger.warning(f"CORR period {period} too large, using maximum of 252")
                period = 252

            return {'period': period}

        # Handle just "CORR" without parentheses
        if clean_string == "CORR":
            return {'period': 30}

    except Exception as e:
        logger.error(f"Error parsing CORR parameters '{param_string}': {e}")

    # Return defaults if parsing fails
    logger.warning(f"Could not parse CORR parameters '{param_string}', using defaults")
    return {'period': 30}


def calculate_corr_for_chart(data1: pd.DataFrame,
                            data2: pd.DataFrame,
                            period: int = 30,
                            price_column: str = 'Close') -> Dict[str, Union[pd.Series, Dict]]:
    """
    Calculate rolling correlation between two assets for chart visualization.

    Correlation measures the linear relationship between two variables.
    Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).

    Args:
        data1: Price data for first asset (DataFrame with OHLCV)
        data2: Price data for second asset (DataFrame with OHLCV)
        period: Rolling window period for correlation calculation (default 30)
        price_column: Column to use for correlation calculation (default 'Close')

    Returns:
        Dict containing:
        - 'correlation': Rolling correlation values
        - 'zero_line': Zero reference line (no correlation)
        - 'high_positive_line': +0.8 threshold line (strong positive)
        - 'low_negative_line': -0.8 threshold line (strong negative)
        - 'moderate_positive_line': +0.5 threshold line (moderate positive)
        - 'moderate_negative_line': -0.5 threshold line (moderate negative)
        - 'metadata': Chart metadata
    """
    try:
        # Validate input data
        if data1.empty or data2.empty:
            raise ValueError("Input data cannot be empty")

        if price_column not in data1.columns:
            available_cols = list(data1.columns)
            if not available_cols:
                raise ValueError("First dataset has no columns")
            price_column = available_cols[0]
            logger.info(f"Price column '{price_column}' not found in data1, using '{price_column}'")

        if price_column not in data2.columns:
            available_cols = list(data2.columns)
            if not available_cols:
                raise ValueError("Second dataset has no columns")
            alt_column = available_cols[0]
            logger.info(f"Price column '{price_column}' not found in data2, using '{alt_column}'")
            price_series2 = data2[alt_column]
        else:
            price_series2 = data2[price_column]

        price_series1 = data1[price_column]

        # Align data by index (inner join to get overlapping dates)
        aligned_data1, aligned_data2 = price_series1.align(price_series2, join='inner')

        if len(aligned_data1) < period + 1:
            raise ValueError(f"Insufficient overlapping data: need at least {period + 1} periods, got {len(aligned_data1)}")

        # Calculate returns for correlation
        returns1 = aligned_data1.pct_change()
        returns2 = aligned_data2.pct_change()

        # Drop first NaN value from pct_change
        returns1 = returns1.dropna()
        returns2 = returns2.dropna()

        # Ensure same index
        returns1, returns2 = returns1.align(returns2, join='inner')

        # Calculate rolling correlation
        correlation = returns1.rolling(window=period, min_periods=period).corr(returns2)

        # Create reference lines with same index as correlation
        correlation_index = correlation.dropna().index
        zero_line = pd.Series(0.0, index=correlation_index, name='zero_line')
        high_positive_line = pd.Series(0.8, index=correlation_index, name='high_positive')
        low_negative_line = pd.Series(-0.8, index=correlation_index, name='low_negative')
        moderate_positive_line = pd.Series(0.5, index=correlation_index, name='moderate_positive')
        moderate_negative_line = pd.Series(-0.5, index=correlation_index, name='moderate_negative')

        # Calculate some statistics for metadata
        valid_corr = correlation.dropna()
        avg_correlation = valid_corr.mean() if len(valid_corr) > 0 else 0.0
        max_correlation = valid_corr.max() if len(valid_corr) > 0 else 0.0
        min_correlation = valid_corr.min() if len(valid_corr) > 0 else 0.0

        # Determine correlation strength
        abs_avg = abs(avg_correlation)
        if abs_avg >= 0.8:
            strength = "Very Strong"
        elif abs_avg >= 0.6:
            strength = "Strong"
        elif abs_avg >= 0.4:
            strength = "Moderate"
        elif abs_avg >= 0.2:
            strength = "Weak"
        else:
            strength = "Very Weak"

        # Set correlation name
        correlation.name = 'correlation'

        result = {
            'correlation': correlation,
            'zero_line': zero_line,
            'high_positive_line': high_positive_line,
            'low_negative_line': low_negative_line,
            'moderate_positive_line': moderate_positive_line,
            'moderate_negative_line': moderate_negative_line,
            'metadata': {
                'indicator_type': 'correlation',
                'period': period,
                'price_column': price_column,
                'data_points': len(valid_corr),
                'avg_correlation': round(avg_correlation, 4),
                'max_correlation': round(max_correlation, 4),
                'min_correlation': round(min_correlation, 4),
                'correlation_strength': strength,
                'chart_type': 'oscillator',
                'y_axis_range': [-1.0, 1.0],
                'reference_lines': {
                    'zero': 0.0,
                    'high_positive': 0.8,
                    'moderate_positive': 0.5,
                    'moderate_negative': -0.5,
                    'low_negative': -0.8
                }
            }
        }

        logger.info(f"CORR calculation completed: {period}-period correlation, "
                   f"{len(valid_corr)} data points, avg correlation: {avg_correlation:.4f}")

        return result

    except Exception as e:
        logger.error(f"Error calculating CORR: {e}")
        # Return empty result with error indication
        return {
            'correlation': pd.Series(dtype=float, name='correlation'),
            'zero_line': pd.Series(dtype=float, name='zero_line'),
            'high_positive_line': pd.Series(dtype=float, name='high_positive'),
            'low_negative_line': pd.Series(dtype=float, name='low_negative'),
            'moderate_positive_line': pd.Series(dtype=float, name='moderate_positive'),
            'moderate_negative_line': pd.Series(dtype=float, name='moderate_negative'),
            'metadata': {
                'indicator_type': 'correlation',
                'period': period,
                'error': str(e),
                'chart_type': 'oscillator',
                'y_axis_range': [-1.0, 1.0]
            }
        }