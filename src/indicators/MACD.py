"""
Moving Average Convergence Divergence (MACD) Module
==================================================

Chart-focused MACD implementation for multi-panel chart generation.
Supports CSV parameter parsing and chart-ready output.

MACD is a trend-following momentum indicator that shows the relationship
between two moving averages of a security's price.

Usage:
    from src.indicators.MACD import calculate_macd_for_chart, parse_macd_params

    params = parse_macd_params("MACD(12,26,9)")
    result = calculate_macd_for_chart(data, **params)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Union, Optional
from .indicators_calculation import calculate_macd


def parse_macd_params(param_string: str) -> Dict[str, int]:
    """
    Parse MACD parameters from CSV string format.

    Args:
        param_string: String like "MACD(12,26,9)" or "MACD(12, 26, 9)"

    Returns:
        Dict with 'fast_period', 'slow_period', 'signal_period'

    Example:
        >>> parse_macd_params("MACD(12,26,9)")
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}

        >>> parse_macd_params("MACD()")
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
    """
    # Remove spaces and extract parameters
    clean_string = param_string.replace(" ", "")

    # Handle empty parameters (use defaults)
    if clean_string == "MACD()":
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }

    # Extract parameters using regex
    match = re.match(r'MACD\((\d+),(\d+),(\d+)\)', clean_string)

    if not match:
        raise ValueError(f"Invalid MACD parameter string: {param_string}")

    fast, slow, signal = match.groups()

    return {
        'fast_period': int(fast),
        'slow_period': int(slow),
        'signal_period': int(signal)
    }


def calculate_macd_for_chart(data: Union[pd.Series, pd.DataFrame],
                            fast_period: int = 12,
                            slow_period: int = 26,
                            signal_period: int = 9,
                            price_column: str = 'Close') -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence) for chart visualization.

    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD Line - Signal Line

    Args:
        data: Price data (Series or DataFrame)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        price_column: Column to use if DataFrame (default 'Close')

    Returns:
        Dict containing:
        - 'macd': MACD line values
        - 'signal': Signal line values
        - 'histogram': MACD - Signal histogram
        - 'metadata': Chart metadata
    """
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        if price_column not in data.columns:
            available_cols = list(data.columns)
            if not available_cols:
                raise ValueError("DataFrame has no columns")
            price_column = available_cols[0]

        # Create a DataFrame with the required structure for calculate_macd
        price_data = pd.DataFrame({'Close': data[price_column]}, index=data.index)
    else:
        # Convert Series to DataFrame format expected by calculate_macd
        price_data = pd.DataFrame({'Close': data}, index=data.index)

    # Validate input
    if len(price_data) < max(fast_period, slow_period):
        raise ValueError(f"Insufficient data: need at least {max(fast_period, slow_period)} periods")

    try:
        # Use existing MACD calculation from indicators_calculation.py
        macd_result = calculate_macd(price_data, fast=fast_period, slow=slow_period, signal=signal_period)

        # Extract components
        macd_line = macd_result['MACD']
        signal_line = macd_result['MACD_Signal']
        histogram = macd_result['MACD_Hist']

    except Exception as e:
        # Fallback with NaN values if calculation fails
        index = price_data.index
        macd_line = pd.Series(np.nan, index=index, name='MACD')
        signal_line = pd.Series(np.nan, index=index, name='Signal')
        histogram = pd.Series(np.nan, index=index, name='Histogram')

    # Chart metadata
    metadata = {
        'indicator': 'MACD',
        'parameters': f'({fast_period},{slow_period},{signal_period})',
        'fast_period': fast_period,
        'slow_period': slow_period,
        'signal_period': signal_period,
        'chart_type': 'oscillator',
        'zero_line': True,
        'color_scheme': {
            'macd': 'blue',
            'signal': 'red',
            'histogram_positive': 'green',
            'histogram_negative': 'red'
        }
    }

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram,
        'metadata': metadata
    }


def validate_macd_input(data: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Validate input data for MACD calculation.

    Args:
        data: Input price data

    Returns:
        True if valid, False otherwise
    """
    if data is None or len(data) == 0:
        return False

    if isinstance(data, pd.DataFrame):
        return len(data.columns) > 0 and len(data) >= 26  # Minimum for default parameters

    return len(data) >= 26


def get_macd_chart_config() -> Dict:
    """
    Get standard chart configuration for MACD visualization.

    Returns:
        Chart configuration dictionary
    """
    return {
        'subplot': True,  # MACD typically shown in separate subplot
        'y_axis_label': 'MACD',
        'zero_line': True,
        'grid': True,
        'legend': True,
        'height_ratio': 0.3,  # 30% of main chart height
        'indicators': {
            'macd': {'type': 'line', 'color': 'blue', 'width': 1},
            'signal': {'type': 'line', 'color': 'red', 'width': 1},
            'histogram': {'type': 'bar', 'color_positive': 'green', 'color_negative': 'red'}
        }
    }