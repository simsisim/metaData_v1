"""
Percentage Price Oscillator (PPO) Module
========================================

Chart-focused PPO implementation for multi-panel chart generation.
Supports CSV parameter parsing and chart-ready output.

Usage:
    from src.indicators.PPO import calculate_ppo_for_chart, parse_ppo_params

    params = parse_ppo_params("PPO(12,26,9)")
    result = calculate_ppo_for_chart(data, **params)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Union, Optional


def parse_ppo_params(param_string: str) -> Dict[str, int]:
    """
    Parse PPO parameters from CSV string format.

    Args:
        param_string: String like "PPO(12,26,9)" or "PPO(12, 26, 9)"

    Returns:
        Dict with 'fast_period', 'slow_period', 'signal_period'

    Example:
        >>> parse_ppo_params("PPO(12,26,9)")
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
    """
    # Remove spaces and extract parameters
    clean_string = param_string.replace(" ", "")

    # Extract parameters using regex
    match = re.match(r'PPO\((\d+),(\d+),(\d+)\)', clean_string)

    if not match:
        raise ValueError(f"Invalid PPO parameter string: {param_string}")

    fast, slow, signal = match.groups()

    return {
        'fast_period': int(fast),
        'slow_period': int(slow),
        'signal_period': int(signal)
    }


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_ppo_for_chart(data: Union[pd.Series, pd.DataFrame],
                           fast_period: int = 12,
                           slow_period: int = 26,
                           signal_period: int = 9,
                           price_column: str = 'Close') -> Dict[str, pd.Series]:
    """
    Calculate PPO (Percentage Price Oscillator) for chart visualization.

    PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100

    Args:
        data: Price data (Series or DataFrame)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        price_column: Column to use if DataFrame (default 'Close')

    Returns:
        Dict containing:
        - 'ppo': PPO line values
        - 'signal': Signal line values
        - 'histogram': PPO - Signal histogram
        - 'metadata': Chart metadata
    """
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        if price_column not in data.columns:
            available_cols = list(data.columns)
            price_column = available_cols[0] if available_cols else 'Close'
        price_series = data[price_column]
    else:
        price_series = data

    # Validate input
    if len(price_series) < max(fast_period, slow_period):
        raise ValueError(f"Insufficient data: need at least {max(fast_period, slow_period)} periods")

    # Calculate EMAs
    fast_ema = calculate_ema(price_series, fast_period)
    slow_ema = calculate_ema(price_series, slow_period)

    # Calculate PPO
    ppo = ((fast_ema - slow_ema) / slow_ema) * 100

    # Calculate Signal line
    signal = calculate_ema(ppo, signal_period)

    # Calculate Histogram
    histogram = ppo - signal

    # Chart metadata
    metadata = {
        'indicator': 'PPO',
        'parameters': f'({fast_period},{slow_period},{signal_period})',
        'fast_period': fast_period,
        'slow_period': slow_period,
        'signal_period': signal_period,
        'chart_type': 'oscillator',
        'zero_line': True,
        'color_scheme': {
            'ppo': 'blue',
            'signal': 'red',
            'histogram_positive': 'green',
            'histogram_negative': 'red'
        }
    }

    return {
        'ppo': ppo,
        'signal': signal,
        'histogram': histogram,
        'metadata': metadata
    }


def validate_ppo_input(data: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Validate input data for PPO calculation.

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


def get_ppo_chart_config() -> Dict:
    """
    Get standard chart configuration for PPO visualization.

    Returns:
        Chart configuration dictionary
    """
    return {
        'subplot': True,  # PPO typically shown in separate subplot
        'y_axis_label': 'PPO (%)',
        'zero_line': True,
        'grid': True,
        'legend': True,
        'height_ratio': 0.3,  # 30% of main chart height
        'indicators': {
            'ppo': {'type': 'line', 'color': 'blue', 'width': 1},
            'signal': {'type': 'line', 'color': 'red', 'width': 1},
            'histogram': {'type': 'bar', 'color_positive': 'green', 'color_negative': 'red'}
        }
    }