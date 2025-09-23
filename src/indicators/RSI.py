"""
Relative Strength Index (RSI) Module
====================================

Chart-focused RSI implementation for multi-panel chart generation.
Supports CSV parameter parsing and chart-ready output.

Usage:
    from src.indicators.RSI import calculate_rsi_for_chart, parse_rsi_params

    params = parse_rsi_params("RSI(14, 70, 30)")
    result = calculate_rsi_for_chart(data, **params)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Union, Optional


def parse_rsi_params(param_string: str) -> Dict[str, Union[int, float]]:
    """
    Parse RSI parameters from CSV string format.

    Args:
        param_string: String like "RSI(14)" or "RSI(14, 70, 30)" or "RSI(12, 6, 9)"

    Returns:
        Dict with 'period', 'overbought', 'oversold'

    Examples:
        >>> parse_rsi_params("RSI(14)")
        {'period': 14, 'overbought': 70, 'oversold': 30}
        >>> parse_rsi_params("RSI(12, 6, 9)")
        {'period': 12, 'overbought': 6, 'oversold': 9}
    """
    # Remove spaces and extract parameters
    clean_string = param_string.replace(" ", "")

    # Try different parameter formats
    # Format 1: RSI(period, overbought, oversold)
    match3 = re.match(r'RSI\((\d+),(\d+),(\d+)\)', clean_string)
    if match3:
        period, param2, param3 = match3.groups()
        return {
            'period': int(period),
            'overbought': float(param2),
            'oversold': float(param3)
        }

    # Format 2: RSI(period, overbought)
    match2 = re.match(r'RSI\((\d+),(\d+)\)', clean_string)
    if match2:
        period, overbought = match2.groups()
        return {
            'period': int(period),
            'overbought': float(overbought),
            'oversold': 100 - float(overbought)  # Symmetric
        }

    # Format 3: RSI(period)
    match1 = re.match(r'RSI\((\d+)\)', clean_string)
    if match1:
        period = match1.groups()[0]
        return {
            'period': int(period),
            'overbought': 70.0,
            'oversold': 30.0
        }

    raise ValueError(f"Invalid RSI parameter string: {param_string}")


def calculate_rsi_for_chart(data: Union[pd.Series, pd.DataFrame],
                           period: int = 14,
                           overbought: float = 70.0,
                           oversold: float = 30.0,
                           price_column: str = 'Close') -> Dict[str, Union[pd.Series, Dict]]:
    """
    Calculate RSI (Relative Strength Index) for chart visualization.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Args:
        data: Price data (Series or DataFrame)
        period: RSI calculation period (default 14)
        overbought: Overbought threshold (default 70)
        oversold: Oversold threshold (default 30)
        price_column: Column to use if DataFrame (default 'Close')

    Returns:
        Dict containing:
        - 'rsi': RSI values
        - 'overbought_line': Overbought threshold line
        - 'oversold_line': Oversold threshold line
        - 'signals': Buy/sell signals
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
    if len(price_series) < period + 1:
        raise ValueError(f"Insufficient data: need at least {period + 1} periods")

    # Calculate price changes
    delta = price_series.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses using Wilder's smoothing
    avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    # Create threshold lines
    overbought_line = pd.Series(overbought, index=rsi.index)
    oversold_line = pd.Series(oversold, index=rsi.index)

    # Generate signals
    signals = pd.Series(0, index=rsi.index)
    signals[rsi < oversold] = 1  # Buy signal
    signals[rsi > overbought] = -1  # Sell signal

    # Chart metadata
    metadata = {
        'indicator': 'RSI',
        'parameters': f'({period},{overbought},{oversold})',
        'period': period,
        'overbought': overbought,
        'oversold': oversold,
        'chart_type': 'oscillator',
        'range': [0, 100],
        'color_scheme': {
            'rsi': 'purple',
            'overbought': 'red',
            'oversold': 'green',
            'signal_buy': 'green',
            'signal_sell': 'red'
        }
    }

    return {
        'rsi': rsi,
        'overbought_line': overbought_line,
        'oversold_line': oversold_line,
        'signals': signals,
        'metadata': metadata
    }


def calculate_rsi_divergence(price_data: pd.Series, rsi_data: pd.Series,
                           lookback: int = 20) -> Dict[str, pd.Series]:
    """
    Detect RSI divergence patterns.

    Args:
        price_data: Price series
        rsi_data: RSI series
        lookback: Lookback period for divergence detection

    Returns:
        Dict with bullish and bearish divergence signals
    """
    # Find local highs and lows
    price_highs = price_data.rolling(window=lookback, center=True).max() == price_data
    price_lows = price_data.rolling(window=lookback, center=True).min() == price_data
    rsi_highs = rsi_data.rolling(window=lookback, center=True).max() == rsi_data
    rsi_lows = rsi_data.rolling(window=lookback, center=True).min() == rsi_data

    bullish_divergence = pd.Series(False, index=price_data.index)
    bearish_divergence = pd.Series(False, index=price_data.index)

    # Simple divergence detection (can be enhanced)
    for i in range(lookback, len(price_data) - lookback):
        # Bullish divergence: price making lower lows, RSI making higher lows
        if price_lows.iloc[i] and rsi_lows.iloc[i]:
            prev_price_low = price_data.iloc[i-lookback:i][price_lows.iloc[i-lookback:i]].iloc[-1] if any(price_lows.iloc[i-lookback:i]) else None
            prev_rsi_low = rsi_data.iloc[i-lookback:i][rsi_lows.iloc[i-lookback:i]].iloc[-1] if any(rsi_lows.iloc[i-lookback:i]) else None

            if prev_price_low is not None and prev_rsi_low is not None:
                if price_data.iloc[i] < prev_price_low and rsi_data.iloc[i] > prev_rsi_low:
                    bullish_divergence.iloc[i] = True

        # Bearish divergence: price making higher highs, RSI making lower highs
        if price_highs.iloc[i] and rsi_highs.iloc[i]:
            prev_price_high = price_data.iloc[i-lookback:i][price_highs.iloc[i-lookback:i]].iloc[-1] if any(price_highs.iloc[i-lookback:i]) else None
            prev_rsi_high = rsi_data.iloc[i-lookback:i][rsi_highs.iloc[i-lookback:i]].iloc[-1] if any(rsi_highs.iloc[i-lookback:i]) else None

            if prev_price_high is not None and prev_rsi_high is not None:
                if price_data.iloc[i] > prev_price_high and rsi_data.iloc[i] < prev_rsi_high:
                    bearish_divergence.iloc[i] = True

    return {
        'bullish_divergence': bullish_divergence,
        'bearish_divergence': bearish_divergence
    }


def validate_rsi_input(data: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Validate input data for RSI calculation.

    Args:
        data: Input price data

    Returns:
        True if valid, False otherwise
    """
    if data is None or len(data) == 0:
        return False

    if isinstance(data, pd.DataFrame):
        return len(data.columns) > 0 and len(data) >= 15  # Minimum for default parameters

    return len(data) >= 15


def get_rsi_chart_config() -> Dict:
    """
    Get standard chart configuration for RSI visualization.

    Returns:
        Chart configuration dictionary
    """
    return {
        'subplot': True,  # RSI typically shown in separate subplot
        'y_axis_label': 'RSI',
        'y_axis_range': [0, 100],
        'horizontal_lines': [30, 50, 70],
        'grid': True,
        'legend': True,
        'height_ratio': 0.25,  # 25% of main chart height
        'indicators': {
            'rsi': {'type': 'line', 'color': 'purple', 'width': 2},
            'overbought': {'type': 'line', 'color': 'red', 'style': 'dashed'},
            'oversold': {'type': 'line', 'color': 'green', 'style': 'dashed'},
            'signals': {'type': 'scatter', 'size': 8}
        }
    }