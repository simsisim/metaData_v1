"""
Accumulation Distribution Line (ADLINE) Module
==============================================

Chart-focused ADLINE implementation for multi-panel chart generation.
Supports CSV parameter parsing and chart-ready output.

The Accumulation/Distribution Line (ADL) is a volume-based technical indicator
that measures the cumulative flow of money into and out of a security.

Formula: AD = AD + ((Close - Low) - (High - Close)) / (High - Low) * Volume

Usage:
    from src.indicators.ADLINE import calculate_adline_for_chart, parse_adline_params

    params = parse_adline_params("ADLINE(50)")
    result = calculate_adline_for_chart(data, **params)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Union, Optional


def parse_adline_params(param_string: str) -> Dict[str, int]:
    """
    Parse ADLINE parameters from CSV string format.

    Args:
        param_string: String like "ADLINE(50)" or "ADLINE()"

    Returns:
        Dict with 'lookback_period'

    Examples:
        >>> parse_adline_params("ADLINE(50)")
        {'lookback_period': 50}

        >>> parse_adline_params("ADLINE()")
        {'lookback_period': 50}
    """
    # Remove spaces and extract parameters
    clean_string = param_string.replace(" ", "")

    # Handle empty parameters (use defaults)
    if clean_string == "ADLINE()":
        return {'lookback_period': 50}

    # Extract parameters using regex
    match = re.match(r'ADLINE\((\d+)\)', clean_string)

    if not match:
        raise ValueError(f"Invalid ADLINE parameter string: {param_string}")

    lookback_period = match.groups()[0]

    return {'lookback_period': int(lookback_period)}


def calculate_adline_for_chart(data: Union[pd.Series, pd.DataFrame],
                              lookback_period: int = 50,
                              price_column: str = 'Close') -> Dict[str, Union[pd.Series, Dict]]:
    """
    Calculate ADLINE (Accumulation Distribution Line) for chart visualization.

    Formula: AD = AD + ((Close - Low) - (High - Close)) / (High - Low) * Volume

    Args:
        data: OHLCV data (must be DataFrame with High, Low, Close, Volume)
        lookback_period: Period for trend analysis (default 50)
        price_column: Column to use for price (default 'Close')

    Returns:
        Dict containing:
        - 'adline': ADL cumulative values
        - 'mfm': Money Flow Multiplier values
        - 'mfv': Money Flow Volume values
        - 'trend': Trend analysis over lookback period
        - 'metadata': Chart metadata
    """
    # Validate input - ADLINE requires OHLCV DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("ADLINE requires DataFrame with OHLCV data")

    required_cols = ['High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for ADLINE: {missing_cols}")

    # Validate minimum data length
    if len(data) < lookback_period:
        raise ValueError(f"Insufficient data: need at least {lookback_period} periods")

    try:
        # Calculate Money Flow Multiplier (MFM)
        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        high_low_diff = data['High'] - data['Low']

        # Avoid division by zero for doji/narrow range bars
        high_low_diff = high_low_diff.replace(0, np.nan)

        close_position = ((data['Close'] - data['Low']) - (data['High'] - data['Close']))
        mfm = close_position / high_low_diff

        # Handle NaN values (doji bars) - assume neutral (0)
        mfm = mfm.fillna(0)

        # Calculate Money Flow Volume (MFV)
        # MFV = MFM * Volume
        mfv = mfm * data['Volume']

        # Calculate ADL (cumulative sum of MFV)
        adl = mfv.cumsum()

        # Calculate trend analysis over lookback period
        trend = pd.Series(0, index=data.index)
        for i in range(lookback_period, len(data)):
            recent_adl = adl.iloc[i-lookback_period:i+1]
            slope = np.polyfit(range(len(recent_adl)), recent_adl, 1)[0]

            if slope > 0:
                trend.iloc[i] = 1  # Accumulation trend
            elif slope < 0:
                trend.iloc[i] = -1  # Distribution trend
            # else: neutral (0)

    except Exception as e:
        # Fallback with NaN values if calculation fails
        index = data.index
        adl = pd.Series(np.nan, index=index, name='ADLINE')
        mfm = pd.Series(np.nan, index=index, name='MFM')
        mfv = pd.Series(np.nan, index=index, name='MFV')
        trend = pd.Series(0, index=index, name='Trend')

    # Chart metadata
    metadata = {
        'indicator': 'ADLINE',
        'parameters': f'({lookback_period})',
        'lookback_period': lookback_period,
        'chart_type': 'overlay',  # ADLINE typically overlays the main price chart
        'zero_line': False,
        'color_scheme': {
            'adline': 'orange',
            'trend_accumulation': 'green',
            'trend_distribution': 'red',
            'trend_neutral': 'gray'
        }
    }

    return {
        'adline': adl,
        'mfm': mfm,
        'mfv': mfv,
        'trend': trend,
        'metadata': metadata
    }


def calculate_adline_divergence(price_data: pd.Series, adl_data: pd.Series,
                               lookback: int = 20) -> Dict[str, pd.Series]:
    """
    Detect ADLINE divergence patterns.

    Args:
        price_data: Price series
        adl_data: ADLINE series
        lookback: Lookback period for divergence detection

    Returns:
        Dict with bullish and bearish divergence signals
    """
    # Find local highs and lows
    price_highs = price_data.rolling(window=lookback, center=True).max() == price_data
    price_lows = price_data.rolling(window=lookback, center=True).min() == price_data
    adl_highs = adl_data.rolling(window=lookback, center=True).max() == adl_data
    adl_lows = adl_data.rolling(window=lookback, center=True).min() == adl_data

    bullish_divergence = pd.Series(False, index=price_data.index)
    bearish_divergence = pd.Series(False, index=price_data.index)

    # Simple divergence detection
    for i in range(lookback, len(price_data) - lookback):
        # Bullish divergence: price making lower lows, ADLINE making higher lows
        if price_lows.iloc[i] and adl_lows.iloc[i]:
            prev_price_low = price_data.iloc[i-lookback:i][price_lows.iloc[i-lookback:i]].iloc[-1] if any(price_lows.iloc[i-lookback:i]) else None
            prev_adl_low = adl_data.iloc[i-lookback:i][adl_lows.iloc[i-lookback:i]].iloc[-1] if any(adl_lows.iloc[i-lookback:i]) else None

            if prev_price_low is not None and prev_adl_low is not None:
                if price_data.iloc[i] < prev_price_low and adl_data.iloc[i] > prev_adl_low:
                    bullish_divergence.iloc[i] = True

        # Bearish divergence: price making higher highs, ADLINE making lower highs
        if price_highs.iloc[i] and adl_highs.iloc[i]:
            prev_price_high = price_data.iloc[i-lookback:i][price_highs.iloc[i-lookback:i]].iloc[-1] if any(price_highs.iloc[i-lookback:i]) else None
            prev_adl_high = adl_data.iloc[i-lookback:i][adl_highs.iloc[i-lookback:i]].iloc[-1] if any(adl_highs.iloc[i-lookback:i]) else None

            if prev_price_high is not None and prev_adl_high is not None:
                if price_data.iloc[i] > prev_price_high and adl_data.iloc[i] < prev_adl_high:
                    bearish_divergence.iloc[i] = True

    return {
        'bullish_divergence': bullish_divergence,
        'bearish_divergence': bearish_divergence
    }


def validate_adline_input(data: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Validate input data for ADLINE calculation.

    Args:
        data: Input OHLCV data

    Returns:
        True if valid, False otherwise
    """
    if data is None or len(data) == 0:
        return False

    if not isinstance(data, pd.DataFrame):
        return False  # ADLINE requires DataFrame with OHLCV

    required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        return False

    return len(data) >= 50  # Minimum for default parameters


def get_adline_chart_config() -> Dict:
    """
    Get standard chart configuration for ADLINE visualization.

    Returns:
        Chart configuration dictionary
    """
    return {
        'subplot': False,  # ADLINE typically overlays main chart
        'y_axis_label': 'ADLINE',
        'overlay': True,
        'separate_y_axis': True,  # Use separate y-axis scale
        'grid': True,
        'legend': True,
        'indicators': {
            'adline': {'type': 'line', 'color': 'orange', 'width': 2},
            'trend_accumulation': {'type': 'background', 'color': 'green', 'alpha': 0.2},
            'trend_distribution': {'type': 'background', 'color': 'red', 'alpha': 0.2}
        }
    }