"""
Moving Averages (MAs) Module
============================

Chart-focused Moving Average implementations for multi-panel chart generation.
Includes EMA, SMA, and various MA-based indicators.

Usage:
    from src.indicators.MAs import calculate_ema_for_chart, calculate_sma_for_chart

    ema_result = calculate_ema_for_chart(data, period=20)
    sma_result = calculate_sma_for_chart(data, period=50)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Union, Optional, List


def parse_ma_params(param_string: str) -> Dict[str, Union[int, str]]:
    """
    Parse Moving Average parameters from CSV string format.

    Args:
        param_string: String like "EMA(20)" or "SMA(50)" or "MA(20,SMA)"

    Returns:
        Dict with 'period', 'type'

    Examples:
        >>> parse_ma_params("EMA(20)")
        {'period': 20, 'type': 'EMA'}
        >>> parse_ma_params("SMA(50)")
        {'period': 50, 'type': 'SMA'}
    """
    # Remove spaces
    clean_string = param_string.replace(" ", "")

    # Extract MA type and parameters
    if clean_string.startswith('EMA'):
        match = re.match(r'EMA\((\d+)\)', clean_string)
        if match:
            return {'period': int(match.group(1)), 'type': 'EMA'}
    elif clean_string.startswith('SMA'):
        match = re.match(r'SMA\((\d+)\)', clean_string)
        if match:
            return {'period': int(match.group(1)), 'type': 'SMA'}
    elif clean_string.startswith('MA'):
        # Generic MA format: MA(period, type)
        match = re.match(r'MA\((\d+),(\w+)\)', clean_string)
        if match:
            return {'period': int(match.group(1)), 'type': match.group(2).upper()}
        # Simple MA format: MA(period) - defaults to SMA
        match = re.match(r'MA\((\d+)\)', clean_string)
        if match:
            return {'period': int(match.group(1)), 'type': 'SMA'}

    raise ValueError(f"Invalid MA parameter string: {param_string}")


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Price series
        period: EMA period

    Returns:
        EMA series
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"🔧 CORE EMA CALCULATION:")
        logger.info(f"   Input data type: {type(data)}")
        logger.info(f"   Input data length: {len(data)}")
        logger.info(f"   Period: {period}")

        if not isinstance(data, pd.Series):
            logger.error(f"❌ INVALID DATA TYPE: Expected pd.Series, got {type(data)}")
            raise TypeError(f"Expected pd.Series, got {type(data)}")

        if period <= 0:
            logger.error(f"❌ INVALID PERIOD: Period must be positive, got {period}")
            raise ValueError(f"Period must be positive, got {period}")

        if len(data) == 0:
            logger.error(f"❌ EMPTY DATA: Cannot calculate EMA on empty series")
            raise ValueError("Cannot calculate EMA on empty series")

        logger.info(f"   Calling data.ewm(span={period}, adjust=False).mean()")
        result = data.ewm(span=period, adjust=False).mean()

        logger.info(f"✅ EMA CORE CALCULATION SUCCESS:")
        logger.info(f"   Result type: {type(result)}")
        logger.info(f"   Result length: {len(result)}")
        logger.info(f"   Non-null values: {result.notna().sum()}")
        logger.info(f"   First non-null value: {result.dropna().iloc[0] if len(result.dropna()) > 0 else 'None'}")

        return result

    except Exception as e:
        logger.error(f"❌ CORE EMA CALCULATION FAILED:")
        logger.error(f"   Period: {period}")
        logger.error(f"   Data type: {type(data)}")
        logger.error(f"   Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        logger.error(f"   Exception: {e}")
        raise e


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        data: Price series
        period: SMA period

    Returns:
        SMA series
    """
    return data.rolling(window=period).mean()


def calculate_ema_for_chart(data: Union[pd.Series, pd.DataFrame],
                           period: int = 20,
                           price_column: str = 'Close') -> Dict[str, Union[pd.Series, Dict]]:
    """
    Calculate EMA for chart visualization.

    Args:
        data: Price data (Series or DataFrame)
        period: EMA period (default 20)
        price_column: Column to use if DataFrame (default 'Close')

    Returns:
        Dict containing:
        - 'ema': EMA values
        - 'price': Original price data
        - 'signals': Crossover signals
        - 'metadata': Chart metadata
    """
    import logging
    logger = logging.getLogger(__name__)

    # DEBUG: Function entry logging
    logger.info(f"🚀 FUNCTION CALL: calculate_ema_for_chart()")
    logger.info(f"   period: {period}")
    logger.info(f"   price_column: '{price_column}'")
    logger.info(f"   data type: {type(data)}")
    if hasattr(data, 'shape'):
        logger.info(f"   data shape: {data.shape}")
    if hasattr(data, 'columns'):
        logger.info(f"   data columns: {list(data.columns)}")
    # Handle different input types
    logger.info(f"📊 DATA TYPE HANDLING:")
    if isinstance(data, pd.DataFrame):
        logger.info(f"   Input is DataFrame")
        if price_column not in data.columns:
            available_cols = list(data.columns)
            old_price_column = price_column
            price_column = available_cols[0] if available_cols else 'Close'
            logger.warning(f"   Price column '{old_price_column}' not found, using '{price_column}'")
            logger.info(f"   Available columns: {available_cols}")
        else:
            logger.info(f"   Using price column: '{price_column}'")
        price_series = data[price_column]
        logger.info(f"   Extracted price series length: {len(price_series)}")
    else:
        logger.info(f"   Input is Series")
        price_series = data
        logger.info(f"   Price series length: {len(price_series)}")

    # Validate input
    logger.info(f"🔍 INPUT VALIDATION:")
    logger.info(f"   Required period: {period}")
    logger.info(f"   Available data length: {len(price_series)}")
    if len(price_series) < period:
        logger.error(f"❌ INSUFFICIENT DATA: need at least {period} periods, got {len(price_series)}")
        raise ValueError(f"Insufficient data: need at least {period} periods")
    logger.info(f"✅ Data validation passed")

    # Calculate EMA
    logger.info(f"📈 CALCULATING EMA:")
    logger.info(f"   Calling calculate_ema(price_series, period={period})")
    ema = calculate_ema(price_series, period)
    logger.info(f"   EMA calculation complete")
    logger.info(f"   EMA series length: {len(ema)}")
    logger.info(f"   EMA value range: {ema.min():.4f} to {ema.max():.4f}")
    logger.info(f"   EMA first 3 values: {ema.head(3).tolist()}")
    logger.info(f"   EMA last 3 values: {ema.tail(3).tolist()}")

    # Generate crossover signals
    logger.info(f"📊 GENERATING SIGNALS:")
    signals = pd.Series(0, index=price_series.index)
    signals[(price_series > ema) & (price_series.shift(1) <= ema.shift(1))] = 1  # Bullish crossover
    signals[(price_series < ema) & (price_series.shift(1) >= ema.shift(1))] = -1  # Bearish crossover

    bullish_signals = (signals == 1).sum()
    bearish_signals = (signals == -1).sum()
    logger.info(f"   Bullish crossovers: {bullish_signals}")
    logger.info(f"   Bearish crossovers: {bearish_signals}")

    # Chart metadata
    logger.info(f"🎨 BUILDING METADATA:")
    metadata = {
        'indicator': 'EMA',
        'parameters': f'({period})',
        'period': period,
        'chart_type': 'overlay',
        'color_scheme': {
            'ema': 'blue',
            'price': 'black',
            'signal_bullish': 'green',
            'signal_bearish': 'red'
        }
    }
    logger.info(f"   Metadata created: {metadata}")

    result = {
        'ema': ema,
        'price': price_series,
        'signals': signals,
        'metadata': metadata
    }

    logger.info(f"✅ EMA CALCULATION COMPLETE:")
    logger.info(f"   Result keys: {list(result.keys())}")
    for key, value in result.items():
        if key != 'metadata':
            logger.info(f"   {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
        else:
            logger.info(f"   {key}: {type(value)}")

    return result


def calculate_sma_for_chart(data: Union[pd.Series, pd.DataFrame],
                           period: int = 50,
                           price_column: str = 'Close') -> Dict[str, Union[pd.Series, Dict]]:
    """
    Calculate SMA for chart visualization.

    Args:
        data: Price data (Series or DataFrame)
        period: SMA period (default 50)
        price_column: Column to use if DataFrame (default 'Close')

    Returns:
        Dict containing:
        - 'sma': SMA values
        - 'price': Original price data
        - 'signals': Crossover signals
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
    if len(price_series) < period:
        raise ValueError(f"Insufficient data: need at least {period} periods")

    # Calculate SMA
    sma = calculate_sma(price_series, period)

    # Generate crossover signals
    signals = pd.Series(0, index=price_series.index)
    signals[(price_series > sma) & (price_series.shift(1) <= sma.shift(1))] = 1  # Bullish crossover
    signals[(price_series < sma) & (price_series.shift(1) >= sma.shift(1))] = -1  # Bearish crossover

    # Chart metadata
    metadata = {
        'indicator': 'SMA',
        'parameters': f'({period})',
        'period': period,
        'chart_type': 'overlay',
        'color_scheme': {
            'sma': 'red',
            'price': 'black',
            'signal_bullish': 'green',
            'signal_bearish': 'red'
        }
    }

    return {
        'sma': sma,
        'price': price_series,
        'signals': signals,
        'metadata': metadata
    }


def calculate_ma_ribbon(data: Union[pd.Series, pd.DataFrame],
                       periods: List[int] = [10, 20, 50, 100, 200],
                       ma_type: str = 'EMA',
                       price_column: str = 'Close') -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Calculate multiple moving averages for ribbon visualization.

    Args:
        data: Price data
        periods: List of MA periods
        ma_type: Type of MA ('EMA' or 'SMA')
        price_column: Column to use if DataFrame

    Returns:
        Dict with MA ribbon data and metadata
    """
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        if price_column not in data.columns:
            available_cols = list(data.columns)
            price_column = available_cols[0] if available_cols else 'Close'
        price_series = data[price_column]
    else:
        price_series = data

    # Calculate function based on type
    calc_func = calculate_ema if ma_type.upper() == 'EMA' else calculate_sma

    # Calculate all MAs
    ma_data = pd.DataFrame(index=price_series.index)
    for period in periods:
        if len(price_series) >= period:
            ma_data[f'{ma_type}{period}'] = calc_func(price_series, period)

    # Generate trend signals based on MA alignment
    trend_signal = pd.Series(0, index=price_series.index)
    if len(ma_data.columns) >= 2:
        # Bullish when shorter MA > longer MA
        bullish_alignment = True
        for i in range(len(periods) - 1):
            short_ma = f'{ma_type}{periods[i]}'
            long_ma = f'{ma_type}{periods[i+1]}'
            if short_ma in ma_data.columns and long_ma in ma_data.columns:
                bullish_alignment &= (ma_data[short_ma] > ma_data[long_ma])

        trend_signal[bullish_alignment] = 1
        trend_signal[~bullish_alignment] = -1

    metadata = {
        'indicator': f'{ma_type}_Ribbon',
        'parameters': f'({",".join(map(str, periods))})',
        'periods': periods,
        'ma_type': ma_type,
        'chart_type': 'overlay',
        'color_scheme': {
            'fast_ma': 'blue',
            'medium_ma': 'orange',
            'slow_ma': 'red',
            'trend_bullish': 'green',
            'trend_bearish': 'red'
        }
    }

    return {
        'ma_ribbon': ma_data,
        'price': price_series,
        'trend_signal': trend_signal,
        'metadata': metadata
    }


def calculate_ma_envelope(data: Union[pd.Series, pd.DataFrame],
                         period: int = 20,
                         envelope_pct: float = 2.0,
                         ma_type: str = 'SMA',
                         price_column: str = 'Close') -> Dict[str, Union[pd.Series, Dict]]:
    """
    Calculate Moving Average Envelope (Bollinger-style bands).

    Args:
        data: Price data
        period: MA period
        envelope_pct: Envelope percentage (e.g., 2.0 for ±2%)
        ma_type: Type of MA ('EMA' or 'SMA')
        price_column: Column to use if DataFrame

    Returns:
        Dict with envelope data and metadata
    """
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        if price_column not in data.columns:
            available_cols = list(data.columns)
            price_column = available_cols[0] if available_cols else 'Close'
        price_series = data[price_column]
    else:
        price_series = data

    # Calculate MA
    calc_func = calculate_ema if ma_type.upper() == 'EMA' else calculate_sma
    ma = calc_func(price_series, period)

    # Calculate envelope
    envelope_multiplier = envelope_pct / 100
    upper_band = ma * (1 + envelope_multiplier)
    lower_band = ma * (1 - envelope_multiplier)

    # Generate signals
    signals = pd.Series(0, index=price_series.index)
    signals[price_series <= lower_band] = 1  # Oversold
    signals[price_series >= upper_band] = -1  # Overbought

    metadata = {
        'indicator': f'{ma_type}_Envelope',
        'parameters': f'({period},{envelope_pct}%)',
        'period': period,
        'envelope_pct': envelope_pct,
        'ma_type': ma_type,
        'chart_type': 'overlay',
        'color_scheme': {
            'ma': 'blue',
            'upper_band': 'red',
            'lower_band': 'green',
            'signal_overbought': 'red',
            'signal_oversold': 'green'
        }
    }

    return {
        'ma': ma,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'price': price_series,
        'signals': signals,
        'metadata': metadata
    }


def validate_ma_input(data: Union[pd.Series, pd.DataFrame], period: int) -> bool:
    """
    Validate input data for MA calculation.

    Args:
        data: Input price data
        period: MA period

    Returns:
        True if valid, False otherwise
    """
    if data is None or len(data) == 0:
        return False

    if isinstance(data, pd.DataFrame):
        return len(data.columns) > 0 and len(data) >= period

    return len(data) >= period


def get_ma_chart_config(ma_type: str = 'EMA') -> Dict:
    """
    Get standard chart configuration for MA visualization.

    Args:
        ma_type: Type of MA ('EMA' or 'SMA')

    Returns:
        Chart configuration dictionary
    """
    return {
        'subplot': False,  # MAs typically overlaid on price chart
        'overlay': True,
        'legend': True,
        'indicators': {
            f'{ma_type.lower()}': {'type': 'line', 'width': 2},
            'price': {'type': 'candlestick'},
            'signals': {'type': 'scatter', 'size': 8}
        }
    }