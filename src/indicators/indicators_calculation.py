"""
Technical Indicators Calculation Module
======================================

Comprehensive technical indicator calculations based on TradingView-validated implementations.
All indicators have been validated against TradingView Pine Script for accuracy.

Includes:
- Kurutoga Histogram (multi-timeframe divergence indicator)
- TSI (True Strength Index)
- MACD (Moving Average Convergence Divergence)
- MFI (Money Flow Index)
- COG (Center of Gravity)
- Momentum, RSI (Relative Strength Index)
- MA Crosses (Moving Average crossovers)
- Easy Trade (trend color system)
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional, Union, List


def calculate_kurutoga(data: pd.DataFrame, length: int = 14, source: str = 'Close') -> pd.DataFrame:
    """
    Calculate the Kurutoga histogram with current, 2x, and 4x timeframes.
    
    Kurutoga measures price divergence from the midpoint of high-low range
    across multiple timeframes for trend and momentum analysis.
    
    Args:
        data: DataFrame with OHLC data
        length: The lookback period (default 14)
        source: The price source ('Close', 'Open', 'High', 'Low')
        
    Returns:
        DataFrame with Kurutoga histogram values for current, 2x, and 4x timeframes
    """
    def _calculate_kurutoga_single(data: pd.DataFrame, length: int, source: str) -> pd.Series:
        # Calculate the 50% level (midpoint of range)
        midpoint = (data['High'].rolling(window=length).max() + 
                   data['Low'].rolling(window=length).min()) / 2
        
        # Calculate price divergence from the 50% level
        divergence = data[source] - midpoint
        
        return divergence
    
    # Calculate Kurutoga for different timeframes
    kurutoga_current = _calculate_kurutoga_single(data, length, source)
    kurutoga_2x = _calculate_kurutoga_single(data, length * 2, source)
    kurutoga_4x = _calculate_kurutoga_single(data, length * 4, source)
    
    return pd.DataFrame({
        'Kurutoga_Current': kurutoga_current,
        'Kurutoga_2x': kurutoga_2x,
        'Kurutoga_4x': kurutoga_4x
    }, index=data.index)


def calculate_tsi(data: pd.DataFrame, fast: int = 13, slow: int = 25, signal: int = 13) -> pd.DataFrame:
    """
    Calculate True Strength Index (TSI) with signal line.
    
    TSI is a momentum oscillator that uses moving averages of price changes
    to filter out price noise and identify trend changes.
    
    Args:
        data: DataFrame with OHLC data
        fast: Fast EMA period (default 13)
        slow: Slow EMA period (default 25)
        signal: Signal line EMA period (default 13)
        
    Returns:
        DataFrame with TSI and TSI_Signal columns
    """
    try:
        tsi_result = ta.tsi(data['Close'], fast=fast, slow=slow, signal=signal)
        
        if tsi_result is not None and len(tsi_result.columns) >= 2:
            return pd.DataFrame({
                'TSI': tsi_result.iloc[:, 0],
                'TSI_Signal': tsi_result.iloc[:, 1]
            }, index=data.index)
        else:
            # Fallback to NaN values if calculation fails
            return pd.DataFrame({
                'TSI': pd.Series(np.nan, index=data.index),
                'TSI_Signal': pd.Series(np.nan, index=data.index)
            }, index=data.index)
    except Exception:
        return pd.DataFrame({
            'TSI': pd.Series(np.nan, index=data.index),
            'TSI_Signal': pd.Series(np.nan, index=data.index)
        }, index=data.index)


def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    
    Args:
        data: DataFrame with OHLC data
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26) 
        signal: Signal line EMA period (default 9)
        
    Returns:
        DataFrame with MACD, MACD_Signal, and MACD_Hist columns
    """
    try:
        macd_result = ta.macd(data['Close'], fast=fast, slow=slow, signal=signal)
        
        if macd_result is not None:
            macd_col = f'MACD_{fast}_{slow}_{signal}'
            signal_col = f'MACDs_{fast}_{slow}_{signal}'
            hist_col = f'MACDh_{fast}_{slow}_{signal}'
            
            return pd.DataFrame({
                'MACD': macd_result[macd_col],
                'MACD_Signal': macd_result[signal_col],
                'MACD_Hist': macd_result[hist_col]
            }, index=data.index)
        else:
            return pd.DataFrame({
                'MACD': pd.Series(np.nan, index=data.index),
                'MACD_Signal': pd.Series(np.nan, index=data.index),
                'MACD_Hist': pd.Series(np.nan, index=data.index)
            }, index=data.index)
    except Exception:
        return pd.DataFrame({
            'MACD': pd.Series(np.nan, index=data.index),
            'MACD_Signal': pd.Series(np.nan, index=data.index),
            'MACD_Hist': pd.Series(np.nan, index=data.index)
        }, index=data.index)


def calculate_mfi(data: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """
    Calculate Money Flow Index (MFI) - TradingView validated implementation.
    
    MFI is a volume-weighted version of RSI that incorporates both price and volume
    to measure buying and selling pressure.
    
    Args:
        data: DataFrame with OHLCV data
        length: The lookback period (default 14)
        
    Returns:
        DataFrame with MFI and related calculations
    """
    df = data.copy()
    
    # Calculate typical price (HLC/3)
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate raw money flow
    df['raw_money_flow'] = df['typical_price'] * df['Volume']
    
    # Identify positive and negative money flow
    df['positive_flow'] = (df['typical_price'] > df['typical_price'].shift(1)).astype(int) * df['raw_money_flow']
    df['negative_flow'] = (df['typical_price'] < df['typical_price'].shift(1)).astype(int) * df['raw_money_flow']
    
    # Calculate positive and negative money flow sums
    df['positive_mf'] = df['positive_flow'].rolling(window=length).sum()
    df['negative_mf'] = df['negative_flow'].rolling(window=length).sum()
    
    # Calculate money flow ratio and MFI
    df['money_flow_ratio'] = df['positive_mf'] / df['negative_mf']
    df['MFI'] = 100 - (100 / (1 + df['money_flow_ratio']))
    
    return pd.DataFrame({
        'MFI': df['MFI'],
        'Positive_MF': df['positive_mf'],
        'Negative_MF': df['negative_mf'],
        'Money_Flow_Ratio': df['money_flow_ratio']
    }, index=data.index)


def calculate_mfi_signal(mfi_data: pd.DataFrame, signal_period: int = 9) -> pd.Series:
    """
    Calculate MFI signal line using EMA smoothing.
    
    Args:
        mfi_data: DataFrame containing MFI column
        signal_period: EMA period for signal line (default 9)
        
    Returns:
        Series with MFI signal values
    """
    return ta.ema(mfi_data['MFI'], length=signal_period)


def calculate_cog(data: pd.DataFrame, length: int = 9, source: str = 'Close') -> pd.DataFrame:
    """
    Calculate Center of Gravity (COG) indicator with ALMA and LSMA smoothing.
    
    COG is an oscillator that identifies turning points with minimal lag,
    based on the mathematical center of gravity concept.
    
    Args:
        data: DataFrame with OHLC data
        length: The lookback period (default 9)
        source: The price source (default 'Close')
        
    Returns:
        DataFrame with COG, COG_ALMA, and COG_LSMA columns
    """
    def _cog_single(series: pd.Series, length: int) -> float:
        """Calculate single COG value for a price series"""
        if len(series) < length:
            return np.nan
            
        num = sum([(i+1) * series.iloc[-i-1] for i in range(length)])
        den = sum([series.iloc[-i-1] for i in range(length)])
        
        return -1 * (num / den - (length + 1) / 2) if den != 0 else 0
    
    # Calculate COG for entire series
    cog_values = data[source].rolling(window=length).apply(
        lambda x: _cog_single(x, length), raw=False
    )
    
    # Calculate smoothed versions
    cog_alma = ta.alma(cog_values, length=3, offset=0.85, sigma=6)
    cog_lsma = ta.linreg(cog_values, length=200)
    
    return pd.DataFrame({
        'COG': cog_values,
        'COG_ALMA': cog_alma,
        'COG_LSMA': cog_lsma
    }, index=data.index)


def calculate_momentum(data: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """
    Calculate Momentum indicator with normalization.
    
    Args:
        data: DataFrame with OHLC data
        length: The lookback period (default 20)
        
    Returns:
        DataFrame with MOM and MOM_Norm columns
    """
    mom = ta.mom(data['Close'], length=length)
    mom_norm = _normalize_series(mom, 0, 100)
    
    return pd.DataFrame({
        'MOM': mom,
        'MOM_Norm': mom_norm
    }, index=data.index)


def calculate_rsi(data: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: DataFrame with OHLC data
        length: The lookback period (default 14)
        
    Returns:
        Series with RSI values
    """
    try:
        rsi_result = ta.rsi(data['Close'], length=length)
        return rsi_result if rsi_result is not None else pd.Series(np.nan, index=data.index)
    except Exception:
        return pd.Series(np.nan, index=data.index)


def calculate_ma_crosses(data: pd.DataFrame, fast_period: int = 50, slow_period: int = 200) -> pd.DataFrame:
    """
    Calculate moving average crossovers (Golden Cross and Death Cross).
    
    Args:
        data: DataFrame with OHLC data
        fast_period: Fast MA period (default 50)
        slow_period: Slow MA period (default 200)
        
    Returns:
        DataFrame with MA values and crossover signals
    """
    fast_ma = ta.sma(data['Close'], length=fast_period)
    slow_ma = ta.sma(data['Close'], length=slow_period)
    
    # Calculate crossovers
    golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    
    return pd.DataFrame({
        f'MA_{fast_period}': fast_ma,
        f'MA_{slow_period}': slow_ma,
        'Golden_Cross': golden_cross,
        'Death_Cross': death_cross
    }, index=data.index)


def calculate_easy_trade(data: pd.DataFrame, fast_length: int = 12, slow_length: int = 26, 
                        signal_length: int = 9, fast_period: int = 50, slow_period: int = 200,
                        multiscale: int = 10000, scale_factor: float = 100/30) -> pd.DataFrame:
    """
    Calculate Easy Entry/Exit Trend Colors indicator.
    
    Combines MACD with Bollinger Bands and MA crossovers for trend identification.
    
    Args:
        data: DataFrame with OHLC data
        fast_length: MACD fast EMA period (default 12)
        slow_length: MACD slow EMA period (default 26)
        signal_length: MACD signal period (default 9)
        fast_period: Fast SMA period for Death/Golden Cross (default 50)
        slow_period: Slow SMA period for Death/Golden Cross (default 200)
        multiscale: Scaling factor for ALTS (default 10000)
        scale_factor: Vertical space scaling factor (default 100/30)
        
    Returns:
        DataFrame with Easy Trade indicator columns
    """
    result_data = data.copy()
    
    # MACD calculation
    result_data['FastMA'] = ta.ema(data['Close'], length=fast_length)
    result_data['SlowMA'] = ta.ema(data['Close'], length=slow_length)
    result_data['MACD'] = result_data['FastMA'] - result_data['SlowMA']
    result_data['MACD_Signal'] = ta.ema(result_data['MACD'], length=signal_length)
    result_data['MACD_Hist'] = result_data['MACD'] - result_data['MACD_Signal']
    
    # Scale MACD
    result_data['MACD_Scaled'] = result_data['MACD'] * multiscale * scale_factor
    
    # Death and Golden Cross
    result_data['FastMA_Cross'] = ta.sma(data['Close'], length=fast_period)
    result_data['SlowMA_Cross'] = ta.sma(data['Close'], length=slow_period)
    
    # Calculate crossovers
    result_data['Golden_Cross'] = ((result_data['FastMA_Cross'] > result_data['SlowMA_Cross']) & 
                                  (result_data['FastMA_Cross'].shift(1) <= result_data['SlowMA_Cross'].shift(1)))
    result_data['Death_Cross'] = ((result_data['FastMA_Cross'] < result_data['SlowMA_Cross']) & 
                                 (result_data['FastMA_Cross'].shift(1) >= result_data['SlowMA_Cross'].shift(1)))
    
    # Bollinger Bands on MACD
    bb = ta.bbands(result_data['MACD'], length=10, std=1)
    result_data['BB_Upper'] = bb['BBU_10_1.0']
    result_data['BB_Lower'] = bb['BBL_10_1.0']
    
    # Trend Colors
    result_data['Trend'] = np.where(result_data['MACD'] >= result_data['BB_Upper'], 'Bullish',
                                   np.where(result_data['MACD'] <= result_data['BB_Lower'], 'Bearish', 'Neutral'))
    
    # Crossover signals
    result_data['Crossover_Bear'] = ((result_data['MACD'] < result_data['BB_Upper']) & 
                                    (result_data['MACD'].shift(1) >= result_data['BB_Upper'].shift(1)))
    result_data['Crossover_Bull'] = ((result_data['MACD'] > result_data['BB_Upper']) & 
                                    (result_data['MACD'].shift(1) <= result_data['BB_Upper'].shift(1)))
    
    return result_data[['MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Scaled', 'FastMA_Cross', 
                       'SlowMA_Cross', 'Golden_Cross', 'Death_Cross', 'BB_Upper', 'BB_Lower',
                       'Trend', 'Crossover_Bear', 'Crossover_Bull']]


def calculate_all_indicators(data: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate all technical indicators for a single ticker.
    
    Args:
        data: DataFrame with OHLCV data
        config: Optional configuration dict with indicator parameters
        
    Returns:
        DataFrame with all calculated indicators
    """
    if config is None:
        config = _get_default_config()
    
    result = data.copy()
    
    # Calculate each indicator if enabled
    indicators_data = []
    
    if config.get('kurutoga', {}).get('enabled', True):
        kurutoga_params = config.get('kurutoga', {})
        kurutoga_result = calculate_kurutoga(
            data, 
            length=kurutoga_params.get('length', 14),
            source=kurutoga_params.get('source', 'Close')
        )
        indicators_data.append(kurutoga_result)
    
    if config.get('tsi', {}).get('enabled', True):
        tsi_params = config.get('tsi', {})
        tsi_result = calculate_tsi(
            data,
            fast=tsi_params.get('fast', 13),
            slow=tsi_params.get('slow', 25), 
            signal=tsi_params.get('signal', 13)
        )
        indicators_data.append(tsi_result)
    
    if config.get('macd', {}).get('enabled', True):
        macd_params = config.get('macd', {})
        macd_result = calculate_macd(
            data,
            fast=macd_params.get('fast', 12),
            slow=macd_params.get('slow', 26),
            signal=macd_params.get('signal', 9)
        )
        indicators_data.append(macd_result)
    
    if config.get('mfi', {}).get('enabled', True):
        mfi_params = config.get('mfi', {})
        mfi_result = calculate_mfi(
            data,
            length=mfi_params.get('length', 14)
        )
        if config.get('mfi', {}).get('include_signal', True):
            mfi_signal = calculate_mfi_signal(mfi_result, signal_period=mfi_params.get('signal_period', 9))
            mfi_result['MFI_Signal'] = mfi_signal
        indicators_data.append(mfi_result)
    
    if config.get('cog', {}).get('enabled', True):
        cog_params = config.get('cog', {})
        cog_result = calculate_cog(
            data,
            length=cog_params.get('length', 9),
            source=cog_params.get('source', 'Close')
        )
        indicators_data.append(cog_result)
    
    if config.get('momentum', {}).get('enabled', True):
        mom_params = config.get('momentum', {})
        mom_result = calculate_momentum(
            data,
            length=mom_params.get('length', 20)
        )
        indicators_data.append(mom_result)
    
    if config.get('rsi', {}).get('enabled', True):
        rsi_params = config.get('rsi', {})
        rsi_result = calculate_rsi(
            data,
            length=rsi_params.get('length', 14)
        )
        result['RSI'] = rsi_result
    
    if config.get('ma_crosses', {}).get('enabled', True):
        ma_params = config.get('ma_crosses', {})
        ma_result = calculate_ma_crosses(
            data,
            fast_period=ma_params.get('fast_period', 50),
            slow_period=ma_params.get('slow_period', 200)
        )
        indicators_data.append(ma_result)
    
    if config.get('easy_trade', {}).get('enabled', True):
        easy_params = config.get('easy_trade', {})
        easy_result = calculate_easy_trade(
            data,
            fast_length=easy_params.get('fast_length', 12),
            slow_length=easy_params.get('slow_length', 26),
            signal_length=easy_params.get('signal_length', 9),
            fast_period=easy_params.get('fast_period', 50),
            slow_period=easy_params.get('slow_period', 200)
        )
        indicators_data.append(easy_result)
    
    # Combine all indicators
    for indicator_df in indicators_data:
        result = pd.concat([result, indicator_df], axis=1)
    
    return result


def calculate_momentum_mfi_rsi_signals(data: pd.DataFrame, mom_length: int = 20, 
                                     mfi_length: int = 20, rsi_length: int = 20) -> pd.DataFrame:
    """
    Calculate combined MOM, MFI, RSI signals with bullish/bearish classifications.
    
    Args:
        data: DataFrame with OHLCV data
        mom_length: Momentum period (default 20)
        mfi_length: MFI period (default 20)
        rsi_length: RSI period (default 20)
        
    Returns:
        DataFrame with signal classifications
    """
    result = data.copy()
    
    # Calculate indicators
    mom_result = calculate_momentum(data, mom_length)
    mfi_result = calculate_mfi(data, mfi_length)
    rsi_result = calculate_rsi(data, rsi_length)
    
    result['MOM'] = mom_result['MOM']
    result['MOM_Norm'] = mom_result['MOM_Norm']
    result['MFI'] = mfi_result['MFI']
    result['RSI'] = rsi_result
    
    # Generate signals
    result['MOM_Signal'] = np.select(
        [result['MOM_Norm'] > 50, result['MOM_Norm'] < 50, result['MOM_Norm'] == 50],
        ['Bullish_Momentum', 'Bearish_Momentum', 'Neutral_Momentum']
    )
    result['MFI_Signal'] = np.select(
        [result['MFI'] > 50, result['MFI'] < 50, result['MFI'] == 50],
        ['Bullish_Money_Flow', 'Bearish_Money_Flow', 'Neutral_Money_Flow']
    )
    result['RSI_Signal'] = np.select(
        [result['RSI'] > 50, result['RSI'] < 50, result['RSI'] == 50],
        ['Bullish_RSI', 'Bearish_RSI', 'Neutral_RSI']
    )
    
    return result


def _normalize_series(src: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """
    Normalize a series to a specified range using rolling historical min/max.
    
    Args:
        src: Input series
        min_val: Target minimum value
        max_val: Target maximum value
        
    Returns:
        Normalized series
    """
    historic_min = np.inf
    historic_max = -np.inf
    
    result = np.zeros_like(src)
    
    for i in range(len(src)):
        value = src.iloc[i]
        if not np.isnan(value):
            historic_min = min(historic_min, value)
            historic_max = max(historic_max, value)
        
        if historic_max - historic_min > 1e-10:
            result[i] = min_val + (max_val - min_val) * (value - historic_min) / (historic_max - historic_min)
        else:
            result[i] = min_val
    
    return pd.Series(result, index=src.index)


def _get_default_config() -> Dict:
    """
    Get default configuration for all indicators.
    
    Returns:
        Dict with default parameters for each indicator
    """
    return {
        'kurutoga': {
            'enabled': True,
            'length': 14,
            'source': 'Close'
        },
        'tsi': {
            'enabled': True,
            'fast': 13,
            'slow': 25,
            'signal': 13
        },
        'macd': {
            'enabled': True,
            'fast': 12,
            'slow': 26,
            'signal': 9
        },
        'mfi': {
            'enabled': True,
            'length': 14,
            'include_signal': True,
            'signal_period': 9
        },
        'cog': {
            'enabled': True,
            'length': 9,
            'source': 'Close'
        },
        'momentum': {
            'enabled': True,
            'length': 20
        },
        'rsi': {
            'enabled': True,
            'length': 14
        },
        'ma_crosses': {
            'enabled': True,
            'fast_period': 50,
            'slow_period': 200
        },
        'easy_trade': {
            'enabled': True,
            'fast_length': 12,
            'slow_length': 26,
            'signal_length': 9,
            'fast_period': 50,
            'slow_period': 200
        }
    }


def analyze_combined_signals(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Analyze combined signals from multiple indicators.
    
    Args:
        data: DataFrame with indicator signals
        
    Returns:
        Dict with signal analysis results
    """
    latest = data.iloc[-1]
    
    analysis = {
        'all_bullish': False,
        'all_bearish': False, 
        'mixed_signals': False,
        'golden_cross': False,
        'death_cross': False,
        'trend_bullish': False,
        'trend_bearish': False
    }
    
    # Check for combined signals
    signal_columns = [col for col in data.columns if col.endswith('_Signal')]
    if signal_columns:
        bullish_signals = sum(1 for col in signal_columns if 'Bullish' in str(latest[col]))
        bearish_signals = sum(1 for col in signal_columns if 'Bearish' in str(latest[col]))
        
        analysis['all_bullish'] = bullish_signals == len(signal_columns)
        analysis['all_bearish'] = bearish_signals == len(signal_columns)
        analysis['mixed_signals'] = not (analysis['all_bullish'] or analysis['all_bearish'])
    
    # Check crossover signals
    if 'Golden_Cross' in data.columns:
        analysis['golden_cross'] = latest['Golden_Cross']
    if 'Death_Cross' in data.columns:
        analysis['death_cross'] = latest['Death_Cross']
    
    # Check trend signals
    if 'Trend' in data.columns:
        analysis['trend_bullish'] = latest['Trend'] == 'Bullish'
        analysis['trend_bearish'] = latest['Trend'] == 'Bearish'
    
    return analysis