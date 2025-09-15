"""
ATR1 Screener - TradingView-validated implementation
====================================================

Implements volatility screening using exact TradingView ATR logic with RMA smoothing.
Based on atr_cloud.py implementation validated against TradingView indicators.

Screening Logic:
- vStop and vStop2 calculations using RMA smoothing
- CrossUp/CrossDown signal detection
- Volatility trend analysis with dual stop levels
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_true_range(df):
    """Calculate True Range exactly as in TradingView validation."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range


def calculate_atr_rma(df, length=14):
    """Calculate ATR using RMA smoothing (TradingView method)."""
    true_range = calculate_true_range(df)
    atr = true_range.ewm(alpha=1/length, min_periods=length).mean()
    return atr


def vol_stop(df, src_column, length, factor):
    """
    Calculate volatility stop levels using TradingView logic.
    Exact implementation from atr_cloud.py validated against TradingView.
    """
    df_calc = df.copy()
    df_calc['TR'] = calculate_true_range(df_calc)
    df_calc['ATR'] = calculate_atr_rma(df_calc, length)
    df_calc['src'] = df_calc[src_column]
    df_calc['max'] = df_calc['src'].copy()
    df_calc['min'] = df_calc['src'].copy()
    df_calc['stop'] = 0.0
    df_calc['uptrend'] = True

    for i in range(1, len(df_calc)):
        if np.isnan(df_calc['ATR'].iloc[i]):
            df_calc.loc[df_calc.index[i], 'atr_m'] = df_calc['TR'].iloc[i]  # Fallback to True Range
        else:
            df_calc.loc[df_calc.index[i], 'atr_m'] = df_calc['ATR'].iloc[i] * factor
        
        # Update max and min
        df_calc.loc[df_calc.index[i], 'max'] = max(df_calc['max'].iloc[i-1], df_calc['src'].iloc[i])
        df_calc.loc[df_calc.index[i], 'min'] = min(df_calc['min'].iloc[i-1], df_calc['src'].iloc[i])
        
        # Calculate new stop
        if df_calc['uptrend'].iloc[i-1]:
            new_stop = max(df_calc['stop'].iloc[i-1], df_calc['max'].iloc[i] - df_calc['atr_m'].iloc[i])
        else:
            new_stop = min(df_calc['stop'].iloc[i-1], df_calc['min'].iloc[i] + df_calc['atr_m'].iloc[i])

        if np.isnan(new_stop):
            df_calc.loc[df_calc.index[i], 'stop'] = df_calc['src'].iloc[i]
        else:
            df_calc.loc[df_calc.index[i], 'stop'] = new_stop
        
        # Update uptrend
        df_calc.loc[df_calc.index[i], 'uptrend'] = df_calc['src'].iloc[i] - df_calc['stop'].iloc[i] >= 0.0

        # Check for trend reversal
        if df_calc['uptrend'].iloc[i] != df_calc['uptrend'].iloc[i-1]:
            df_calc.loc[df_calc.index[i], 'max'] = df_calc['src'].iloc[i]
            df_calc.loc[df_calc.index[i], 'min'] = df_calc['src'].iloc[i]
            df_calc.loc[df_calc.index[i], 'stop'] = df_calc['src'].iloc[i] - df_calc['atr_m'].iloc[i] if df_calc['uptrend'].iloc[i] else df_calc['src'].iloc[i] + df_calc['atr_m'].iloc[i]

    return df_calc[['stop', 'uptrend']]


def calculate_atr_cloud(df, length=20, factor=3.0, length2=20, factor2=1.5, src='Close', src2='Close'):
    """
    Calculate ATR cloud with dual volatility stops and signals.
    Exact implementation from atr_cloud.py validated against TradingView.
    """
    v_stop = vol_stop(df, src, length, factor)
    v_stop2 = vol_stop(df, src2, length2, factor2)
    
    df_result = df.copy()
    df_result['vStop'] = v_stop['stop']
    df_result['uptrend'] = v_stop['uptrend']
    df_result['vStop2'] = v_stop2['stop']
    df_result['uptrend2'] = v_stop2['uptrend']
    
    df_result['vstopseries'] = (df_result['vStop'] + df_result['vStop2']) / 2
    
    df_result['crossUp'] = (df_result['vStop2'] > df_result['vStop']) & (df_result['vStop2'].shift(1) <= df_result['vStop'].shift(1))
    df_result['crossDn'] = (df_result['vStop2'] < df_result['vStop']) & (df_result['vStop2'].shift(1) >= df_result['vStop'].shift(1))
    
    return df_result


def _calculate_atr1_score(signal_type: str, days_since: int, price_change: float, current_uptrend: bool) -> float:
    """
    Calculate ATR1 screening score based on signal quality and recency.
    
    Args:
        signal_type: 'Long' or 'Short' signal
        days_since: Days since signal was generated
        price_change: Price change percentage since signal
        current_uptrend: Current uptrend status
        
    Returns:
        Numeric score for ranking (higher = better)
    """
    score = 0.0
    
    # Base score for having a signal
    score += 20
    
    # Recency bonus (newer signals are better)
    if days_since <= 5:
        score += 15 - (days_since * 2)
    elif days_since <= 10:
        score += 10 - days_since
    
    # Performance bonus/penalty
    if signal_type == 'Long' and price_change > 0:
        score += min(price_change, 10)  # Cap at 10 points
    elif signal_type == 'Short' and price_change < 0:
        score += min(abs(price_change), 10)  # Cap at 10 points
    elif signal_type == 'Long' and price_change < -5:
        score -= 5  # Penalty for failed long signals
    elif signal_type == 'Short' and price_change > 5:
        score -= 5  # Penalty for failed short signals
    
    # Trend alignment bonus
    if (signal_type == 'Long' and current_uptrend) or (signal_type == 'Short' and not current_uptrend):
        score += 5
    
    return max(score, 0)


def atr1_screener(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    ATR1 screener using TradingView-validated ATR cloud logic.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with ATR1 parameters
        
    Returns:
        List of dictionaries with screening results sorted by score
    """
    # Apply data capping for performance optimization
    from ..data_optimization import cap_historical_data
    timeframe = params.get('timeframe', 'daily') if params else 'daily'
    cap_setting = params.get('cap_history_data', 2) if params else 2
    
    if cap_setting > 0:
        batch_data = cap_historical_data(batch_data, timeframe, cap_setting)
    if params is None:
        params = {
            'length': 20,
            'factor': 3.0,
            'length2': 20,
            'factor2': 1.5,
            'src': 'Close',
            'src2': 'Close',
            'min_volume': 10000,
            'min_price': 1.0
        }
    
    results = []
    
    logger.info(f"Running ATR1 screener on {len(batch_data)} tickers")
    
    for ticker, df in batch_data.items():
        try:
            # Data validation
            if df is None or df.empty:
                continue
                
            required_columns = ['Close', 'High', 'Low', 'Volume']
            if not all(col in df.columns for col in required_columns):
                logger.debug(f"Skipping {ticker}: Missing required columns")
                continue
                
            # Minimum data length check
            min_length = max(params['length'], params['length2']) + 10
            if len(df) < min_length:
                logger.debug(f"Skipping {ticker}: Insufficient data ({len(df)} < {min_length})")
                continue
                
            # Current price and volume filters
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            if (current_price < params['min_price'] or 
                current_volume < params['min_volume']):
                continue
                
            # Calculate ATR cloud with TradingView logic
            df_cloud = calculate_atr_cloud(
                df,
                length=params['length'],
                factor=params['factor'],
                length2=params['length2'],
                factor2=params['factor2'],
                src=params['src'],
                src2=params['src2']
            )
            
            # Find the most recent signal (crossUp or crossDn) - exact logic from atr_cloud.py
            signal_df = df_cloud[df_cloud['crossUp'] | df_cloud['crossDn']]
            
            if not signal_df.empty:
                # Get the most recent signal (exactly as in atr_cloud.py:149)
                last_signal = signal_df.iloc[-1]
                signal_date = last_signal.name
                signal_close_price = last_signal['Close']
                
                # Calculate signal metrics (exactly as in atr_cloud.py:152-154)
                signal_type = 'Long' if last_signal['crossUp'] else 'Short'
                price_change = ((current_price - signal_close_price) / signal_close_price) * 100
                days_since_signal = (df.index[-1] - signal_date).days
                current_uptrend = df_cloud['uptrend'].iloc[-1]
                
                # Signal Day Price Analysis
                signal_open = last_signal['Open']
                signal_close = last_signal['Close']
                signal_day_change_abs = abs(signal_close - signal_open)
                signal_day_change_pct = ((signal_close - signal_open) / signal_open) * 100 if signal_open != 0 else 0
                
                # Performance Tracking Since Signal (same as price_change but explicitly named)
                performance_since_signal = price_change
                
                # Get volatility stop levels from signal date (not current date)
                signal_vstop = last_signal['vStop']
                signal_vstop2 = last_signal['vStop2']
                signal_vstopseries = last_signal['vstopseries']
                
                # Calculate score
                score = _calculate_atr1_score(signal_type, days_since_signal, price_change, current_uptrend)
                
                results.append({
                    'ticker': ticker,
                    'screen_type': 'atr1',
                    'signal_type': signal_type,
                    'signal_date': signal_date,
                    'signal_close_price': signal_close_price,
                    'current_price': current_price,
                    'price_change_pct': price_change,
                    'days_since_signal': days_since_signal,
                    'vstop': signal_vstop,
                    'vstop2': signal_vstop2,
                    'vstopseries': signal_vstopseries,
                    'uptrend': current_uptrend,
                    'volume': current_volume,
                    'score': score,
                    'signal_day_change_abs': signal_day_change_abs,
                    'signal_day_change_pct': signal_day_change_pct,
                    'performance_since_signal': performance_since_signal
                })
            else:
                # No signal found - add with current stop levels
                current_vstop = df_cloud['vStop'].iloc[-1]
                current_vstop2 = df_cloud['vStop2'].iloc[-1]
                current_vstopseries = df_cloud['vstopseries'].iloc[-1]
                current_uptrend = df_cloud['uptrend'].iloc[-1]
                
                results.append({
                    'ticker': ticker,
                    'screen_type': 'atr1',
                    'signal_type': 'No Signal',
                    'signal_date': None,
                    'signal_close_price': None,
                    'current_price': current_price,
                    'price_change_pct': None,
                    'days_since_signal': None,
                    'vstop': current_vstop,
                    'vstop2': current_vstop2,
                    'vstopseries': current_vstopseries,
                    'uptrend': current_uptrend,
                    'volume': current_volume,
                    'score': 0,
                    'signal_day_change_abs': None,
                    'signal_day_change_pct': None,
                    'performance_since_signal': None
                })
                    
        except Exception as e:
            logger.debug(f"Error processing {ticker} in ATR1 screener: {e}")
            continue
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"ATR1 screener found {len(results)} results")
    return results


def _save_atr1_results(results: List[Dict], params: Optional[Dict] = None) -> None:
    """
    Save ATR1 screening results in TradingView format.
    
    Args:
        results: List of ATR1 screening results
        params: ATR1 parameters (contains ticker_choice and timeframe for filename)
    """
    try:
        if not results:
            return
            
        # Create DataFrame with TradingView-style format
        output_data = []
        for result in results:
            signal_date_str = 'N/A'
            price_change_str = 'N/A'
            days_since_str = 'N/A'
            
            if result['signal_date'] is not None:
                signal_date_str = result['signal_date'].strftime('%Y-%m-%d')
                price_change_str = round(result['price_change_pct'], 2)
                days_since_str = result['days_since_signal']
            
            # Signal Day Price Analysis
            signal_day_abs_str = 'N/A'
            signal_day_pct_str = 'N/A'
            performance_since_str = 'N/A'
            
            if result['signal_day_change_abs'] is not None:
                signal_day_abs_str = round(result['signal_day_change_abs'], 2)
                signal_day_pct_str = round(result['signal_day_change_pct'], 2)
                performance_since_str = round(result['performance_since_signal'], 2)
            
            output_data.append({
                'Symbol': result['ticker'],
                'Signal Date': signal_date_str,
                'Current Price': round(result['current_price'], 2),
                'Signal': result['signal_type'],
                'vStop': round(result['vstop'], 2),
                'vStop2': round(result['vstop2'], 2),
                'vstopseries': round(result['vstopseries'], 2),
                'Price Change %': price_change_str,
                'Days Since Signal': days_since_str,
                'Signal Day Open-Close $': signal_day_abs_str,
                'Signal Day Open-Close %': signal_day_pct_str,
                'Performance Since Signal %': performance_since_str,
                'Uptrend': result['uptrend'],
                'Volume': int(result['volume']),
                'Score': round(result['score'], 2)
            })
        
        # Create DataFrame and sort by Score
        df = pd.DataFrame(output_data)
        df = df.sort_values('Score', ascending=False)
        
        # Create output directory and filename
        output_dir = Path('results/screeners/atr')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ticker_choice and timeframe from params
        ticker_choice = params.get('ticker_choice', 0) if params else 0
        timeframe = params.get('timeframe', '') if params else ''
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Create filename with ATR1 prefix and timeframe
        if timeframe:
            filename = f'ATR1_{ticker_choice}_{date_str}_{timeframe}.csv'
        else:
            filename = f'ATR1_{ticker_choice}_{date_str}.csv'
        
        output_file = output_dir / filename
        
        # Save with proper formatting
        df.to_csv(output_file, index=False, float_format='%.2f')
        
        print(f"📊 ATR1 screening results saved to: {output_file}")
        print(f"📈 Total ATR1 results: {len(results)}")
        
        # Show signal breakdown
        signal_counts = df['Signal'].value_counts()
        for signal_type, count in signal_counts.items():
            print(f"   • {signal_type}: {count}")
            
        logger.info(f"ATR1 screening results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving ATR1 screening results: {e}")
        print(f"❌ Error saving ATR1 screening results: {e}")


def atr1_screener_with_output(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    ATR1 screener wrapper that automatically saves results to file.
    Also generates ATR1 trailing stops in reference format.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with ATR1 parameters
        
    Returns:
        List of dictionaries with screening results
    """
    # Call the main screener logic
    results = atr1_screener(batch_data, params)
    
    # Save screening results if any found
    if results:
        _save_atr1_results(results, params)
    
    # Also generate ATR1 trailing stops output (reference format)
    from .atr1_trailing_stops import atr1_trailing_stops_with_output
    atr1_trailing_stops_with_output(batch_data, params)
    
    return results