"""
Price Volume Breakout (PVB) Screener
====================================

Implements the TradeDots Price Volume Breakout strategy as a screener.
Identifies stocks with simultaneous price and volume breakouts with trend confirmation.

Based on the TradingView-tested implementation from intro_PVB/price_volume_breakout.py
Optimal for highly volatile assets with pronounced price and volume swings.

Signal Logic:
- Buy: Close > prev_high AND Volume > prev_volume_high AND Close > SMA
- Sell: Close < prev_low AND Volume > prev_volume_high AND Close < SMA
- Close: 5+ consecutive days violating trend (below/above SMA)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def _calculate_pvb_indicators(
    data: pd.DataFrame, 
    price_breakout_period: int, 
    volume_breakout_period: int, 
    trendline_length: int
) -> pd.DataFrame:
    """
    Calculate PVB technical indicators.
    
    Args:
        data: DataFrame with OHLCV data
        price_breakout_period: Window for price high/low calculation
        volume_breakout_period: Window for volume high calculation
        trendline_length: Window for SMA trend calculation
        
    Returns:
        DataFrame with added indicator columns
    """
    try:
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Price breakout indicators
        df['Price_Highest'] = df['High'].rolling(window=price_breakout_period).max()
        df['Price_Lowest'] = df['Low'].rolling(window=price_breakout_period).min()
        
        # Volume breakout indicator
        df['Volume_Highest'] = df['Volume'].rolling(window=volume_breakout_period).max()
        
        # Trend indicator
        df['SMA'] = df['Close'].rolling(window=trendline_length).mean()
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating PVB indicators: {e}")
        raise


def _generate_pvb_signals(data: pd.DataFrame, order_direction: str) -> List[Dict]:
    """
    Generate PVB buy/sell signals based on breakout conditions.
    
    Args:
        data: DataFrame with OHLCV and indicator data
        order_direction: "Long", "Short", or "Long and Short"
        
    Returns:
        List of signal dictionaries
    """
    signals = []
    current_signal = "No Signal"
    signal_date = None
    consecutive_days = 0

    try:
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            prev_row = data.iloc[i-1]
            
            # Skip if required indicators are NaN
            if pd.isna(current_row['SMA']) or pd.isna(prev_row['Price_Highest']):
                continue

            # Long Signal Detection
            if (order_direction in ["Long", "Long and Short"] and
                current_row['Close'] > prev_row['Price_Highest'] and
                current_row['Volume'] > prev_row['Volume_Highest'] and
                current_row['Close'] > current_row['SMA'] and
                current_signal not in ["Buy"]):
                
                current_signal = "Buy"
                signal_date = data.index[i]
                consecutive_days = 0

            # Short Signal Detection
            elif (order_direction in ["Short", "Long and Short"] and
                  current_row['Close'] < prev_row['Price_Lowest'] and
                  current_row['Volume'] > prev_row['Volume_Highest'] and
                  current_row['Close'] < current_row['SMA'] and
                  current_signal not in ["Sell"]):
                
                current_signal = "Sell"
                signal_date = data.index[i]
                consecutive_days = 0

            # Check for closing conditions
            if current_signal == "Buy":
                if current_row['Close'] < current_row['SMA']:
                    consecutive_days += 1
                else:
                    consecutive_days = 0
                if consecutive_days >= 5:
                    current_signal = "Close Buy"
                    signal_date = data.index[i]
                    consecutive_days = 0

            elif current_signal == "Sell":
                if current_row['Close'] > current_row['SMA']:
                    consecutive_days += 1
                else:
                    consecutive_days = 0
                if consecutive_days >= 5:
                    current_signal = "Close Sell"
                    signal_date = data.index[i]
                    consecutive_days = 0

            # Record signal if it's new and valid
            if (current_signal != "No Signal" and 
                signal_date is not None and 
                (not signals or signals[-1]['signal_type'] != current_signal)):
                
                signals.append({
                    'date': signal_date,
                    'signal_type': current_signal,
                    'close_price': current_row['Close'],
                    'volume': current_row['Volume'],
                    'sma': current_row['SMA'],
                    'price_highest': prev_row['Price_Highest'],
                    'volume_highest': prev_row['Volume_Highest']
                })

        return signals
        
    except Exception as e:
        logger.error(f"Error generating PVB signals: {e}")
        return []


def _calculate_pvb_score(
    latest_signal: Dict, 
    current_data: pd.Series, 
    days_since_signal: int
) -> float:
    """
    Calculate PVB screening score based on signal strength and characteristics.
    
    Args:
        latest_signal: Dictionary with latest signal data
        current_data: Current row of market data
        days_since_signal: Days elapsed since signal
        
    Returns:
        Numeric score for ranking (higher = better)
    """
    try:
        score = 0.0
        
        # Base signal strength (breakout magnitude)
        if latest_signal['signal_type'] in ['Buy', 'Sell']:
            # Price breakout strength
            if latest_signal['signal_type'] == 'Buy':
                price_breakout = (latest_signal['close_price'] / latest_signal['price_highest'] - 1) * 100
            else:
                price_breakout = (latest_signal['price_highest'] / latest_signal['close_price'] - 1) * 100
            
            # Volume surge strength
            volume_surge = (latest_signal['volume'] / latest_signal['volume_highest'] - 1) * 100
            
            # Trend alignment bonus
            sma_distance = abs((latest_signal['close_price'] / latest_signal['sma'] - 1) * 100)
            
            score = price_breakout * 2 + volume_surge + min(sma_distance, 10)
            
            # Recency bonus (recent signals are more valuable)
            recency_factor = max(0, (10 - days_since_signal) / 10)
            score *= (0.7 + 0.3 * recency_factor)
            
        # Penalty for close signals (exit conditions)
        elif latest_signal['signal_type'] in ['Close Buy', 'Close Sell']:
            score = -10  # Negative score indicates exit condition
            
        return max(score, 0)  # Ensure non-negative score
        
    except Exception as e:
        logger.error(f"Error calculating PVB score: {e}")
        return 0.0


def pvb_screener(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    Price Volume Breakout screener for identifying breakout signals.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with PVB parameters
        
    Returns:
        List of dictionaries with screening results sorted by score
    """
    # Call the main screener logic
    results = _pvb_screener_logic(batch_data, params)
    
    # Save results in TradingView-compatible format if results exist
    if results:
        _save_pvb_results(results, params)
    
    return results


def _pvb_screener_logic(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    Core PVB screener logic separated for testing and reuse.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with PVB parameters
        
    Returns:
        List of dictionaries with screening results
    """
    if params is None:
        params = {
            'price_breakout_period': 20,
            'volume_breakout_period': 20, 
            'trendline_length': 50,
            'order_direction': 'Long and Short',
            'close_threshold': 5,
            'min_volume': 10000,
            'min_price': 1.0
        }
    
    results = []
    
    logger.info(f"Running PVB screener on {len(batch_data)} tickers")
    
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
            min_length = max(
                params['price_breakout_period'], 
                params['volume_breakout_period'],
                params['trendline_length']
            ) + 10
            
            if len(df) < min_length:
                logger.debug(f"Skipping {ticker}: Insufficient data ({len(df)} < {min_length})")
                continue
                
            # Current price and volume filters
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            if (current_price < params['min_price'] or 
                current_volume < params['min_volume']):
                continue
                
            # Calculate PVB indicators
            df_with_indicators = _calculate_pvb_indicators(
                df,
                params['price_breakout_period'],
                params['volume_breakout_period'], 
                params['trendline_length']
            )
            
            # Generate signals
            signals = _generate_pvb_signals(df_with_indicators, params['order_direction'])
            
            if not signals:
                continue
                
            # Get latest signal
            latest_signal = signals[-1]
            current_row = df_with_indicators.iloc[-1]
            
            # Calculate days since signal
            days_since_signal = (df_with_indicators.index[-1] - latest_signal['date']).days
            
            # Include all signal types within max age (updated to match user config: 50 days)
            max_age = params.get('signal_max_age', 50)
            if days_since_signal <= max_age:
                
                # Calculate screening score
                score = _calculate_pvb_score(latest_signal, current_row, days_since_signal)
                
                # Calculate additional metrics for TradingView format
                price_change_pct = ((current_price / latest_signal['sma']) - 1) * 100
                volume_change_pct = ((current_volume / latest_signal['volume_highest']) - 1) * 100
                
                # Signal Day Price Analysis
                signal_row = df_with_indicators.loc[latest_signal['date']]
                signal_open = signal_row['Open']
                signal_close = signal_row['Close']
                signal_day_change_abs = abs(signal_close - signal_open)
                signal_day_change_pct = ((signal_close - signal_open) / signal_open) * 100 if signal_open != 0 else 0
                
                # Performance Tracking Since Signal
                performance_since_signal = ((current_price - latest_signal['close_price']) / latest_signal['close_price']) * 100
                
                results.append({
                    'ticker': ticker,
                    'screen_type': 'pvb',
                    'signal_type': latest_signal['signal_type'],
                    'signal_date': latest_signal['date'],
                    'current_price': current_price,
                    'signal_price': latest_signal['close_price'],
                    'sma': latest_signal['sma'],
                    'volume': latest_signal['volume'],
                    'volume_highest': latest_signal['volume_highest'],
                    'days_since_signal': days_since_signal,
                    'price_change_pct': price_change_pct,
                    'volume_change_pct': volume_change_pct,
                    'volume_surge': ((latest_signal['volume'] / latest_signal['volume_highest']) - 1) * 100,
                    'score': score,
                    'signal_day_change_abs': signal_day_change_abs,
                    'signal_day_change_pct': signal_day_change_pct,
                    'performance_since_signal': performance_since_signal
                })
                    
        except Exception as e:
            logger.debug(f"Error processing {ticker} in PVB screener: {e}")
            continue
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"PVB screener found {len(results)} signals")
    return results


def _save_pvb_results(results: List[Dict], params: Optional[Dict] = None) -> None:
    """
    Save PVB results in TradingView-compatible CSV format.
    
    Args:
        results: List of PVB screening results
        params: PVB parameters (contains ticker_choice for filename)
    """
    try:
        if not results:
            return
            
        # Create DataFrame with exact TradingView column structure plus new columns
        output_data = []
        for result in results:
            output_data.append({
                'Symbol': result['ticker'],
                'Signal Date': result['signal_date'].strftime('%Y-%m-%d'),
                'Current Price': round(result['current_price'], 2),
                'Signal': result['signal_type'],
                'SMA': round(result['sma'], 2),
                'Volume': int(result['volume']),
                'Volume Highest': int(result['volume_highest']),
                'Price Change %': round(result['price_change_pct'], 2),
                'Volume Change %': round(result['volume_change_pct'], 2),
                'Days Since Signal': result['days_since_signal'],
                'Signal Day Open-Close $': round(result['signal_day_change_abs'], 2),
                'Signal Day Open-Close %': round(result['signal_day_change_pct'], 2),
                'Performance Since Signal %': round(result['performance_since_signal'], 2)
            })
        
        # Create DataFrame and sort by Days Since Signal
        df = pd.DataFrame(output_data)
        df = df.sort_values('Days Since Signal')
        
        # Create output directory and filename
        output_dir = Path('results/screeners/pvb')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ticker_choice and timeframe from params
        ticker_choice = params.get('ticker_choice', 0) if params else 0
        timeframe = params.get('timeframe', '') if params else ''
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Create filename with timeframe if available
        if timeframe:
            filename = f'PVB_{ticker_choice}_{date_str}_{timeframe}.csv'
        else:
            filename = f'PVB_{ticker_choice}_{date_str}.csv'
        
        output_file = output_dir / filename
        
        # Save with exact format matching TradingView output
        df.to_csv(output_file, index=False, float_format='%.2f')
        
        print(f"📊 PVB results saved to: {output_file}")
        print(f"📈 Total PVB signals found: {len(results)}")
        
        # Show signal breakdown
        signal_counts = df['Signal'].value_counts()
        for signal_type, count in signal_counts.items():
            print(f"   • {signal_type}: {count}")
            
        logger.info(f"PVB results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving PVB results: {e}")
        print(f"❌ Error saving PVB results: {e}")