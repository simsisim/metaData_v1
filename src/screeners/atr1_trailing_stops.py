"""
ATR1 Trailing Stops - TradingView-validated implementation
==========================================================

Generates trailing stop signals using exact TradingView ATR cloud logic.
Matches the format and calculations from the original atr_cloud.py implementation.

This module creates the ATR1 trailing stops output in the reference format,
using the validated TradingView calculations for signal detection.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from .atr1_screener import calculate_atr_cloud

logger = logging.getLogger(__name__)


def generate_atr1_trailing_stops(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    Generate ATR1 trailing stops using exact atr_cloud.py logic.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with ATR1 parameters
        
    Returns:
        List of dictionaries with trailing stop results matching reference format
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
    
    logger.info(f"Generating ATR1 trailing stops for {len(batch_data)} tickers")
    
    for ticker, df in batch_data.items():
        try:
            # Data validation
            if df is None or df.empty:
                continue
                
            required_columns = ['Close', 'High', 'Low', 'Volume']
            if not all(col in df.columns for col in required_columns):
                continue
                
            # Minimum data length check
            min_length = max(params['length'], params['length2']) + 10
            if len(df) < min_length:
                continue
                
            # Current price and volume filters
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            if (current_price < params['min_price'] or 
                current_volume < params['min_volume']):
                continue
                
            # Calculate ATR cloud (exact logic from atr_cloud.py)
            df_cloud = calculate_atr_cloud(
                df,
                length=params['length'],
                factor=params['factor'],
                length2=params['length2'],
                factor2=params['factor2'],
                src=params['src'],
                src2=params['src2']
            )
            
            # Find the most recent signal (exact logic from atr_cloud.py:149)
            signal_df = df_cloud[df_cloud['crossUp'] | df_cloud['crossDn']]
            
            if not signal_df.empty:
                # Get the most recent signal
                last_signal = signal_df.iloc[-1]
                signal_date = last_signal.name
                signal_close_price = last_signal['Close']
                
                # Calculate signal metrics (exact logic from atr_cloud.py:152-154)
                signal_type = 'Long' if last_signal['crossUp'] else 'Short'
                price_change = ((current_price - signal_close_price) / signal_close_price) * 100
                days_since_signal = (df_cloud.index[-1] - signal_date).days
                
                # Signal Day Price Analysis
                signal_open = last_signal['Open']
                signal_close = last_signal['Close']
                signal_day_change_abs = abs(signal_close - signal_open)
                signal_day_change_pct = ((signal_close - signal_open) / signal_open) * 100 if signal_open != 0 else 0
                
                # Performance Tracking Since Signal
                performance_since_signal = price_change
                
                results.append({
                    'ticker': ticker,
                    'signal_date': signal_date,
                    'current_price': current_price,
                    'signal': signal_type,
                    'vStop': last_signal['vStop'],
                    'vStop2': last_signal['vStop2'],
                    'vstopseries': last_signal['vstopseries'],
                    'price_change_pct': price_change,
                    'days_since_signal': days_since_signal,
                    'signal_day_change_abs': signal_day_change_abs,
                    'signal_day_change_pct': signal_day_change_pct,
                    'performance_since_signal': performance_since_signal
                })
            else:
                # No signal found (exact logic from atr_cloud.py:167-178)
                results.append({
                    'ticker': ticker,
                    'signal_date': df_cloud.index[-1],  # Latest date
                    'current_price': current_price,
                    'signal': 'No Signal',
                    'vStop': df_cloud['vStop'].iloc[-1],
                    'vStop2': df_cloud['vStop2'].iloc[-1],
                    'vstopseries': df_cloud['vstopseries'].iloc[-1],
                    'price_change_pct': 0.0,  # No signal, no change
                    'days_since_signal': 0,
                    'signal_day_change_abs': 0.0,
                    'signal_day_change_pct': 0.0,
                    'performance_since_signal': 0.0
                })
                    
        except Exception as e:
            logger.debug(f"Error processing {ticker} for ATR1 trailing stops: {e}")
            continue
    
    logger.info(f"Generated {len(results)} ATR1 trailing stop signals")
    return results


def save_atr1_trailing_stops(results: List[Dict], params: Optional[Dict] = None) -> None:
    """
    Save ATR1 trailing stops in exact reference format.
    
    Args:
        results: List of ATR1 trailing stop results
        params: ATR1 parameters (contains ticker_choice and timeframe for filename)
    """
    try:
        if not results:
            return
            
        # Create DataFrame with exact reference format
        output_data = []
        for result in results:
            signal_date_str = result['signal_date'].strftime('%Y-%m-%d')
            price_change_str = round(result['price_change_pct'], 2)
            days_since_str = result['days_since_signal']
            
            # Signal Day Price Analysis
            signal_day_abs_str = round(result['signal_day_change_abs'], 2) if result['signal_day_change_abs'] is not None else 'N/A'
            signal_day_pct_str = round(result['signal_day_change_pct'], 2) if result['signal_day_change_pct'] is not None else 'N/A'
            performance_since_str = round(result['performance_since_signal'], 2) if result['performance_since_signal'] is not None else 'N/A'
            
            output_data.append({
                'Symbol': result['ticker'],
                'Signal Date': signal_date_str,
                'Current Price': round(result['current_price'], 2),
                'Signal': result['signal'],
                'vStop': round(result['vStop'], 2),
                'vStop2': round(result['vStop2'], 2),
                'vstopseries': round(result['vstopseries'], 2),
                'Price Change %': price_change_str,
                'Days Since Signal': days_since_str,
                'Signal Day Open-Close $': signal_day_abs_str,
                'Signal Day Open-Close %': signal_day_pct_str,
                'Performance Since Signal %': performance_since_str
            })
        
        # Create DataFrame
        df = pd.DataFrame(output_data)
        
        # Create output directory and filename
        output_dir = Path('results/screeners/atr')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ticker_choice and timeframe from params
        ticker_choice = params.get('ticker_choice', 0) if params else 0
        timeframe = params.get('timeframe', '') if params else ''
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Create filename matching reference format (ATR1_trailing_stops)
        if timeframe:
            filename = f'ATR1_trailing_stops_{ticker_choice}_{date_str}_{timeframe}.csv'
        else:
            filename = f'ATR1_trailing_stops_{ticker_choice}_{date_str}.csv'
        
        output_file = output_dir / filename
        
        # Save with exact format matching reference
        df.to_csv(output_file, index=False, float_format='%.2f')
        
        print(f"📊 ATR1 trailing stops saved to: {output_file}")
        print(f"📈 Total ATR1 trailing stop signals: {len(results)}")
        
        # Show signal breakdown
        signal_counts = df['Signal'].value_counts()
        for signal_type, count in signal_counts.items():
            print(f"   • {signal_type}: {count}")
            
        logger.info(f"ATR1 trailing stops saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving ATR1 trailing stops: {e}")
        print(f"❌ Error saving ATR1 trailing stops: {e}")


def atr1_trailing_stops_with_output(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    ATR1 trailing stops wrapper that automatically saves results to file.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with ATR1 parameters
        
    Returns:
        List of dictionaries with trailing stop results
    """
    # Generate trailing stops using exact atr_cloud.py logic
    results = generate_atr1_trailing_stops(batch_data, params)
    
    # Save results if any found
    if results:
        save_atr1_trailing_stops(results, params)
    
    return results