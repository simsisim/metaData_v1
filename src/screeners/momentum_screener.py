"""
Momentum Screener
================

Screens for stocks showing strong price momentum based on:
- 1-month and 3-month returns
- Volume analysis
- RSI levels
- Price filters
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def momentum_screener(batch_data, params=None):
    """
    Screen for stocks showing strong price momentum.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame}
        params: Dictionary with screening parameters
        
    Returns:
        List of dictionaries with screening results sorted by score
    """
    if params is None:
        params = {
            'min_return_1m': 0.05,    # 5% minimum 1-month return
            'min_return_3m': 0.10,    # 10% minimum 3-month return
            'min_volume_ratio': 1.2,  # 20% above average volume
            'rsi_max': 70,            # RSI below 70 (not overbought)
            'min_price': 5.0          # Minimum stock price
        }
    
    results = []
    
    for ticker, df in batch_data.items():
        try:
            if df is None or df.empty or 'Close' not in df.columns or len(df) < 66:
                continue
                
            close = df['Close']
            current_price = close.iloc[-1]
            
            # Price filter
            if current_price < params['min_price']:
                continue
                
            # Calculate returns
            if len(df) >= 22:
                return_1m = (close.iloc[-1] / close.iloc[-22]) - 1
            else:
                return_1m = 0
                
            if len(df) >= 66:
                return_3m = (close.iloc[-1] / close.iloc[-66]) - 1
            else:
                return_3m = 0
                
            # Volume analysis
            volume_ratio = 1.0
            if 'Volume' in df.columns and len(df) >= 20:
                current_volume = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
            # RSI calculation
            rsi = 50  # Default neutral RSI
            if len(df) >= 15:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                if not np.isnan(rs.iloc[-1]) and rs.iloc[-1] > 0:
                    rsi = 100 - (100 / (1 + rs.iloc[-1]))
                    
            # Apply screening criteria
            passes_screen = (
                return_1m >= params['min_return_1m'] and
                return_3m >= params['min_return_3m'] and
                volume_ratio >= params['min_volume_ratio'] and
                rsi <= params['rsi_max']
            )
            
            if passes_screen:
                results.append({
                    'ticker': ticker,
                    'screen_type': 'momentum',
                    'current_price': current_price,
                    'return_1m': return_1m,
                    'return_3m': return_3m,
                    'volume_ratio': volume_ratio,
                    'rsi': rsi,
                    'score': return_1m + return_3m + (volume_ratio - 1) - (rsi - 50) / 100
                })
                
        except Exception as e:
            logger.debug(f"Error in momentum screener for {ticker}: {e}")
            continue
            
    return sorted(results, key=lambda x: x['score'], reverse=True)