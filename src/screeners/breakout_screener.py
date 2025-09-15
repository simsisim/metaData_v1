"""
Breakout Screener
================

Screens for stocks breaking out of consolidation patterns based on:
- Consolidation pattern analysis
- Volume surge on breakout
- Price breakout above resistance levels
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def breakout_screener(batch_data, params=None):
    """
    Screen for stocks breaking out of consolidation patterns.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame}
        params: Dictionary with screening parameters
        
    Returns:
        List of dictionaries with screening results sorted by score
    """
    if params is None:
        params = {
            'min_consolidation_days': 20,
            'max_consolidation_range': 0.15,  # 15% max range during consolidation
            'breakout_threshold': 0.02,       # 2% breakout above resistance
            'min_volume_surge': 1.5,          # 50% volume increase on breakout
            'min_price': 5.0
        }
    
    results = []
    
    for ticker, df in batch_data.items():
        try:
            if df is None or df.empty or len(df) < 50:
                continue
                
            close = df['Close']
            current_price = close.iloc[-1]
            
            if current_price < params['min_price']:
                continue
                
            # Analyze recent consolidation period
            lookback = min(params['min_consolidation_days'], len(df) - 5)
            consolidation_period = close.iloc[-(lookback+5):-5]  # Exclude last 5 days
            recent_period = close.iloc[-5:]  # Last 5 days for breakout detection
            
            if len(consolidation_period) < params['min_consolidation_days']:
                continue
                
            # Check if consolidation range is tight enough
            consolidation_high = consolidation_period.max()
            consolidation_low = consolidation_period.min()
            consolidation_range = (consolidation_high - consolidation_low) / consolidation_low
            
            if consolidation_range > params['max_consolidation_range']:
                continue
                
            # Check for breakout
            breakout_level = consolidation_high * (1 + params['breakout_threshold'])
            has_breakout = current_price >= breakout_level
            
            # Volume analysis
            volume_surge = 1.0
            if 'Volume' in df.columns and len(df) >= 20:
                recent_avg_volume = df['Volume'].tail(5).mean()
                historical_avg_volume = df['Volume'].iloc[-(lookback+10):-10].mean()
                volume_surge = recent_avg_volume / historical_avg_volume if historical_avg_volume > 0 else 1.0
                
            if has_breakout and volume_surge >= params['min_volume_surge']:
                results.append({
                    'ticker': ticker,
                    'screen_type': 'breakout',
                    'current_price': current_price,
                    'consolidation_range': consolidation_range,
                    'breakout_level': breakout_level,
                    'breakout_percentage': (current_price / consolidation_high) - 1,
                    'volume_surge': volume_surge,
                    'score': ((current_price / breakout_level) - 1) * 100 + (volume_surge - 1) * 10
                })
                
        except Exception as e:
            logger.debug(f"Error in breakout screener for {ticker}: {e}")
            continue
            
    return sorted(results, key=lambda x: x['score'], reverse=True)