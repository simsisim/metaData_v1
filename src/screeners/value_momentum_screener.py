"""
Value Momentum Screener
======================

Combines value and momentum factors for screening based on:
- 6-month returns
- Volatility analysis
- Relative strength vs market benchmarks
- Price-based value proxies
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def value_momentum_screener(batch_data, params=None):
    """
    Screen combining value and momentum factors.
    Note: Limited fundamental data available, so uses price-based proxies.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame}
        params: Dictionary with screening parameters
        
    Returns:
        List of dictionaries with screening results sorted by score
    """
    if params is None:
        params = {
            'min_return_6m': 0.15,     # 15% minimum 6-month return
            'max_volatility': 0.6,     # 60% max annualized volatility
            'min_relative_strength': 1.1,  # Outperforming market by 10%
            'min_price': 10.0
        }
    
    results = []
    
    # Try to find a market benchmark (SPY, QQQ, etc.)
    market_benchmark = None
    benchmark_return = 0
    
    for ticker in ['SPY', 'QQQ', '^GSPC', '^IXIC']:
        if ticker in batch_data and batch_data[ticker] is not None:
            benchmark_df = batch_data[ticker]
            if 'Close' in benchmark_df.columns and len(benchmark_df) >= 132:
                benchmark_return = (benchmark_df['Close'].iloc[-1] / benchmark_df['Close'].iloc[-132]) - 1
                market_benchmark = ticker
                break
    
    for ticker, df in batch_data.items():
        try:
            if df is None or df.empty or len(df) < 132:
                continue
                
            close = df['Close']
            current_price = close.iloc[-1]
            
            if current_price < params['min_price']:
                continue
                
            # Calculate 6-month return
            return_6m = (close.iloc[-1] / close.iloc[-132]) - 1
            
            # Calculate volatility
            daily_returns = close.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            
            # Relative strength vs market
            relative_strength = (return_6m / benchmark_return) if benchmark_return != 0 else 1.0
            
            # Apply screening criteria
            passes_screen = (
                return_6m >= params['min_return_6m'] and
                volatility <= params['max_volatility'] and
                relative_strength >= params['min_relative_strength']
            )
            
            if passes_screen:
                # Simple momentum score
                momentum_score = return_6m * 100
                volatility_score = max(0, (0.6 - volatility)) * 100
                relative_strength_score = (relative_strength - 1) * 100
                
                results.append({
                    'ticker': ticker,
                    'screen_type': 'value_momentum',
                    'current_price': current_price,
                    'return_6m': return_6m,
                    'volatility': volatility,
                    'relative_strength': relative_strength,
                    'vs_benchmark': market_benchmark,
                    'score': momentum_score + volatility_score + relative_strength_score
                })
                
        except Exception as e:
            logger.debug(f"Error in value_momentum screener for {ticker}: {e}")
            continue
            
    return sorted(results, key=lambda x: x['score'], reverse=True)