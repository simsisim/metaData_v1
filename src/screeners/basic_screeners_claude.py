"""
Basic Screeners Collection - Claude
===================================

Contains the basic momentum, breakout, and value-momentum screeners
that were previously hardcoded to always run. Now configurable via user_data.csv.
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


def run_basic_screeners(batch_data, config):
    """
    Run basic screeners based on configuration settings.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame}
        config: User configuration object with basic screener settings
        
    Returns:
        dict: Results from enabled basic screeners
    """
    results = {
        'momentum': [],
        'breakout': [],
        'value_momentum': []
    }
    
    # Run only enabled screeners
    if getattr(config, 'basic_momentum_enable', False):
        print("🔍 Running momentum screener...")
        results['momentum'] = momentum_screener(batch_data)
        
    if getattr(config, 'basic_breakout_enable', False):
        print("🔍 Running breakout screener...")
        results['breakout'] = breakout_screener(batch_data)
        
    if getattr(config, 'basic_value_momentum_enable', False):
        print("🔍 Running value-momentum screener...")
        results['value_momentum'] = value_momentum_screener(batch_data)
    
    return results