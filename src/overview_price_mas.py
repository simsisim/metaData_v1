"""
Overview Price and Moving Averages Module
==========================================

Calculates comprehensive price metrics and moving averages for index overview.
Generates overview_template_part1.csv with price + MA data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def calculate_moving_averages(df: pd.DataFrame, periods: List[int] = None) -> Dict[str, float]:
    """
    Calculate various moving averages for a ticker.
    
    Args:
        df: DataFrame with OHLCV data
        periods: List of MA periods to calculate
        
    Returns:
        Dict with MA calculations
    """
    if periods is None:
        periods = [5, 10, 20, 50, 100, 200]
    
    if df is None or df.empty or 'Close' not in df.columns:
        return {}
    
    close = df['Close']
    ma_data = {}
    
    # Calculate Simple Moving Averages
    for period in periods:
        if len(df) >= period:
            ma_data[f'sma_{period}'] = close.rolling(window=period).mean().iloc[-1]
            
    # Calculate Exponential Moving Averages
    for period in [10, 20, 50]:
        if len(df) >= period:
            ma_data[f'ema_{period}'] = close.ewm(span=period).mean().iloc[-1]
    
    # Calculate MA slopes (trend direction)
    for period in [20, 50]:
        if len(df) >= period + 5:
            ma_series = close.rolling(window=period).mean()
            if len(ma_series.dropna()) >= 5:
                recent_ma = ma_series.tail(5)
                slope = (recent_ma.iloc[-1] - recent_ma.iloc[0]) / len(recent_ma)
                ma_data[f'sma_{period}_slope'] = slope
    
    return ma_data


def calculate_price_metrics(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """
    Calculate comprehensive price metrics for a ticker.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Ticker symbol
        
    Returns:
        Dict with price metrics
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return {'ticker': ticker, 'error': 'No data available'}
    
    close = df['Close']
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series([0] * len(df))
    
    current_price = close.iloc[-1]
    
    # Basic price metrics
    metrics = {
        'ticker': ticker,
        'current_price': current_price,
        'data_points': len(df),
    }
    
    # Returns calculation
    return_periods = [5, 10, 22, 66, 132, 252]  # 1w, 2w, 1m, 3m, 6m, 1y
    return_names = ['1w', '2w', '1m', '3m', '6m', '1y']
    
    for period, name in zip(return_periods, return_names):
        if len(df) >= period + 1:
            past_price = close.iloc[-(period+1)]
            metrics[f'return_{name}'] = (current_price / past_price) - 1
        else:
            metrics[f'return_{name}'] = np.nan
    
    # Volatility (annualized)
    if len(df) >= 22:
        daily_returns = close.pct_change().dropna()
        if len(daily_returns) > 0:
            metrics['volatility'] = daily_returns.std() * np.sqrt(252)
        else:
            metrics['volatility'] = np.nan
    else:
        metrics['volatility'] = np.nan
    
    # Volume metrics
    if len(df) >= 20 and 'Volume' in df.columns:
        avg_volume_20d = volume.tail(20).mean()
        current_volume = volume.iloc[-1]
        metrics['volume_20d_avg'] = avg_volume_20d
        metrics['volume_ratio'] = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
    else:
        metrics['volume_20d_avg'] = np.nan
        metrics['volume_ratio'] = np.nan
    
    # Price position metrics
    if len(df) >= 52:
        high_52w = high.tail(252).max()
        low_52w = low.tail(252).min()
        metrics['high_52w'] = high_52w
        metrics['low_52w'] = low_52w
        metrics['price_vs_52w_high'] = (current_price / high_52w) - 1
        metrics['price_vs_52w_low'] = (current_price / low_52w) - 1
    else:
        metrics['high_52w'] = np.nan
        metrics['low_52w'] = np.nan
        metrics['price_vs_52w_high'] = np.nan
        metrics['price_vs_52w_low'] = np.nan
    
    # RSI calculation
    if len(df) >= 15:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) and rs.iloc[-1] > 0 else 50
        metrics['rsi'] = rsi
    else:
        metrics['rsi'] = np.nan
    
    # Add moving averages
    ma_data = calculate_moving_averages(df)
    metrics.update(ma_data)
    
    return metrics


def generate_overview_template_part1(batch_data: Dict[str, pd.DataFrame], 
                                   output_path: Path, 
                                   user_config=None) -> str:
    """
    Generate overview_template_part1.csv with price and MA data.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        output_path: Path to save the CSV file
        user_config: User configuration
        
    Returns:
        str: Path to generated file
    """
    logger.info(f"Generating overview template part 1 for {len(batch_data)} tickers")
    
    all_metrics = []
    
    # Process each ticker
    for ticker, df in batch_data.items():
        try:
            metrics = calculate_price_metrics(df, ticker)
            all_metrics.append(metrics)
        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")
            all_metrics.append({'ticker': ticker, 'error': str(e)})
    
    # Create DataFrame and save to CSV
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        
        # Sort by ticker
        results_df = results_df.sort_values('ticker')
        
        # Generate output file
        output_file = output_path / 'overview_template_part1.csv'
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_file, index=False, float_format='%.4f')
        
        logger.info(f"Overview template part 1 saved: {output_file}")
        print(f"📊 Overview template part 1 generated: {output_file.name}")
        print(f"  • {len(results_df)} tickers processed")
        print(f"  • {len([c for c in results_df.columns if c != 'ticker'])} metrics calculated")
        
        return str(output_file)
    else:
        logger.warning("No metrics calculated for overview template")
        return ""


def get_price_ma_summary(batch_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Get summary statistics for price and MA data.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        
    Returns:
        Dict with summary statistics
    """
    if not batch_data:
        return {}
    
    all_returns = []
    all_volatilities = []
    all_rsi = []
    
    for ticker, df in batch_data.items():
        try:
            metrics = calculate_price_metrics(df, ticker)
            if 'return_1m' in metrics and not np.isnan(metrics['return_1m']):
                all_returns.append(metrics['return_1m'])
            if 'volatility' in metrics and not np.isnan(metrics['volatility']):
                all_volatilities.append(metrics['volatility'])
            if 'rsi' in metrics and not np.isnan(metrics['rsi']):
                all_rsi.append(metrics['rsi'])
        except Exception:
            continue
    
    summary = {
        'total_tickers': len(batch_data),
        'avg_return_1m': np.mean(all_returns) if all_returns else 0,
        'avg_volatility': np.mean(all_volatilities) if all_volatilities else 0,
        'avg_rsi': np.mean(all_rsi) if all_rsi else 50,
        'tickers_with_data': len(all_returns)
    }
    
    return summary