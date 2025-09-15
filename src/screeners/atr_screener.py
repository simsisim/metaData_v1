"""
ATR2 Screener - Volatility Analysis (formerly ATR Screener)
==========================================================

Implements volatility-based screening using ATR and ATRext calculations.
Based on Steve Jacobs' ATR methodology with ATR Extension analysis.
Uses Wilder smoothing method for ATR calculation.

Screening Logic:
- High Volatility: ATR percentile > 80% (breakout potential)
- Low Volatility: ATR percentile < 20% (compression setups)
- Extended Above: ATRext_$ > +2.0 (overbought)
- Extended Below: ATRext_$ < -2.0 (oversold)
- Optimal Entry: ATRext between -1.0 and +1.0 (balanced)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Import existing ATR calculation function
from ..basic_calculations import calculate_atr_and_atrext

logger = logging.getLogger(__name__)


def _calculate_atr_screening_score(atr_metrics: Dict, screening_type: str) -> float:
    """
    Calculate ATR screening score based on volatility and extension metrics.
    
    Args:
        atr_metrics: Dictionary with ATR calculation results
        screening_type: Type of ATR screening ('high_volatility', 'low_volatility', 'extended', 'optimal_entry')
        
    Returns:
        Numeric score for ranking (higher = better)
    """
    try:
        score = 0.0
        
        atr_pct = atr_metrics.get('atr_pct', 0)
        atrext_dollar = atr_metrics.get('atrext_dollar', 0)
        atr_percentile = atr_metrics.get('atr_percentile_100', 0.5)
        
        if screening_type == 'high_volatility':
            # High volatility screening - prefer high ATR percentile
            if atr_percentile > 0.8:
                score = (atr_percentile - 0.8) * 100 + atr_pct * 2
            
        elif screening_type == 'low_volatility':
            # Low volatility screening - prefer low ATR percentile (compression)
            if atr_percentile < 0.2:
                score = (0.2 - atr_percentile) * 100 + (5 - atr_pct)
                
        elif screening_type == 'extended_above':
            # Extended above SMA - overbought conditions
            if atrext_dollar > 2.0:
                score = atrext_dollar * 10 + atr_percentile * 20
                
        elif screening_type == 'extended_below':
            # Extended below SMA - oversold conditions  
            if atrext_dollar < -2.0:
                score = abs(atrext_dollar) * 10 + atr_percentile * 20
                
        elif screening_type == 'optimal_entry':
            # Optimal entry zone - balanced ATRext
            if -1.0 <= atrext_dollar <= 1.0:
                # Closer to zero is better
                distance_from_zero = abs(atrext_dollar)
                score = (1.0 - distance_from_zero) * 50 + atr_percentile * 20
        
        return max(score, 0)  # Ensure non-negative
        
    except Exception as e:
        logger.error(f"Error calculating ATR score: {e}")
        return 0.0


def atr2_screener(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    ATR-based screener for volatility and extension analysis.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with ATR parameters
        
    Returns:
        List of dictionaries with screening results sorted by score
    """
    if params is None:
        params = {
            'atr_period': 14,
            'sma_period': 50,
            'enable_percentile': True,
            'percentile_period': 100,
            'min_volume': 10000,
            'min_price': 1.0,
            'high_volatility_threshold': 0.8,
            'low_volatility_threshold': 0.2,
            'extended_threshold': 2.0,
            'optimal_range': 1.0
        }
    
    results = []
    
    logger.info(f"Running ATR2 screener on {len(batch_data)} tickers")
    
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
            min_length = max(params['atr_period'], params['sma_period'], params['percentile_period']) + 10
            if len(df) < min_length:
                logger.debug(f"Skipping {ticker}: Insufficient data ({len(df)} < {min_length})")
                continue
                
            # Current price and volume filters
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            if (current_price < params['min_price'] or 
                current_volume < params['min_volume']):
                continue
                
            # Calculate ATR metrics
            atr_metrics = calculate_atr_and_atrext(
                df,
                atr_period=params['atr_period'],
                sma_period=params['sma_period'],
                enable_percentile=params['enable_percentile'],
                percentile_period=params['percentile_period']
            )
            
            if not atr_metrics:
                continue
                
            # Determine screening categories and calculate scores
            atr_percentile = atr_metrics.get('atr_percentile_100', 0.5)
            atrext_dollar = atr_metrics.get('atrext_dollar', 0)
            
            screening_results = []
            
            # High Volatility Screening
            if atr_percentile > params['high_volatility_threshold']:
                score = _calculate_atr_screening_score(atr_metrics, 'high_volatility')
                if score > 0:
                    screening_results.append({
                        'screening_type': 'high_volatility',
                        'score': score,
                        'reason': f"ATR percentile {atr_percentile:.1%} (>80%)"
                    })
            
            # Low Volatility Screening
            if atr_percentile < params['low_volatility_threshold']:
                score = _calculate_atr_screening_score(atr_metrics, 'low_volatility')
                if score > 0:
                    screening_results.append({
                        'screening_type': 'low_volatility',
                        'score': score,
                        'reason': f"ATR percentile {atr_percentile:.1%} (<20%)"
                    })
            
            # Extended Above Screening
            if atrext_dollar > params['extended_threshold']:
                score = _calculate_atr_screening_score(atr_metrics, 'extended_above')
                if score > 0:
                    screening_results.append({
                        'screening_type': 'extended_above',
                        'score': score,
                        'reason': f"ATRext {atrext_dollar:.1f} (>+2.0)"
                    })
            
            # Extended Below Screening
            if atrext_dollar < -params['extended_threshold']:
                score = _calculate_atr_screening_score(atr_metrics, 'extended_below')
                if score > 0:
                    screening_results.append({
                        'screening_type': 'extended_below',
                        'score': score,
                        'reason': f"ATRext {atrext_dollar:.1f} (<-2.0)"
                    })
            
            # Optimal Entry Screening
            if abs(atrext_dollar) <= params['optimal_range']:
                score = _calculate_atr_screening_score(atr_metrics, 'optimal_entry')
                if score > 0:
                    screening_results.append({
                        'screening_type': 'optimal_entry',
                        'score': score,
                        'reason': f"ATRext {atrext_dollar:.1f} (balanced)"
                    })
            
            # Add results for each screening type that qualified
            for screen_result in screening_results:
                results.append({
                    'ticker': ticker,
                    'screen_type': 'atr',
                    'atr_screening_type': screen_result['screening_type'],
                    'score': screen_result['score'],
                    'reason': screen_result['reason'],
                    'current_price': current_price,
                    'atr': atr_metrics.get('atr', 0),
                    'atr_pct': atr_metrics.get('atr_pct', 0),
                    'atrext_dollar': atrext_dollar,
                    'atrext_percent': atr_metrics.get('atrext_percent', 0),
                    'atr_percentile': atr_percentile,
                    'sma_50': atr_metrics.get('sma_50', 0),
                    'volume': current_volume
                })
                    
        except Exception as e:
            logger.debug(f"Error processing {ticker} in ATR screener: {e}")
            continue
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"ATR2 screener found {len(results)} screening hits")
    return results


def _save_atr2_results(results: List[Dict], params: Optional[Dict] = None) -> None:
    """
    Save ATR screening results to dedicated output directory.
    
    Args:
        results: List of ATR screening results
        params: ATR parameters (contains ticker_choice and timeframe for filename)
    """
    try:
        if not results:
            return
            
        # Create DataFrame with ATR screening columns
        output_data = []
        for result in results:
            output_data.append({
                'Symbol': result['ticker'],
                'ATR_Screening_Type': result['atr_screening_type'],
                'Current_Price': result['current_price'],
                'ATR': result['atr'],
                'ATR_Pct': result['atr_pct'],
                'ATRext_Dollar': result['atrext_dollar'],
                'ATRext_Percent': result['atrext_percent'],
                'ATR_Percentile': result['atr_percentile'],
                'SMA_50': result['sma_50'],
                'Volume': result['volume'],
                'Score': result['score'],
                'Reason': result['reason']
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
        
        # Create filename with ATR2 prefix and timeframe if available
        if timeframe:
            filename = f'ATR2_{ticker_choice}_{date_str}_{timeframe}.csv'
        else:
            filename = f'ATR2_{ticker_choice}_{date_str}.csv'
        
        output_file = output_dir / filename
        
        # Save with proper formatting
        df.to_csv(output_file, index=False, float_format='%.3f')
        
        print(f"📊 ATR2 screening results saved to: {output_file}")
        print(f"📈 Total ATR2 screening hits: {len(results)}")
        
        # Show screening type breakdown
        type_counts = df['ATR_Screening_Type'].value_counts()
        for screen_type, count in type_counts.items():
            print(f"   • {screen_type}: {count}")
            
        logger.info(f"ATR2 screening results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving ATR2 screening results: {e}")
        print(f"❌ Error saving ATR2 screening results: {e}")


def _calculate_volatility_stops_atr_cloud_style(data: pd.DataFrame, length: int = 20, factor: float = 3.0, length2: int = 20, factor2: float = 1.5) -> Dict:
    """
    Calculate volatility stops using ATR cloud methodology from atr_cloud.py.
    This matches the original working implementation.
    
    Args:
        data: DataFrame with OHLCV data
        length: ATR length for vStop calculation
        factor: ATR multiplier for vStop
        length2: ATR length for vStop2 calculation
        factor2: ATR multiplier for vStop2
        
    Returns:
        Dictionary with volatility stop data matching atr_cloud.py format
    """
    try:
        from ..screeners.atr1_screener import calculate_atr_cloud
        
        # Use the exact ATR cloud calculation
        df_cloud = calculate_atr_cloud(data, length, factor, length2, factor2, 'Close', 'Close')
        
        # Find the most recent signal (exactly as in atr_cloud.py:149)
        signal_df = df_cloud[df_cloud['crossUp'] | df_cloud['crossDn']]
        
        if not signal_df.empty:
            # Get the most recent signal
            last_signal = signal_df.iloc[-1]
            signal_date = last_signal.name
            signal_close_price = last_signal['Close']
            current_price = df_cloud['Close'].iloc[-1]
            
            # Calculate signal metrics (exactly as in atr_cloud.py:152-154)
            signal_type = 'Long' if last_signal['crossUp'] else 'Short'
            price_change = ((current_price - signal_close_price) / signal_close_price) * 100
            days_since_signal = (df_cloud.index[-1] - signal_date).days
            
            return {
                'current_price': current_price,
                'signal': signal_type,
                'vStop': last_signal['vStop'],
                'vStop2': last_signal['vStop2'],
                'vstopseries': last_signal['vstopseries'],
                'signal_date': signal_date,
                'signal_close_price': signal_close_price,
                'price_change_pct': price_change,
                'days_since_signal': days_since_signal
            }
        else:
            # No signal found
            current_price = df_cloud['Close'].iloc[-1]
            return {
                'current_price': current_price,
                'signal': 'No Signal',
                'vStop': df_cloud['vStop'].iloc[-1],
                'vStop2': df_cloud['vStop2'].iloc[-1],
                'vstopseries': df_cloud['vstopseries'].iloc[-1],
                'signal_date': df_cloud.index[-1],  # Use latest date
                'signal_close_price': current_price,
                'price_change_pct': 0.0,
                'days_since_signal': 0
            }
        
    except Exception as e:
        logger.error(f"Error calculating volatility stops: {e}")
        return {}


def _save_atr_trailing_stop_results(results: List[Dict], params: Optional[Dict] = None) -> None:
    """
    Save ATR trailing stop results in the reference format.
    
    Args:
        results: List of ATR trailing stop results
        params: ATR parameters (contains ticker_choice and timeframe for filename)
    """
    try:
        if not results:
            return
            
        # Create DataFrame with reference format columns
        output_data = []
        for result in results:
            # Use actual calculated values (not simplified)
            price_change_pct = result.get('price_change_pct', 0.0)
            days_since_signal = result.get('days_since_signal', 0)
            
            output_data.append({
                'Symbol': result['ticker'],
                'Signal Date': result['signal_date'].strftime('%Y-%m-%d'),
                'Current Price': round(result['current_price'], 2),
                'Signal': result['signal'],
                'vStop': round(result['vStop'], 2),
                'vStop2': round(result['vStop2'], 2),
                'vstopseries': round(result['vstopseries'], 2),
                'Price Change %': round(price_change_pct, 2),
                'Days Since Signal': days_since_signal
            })
        
        # Create DataFrame
        df = pd.DataFrame(output_data)
        
        # Create output directory and filename (same as screening results)
        output_dir = Path('results/screeners/atr')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ticker_choice and timeframe from params
        ticker_choice = params.get('ticker_choice', 0) if params else 0
        timeframe = params.get('timeframe', '') if params else ''
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Create filename with ATR2_trailing_stops prefix to differentiate
        if timeframe:
            filename = f'ATR2_trailing_stops_{ticker_choice}_{date_str}_{timeframe}.csv'
        else:
            filename = f'ATR2_trailing_stops_{ticker_choice}_{date_str}.csv'
        
        output_file = output_dir / filename
        
        # Save with exact format matching reference
        df.to_csv(output_file, index=False, float_format='%.2f')
        
        print(f"📊 ATR2 trailing stops saved to: {output_file}")
        print(f"📈 Total ATR2 trailing stop signals: {len(results)}")
        
        # Show signal breakdown
        signal_counts = df['Signal'].value_counts()
        for signal_type, count in signal_counts.items():
            print(f"   • {signal_type}: {count}")
            
        logger.info(f"ATR2 trailing stops saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving ATR2 trailing stops: {e}")
        print(f"❌ Error saving ATR2 trailing stops: {e}")


def generate_atr_trailing_stops(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    Generate ATR-based trailing stop signals for all tickers.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with ATR parameters
        
    Returns:
        List of dictionaries with trailing stop results
    """
    if params is None:
        params = {
            'atr_period': 14,
            'atr_multiplier1': 2.0,
            'atr_multiplier2': 1.5,
            'min_volume': 10000,
            'min_price': 1.0
        }
    
    results = []
    
    logger.info(f"Generating ATR trailing stops for {len(batch_data)} tickers")
    
    for ticker, df in batch_data.items():
        try:
            # Data validation
            if df is None or df.empty:
                continue
                
            required_columns = ['Close', 'High', 'Low', 'Volume']
            if not all(col in df.columns for col in required_columns):
                continue
                
            # Minimum data length check
            if len(df) < params.get('atr_period', 14) + 10:
                continue
                
            # Current price and volume filters
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            if (current_price < params.get('min_price', 1.0) or 
                current_volume < params.get('min_volume', 10000)):
                continue
                
            # Calculate volatility stops using ATR cloud style
            stop_data = _calculate_volatility_stops_atr_cloud_style(
                df,
                length=params.get('atr_period', 20),
                factor=params.get('atr_multiplier1', 3.0),
                length2=params.get('atr_period', 20),
                factor2=params.get('atr_multiplier2', 1.5)
            )
            
            if stop_data:
                stop_data['ticker'] = ticker
                results.append(stop_data)
                    
        except Exception as e:
            logger.debug(f"Error processing {ticker} for ATR trailing stops: {e}")
            continue
    
    # Save results in reference format
    if results:
        _save_atr_trailing_stop_results(results, params)
    
    logger.info(f"Generated {len(results)} ATR trailing stop signals")
    return results


# Enhanced ATR2 screener that saves results automatically
def atr2_screener_with_output(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    ATR2 screener wrapper that automatically saves results to file.
    Also generates ATR trailing stop output in reference format.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with ATR2 parameters
        
    Returns:
        List of dictionaries with screening results
    """
    # Call the main screener logic
    results = atr2_screener(batch_data, params)
    
    # Save screening results if any found
    if results:
        _save_atr2_results(results, params)
    
    # Also generate ATR2 trailing stops output (reference format)
    trailing_params = params.copy() if params else {}
    trailing_params.update({
        'atr_multiplier1': 3.0,
        'atr_multiplier2': 1.5
    })
    
    trailing_stops = generate_atr_trailing_stops(batch_data, trailing_params)
    
    return results