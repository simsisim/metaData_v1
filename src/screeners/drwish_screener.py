"""
Dr. Wish Suite Screener Module
===============================

Implements Dr. Eric Wish's trading system methodologies including:
1. GLB (Green Line Breakout) - Momentum breakout signals
2. Blue Dot - Oversold bounce with trend confirmation  
3. Black Dot - Oversold bounce with stronger trend criteria

This screener integrates all three strategies with optimized batch processing
for heavy pivot and stochastic calculations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class DrWishCalculator:
    """
    Optimized Dr. Wish suite calculator with batch processing for heavy calculations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeframe = config.get('timeframe', 'daily')

        # Timeframe multipliers for adjusting parameters
        self.timeframe_multipliers = {
            'daily': 1.0,
            'weekly': 0.2,  # 1 week = ~5 days, so 1/5 = 0.2
            'monthly': 0.05  # 1 month = ~21 days, so 1/21 ≈ 0.05
        }

        multiplier = self.timeframe_multipliers.get(self.timeframe, 1.0)

        # GLB parameters (adjusted for timeframe)
        base_pivot_strength = int(config.get('pivot_strength', 10))
        self.pivot_strength = max(1, int(base_pivot_strength * multiplier))
        self.lookback_period = config.get('lookback_period', '3m')
        self.calculate_historical_GLB = config.get('calculate_historical_GLB', '1y')
        self.confirmation_period = config.get('confirmation_period', '2w')
        self.require_confirmation = config.get('require_confirmation', True)

        # Blue Dot parameters (adjusted for timeframe)
        base_stoch_period = int(config.get('blue_dot_stoch_period', 10))
        base_sma_period = int(config.get('blue_dot_sma_period', 50))
        self.blue_dot_stoch_period = max(2, int(base_stoch_period * multiplier))
        self.blue_dot_stoch_threshold = float(config.get('blue_dot_stoch_threshold', 20.0))
        self.blue_dot_sma_period = max(3, int(base_sma_period * multiplier))

        # Black Dot parameters (adjusted for timeframe)
        base_black_stoch_period = int(config.get('black_dot_stoch_period', 10))
        base_black_sma_period = int(config.get('black_dot_sma_period', 30))
        base_black_ema_period = int(config.get('black_dot_ema_period', 21))
        self.black_dot_stoch_period = max(2, int(base_black_stoch_period * multiplier))
        self.black_dot_stoch_threshold = float(config.get('black_dot_stoch_threshold', 25.0))
        self.black_dot_lookback = max(1, int(config.get('black_dot_lookback', 3) * multiplier))
        self.black_dot_sma_period = max(3, int(base_black_sma_period * multiplier))
        self.black_dot_ema_period = max(3, int(base_black_ema_period * multiplier))

        # General parameters (volume adjusted for timeframe)
        self.min_price = float(config.get('min_price', 5.0))
        base_min_volume = int(config.get('min_volume', 100000))

        # Adjust volume threshold for timeframe (weekly/monthly have aggregated volume)
        volume_divisors = {'daily': 1, 'weekly': 3, 'monthly': 10}  # Conservative scaling
        self.min_volume = base_min_volume // volume_divisors.get(self.timeframe, 1)

        # Minimum data requirements (adjusted for timeframe)
        self.min_data_points = {'daily': 100, 'weekly': 30, 'monthly': 12}.get(self.timeframe, 100)

        logger.info(f"DrWish calculator initialized for {self.timeframe} timeframe")
        logger.info(f"Adjusted parameters: pivot_strength={self.pivot_strength}, stoch_period={self.blue_dot_stoch_period}, "
                   f"sma_period={self.blue_dot_sma_period}, min_volume={self.min_volume}, min_data_points={self.min_data_points}")

    def calculate_stochastic(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                           period: int = 10) -> pd.Series:
        """Calculate Stochastic %K indicator (optimized for batch processing)"""
        try:
            lowest_low = lows.rolling(window=period, min_periods=period).min()
            highest_high = highs.rolling(window=period, min_periods=period).max()
            
            # Avoid division by zero
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, np.nan)
            
            stoch_k = 100 * (closes - lowest_low) / denominator
            return stoch_k.fillna(0)
        except Exception as e:
            logger.warning(f"Stochastic calculation error: {e}")
            return pd.Series(0, index=closes.index)

    def is_pivot_high(self, highs: pd.Series, index: int, strength: int) -> bool:
        """Detect pivot high with specified strength (exact PineScript logic)"""
        try:
            if index < strength or index >= len(highs) - strength:
                return False
            
            center_high = highs.iloc[index]
            
            # Check left side
            for i in range(max(0, index - strength), index):
                if highs.iloc[i] >= center_high:
                    return False
            
            # Check right side  
            for i in range(index + 1, min(len(highs), index + strength + 1)):
                if highs.iloc[i] > center_high:
                    return False
                    
            return True
        except Exception:
            return False

    def get_lookback_bars(self, period_str: str, timeframe: str, df_length: int = None) -> int:
        """Convert period string to number of bars"""
        multipliers = {
            'daily': {'w': 5, 'm': 21, 'y': 252},
            'weekly': {'w': 1, 'm': 4, 'y': 52}, 
            'monthly': {'w': 0.25, 'm': 1, 'y': 12}
        }
        
        try:
            # Handle special cases
            if period_str.lower() == 'complete':
                # Use all available data (with reasonable minimum)
                return max(df_length - 20, 100) if df_length else 500
            
            if period_str[-1] in ['w', 'm', 'y']:
                unit = period_str[-1]
                value = int(period_str[:-1])
                calculated_bars = int(value * multipliers[timeframe][unit])
                
                # For very long periods, ensure we don't exceed available data
                if df_length and calculated_bars > df_length - 20:
                    return max(df_length - 20, 100)
                return calculated_bars
                
            return int(period_str)
        except:
            return 63  # Default 3m for daily

    def calculate_historical_glb_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculate all historical GLB levels within the historical calculation period
        Similar to original GLBRecordManager approach
        
        Args:
            df: Price data DataFrame
            
        Returns:
            List of GLB level records with detection/breakout dates
        """
        try:
            if len(df) < self.min_data_points:
                return []
            
            highs = df['High']
            closes = df['Close']
            
            # Calculate bars for historical GLB calculation period
            historical_bars = self.get_lookback_bars(self.calculate_historical_GLB, self.timeframe, len(df))
            lookback_bars = self.get_lookback_bars(self.lookback_period, self.timeframe, len(df))
            confirmation_bars = self.get_lookback_bars(self.confirmation_period, self.timeframe, len(df))
            
            # Start from the historical period
            start_bar = max(self.pivot_strength, len(df) - historical_bars)
            
            # Track all GLB levels found
            glb_records = []
            
            # Find all GLB levels in historical period
            for bar_index in range(start_bar, len(df) - self.pivot_strength):
                
                # Detect pivot high at current bar
                if self.is_pivot_high(highs, bar_index, self.pivot_strength):
                    
                    # Find highest high in lookback period from this bar
                    start_idx = max(0, bar_index - lookback_bars)
                    lookback_slice = highs.iloc[start_idx:bar_index+1]
                    
                    if not lookback_slice.empty:
                        highest_high = lookback_slice.max()
                        current_high = highs.iloc[bar_index]
                        
                        # If current pivot is the highest in lookback period, it's a GLB
                        if current_high >= highest_high:
                            
                            # Find when this GLB gets broken (if at all)
                            breakout_bar = None
                            breakout_date = None
                            
                            # Look forward to find breakout
                            for future_bar in range(bar_index + confirmation_bars, len(df)):
                                if closes.iloc[future_bar] > current_high:
                                    breakout_bar = future_bar
                                    breakout_date = df.index[future_bar]
                                    break
                            
                            # Check if confirmed (wasn't broken during confirmation)
                            is_confirmed = True
                            if self.require_confirmation:
                                conf_end = min(bar_index + confirmation_bars, len(df))
                                for check_bar in range(bar_index + 1, conf_end):
                                    if highs.iloc[check_bar] > current_high:
                                        is_confirmed = False
                                        break
                            
                            # Ensure timezone-naive dates
                            detection_dt = df.index[bar_index]
                            if hasattr(detection_dt, 'tz_convert') and detection_dt.tz is not None:
                                detection_dt = detection_dt.tz_convert(None)
                            
                            breakout_dt = breakout_date
                            if breakout_dt and hasattr(breakout_dt, 'tz_convert') and breakout_dt.tz is not None:
                                breakout_dt = breakout_dt.tz_convert(None)
                            
                            glb_records.append({
                                'glb_level': float(current_high),
                                'detection_date': detection_dt,
                                'detection_bar': int(bar_index),
                                'breakout_date': breakout_dt,
                                'breakout_bar': int(breakout_bar) if breakout_bar else None,
                                'is_confirmed': bool(is_confirmed),
                                'is_broken': bool(breakout_bar is not None)
                            })
            
            return glb_records
            
        except Exception as e:
            logger.warning(f"Historical GLB calculation error: {e}")
            return []

    def detect_glb_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect GLB (Green Line Breakout) signals with batch optimization"""
        try:
            if len(df) < self.min_data_points:  # Need sufficient data
                return pd.DataFrame()
            
            highs = df['High']
            closes = df['Close']
            volumes = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=df.index)
            
            lookback_bars = self.get_lookback_bars(self.lookback_period, self.timeframe, len(df))
            confirmation_bars = self.get_lookback_bars(self.confirmation_period, self.timeframe, len(df))
            
            # Find pivot highs (batch processing for efficiency)
            pivot_indices = []
            for i in range(self.pivot_strength, len(highs) - self.pivot_strength):
                if self.is_pivot_high(highs, i, self.pivot_strength):
                    pivot_indices.append(i)
            
            glb_signals = []
            
            for i in range(lookback_bars, len(df)):
                current_date = df.index[i]
                current_high = highs.iloc[i]
                current_close = closes.iloc[i]
                current_volume = volumes.iloc[i]
                
                # Find relevant pivots in lookback period
                start_idx = max(0, i - lookback_bars)
                relevant_pivots = [p for p in pivot_indices if start_idx <= p < i]
                
                if not relevant_pivots:
                    continue
                
                # Find GLB (highest pivot in lookback period)
                glb_idx = max(relevant_pivots, key=lambda x: highs.iloc[x])
                glb_level = highs.iloc[glb_idx]
                
                # Collect all pivot levels for chart display
                all_pivot_levels = [{'date': df.index[p], 'level': highs.iloc[p], 'is_glb': p == glb_idx} 
                                  for p in relevant_pivots]
                glb_date = df.index[glb_idx]
                
                # Check if confirmation period has passed
                bars_since_pivot = i - glb_idx
                is_confirmed = bars_since_pivot >= confirmation_bars if self.require_confirmation else True
                
                # Detect breakout
                if current_high > glb_level and is_confirmed:
                    signal_strength = (current_high - glb_level) / glb_level * 100
                    
                    # Get historical GLB levels for this signal
                    historical_glb_levels = self.calculate_historical_glb_levels(df)
                    
                    glb_signals.append({
                        'date': current_date,
                        'glb_level': glb_level,
                        'glb_date': glb_date,
                        'breakout_high': current_high,
                        'close': current_close,
                        'volume': current_volume,
                        'bars_since_pivot': bars_since_pivot,
                        'signal_strength': signal_strength,
                        'confirmed': is_confirmed,
                        'all_pivot_levels': all_pivot_levels,
                        'historical_glb_levels': historical_glb_levels
                    })
            
            return pd.DataFrame(glb_signals)
            
        except Exception as e:
            logger.warning(f"GLB calculation error: {e}")
            return pd.DataFrame()

    def detect_blue_dot_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Blue Dot oversold bounce signals"""
        try:
            min_required = max(self.blue_dot_stoch_period, self.blue_dot_sma_period, self.min_data_points)
            if len(df) < min_required:
                return pd.DataFrame()
            
            highs = df['High']
            lows = df['Low'] 
            closes = df['Close']
            volumes = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=df.index)
            
            # Calculate indicators
            stoch_k = self.calculate_stochastic(highs, lows, closes, self.blue_dot_stoch_period)
            sma = closes.rolling(window=self.blue_dot_sma_period).mean()
            sma_rising = sma.diff() > 0
            
            blue_dot_signals = []
            
            for i in range(1, len(df)):
                current_date = df.index[i]
                
                # Blue Dot conditions
                stoch_yesterday = stoch_k.iloc[i-1] if i > 0 else 0
                stoch_today = stoch_k.iloc[i]
                sma_is_rising = sma_rising.iloc[i]
                
                if (stoch_yesterday < self.blue_dot_stoch_threshold and 
                    stoch_today > self.blue_dot_stoch_threshold and
                    sma_is_rising):
                    
                    blue_dot_signals.append({
                        'date': current_date,
                        'close': closes.iloc[i],
                        'volume': volumes.iloc[i],
                        'stoch_yesterday': stoch_yesterday,
                        'stoch_today': stoch_today,
                        'sma_50': sma.iloc[i],
                        'sma_rising': sma_is_rising
                    })
            
            return pd.DataFrame(blue_dot_signals)
            
        except Exception as e:
            logger.warning(f"Blue Dot calculation error: {e}")
            return pd.DataFrame()

    def detect_black_dot_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Black Dot oversold bounce signals"""
        try:
            min_required = max(self.black_dot_stoch_period, self.black_dot_sma_period,
                              self.black_dot_ema_period, self.min_data_points)
            if len(df) < min_required:
                return pd.DataFrame()
            
            highs = df['High']
            lows = df['Low']
            closes = df['Close']
            volumes = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=df.index)
            
            # Calculate indicators
            stoch_k = self.calculate_stochastic(highs, lows, closes, self.black_dot_stoch_period)
            sma = closes.rolling(window=self.black_dot_sma_period).mean()
            ema = closes.ewm(span=self.black_dot_ema_period).mean()
            
            black_dot_signals = []
            
            for i in range(self.black_dot_lookback, len(df)):
                current_date = df.index[i]
                current_close = closes.iloc[i]
                previous_close = closes.iloc[i-1] if i > 0 else current_close
                
                # Check if stochastic was <= threshold in lookback period
                lookback_start = max(0, i - self.black_dot_lookback)
                stoch_lookback = stoch_k.iloc[lookback_start:i]
                was_oversold = (stoch_lookback <= self.black_dot_stoch_threshold).any()
                
                # Black Dot conditions
                closing_higher = current_close > previous_close
                above_sma = current_close > sma.iloc[i]
                above_ema = current_close > ema.iloc[i]
                trend_confirmed = above_sma or above_ema
                
                if was_oversold and closing_higher and trend_confirmed:
                    min_stoch = stoch_lookback.min()
                    
                    black_dot_signals.append({
                        'date': current_date,
                        'close': current_close,
                        'volume': volumes.iloc[i],
                        'min_stoch_lookback': min_stoch,
                        'closing_higher': closing_higher,
                        'above_sma': above_sma,
                        'above_ema': above_ema,
                        'sma_30': sma.iloc[i],
                        'ema_21': ema.iloc[i]
                    })
            
            return pd.DataFrame(black_dot_signals)
            
        except Exception as e:
            logger.warning(f"Black Dot calculation error: {e}")
            return pd.DataFrame()


def drwish_screener(batch_data, params=None):
    """
    Run Dr. Wish suite screener on batch data.

    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with screener parameters

    Returns:
        list: List of screening results with ticker, strategy, and signal data
    """
    if not batch_data:
        logger.warning("DrWish screener: No batch data provided")
        return []

    # Default parameters
    default_params = {
        'timeframe': 'daily',  # Add timeframe parameter
        'min_price': 5.0,
        'min_volume': 100000,
        'pivot_strength': 10,
        'lookback_period': '3m',
        'confirmation_period': '2w',
        'require_confirmation': True,
        'blue_dot_stoch_period': 10,
        'blue_dot_stoch_threshold': 20.0,
        'blue_dot_sma_period': 50,
        'black_dot_stoch_period': 10,
        'black_dot_stoch_threshold': 25.0,
        'black_dot_lookback': 3,
        'black_dot_sma_period': 30,
        'black_dot_ema_period': 21,
        'enable_glb': True,
        'enable_blue_dot': True,
        'enable_black_dot': True,
        'show_all_stocks': False
    }
    
    if params:
        default_params.update(params)
    
    logger.info(f"Running DrWish screener on {len(batch_data)} tickers")
    
    try:
        calculator = DrWishCalculator(default_params)
        results = []
        analysis_date = datetime.now()
        
        for ticker, df in batch_data.items():
            if df is None or df.empty:
                continue
                
            try:
                # Basic filters (use calculator's adjusted requirements)
                if len(df) < calculator.min_data_points:
                    continue

                latest_price = df['Close'].iloc[-1]
                latest_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0

                if latest_price < calculator.min_price or latest_volume < calculator.min_volume:
                    continue
                
                # Run GLB analysis
                if default_params['enable_glb']:
                    glb_signals = calculator.detect_glb_signals(df)
                    
                    for _, signal in glb_signals.iterrows():
                        # Calculate days since signal
                        days_since = (analysis_date.date() - signal['date'].date()).days
                        
                        if days_since <= 30:  # Recent signals only
                            score = 70 + min(signal['signal_strength'], 30)  # Base 70 + strength bonus
                            
                            results.append({
                                'ticker': ticker,
                                'screen_type': 'drwish_glb',
                                'score': round(score, 2),
                                'price': round(latest_price, 2),
                                'volume': int(latest_volume),
                                'signal_date': signal['date'].strftime('%Y-%m-%d'),
                                'glb_level': round(signal['glb_level'], 2),
                                'breakout_high': round(signal['breakout_high'], 2),
                                'signal_strength': round(signal['signal_strength'], 2),
                                'bars_since_pivot': int(signal['bars_since_pivot']),
                                'confirmed': signal['confirmed'],
                                'days_since_signal': days_since,
                                'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                                'screener': 'DrWish GLB'
                            })
                
                # Run Blue Dot analysis
                if default_params['enable_blue_dot']:
                    blue_signals = calculator.detect_blue_dot_signals(df)
                    
                    for _, signal in blue_signals.iterrows():
                        days_since = (analysis_date.date() - signal['date'].date()).days
                        
                        if days_since <= 10:  # Very recent signals
                            score = 60 + (calculator.blue_dot_stoch_threshold - signal['stoch_yesterday'])
                            
                            results.append({
                                'ticker': ticker,
                                'screen_type': 'drwish_blue_dot',
                                'score': round(score, 2),
                                'price': round(latest_price, 2),
                                'volume': int(latest_volume),
                                'signal_date': signal['date'].strftime('%Y-%m-%d'),
                                'stoch_yesterday': round(signal['stoch_yesterday'], 2),
                                'stoch_today': round(signal['stoch_today'], 2),
                                'sma_50': round(signal['sma_50'], 2),
                                'sma_rising': signal['sma_rising'],
                                'days_since_signal': days_since,
                                'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                                'screener': 'DrWish Blue Dot'
                            })
                
                # Run Black Dot analysis
                if default_params['enable_black_dot']:
                    black_signals = calculator.detect_black_dot_signals(df)
                    
                    for _, signal in black_signals.iterrows():
                        days_since = (analysis_date.date() - signal['date'].date()).days
                        
                        if days_since <= 10:  # Very recent signals
                            score = 65 + (calculator.black_dot_stoch_threshold - signal['min_stoch_lookback'])
                            
                            results.append({
                                'ticker': ticker,
                                'screen_type': 'drwish_black_dot',
                                'score': round(score, 2),
                                'price': round(latest_price, 2),
                                'volume': int(latest_volume),
                                'signal_date': signal['date'].strftime('%Y-%m-%d'),
                                'min_stoch_lookback': round(signal['min_stoch_lookback'], 2),
                                'above_sma': signal['above_sma'],
                                'above_ema': signal['above_ema'],
                                'sma_30': round(signal['sma_30'], 2),
                                'ema_21': round(signal['ema_21'], 2),
                                'days_since_signal': days_since,
                                'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                                'screener': 'DrWish Black Dot'
                            })
                
                # Generate individual ticker result files if enabled
                _save_individual_ticker_results(ticker, calculator, df, default_params)
                
            except Exception as e:
                logger.warning(f"DrWish screener: Error processing {ticker}: {e}")
                continue
        
        # Filter results if not showing all stocks
        if not default_params['show_all_stocks']:
            # Keep only top scoring signals from each strategy
            glb_results = [r for r in results if r['screen_type'] == 'drwish_glb']
            blue_results = [r for r in results if r['screen_type'] == 'drwish_blue_dot']
            black_results = [r for r in results if r['screen_type'] == 'drwish_black_dot']
            
            # Top 20 from each strategy
            glb_top = sorted(glb_results, key=lambda x: x['score'], reverse=True)[:20]
            blue_top = sorted(blue_results, key=lambda x: x['score'], reverse=True)[:20]
            black_top = sorted(black_results, key=lambda x: x['score'], reverse=True)[:20]
            
            results = glb_top + blue_top + black_top
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"DrWish screener completed: {len(results)} results generated")
        
        # Save results in DrWish-specific format if results exist
        if results:
            _save_drwish_results(results, default_params)
            
            # Generate charts if enabled
            if default_params.get('enable_charts', False):
                try:
                    from .drwish_charts import generate_drwish_charts
                    chart_paths = generate_drwish_charts(results, batch_data, default_params)
                    logger.info(f"Generated {len(chart_paths)} DrWish charts")
                except ImportError:
                    logger.warning("Chart generation requested but matplotlib not available")
                except Exception as e:
                    logger.warning(f"Chart generation failed: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"DrWish screener failed: {e}")
        return []


def _save_individual_ticker_results(ticker: str, calculator: DrWishCalculator, df: pd.DataFrame, 
                                   params: Dict[str, Any]) -> None:
    """
    Save individual ticker result files matching original format
    
    Args:
        ticker: Stock ticker symbol
        calculator: DrWish calculator instance
        df: Price data for ticker
        params: Screener parameters
    """
    if not params.get('generate_individual_files', False):
        return
        
    try:
        # Create individual results directory
        timeframe = params.get('timeframe', 'daily')
        user_choice = params.get('ticker_choice', 0)
        timestamp = datetime.now().strftime('%Y%m%d')
        
        results_dir = Path("results/screeners/drwish/individual")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate GLB results file
        historical_glbs = calculator.calculate_historical_glb_levels(df)
        if historical_glbs:
            glb_df = pd.DataFrame(historical_glbs)
            glb_df = glb_df[[
                'glb_level', 'detection_date', 'detection_bar', 
                'breakout_date', 'breakout_bar', 'is_confirmed', 'is_broken'
            ]].copy()
            # Add confirmation_date and confirmation_bar columns to match original
            glb_df['confirmation_date'] = glb_df['detection_date']  # Placeholder
            glb_df['confirmation_bar'] = -1  # Placeholder
            
            # Reorder columns to match original
            glb_df = glb_df[[
                'glb_level', 'detection_date', 'detection_bar',
                'confirmation_date', 'confirmation_bar', 
                'breakout_date', 'breakout_bar', 'is_confirmed', 'is_broken'
            ]]
            glb_df.columns = ['level', 'detection_date', 'detection_bar', 
                             'confirmation_date', 'confirmation_bar',
                             'breakout_date', 'breakout_bar', 'is_confirmed', 'is_broken']
            
            glb_file = results_dir / f"{ticker}_glb_results.csv"
            glb_df.to_csv(glb_file, index=False)
            logger.info(f"Individual GLB results saved: {glb_file}")
        
        # Generate Blue Dot results file
        blue_signals = calculator.detect_blue_dot_signals(df)
        if not blue_signals.empty:
            blue_df = blue_signals.copy()
            blue_df['signal_number'] = range(1, len(blue_df) + 1)
            # Add bar_index (placeholder) and price_level to match original format
            blue_df['bar_index'] = range(len(blue_df))  # Placeholder bar index
            blue_df['price_level'] = blue_df['close']  # Use close as price level
            blue_df = blue_df[['signal_number', 'date', 'bar_index', 'price_level', 'close', 'stoch_yesterday', 'sma_50']].copy()
            blue_df.columns = ['signal_number', 'date', 'bar_index', 'price_level', 'close_price', 'stochastic_value', 'sma_value']
            
            blue_file = results_dir / f"{ticker}_blue_dot_results.csv"
            blue_df.to_csv(blue_file, index=False)
            logger.info(f"Individual Blue Dot results saved: {blue_file}")
        
        # Generate Black Dot results file  
        black_signals = calculator.detect_black_dot_signals(df)
        if not black_signals.empty:
            black_df = black_signals.copy()
            black_df['signal_number'] = range(1, len(black_df) + 1)
            # Add bar_index (placeholder) and price_level to match original format
            black_df['bar_index'] = range(len(black_df))  # Placeholder bar index
            black_df['price_level'] = black_df['close']  # Use close as price level
            black_df = black_df[['signal_number', 'date', 'bar_index', 'price_level', 'close', 'min_stoch_lookback', 'sma_30', 'ema_21']].copy()
            black_df.columns = ['signal_number', 'date', 'bar_index', 'price_level', 'close_price', 'stochastic_value', 'sma_value', 'ema_value']
            
            black_file = results_dir / f"{ticker}_black_dot_results.csv"
            black_df.to_csv(black_file, index=False)
            logger.info(f"Individual Black Dot results saved: {black_file}")
            
    except Exception as e:
        logger.error(f"Failed to save individual results for {ticker}: {e}")


def _save_drwish_results(results: List[Dict], params: Optional[Dict] = None) -> None:
    """
    Save DrWish screener results in the expected format.
    
    Args:
        results: List of screening results
        params: Dictionary with screener parameters
    """
    if not params:
        logger.warning("No parameters provided for DrWish results saving")
        return
        
    try:
        # Extract parameters
        user_choice = params.get('ticker_choice', 0)
        timeframe = params.get('timeframe', 'daily')
        
        # Create drwish subdirectory in screeners folder
        screeners_dir = Path("results/screeners")
        drwish_dir = screeners_dir / 'drwish'
        drwish_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timeframe to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"results_drwish_{timeframe}_{user_choice}_{timestamp}.csv"
        output_file = drwish_dir / filename
        
        if results:
            # Convert to DataFrame and save
            df = pd.DataFrame(results)
            
            # Reorder columns for readability
            column_order = [
                'ticker', 'screen_type', 'price', 'volume', 'score',
                'signal_date', 'days_since_signal', 'analysis_date', 'screener'
            ]
            
            # Add strategy-specific columns that exist
            for col in df.columns:
                if col not in column_order:
                    column_order.append(col)
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in df.columns]
            df_ordered = df[available_columns]
            
            df_ordered.to_csv(output_file, index=False)
            logger.info(f"DrWish results saved: {output_file}")
            
            # Save strategy-specific files
            for strategy in ['drwish_glb', 'drwish_blue_dot', 'drwish_black_dot']:
                strategy_results = df[df['screen_type'] == strategy]
                if not strategy_results.empty:
                    strategy_file = drwish_dir / f"{strategy}_{timeframe}_{user_choice}_{timestamp}.csv"
                    strategy_results.to_csv(strategy_file, index=False)
                    logger.info(f"DrWish {strategy} results saved: {strategy_file}")
        else:
            # Create empty file with headers
            headers = [
                'ticker', 'screen_type', 'price', 'volume', 'score',
                'signal_date', 'days_since_signal', 'analysis_date', 'screener'
            ]
            
            empty_df = pd.DataFrame(columns=headers)
            empty_df.to_csv(output_file, index=False)
            logger.info(f"DrWish results (empty) saved: {output_file}")
            
    except Exception as e:
        logger.error(f"Failed to save DrWish results: {e}")


# Maintain module interface consistency
__all__ = ['drwish_screener', '_save_drwish_results', 'DrWishCalculator']