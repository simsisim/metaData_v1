"""
Basic Calculations Module
========================

Performs fundamental technical analysis calculations on market data batches.
Includes price momentum, volume analysis, returns, and risk metrics.
"""

import pandas as pd
import numpy as np
import gc
from pathlib import Path
from datetime import datetime
import logging
# Stage analysis import removed - now handled by stage_analysis_processor.py
# Date normalization imports removed - using simple pandas operations

logger = logging.getLogger(__name__)


def find_latest_basic_calculation_file(config, timeframe, user_choice):
    """
    Find the latest basic_calculation file, trying date-stamped version first.
    Supports both hyphen (0-5) and underscore (0_5) formats for backward compatibility.
    
    Args:
        config: Config object with directories
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_choice: User ticker choice
        
    Returns:
        Path: Path to the most recent basic_calculation file or None if not found
    """
    from src.config import get_file_safe_user_choice
    
    # Try new hyphen format first (0-5), then fall back to legacy underscore format (0_5)
    user_choice_formats = [
        get_file_safe_user_choice(user_choice, preserve_hyphens=True),   # 0-5 format
        get_file_safe_user_choice(user_choice, preserve_hyphens=False)   # 0_5 format (legacy)
    ]
    
    from datetime import timedelta
    
    for user_choice_format in user_choice_formats:
        base_name = f'basic_calculation_{user_choice_format}_{timeframe}'
        
        # Try date-stamped version first (newest format) - check today and recent dates
        for days_back in range(7):  # Look back up to 7 days for date-stamped files
            check_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            dated_file = config.directories['BASIC_CALCULATION_DIR'] / f'{base_name}_{check_date}.csv'
            if dated_file.exists():
                return dated_file
        
        # Fall back to non-dated version (legacy format)
        legacy_file = config.directories['BASIC_CALCULATION_DIR'] / f'{base_name}.csv'
        if legacy_file.exists():
            return legacy_file
    
    # If no files found in either format
    return None


def calculate_returns(df):
    """Calculate various return metrics for a stock."""
    returns = {}
    
    if 'Close' in df.columns and len(df) > 1:
        # Daily returns
        returns['daily_returns'] = df['Close'].pct_change().dropna()
        
        # Cumulative returns
        returns['cumulative_return'] = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        # Volatility (annualized)
        returns['volatility'] = returns['daily_returns'].std() * np.sqrt(252)
        
        # Sharpe ratio approximation (assuming risk-free rate = 2%)
        excess_returns = returns['daily_returns'] - (0.02 / 252)
        returns['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns['daily_returns']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        returns['max_drawdown'] = drawdown.min()
        
    return returns


def calculate_candle_strength(df, timeframe='daily'):
    """
    Calculate Candle Strength indicator based on the relationship between
    candle body, wicks, and total range.
    
    Based on: https://www.tradingview.com/script/ecnJXleY-Candle-Strength/
    
    The indicator measures:
    1. Total candle area (High - Low)
    2. Upper and lower wick areas
    3. Body strength relative to total range
    4. Oppression factor of opposing forces
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        dict: Dictionary with candle strength metrics
    """
    candle_metrics = {}
    
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        return candle_metrics
    
    if len(df) == 0:
        return candle_metrics
    
    try:
        # Get latest candle data
        latest = df.iloc[-1]
        open_price = latest['Open']
        high_price = latest['High']
        low_price = latest['Low']
        close_price = latest['Close']
        
        # Calculate candle components
        total_range = high_price - low_price
        body_size = abs(close_price - open_price)
        
        # Determine candle direction
        is_bullish = close_price > open_price
        
        # Calculate wick sizes
        if is_bullish:
            upper_wick = high_price - close_price
            lower_wick = open_price - low_price
            body_top = close_price
            body_bottom = open_price
        else:
            upper_wick = high_price - open_price
            lower_wick = close_price - low_price
            body_top = open_price
            body_bottom = close_price
        
        # Avoid division by zero
        if total_range == 0:
            candle_metrics[f'{timeframe}_candle_strength'] = 0
            candle_metrics[f'{timeframe}_candle_strength_pct'] = 0
            candle_metrics[f'{timeframe}_body_ratio'] = 0
            candle_metrics[f'{timeframe}_upper_wick_ratio'] = 0
            candle_metrics[f'{timeframe}_lower_wick_ratio'] = 0
            candle_metrics[f'{timeframe}_candle_type'] = 'Doji'
            return candle_metrics
        
        # Calculate ratios
        body_ratio = body_size / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        
        # Calculate candle strength based on body dominance vs wick oppression
        # Strong candles have large bodies and small wicks
        # Weak candles have small bodies and large wicks
        
        # Primary strength: Body size relative to total range
        primary_strength = body_ratio
        
        # Oppression factor: How much the opposing side (wicks) reduces strength
        if is_bullish:
            # For bullish candles, lower wick represents selling pressure (oppression)
            oppression_factor = lower_wick_ratio
        else:
            # For bearish candles, upper wick represents buying pressure (oppression)
            oppression_factor = upper_wick_ratio
        
        # Final candle strength: Primary strength reduced by oppression
        candle_strength = primary_strength * (1 - oppression_factor)
        
        # Normalize to percentage
        candle_strength_pct = candle_strength * 100
        
        # Classify candle type
        if body_ratio > 0.7:
            candle_type = 'Strong Body'
        elif body_ratio > 0.4:
            candle_type = 'Moderate Body'
        elif body_ratio > 0.1:
            candle_type = 'Weak Body'
        else:
            candle_type = 'Doji'
        
        # Store metrics
        candle_metrics[f'{timeframe}_candle_strength'] = round(candle_strength, 4)
        candle_metrics[f'{timeframe}_candle_strength_pct'] = round(candle_strength_pct, 2)
        candle_metrics[f'{timeframe}_body_ratio'] = round(body_ratio, 4)
        candle_metrics[f'{timeframe}_upper_wick_ratio'] = round(upper_wick_ratio, 4)
        candle_metrics[f'{timeframe}_lower_wick_ratio'] = round(lower_wick_ratio, 4)
        candle_metrics[f'{timeframe}_oppression_factor'] = round(oppression_factor, 4)
        candle_metrics[f'{timeframe}_candle_type'] = candle_type
        candle_metrics[f'{timeframe}_is_bullish'] = is_bullish
        candle_metrics['total_range'] = round(total_range, 4)
        candle_metrics['body_size'] = round(body_size, 4)
        
        # Additional strength metrics
        candle_metrics[f'{timeframe}_wick_dominance'] = round(upper_wick_ratio + lower_wick_ratio, 4)
        candle_metrics['directional_strength'] = round(primary_strength, 4)
        
        # Multi-period candle strength (average of last 5 candles)
        if len(df) >= 5:
            recent_strengths = []
            for i in range(-5, 0):
                try:
                    candle = df.iloc[i]
                    c_total_range = candle['High'] - candle['Low']
                    c_body_size = abs(candle['Close'] - candle['Open'])
                    c_is_bullish = candle['Close'] > candle['Open']
                    
                    if c_total_range > 0:
                        c_body_ratio = c_body_size / c_total_range
                        if c_is_bullish:
                            c_oppression = (candle['Open'] - candle['Low']) / c_total_range
                        else:
                            c_oppression = (candle['High'] - candle['Open']) / c_total_range
                        
                        c_strength = c_body_ratio * (1 - c_oppression)
                        recent_strengths.append(c_strength)
                except:
                    continue
            
            if recent_strengths:
                candle_metrics[f'{timeframe}_avg_candle_strength_5'] = round(np.mean(recent_strengths), 4)
                candle_metrics[f'{timeframe}_avg_candle_strength_5_pct'] = round(np.mean(recent_strengths) * 100, 2)
        
    except Exception as e:
        logger.error(f"Candle strength calculation error: {e}")
        candle_metrics[f'{timeframe}_candle_strength'] = 0
        candle_metrics[f'{timeframe}_candle_strength_pct'] = 0
    
    return candle_metrics


def calculate_technical_indicators(df, timeframe, user_config):
    """Calculate configurable technical indicators based on timeframe."""
    indicators = {}
    
    if 'Close' in df.columns and len(df) > 0:
        close = df['Close']
        current_price = close.iloc[-1]
        
        # Get indicator periods for this timeframe
        ema_periods = []
        sma_periods = []
        
        if timeframe == 'daily':
            ema_periods = [int(x.strip()) for x in user_config.daily_ema_periods.split(';') if x.strip()]
            sma_periods = [int(x.strip()) for x in user_config.daily_sma_periods.split(';') if x.strip()]
        elif timeframe == 'weekly':
            ema_periods = [int(x.strip()) for x in user_config.weekly_ema_periods.split(';') if x.strip()]
            sma_periods = [int(x.strip()) for x in user_config.weekly_sma_periods.split(';') if x.strip()]
        elif timeframe == 'monthly':
            ema_periods = [int(x.strip()) for x in user_config.monthly_ema_periods.split(';') if x.strip()]
            sma_periods = [int(x.strip()) for x in user_config.monthly_sma_periods.split(';') if x.strip()]
        
        # Calculate EMAs with timeframe prefix
        for period in ema_periods:
            if len(df) >= period:
                ema_value = close.ewm(span=period).mean().iloc[-1]
                indicators[f'{timeframe}_ema{period}'] = ema_value
                
                # Calculate EMA slope (trend)
                if len(df) >= period + 5:
                    older_ema = close.ewm(span=period).mean().iloc[-6]
                    indicators[f'{timeframe}_ema{period}slope'] = (ema_value - older_ema) / 5
        
        # Calculate SMAs with timeframe prefix
        for period in sma_periods:
            if len(df) >= period:
                sma_value = close.rolling(period).mean().iloc[-1]
                indicators[f'{timeframe}_sma{period}'] = sma_value
                
                # Calculate SMA slope (trend)
                if len(df) >= period + 5:
                    older_sma = close.rolling(period).mean().iloc[-6]
                    indicators[f'{timeframe}_sma{period}slope'] = (sma_value - older_sma) / 5
                    
                # Calculate price distance to SMA (percentage)
                if sma_value != 0:
                    indicators[f'{timeframe}_price2_sma{period}pct'] = ((current_price / sma_value) - 1) * 100
                    
        # Always calculate SMA 10 for INDEX_OVERVIEW (regardless of user config) with timeframe prefix
        if len(df) >= 10:
            sma_10 = close.rolling(10).mean().iloc[-1]
            indicators[f'{timeframe}_sma10'] = sma_10
            if len(df) >= 15:
                older_sma_10 = close.rolling(10).mean().iloc[-6]
                indicators[f'{timeframe}_sma10slope'] = (sma_10 - older_sma_10) / 5
            if sma_10 != 0:
                indicators[f'{timeframe}_price2_sma10pct'] = ((current_price / sma_10) - 1) * 100
            
        # RSI (14-period)
        if len(df) >= 15:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators[f'{timeframe}_rsi_14'] = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) and rs.iloc[-1] != 0 else 50
            
        # Price position relative to highs/lows
        if len(df) >= 52:
            high_52w = close.rolling(252).max().iloc[-1] if len(df) >= 252 else close.max()
            low_52w = close.rolling(252).min().iloc[-1] if len(df) >= 252 else close.min()
            current_price = close.iloc[-1]
            
            if high_52w != low_52w:
                indicators[f'{timeframe}_price_position_52w'] = (current_price - low_52w) / (high_52w - low_52w)
            else:
                indicators[f'{timeframe}_price_position_52w'] = 0.5
                
        # MACD
        if len(df) >= 35:
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            indicators[f'{timeframe}_macd'] = macd_line.iloc[-1]
            indicators[f'{timeframe}_macd_signal'] = signal_line.iloc[-1]
            indicators[f'{timeframe}_macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
            
    return indicators


def calculate_atr_and_atrext(df, atr_period=14, sma_period=50, enable_percentile=True, percentile_period=100):
    """
    Calculate ATR (Average True Range) and ATRext (ATR Extension) based on Steve Jacobs' method.
    
    Args:
        df: DataFrame with OHLC data
        atr_period: Period for ATR calculation (default: 14)
        sma_period: Period for SMA calculation (default: 50)
        enable_percentile: Whether to calculate ATR percentile ranking (default: True)
        percentile_period: Period for ATR percentile calculation (default: 100)
    
    Returns:
        dict: Dictionary containing ATR and ATRext values
    """
    atr_metrics = {}
    
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        return atr_metrics
    
    if len(df) < atr_period:
        return atr_metrics
    
    try:
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Step 1: Calculate True Range (TR)
        data['Prev_Close'] = data['Close'].shift(1)
        data['TR'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Prev_Close']),
                abs(data['Low'] - data['Prev_Close'])
            )
        )
        
        # Handle first row where no previous close exists
        data.loc[data.index[0], 'TR'] = data['High'].iloc[0] - data['Low'].iloc[0]
        
        # Step 2: Calculate ATR using Wilder's method
        atr_values = [np.nan] * atr_period
        
        # Initial ATR is simple average of first 14 TR values
        atr_values[atr_period - 1] = data['TR'][:atr_period].mean()
        
        # Subsequent ATR values use Wilder's exponential smoothing
        for i in range(atr_period, len(data)):
            prev_atr = atr_values[i - 1]
            current_tr = data['TR'].iloc[i]
            atr_values.append((prev_atr * (atr_period - 1) + current_tr) / atr_period)
        
        # Get current ATR value
        current_atr = atr_values[-1]
        atr_metrics['atr'] = current_atr
        
        # Calculate ATR as percentage of close price
        current_close = data['Close'].iloc[-1]
        atr_metrics['atr_pct'] = (current_atr / current_close) * 100
        
        # Step 3: Calculate SMA if we have enough data
        if len(df) >= sma_period:
            sma_value = data['Close'].rolling(window=sma_period).mean().iloc[-1]
            atr_metrics[f'sma{sma_period}'] = sma_value
            
            # Step 4: Calculate ATRext ($ based - primary method)
            atrext_dollar = (current_close - sma_value) / current_atr
            atr_metrics['atrext_dollar'] = atrext_dollar
            
            # Step 5: Calculate ATRext (% based - alternative method)
            atr_pct_decimal = current_atr / current_close
            price_deviation_pct = (current_close / sma_value) - 1
            atrext_percent = price_deviation_pct / atr_pct_decimal
            atr_metrics['atrext_percent'] = atrext_percent
            
        # Add ATR percentile ranking if enabled and enough data available
        if enable_percentile and len(data) >= percentile_period:
            recent_atr_series = pd.Series(atr_values[-percentile_period:])
            current_atr_percentile = (recent_atr_series < current_atr).sum() / len(recent_atr_series)
            atr_metrics[f'atr_percentile_{percentile_period}'] = current_atr_percentile
            
    except Exception as e:
        logger.error(f"Error calculating ATR/ATRext: {e}")
        
    return atr_metrics


def calculate_percentage_moves(df, timeframe='daily', user_config=None):
    """
    Calculate daily and weekly percentage moves for a stock.
    
    Args:
        df: DataFrame with OHLC data sorted by date
        timeframe: Timeframe of the data ('daily', 'weekly', 'monthly')
    
    Returns:
        dict: Dictionary containing percentage move information
    """
    move_metrics = {}
    
    try:
        if 'Close' not in df.columns or len(df) < 2:
            return move_metrics
        
        # Calculate latest period move
        current_price = df['Close'].iloc[-1]
        previous_price = df['Close'].iloc[-2]
        
        # Unified intraday calculation for all timeframes (current period Open to Close)
        if 'Open' in df.columns:
            # Use intraday logic: (Close - Open) / Open * 100 for current period
            current_open = df['Open'].iloc[-1]
            pct_change = ((current_price - current_open) / current_open) * 100
        else:
            # Fallback to close-to-close if Open not available
            pct_change = ((current_price - previous_price) / previous_price) * 100
        
        # Store with timeframe-specific naming
        if timeframe == 'daily':
            move_metrics[f'{timeframe}_{timeframe}_{timeframe}_1d_pct_change'] = pct_change
        elif timeframe == 'weekly':
            move_metrics[f'{timeframe}_{timeframe}_{timeframe}_1w_pct_change'] = pct_change
        elif timeframe == 'monthly':
            move_metrics[f'{timeframe}_{timeframe}_{timeframe}_1m_pct_change'] = pct_change
        else:
            # Fallback for other timeframes
            move_metrics[f'{timeframe}_pct_change'] = pct_change
        
        # Calculate longer-period moves if enough data
        if len(df) >= 5:  # 5-period move
            five_periods_ago = df['Close'].iloc[-5]
            five_period_change = ((current_price - five_periods_ago) / five_periods_ago) * 100
            move_metrics[f'{timeframe}_5period_pct_change'] = five_period_change
        
        if len(df) >= 10:  # 10-period move
            ten_periods_ago = df['Close'].iloc[-10]
            ten_period_change = ((current_price - ten_periods_ago) / ten_periods_ago) * 100
            move_metrics[f'{timeframe}_10period_pct_change'] = ten_period_change
            
        # For daily data, also calculate weekly moves if we have enough data
        # Configurable daily period percent changes
        if timeframe == 'daily':
            # Daily periods (2d, 3d, 5d, etc.) - skip 1d as it's handled specially above
            if hasattr(user_config, 'daily_daily_periods'):
                daily_periods_str = str(user_config.daily_daily_periods)
                if daily_periods_str and daily_periods_str.lower() != 'nan':
                    daily_periods = [int(p.strip()) for p in daily_periods_str.split(';') if p.strip()]
                    for i, period in enumerate(daily_periods, 1):
                        if period == 1:
                            # Skip 1d period - already calculated as intraday above
                            continue
                        if len(df) >= period:
                            daily_price = df['Close'].iloc[-period]
                            daily_change = ((current_price - daily_price) / daily_price) * 100
                            move_metrics[f'daily_daily_daily_{period}d_pct_change'] = daily_change
            
            # Weekly periods
            if hasattr(user_config, 'daily_weekly_periods'):
                weekly_periods_str = str(user_config.daily_weekly_periods)
                if weekly_periods_str and weekly_periods_str.lower() != 'nan':
                    weekly_periods = [int(p.strip()) for p in weekly_periods_str.split(';') if p.strip()]
                    for period in weekly_periods:
                        if len(df) >= period:
                            weekly_price = df['Close'].iloc[-period]
                            weekly_change = ((current_price - weekly_price) / weekly_price) * 100
                            move_metrics[f'daily_daily_weekly_{period}d_pct_change'] = weekly_change
            
            # Monthly periods
            if hasattr(user_config, 'daily_monthly_periods'):
                monthly_periods_str = str(user_config.daily_monthly_periods)
                if monthly_periods_str and monthly_periods_str.lower() != 'nan':
                    monthly_periods = [int(p.strip()) for p in monthly_periods_str.split(';') if p.strip()]
                    for period in monthly_periods:
                        if len(df) >= period:
                            monthly_price = df['Close'].iloc[-period]
                            monthly_change = ((current_price - monthly_price) / monthly_price) * 100
                            move_metrics[f'daily_daily_monthly_{period}d_pct_change'] = monthly_change
            
            # Quarterly periods
            if hasattr(user_config, 'daily_quarterly_periods'):
                quarterly_periods_str = str(user_config.daily_quarterly_periods)
                if quarterly_periods_str and quarterly_periods_str.lower() != 'nan':
                    quarterly_periods = [int(p.strip()) for p in quarterly_periods_str.split(';') if p.strip()]
                    for period in quarterly_periods:
                        if len(df) >= period:
                            quarterly_price = df['Close'].iloc[-period]
                            quarterly_change = ((current_price - quarterly_price) / quarterly_price) * 100
                            move_metrics[f'daily_daily_quarterly_{period}d_pct_change'] = quarterly_change
            
            # Yearly periods
            if hasattr(user_config, 'daily_yearly_periods'):
                yearly_periods_str = str(user_config.daily_yearly_periods)
                if yearly_periods_str and yearly_periods_str.lower() != 'nan':
                    yearly_periods = [int(p.strip()) for p in yearly_periods_str.split(';') if p.strip().isdigit()]
                    for period in yearly_periods:
                        if len(df) >= period:
                            yearly_price = df['Close'].iloc[-period]
                            yearly_change = ((current_price - yearly_price) / yearly_price) * 100
                            move_metrics[f'daily_daily_yearly_{period}d_pct_change'] = yearly_change
        
        # Configurable weekly period percent changes
        elif timeframe == 'weekly':
            # Weekly periods (2w, 4w, etc.)
            if hasattr(user_config, 'weekly_weekly_periods'):
                weekly_periods_str = str(user_config.weekly_weekly_periods)
                if weekly_periods_str and weekly_periods_str.lower() != 'nan':
                    weekly_periods = [int(p.strip()) for p in weekly_periods_str.split(';') if p.strip()]
                    for i, period in enumerate(weekly_periods, 1):
                        if len(df) >= period:
                            weekly_price = df['Close'].iloc[-period]
                            weekly_change = ((current_price - weekly_price) / weekly_price) * 100
                            move_metrics[f'weekly_weekly_weekly_{period}w_pct_change'] = weekly_change
            
            # Monthly periods (1m, 2m based on weeks)
            if hasattr(user_config, 'weekly_monthly_periods'):
                monthly_periods_str = str(user_config.weekly_monthly_periods)
                if monthly_periods_str and monthly_periods_str.lower() != 'nan':
                    monthly_periods = [int(p.strip()) for p in monthly_periods_str.split(';') if p.strip()]
                    for period in monthly_periods:
                        if len(df) >= period:
                            monthly_price = df['Close'].iloc[-period]
                            monthly_change = ((current_price - monthly_price) / monthly_price) * 100
                            move_metrics[f'weekly_weekly_monthly_{period}w_pct_change'] = monthly_change
        
        # Configurable monthly period percent changes
        elif timeframe == 'monthly':
            # Monthly periods (2m, 3m, 6m, etc.)
            if hasattr(user_config, 'monthly_monthly_periods'):
                monthly_periods_str = str(user_config.monthly_monthly_periods)
                if monthly_periods_str and monthly_periods_str.lower() != 'nan':
                    monthly_periods = [int(p.strip()) for p in monthly_periods_str.split(';') if p.strip()]
                    for period in monthly_periods:
                        if len(df) >= period:
                            monthly_price = df['Close'].iloc[-period]
                            monthly_change = ((current_price - monthly_price) / monthly_price) * 100
                            move_metrics[f'monthly_monthly_monthly_{period}m_pct_change'] = monthly_change
            
        # Price momentum indicators
        if len(df) >= 20:
            # Recent vs older period comparison
            recent_avg = df['Close'].tail(5).mean()
            older_avg = df['Close'].iloc[-20:-15].mean()
            momentum = ((recent_avg - older_avg) / older_avg) * 100
            move_metrics[f'{timeframe}_momentum_20'] = momentum
            
        # Add longer period returns for INDEX_OVERVIEW
        if len(df) >= 60:  # Quarter (3 months ≈ 60 trading days)
            quarter_price = df['Close'].iloc[-60]
            quarter_change = ((current_price - quarter_price) / quarter_price) * 100
            move_metrics[f'{timeframe}_quarter_pct_change'] = quarter_change
            
        if len(df) >= 120:  # Half year (6 months ≈ 120 trading days)
            half_year_price = df['Close'].iloc[-120]
            half_year_change = ((current_price - half_year_price) / half_year_price) * 100
            move_metrics[f'{timeframe}_half_year_pct_change'] = half_year_change
            
        if len(df) >= 252:  # Year (252 trading days)
            year_price = df['Close'].iloc[-252]
            year_change = ((current_price - year_price) / year_price) * 100
            move_metrics[f'{timeframe}_year_pct_change'] = year_change
            
    except Exception as e:
        logger.error(f"Error calculating percentage moves: {e}")
        
    return move_metrics


# NOTE: calculate_index_overview_metrics function removed
# as index_overview module has been removed from BASIC calculations


def calculate_ath_atl(df, timeframe='daily'):
    """
    Calculate All-Time High (ATH) and All-Time Low (ATL) across entire historical data.
    Also detects if current price is a new ATH or ATL.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Timeframe for column naming ('daily', 'weekly', 'monthly')
        
    Returns:
        dict: Dictionary with ATH/ATL metrics
    """
    ath_atl_metrics = {}
    
    if 'Close' not in df.columns or len(df) == 0:
        return ath_atl_metrics
    
    try:
        close = df['Close']
        current_close = close.iloc[-1]
        
        # Calculate cumulative maximum (ATH) and minimum (ATL) across all historical data
        cumulative_max = close.cummax()
        cumulative_min = close.cummin()
        
        # Current ATH and ATL values
        current_ath = cumulative_max.iloc[-1]
        current_atl = cumulative_min.iloc[-1]
        
        # Check if current close is a new ATH or ATL
        is_new_ath = current_close == current_ath
        is_new_atl = current_close == current_atl
        
        # Store metrics with timeframe-specific naming
        ath_atl_metrics[f'{timeframe}_ATH'] = current_ath
        ath_atl_metrics[f'{timeframe}_ATL'] = current_atl
        ath_atl_metrics[f'{timeframe}_is_new_ATH'] = is_new_ath
        ath_atl_metrics[f'{timeframe}_is_new_ATL'] = is_new_atl
        
        # Additional metrics: Distance from ATH/ATL in percentage
        if current_ath != 0:
            ath_atl_metrics[f'{timeframe}_distance_from_ATH_pct'] = ((current_close - current_ath) / current_ath) * 100
        
        if current_atl != 0:
            ath_atl_metrics[f'{timeframe}_distance_from_ATL_pct'] = ((current_close - current_atl) / current_atl) * 100
        
        # Calculate ATH/ATL range position (0-100, where 0 = at ATL, 100 = at ATH)
        if current_ath != current_atl:
            ath_atl_metrics[f'{timeframe}_ATH_ATL_position_pct'] = ((current_close - current_atl) / (current_ath - current_atl)) * 100
        else:
            # If ATH == ATL (single data point), position is 50%
            ath_atl_metrics[f'{timeframe}_ATH_ATL_position_pct'] = 50.0
            
    except Exception as e:
        logger.error(f"Error calculating ATH/ATL for {timeframe}: {e}")
        
    return ath_atl_metrics


def calculate_volume_metrics(df, timeframe='daily'):
    """Calculate volume-based metrics."""
    volume_metrics = {}
    
    if 'Volume' in df.columns and len(df) > 1:
        volume = df['Volume']
        
        # Average volume
        volume_metrics[f'{timeframe}_avg_volume_20'] = volume.rolling(20).mean().iloc[-1] if len(df) >= 20 else volume.mean()
        
        # Volume trend
        if len(df) >= 10:
            recent_avg = volume.tail(5).mean()
            older_avg = volume.tail(10).head(5).mean()
            volume_metrics[f'{timeframe}_volume_trend'] = (recent_avg / older_avg - 1) if older_avg != 0 else 0
            
        # On-balance volume
        if 'Close' in df.columns:
            price_change = df['Close'].diff()
            obv = np.where(price_change > 0, volume, 
                          np.where(price_change < 0, -volume, 0)).cumsum()
            volume_metrics['obv'] = obv[-1] if len(obv) > 0 else 0
            
            # OBV trend
            if len(df) >= 20:
                obv_series = pd.Series(obv, index=df.index)
                obv_slope = (obv_series.iloc[-1] - obv_series.iloc[-20]) / 20
                volume_metrics['obv_trend'] = obv_slope
                
    return volume_metrics


def load_index_boolean_data(config):
    """
    Load ticker universe data from ticker_universe_all.csv.
    Includes all columns except description.

    Returns:
        dict: Dictionary mapping ticker -> universe data
    """
    universe_data = {}

    try:
        # Load the ticker universe all file
        universe_file = config.directories['RESULTS_DIR'] / 'ticker_universes' / 'ticker_universe_all.csv'
        if not universe_file.exists():
            logger.warning(f"Ticker universe file not found: {universe_file}")
            return universe_data

        # Load CSV file
        universe_df = pd.read_csv(universe_file)

        # Check for ticker column
        if 'ticker' not in universe_df.columns:
            logger.warning("Ticker universe file missing ticker column")
            return universe_data

        # Get all columns except ticker and description
        exclude_columns = ['ticker', 'description']
        available_columns = [col for col in universe_df.columns if col not in exclude_columns]

        if not available_columns:
            logger.warning("No universe data columns found in file")
            return universe_data

        # Create dictionary mapping ticker to universe data
        for _, row in universe_df.iterrows():
            ticker = row['ticker']
            ticker_data = {}

            for col in available_columns:
                value = row[col]
                # Handle boolean conversion for True/False columns
                if isinstance(value, bool):
                    ticker_data[col] = value
                elif pd.isna(value):
                    # Handle NaN values based on column type
                    if col.startswith(('SP500', 'NASDAQ', 'Russell', 'Dow', 'exchange_', 'rating_')):
                        ticker_data[col] = False  # Boolean columns default to False
                    else:
                        ticker_data[col] = value  # Keep NaN for other columns
                elif isinstance(value, str) and value.lower() in ['true', 'false']:
                    ticker_data[col] = value.lower() == 'true'
                else:
                    ticker_data[col] = value

            universe_data[ticker] = ticker_data

        logger.info(f"Loaded universe data for {len(universe_data)} tickers with {len(available_columns)} data columns")

    except Exception as e:
        logger.error(f"Error loading ticker universe data: {e}")

    return universe_data


def basic_calculations(batch_data, output_path, timeframe, user_config, config=None):
    """
    Perform basic calculations on a batch of stock data.
    Creates a matrix with tickers as rows and calculated indicators as columns.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        output_path: Path to save calculation results
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration object with indicator settings
        config: Config object with directory paths (optional)
    """
    logger.info(f"Starting basic calculations for {len(batch_data)} tickers ({timeframe})")
    
    # Normalize all date indices in batch_data for consistent calculations
    normalized_batch_data = {}
    for ticker, df in batch_data.items():
        if df is not None and not df.empty:
            normalized_df = df.copy()  # Simple copy without complex date normalization
            normalized_batch_data[ticker] = normalized_df
            logger.debug(f"Normalized dates for {ticker}: {len(normalized_df)} rows")
        else:
            normalized_batch_data[ticker] = df
    
    # Use normalized data for all calculations
    batch_data = normalized_batch_data
    
    # Initialize results dictionary to accumulate data across all batches
    if not hasattr(basic_calculations, 'all_results'):
        basic_calculations.all_results = {}
    if timeframe not in basic_calculations.all_results:
        basic_calculations.all_results[timeframe] = {}
    
    # Load boolean classification data (once per batch for efficiency)
    boolean_classifications = {}
    if config:
        boolean_classifications = load_index_boolean_data(config)
    
    processed_count = 0
    error_count = 0
    
    for ticker, df in batch_data.items():
        try:
            if df is None or df.empty:
                logger.debug(f"Skipping {ticker}: No data")
                error_count += 1
                continue
            
            # Initialize ticker result dictionary with normalized date
            calculation_date = df.index.max() if len(df) > 0 else None
            ticker_result = {
                'date': calculation_date,
                'timeframe': timeframe,
                'current_price': df['Close'].iloc[-1] if 'Close' in df.columns and len(df) > 0 else None,
                'data_points': len(df)
            }
            
            # Calculate technical indicators with user configuration
            indicators = calculate_technical_indicators(df, timeframe, user_config)
            
            # Calculate ATR and ATRext metrics (if enabled)
            atr_metrics = {}
            if user_config.enable_atr_calculation:
                atr_metrics = calculate_atr_and_atrext(
                    df, 
                    atr_period=user_config.atr_period,
                    sma_period=user_config.atr_sma_period,
                    enable_percentile=user_config.enable_atr_percentile,
                    percentile_period=user_config.atr_percentile_period
                )
            
            # Stage analysis calculation removed - now handled separately by stage_analysis_processor.py
            stage_metrics = {}  # Keep empty dict for backward compatibility
            
            # Calculate percentage moves
            move_metrics = calculate_percentage_moves(df, timeframe, user_config)
            
            # Calculate volume metrics
            volume_metrics = calculate_volume_metrics(df, timeframe)
            
            # Calculate ATH/ATL metrics
            ath_atl_metrics = calculate_ath_atl(df, timeframe)
            
            # Calculate candle strength metrics
            candle_strength_metrics = calculate_candle_strength(df, timeframe)

            # NOTE: index_overview_metrics calculation removed (obsolete module)
            
            # Calculate technical indicators if enabled
            advanced_indicators = {}
            if user_config and hasattr(user_config, 'indicators_enable') and user_config.indicators_enable:
                try:
                    from .indicators.indicators_calculation import calculate_all_indicators, _get_default_config
                    
                    # Build config from user settings
                    indicators_config = {
                        'kurutoga': {
                            'enabled': getattr(user_config, 'indicators_kurutoga_enable', True),
                            'length': getattr(user_config, 'indicators_kurutoga_length', 14),
                            'source': getattr(user_config, 'indicators_kurutoga_source', 'Close')
                        },
                        'tsi': {
                            'enabled': getattr(user_config, 'indicators_tsi_enable', True),
                            'fast': getattr(user_config, 'indicators_tsi_fast', 13),
                            'slow': getattr(user_config, 'indicators_tsi_slow', 25),
                            'signal': getattr(user_config, 'indicators_tsi_signal', 13)
                        },
                        'macd': {
                            'enabled': getattr(user_config, 'indicators_macd_enable', True),
                            'fast': getattr(user_config, 'indicators_macd_fast', 12),
                            'slow': getattr(user_config, 'indicators_macd_slow', 26),
                            'signal': getattr(user_config, 'indicators_macd_signal', 9)
                        },
                        'mfi': {
                            'enabled': getattr(user_config, 'indicators_mfi_enable', True),
                            'length': getattr(user_config, 'indicators_mfi_length', 14),
                            'include_signal': getattr(user_config, 'indicators_mfi_signal_enable', True),
                            'signal_period': getattr(user_config, 'indicators_mfi_signal_period', 9)
                        },
                        'rsi': {
                            'enabled': getattr(user_config, 'indicators_rsi_enable', True),
                            'length': getattr(user_config, 'indicators_rsi_length', 14)
                        }
                    }
                    
                    # Calculate key indicators for basic calculations
                    indicators_data = calculate_all_indicators(df, indicators_config)
                    
                    # Extract latest values for basic calculations matrix
                    latest = indicators_data.iloc[-1]
                    if 'TSI' in indicators_data.columns:
                        advanced_indicators['tsi'] = latest['TSI']
                    if 'TSI_Signal' in indicators_data.columns:
                        advanced_indicators['tsi_signal'] = latest['TSI_Signal']
                    if 'MACD' in indicators_data.columns:
                        advanced_indicators['macd'] = latest['MACD']
                        advanced_indicators['macd_signal'] = latest['MACD_Signal']
                        advanced_indicators['macd_hist'] = latest['MACD_Hist']
                    if 'MFI' in indicators_data.columns:
                        advanced_indicators['mfi'] = latest['MFI']
                    if 'RSI' in indicators_data.columns:
                        advanced_indicators['rsi'] = latest['RSI']
                    if 'Kurutoga_Current' in indicators_data.columns:
                        advanced_indicators['kurutoga_current'] = latest['Kurutoga_Current']
                        advanced_indicators['kurutoga_2x'] = latest['Kurutoga_2x']
                        advanced_indicators['kurutoga_4x'] = latest['Kurutoga_4x']
                        
                except Exception as e:
                    logger.debug(f"Advanced indicators calculation failed for {ticker}: {e}")
            
            # Store technical indicators
            for key, value in indicators.items():
                ticker_result[key] = value
            
            # Store ATR and ATRext metrics
            for key, value in atr_metrics.items():
                ticker_result[key] = value
            
            # Stage analysis metrics storage removed - handled separately by stage_analysis_processor.py
            
            # Store percentage moves
            for key, value in move_metrics.items():
                ticker_result[key] = value
            
            # Store volume metrics
            for key, value in volume_metrics.items():
                ticker_result[key] = value
            
            # Store ATH/ATL metrics
            for key, value in ath_atl_metrics.items():
                ticker_result[key] = value
            
            # Store candle strength metrics
            for key, value in candle_strength_metrics.items():
                ticker_result[key] = value

            # NOTE: index_overview_metrics storage removed (obsolete module)
            
            # Store advanced indicators
            for key, value in advanced_indicators.items():
                ticker_result[key] = value
            
            # Add boolean classifications for this ticker
            if ticker in boolean_classifications:
                for key, value in boolean_classifications[ticker].items():
                    ticker_result[key] = value
            
            # Store result for this ticker
            basic_calculations.all_results[timeframe][ticker] = ticker_result
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            error_count += 1
            # Store empty result for failed ticker
            basic_calculations.all_results[timeframe][ticker] = {'error': str(e)}
            continue
    
    # Display configured indicators for this timeframe
    ema_periods = []
    sma_periods = []
    if timeframe == 'daily':
        ema_periods = [x.strip() for x in user_config.daily_ema_periods.split(';') if x.strip()]
        sma_periods = [x.strip() for x in user_config.daily_sma_periods.split(';') if x.strip()]
    elif timeframe == 'weekly':
        ema_periods = [x.strip() for x in user_config.weekly_ema_periods.split(';') if x.strip()]
        sma_periods = [x.strip() for x in user_config.weekly_sma_periods.split(';') if x.strip()]
    elif timeframe == 'monthly':
        ema_periods = [x.strip() for x in user_config.monthly_ema_periods.split(';') if x.strip()]
        sma_periods = [x.strip() for x in user_config.monthly_sma_periods.split(';') if x.strip()]
    
    # Print batch summary
    print(f"📊 Basic Calculations Batch Summary ({timeframe}):")
    print(f"  • Batch processed: {processed_count} tickers")
    print(f"  • EMA periods: {', '.join(ema_periods) if ema_periods else 'None'}")
    print(f"  • SMA periods: {', '.join(sma_periods) if sma_periods else 'None'}")
    
    logger.info(f"Batch completed: {processed_count} processed, {error_count} errors")
    
    return processed_count


def combine_daily_weekly_calculations(config, user_config):
    """
    Combine daily and weekly basic calculations into a single file with _d/_w suffixes.
    
    Args:
        config: Config object with directory paths
        user_config: User configuration object
        
    Returns:
        str: Path to combined file or None if files don't exist
    """
    # Define file paths using helper function to find latest files
    daily_file = find_latest_basic_calculation_file(config, 'daily', user_config.ticker_choice)
    weekly_file = find_latest_basic_calculation_file(config, 'weekly', user_config.ticker_choice)
    
    # Use date-stamped naming for combined file
    date_stamp = datetime.now().strftime("%Y%m%d")
    combined_file = config.directories['BASIC_CALCULATION_DIR'] / f'basic_calculation_DWM_{date_stamp}.csv'
    
    try:
        # Check if both files exist
        if not daily_file or not weekly_file or not daily_file.exists() or not weekly_file.exists():
            logger.warning(f"Cannot combine: daily file exists: {daily_file.exists()}, weekly file exists: {weekly_file.exists()}")
            return None
            
        # Read both files with date normalization
        daily_df = pd.read_csv(daily_file, index_col=0, parse_dates=True) if daily_file.exists() else pd.DataFrame()
        weekly_df = pd.read_csv(weekly_file, index_col=0, parse_dates=True) if weekly_file.exists() else pd.DataFrame()
        
        # Fall back to regular loading if normalization fails
        if daily_df.empty and daily_file.exists():
            daily_df = pd.read_csv(daily_file)
        if weekly_df.empty and weekly_file.exists():
            weekly_df = pd.read_csv(weekly_file)
        
        logger.info(f"Combining calculations: {len(daily_df)} daily records, {len(weekly_df)} weekly records")
        
        # Define columns that should remain unchanged (common columns)
        common_columns = {'ticker', 'current_price'}  # ticker is the merge key, current_price should be the same
        
        # Rename columns in daily dataframe (add _d suffix to non-common columns)
        daily_renamed = daily_df.copy()
        for col in daily_df.columns:
            if col not in common_columns:
                daily_renamed = daily_renamed.rename(columns={col: f'{col}_d'})
        
        # Rename columns in weekly dataframe (add _w suffix to non-common columns)
        weekly_renamed = weekly_df.copy()
        for col in weekly_df.columns:
            if col not in common_columns:
                weekly_renamed = weekly_renamed.rename(columns={col: f'{col}_w'})
        
        # Merge on ticker
        combined_df = pd.merge(daily_renamed, weekly_renamed, on='ticker', how='outer', suffixes=('', '_w_dup'))
        
        # Handle current_price - use daily if available, otherwise weekly
        if 'current_price_w' in combined_df.columns:
            combined_df['current_price'] = combined_df['current_price'].fillna(combined_df['current_price_w'])
            combined_df = combined_df.drop(columns=['current_price_w'])
        
        # Clean up any duplicate columns from merge
        duplicate_cols = [col for col in combined_df.columns if col.endswith('_w_dup')]
        if duplicate_cols:
            combined_df = combined_df.drop(columns=duplicate_cols)
            logger.info(f"Removed duplicate columns: {duplicate_cols}")
        
        # Sort by ticker
        combined_df = combined_df.sort_values('ticker')
        
        # Save combined file
        # Ensure directory exists before saving
        combined_file.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"Combined calculations saved: {combined_file}")
        logger.info(f"Combined file contains {len(combined_df)} tickers with {len(combined_df.columns)} total columns")

        # Memory cleanup after CSV write
        del combined_df
        gc.collect()
        logger.debug("Memory cleaned up after combined calculations CSV write")

        return str(combined_file)
        
    except Exception as e:
        logger.error(f"Error combining daily and weekly calculations: {e}")
        return None


def save_basic_calculations_matrix(config, user_config, timeframe):
    """
    Save the accumulated basic calculations as a matrix file.
    Called after all batches are processed for a timeframe.
    
    Args:
        config: Config object with directory paths
        user_config: User configuration object
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
    """
    if not hasattr(basic_calculations, 'all_results'):
        print(f"⚠️  No calculation results to save for {timeframe}")
        return None
        
    if timeframe not in basic_calculations.all_results:
        print(f"⚠️  No {timeframe} calculation results to save")
        return None
    
    results_data = basic_calculations.all_results[timeframe]
    
    if not results_data:
        print(f"⚠️  No valid calculation data for {timeframe}")
        return None
    
    # Try to load the info file to get all expected tickers
    # Use only tickers with actual calculation results (like RS and stage_analysis)
    all_tickers = list(results_data.keys())
    print(f"📋 Using {len(all_tickers)} tickers with valid calculations (consistent with RS/stage_analysis)")
    
    # Create matrix DataFrame
    matrix_data = []
    
    for ticker in all_tickers:
        row = {'ticker': ticker}
        ticker_data = results_data[ticker]
        # Add all calculated indicators
        for key, value in ticker_data.items():
            if key not in ['error']:  # Skip error entries
                row[key] = value
        
        matrix_data.append(row)
    
    # Create DataFrame
    matrix_df = pd.DataFrame(matrix_data)
    
    # COMPREHENSIVE 11-GROUP COLUMN ORDERING SYSTEM
    # ==============================================
    # Based on logical grouping rules for financial analysis
    
    all_cols = list(matrix_df.columns)
    ordered_cols = []
    remaining_cols = set(all_cols)
    
    # Helper function to add columns if they exist
    def add_columns_if_exist(column_list):
        added = []
        for col in column_list:
            if col in remaining_cols:
                ordered_cols.append(col)
                remaining_cols.remove(col)
                added.append(col)
        return added
    
    # Helper function to add pattern-matched columns
    def add_pattern_columns(patterns, sort_func=None):
        matched = []
        for pattern in patterns:
            if callable(pattern):
                # Pattern is a function that filters columns
                found = [col for col in remaining_cols if pattern(col)]
            else:
                # Pattern is a string to match
                found = [col for col in remaining_cols if pattern in col]
            
            if sort_func:
                found = sorted(found, key=sort_func)
            else:
                found = sorted(found)
                
            for col in found:
                ordered_cols.append(col)
                matched.append(col)
                remaining_cols.discard(col)
        return matched
    
    # 1. IDENTIFIERS & DATE GROUP
    add_columns_if_exist(['ticker'])
    # Format date as latest_date with ISO format (YYYY-MM-DD)
    if 'date' in remaining_cols:
        ordered_cols.append('date')
        remaining_cols.remove('date')
    
    # 2. PRICE & CHANGE METRICS GROUP  
    add_columns_if_exist(['current_price', 'data_points'])
    
    # Add daily percent change (current timeframe change) with systematic naming
    if timeframe == 'daily':
        add_columns_if_exist([f'{timeframe}_{timeframe}_{timeframe}_1d_pct_change', f'{timeframe}_{timeframe}_{timeframe}_1d_abs_pct_change'])
    elif timeframe == 'weekly':
        add_columns_if_exist([f'{timeframe}_{timeframe}_{timeframe}_1w_pct_change', f'{timeframe}_{timeframe}_{timeframe}_1w_abs_pct_change'])
    elif timeframe == 'monthly':
        add_columns_if_exist([f'{timeframe}_{timeframe}_{timeframe}_1m_pct_change', f'{timeframe}_{timeframe}_{timeframe}_1m_abs_pct_change'])
    else:
        # Fallback for other timeframes
        add_columns_if_exist([f'{timeframe}_pct_change', f'{timeframe}_abs_pct_change'])
    
    # Add configurable period changes in logical order
    if timeframe == 'daily':
        # Daily periods with systematic naming (daily_daily_daily_Nd_pct_change)
        add_pattern_columns(['daily_daily_daily_'], lambda x: int(x.split('_')[3][:-1]) if len(x.split('_')) >= 4 and x.split('_')[3][:-1].isdigit() else 0)
        # Weekly periods (daily_daily_weekly_Nd_pct_change)
        add_pattern_columns(['daily_daily_weekly_'], lambda x: int(x.split('_')[3][:-1]) if len(x.split('_')) >= 4 and x.split('_')[3][:-1].isdigit() else 0)
        # Monthly periods (daily_daily_monthly_Nd_pct_change)
        add_pattern_columns(['daily_daily_monthly_'], lambda x: int(x.split('_')[3][:-1]) if len(x.split('_')) >= 4 and x.split('_')[3][:-1].isdigit() else 0)
        # Quarterly periods (daily_daily_quarterly_Nd_pct_change)
        add_pattern_columns(['daily_daily_quarterly_'], lambda x: int(x.split('_')[3][:-1]) if len(x.split('_')) >= 4 and x.split('_')[3][:-1].isdigit() else 0)
        # Yearly periods (daily_daily_yearly_Nd_pct_change)
        add_pattern_columns(['daily_daily_yearly_'], lambda x: int(x.split('_')[3][:-1]) if len(x.split('_')) >= 4 and x.split('_')[3][:-1].isdigit() else 0)
    elif timeframe == 'weekly':
        # Weekly periods with systematic naming (weekly_weekly_weekly_Nw_pct_change)
        add_pattern_columns(['weekly_weekly_weekly_'], lambda x: int(x.split('_')[3][:-1]) if len(x.split('_')) >= 4 and x.split('_')[3][:-1].isdigit() else 0)
        # Monthly periods in weeks (weekly_weekly_monthly_Nw_pct_change)
        add_pattern_columns(['weekly_weekly_monthly_'], lambda x: int(x.split('_')[3][:-1]) if len(x.split('_')) >= 4 and x.split('_')[3][:-1].isdigit() else 0)
    elif timeframe == 'monthly':
        # Monthly periods with systematic naming (monthly_monthly_monthly_Nm_pct_change)
        add_pattern_columns(['monthly_monthly_monthly_'], lambda x: int(x.split('_')[3][:-1]) if len(x.split('_')) >= 4 and x.split('_')[3][:-1].isdigit() else 0)
    
    # Add older period changes for compatibility
    add_columns_if_exist([f'{timeframe}_5period_pct_change', f'{timeframe}_10period_pct_change'])
    add_columns_if_exist([f'{timeframe}_quarter_pct_change'])  # INDEX_OVERVIEW compatibility
    
    # 3. MOVING AVERAGES AND SLOPES GROUP
    # EMAs with their slopes using timeframe-prefixed naming - sorted by period
    ema_pattern = f'{timeframe}_ema'
    ema_base = sorted([col for col in remaining_cols if col.startswith(ema_pattern) and not col.endswith('slope') and col[len(ema_pattern):].isdigit()],
                      key=lambda x: int(x[len(ema_pattern):]) if x[len(ema_pattern):].isdigit() else 0)
    for base in ema_base:
        add_columns_if_exist([base])
        slope_col = f"{base}slope"
        add_columns_if_exist([slope_col])
    
    # SMAs with their slopes using timeframe-prefixed naming - sorted by period  
    sma_pattern = f'{timeframe}_sma'
    sma_base = sorted([col for col in remaining_cols if col.startswith(sma_pattern) and not col.endswith('slope') and col[len(sma_pattern):].isdigit()],
                      key=lambda x: int(x[len(sma_pattern):]) if x[len(sma_pattern):].isdigit() else 0)
    for base in sma_base:
        add_columns_if_exist([base])
        slope_col = f"{base}slope"
        add_columns_if_exist([slope_col])
    
    # 4. TECHNICAL POSITION METRICS GROUP
    # Price to MA percentages and boolean positions (timeframe-prefixed)
    add_pattern_columns([lambda x: f'{timeframe}_price2_' in x and 'pct' in x])  # daily_price2_sma10pct, etc.
    add_pattern_columns([lambda x: 'priceabove' in x])  # daily_priceabovesma10, etc.
    add_pattern_columns([lambda x: 'closeabove' in x])  # closeabovesma20, etc.
    add_pattern_columns([lambda x: f'{timeframe}_price_vs_' in x])  # Stage analysis position metrics
    
    # 5. MOVING AVERAGE RELATIONSHIPS GROUP
    add_pattern_columns([lambda x: 'vs' in x and 'sma' in x])  # daily_sma10vssma20, etc.
    add_pattern_columns([lambda x: 'perfectbullish' in x or 'perfectbearish' in x])  # daily_perfectbullishalignment, etc.
    add_pattern_columns([lambda x: '5day_low_vs_30day_high' in x])  # daily_5day_low_vs_30day_high
    add_pattern_columns([lambda x: 'maalignment' in x])  # Stage analysis MA alignment
    
    # 6. TREND & STRENGTH INDICATORS GROUP  
    add_columns_if_exist([f'{timeframe}_macd', f'{timeframe}_macd_signal', f'{timeframe}_macd_histogram'])
    add_columns_if_exist([f'{timeframe}_rsi_14'])
    add_pattern_columns([lambda x: 'momentum_' in x])  # momentum indicators
    # Stage analysis columns removed from basic calculations output
    add_columns_if_exist([f'{timeframe}_atr_ratio'])
    
    # 7. VOLATILITY MEASURES GROUP
    add_columns_if_exist(['atr', 'atr_pct', 'atr_percentile_100'])
    add_columns_if_exist(['atrext_dollar', 'atrext_percent'])
    add_pattern_columns([lambda x: 'volatility' in x])
    
    # 8. HIGH/LOW POSITION METRICS GROUP
    add_columns_if_exist([f'{timeframe}_price_position_52w'])
    # ATH/ATL metrics
    add_columns_if_exist([f'{timeframe}_ATH', f'{timeframe}_ATL'])
    add_columns_if_exist([f'{timeframe}_is_new_ATH', f'{timeframe}_is_new_ATL'])
    add_columns_if_exist([f'{timeframe}_distance_from_ATH_pct', f'{timeframe}_distance_from_ATL_pct'])
    add_columns_if_exist([f'{timeframe}_ATH_ATL_position_pct'])
    # Other high/low position metrics
    add_pattern_columns([lambda x: f'{timeframe}_at_' in x and ('high' in x or 'low' in x)])  # daily_at_20day_high, etc.
    add_pattern_columns([lambda x: 'near_' in x and ('high' in x or 'low' in x)])  # near_52w_high, etc.
    add_pattern_columns([lambda x: 'position_' in x and ('range' in x or 'day' in x)])  # position metrics
    
    # 9. VOLUME METRICS GROUP
    add_pattern_columns([lambda x: 'volume' in x and 'avg' in x])  # daily_avg_volume_20, etc.
    add_pattern_columns([lambda x: 'volume' in x and 'trend' in x])  # daily_volume_trend, etc.
    add_pattern_columns([lambda x: 'volume' in x and x not in ordered_cols])  # Other volume metrics
    
    # 10. CANDLE/BODY METRICS GROUP
    add_pattern_columns([lambda x: 'body_ratio' in x])  # daily_body_ratio, etc.
    add_pattern_columns([lambda x: 'wick_ratio' in x])  # daily_upper_wick_ratio, daily_lower_wick_ratio, etc.
    add_pattern_columns([lambda x: 'candle_strength' in x])  # daily_candle_strength, daily_candle_strength_pct, etc.
    add_pattern_columns([lambda x: 'candle_type' in x or 'is_bullish' in x or 'wick_dominance' in x])  # daily_candle_type, etc.
    add_pattern_columns([lambda x: 'oppression_factor' in x])  # daily_oppression_factor, etc.
    add_pattern_columns([lambda x: 'candle' in x and x not in ordered_cols])  # Other candle metrics
    
    # 11. INDEX MEMBERSHIP BOOLEANS GROUP
    add_pattern_columns([lambda x: x.startswith(('SP500', 'NASDAQ100', 'Russell', 'Dow'))])  # Index membership
    add_pattern_columns([lambda x: x.startswith('exchange_')])  # Exchange flags
    add_pattern_columns([lambda x: x.startswith('rating_')])  # Analyst rating flags

    # 12. UNIVERSE METADATA GROUP
    add_pattern_columns([lambda x: x in ['market_cap', 'market_cap_currency', 'sector', 'industry', 'exchange', 'analyst rating']])  # Core metadata
    add_pattern_columns([lambda x: x in ['upcoming earnings date', 'recent earnings date', 'index']])  # Additional metadata
    add_pattern_columns([lambda x: x.startswith(('filter_applied', 'universe_name', 'generation_source', 'generation_date'))])  # File metadata
    
    # Add any remaining columns at the end (preserves any new indicators we missed)
    if remaining_cols:
        ordered_cols.extend(sorted(remaining_cols))
    
    matrix_df = matrix_df[ordered_cols]
    
    # Format numerical columns to 2 decimal places
    numeric_cols = ['current_price'] + [col for col in matrix_df.columns if col.startswith(('ema', 'sma'))]
    for col in numeric_cols:
        if col in matrix_df.columns:
            matrix_df[col] = pd.to_numeric(matrix_df[col], errors='coerce').round(2)
    
    # Save matrix file with the requested naming format - include timeframe and data date timestamp
    # Use centralized naming function to preserve hyphens (0-5 format)
    from src.config import get_file_safe_user_choice
    safe_user_choice = get_file_safe_user_choice(user_config.ticker_choice, preserve_hyphens=True)
    
    # Extract data date from the first ticker's results (use data date instead of file generation date)
    data_date = None
    for ticker_data in results_data.values():
        if 'date' in ticker_data and ticker_data['date']:
            # Handle both string dates and Timestamp objects
            if isinstance(ticker_data['date'], str):
                data_date = ticker_data['date'].replace('-', '')  # Convert 2025-08-29 to 20250829
            else:
                data_date = ticker_data['date'].strftime('%Y%m%d')  # Handle pandas Timestamp
            break
    
    # Fallback to file generation date if no data date found
    if not data_date:
        data_date = datetime.now().strftime("%Y%m%d")
        print(f"⚠️  Using file generation date as fallback: {data_date}")
    else:
        print(f"📅 Using data date for filename: {data_date}")
    
    output_file = config.directories['BASIC_CALCULATION_DIR'] / f'basic_calculation_{safe_user_choice}_{timeframe}_{data_date}.csv'
    
    # Ensure directory exists before saving
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    matrix_df.to_csv(output_file, index=False, float_format='%.2f')

    # Memory cleanup after CSV write - free accumulated results data
    del matrix_df
    del basic_calculations.all_results[timeframe]
    gc.collect()
    logger.debug(f"Memory cleaned up after {timeframe} matrix CSV write and results cleared")

    # Print results summary
    successful_calculations = len([t for t in results_data if 'error' not in results_data[t]])
    
    print(f"✅ Basic Calculations Matrix Saved ({timeframe.upper()} DATA):")
    print(f"  • Output file: {output_file}")
    print(f"  • Data source: {timeframe} market data files")
    print(f"  • Total tickers: {len(matrix_df)}")
    print(f"  • Successfully calculated: {successful_calculations}")
    print(f"  • Total columns: {len(matrix_df.columns)}")
    
    # Show indicators calculated
    ema_cols = [col for col in matrix_df.columns if col.startswith('ema') and not col.endswith('slope') and col[3:].isdigit()]
    sma_cols = [col for col in matrix_df.columns if col.startswith('sma') and not col.endswith('slope') and col[3:].isdigit()]
    slope_cols = [col for col in matrix_df.columns if col.endswith('slope')]
    
    if ema_cols:
        print(f"  • EMAs calculated: {', '.join(sorted(ema_cols))}")
    if sma_cols:
        print(f"  • SMAs calculated: {', '.join(sorted(sma_cols, key=lambda x: int(x[3:]) if x[3:].isdigit() else 0))}")
    if slope_cols:
        print(f"  • Slopes included: {len(slope_cols)} trend indicators")
        
    # Show column order sample
    print(f"  • Column order: 11 logical groups - Identifiers, Price/Change, MA/Slopes, Position, Relationships, Trend, Volatility, High/Low, Volume, Candle, Index Membership")
    print(f"  • Number format: All values rounded to 2 decimal places")
    print(f"  • Date: Last trading date from market data (ISO format YYYY-MM-DD)")
    
    logger.info(f"Matrix saved: {output_file} with {len(matrix_df)} rows and {len(matrix_df.columns)} columns")
    
    # Database sync removed - using CSV files directly
    
    # Return both output file path and data date for central date management
    return {
        'output_file': str(output_file),
        'data_date': data_date,
        'formatted_date': data_date[:4] + '-' + data_date[4:6] + '-' + data_date[6:8] if len(data_date) >= 8 else data_date,
        'timeframe': timeframe
    }