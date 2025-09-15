"""
Stage Analysis Module
====================

Implements market stage classification using moving averages and ATR based on 
tested TradingView Pine Script implementation. Classifies each ticker into one 
of 11 market stages across 4 main phases (Basing, Advancing, Distribution, Declining).

Based on: /intro_stage_analysis/stage_analysis/stage_analysis.pine
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
# Date normalization removed - using simple date operations

logger = logging.getLogger(__name__)


class StageAnalyzer:
    """
    Market stage analyzer using moving averages and ATR volatility measures.
    
    Implements 11-stage classification system:
    - Stage 1 (Basing): 1A, 1B, 1C
    - Stage 2 (Advancing): 2A, 2B, 2C  
    - Stage 3 (Distribution): 3A, 3B, 3C
    - Stage 4 (Declining): 4A, 4B, 4C
    - Special: LP (Launch Pad), ?? (Undefined)
    """
    
    def __init__(self, user_config):
        """
        Initialize stage analyzer with user configuration.
        
        Args:
            user_config: UserConfiguration object with stage analysis settings
        """
        # Moving Average Periods (user-configurable)
        self.ema_fast_period = getattr(user_config, 'stage_ema_fast_period', 10)
        self.sma_medium_period = getattr(user_config, 'stage_sma_medium_period', 20)
        self.sma_slow_period = getattr(user_config, 'stage_sma_slow_period', 50)
        
        # ATR Configuration
        self.atr_period = getattr(user_config, 'stage_atr_period', 14)
        self.atr_threshold_low = getattr(user_config, 'stage_atr_threshold_low', 4.0)
        self.atr_threshold_high = getattr(user_config, 'stage_atr_threshold_high', 7.0)
        
        # MA Convergence Threshold
        self.ma_convergence_threshold = getattr(user_config, 'stage_ma_convergence_threshold', 1.0)
        
        # Stage definitions with colors (for future visualization)
        self.stage_definitions = {
            'LP': {'name': 'Launch Pad', 'color': 'purple'},
            '4C': {'name': 'Bearish Extended', 'color': 'maroon'}, 
            '4B': {'name': 'Bearish Confirmation', 'color': 'red'},
            '4A': {'name': 'Bearish Trend', 'color': 'red_light'},
            '3C': {'name': 'Volatile Distribution', 'color': 'orange'},
            '3B': {'name': 'Fade Confirmation', 'color': 'orange_dark'},
            '3A': {'name': 'Bullish Fade', 'color': 'orange_light'},
            '2C': {'name': 'Bullish Extended', 'color': 'green_dark'},
            '2B': {'name': 'Breakout Confirmation', 'color': 'lime'}, 
            '2A': {'name': 'Bullish Trend', 'color': 'green_light'},
            '1C': {'name': 'Pullback', 'color': 'yellow_light'},
            '1B': {'name': 'Mean Reversion', 'color': 'yellow'},
            '1A': {'name': 'Upward Pivot', 'color': 'gray_light'},
            '??': {'name': 'Undefined', 'color': 'gray'}
        }
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate required moving averages for stage analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added MA columns
        """
        if df is None or df.empty or 'Close' not in df.columns:
            return df
            
        # Normalize dates before calculations
        df = df.copy()  # Simple copy without date normalization
        
        try:
            # Calculate moving averages
            df[f'EMA_{self.ema_fast_period}'] = df['Close'].ewm(span=self.ema_fast_period).mean()
            df[f'SMA_{self.sma_medium_period}'] = df['Close'].rolling(self.sma_medium_period).mean()
            df[f'SMA_{self.sma_slow_period}'] = df['Close'].rolling(self.sma_slow_period).mean()
            
            # Calculate ATR
            df['High_Low'] = df['High'] - df['Low']
            df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
            df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
            df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
            df['ATR'] = df['True_Range'].rolling(self.atr_period).mean()
            
            # Calculate ATR ratio (ATR / SMA_slow * 100)
            df['ATR_Ratio'] = (df['ATR'] / df[f'SMA_{self.sma_slow_period}']) * 100
            
            # Clean up intermediate columns
            df.drop(['High_Low', 'High_Close', 'Low_Close', 'True_Range'], axis=1, inplace=True)
            
        except Exception as e:
            logger.debug(f"Error calculating moving averages: {e}")
            
        return df
    
    def detect_stage(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Detect market stage for a single row of data.
        
        Args:
            row: Series with Close price, MAs, and ATR_Ratio
            
        Returns:
            Tuple of (stage_code, stage_name, stage_color)
        """
        try:
            close = row['Close']
            ema_fast = row[f'EMA_{self.ema_fast_period}']
            sma_medium = row[f'SMA_{self.sma_medium_period}']
            sma_slow = row[f'SMA_{self.sma_slow_period}']
            atr_ratio = row['ATR_Ratio']
            
            # Handle NaN values
            if pd.isna(close) or pd.isna(ema_fast) or pd.isna(sma_medium) or pd.isna(sma_slow) or pd.isna(atr_ratio):
                return '??', 'Undefined', 'gray'
            
            # Stage condition checks (following TradingView logic exactly)
            stage_1a = close >= ema_fast and close <= sma_medium and close <= sma_slow
            stage_1b = close >= ema_fast and close >= sma_medium and close <= sma_slow
            stage_1c = close < ema_fast and close >= sma_medium and close >= sma_slow
            
            stage_2a = (close >= ema_fast and close >= sma_medium and close >= sma_slow and 
                        atr_ratio < self.atr_threshold_low)
            stage_2b = (close >= ema_fast and close >= sma_medium and close >= sma_slow and 
                        ema_fast > sma_medium and sma_medium > sma_slow and 
                        atr_ratio >= self.atr_threshold_low and atr_ratio < self.atr_threshold_high)
            stage_2c = (close >= ema_fast and close >= sma_medium and close >= sma_slow and 
                        atr_ratio >= self.atr_threshold_high)
            
            stage_3a = (close <= ema_fast and close <= sma_medium and close >= sma_slow and 
                        atr_ratio < self.atr_threshold_low)
            stage_3b = close <= ema_fast and close <= sma_medium and close <= sma_slow
            stage_3c = (close <= ema_fast and close <= sma_medium and close >= sma_slow and 
                        atr_ratio >= self.atr_threshold_low)
            
            stage_4a = (close <= ema_fast and close <= sma_medium and close <= sma_slow and 
                        atr_ratio < self.atr_threshold_low)
            stage_4b = (close <= ema_fast and close <= sma_medium and close <= sma_slow and 
                        ema_fast < sma_medium and sma_medium < sma_slow and 
                        atr_ratio >= self.atr_threshold_low and atr_ratio < self.atr_threshold_high)
            stage_4c = (close <= ema_fast and close <= sma_medium and close <= sma_slow and 
                        atr_ratio >= self.atr_threshold_high)
            
            # Launch Pad: MA Convergence Zone (all MAs within threshold % of each other)
            ma_convergence = (abs(ema_fast - sma_medium) / sma_medium <= self.ma_convergence_threshold / 100 and 
                             abs(sma_medium - sma_slow) / sma_slow <= self.ma_convergence_threshold / 100 and 
                             abs(ema_fast - sma_slow) / sma_slow <= self.ma_convergence_threshold / 100)
            
            # Priority system (ordered by stage significance - matches TradingView exactly)
            if ma_convergence:
                stage_code = 'LP'
            elif stage_4c:
                stage_code = '4C'
            elif stage_4b:
                stage_code = '4B'
            elif stage_4a:
                stage_code = '4A'
            elif stage_3c:
                stage_code = '3C'
            elif stage_3b:
                stage_code = '3B'
            elif stage_3a:
                stage_code = '3A'
            elif stage_2c:
                stage_code = '2C'
            elif stage_2b:
                stage_code = '2B'
            elif stage_2a:
                stage_code = '2A'
            elif stage_1c:
                stage_code = '1C'
            elif stage_1b:
                stage_code = '1B'
            elif stage_1a:
                stage_code = '1A'
            else:
                stage_code = '??'
            
            # Get stage name and color
            stage_info = self.stage_definitions[stage_code]
            return stage_code, stage_info['name'], stage_info['color']
            
        except Exception as e:
            logger.debug(f"Error in stage detection: {e}")
            return '??', 'Undefined', 'gray'
    
    def calculate_additional_metrics(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate additional metrics for the stage analysis.
        
        Args:
            row: Series with Close price, MAs, and ATR_Ratio
            
        Returns:
            Dictionary with additional metrics
        """
        try:
            close = row['Close']
            ema_fast = row[f'EMA_{self.ema_fast_period}']
            sma_medium = row[f'SMA_{self.sma_medium_period}']
            sma_slow = row[f'SMA_{self.sma_slow_period}']
            atr_ratio = row['ATR_Ratio']
            
            # Handle NaN values
            if pd.isna(close) or pd.isna(ema_fast) or pd.isna(sma_medium) or pd.isna(sma_slow):
                return {
                    'atr_ratio': np.nan,
                    'ma_alignment': 'Unknown',
                    'price_vs_ema_fast': np.nan,
                    'price_vs_sma_medium': np.nan,
                    'price_vs_sma_slow': np.nan
                }
            
            # Calculate price vs MA percentages
            price_vs_ema_fast = (close - ema_fast) / ema_fast * 100
            price_vs_sma_medium = (close - sma_medium) / sma_medium * 100
            price_vs_sma_slow = (close - sma_slow) / sma_slow * 100
            
            # Determine MA alignment
            if ema_fast > sma_medium and sma_medium > sma_slow:
                ma_alignment = 'Bullish'
            elif ema_fast < sma_medium and sma_medium < sma_slow:
                ma_alignment = 'Bearish'
            else:
                ma_alignment = 'Mixed'
            
            return {
                'atr_ratio': atr_ratio if not pd.isna(atr_ratio) else np.nan,
                'ma_alignment': ma_alignment,
                'price_vs_ema_fast': price_vs_ema_fast,
                'price_vs_sma_medium': price_vs_sma_medium,
                'price_vs_sma_slow': price_vs_sma_slow
            }
            
        except Exception as e:
            logger.debug(f"Error calculating additional metrics: {e}")
            return {
                'atr_ratio': np.nan,
                'ma_alignment': 'Unknown',
                'price_vs_ema_fast': np.nan,
                'price_vs_sma_medium': np.nan,
                'price_vs_sma_slow': np.nan
            }
    
    def analyze_ticker(self, ticker: str, df: pd.DataFrame, timeframe: str = 'daily') -> Dict[str, any]:
        """
        Perform complete stage analysis for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data
            timeframe: 'daily' or 'weekly'
            
        Returns:
            Dictionary with stage analysis results
        """
        if df is None or df.empty:
            return self._get_empty_results(ticker, timeframe)
        
        try:
            # Calculate moving averages and ATR
            df_with_mas = self.calculate_moving_averages(df)
            
            # Get the most recent row for analysis
            if len(df_with_mas) == 0:
                return self._get_empty_results(ticker, timeframe)
            
            latest_row = df_with_mas.iloc[-1]
            
            # Detect stage
            stage_code, stage_name, stage_color = self.detect_stage(latest_row)
            
            # Calculate additional metrics
            metrics = self.calculate_additional_metrics(latest_row)
            
            # Compile results
            results = {
                'ticker': ticker,
                'timeframe': timeframe,
                f'{timeframe}_sa_code': stage_code,
                f'{timeframe}_sa_name': stage_name,
                f'{timeframe}_sa_color_code': stage_color,
                f'{timeframe}_atr_ratio_sma_50': metrics['atr_ratio'],
                f'{timeframe}_ma_alignment': metrics['ma_alignment'],
                f'{timeframe}_price_vs_ema_fast': metrics['price_vs_ema_fast'],
                f'{timeframe}_price_vs_sma_medium': metrics['price_vs_sma_medium'],
                f'{timeframe}_price_vs_sma_slow': metrics['price_vs_sma_slow']
            }
            
            return results
            
        except Exception as e:
            logger.debug(f"Error analyzing ticker {ticker}: {e}")
            return self._get_empty_results(ticker, timeframe)
    
    def _get_empty_results(self, ticker: str, timeframe: str) -> Dict[str, any]:
        """Get empty results structure for failed analysis."""
        return {
            'ticker': ticker,
            'timeframe': timeframe,
            f'{timeframe}_sa_code': '??',
            f'{timeframe}_sa_name': 'Undefined',
            f'{timeframe}_sa_color_code': 'gray',
            f'{timeframe}_atr_ratio_sma_50': np.nan,
            f'{timeframe}_ma_alignment': 'Unknown',
            f'{timeframe}_price_vs_ema_fast': np.nan,
            f'{timeframe}_price_vs_sma_medium': np.nan,
            f'{timeframe}_price_vs_sma_slow': np.nan
        }


def get_stage_summary_stats(stage_results: Dict[str, Dict]) -> Dict[str, int]:
    """
    Get summary statistics of stage distribution across all tickers.
    
    Args:
        stage_results: Dictionary of ticker -> stage results
        
    Returns:
        Dictionary with stage counts
    """
    daily_stages = {}
    weekly_stages = {}
    
    for ticker, results in stage_results.items():
        if 'daily_sa_code' in results:
            daily_stage = results['daily_sa_code']
            daily_stages[daily_stage] = daily_stages.get(daily_stage, 0) + 1
            
        if 'weekly_sa_code' in results:
            weekly_stage = results['weekly_sa_code']
            weekly_stages[weekly_stage] = weekly_stages.get(weekly_stage, 0) + 1
    
    return {
        'daily_stages': daily_stages,
        'weekly_stages': weekly_stages,
        'total_tickers': len(stage_results)
    }