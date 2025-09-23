"""
SR Intermarket Ratios Module
============================

Calculate intermarket ratios and market breadth indicators for sustainability analysis.

Key Ratios:
- QQQ:SPY - Tech vs Market strength
- XLY:XLP - Risk-on vs Risk-off sentiment
- IWF:IWD - Growth vs Value rotation
- TRAN:UTIL - Transportation vs Utilities (economic strength)

Market Breadth:
- Advance/Decline metrics
- New highs/lows
- % above moving averages
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def calculate_ratio(numerator_data: pd.Series, denominator_data: pd.Series,
                   ratio_name: str = "Ratio") -> pd.Series:
    """
    Calculate ratio between two price series.

    Args:
        numerator_data: Price series for numerator
        denominator_data: Price series for denominator
        ratio_name: Name for the ratio

    Returns:
        Ratio series
    """
    try:
        # Align series by index
        aligned_num, aligned_den = numerator_data.align(denominator_data, join='inner')

        # Avoid division by zero
        ratio = aligned_num / aligned_den.replace(0, np.nan)

        ratio.name = ratio_name
        return ratio

    except Exception as e:
        logger.error(f"Error calculating ratio {ratio_name}: {e}")
        return pd.Series(dtype=float, name=ratio_name)


def calculate_intermarket_ratios(market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate key intermarket ratios for sustainability analysis.

    Args:
        market_data: Dict with ticker DataFrames containing OHLCV data

    Returns:
        DataFrame with calculated ratios
    """
    try:
        ratios_df = pd.DataFrame()

        # Define ratio pairs
        ratio_definitions = [
            ('QQQ', 'SPY', 'QQQ_SPY', 'Tech vs Market Strength'),
            ('XLY', 'XLP', 'XLY_XLP', 'Risk-On vs Risk-Off'),
            ('IWF', 'IWD', 'IWF_IWD', 'Growth vs Value'),
            ('QQQ', 'XLP', 'QQQ_XLP', 'Tech vs Defensive'),
            ('SPY', 'TLT', 'SPY_TLT', 'Stocks vs Bonds'),
        ]

        # Calculate each ratio
        for num_ticker, den_ticker, ratio_name, description in ratio_definitions:
            if num_ticker in market_data and den_ticker in market_data:
                try:
                    # Use Close prices for ratio calculation
                    num_close = market_data[num_ticker]['Close']
                    den_close = market_data[den_ticker]['Close']

                    ratio = calculate_ratio(num_close, den_close, ratio_name)
                    if not ratio.empty:
                        ratios_df[ratio_name] = ratio
                        logger.debug(f"Calculated {ratio_name}: {description}")

                except Exception as e:
                    logger.warning(f"Error calculating {ratio_name}: {e}")

        # Calculate ratio trends and momentum
        if not ratios_df.empty:
            ratios_df = add_ratio_indicators(ratios_df)

        logger.info(f"Calculated {len(ratios_df.columns)} intermarket ratios")
        return ratios_df

    except Exception as e:
        logger.error(f"Error calculating intermarket ratios: {e}")
        return pd.DataFrame()


def add_ratio_indicators(ratios_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to ratio data.

    Args:
        ratios_df: DataFrame with ratio values

    Returns:
        DataFrame with additional indicator columns
    """
    try:
        enhanced_df = ratios_df.copy()

        for col in ratios_df.columns:
            ratio_data = ratios_df[col].dropna()

            if len(ratio_data) < 20:
                continue

            try:
                # Calculate moving averages
                enhanced_df[f'{col}_SMA20'] = ratio_data.rolling(20).mean()
                enhanced_df[f'{col}_SMA50'] = ratio_data.rolling(50).mean()

                # Calculate percentage change
                enhanced_df[f'{col}_Pct_1d'] = ratio_data.pct_change(1) * 100
                enhanced_df[f'{col}_Pct_5d'] = ratio_data.pct_change(5) * 100
                enhanced_df[f'{col}_Pct_20d'] = ratio_data.pct_change(20) * 100

                # Calculate Z-score (normalized position)
                rolling_mean = ratio_data.rolling(50).mean()
                rolling_std = ratio_data.rolling(50).std()
                enhanced_df[f'{col}_ZScore'] = (ratio_data - rolling_mean) / rolling_std

                # Trend direction
                sma20 = enhanced_df[f'{col}_SMA20']
                sma50 = enhanced_df[f'{col}_SMA50']
                enhanced_df[f'{col}_Trend'] = np.where(sma20 > sma50, 1, -1)

            except Exception as e:
                logger.warning(f"Error adding indicators for {col}: {e}")

        return enhanced_df

    except Exception as e:
        logger.error(f"Error adding ratio indicators: {e}")
        return ratios_df


def calculate_market_breadth(market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Calculate market breadth indicators.

    Args:
        market_data: Dict with ticker DataFrames

    Returns:
        Dict with breadth indicators
    """
    try:
        breadth_results = {}

        if len(market_data) < 10:
            logger.warning("Insufficient data for meaningful breadth calculation")
            return breadth_results

        # Collect all close prices
        all_closes = pd.DataFrame()
        for ticker, data in market_data.items():
            if 'Close' in data.columns and not data['Close'].empty:
                all_closes[ticker] = data['Close']

        if all_closes.empty:
            logger.warning("No valid close price data for breadth calculation")
            return breadth_results

        # Calculate daily percentage changes
        daily_changes = all_closes.pct_change()

        # Advance/Decline metrics
        advances = (daily_changes > 0).sum(axis=1)
        declines = (daily_changes < 0).sum(axis=1)
        unchanged = (daily_changes == 0).sum(axis=1)

        breadth_results['Advances'] = advances
        breadth_results['Declines'] = declines
        breadth_results['Unchanged'] = unchanged
        breadth_results['AD_Ratio'] = advances / (declines + 0.001)  # Avoid division by zero
        breadth_results['AD_Diff'] = advances - declines
        breadth_results['Advance_Pct'] = (advances / len(all_closes.columns)) * 100

        # New highs/lows (52-week)
        if len(all_closes) >= 252:
            rolling_highs = all_closes.rolling(252).max()
            rolling_lows = all_closes.rolling(252).min()

            new_highs = (all_closes == rolling_highs).sum(axis=1)
            new_lows = (all_closes == rolling_lows).sum(axis=1)

            breadth_results['New_Highs'] = new_highs
            breadth_results['New_Lows'] = new_lows
            breadth_results['HL_Ratio'] = new_highs / (new_lows + 0.001)

        # % above moving averages
        for ma_period in [20, 50, 200]:
            if len(all_closes) >= ma_period:
                mas = all_closes.rolling(ma_period).mean()
                above_ma = (all_closes > mas).sum(axis=1)
                breadth_results[f'Above_SMA{ma_period}_Pct'] = (above_ma / len(all_closes.columns)) * 100

        # McClellan Oscillator (simplified)
        if 'AD_Diff' in breadth_results:
            ad_diff = breadth_results['AD_Diff']
            ema19 = ad_diff.ewm(span=19).mean()
            ema39 = ad_diff.ewm(span=39).mean()
            breadth_results['McClellan_Osc'] = ema19 - ema39

        logger.info(f"Calculated {len(breadth_results)} market breadth indicators")
        return breadth_results

    except Exception as e:
        logger.error(f"Error calculating market breadth: {e}")
        return {}


def calculate_sector_rotation(sector_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate sector rotation metrics.

    Args:
        sector_data: Dict with sector ETF data

    Returns:
        DataFrame with rotation analysis
    """
    try:
        # Common sector ETFs
        sector_etfs = ['XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLU', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC']

        rotation_df = pd.DataFrame()

        # Calculate relative performance vs SPY benchmark
        spy_data = sector_data.get('SPY')
        if spy_data is None or 'Close' not in spy_data.columns:
            logger.warning("SPY data not available for sector rotation calculation")
            return rotation_df

        spy_returns = spy_data['Close'].pct_change()

        for sector_etf in sector_etfs:
            if sector_etf in sector_data and 'Close' in sector_data[sector_etf].columns:
                sector_returns = sector_data[sector_etf]['Close'].pct_change()

                # Calculate relative strength
                rel_strength = (sector_returns - spy_returns).rolling(20).mean()
                rotation_df[f'{sector_etf}_RelStrength'] = rel_strength

                # Calculate momentum
                momentum = sector_data[sector_etf]['Close'].pct_change(20)
                rotation_df[f'{sector_etf}_Momentum'] = momentum

        logger.info(f"Calculated sector rotation for {len(rotation_df.columns)//2} sectors")
        return rotation_df

    except Exception as e:
        logger.error(f"Error calculating sector rotation: {e}")
        return pd.DataFrame()


def get_ratio_signals(ratios_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate trading signals from ratio analysis.

    Args:
        ratios_df: DataFrame with ratio data and indicators

    Returns:
        Dict with current signals for each ratio
    """
    try:
        signals = {}

        for col in ratios_df.columns:
            if not col.endswith(('_SMA20', '_SMA50', '_ZScore', '_Trend', '_Pct_1d', '_Pct_5d', '_Pct_20d')):
                # This is a base ratio column
                try:
                    latest_value = ratios_df[col].iloc[-1]
                    zscore_col = f'{col}_ZScore'
                    trend_col = f'{col}_Trend'

                    if zscore_col in ratios_df.columns and trend_col in ratios_df.columns:
                        latest_zscore = ratios_df[zscore_col].iloc[-1]
                        latest_trend = ratios_df[trend_col].iloc[-1]

                        # Generate signal based on Z-score and trend
                        if pd.notna(latest_zscore) and pd.notna(latest_trend):
                            if latest_zscore > 1.5 and latest_trend == 1:
                                signals[col] = "Strong Bullish"
                            elif latest_zscore > 0.5 and latest_trend == 1:
                                signals[col] = "Bullish"
                            elif latest_zscore < -1.5 and latest_trend == -1:
                                signals[col] = "Strong Bearish"
                            elif latest_zscore < -0.5 and latest_trend == -1:
                                signals[col] = "Bearish"
                            else:
                                signals[col] = "Neutral"
                        else:
                            signals[col] = "No Signal"

                except Exception as e:
                    logger.debug(f"Error generating signal for {col}: {e}")
                    signals[col] = "Error"

        return signals

    except Exception as e:
        logger.error(f"Error generating ratio signals: {e}")
        return {}