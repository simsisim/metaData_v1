"""
ADL Calculator Module
====================

Core ADL (Accumulation/Distribution Line) calculation logic.

Formula: AD = AD + ((Close - Low) - (High - Close)) / (High - Low) * Volume

Refactored from existing ADLINE.py and adl_screener.py implementations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

from .adl_utils import validate_ohlcv_data, safe_division

logger = logging.getLogger(__name__)


class ADLCalculator:
    """
    Core ADL calculation class with support for percentage changes
    and rolling window analysis.
    """

    def __init__(self):
        """Initialize ADL calculator."""
        pass

    def calculate_adl(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculate Accumulation/Distribution Line.

        Formula:
        1. Money Flow Multiplier (MFM) = ((Close - Low) - (High - Close)) / (High - Low)
        2. Money Flow Volume (MFV) = MFM × Volume
        3. ADL = Cumulative Sum of MFV

        Args:
            df: OHLCV DataFrame

        Returns:
            ADL Series or None if calculation fails
        """
        try:
            # Validate input
            if not validate_ohlcv_data(df):
                logger.warning("Invalid OHLCV data for ADL calculation")
                return None

            # Calculate Money Flow Multiplier (MFM)
            high_low_diff = df['High'] - df['Low']

            # Avoid division by zero for doji/narrow range bars
            # Replace zero with NaN to handle separately
            high_low_diff = high_low_diff.replace(0, np.nan)

            # Calculate close position in range
            close_position = ((df['Close'] - df['Low']) - (df['High'] - df['Close']))

            # Calculate MFM
            mfm = close_position / high_low_diff

            # Handle NaN values (doji bars) - assume neutral (0)
            mfm = mfm.fillna(0)

            # Calculate Money Flow Volume (MFV)
            mfv = mfm * df['Volume']

            # Calculate ADL (cumulative sum of MFV)
            adl = mfv.cumsum()

            # Set name for series
            adl.name = 'ADL'

            return adl

        except Exception as e:
            logger.error(f"Error calculating ADL: {e}")
            return None

    def calculate_adl_pct_change(self, adl_series: pd.Series,
                                  period: int) -> Optional[pd.Series]:
        """
        Calculate percentage change of ADL over specified period.

        Args:
            adl_series: ADL Series
            period: Lookback period for percentage change

        Returns:
            Percentage change Series or None if calculation fails
        """
        try:
            if adl_series is None or len(adl_series) < period:
                return None

            # Calculate percentage change
            pct_change = adl_series.pct_change(periods=period) * 100

            # Handle inf and NaN values
            pct_change = pct_change.replace([np.inf, -np.inf], np.nan)

            pct_change.name = f'ADL_{period}d_pct_change'

            return pct_change

        except Exception as e:
            logger.error(f"Error calculating ADL percentage change: {e}")
            return None

    def calculate_multiple_pct_changes(self, adl_series: pd.Series,
                                       periods: list) -> Dict[str, pd.Series]:
        """
        Calculate ADL percentage changes for multiple periods.

        Args:
            adl_series: ADL Series
            periods: List of periods (e.g., [5, 10, 20])

        Returns:
            Dictionary of {period: pct_change_series}
        """
        results = {}

        for period in periods:
            pct_change = self.calculate_adl_pct_change(adl_series, period)
            if pct_change is not None:
                results[f'{period}d'] = pct_change

        return results

    def calculate_rolling_adl(self, df: pd.DataFrame,
                             window: int) -> Optional[pd.DataFrame]:
        """
        Calculate ADL over rolling windows.

        Args:
            df: OHLCV DataFrame
            window: Rolling window size

        Returns:
            DataFrame with rolling ADL values or None if calculation fails
        """
        try:
            if not validate_ohlcv_data(df, min_length=window):
                return None

            # Calculate full ADL first
            adl = self.calculate_adl(df)
            if adl is None:
                return None

            # Calculate rolling statistics
            rolling_mean = adl.rolling(window=window).mean()
            rolling_std = adl.rolling(window=window).std()
            rolling_min = adl.rolling(window=window).min()
            rolling_max = adl.rolling(window=window).max()

            # Create result DataFrame
            result = pd.DataFrame({
                'ADL': adl,
                f'ADL_MA_{window}': rolling_mean,
                f'ADL_STD_{window}': rolling_std,
                f'ADL_MIN_{window}': rolling_min,
                f'ADL_MAX_{window}': rolling_max
            }, index=df.index)

            return result

        except Exception as e:
            logger.error(f"Error calculating rolling ADL: {e}")
            return None

    def calculate_adl_with_components(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate ADL along with intermediate components.

        Returns all calculation steps for transparency and debugging.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary containing:
            - 'adl': ADL Series
            - 'mfm': Money Flow Multiplier Series
            - 'mfv': Money Flow Volume Series
        """
        try:
            if not validate_ohlcv_data(df):
                return {}

            # Calculate components
            high_low_diff = df['High'] - df['Low']
            high_low_diff = high_low_diff.replace(0, np.nan)

            close_position = ((df['Close'] - df['Low']) - (df['High'] - df['Close']))
            mfm = (close_position / high_low_diff).fillna(0)

            mfv = mfm * df['Volume']
            adl = mfv.cumsum()

            return {
                'adl': adl,
                'mfm': mfm,
                'mfv': mfv
            }

        except Exception as e:
            logger.error(f"Error calculating ADL with components: {e}")
            return {}

    def calculate_adl_trend(self, adl_series: pd.Series,
                           lookback: int = 50) -> pd.Series:
        """
        Calculate trend direction of ADL using linear regression slope.

        Args:
            adl_series: ADL Series
            lookback: Lookback period for trend analysis

        Returns:
            Series with trend values: 1 (accumulation), -1 (distribution), 0 (neutral)
        """
        try:
            if adl_series is None or len(adl_series) < lookback:
                return pd.Series(0, index=adl_series.index)

            trend = pd.Series(0, index=adl_series.index)

            for i in range(lookback, len(adl_series)):
                recent_adl = adl_series.iloc[i-lookback:i+1]

                # Calculate slope using linear regression
                x = np.arange(len(recent_adl))
                slope = np.polyfit(x, recent_adl.values, 1)[0]

                if slope > 0:
                    trend.iloc[i] = 1  # Accumulation
                elif slope < 0:
                    trend.iloc[i] = -1  # Distribution
                # else: neutral (0)

            trend.name = 'ADL_Trend'
            return trend

        except Exception as e:
            logger.error(f"Error calculating ADL trend: {e}")
            return pd.Series(0, index=adl_series.index)

    def calculate_adl_strength(self, adl_series: pd.Series,
                              price_series: pd.Series,
                              lookback: int = 20) -> pd.Series:
        """
        Calculate ADL strength relative to price movement.

        Measures how well ADL confirms price trends.

        Args:
            adl_series: ADL Series
            price_series: Price Series
            lookback: Lookback period

        Returns:
            Series with strength values (0-100)
        """
        try:
            if adl_series is None or price_series is None:
                return pd.Series(50, index=adl_series.index)

            strength = pd.Series(50, index=adl_series.index)

            for i in range(lookback, len(adl_series)):
                # Calculate ADL and price changes over lookback
                adl_change = adl_series.iloc[i] - adl_series.iloc[i-lookback]
                price_change = price_series.iloc[i] - price_series.iloc[i-lookback]

                # Both positive or both negative = strong confirmation
                if (adl_change > 0 and price_change > 0) or (adl_change < 0 and price_change < 0):
                    # Calculate correlation strength
                    correlation = min(abs(adl_change), abs(price_change)) / max(abs(adl_change), abs(price_change), 1)
                    strength.iloc[i] = 50 + (correlation * 50)  # 50-100 range
                else:
                    # Divergence - weak confirmation
                    strength.iloc[i] = 50 - (abs(adl_change - price_change) / (abs(adl_change) + abs(price_change) + 1) * 50)

            strength.name = 'ADL_Strength'
            return strength

        except Exception as e:
            logger.error(f"Error calculating ADL strength: {e}")
            return pd.Series(50, index=adl_series.index)


def calculate_adl_for_ticker(df: pd.DataFrame, ticker: str = None) -> Optional[pd.Series]:
    """
    Convenience function to calculate ADL for a single ticker.

    Args:
        df: OHLCV DataFrame
        ticker: Ticker symbol (optional, for logging)

    Returns:
        ADL Series or None
    """
    calculator = ADLCalculator()
    adl = calculator.calculate_adl(df)

    if adl is None and ticker:
        logger.warning(f"Failed to calculate ADL for {ticker}")

    return adl