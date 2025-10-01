"""
ADL Moving Average Analysis Module
==================================

Analyzes ADL moving averages and alignment patterns (Step 4).

Calculates MAs on the ADL line, checks for bullish/bearish alignment,
detects crossovers, and measures trend strength.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple

from .adl_calculator import ADLCalculator
from .adl_utils import parse_semicolon_list, detect_trend, normalize_score

logger = logging.getLogger(__name__)


class ADLMAAnalyzer:
    """
    Moving Average analysis for ADL line.

    Analyzes MA alignment, crossovers, and trend confirmation.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize MA analyzer.

        Args:
            params: Dictionary containing:
                - ma_periods: String like "20;50;100" or list [20, 50, 100]
                - ma_type: 'SMA' or 'EMA'
                - ma_bullish_alignment_required: Require bullish alignment (default True)
                - ma_crossover_detection: Enable crossover detection (default True)
                - ma_crossover_lookback: Periods to check for crossovers (default 10)
                - ma_min_slope_threshold: Minimum positive slope (default 0.01)
                - ma_min_alignment_score: Minimum score threshold (default 70)
        """
        # Parse MA periods
        periods_param = params.get('ma_periods', '20;50;100')
        if isinstance(periods_param, str):
            self.ma_periods = parse_semicolon_list(periods_param, int)
        else:
            self.ma_periods = periods_param

        self.ma_type = params.get('ma_type', 'SMA').upper()
        self.bullish_alignment_required = params.get('ma_bullish_alignment_required', True)
        self.crossover_detection = params.get('ma_crossover_detection', True)
        self.crossover_lookback = params.get('ma_crossover_lookback', 10)
        self.min_slope_threshold = params.get('ma_min_slope_threshold', 0.01)
        self.min_alignment_score = params.get('ma_min_alignment_score', 70.0)

        self.calculator = ADLCalculator()

        logger.debug(f"MA Analyzer initialized: periods={self.ma_periods}, "
                    f"type={self.ma_type}")

    def calculate_adl_mas(self, df: pd.DataFrame,
                          ticker: str = None) -> Optional[Dict[str, Any]]:
        """
        Calculate ADL moving averages and analyze alignment.

        Args:
            df: OHLCV DataFrame
            ticker: Ticker symbol (for logging)

        Returns:
            Dictionary with MA analysis results or None if insufficient data
        """
        try:
            # Calculate ADL
            adl = self.calculator.calculate_adl(df)
            if adl is None:
                return None

            # Check if we have enough data
            max_period = max(self.ma_periods)
            if len(adl) < max_period + self.crossover_lookback:
                logger.debug(f"{ticker}: Insufficient data for MA analysis "
                           f"(need {max_period + self.crossover_lookback}, have {len(adl)})")
                return None

            # Calculate MAs
            ma_values = {}
            for period in self.ma_periods:
                ma = self._calculate_ma(adl, period)
                if ma is not None:
                    ma_values[f'adl_ma_{period}'] = ma.iloc[-1]

            if len(ma_values) != len(self.ma_periods):
                logger.debug(f"{ticker}: Failed to calculate all MAs")
                return None

            # Check MA alignment
            ma_alignment = self._check_ma_alignment(ma_values)

            # Calculate alignment score
            alignment_score = self._calculate_alignment_score(ma_values, ma_alignment)

            # Detect crossovers if enabled
            recent_crossover = None
            crossover_date = None
            if self.crossover_detection:
                recent_crossover, crossover_date = self._detect_ma_crossovers(adl, df.index)

            # Calculate MA slopes
            ma_slopes = self._calculate_ma_slopes(adl)

            result = {
                'ticker': ticker,
                **{k: round(v, 2) for k, v in ma_values.items()},
                'ma_alignment': ma_alignment,
                'ma_alignment_score': round(alignment_score, 2),
                'recent_crossover': recent_crossover,
                'crossover_date': str(crossover_date) if crossover_date else None,
                **{f'ma_{p}_slope': round(ma_slopes.get(p, 0), 6) for p in self.ma_periods},
                'all_slopes_positive': all(ma_slopes.get(p, 0) > 0 for p in self.ma_periods),
                'meets_criteria': (
                    alignment_score >= self.min_alignment_score and
                    (not self.bullish_alignment_required or ma_alignment == 'bullish')
                )
            }

            return result

        except Exception as e:
            logger.error(f"Error in MA analysis for {ticker}: {e}")
            return None

    def _calculate_ma(self, series: pd.Series, period: int) -> Optional[pd.Series]:
        """
        Calculate moving average (SMA or EMA).

        Args:
            series: Data series
            period: MA period

        Returns:
            MA series or None
        """
        try:
            if self.ma_type == 'EMA':
                return series.ewm(span=period, adjust=False).mean()
            else:  # SMA
                return series.rolling(window=period).mean()
        except Exception as e:
            logger.error(f"Error calculating {self.ma_type}{period}: {e}")
            return None

    def _check_ma_alignment(self, ma_values: Dict[str, float]) -> str:
        """
        Check MA alignment pattern.

        Args:
            ma_values: Dictionary of MA values

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        # Extract MA values in order (shortest to longest)
        sorted_periods = sorted(self.ma_periods)
        mas = [ma_values.get(f'adl_ma_{p}', 0) for p in sorted_periods]

        if not mas or len(mas) < 2:
            return 'neutral'

        # Check for bullish alignment (shorter > longer)
        is_bullish = all(mas[i] > mas[i+1] for i in range(len(mas)-1))
        if is_bullish:
            return 'bullish'

        # Check for bearish alignment (shorter < longer)
        is_bearish = all(mas[i] < mas[i+1] for i in range(len(mas)-1))
        if is_bearish:
            return 'bearish'

        return 'neutral'

    def _calculate_alignment_score(self, ma_values: Dict[str, float],
                                   ma_alignment: str) -> float:
        """
        Calculate MA alignment quality score (0-100).

        Args:
            ma_values: Dictionary of MA values
            ma_alignment: Alignment classification

        Returns:
            Alignment score (0-100)
        """
        sorted_periods = sorted(self.ma_periods)
        mas = [ma_values.get(f'adl_ma_{p}', 0) for p in sorted_periods]

        if not mas or len(mas) < 2:
            return 0.0

        # Base score from alignment type
        if ma_alignment == 'bullish':
            base_score = 60
        elif ma_alignment == 'bearish':
            base_score = 20
        else:
            base_score = 40

        # Calculate separation between MAs (wider = stronger)
        separations = []
        for i in range(len(mas)-1):
            if mas[i+1] != 0:
                sep = abs((mas[i] - mas[i+1]) / mas[i+1]) * 100
                separations.append(sep)

        # Average separation as % of longer MA
        avg_separation = np.mean(separations) if separations else 0

        # Normalize separation: 0-5% maps to 0-30 bonus points
        separation_bonus = min(30, avg_separation * 6)

        # Check for good separation (not too tight, not too wide)
        if 1 < avg_separation < 5:  # Ideal range
            separation_bonus += 10

        total_score = min(100, base_score + separation_bonus)

        return total_score

    def _detect_ma_crossovers(self, adl: pd.Series,
                             index: pd.DatetimeIndex) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
        """
        Detect recent MA crossovers.

        Args:
            adl: ADL Series
            index: DateTime index

        Returns:
            Tuple of (crossover_type, crossover_date)
            crossover_type: 'golden', 'death', or None
        """
        try:
            if len(self.ma_periods) < 2:
                return None, None

            # Use first two periods (fastest crossover pair)
            short_period = self.ma_periods[0]
            long_period = self.ma_periods[1]

            # Calculate MAs for lookback period
            ma_short = self._calculate_ma(adl, short_period)
            ma_long = self._calculate_ma(adl, long_period)

            if ma_short is None or ma_long is None:
                return None, None

            # Check recent period for crossovers
            lookback_start = max(0, len(adl) - self.crossover_lookback)
            recent_short = ma_short.iloc[lookback_start:]
            recent_long = ma_long.iloc[lookback_start:]

            # Detect crossover
            for i in range(1, len(recent_short)):
                prev_short = recent_short.iloc[i-1]
                prev_long = recent_long.iloc[i-1]
                curr_short = recent_short.iloc[i]
                curr_long = recent_long.iloc[i]

                # Golden cross: short crosses above long
                if prev_short <= prev_long and curr_short > curr_long:
                    crossover_date = index[lookback_start + i]
                    return 'golden', crossover_date

                # Death cross: short crosses below long
                elif prev_short >= prev_long and curr_short < curr_long:
                    crossover_date = index[lookback_start + i]
                    return 'death', crossover_date

            return None, None

        except Exception as e:
            logger.debug(f"Error detecting crossovers: {e}")
            return None, None

    def _calculate_ma_slopes(self, adl: pd.Series) -> Dict[int, float]:
        """
        Calculate slope for each MA using linear regression.

        Args:
            adl: ADL Series

        Returns:
            Dictionary of {period: slope}
        """
        slopes = {}

        for period in self.ma_periods:
            ma = self._calculate_ma(adl, period)
            if ma is None:
                slopes[period] = 0.0
                continue

            # Use recent data for slope calculation
            lookback = min(period, len(ma))
            recent_ma = ma.iloc[-lookback:].dropna()

            if len(recent_ma) < 2:
                slopes[period] = 0.0
                continue

            # Linear regression
            try:
                x = np.arange(len(recent_ma))
                slope = np.polyfit(x, recent_ma.values, 1)[0]
                slopes[period] = slope
            except:
                slopes[period] = 0.0

        return slopes

    def filter_by_criteria(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results by MA criteria thresholds.

        Args:
            results: List of MA analysis results

        Returns:
            Filtered list of results meeting criteria
        """
        filtered = [
            r for r in results
            if r.get('meets_criteria', False)
        ]

        logger.info(f"MA filter: {len(filtered)}/{len(results)} pass criteria")

        return filtered

    def rank_by_alignment(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank results by MA alignment score (descending).

        Args:
            results: List of MA analysis results

        Returns:
            Sorted list of results
        """
        sorted_results = sorted(
            results,
            key=lambda x: (x.get('ma_alignment_score', 0), x.get('all_slopes_positive', False)),
            reverse=True
        )

        # Add rank
        for i, result in enumerate(sorted_results, 1):
            result['ma_rank'] = i

        return sorted_results