"""
ADL Month-over-Month Analysis Module
====================================

Analyzes monthly accumulation consistency patterns (Step 2).

Identifies stocks with consistent accumulation over multiple months
within specified growth thresholds.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

from .adl_calculator import ADLCalculator
from .adl_utils import calculate_percentage_change, normalize_score

logger = logging.getLogger(__name__)


class ADLMoMAnalyzer:
    """
    Month-over-month accumulation analysis.

    Tracks ADL growth consistency over monthly intervals.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize MoM analyzer.

        Args:
            params: Dictionary containing:
                - mom_period: Trading days per month (default 22)
                - mom_min_threshold_pct: Minimum monthly growth % (default 15)
                - mom_max_threshold_pct: Maximum monthly growth % (default 30)
                - mom_consecutive_months: Required consecutive months (default 3)
                - mom_lookback_months: Historical months to analyze (default 6)
                - mom_min_consistency_score: Minimum score threshold (default 60)
        """
        self.period = params.get('mom_period', 22)
        self.min_threshold = params.get('mom_min_threshold_pct', 15.0)
        self.max_threshold = params.get('mom_max_threshold_pct', 30.0)
        self.consecutive_months = params.get('mom_consecutive_months', 3)
        self.lookback_months = params.get('mom_lookback_months', 6)
        self.min_consistency_score = params.get('mom_min_consistency_score', 60.0)

        self.calculator = ADLCalculator()

        logger.debug(f"MoM Analyzer initialized: period={self.period}, "
                    f"threshold={self.min_threshold}-{self.max_threshold}%")

    def analyze_monthly_accumulation(self, df: pd.DataFrame,
                                     ticker: str = None) -> Optional[Dict[str, Any]]:
        """
        Analyze monthly accumulation patterns.

        Args:
            df: OHLCV DataFrame
            ticker: Ticker symbol (for logging)

        Returns:
            Dictionary with analysis results or None if insufficient data
        """
        try:
            # Calculate ADL
            adl = self.calculator.calculate_adl(df)
            if adl is None:
                return None

            # Check if we have enough data
            required_length = self.period * self.lookback_months
            if len(adl) < required_length:
                logger.debug(f"{ticker}: Insufficient data for MoM analysis "
                           f"(need {required_length}, have {len(adl)})")
                return None

            # Extract monthly ADL values
            monthly_adl = self._extract_monthly_values(adl)

            if len(monthly_adl) < 2:
                return None

            # Calculate month-over-month percentage changes
            monthly_changes = self._calculate_monthly_changes(monthly_adl)

            if not monthly_changes:
                return None

            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(monthly_changes)

            # Detect accumulation streaks
            streak_info = self._detect_accumulation_streaks(monthly_changes)

            # Calculate average monthly change
            avg_monthly_change = np.mean(monthly_changes) if monthly_changes else 0.0

            # Check if meets criteria
            meets_criteria = (
                consistency_score >= self.min_consistency_score and
                streak_info['current_streak'] >= self.consecutive_months
            )

            result = {
                'ticker': ticker,
                'monthly_changes': monthly_changes,
                'avg_monthly_change': round(avg_monthly_change, 2),
                'consistency_score': round(consistency_score, 2),
                'qualifying_streak': streak_info['qualifying_streak'],
                'current_streak': streak_info['current_streak'],
                'total_months_analyzed': len(monthly_changes),
                'months_in_range': streak_info['months_in_range'],
                'meets_criteria': meets_criteria
            }

            return result

        except Exception as e:
            logger.error(f"Error in MoM analysis for {ticker}: {e}")
            return None

    def _extract_monthly_values(self, adl: pd.Series) -> List[float]:
        """
        Extract ADL values at monthly intervals.

        Args:
            adl: ADL Series

        Returns:
            List of monthly ADL values
        """
        monthly_values = []

        # Start from the end and work backwards
        total_months = min(self.lookback_months + 1, len(adl) // self.period)

        for i in range(total_months):
            idx = len(adl) - 1 - (i * self.period)
            if idx >= 0:
                monthly_values.append(adl.iloc[idx])

        # Reverse to chronological order
        return monthly_values[::-1]

    def _calculate_monthly_changes(self, monthly_values: List[float]) -> List[float]:
        """
        Calculate month-over-month percentage changes.

        Args:
            monthly_values: List of monthly ADL values

        Returns:
            List of percentage changes
        """
        changes = []

        for i in range(1, len(monthly_values)):
            previous = monthly_values[i-1]
            current = monthly_values[i]

            pct_change = calculate_percentage_change(current, previous)
            changes.append(pct_change)

        return changes

    def _calculate_consistency_score(self, monthly_changes: List[float]) -> float:
        """
        Calculate consistency score (0-100) based on accumulation patterns.

        Higher score = more months within ideal threshold range.

        Args:
            monthly_changes: List of monthly percentage changes

        Returns:
            Consistency score (0-100)
        """
        if not monthly_changes:
            return 0.0

        # Count months within threshold range
        in_range_count = sum(
            1 for change in monthly_changes
            if self.min_threshold <= change <= self.max_threshold
        )

        # Count months with positive growth (even if outside ideal range)
        positive_count = sum(1 for change in monthly_changes if change > 0)

        # Calculate base score from in-range months
        base_score = (in_range_count / len(monthly_changes)) * 70

        # Add bonus for positive (but out of range) months
        positive_bonus = ((positive_count - in_range_count) / len(monthly_changes)) * 20

        # Add bonus for consistency (low variance)
        variance_bonus = self._calculate_variance_bonus(monthly_changes)

        total_score = min(100, base_score + positive_bonus + variance_bonus)

        return total_score

    def _calculate_variance_bonus(self, monthly_changes: List[float]) -> float:
        """
        Calculate bonus score for low variance (consistent growth).

        Args:
            monthly_changes: List of monthly percentage changes

        Returns:
            Variance bonus (0-10)
        """
        if len(monthly_changes) < 2:
            return 0.0

        # Calculate standard deviation
        std_dev = np.std(monthly_changes)

        # Low std dev = high consistency = higher bonus
        # Normalize: std_dev of 0-10 maps to bonus of 10-0
        variance_bonus = max(0, 10 - (std_dev / 2))

        return variance_bonus

    def _detect_accumulation_streaks(self, monthly_changes: List[float]) -> Dict[str, int]:
        """
        Detect consecutive months meeting accumulation criteria.

        Args:
            monthly_changes: List of monthly percentage changes

        Returns:
            Dictionary with streak information
        """
        if not monthly_changes:
            return {
                'qualifying_streak': 0,
                'current_streak': 0,
                'months_in_range': 0
            }

        # Count total months in range
        months_in_range = sum(
            1 for change in monthly_changes
            if self.min_threshold <= change <= self.max_threshold
        )

        # Find longest qualifying streak
        max_streak = 0
        current_streak = 0

        for change in monthly_changes:
            if self.min_threshold <= change <= self.max_threshold:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Calculate current streak (from the end)
        current_end_streak = 0
        for change in reversed(monthly_changes):
            if self.min_threshold <= change <= self.max_threshold:
                current_end_streak += 1
            else:
                break

        return {
            'qualifying_streak': max_streak,
            'current_streak': current_end_streak,
            'months_in_range': months_in_range
        }

    def filter_by_criteria(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results by MoM criteria thresholds.

        Args:
            results: List of MoM analysis results

        Returns:
            Filtered list of results meeting criteria
        """
        filtered = [
            r for r in results
            if r.get('meets_criteria', False)
        ]

        logger.info(f"MoM filter: {len(filtered)}/{len(results)} pass criteria")

        return filtered

    def rank_by_consistency(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank results by consistency score (descending).

        Args:
            results: List of MoM analysis results

        Returns:
            Sorted list of results
        """
        sorted_results = sorted(
            results,
            key=lambda x: (x.get('consistency_score', 0), x.get('current_streak', 0)),
            reverse=True
        )

        # Add rank
        for i, result in enumerate(sorted_results, 1):
            result['mom_rank'] = i

        return sorted_results