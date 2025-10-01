"""
ADL Short-term Momentum Analysis Module
=======================================

Analyzes short-term ADL percentage changes and momentum shifts (Step 3).

Calculates 5-day, 10-day, 20-day percentage changes and detects
acceleration/deceleration patterns.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

from .adl_calculator import ADLCalculator
from .adl_utils import parse_semicolon_list, normalize_score

logger = logging.getLogger(__name__)


class ADLShortTermAnalyzer:
    """
    Short-term momentum analysis for ADL line.

    Detects momentum shifts and acceleration patterns.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize short-term analyzer.

        Args:
            params: Dictionary containing:
                - short_term_periods: String like "5;10;20" or list [5, 10, 20]
                - short_term_momentum_threshold: Minimum % change for signal (default 5)
                - short_term_acceleration_detect: Enable acceleration detection (default True)
                - short_term_min_score: Minimum momentum score (default 50)
        """
        # Parse periods
        periods_param = params.get('short_term_periods', '5;10;20')
        if isinstance(periods_param, str):
            self.periods = parse_semicolon_list(periods_param, int)
        else:
            self.periods = periods_param

        self.momentum_threshold = params.get('short_term_momentum_threshold', 5.0)
        self.acceleration_detect = params.get('short_term_acceleration_detect', True)
        self.min_score = params.get('short_term_min_score', 50.0)

        self.calculator = ADLCalculator()

        logger.debug(f"Short-term Analyzer initialized: periods={self.periods}, "
                    f"threshold={self.momentum_threshold}%")

    def calculate_short_term_changes(self, df: pd.DataFrame,
                                     ticker: str = None) -> Optional[Dict[str, Any]]:
        """
        Calculate short-term ADL percentage changes.

        Args:
            df: OHLCV DataFrame
            ticker: Ticker symbol (for logging)

        Returns:
            Dictionary with short-term metrics or None if insufficient data
        """
        try:
            # Calculate ADL
            adl = self.calculator.calculate_adl(df)
            if adl is None:
                return None

            # Check if we have enough data
            max_period = max(self.periods)
            if len(adl) < max_period:
                logger.debug(f"{ticker}: Insufficient data for short-term analysis "
                           f"(need {max_period}, have {len(adl)})")
                return None

            # Calculate percentage changes for each period
            pct_changes = {}
            for period in self.periods:
                pct_change = self.calculator.calculate_adl_pct_change(adl, period)
                if pct_change is not None:
                    current_change = pct_change.iloc[-1]
                    pct_changes[f'{period}d'] = current_change

            if not pct_changes:
                return None

            # Detect momentum signal
            momentum_signal = self._detect_momentum_signal(pct_changes)

            # Detect acceleration pattern if enabled
            acceleration_pattern = None
            if self.acceleration_detect:
                acceleration_pattern = self._detect_acceleration_pattern(pct_changes)

            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(pct_changes, momentum_signal)

            # Detect inflection point
            inflection_point = self._detect_inflection_point(adl, pct_changes)

            result = {
                'ticker': ticker,
                **{f'adl_{k}_pct_change': round(v, 2) for k, v in pct_changes.items()},
                'momentum_signal': momentum_signal,
                'acceleration_pattern': acceleration_pattern,
                'momentum_score': round(momentum_score, 2),
                'inflection_point': inflection_point,
                'meets_criteria': momentum_score >= self.min_score
            }

            return result

        except Exception as e:
            logger.error(f"Error in short-term analysis for {ticker}: {e}")
            return None

    def _detect_momentum_signal(self, pct_changes: Dict[str, float]) -> str:
        """
        Detect momentum signal based on percentage changes.

        Args:
            pct_changes: Dictionary of {period: pct_change}

        Returns:
            'acceleration', 'deceleration', 'momentum', or 'neutral'
        """
        # Get changes in order
        changes = [pct_changes.get(f'{p}d', 0) for p in self.periods]

        if not changes or all(c == 0 for c in changes):
            return 'neutral'

        # Check for acceleration (shorter periods > longer periods)
        if len(changes) >= 2:
            if all(changes[i] > changes[i+1] for i in range(len(changes)-1)):
                # Each shorter period is stronger than longer period
                if changes[0] > self.momentum_threshold:
                    return 'acceleration'

            # Check for deceleration (shorter periods < longer periods)
            elif all(changes[i] < changes[i+1] for i in range(len(changes)-1)):
                return 'deceleration'

        # Check for general positive momentum
        positive_count = sum(1 for c in changes if c > 0)
        if positive_count >= len(changes) / 2:
            if any(c > self.momentum_threshold for c in changes):
                return 'momentum'

        return 'neutral'

    def _detect_acceleration_pattern(self, pct_changes: Dict[str, float]) -> Optional[str]:
        """
        Detect specific acceleration/deceleration patterns.

        Args:
            pct_changes: Dictionary of {period: pct_change}

        Returns:
            Pattern description or None
        """
        changes = [pct_changes.get(f'{p}d', 0) for p in self.periods]

        if len(changes) < 3:
            return None

        # Strong acceleration: all positive and increasing
        if all(c > 0 for c in changes) and all(changes[i] > changes[i+1] for i in range(len(changes)-1)):
            return 'strong_acceleration'

        # Moderate acceleration: mostly positive and trending up
        positive_count = sum(1 for c in changes if c > 0)
        if positive_count >= 2 and changes[0] > changes[-1]:
            return 'moderate_acceleration'

        # Deceleration: positive but slowing
        if all(c > 0 for c in changes) and all(changes[i] < changes[i+1] for i in range(len(changes)-1)):
            return 'slowing_momentum'

        # Negative acceleration: all negative and worsening
        if all(c < 0 for c in changes) and all(changes[i] < changes[i+1] for i in range(len(changes)-1)):
            return 'negative_acceleration'

        return None

    def _calculate_momentum_score(self, pct_changes: Dict[str, float],
                                  momentum_signal: str) -> float:
        """
        Calculate momentum score (0-100) based on short-term changes.

        Args:
            pct_changes: Dictionary of {period: pct_change}
            momentum_signal: Momentum signal classification

        Returns:
            Momentum score (0-100)
        """
        if not pct_changes:
            return 0.0

        # Base score from average percentage change
        changes = list(pct_changes.values())
        avg_change = np.mean(changes)

        # Normalize average change to 0-100 scale
        # Typical range: -20% to +20%
        base_score = normalize_score(avg_change, -20, 20, reverse=False)

        # Apply weights: shorter periods = higher weight
        weighted_changes = []
        weights = [3, 2, 1]  # Weight for 5d, 10d, 20d
        for i, period in enumerate(self.periods):
            change = pct_changes.get(f'{period}d', 0)
            weight = weights[i] if i < len(weights) else 1
            weighted_changes.append(change * weight)

        weighted_avg = sum(weighted_changes) / sum(weights[:len(self.periods)])
        weighted_score = normalize_score(weighted_avg, -20, 20, reverse=False)

        # Combine base and weighted scores
        combined_score = (base_score * 0.4) + (weighted_score * 0.6)

        # Apply momentum signal bonuses
        if momentum_signal == 'acceleration':
            combined_score = min(100, combined_score + 15)
        elif momentum_signal == 'momentum':
            combined_score = min(100, combined_score + 10)
        elif momentum_signal == 'deceleration':
            combined_score = max(0, combined_score - 10)

        return combined_score

    def _detect_inflection_point(self, adl: pd.Series,
                                 pct_changes: Dict[str, float]) -> bool:
        """
        Detect if ADL is at a significant inflection point.

        An inflection point occurs when momentum shifts significantly.

        Args:
            adl: ADL Series
            pct_changes: Dictionary of {period: pct_change}

        Returns:
            True if inflection point detected
        """
        try:
            if len(adl) < 20:
                return False

            # Check for recent sign change in shortest period
            shortest_period = min(self.periods)
            recent_changes = []

            for i in range(shortest_period, min(shortest_period + 5, len(adl))):
                lookback_adl = adl.iloc[-i-1]
                current_adl = adl.iloc[-i]
                change = current_adl - lookback_adl
                recent_changes.append(change)

            if len(recent_changes) < 2:
                return False

            # Check for sign change (positive to negative or vice versa)
            sign_changes = sum(
                1 for i in range(len(recent_changes)-1)
                if (recent_changes[i] > 0) != (recent_changes[i+1] > 0)
            )

            # Also check if magnitude is significant
            current_change = pct_changes.get(f'{shortest_period}d', 0)
            is_significant = abs(current_change) > self.momentum_threshold

            return sign_changes >= 1 and is_significant

        except Exception as e:
            logger.debug(f"Error detecting inflection point: {e}")
            return False

    def filter_by_criteria(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results by short-term criteria thresholds.

        Args:
            results: List of short-term analysis results

        Returns:
            Filtered list of results meeting criteria
        """
        filtered = [
            r for r in results
            if r.get('meets_criteria', False)
        ]

        logger.info(f"Short-term filter: {len(filtered)}/{len(results)} pass criteria")

        return filtered

    def rank_by_momentum(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank results by momentum score (descending).

        Args:
            results: List of short-term analysis results

        Returns:
            Sorted list of results
        """
        sorted_results = sorted(
            results,
            key=lambda x: x.get('momentum_score', 0),
            reverse=True
        )

        # Add rank
        for i, result in enumerate(sorted_results, 1):
            result['momentum_rank'] = i

        return sorted_results