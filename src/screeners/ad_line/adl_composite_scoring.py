"""
ADL Composite Scoring Module
============================

Combines all analysis modules into composite score and ranking (Step 5).

Applies configurable weights to long-term, short-term, and MA scores
to generate final composite score and rank stocks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

from .adl_utils import calculate_score_category

logger = logging.getLogger(__name__)


class ADLCompositeScorer:
    """
    Composite scoring and ranking system.

    Combines component scores with configurable weights.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize composite scorer.

        Args:
            params: Dictionary containing:
                - composite_weight_longterm: Weight for long-term score (default 0.4)
                - composite_weight_shortterm: Weight for short-term score (default 0.3)
                - composite_weight_ma_align: Weight for MA alignment score (default 0.3)
                - composite_min_score: Minimum composite score threshold (default 70)
                - ranking_method: 'composite', 'momentum', or 'accumulation'
                - top_candidates_count: Number of top candidates (default 50)
        """
        self.weight_longterm = params.get('composite_weight_longterm', 0.4)
        self.weight_shortterm = params.get('composite_weight_shortterm', 0.3)
        self.weight_ma_align = params.get('composite_weight_ma_align', 0.3)
        self.min_score = params.get('composite_min_score', 70.0)
        self.ranking_method = params.get('ranking_method', 'composite')
        self.top_candidates_count = params.get('top_candidates_count', 50)

        # Validate weights sum to 1.0
        total_weight = self.weight_longterm + self.weight_shortterm + self.weight_ma_align
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weight_longterm /= total_weight
            self.weight_shortterm /= total_weight
            self.weight_ma_align /= total_weight

        logger.debug(f"Composite Scorer initialized: weights=({self.weight_longterm:.2f}, "
                    f"{self.weight_shortterm:.2f}, {self.weight_ma_align:.2f})")

    def calculate_composite_score(self,
                                  longterm_score: float,
                                  shortterm_score: float,
                                  ma_score: float) -> float:
        """
        Calculate weighted composite score.

        Args:
            longterm_score: Long-term accumulation score (0-100)
            shortterm_score: Short-term momentum score (0-100)
            ma_score: MA alignment score (0-100)

        Returns:
            Composite score (0-100)
        """
        composite = (
            (self.weight_longterm * longterm_score) +
            (self.weight_shortterm * shortterm_score) +
            (self.weight_ma_align * ma_score)
        )

        return composite

    def score_ticker(self,
                    ticker: str,
                    mom_result: Optional[Dict[str, Any]],
                    shortterm_result: Optional[Dict[str, Any]],
                    ma_result: Optional[Dict[str, Any]],
                    price: float,
                    volume: int,
                    date: str) -> Optional[Dict[str, Any]]:
        """
        Calculate composite score for a single ticker.

        Args:
            ticker: Ticker symbol
            mom_result: MoM analysis result dictionary
            shortterm_result: Short-term analysis result dictionary
            ma_result: MA analysis result dictionary
            price: Current price
            volume: Current volume
            date: Current date

        Returns:
            Composite score dictionary or None if insufficient data
        """
        # Extract component scores (default to 0 if module disabled or failed)
        longterm_score = mom_result.get('consistency_score', 0) if mom_result else 0
        shortterm_score = shortterm_result.get('momentum_score', 0) if shortterm_result else 0
        ma_score = ma_result.get('ma_alignment_score', 0) if ma_result else 0

        # Calculate composite score
        composite_score = self.calculate_composite_score(
            longterm_score, shortterm_score, ma_score
        )

        # Calculate component contributions
        longterm_contribution = self.weight_longterm * longterm_score
        shortterm_contribution = self.weight_shortterm * shortterm_score
        ma_contribution = self.weight_ma_align * ma_score

        # Determine if meets threshold
        meets_threshold = composite_score >= self.min_score

        # Calculate score category
        score_category = calculate_score_category(composite_score)

        # Identify key strengths and weaknesses
        component_scores = {
            'longterm': longterm_score,
            'shortterm': shortterm_score,
            'ma_alignment': ma_score
        }
        key_strengths = [k for k, v in component_scores.items() if v >= 80]
        key_weaknesses = [k for k, v in component_scores.items() if v < 60]

        result = {
            'ticker': ticker,
            'date': date,
            'price': round(price, 2),
            'volume': int(volume),
            'composite_score': round(composite_score, 2),
            'score_category': score_category,
            'longterm_score': round(longterm_score, 2),
            'shortterm_score': round(shortterm_score, 2),
            'ma_alignment_score': round(ma_score, 2),
            'longterm_contribution': round(longterm_contribution, 2),
            'shortterm_contribution': round(shortterm_contribution, 2),
            'ma_contribution': round(ma_contribution, 2),
            'meets_threshold': meets_threshold,
            'key_strengths': ';'.join(key_strengths) if key_strengths else 'none',
            'key_weaknesses': ';'.join(key_weaknesses) if key_weaknesses else 'none'
        }

        # Add component-specific details
        if mom_result:
            result['mom_current_streak'] = mom_result.get('current_streak', 0)
            result['mom_avg_monthly_change'] = mom_result.get('avg_monthly_change', 0)

        if shortterm_result:
            result['momentum_signal'] = shortterm_result.get('momentum_signal', 'neutral')
            result['acceleration_pattern'] = shortterm_result.get('acceleration_pattern', None)

        if ma_result:
            result['ma_alignment'] = ma_result.get('ma_alignment', 'neutral')
            result['recent_crossover'] = ma_result.get('recent_crossover', None)

        return result

    def rank_stocks(self, composite_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank stocks by composite score or alternative method.

        Args:
            composite_results: List of composite score dictionaries

        Returns:
            Ranked list of results
        """
        if not composite_results:
            return []

        # Determine sort key based on ranking method
        if self.ranking_method == 'momentum':
            sort_key = lambda x: x.get('shortterm_score', 0)
        elif self.ranking_method == 'accumulation':
            sort_key = lambda x: x.get('longterm_score', 0)
        else:  # composite
            sort_key = lambda x: x.get('composite_score', 0)

        # Sort descending
        ranked_results = sorted(composite_results, key=sort_key, reverse=True)

        # Add rank
        for i, result in enumerate(ranked_results, 1):
            result['rank'] = i

        logger.info(f"Ranked {len(ranked_results)} stocks by {self.ranking_method}")

        return ranked_results

    def filter_by_threshold(self, ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results by minimum composite score threshold.

        Args:
            ranked_results: List of ranked results

        Returns:
            Filtered list meeting threshold
        """
        filtered = [
            r for r in ranked_results
            if r.get('composite_score', 0) >= self.min_score
        ]

        logger.info(f"Composite filter: {len(filtered)}/{len(ranked_results)} "
                   f"meet threshold ({self.min_score})")

        return filtered

    def generate_top_candidates(self, ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract top N candidates from ranked results.

        Args:
            ranked_results: List of ranked results

        Returns:
            Top N candidates
        """
        top_candidates = ranked_results[:self.top_candidates_count]

        logger.info(f"Generated top {len(top_candidates)} candidates")

        return top_candidates

    def generate_score_breakdown(self, composite_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed score breakdown for a single ticker.

        Args:
            composite_result: Composite score dictionary

        Returns:
            Detailed breakdown dictionary
        """
        breakdown = {
            'ticker': composite_result.get('ticker'),
            'composite_score': composite_result.get('composite_score', 0),
            'rank': composite_result.get('rank'),
            'components': {
                'longterm': {
                    'score': composite_result.get('longterm_score', 0),
                    'weight': self.weight_longterm,
                    'contribution': composite_result.get('longterm_contribution', 0),
                    'percentage_of_total': round(
                        (composite_result.get('longterm_contribution', 0) /
                         composite_result.get('composite_score', 1)) * 100, 1
                    )
                },
                'shortterm': {
                    'score': composite_result.get('shortterm_score', 0),
                    'weight': self.weight_shortterm,
                    'contribution': composite_result.get('shortterm_contribution', 0),
                    'percentage_of_total': round(
                        (composite_result.get('shortterm_contribution', 0) /
                         composite_result.get('composite_score', 1)) * 100, 1
                    )
                },
                'ma_alignment': {
                    'score': composite_result.get('ma_alignment_score', 0),
                    'weight': self.weight_ma_align,
                    'contribution': composite_result.get('ma_contribution', 0),
                    'percentage_of_total': round(
                        (composite_result.get('ma_contribution', 0) /
                         composite_result.get('composite_score', 1)) * 100, 1
                    )
                }
            },
            'strengths': composite_result.get('key_strengths', 'none'),
            'weaknesses': composite_result.get('key_weaknesses', 'none'),
            'score_category': composite_result.get('score_category', 'weak')
        }

        return breakdown

    def create_summary_statistics(self, composite_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary statistics for all composite results.

        Args:
            composite_results: List of composite score dictionaries

        Returns:
            Summary statistics dictionary
        """
        if not composite_results:
            return {}

        composite_scores = [r.get('composite_score', 0) for r in composite_results]
        longterm_scores = [r.get('longterm_score', 0) for r in composite_results]
        shortterm_scores = [r.get('shortterm_score', 0) for r in composite_results]
        ma_scores = [r.get('ma_alignment_score', 0) for r in composite_results]

        # Count by score category
        category_counts = {}
        for result in composite_results:
            category = result.get('score_category', 'weak')
            category_counts[category] = category_counts.get(category, 0) + 1

        summary = {
            'total_stocks': len(composite_results),
            'meeting_threshold': sum(1 for r in composite_results if r.get('meets_threshold')),
            'composite_score': {
                'mean': round(np.mean(composite_scores), 2),
                'median': round(np.median(composite_scores), 2),
                'std': round(np.std(composite_scores), 2),
                'min': round(np.min(composite_scores), 2),
                'max': round(np.max(composite_scores), 2)
            },
            'longterm_score': {
                'mean': round(np.mean(longterm_scores), 2),
                'median': round(np.median(longterm_scores), 2)
            },
            'shortterm_score': {
                'mean': round(np.mean(shortterm_scores), 2),
                'median': round(np.median(shortterm_scores), 2)
            },
            'ma_score': {
                'mean': round(np.mean(ma_scores), 2),
                'median': round(np.median(ma_scores), 2)
            },
            'category_counts': category_counts
        }

        return summary