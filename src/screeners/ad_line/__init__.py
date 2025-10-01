"""
ADL (Accumulation/Distribution Line) Enhanced Screener Module
=============================================================

Multi-dimensional accumulation analysis with composite scoring and ranking.

This module implements a 5-step ADL analysis methodology:
1. Basic ADL calculation (existing functionality preserved)
2. Month-over-month accumulation analysis
3. Short-term momentum detection
4. Moving average overlay and alignment
5. Composite scoring and ranking

Author: Claude Code
Date: 2025-09-30
"""

from .adl_calculator import ADLCalculator
from .adl_mom_analysis import ADLMoMAnalyzer
from .adl_short_term import ADLShortTermAnalyzer
from .adl_ma_analysis import ADLMAAnalyzer
from .adl_composite_scoring import ADLCompositeScorer
from .adl_utils import (
    validate_ohlcv_data,
    parse_semicolon_list,
    calculate_score_category
)

__all__ = [
    'ADLCalculator',
    'ADLMoMAnalyzer',
    'ADLShortTermAnalyzer',
    'ADLMAAnalyzer',
    'ADLCompositeScorer',
    'validate_ohlcv_data',
    'parse_semicolon_list',
    'calculate_score_category',
]

__version__ = '1.0.0'