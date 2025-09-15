"""
Market Pulse Analysis Module
===========================

Comprehensive market timing and trend analysis for major indexes.
Provides pipeline-level market analysis separate from individual stock screening.

Components:
- MarketPulseManager: Main orchestrator for all market analysis
- calculators/: Specific metric calculators (GMI, etc.)
- indicators/: Market trend and timing indicators
"""

from .market_pulse_manager import MarketPulseManager

__all__ = ['MarketPulseManager']