"""
Technical Indicators Module
==========================

This module provides comprehensive technical indicator calculations and charting capabilities.

Submodules:
- indicators_calculation: Core technical indicator calculations (TSI, MACD, MFI, COG, etc.)
- indicators_charts: Charting and visualization functionality
- PPO: Percentage Price Oscillator for chart generation
- RSI: Relative Strength Index for chart generation
- MAs: Moving Averages (EMA, SMA) for chart generation
- indicator_parser: Universal parameter parsing for CSV-driven indicators
"""

from .indicators_calculation import *
from .indicators_charts import *
from .indicators_processor import *

# Import new modular indicators for SR module
from .PPO import calculate_ppo_for_chart, parse_ppo_params
from .RSI import calculate_rsi_for_chart, parse_rsi_params
from .MAs import calculate_ema_for_chart, calculate_sma_for_chart, parse_ma_params
from .MACD import calculate_macd_for_chart, parse_macd_params
from .ADLINE import calculate_adline_for_chart, parse_adline_params
from .indicator_parser import (
    parse_indicator_string, calculate_indicator, get_indicator_function,
    validate_indicator_string, get_supported_indicators
)

__all__ = [
    # Existing indicators
    'calculate_kurutoga',
    'calculate_tsi',
    'calculate_macd',
    'calculate_mfi',
    'calculate_cog',
    'calculate_momentum',
    'calculate_rsi',
    'calculate_ma_crosses',
    'calculate_easy_trade',
    'create_indicator_chart',
    'process_indicators_batch',
    'get_indicators_config_for_timeframe',

    # New modular indicators for SR module
    'calculate_ppo_for_chart',
    'parse_ppo_params',
    'calculate_rsi_for_chart',
    'parse_rsi_params',
    'calculate_ema_for_chart',
    'calculate_sma_for_chart',
    'parse_ma_params',
    'calculate_macd_for_chart',
    'parse_macd_params',
    'calculate_adline_for_chart',
    'parse_adline_params',

    # Universal indicator parser
    'parse_indicator_string',
    'calculate_indicator',
    'get_indicator_function',
    'validate_indicator_string',
    'get_supported_indicators'
]