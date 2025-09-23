"""
Sustainability Ratios (SR) Module
=================================

Market timing and sustainability analysis through intermarket ratios,
sentiment indicators, and multi-panel chart generation.

Main Components:
- sr_calculations: Core SR calculation engine
- sr_config_reader: CSV panel configuration parser
- sr_dashboard_generator: Multi-panel chart generation
- sr_ratios: Intermarket ratio calculations

Usage:
    from src.sustainability_ratios import run_sr_analysis

    # Run complete SR analysis
    run_sr_analysis(config, data_reader)
"""

from .sr_calculations import run_sr_analysis, SRProcessor
from .sr_config_reader import parse_panel_config, load_sr_configuration
from .sr_ratios import calculate_intermarket_ratios, calculate_market_breadth
from .sr_dashboard_generator import generate_sr_dashboard, create_multi_panel_chart

__all__ = [
    'run_sr_analysis',
    'SRProcessor',
    'parse_panel_config',
    'load_sr_configuration',
    'calculate_intermarket_ratios',
    'calculate_market_breadth',
    'generate_sr_dashboard',
    'create_multi_panel_chart'
]