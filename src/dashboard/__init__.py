"""
Trading Dashboard Module
=======================

Production dashboard system for daily market overview generation.
Integrates with the main trading pipeline to create Excel-based market briefings.
"""

from .dashboard_builder import TradingDashboardBuilder
from .real_data_connector import RealDataConnector

__all__ = ['TradingDashboardBuilder', 'RealDataConnector']