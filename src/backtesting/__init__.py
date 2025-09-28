"""
BACKTESTING Module - Comprehensive Strategy Performance Analysis
================================================================

Post-screener backtesting engine for evaluating screener strategy performance.
Analyzes historical signals from all screeners to provide performance insights.

Main Components:
- BacktestManager: Main orchestrator for backtesting workflow
- SignalProcessor: Load and process screener CSV outputs
- StrategyBacktester: Individual strategy performance analysis
- PortfolioBacktester: Multi-strategy portfolio simulation
- PerformanceAnalyzer: Metrics calculation and risk analysis
- ReportGenerator: Performance reporting and visualization

Main Entry Point:
    from src.backtesting import run_backtesting_analysis

    results = run_backtesting_analysis(config, user_config)
"""

from .backtest_manager import run_backtesting_analysis, BacktestManager

__all__ = ['run_backtesting_analysis', 'BacktestManager']