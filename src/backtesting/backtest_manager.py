"""
Backtesting Manager - Main Orchestrator
=======================================

Main coordination module for comprehensive backtesting workflow.
Manages the entire backtesting process from loading screener results
to generating performance reports.

Workflow:
1. Load all screener CSV outputs
2. Process and validate signals
3. Run individual strategy backtests
4. Run portfolio-level backtesting
5. Generate performance analysis
6. Create comprehensive reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

from .signal_processor import SignalProcessor
from .strategy_backtester import StrategyBacktester
from .portfolio_backtester import PortfolioBacktester
from .performance_analyzer import PerformanceAnalyzer
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class BacktestManager:
    """
    Main orchestrator for backtesting workflow.

    Coordinates all backtesting components to provide comprehensive
    performance analysis of screener strategies.
    """

    def __init__(self, config, user_config):
        """
        Initialize BacktestManager.

        Args:
            config: System configuration object
            user_config: User configuration with backtesting parameters
        """
        self.config = config
        self.user_config = user_config
        self.signal_processor = SignalProcessor(config)
        self.strategy_backtester = StrategyBacktester(config, user_config)
        self.portfolio_backtester = PortfolioBacktester(config, user_config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.report_generator = ReportGenerator(config, user_config)

        # Results storage
        self.strategy_results = {}
        self.portfolio_results = {}
        self.performance_metrics = {}

    def run_backtesting_workflow(self) -> Dict:
        """
        Execute complete backtesting workflow.

        Returns:
            Dict with backtesting results and performance metrics
        """
        try:
            logger.info("Starting comprehensive backtesting workflow")

            # Step 1: Load and process all screener signals
            logger.info("Step 1: Loading screener signals")
            signals_data = self._load_screener_signals()

            if not signals_data:
                logger.warning("No screener signals found - skipping backtesting")
                return {'status': 'skipped', 'reason': 'no_signals'}

            # Step 2: Run individual strategy backtests
            logger.info("Step 2: Running individual strategy backtests")
            self.strategy_results = self._run_strategy_backtests(signals_data)

            # Step 3: Run portfolio backtesting
            logger.info("Step 3: Running portfolio backtesting")
            self.portfolio_results = self._run_portfolio_backtest(signals_data)

            # Step 4: Analyze performance metrics
            logger.info("Step 4: Analyzing performance metrics")
            self.performance_metrics = self._analyze_performance()

            # Step 5: Generate reports
            logger.info("Step 5: Generating performance reports")
            report_paths = self._generate_reports()

            logger.info("Backtesting workflow completed successfully")

            return {
                'status': 'completed',
                'strategy_results': self.strategy_results,
                'portfolio_results': self.portfolio_results,
                'performance_metrics': self.performance_metrics,
                'report_paths': report_paths,
                'summary': self._create_summary()
            }

        except Exception as e:
            logger.error(f"Backtesting workflow failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _load_screener_signals(self) -> Dict:
        """Load and process signals from all screener CSV files."""
        try:
            # Find all screener output files
            screener_files = self.signal_processor.find_screener_files()

            if not screener_files:
                logger.warning("No screener output files found")
                return {}

            # Load and process signals
            signals_data = self.signal_processor.load_all_signals(screener_files)

            logger.info(f"Loaded signals from {len(screener_files)} screener files")
            return signals_data

        except Exception as e:
            logger.error(f"Error loading screener signals: {e}")
            return {}

    def _run_strategy_backtests(self, signals_data: Dict) -> Dict:
        """Run backtests for individual strategies."""
        strategy_results = {}

        for strategy_name, signals in signals_data.items():
            try:
                logger.info(f"Backtesting strategy: {strategy_name}")

                # Run strategy backtest
                result = self.strategy_backtester.backtest_strategy(
                    strategy_name, signals
                )

                strategy_results[strategy_name] = result

                logger.info(f"Completed backtest for {strategy_name}: "
                          f"{len(signals)} signals, "
                          f"{result.get('total_return', 0):.2f}% return")

            except Exception as e:
                logger.error(f"Error backtesting {strategy_name}: {e}")
                continue

        return strategy_results

    def _run_portfolio_backtest(self, signals_data: Dict) -> Dict:
        """Run portfolio-level backtesting."""
        try:
            return self.portfolio_backtester.backtest_portfolio(signals_data)
        except Exception as e:
            logger.error(f"Error in portfolio backtesting: {e}")
            return {}

    def _analyze_performance(self) -> Dict:
        """Analyze performance metrics for all strategies and portfolio."""
        try:
            return self.performance_analyzer.analyze_all_performance(
                self.strategy_results, self.portfolio_results
            )
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {}

    def _generate_reports(self) -> Dict:
        """Generate comprehensive performance reports."""
        try:
            return self.report_generator.generate_all_reports(
                self.strategy_results,
                self.portfolio_results,
                self.performance_metrics
            )
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            return {}

    def _create_summary(self) -> Dict:
        """Create high-level summary of backtesting results."""
        try:
            summary = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'strategies_tested': len(self.strategy_results),
                'total_signals': sum(
                    len(result.get('trades', []))
                    for result in self.strategy_results.values()
                ),
                'best_strategy': None,
                'worst_strategy': None,
                'portfolio_return': self.portfolio_results.get('total_return', 0),
                'portfolio_sharpe': self.portfolio_results.get('sharpe_ratio', 0)
            }

            # Find best and worst performing strategies
            if self.strategy_results:
                strategy_returns = {
                    name: result.get('total_return', 0)
                    for name, result in self.strategy_results.items()
                }

                if strategy_returns:
                    best_strategy = max(strategy_returns, key=strategy_returns.get)
                    worst_strategy = min(strategy_returns, key=strategy_returns.get)

                    summary['best_strategy'] = {
                        'name': best_strategy,
                        'return': strategy_returns[best_strategy]
                    }
                    summary['worst_strategy'] = {
                        'name': worst_strategy,
                        'return': strategy_returns[worst_strategy]
                    }

            return summary

        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return {}


def run_backtesting_analysis(config, user_config) -> Dict:
    """
    Main entry point for backtesting analysis.

    Args:
        config: System configuration object
        user_config: User configuration with backtesting settings

    Returns:
        Dict with comprehensive backtesting results
    """
    logger.info("Starting backtesting analysis")

    # Check if backtesting is enabled
    if not getattr(user_config, 'backtesting', False):
        logger.info("Backtesting disabled - skipping analysis")
        return {'status': 'disabled'}

    # Initialize and run backtesting manager
    backtest_manager = BacktestManager(config, user_config)
    results = backtest_manager.run_backtesting_workflow()

    logger.info("Backtesting analysis completed")
    return results