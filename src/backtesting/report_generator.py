"""
Report Generator - Backtesting Performance Reports
==================================================

Generates comprehensive reports and visualizations for backtesting results.
Creates both text-based summaries and data export files.

Key Features:
- Strategy performance reports
- Portfolio analysis reports
- Comparative analysis charts
- Risk assessment reports
- CSV data exports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive backtesting reports.

    Creates detailed performance reports, comparative analysis,
    and data exports for backtesting results.
    """

    def __init__(self, config, user_config):
        """
        Initialize ReportGenerator.

        Args:
            config: System configuration object
            user_config: User configuration with reporting parameters
        """
        self.config = config
        self.user_config = user_config

        # Output directories
        self.results_dir = Path(config.directories.get('RESULTS_DIR', 'results'))
        self.reports_dir = self.results_dir / 'backtesting'
        self.reports_dir.mkdir(exist_ok=True)

        # Report parameters
        self.include_charts = getattr(user_config, 'bt_include_charts', True)
        self.export_trades = getattr(user_config, 'bt_export_trades', True)
        self.detailed_analysis = getattr(user_config, 'bt_detailed_analysis', True)

    def generate_all_reports(
        self,
        strategy_results: Dict,
        portfolio_results: Dict,
        performance_metrics: Dict
    ) -> Dict[str, str]:
        """
        Generate all backtesting reports.

        Args:
            strategy_results: Individual strategy results
            portfolio_results: Portfolio results
            performance_metrics: Performance analysis results

        Returns:
            Dict mapping report types to file paths
        """
        try:
            logger.info("Generating backtesting reports")

            report_paths = {}
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Generate strategy summary report
            strategy_summary_path = self._generate_strategy_summary_report(
                strategy_results, performance_metrics, timestamp
            )
            report_paths['strategy_summary'] = strategy_summary_path

            # Generate portfolio report
            if portfolio_results:
                portfolio_report_path = self._generate_portfolio_report(
                    portfolio_results, performance_metrics, timestamp
                )
                report_paths['portfolio_report'] = portfolio_report_path

            # Generate comparative analysis report
            comparative_report_path = self._generate_comparative_report(
                strategy_results, portfolio_results, performance_metrics, timestamp
            )
            report_paths['comparative_analysis'] = comparative_report_path

            # Generate risk analysis report
            risk_report_path = self._generate_risk_analysis_report(
                performance_metrics, timestamp
            )
            report_paths['risk_analysis'] = risk_report_path

            # Export detailed data
            if self.export_trades:
                data_exports = self._export_detailed_data(
                    strategy_results, portfolio_results, timestamp
                )
                report_paths.update(data_exports)

            # Generate executive summary
            executive_summary_path = self._generate_executive_summary(
                strategy_results, portfolio_results, performance_metrics, timestamp
            )
            report_paths['executive_summary'] = executive_summary_path

            logger.info(f"Generated {len(report_paths)} backtesting reports")
            return report_paths

        except Exception as e:
            logger.error(f"Error generating backtesting reports: {e}")
            return {}

    def _generate_strategy_summary_report(
        self,
        strategy_results: Dict,
        performance_metrics: Dict,
        timestamp: str
    ) -> str:
        """Generate strategy summary report."""
        try:
            report_path = self.reports_dir / f'strategy_summary_{timestamp}.txt'

            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("BACKTESTING STRATEGY SUMMARY REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Overall summary
                f.write("OVERALL SUMMARY\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total Strategies Analyzed: {len(strategy_results)}\n")

                if strategy_results:
                    all_trades = sum(
                        len(result.get('trades', []))
                        for result in strategy_results.values()
                    )
                    f.write(f"Total Trades Executed: {all_trades}\n")

                    # Best performing strategy
                    best_strategy = self._find_best_strategy(strategy_results)
                    if best_strategy:
                        f.write(f"Best Performing Strategy: {best_strategy['name']} "
                               f"({best_strategy['return']:.2f}% return)\n")

                f.write("\n")

                # Individual strategy details
                f.write("INDIVIDUAL STRATEGY PERFORMANCE\n")
                f.write("-" * 50 + "\n")

                for strategy_name, result in strategy_results.items():
                    f.write(f"\n{strategy_name.upper()}\n")
                    f.write("-" * len(strategy_name) + "\n")

                    performance = result.get('performance', {})
                    f.write(f"Total Return: {performance.get('total_return', 0):.2f}%\n")
                    f.write(f"Total Trades: {performance.get('total_trades', 0)}\n")
                    f.write(f"Win Rate: {performance.get('win_rate', 0):.1f}%\n")
                    f.write(f"Profit Factor: {performance.get('profit_factor', 0):.2f}\n")
                    f.write(f"Max Drawdown: {performance.get('max_drawdown', 0):.2f}%\n")

                    # Enhanced metrics from performance analysis
                    strategy_analysis = performance_metrics.get('strategy_analysis', {}).get(strategy_name, {})
                    enhanced_metrics = strategy_analysis.get('enhanced_risk_metrics', {})

                    if enhanced_metrics:
                        f.write(f"Sharpe Ratio: {enhanced_metrics.get('sharpe_ratio', 0):.2f}\n")
                        f.write(f"Sortino Ratio: {enhanced_metrics.get('sortino_ratio', 0):.2f}\n")

                # Strategy rankings
                comparative_analysis = performance_metrics.get('comparative_analysis', {})
                rankings = comparative_analysis.get('strategy_rankings', {})

                if rankings:
                    f.write(f"\nSTRATEGY RANKINGS\n")
                    f.write("-" * 50 + "\n")
                    for metric, ranking in rankings.items():
                        f.write(f"\nBy {metric.replace('_', ' ').title()}:\n")
                        for i, (strategy, value) in enumerate(ranking[:5], 1):
                            f.write(f"  {i}. {strategy}: {value:.2f}\n")

            logger.info(f"Generated strategy summary report: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating strategy summary report: {e}")
            return ""

    def _generate_portfolio_report(
        self,
        portfolio_results: Dict,
        performance_metrics: Dict,
        timestamp: str
    ) -> str:
        """Generate portfolio analysis report."""
        try:
            report_path = self.reports_dir / f'portfolio_analysis_{timestamp}.txt'

            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("BACKTESTING PORTFOLIO ANALYSIS REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Portfolio overview
                f.write("PORTFOLIO OVERVIEW\n")
                f.write("-" * 50 + "\n")

                strategies = portfolio_results.get('strategies_included', [])
                f.write(f"Strategies Included: {', '.join(strategies)}\n")
                f.write(f"Allocation Method: {portfolio_results.get('allocation_method', 'N/A')}\n")
                f.write(f"Total Portfolio Trades: {portfolio_results.get('total_portfolio_trades', 0)}\n")

                # Portfolio performance
                performance = portfolio_results.get('performance', {})
                f.write(f"\nPORTFOLIO PERFORMANCE\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total Return: {performance.get('total_return', 0):.2f}%\n")
                f.write(f"Win Rate: {performance.get('win_rate', 0):.1f}%\n")
                f.write(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Max Drawdown: {performance.get('max_drawdown', 0):.2f}%\n")
                f.write(f"Diversification Ratio: {performance.get('diversification_ratio', 0):.2f}\n")

                # Strategy contributions
                strategy_contributions = performance.get('strategy_contributions', {})
                if strategy_contributions:
                    f.write(f"\nSTRATEGY CONTRIBUTIONS\n")
                    f.write("-" * 50 + "\n")
                    for strategy, contribution in strategy_contributions.items():
                        f.write(f"{strategy}: ${contribution:.2f}\n")

                # Portfolio analysis from performance metrics
                portfolio_analysis = performance_metrics.get('portfolio_analysis', {})
                if portfolio_analysis:
                    contribution_analysis = portfolio_analysis.get('contribution_analysis', {})
                    if contribution_analysis:
                        f.write(f"\nDETAILED STRATEGY ANALYSIS\n")
                        f.write("-" * 50 + "\n")
                        for strategy, analysis in contribution_analysis.items():
                            f.write(f"\n{strategy}:\n")
                            f.write(f"  Trades: {analysis.get('total_trades', 0)}\n")
                            f.write(f"  Return: {analysis.get('return_pct', 0):.2f}%\n")
                            f.write(f"  Win Rate: {analysis.get('win_rate', 0):.1f}%\n")
                            f.write(f"  Portfolio Contribution: {analysis.get('contribution_to_portfolio', 0):.1f}%\n")

            logger.info(f"Generated portfolio report: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating portfolio report: {e}")
            return ""

    def _generate_comparative_report(
        self,
        strategy_results: Dict,
        portfolio_results: Dict,
        performance_metrics: Dict,
        timestamp: str
    ) -> str:
        """Generate comparative analysis report."""
        try:
            report_path = self.reports_dir / f'comparative_analysis_{timestamp}.txt'

            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("BACKTESTING COMPARATIVE ANALYSIS REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                comparative_analysis = performance_metrics.get('comparative_analysis', {})

                # Strategy rankings
                rankings = comparative_analysis.get('strategy_rankings', {})
                if rankings:
                    f.write("STRATEGY PERFORMANCE RANKINGS\n")
                    f.write("-" * 50 + "\n")

                    for metric, ranking in rankings.items():
                        f.write(f"\nRanking by {metric.replace('_', ' ').title()}:\n")
                        for i, (strategy, value) in enumerate(ranking, 1):
                            f.write(f"  {i}. {strategy}: {value:.2f}\n")

                # Portfolio vs best strategy
                portfolio_vs_best = comparative_analysis.get('portfolio_vs_best_strategy', {})
                if portfolio_vs_best:
                    f.write(f"\nPORTFOLIO VS BEST STRATEGY\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Best Strategy: {portfolio_vs_best.get('best_strategy_name', 'N/A')}\n")
                    f.write(f"Best Strategy Return: {portfolio_vs_best.get('best_strategy_return', 0):.2f}%\n")
                    f.write(f"Portfolio Return: {portfolio_vs_best.get('portfolio_return', 0):.2f}%\n")
                    f.write(f"Outperformance: {portfolio_vs_best.get('outperformance', 0):.2f}%\n")

                # Risk-return efficiency
                efficiency_analysis = comparative_analysis.get('risk_return_efficiency', {})
                if efficiency_analysis:
                    f.write(f"\nRISK-RETURN EFFICIENCY ANALYSIS\n")
                    f.write("-" * 50 + "\n")
                    for strategy, metrics in efficiency_analysis.items():
                        f.write(f"{strategy}:\n")
                        f.write(f"  Return: {metrics.get('return', 0):.2f}%\n")
                        f.write(f"  Risk (StdDev): {metrics.get('risk', 0):.2f}%\n")
                        f.write(f"  Risk-Adjusted Return: {metrics.get('risk_adjusted_return', 0):.2f}\n\n")

            logger.info(f"Generated comparative analysis report: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating comparative report: {e}")
            return ""

    def _generate_risk_analysis_report(self, performance_metrics: Dict, timestamp: str) -> str:
        """Generate risk analysis report."""
        try:
            report_path = self.reports_dir / f'risk_analysis_{timestamp}.txt'

            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("BACKTESTING RISK ANALYSIS REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                risk_analysis = performance_metrics.get('risk_analysis', {})

                # Strategy risk metrics
                strategy_risks = risk_analysis.get('strategy_risks', {})
                if strategy_risks:
                    f.write("STRATEGY RISK METRICS\n")
                    f.write("-" * 50 + "\n")

                    for strategy, risk_metrics in strategy_risks.items():
                        f.write(f"\n{strategy}:\n")
                        f.write(f"  Volatility: {risk_metrics.get('volatility', 0):.2f}%\n")
                        f.write(f"  VaR (95%): {risk_metrics.get('var_95', 0):.2f}%\n")
                        f.write(f"  CVaR (95%): {risk_metrics.get('cvar_95', 0):.2f}%\n")
                        f.write(f"  Max Consecutive Losses: {risk_metrics.get('max_consecutive_losses', 0)}\n")

                # Portfolio risk metrics
                portfolio_risk = risk_analysis.get('portfolio_risk', {})
                if portfolio_risk:
                    f.write(f"\nPORTFOLIO RISK METRICS\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Portfolio Volatility: {portfolio_risk.get('volatility', 0):.2f}%\n")
                    f.write(f"Portfolio VaR (95%): {portfolio_risk.get('var_95', 0):.2f}%\n")
                    f.write(f"Portfolio CVaR (95%): {portfolio_risk.get('cvar_95', 0):.2f}%\n")

                # Risk concentration
                risk_concentration = risk_analysis.get('risk_concentration', {})
                if risk_concentration:
                    f.write(f"\nRISK CONCENTRATION ANALYSIS\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Most Risky Strategy: {risk_concentration.get('highest_risk_strategy', 'N/A')}\n")
                    f.write(f"Risk Concentration Ratio: {risk_concentration.get('concentration_ratio', 0):.2f}\n")

            logger.info(f"Generated risk analysis report: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating risk analysis report: {e}")
            return ""

    def _export_detailed_data(
        self,
        strategy_results: Dict,
        portfolio_results: Dict,
        timestamp: str
    ) -> Dict[str, str]:
        """Export detailed data to CSV files."""
        try:
            export_paths = {}

            # Export individual strategy trades
            for strategy_name, result in strategy_results.items():
                trades = result.get('trades', [])
                if trades:
                    trades_df = pd.DataFrame(trades)
                    export_path = self.reports_dir / f'{strategy_name}_trades_{timestamp}.csv'
                    trades_df.to_csv(export_path, index=False)
                    export_paths[f'{strategy_name}_trades'] = str(export_path)

            # Export portfolio trades
            if portfolio_results and portfolio_results.get('portfolio_trades'):
                portfolio_trades_df = pd.DataFrame(portfolio_results['portfolio_trades'])
                export_path = self.reports_dir / f'portfolio_trades_{timestamp}.csv'
                portfolio_trades_df.to_csv(export_path, index=False)
                export_paths['portfolio_trades'] = str(export_path)

            # Export strategy performance summary
            performance_summary = self._create_performance_summary_df(strategy_results)
            if not performance_summary.empty:
                export_path = self.reports_dir / f'strategy_performance_summary_{timestamp}.csv'
                performance_summary.to_csv(export_path, index=False)
                export_paths['performance_summary'] = str(export_path)

            logger.info(f"Exported {len(export_paths)} data files")
            return export_paths

        except Exception as e:
            logger.error(f"Error exporting detailed data: {e}")
            return {}

    def _generate_executive_summary(
        self,
        strategy_results: Dict,
        portfolio_results: Dict,
        performance_metrics: Dict,
        timestamp: str
    ) -> str:
        """Generate executive summary report."""
        try:
            report_path = self.reports_dir / f'executive_summary_{timestamp}.txt'

            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("BACKTESTING EXECUTIVE SUMMARY\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Key highlights
                f.write("KEY HIGHLIGHTS\n")
                f.write("-" * 50 + "\n")

                if strategy_results:
                    total_strategies = len(strategy_results)
                    f.write(f"• {total_strategies} trading strategies analyzed\n")

                    # Best strategy
                    best_strategy = self._find_best_strategy(strategy_results)
                    if best_strategy:
                        f.write(f"• Best performing strategy: {best_strategy['name']} "
                               f"({best_strategy['return']:.1f}% return)\n")

                    # Portfolio performance
                    if portfolio_results:
                        portfolio_return = portfolio_results.get('performance', {}).get('total_return', 0)
                        f.write(f"• Portfolio return: {portfolio_return:.1f}%\n")

                        # Diversification benefit
                        if best_strategy:
                            diversification_benefit = portfolio_return - best_strategy['return']
                            if diversification_benefit > 0:
                                f.write(f"• Diversification added {diversification_benefit:.1f}% additional return\n")
                            else:
                                f.write(f"• Single strategy outperformed portfolio by {abs(diversification_benefit):.1f}%\n")

                # Recommendations
                f.write(f"\nRECOMMENDations\n")
                f.write("-" * 50 + "\n")

                recommendations = self._generate_recommendations(strategy_results, portfolio_results, performance_metrics)
                for i, recommendation in enumerate(recommendations, 1):
                    f.write(f"{i}. {recommendation}\n")

                # Risk summary
                f.write(f"\nRISK ASSESSMENT\n")
                f.write("-" * 50 + "\n")

                risk_summary = self._create_risk_summary(performance_metrics)
                for risk_point in risk_summary:
                    f.write(f"• {risk_point}\n")

            logger.info(f"Generated executive summary: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return ""

    def _find_best_strategy(self, strategy_results: Dict) -> Optional[Dict]:
        """Find the best performing strategy."""
        try:
            best_strategy = None
            best_return = float('-inf')

            for strategy_name, result in strategy_results.items():
                performance = result.get('performance', {})
                total_return = performance.get('total_return', 0)

                if total_return > best_return:
                    best_return = total_return
                    best_strategy = {
                        'name': strategy_name,
                        'return': total_return,
                        'performance': performance
                    }

            return best_strategy

        except Exception:
            return None

    def _create_performance_summary_df(self, strategy_results: Dict) -> pd.DataFrame:
        """Create performance summary DataFrame."""
        try:
            summary_data = []

            for strategy_name, result in strategy_results.items():
                performance = result.get('performance', {})

                summary_data.append({
                    'Strategy': strategy_name,
                    'Total Return (%)': performance.get('total_return', 0),
                    'Total Trades': performance.get('total_trades', 0),
                    'Win Rate (%)': performance.get('win_rate', 0),
                    'Profit Factor': performance.get('profit_factor', 0),
                    'Max Drawdown (%)': performance.get('max_drawdown', 0),
                    'Avg Holding Days': performance.get('avg_holding_days', 0),
                    'Sharpe Ratio': performance.get('sharpe_ratio', 0)
                })

            return pd.DataFrame(summary_data)

        except Exception as e:
            logger.error(f"Error creating performance summary DataFrame: {e}")
            return pd.DataFrame()

    def _generate_recommendations(
        self,
        strategy_results: Dict,
        portfolio_results: Dict,
        performance_metrics: Dict
    ) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []

        try:
            # Best strategy recommendation
            best_strategy = self._find_best_strategy(strategy_results)
            if best_strategy:
                recommendations.append(
                    f"Focus on {best_strategy['name']} strategy which delivered "
                    f"{best_strategy['return']:.1f}% return"
                )

            # Portfolio diversification
            if portfolio_results and best_strategy:
                portfolio_return = portfolio_results.get('performance', {}).get('total_return', 0)
                if portfolio_return > best_strategy['return']:
                    recommendations.append(
                        "Portfolio diversification improved returns - maintain multi-strategy approach"
                    )

            # Risk management
            high_risk_strategies = []
            for strategy_name, result in strategy_results.items():
                max_drawdown = result.get('performance', {}).get('max_drawdown', 0)
                if max_drawdown < -20:  # More than 20% drawdown
                    high_risk_strategies.append(strategy_name)

            if high_risk_strategies:
                recommendations.append(
                    f"Review risk management for high-drawdown strategies: {', '.join(high_risk_strategies)}"
                )

            # Statistical significance
            statistical_analysis = performance_metrics.get('statistical_analysis', {})
            significance_tests = statistical_analysis.get('significance_tests', {})

            insignificant_strategies = [
                strategy for strategy, test in significance_tests.items()
                if not test.get('is_significant_5pct', False)
            ]

            if insignificant_strategies:
                recommendations.append(
                    f"Results for {', '.join(insignificant_strategies)} may not be statistically significant"
                )

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Conduct further analysis to validate results")

        return recommendations

    def _create_risk_summary(self, performance_metrics: Dict) -> List[str]:
        """Create risk assessment summary."""
        risk_points = []

        try:
            risk_analysis = performance_metrics.get('risk_analysis', {})

            # Portfolio risk
            portfolio_risk = risk_analysis.get('portfolio_risk', {})
            if portfolio_risk:
                var_95 = portfolio_risk.get('var_95', 0)
                if abs(var_95) > 5:  # More than 5% VaR
                    risk_points.append(f"High portfolio risk: 95% VaR is {var_95:.1f}%")

            # Strategy concentration
            risk_concentration = risk_analysis.get('risk_concentration', {})
            if risk_concentration:
                concentration_ratio = risk_concentration.get('concentration_ratio', 0)
                if concentration_ratio > 0.7:  # More than 70% concentration
                    risk_points.append("High risk concentration - consider further diversification")

            if not risk_points:
                risk_points.append("Risk levels appear manageable based on current analysis")

        except Exception as e:
            logger.error(f"Error creating risk summary: {e}")
            risk_points.append("Risk assessment requires further analysis")

        return risk_points