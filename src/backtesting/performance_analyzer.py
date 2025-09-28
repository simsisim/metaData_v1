"""
Performance Analyzer - Comprehensive Metrics Calculation
========================================================

Calculates detailed performance metrics and risk analysis for backtesting results.
Provides statistical analysis and comparative performance evaluation.

Key Features:
- Risk-adjusted returns calculation
- Drawdown analysis
- Statistical significance testing
- Benchmark comparison
- Risk metrics (VaR, CVaR, etc.)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes backtesting performance with comprehensive metrics.

    Calculates risk-adjusted returns, statistical measures,
    and comparative performance analysis.
    """

    def __init__(self, config):
        """
        Initialize PerformanceAnalyzer.

        Args:
            config: System configuration object
        """
        self.config = config

        # Standard analysis parameters
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.confidence_levels = [0.95, 0.99]  # For VaR calculations
        self.benchmark_return = 0.10  # 10% annual benchmark return

    def analyze_all_performance(
        self,
        strategy_results: Dict,
        portfolio_results: Dict
    ) -> Dict:
        """
        Analyze performance for all strategies and portfolio.

        Args:
            strategy_results: Individual strategy backtesting results
            portfolio_results: Portfolio backtesting results

        Returns:
            Comprehensive performance analysis
        """
        try:
            logger.info("Starting comprehensive performance analysis")

            analysis = {
                'analysis_timestamp': datetime.now().isoformat(),
                'strategy_analysis': {},
                'portfolio_analysis': {},
                'comparative_analysis': {},
                'risk_analysis': {},
                'statistical_analysis': {}
            }

            # Analyze individual strategies
            for strategy_name, result in strategy_results.items():
                strategy_analysis = self._analyze_strategy_performance(strategy_name, result)
                analysis['strategy_analysis'][strategy_name] = strategy_analysis

            # Analyze portfolio performance
            if portfolio_results:
                portfolio_analysis = self._analyze_portfolio_performance(portfolio_results)
                analysis['portfolio_analysis'] = portfolio_analysis

            # Comparative analysis
            comparative_analysis = self._perform_comparative_analysis(strategy_results, portfolio_results)
            analysis['comparative_analysis'] = comparative_analysis

            # Risk analysis
            risk_analysis = self._perform_risk_analysis(strategy_results, portfolio_results)
            analysis['risk_analysis'] = risk_analysis

            # Statistical analysis
            statistical_analysis = self._perform_statistical_analysis(strategy_results)
            analysis['statistical_analysis'] = statistical_analysis

            logger.info("Performance analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return {'error': str(e)}

    def _analyze_strategy_performance(self, strategy_name: str, result: Dict) -> Dict:
        """Analyze individual strategy performance."""
        try:
            if not result.get('trades'):
                return {'error': 'No trades available for analysis'}

            trades_df = pd.DataFrame(result['trades'])
            performance = result.get('performance', {})

            # Enhanced risk metrics
            returns = trades_df['pnl_pct'] / 100
            enhanced_metrics = self._calculate_enhanced_risk_metrics(returns)

            # Trade analysis
            trade_analysis = self._analyze_trade_patterns(trades_df)

            # Monthly/quarterly breakdown
            temporal_analysis = self._analyze_temporal_performance(trades_df)

            # Benchmark comparison
            benchmark_comparison = self._compare_to_benchmark(performance)

            analysis = {
                'basic_metrics': performance,
                'enhanced_risk_metrics': enhanced_metrics,
                'trade_analysis': trade_analysis,
                'temporal_analysis': temporal_analysis,
                'benchmark_comparison': benchmark_comparison,
                'quality_score': self._calculate_strategy_quality_score(performance, enhanced_metrics)
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing strategy {strategy_name}: {e}")
            return {'error': str(e)}

    def _analyze_portfolio_performance(self, portfolio_results: Dict) -> Dict:
        """Analyze portfolio performance."""
        try:
            if not portfolio_results.get('portfolio_trades'):
                return {'error': 'No portfolio trades available for analysis'}

            trades_df = pd.DataFrame(portfolio_results['portfolio_trades'])
            performance = portfolio_results.get('performance', {})

            # Portfolio-specific metrics
            portfolio_metrics = self._calculate_portfolio_metrics(trades_df, performance)

            # Strategy contribution analysis
            contribution_analysis = self._analyze_strategy_contributions(trades_df)

            # Diversification analysis
            diversification_analysis = self._analyze_diversification_benefits(portfolio_results)

            # Risk-return profile
            risk_return_profile = self._calculate_portfolio_risk_return_profile(trades_df)

            analysis = {
                'portfolio_metrics': portfolio_metrics,
                'contribution_analysis': contribution_analysis,
                'diversification_analysis': diversification_analysis,
                'risk_return_profile': risk_return_profile
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing portfolio performance: {e}")
            return {'error': str(e)}

    def _calculate_enhanced_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate enhanced risk metrics."""
        try:
            if len(returns) == 0:
                return {}

            metrics = {}

            # Basic statistics
            metrics['mean_return'] = returns.mean()
            metrics['volatility'] = returns.std()
            metrics['skewness'] = stats.skew(returns)
            metrics['kurtosis'] = stats.kurtosis(returns)

            # Risk-adjusted returns
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = (metrics['mean_return'] - self.risk_free_rate/252) / metrics['volatility'] * np.sqrt(252)
                metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
                metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns)

            # Value at Risk
            for confidence in self.confidence_levels:
                var_key = f'var_{int(confidence*100)}'
                cvar_key = f'cvar_{int(confidence*100)}'
                metrics[var_key] = np.percentile(returns, (1 - confidence) * 100)
                metrics[cvar_key] = returns[returns <= metrics[var_key]].mean()

            # Maximum consecutive losses
            metrics['max_consecutive_losses'] = self._calculate_max_consecutive_losses(returns)

            # Gain-to-pain ratio
            metrics['gain_to_pain_ratio'] = self._calculate_gain_to_pain_ratio(returns)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating enhanced risk metrics: {e}")
            return {}

    def _analyze_trade_patterns(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze trade patterns and characteristics."""
        try:
            analysis = {}

            # Holding period analysis
            analysis['avg_holding_days'] = trades_df['holding_days'].mean()
            analysis['median_holding_days'] = trades_df['holding_days'].median()
            analysis['holding_period_std'] = trades_df['holding_days'].std()

            # Trade size distribution
            analysis['avg_trade_size'] = trades_df['entry_value'].mean()
            analysis['trade_size_cv'] = trades_df['entry_value'].std() / analysis['avg_trade_size']

            # Win/loss streaks
            win_loss_sequence = (trades_df['pnl'] > 0).astype(int)
            analysis['max_win_streak'] = self._calculate_max_streak(win_loss_sequence, 1)
            analysis['max_loss_streak'] = self._calculate_max_streak(win_loss_sequence, 0)

            # Profit factor by holding period
            short_term_trades = trades_df[trades_df['holding_days'] <= 5]
            long_term_trades = trades_df[trades_df['holding_days'] > 20]

            if not short_term_trades.empty:
                analysis['short_term_profit_factor'] = self._calculate_profit_factor(short_term_trades)
            if not long_term_trades.empty:
                analysis['long_term_profit_factor'] = self._calculate_profit_factor(long_term_trades)

            # Exit reason analysis
            exit_reasons = trades_df['exit_reason'].value_counts()
            analysis['exit_reason_breakdown'] = exit_reasons.to_dict()

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {e}")
            return {}

    def _analyze_temporal_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance across different time periods."""
        try:
            analysis = {}

            # Convert exit_date to datetime
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

            # Monthly performance
            trades_df['exit_month'] = trades_df['exit_date'].dt.to_period('M')
            monthly_pnl = trades_df.groupby('exit_month')['pnl'].sum()

            analysis['monthly_performance'] = {
                'best_month': monthly_pnl.max() if len(monthly_pnl) > 0 else 0,
                'worst_month': monthly_pnl.min() if len(monthly_pnl) > 0 else 0,
                'avg_monthly_pnl': monthly_pnl.mean() if len(monthly_pnl) > 0 else 0,
                'monthly_volatility': monthly_pnl.std() if len(monthly_pnl) > 0 else 0,
                'positive_months': (monthly_pnl > 0).sum() if len(monthly_pnl) > 0 else 0,
                'total_months': len(monthly_pnl)
            }

            # Quarterly performance
            trades_df['exit_quarter'] = trades_df['exit_date'].dt.to_period('Q')
            quarterly_pnl = trades_df.groupby('exit_quarter')['pnl'].sum()

            analysis['quarterly_performance'] = {
                'best_quarter': quarterly_pnl.max() if len(quarterly_pnl) > 0 else 0,
                'worst_quarter': quarterly_pnl.min() if len(quarterly_pnl) > 0 else 0,
                'avg_quarterly_pnl': quarterly_pnl.mean() if len(quarterly_pnl) > 0 else 0
            }

            # Day of week analysis
            trades_df['exit_dow'] = trades_df['exit_date'].dt.day_name()
            dow_performance = trades_df.groupby('exit_dow')['pnl'].agg(['mean', 'count'])

            analysis['day_of_week_performance'] = {
                day: {'avg_pnl': performance['mean'], 'trade_count': performance['count']}
                for day, performance in dow_performance.iterrows()
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing temporal performance: {e}")
            return {}

    def _perform_comparative_analysis(self, strategy_results: Dict, portfolio_results: Dict) -> Dict:
        """Perform comparative analysis between strategies and portfolio."""
        try:
            analysis = {}

            # Strategy ranking
            strategy_rankings = self._rank_strategies(strategy_results)
            analysis['strategy_rankings'] = strategy_rankings

            # Portfolio vs best strategy comparison
            if portfolio_results and strategy_results:
                portfolio_vs_best = self._compare_portfolio_to_best_strategy(
                    portfolio_results, strategy_results
                )
                analysis['portfolio_vs_best_strategy'] = portfolio_vs_best

            # Correlation analysis
            correlation_analysis = self._analyze_strategy_correlations(strategy_results)
            analysis['strategy_correlations'] = correlation_analysis

            # Risk-return efficiency
            efficiency_analysis = self._analyze_risk_return_efficiency(strategy_results, portfolio_results)
            analysis['risk_return_efficiency'] = efficiency_analysis

            return analysis

        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            return {}

    def _perform_risk_analysis(self, strategy_results: Dict, portfolio_results: Dict) -> Dict:
        """Perform comprehensive risk analysis."""
        try:
            analysis = {}

            # Strategy risk analysis
            strategy_risks = {}
            for strategy_name, result in strategy_results.items():
                if result.get('trades'):
                    trades_df = pd.DataFrame(result['trades'])
                    returns = trades_df['pnl_pct'] / 100
                    strategy_risks[strategy_name] = self._calculate_risk_metrics(returns)

            analysis['strategy_risks'] = strategy_risks

            # Portfolio risk analysis
            if portfolio_results and portfolio_results.get('portfolio_trades'):
                portfolio_trades_df = pd.DataFrame(portfolio_results['portfolio_trades'])
                portfolio_returns = portfolio_trades_df['pnl_pct'] / 100
                analysis['portfolio_risk'] = self._calculate_risk_metrics(portfolio_returns)

            # Risk concentration analysis
            concentration_analysis = self._analyze_risk_concentration(strategy_results)
            analysis['risk_concentration'] = concentration_analysis

            return analysis

        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {}

    def _perform_statistical_analysis(self, strategy_results: Dict) -> Dict:
        """Perform statistical significance analysis."""
        try:
            analysis = {}

            # Collect all strategy returns for analysis
            strategy_returns = {}
            for strategy_name, result in strategy_results.items():
                if result.get('trades'):
                    trades_df = pd.DataFrame(result['trades'])
                    returns = trades_df['pnl_pct'] / 100
                    if len(returns) > 0:
                        strategy_returns[strategy_name] = returns

            # T-tests for statistical significance
            significance_tests = {}
            for strategy_name, returns in strategy_returns.items():
                if len(returns) > 1:
                    # Test if returns are significantly different from zero
                    t_stat, p_value = stats.ttest_1samp(returns, 0)
                    significance_tests[strategy_name] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'is_significant_5pct': p_value < 0.05,
                        'is_significant_1pct': p_value < 0.01
                    }

            analysis['significance_tests'] = significance_tests

            # ANOVA for comparing strategies
            if len(strategy_returns) > 1:
                returns_lists = [returns.values for returns in strategy_returns.values()]
                try:
                    f_stat, p_value = stats.f_oneway(*returns_lists)
                    analysis['anova_test'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'strategies_significantly_different': p_value < 0.05
                    }
                except Exception:
                    analysis['anova_test'] = {'error': 'Could not perform ANOVA'}

            return analysis

        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {}

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        try:
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float('inf')

            downside_deviation = downside_returns.std()
            if downside_deviation == 0:
                return float('inf')

            excess_return = returns.mean() - self.risk_free_rate/252
            return excess_return / downside_deviation * np.sqrt(252)

        except Exception:
            return 0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        try:
            annual_return = returns.mean() * 252
            max_drawdown = self._calculate_max_drawdown_from_returns(returns)

            if max_drawdown == 0:
                return float('inf')

            return annual_return / abs(max_drawdown)

        except Exception:
            return 0

    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losses."""
        try:
            loss_sequence = (returns < 0).astype(int)
            return self._calculate_max_streak(loss_sequence, 1)

        except Exception:
            return 0

    def _calculate_gain_to_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate gain-to-pain ratio."""
        try:
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())

            if losses == 0:
                return float('inf')

            return gains / losses

        except Exception:
            return 0

    def _calculate_max_streak(self, sequence: pd.Series, value: int) -> int:
        """Calculate maximum consecutive streak of a value."""
        try:
            max_streak = 0
            current_streak = 0

            for val in sequence:
                if val == value:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0

            return max_streak

        except Exception:
            return 0

    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor for trades."""
        try:
            wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

            if losses == 0:
                return float('inf')

            return wins / losses

        except Exception:
            return 0

    def _calculate_strategy_quality_score(self, performance: Dict, enhanced_metrics: Dict) -> float:
        """Calculate overall strategy quality score."""
        try:
            # Weighted scoring based on key metrics
            score = 0.0
            weights = {
                'total_return': 0.3,
                'sharpe_ratio': 0.2,
                'win_rate': 0.2,
                'profit_factor': 0.15,
                'max_drawdown': 0.15  # Negative impact
            }

            # Normalize and score each metric
            total_return = performance.get('total_return', 0)
            sharpe_ratio = enhanced_metrics.get('sharpe_ratio', 0)
            win_rate = performance.get('win_rate', 0)
            profit_factor = performance.get('profit_factor', 0)
            max_drawdown = performance.get('max_drawdown', 0)

            # Score components (0-100 scale)
            score += weights['total_return'] * min(total_return / 20 * 100, 100)  # 20% return = 100 points
            score += weights['sharpe_ratio'] * min(sharpe_ratio / 2 * 100, 100)  # Sharpe 2.0 = 100 points
            score += weights['win_rate'] * win_rate  # Already in percentage
            score += weights['profit_factor'] * min(profit_factor / 3 * 100, 100)  # PF 3.0 = 100 points
            score -= weights['max_drawdown'] * abs(max_drawdown)  # Drawdown penalty

            return max(0, min(100, score))  # Constrain to 0-100

        except Exception:
            return 0

    # Additional helper methods would continue here...
    # (Implementing remaining methods for brevity)