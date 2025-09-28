"""
Portfolio Backtester - Multi-Strategy Portfolio Analysis
========================================================

Simulates portfolio-level backtesting combining multiple screener strategies.
Analyzes diversification benefits and strategy correlation effects.

Key Features:
- Multi-strategy portfolio simulation
- Strategy allocation and rebalancing
- Correlation analysis between strategies
- Portfolio-level risk metrics
- Strategy contribution analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """
    Backtests multi-strategy portfolios.

    Combines multiple screener strategies into a diversified portfolio
    and analyzes overall performance and risk characteristics.
    """

    def __init__(self, config, user_config):
        """
        Initialize PortfolioBacktester.

        Args:
            config: System configuration object
            user_config: User configuration with portfolio parameters
        """
        self.config = config
        self.user_config = user_config

        # Portfolio parameters
        self.initial_capital = getattr(user_config, 'bt_initial_capital', 100000)
        self.rebalance_frequency = getattr(user_config, 'bt_rebalance_frequency', 'monthly')
        self.max_portfolio_positions = getattr(user_config, 'bt_max_portfolio_positions', 20)
        self.strategy_allocation = getattr(user_config, 'bt_strategy_allocation', 'equal_weight')

    def backtest_portfolio(self, signals_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Backtest portfolio combining multiple strategies.

        Args:
            signals_data: Dict mapping strategy names to signal DataFrames

        Returns:
            Portfolio backtesting results
        """
        try:
            logger.info("Starting portfolio backtesting")

            if not signals_data:
                logger.warning("No strategies provided for portfolio backtesting")
                return self._create_empty_portfolio_result()

            # Combine and prepare all signals
            combined_signals = self._combine_strategy_signals(signals_data)

            if combined_signals.empty:
                logger.warning("No valid signals for portfolio backtesting")
                return self._create_empty_portfolio_result()

            # Simulate portfolio trades
            portfolio_trades = self._simulate_portfolio_trades(combined_signals, signals_data)

            # Calculate portfolio performance
            portfolio_performance = self._calculate_portfolio_performance(portfolio_trades)

            # Analyze strategy contributions
            strategy_analysis = self._analyze_strategy_contributions(portfolio_trades, signals_data)

            # Calculate correlation analysis
            correlation_analysis = self._calculate_strategy_correlations(signals_data)

            result = {
                'backtest_period': self._get_portfolio_backtest_period(combined_signals),
                'strategies_included': list(signals_data.keys()),
                'total_portfolio_trades': len(portfolio_trades),
                'portfolio_trades': portfolio_trades,
                'performance': portfolio_performance,
                'strategy_analysis': strategy_analysis,
                'correlation_analysis': correlation_analysis,
                'allocation_method': self.strategy_allocation
            }

            logger.info(f"Portfolio backtest completed: "
                       f"{len(portfolio_trades)} trades, "
                       f"{portfolio_performance.get('total_return', 0):.2f}% return")

            return result

        except Exception as e:
            logger.error(f"Error in portfolio backtesting: {e}")
            return self._create_empty_portfolio_result(error=str(e))

    def _combine_strategy_signals(self, signals_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine signals from all strategies into unified DataFrame.

        Args:
            signals_data: Dict mapping strategy names to signal DataFrames

        Returns:
            Combined signals DataFrame with strategy identification
        """
        try:
            all_signals = []

            for strategy_name, signals_df in signals_data.items():
                if not signals_df.empty:
                    # Add strategy identification
                    signals_copy = signals_df.copy()
                    signals_copy['source_strategy'] = strategy_name
                    all_signals.append(signals_copy)

            if all_signals:
                combined = pd.concat(all_signals, ignore_index=True)
                combined = combined.sort_values('signal_date').reset_index(drop=True)
                logger.info(f"Combined {len(combined)} signals from {len(all_signals)} strategies")
                return combined
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error combining strategy signals: {e}")
            return pd.DataFrame()

    def _simulate_portfolio_trades(
        self,
        combined_signals: pd.DataFrame,
        signals_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        Simulate portfolio trades with multi-strategy allocation.

        Args:
            combined_signals: Combined signals from all strategies
            signals_data: Original strategy signals for allocation

        Returns:
            List of portfolio trade dictionaries
        """
        try:
            portfolio_trades = []
            positions = {}  # {ticker: position_info}
            strategy_allocations = self._calculate_strategy_allocations(signals_data)

            current_capital = self.initial_capital
            capital_per_strategy = {
                strategy: self.initial_capital * allocation
                for strategy, allocation in strategy_allocations.items()
            }

            for _, signal in combined_signals.iterrows():
                ticker = signal['ticker']
                signal_type = signal['signal_type']
                signal_date = signal['signal_date']
                signal_price = signal['signal_price']
                source_strategy = signal['source_strategy']

                # Calculate position size based on strategy allocation
                strategy_capital = capital_per_strategy.get(source_strategy, 0)
                max_position_size = strategy_capital * 0.1  # 10% of strategy capital per position

                if signal_type == 'Buy':
                    # Open new position
                    position_key = f"{ticker}_{source_strategy}"

                    if (position_key not in positions and
                        len(positions) < self.max_portfolio_positions and
                        strategy_capital > 0):

                        shares = int(max_position_size / signal_price)
                        position_value = shares * signal_price

                        if shares > 0 and position_value <= strategy_capital:
                            positions[position_key] = {
                                'ticker': ticker,
                                'strategy': source_strategy,
                                'shares': shares,
                                'entry_price': signal_price,
                                'entry_date': signal_date,
                                'entry_value': position_value
                            }
                            capital_per_strategy[source_strategy] -= position_value

                elif signal_type in ['Sell', 'Close Buy', 'Close Sell']:
                    # Close existing position
                    position_key = f"{ticker}_{source_strategy}"

                    if position_key in positions:
                        position = positions[position_key]
                        trade = self._create_portfolio_trade(
                            position, signal_price, signal_date, 'signal_close'
                        )
                        portfolio_trades.append(trade)

                        # Return capital to strategy
                        exit_value = trade['exit_value']
                        capital_per_strategy[source_strategy] += exit_value

                        del positions[position_key]

            # Close remaining positions
            if positions:
                last_signal = combined_signals.iloc[-1]
                for position_key, position in positions.items():
                    # Use entry price as approximation for closing
                    closing_price = position['entry_price']
                    trade = self._create_portfolio_trade(
                        position, closing_price, last_signal['signal_date'], 'forced_close'
                    )
                    portfolio_trades.append(trade)

            logger.info(f"Simulated {len(portfolio_trades)} portfolio trades")
            return portfolio_trades

        except Exception as e:
            logger.error(f"Error simulating portfolio trades: {e}")
            return []

    def _calculate_strategy_allocations(self, signals_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate allocation weights for each strategy.

        Args:
            signals_data: Strategy signals data

        Returns:
            Dict mapping strategy names to allocation weights
        """
        try:
            if self.strategy_allocation == 'equal_weight':
                # Equal weight allocation
                num_strategies = len(signals_data)
                return {strategy: 1.0 / num_strategies for strategy in signals_data.keys()}

            elif self.strategy_allocation == 'signal_count_weighted':
                # Weight by number of signals
                signal_counts = {
                    strategy: len(signals_df)
                    for strategy, signals_df in signals_data.items()
                }
                total_signals = sum(signal_counts.values())

                if total_signals > 0:
                    return {
                        strategy: count / total_signals
                        for strategy, count in signal_counts.items()
                    }
                else:
                    # Fallback to equal weight
                    num_strategies = len(signals_data)
                    return {strategy: 1.0 / num_strategies for strategy in signals_data.keys()}

            else:
                # Default to equal weight
                num_strategies = len(signals_data)
                return {strategy: 1.0 / num_strategies for strategy in signals_data.keys()}

        except Exception as e:
            logger.error(f"Error calculating strategy allocations: {e}")
            # Fallback to equal weight
            num_strategies = len(signals_data)
            return {strategy: 1.0 / num_strategies for strategy in signals_data.keys()}

    def _create_portfolio_trade(
        self,
        position: Dict,
        exit_price: float,
        exit_date: datetime,
        exit_reason: str
    ) -> Dict:
        """Create portfolio trade record."""
        shares = position['shares']
        entry_price = position['entry_price']
        entry_date = position['entry_date']

        exit_value = shares * exit_price
        entry_value = shares * entry_price
        pnl = exit_value - entry_value
        pnl_pct = (pnl / entry_value) * 100
        holding_days = (exit_date - entry_date).days

        return {
            'ticker': position['ticker'],
            'strategy': position['strategy'],
            'shares': shares,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_value': entry_value,
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_days': holding_days,
            'exit_reason': exit_reason
        }

    def _calculate_portfolio_performance(self, portfolio_trades: List[Dict]) -> Dict:
        """Calculate portfolio-level performance metrics."""
        try:
            if not portfolio_trades:
                return self._create_empty_portfolio_performance()

            trades_df = pd.DataFrame(portfolio_trades)

            # Basic metrics
            total_pnl = trades_df['pnl'].sum()
            total_return = (total_pnl / self.initial_capital) * 100
            total_trades = len(trades_df)

            # Win/Loss metrics
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]

            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

            # Risk metrics
            returns = trades_df['pnl_pct'] / 100
            sharpe_ratio = self._calculate_portfolio_sharpe(returns)
            max_drawdown = self._calculate_portfolio_drawdown(trades_df)

            # Portfolio-specific metrics
            strategy_contributions = trades_df.groupby('strategy')['pnl'].sum()
            diversification_ratio = self._calculate_diversification_ratio(trades_df)

            performance = {
                'total_return': total_return,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'diversification_ratio': diversification_ratio,
                'strategy_contributions': strategy_contributions.to_dict()
            }

            return performance

        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return self._create_empty_portfolio_performance()

    def _analyze_strategy_contributions(
        self,
        portfolio_trades: List[Dict],
        signals_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Analyze individual strategy contributions to portfolio."""
        try:
            if not portfolio_trades:
                return {}

            trades_df = pd.DataFrame(portfolio_trades)

            # Strategy-level analysis
            strategy_analysis = {}

            for strategy in signals_data.keys():
                strategy_trades = trades_df[trades_df['strategy'] == strategy]

                if not strategy_trades.empty:
                    strategy_pnl = strategy_trades['pnl'].sum()
                    strategy_return = (strategy_pnl / self.initial_capital) * 100
                    strategy_win_rate = (strategy_trades['pnl'] > 0).mean() * 100

                    strategy_analysis[strategy] = {
                        'total_trades': len(strategy_trades),
                        'total_pnl': strategy_pnl,
                        'return_pct': strategy_return,
                        'win_rate': strategy_win_rate,
                        'contribution_to_portfolio': strategy_pnl / trades_df['pnl'].sum() * 100 if trades_df['pnl'].sum() != 0 else 0
                    }

            return strategy_analysis

        except Exception as e:
            logger.error(f"Error analyzing strategy contributions: {e}")
            return {}

    def _calculate_strategy_correlations(self, signals_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate correlations between strategies."""
        try:
            # This is a simplified correlation analysis
            # In practice, you'd need aligned time series data
            correlation_matrix = {}

            strategies = list(signals_data.keys())
            for i, strategy1 in enumerate(strategies):
                for j, strategy2 in enumerate(strategies):
                    if i <= j:  # Only calculate upper triangle
                        # Simplified correlation based on signal timing overlap
                        overlap_score = self._calculate_signal_overlap(
                            signals_data[strategy1], signals_data[strategy2]
                        )
                        correlation_matrix[f"{strategy1}_{strategy2}"] = overlap_score

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error calculating strategy correlations: {e}")
            return {}

    def _calculate_signal_overlap(self, signals1: pd.DataFrame, signals2: pd.DataFrame) -> float:
        """Calculate overlap score between two strategy signals."""
        try:
            if signals1.empty or signals2.empty:
                return 0.0

            # Get unique tickers and dates for each strategy
            tickers1 = set(signals1['ticker'].unique())
            tickers2 = set(signals2['ticker'].unique())

            # Calculate ticker overlap
            ticker_overlap = len(tickers1.intersection(tickers2)) / len(tickers1.union(tickers2))

            return ticker_overlap

        except Exception:
            return 0.0

    def _calculate_portfolio_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate portfolio Sharpe ratio."""
        try:
            if len(returns) == 0 or returns.std() == 0:
                return 0

            excess_returns = returns.mean() - (risk_free_rate / 252)
            return excess_returns / returns.std() * np.sqrt(252)

        except Exception:
            return 0

    def _calculate_portfolio_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate portfolio maximum drawdown."""
        try:
            if trades_df.empty:
                return 0

            cumulative_pnl = trades_df['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / self.initial_capital * 100

            return drawdown.min()

        except Exception:
            return 0

    def _calculate_diversification_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate diversification ratio."""
        try:
            if trades_df.empty:
                return 0

            # Simple diversification measure: number of strategies used
            unique_strategies = trades_df['strategy'].nunique()
            unique_tickers = trades_df['ticker'].nunique()

            # Normalize by maximum possible diversification
            max_possible = len(trades_df)
            actual_diversification = unique_strategies * unique_tickers

            return actual_diversification / max_possible if max_possible > 0 else 0

        except Exception:
            return 0

    def _get_portfolio_backtest_period(self, combined_signals: pd.DataFrame) -> Dict:
        """Get portfolio backtest period information."""
        try:
            if combined_signals.empty:
                return {}

            start_date = combined_signals['signal_date'].min()
            end_date = combined_signals['signal_date'].max()
            duration_days = (end_date - start_date).days

            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'duration_days': duration_days
            }

        except Exception:
            return {}

    def _create_empty_portfolio_result(self, error: str = None) -> Dict:
        """Create empty portfolio result."""
        return {
            'backtest_period': {},
            'strategies_included': [],
            'total_portfolio_trades': 0,
            'portfolio_trades': [],
            'performance': self._create_empty_portfolio_performance(),
            'strategy_analysis': {},
            'correlation_analysis': {},
            'allocation_method': self.strategy_allocation,
            'error': error
        }

    def _create_empty_portfolio_performance(self) -> Dict:
        """Create empty portfolio performance metrics."""
        return {
            'total_return': 0,
            'total_pnl': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'diversification_ratio': 0,
            'strategy_contributions': {}
        }