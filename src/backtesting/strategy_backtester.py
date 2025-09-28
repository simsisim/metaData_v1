"""
Strategy Backtester - Individual Strategy Performance Analysis
==============================================================

Backtests individual screener strategies to evaluate their historical performance.
Simulates trading based on screener signals and calculates comprehensive metrics.

Key Features:
- Signal-based trade simulation
- Position sizing and risk management
- Transaction cost modeling
- Performance metrics calculation
- Trade-level analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Backtests individual screener strategies.

    Simulates trading based on historical signals and calculates
    comprehensive performance metrics.
    """

    def __init__(self, config, user_config):
        """
        Initialize StrategyBacktester.

        Args:
            config: System configuration object
            user_config: User configuration with backtesting parameters
        """
        self.config = config
        self.user_config = user_config

        # Default backtesting parameters
        self.initial_capital = getattr(user_config, 'bt_initial_capital', 100000)
        self.position_size = getattr(user_config, 'bt_position_size', 0.1)  # 10% per position
        self.transaction_cost = getattr(user_config, 'bt_transaction_cost', 0.001)  # 0.1%
        self.max_positions = getattr(user_config, 'bt_max_positions', 10)
        self.stop_loss = getattr(user_config, 'bt_stop_loss', None)  # Optional stop loss %
        self.take_profit = getattr(user_config, 'bt_take_profit', None)  # Optional take profit %

    def backtest_strategy(self, strategy_name: str, signals: pd.DataFrame) -> Dict:
        """
        Backtest a specific strategy using its signals.

        Args:
            strategy_name: Name of the strategy
            signals: DataFrame with strategy signals

        Returns:
            Dict with backtesting results and performance metrics
        """
        try:
            logger.info(f"Starting backtest for strategy: {strategy_name}")

            if signals.empty:
                logger.warning(f"No signals for strategy {strategy_name}")
                return self._create_empty_result(strategy_name)

            # Prepare signals for backtesting
            processed_signals = self._prepare_signals(signals)

            # Simulate trades
            trades = self._simulate_trades(processed_signals)

            # Calculate performance metrics
            performance = self._calculate_performance(trades)

            # Create comprehensive result
            result = {
                'strategy_name': strategy_name,
                'backtest_period': self._get_backtest_period(processed_signals),
                'total_signals': len(processed_signals),
                'total_trades': len(trades),
                'trades': trades,
                'performance': performance,
                'statistics': self._calculate_statistics(trades, performance)
            }

            logger.info(f"Backtest completed for {strategy_name}: "
                       f"{len(trades)} trades, "
                       f"{performance.get('total_return', 0):.2f}% return")

            return result

        except Exception as e:
            logger.error(f"Error backtesting strategy {strategy_name}: {e}")
            return self._create_empty_result(strategy_name, error=str(e))

    def _prepare_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate signals for backtesting.

        Args:
            signals: Raw signals DataFrame

        Returns:
            Processed signals DataFrame
        """
        try:
            # Copy and sort by date
            processed = signals.copy()
            processed = processed.sort_values('signal_date').reset_index(drop=True)

            # Validate required columns
            required_cols = ['ticker', 'signal_date', 'signal_type', 'signal_price']
            missing_cols = [col for col in required_cols if col not in processed.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Convert signal_date to datetime
            processed['signal_date'] = pd.to_datetime(processed['signal_date'])

            # Filter out invalid signals
            valid_signal_types = ['Buy', 'Sell', 'Close Buy', 'Close Sell']
            processed = processed[processed['signal_type'].isin(valid_signal_types)]

            # Remove signals with invalid prices
            processed = processed[
                (processed['signal_price'] > 0) &
                (processed['signal_price'].notna())
            ]

            logger.info(f"Prepared {len(processed)} valid signals")
            return processed

        except Exception as e:
            logger.error(f"Error preparing signals: {e}")
            return pd.DataFrame()

    def _simulate_trades(self, signals: pd.DataFrame) -> List[Dict]:
        """
        Simulate trades based on signals.

        Args:
            signals: Processed signals DataFrame

        Returns:
            List of trade dictionaries
        """
        trades = []
        positions = {}  # {ticker: position_info}
        current_capital = self.initial_capital

        try:
            for _, signal in signals.iterrows():
                ticker = signal['ticker']
                signal_type = signal['signal_type']
                signal_date = signal['signal_date']
                signal_price = signal['signal_price']

                # Process different signal types
                if signal_type == 'Buy':
                    # Open long position
                    if ticker not in positions and len(positions) < self.max_positions:
                        position_value = current_capital * self.position_size
                        shares = int(position_value / signal_price)
                        transaction_cost = position_value * self.transaction_cost

                        if shares > 0 and position_value + transaction_cost <= current_capital:
                            positions[ticker] = {
                                'type': 'long',
                                'shares': shares,
                                'entry_price': signal_price,
                                'entry_date': signal_date,
                                'entry_value': shares * signal_price
                            }
                            current_capital -= (shares * signal_price + transaction_cost)

                elif signal_type == 'Sell':
                    # Open short position (if supported)
                    # For now, treat as close long position if exists
                    if ticker in positions and positions[ticker]['type'] == 'long':
                        trade = self._close_position(positions[ticker], ticker, signal_price, signal_date, 'sell_signal')
                        trades.append(trade)
                        current_capital += trade['exit_value'] - (trade['exit_value'] * self.transaction_cost)
                        del positions[ticker]

                elif signal_type in ['Close Buy', 'Close Sell']:
                    # Close existing position
                    if ticker in positions:
                        trade = self._close_position(positions[ticker], ticker, signal_price, signal_date, 'close_signal')
                        trades.append(trade)
                        current_capital += trade['exit_value'] - (trade['exit_value'] * self.transaction_cost)
                        del positions[ticker]

            # Close any remaining positions at the end
            if positions:
                # Use last available signal date and price for closing
                last_signal = signals.iloc[-1]
                for ticker, position in positions.items():
                    # In real implementation, you'd get the actual closing price
                    closing_price = position['entry_price']  # Simplified
                    trade = self._close_position(position, ticker, closing_price, last_signal['signal_date'], 'forced_close')
                    trades.append(trade)

            logger.info(f"Simulated {len(trades)} trades")
            return trades

        except Exception as e:
            logger.error(f"Error simulating trades: {e}")
            return []

    def _close_position(self, position: Dict, ticker: str, exit_price: float, exit_date: datetime, exit_reason: str) -> Dict:
        """
        Close a position and create trade record.

        Args:
            position: Position information
            ticker: Stock ticker
            exit_price: Exit price
            exit_date: Exit date
            exit_reason: Reason for closing

        Returns:
            Trade dictionary
        """
        shares = position['shares']
        entry_price = position['entry_price']
        entry_date = position['entry_date']

        exit_value = shares * exit_price
        entry_value = shares * entry_price
        pnl = exit_value - entry_value
        pnl_pct = (pnl / entry_value) * 100
        holding_days = (exit_date - entry_date).days

        trade = {
            'ticker': ticker,
            'position_type': position['type'],
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

        return trade

    def _calculate_performance(self, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            trades: List of trade dictionaries

        Returns:
            Performance metrics dictionary
        """
        if not trades:
            return self._create_empty_performance()

        try:
            trades_df = pd.DataFrame(trades)

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
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')

            # Risk metrics
            avg_holding_days = trades_df['holding_days'].mean()
            max_win = trades_df['pnl'].max()
            max_loss = trades_df['pnl'].min()

            # Return-based metrics
            returns = trades_df['pnl_pct'] / 100
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(trades_df)

            performance = {
                'total_return': total_return,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_win': max_win,
                'max_loss': max_loss,
                'avg_holding_days': avg_holding_days,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades)
            }

            return performance

        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return self._create_empty_performance()

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) == 0 or returns.std() == 0:
                return 0

            excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
            return excess_returns / returns.std() * np.sqrt(252)  # Annualized

        except Exception:
            return 0

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        try:
            if trades_df.empty:
                return 0

            # Calculate cumulative returns
            cumulative_pnl = trades_df['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / self.initial_capital * 100

            return drawdown.min()

        except Exception:
            return 0

    def _calculate_statistics(self, trades: List[Dict], performance: Dict) -> Dict:
        """Calculate additional statistics."""
        try:
            if not trades:
                return {}

            trades_df = pd.DataFrame(trades)

            # Monthly breakdown
            trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
            monthly_pnl = trades_df.groupby('exit_month')['pnl'].sum()

            # Ticker breakdown
            ticker_pnl = trades_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False)

            statistics = {
                'best_month': monthly_pnl.max() if len(monthly_pnl) > 0 else 0,
                'worst_month': monthly_pnl.min() if len(monthly_pnl) > 0 else 0,
                'best_ticker': ticker_pnl.index[0] if len(ticker_pnl) > 0 else None,
                'worst_ticker': ticker_pnl.index[-1] if len(ticker_pnl) > 0 else None,
                'avg_trades_per_month': len(trades) / max(len(monthly_pnl), 1),
                'unique_tickers_traded': trades_df['ticker'].nunique()
            }

            return statistics

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}

    def _get_backtest_period(self, signals: pd.DataFrame) -> Dict:
        """Get backtest period information."""
        try:
            if signals.empty:
                return {}

            start_date = signals['signal_date'].min()
            end_date = signals['signal_date'].max()
            duration_days = (end_date - start_date).days

            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'duration_days': duration_days
            }

        except Exception:
            return {}

    def _create_empty_result(self, strategy_name: str, error: str = None) -> Dict:
        """Create empty result for failed backtests."""
        return {
            'strategy_name': strategy_name,
            'backtest_period': {},
            'total_signals': 0,
            'total_trades': 0,
            'trades': [],
            'performance': self._create_empty_performance(),
            'statistics': {},
            'error': error
        }

    def _create_empty_performance(self) -> Dict:
        """Create empty performance metrics."""
        return {
            'total_return': 0,
            'total_pnl': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_win': 0,
            'max_loss': 0,
            'avg_holding_days': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }