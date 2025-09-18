"""
Market Breadth Calculation Module
================================

Calculates comprehensive market breadth indicators including:
- Advance/Decline Line and related metrics
- New Highs/Lows analysis (52-week lookback)
- Moving Average Breadth percentages
- Specialized breadth thresholds and conditions

Follows the same architectural patterns as basic_calculations module.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import os

from src.data_reader import DataReader
from src.models import MarketBreadthConfig

logger = logging.getLogger(__name__)


class MarketBreadthCalculator:
    """
    Calculates comprehensive market breadth indicators following the same patterns
    as BasicCalculations module.
    """
    
    def __init__(self, config, data_reader: DataReader = None):
        """
        Initialize the market breadth calculator.
        
        Args:
            config: Configuration object
            data_reader: Optional DataReader instance
        """
        self.config = config
        self.data_reader = data_reader or DataReader(config)
        self.results_dir = config.directories.get('RESULTS_DIR', Path('results'))
        self.market_data_dir = config.directories.get('MARKET_DATA_DIR', Path('data/market_data'))
        
        # Ensure results directory exists
        os.makedirs(self.results_dir / 'market_breadth', exist_ok=True)
        
        # Column naming conventions following basic_calculations pattern
        self.column_prefix = "market_breadth"

    def _get_timeframe_periods(self, timeframe: str, user_config) -> dict:
        """
        Get timeframe-specific periods configuration.

        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            user_config: User configuration object

        Returns:
            Dictionary with ma_periods and new_high_lows_periods for the timeframe
        """
        # Get timeframe-specific MA periods
        ma_periods_attr = f'market_breadth_{timeframe}_ma_periods'
        ma_periods_str = getattr(user_config, ma_periods_attr, None)

        # Get timeframe-specific new high/lows periods
        hl_periods_attr = f'market_breadth_{timeframe}_new_high_lows_periods'
        hl_periods_str = getattr(user_config, hl_periods_attr, None)

        # Fallback to legacy settings if new ones not available
        if ma_periods_str is None:
            ma_periods_str = getattr(user_config, 'market_breadth_ma_periods', '20;50;200')
        if hl_periods_str is None:
            hl_periods_str = '252;63;20' if timeframe == 'daily' else '52;13;4' if timeframe == 'weekly' else '12;3;1'

        # Parse periods (handle both string and list formats)
        try:
            # If it's already a list (parsed by _parse_period_string), use it directly
            if isinstance(ma_periods_str, list):
                ma_periods = [int(x) for x in ma_periods_str]
            elif ma_periods_str:
                ma_periods = [int(x.strip()) for x in ma_periods_str.split(';')]
            else:
                ma_periods = [20, 50, 200]

            if isinstance(hl_periods_str, list):
                hl_periods = [int(x) for x in hl_periods_str]
            elif hl_periods_str:
                hl_periods = [int(x.strip()) for x in hl_periods_str.split(';')]
            else:
                hl_periods = [252, 63, 20]
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(f"Error parsing {timeframe} periods, using defaults: {e}")
            ma_periods = [20, 50, 200] if timeframe == 'daily' else [10, 20, 50] if timeframe == 'weekly' else [3, 6, 12]
            hl_periods = [252, 63, 20] if timeframe == 'daily' else [52, 13, 4] if timeframe == 'weekly' else [12, 3, 1]

        return {
            'ma_periods': ma_periods,
            'new_high_lows_periods': hl_periods,
            'timeframe': timeframe
        }

    def calculate_universe_breadth_matrix(self, timeframe: str, user_config) -> dict:
        """
        Calculate market breadth indicators for single or multiple universes and accumulate results.
        This method follows the same pattern as basic_calculations for matrix consolidation.
        
        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            user_config: User configuration object with market breadth settings
            
        Returns:
            dict: Results dictionary with universe names as keys
        """
        global market_breadth_calculation
        
        try:
            # Get configuration parameters
            universe_config = getattr(user_config, 'market_breadth_universe', {
                'type': 'single', 'universes': ['all'], 'display_name': 'all'
            })
            # Get timeframe-specific periods instead of fixed lookback_days
            periods_config = self._get_timeframe_periods(timeframe, user_config)
            
            logger.info(f"Calculating market breadth matrix for {timeframe} - type: {universe_config['type']}, universes: {universe_config['universes']}")
            
            # Initialize accumulator if needed
            if not hasattr(market_breadth_calculation, 'all_results'):
                market_breadth_calculation.all_results = {}
            if timeframe not in market_breadth_calculation.all_results:
                market_breadth_calculation.all_results[timeframe] = {}
            
            results = {}
            
            if universe_config['type'] == 'separate':
                # Process each universe separately
                for universe in universe_config['universes']:
                    logger.info(f"Processing separate universe: {universe}")
                    breadth_results = self.calculate_all_breadth_indicators(
                        universe_name=universe,
                        timeframe=timeframe,
                        periods_config=periods_config,
                        user_config=user_config
                    )
                    
                    if not breadth_results.empty:
                        results[universe] = breadth_results
                        market_breadth_calculation.all_results[timeframe][universe] = breadth_results
                        logger.info(f"Calculated {len(breadth_results)} days for universe: {universe}")
                    else:
                        logger.warning(f"No data for separate universe: {universe}")
            
            elif universe_config['type'] == 'combined':
                # Combine all universes into single dataset
                logger.info(f"Processing combined universe: {universe_config['display_name']}")
                combined_data = self._load_combined_universe_data(universe_config['universes'])
                
                if not combined_data.empty:
                    breadth_results = self._calculate_breadth_from_combined_data(combined_data, timeframe, periods_config, user_config)
                    if not breadth_results.empty:
                        display_name = universe_config['display_name']
                        results[display_name] = breadth_results
                        market_breadth_calculation.all_results[timeframe][display_name] = breadth_results
                        logger.info(f"Calculated {len(breadth_results)} days for combined universe: {display_name}")
                    else:
                        logger.warning(f"No breadth results for combined universe: {display_name}")
                else:
                    logger.warning(f"No data for combined universe: {universe_config['display_name']}")
            
            else:  # single universe (backwards compatible)
                universe = universe_config['universes'][0]
                logger.info(f"Processing single universe: {universe}")
                breadth_results = self.calculate_all_breadth_indicators(
                    universe_name=universe,
                    timeframe=timeframe,
                    periods_config=periods_config,
                    user_config=user_config
                )
                
                if not breadth_results.empty:
                    results[universe] = breadth_results
                    market_breadth_calculation.all_results[timeframe][universe] = breadth_results
                    logger.info(f"Calculated {len(breadth_results)} days for single universe: {universe}")
                else:
                    logger.warning(f"No data for single universe: {universe}")
            
            total_results = sum(len(df) for df in results.values())
            logger.info(f"Market breadth matrix calculation completed for {timeframe}: {len(results)} universes, {total_results} total data points")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating market breadth matrix for {timeframe}: {e}")
            return {}
        
    def calculate_all_breadth_indicators(self, universe_name: str = "all",
                                       timeframe: str = "daily", periods_config=None, user_config=None) -> pd.DataFrame:
        """
        Calculate all market breadth indicators for the specified universe.

        Args:
            universe_name: Name of the ticker universe to analyze
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            periods_config: Dictionary with timeframe-specific periods configuration
            
        Returns:
            DataFrame with all breadth calculations indexed by date
        """
        logger.info(f"Calculating market breadth indicators for universe: {universe_name}")
        
        try:
            # Load market data for the universe
            market_data = self._load_universe_market_data(universe_name)
            
            if market_data.empty:
                logger.warning(f"No market data found for universe: {universe_name}")
                return pd.DataFrame()
            
            # Initialize results DataFrame with date index
            dates = sorted(market_data['date'].unique())
            results_df = pd.DataFrame(index=pd.DatetimeIndex(dates, name='date'))
            
            # Calculate advance/decline indicators
            ad_results = self._calculate_advance_decline_indicators(market_data, results_df.index)
            results_df = pd.concat([results_df, ad_results], axis=1)
            
            # Get periods from configuration or fallback to defaults
            if periods_config is None:
                periods_config = self._get_timeframe_periods(timeframe, user_config)

            hl_periods = periods_config['new_high_lows_periods']
            ma_periods = periods_config['ma_periods']

            # Calculate new highs/lows indicators using dynamic periods (long, medium, short)
            for i, period in enumerate(hl_periods):
                period_name = ['long', 'medium', 'short'][i] if i < 3 else f'period_{i+1}'
                hl_results = self._calculate_new_highs_lows_indicators(
                    market_data, results_df.index, period, period_name
                )
                results_df = pd.concat([results_df, hl_results], axis=1)

            # Calculate moving average breadth indicators using dynamic periods
            ma_results = self._calculate_moving_average_breadth(
                market_data, results_df.index, ma_periods
            )
            results_df = pd.concat([results_df, ma_results], axis=1)
            
            # Calculate specialized breadth thresholds
            threshold_results = self._calculate_breadth_thresholds(results_df, user_config)
            results_df = pd.concat([results_df, threshold_results], axis=1)
            
            # Add metadata columns
            results_df[f'{self.column_prefix}_timeframe'] = timeframe
            results_df[f'{self.column_prefix}_universe'] = universe_name
            results_df[f'{self.column_prefix}_calculation_date'] = datetime.now().strftime('%Y-%m-%d')
            results_df[f'{self.column_prefix}_hl_periods'] = ';'.join(map(str, hl_periods))
            results_df[f'{self.column_prefix}_ma_periods'] = ';'.join(map(str, ma_periods))
            
            logger.info(f"Calculated {len(results_df.columns)} breadth indicators for {len(results_df)} trading days")

            # Add advanced analysis columns (signal strength, threshold classification, etc.)
            analysis_results = self._add_analysis_columns(results_df)
            results_df = pd.concat([results_df, analysis_results], axis=1)

            logger.info(f"Enhanced with {len(analysis_results.columns)} analysis columns. Total: {len(results_df.columns)} columns")

            return results_df
            
        except Exception as e:
            logger.error(f"Error calculating market breadth indicators: {e}")
            return pd.DataFrame()
    
    def _load_universe_market_data(self, universe_name: str) -> pd.DataFrame:
        """
        Load market data for all tickers in the specified universe.
        
        Args:
            universe_name: Name of the ticker universe
            
        Returns:
            Combined DataFrame with OHLCV data for all tickers
        """
        try:
            # Load ticker universe
            if universe_name == "all":
                universe_file = self.results_dir / 'ticker_universes' / 'ticker_universe_all.csv'
            else:
                universe_file = self.results_dir / 'ticker_universes' / f'ticker_universe_{universe_name}.csv'
            
            if not universe_file.exists():
                logger.error(f"Universe file not found: {universe_file}")
                return pd.DataFrame()
            
            universe_df = pd.read_csv(universe_file)
            tickers = universe_df['ticker'].tolist()
            
            logger.info(f"Loading market data for {len(tickers)} tickers from universe: {universe_name}")
            
            # Load daily market data for all tickers
            all_data = []
            daily_dir = self.config.get_market_data_dir('daily')
            
            for ticker in tickers:
                ticker_file = daily_dir / f"{ticker}.csv"
                if ticker_file.exists():
                    try:
                        ticker_data = pd.read_csv(ticker_file)
                        ticker_data['ticker'] = ticker
                        
                        # Handle date column naming (could be 'Date' or 'date')
                        date_col = 'Date' if 'Date' in ticker_data.columns else 'date'
                        if date_col in ticker_data.columns:
                            # Handle timezone-aware dates and convert to naive UTC dates
                            ticker_data['date'] = pd.to_datetime(ticker_data[date_col], utc=True).dt.date
                            ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                            
                            # Standardize column names for OHLCV data
                            column_mapping = {
                                'Open': 'open',
                                'High': 'high', 
                                'Low': 'low',
                                'Close': 'close',
                                'Volume': 'volume'
                            }
                            ticker_data = ticker_data.rename(columns=column_mapping)
                            
                            # Keep only essential columns to reduce memory usage
                            essential_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
                            available_columns = [col for col in essential_columns if col in ticker_data.columns]
                            ticker_data = ticker_data[available_columns]
                            
                            all_data.append(ticker_data)
                        else:
                            logger.debug(f"No date column found for {ticker}")
                    except Exception as e:
                        logger.debug(f"Error loading data for {ticker}: {e}")
            
            if not all_data:
                logger.warning("No market data files found")
                return pd.DataFrame()
            
            # Combine all ticker data
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded market data for {len(combined_data['ticker'].unique())} tickers")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error loading universe market data: {e}")
            return pd.DataFrame()
    
    def _calculate_advance_decline_indicators(self, market_data: pd.DataFrame, 
                                            date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate advance/decline line and related indicators.
        
        Args:
            market_data: Combined market data for all tickers
            date_index: Date index for results
            
        Returns:
            DataFrame with advance/decline indicators
        """
        results = pd.DataFrame(index=date_index)
        
        try:
            logger.debug("Calculating advance/decline indicators")
            
            # Calculate daily price changes
            market_data = market_data.sort_values(['ticker', 'date'])
            market_data['prev_close'] = market_data.groupby('ticker')['close'].shift(1)
            market_data['price_change'] = market_data['close'] - market_data['prev_close']
            market_data['pct_change'] = (market_data['price_change'] / market_data['prev_close']) * 100
            
            # Group by date to calculate daily breadth
            daily_breadth = market_data.groupby('date').agg({
                'ticker': 'count',  # Total stocks traded
                'price_change': lambda x: (x > 0).sum(),  # Advancing stocks
                'pct_change': ['mean', 'std']  # Average change and volatility
            }).round(4)
            
            # Flatten column names
            daily_breadth.columns = ['total_stocks', 'advancing_stocks', 'avg_pct_change', 'pct_change_std']
            
            # Calculate declining stocks
            declining_data = market_data.groupby('date')['price_change'].apply(lambda x: (x < 0).sum())
            daily_breadth['declining_stocks'] = declining_data
            
            # Calculate unchanged stocks
            unchanged_data = market_data.groupby('date')['price_change'].apply(lambda x: (x == 0).sum())
            daily_breadth['unchanged_stocks'] = unchanged_data
            
            # Reindex to match our date index
            daily_breadth = daily_breadth.reindex(date_index).fillna(0)
            
            # Calculate core advance/decline metrics
            results[f'{self.column_prefix}_total_stocks'] = daily_breadth['total_stocks']
            results[f'{self.column_prefix}_advancing_stocks'] = daily_breadth['advancing_stocks']
            results[f'{self.column_prefix}_declining_stocks'] = daily_breadth['declining_stocks']
            results[f'{self.column_prefix}_unchanged_stocks'] = daily_breadth['unchanged_stocks']
            
            # Calculate net advances
            results[f'{self.column_prefix}_net_advances'] = (
                results[f'{self.column_prefix}_advancing_stocks'] - 
                results[f'{self.column_prefix}_declining_stocks']
            )
            
            # Calculate advance/decline ratio
            results[f'{self.column_prefix}_ad_ratio'] = (
                results[f'{self.column_prefix}_advancing_stocks'] / 
                results[f'{self.column_prefix}_declining_stocks'].replace(0, np.nan)
            ).round(4)
            
            # Calculate advance/decline line (cumulative net advances)
            results[f'{self.column_prefix}_ad_line'] = results[f'{self.column_prefix}_net_advances'].cumsum()
            
            # Calculate advance percentage
            results[f'{self.column_prefix}_advance_pct'] = (
                (results[f'{self.column_prefix}_advancing_stocks'] / 
                 results[f'{self.column_prefix}_total_stocks'].replace(0, np.nan)) * 100
            ).round(2)
            
            # Calculate decline percentage
            results[f'{self.column_prefix}_decline_pct'] = (
                (results[f'{self.column_prefix}_declining_stocks'] / 
                 results[f'{self.column_prefix}_total_stocks'].replace(0, np.nan)) * 100
            ).round(2)
            
            # Add average price change and volatility
            results[f'{self.column_prefix}_avg_pct_change'] = daily_breadth['avg_pct_change']
            results[f'{self.column_prefix}_pct_change_volatility'] = daily_breadth['pct_change_std']
            
            logger.debug(f"Calculated {len([c for c in results.columns if 'ad_' in c or 'advance' in c or 'decline' in c])} advance/decline indicators")
            
        except Exception as e:
            logger.error(f"Error calculating advance/decline indicators: {e}")
        
        return results

    def _calculate_new_highs_lows_indicators(self, market_data: pd.DataFrame,
                                           date_index: pd.DatetimeIndex, period: int,
                                           period_name: str) -> pd.DataFrame:
        """
        Calculate new highs/lows indicators for any specified period.

        Args:
            market_data: DataFrame with OHLCV data
            date_index: DatetimeIndex for results alignment
            period: Number of periods for lookback (e.g., 252, 63, 20)
            period_name: Name for the period ('long', 'medium', 'short')

        Returns:
            DataFrame with new highs/lows indicators
        """
        results = pd.DataFrame(index=date_index)

        try:
            logger.debug(f"Calculating {period}-period new highs/lows indicators ({period_name})")

            # Calculate rolling highs and lows for the specified period
            market_data = market_data.sort_values(['ticker', 'date'])
            market_data[f'{period}d_high'] = market_data.groupby('ticker')['high'].rolling(
                window=period, min_periods=max(1, period//4)
            ).max().reset_index(0, drop=True)

            market_data[f'{period}d_low'] = market_data.groupby('ticker')['low'].rolling(
                window=period, min_periods=max(1, period//4)
            ).min().reset_index(0, drop=True)

            # Identify new highs and lows
            market_data[f'is_{period}d_new_high'] = (market_data['high'] >= market_data[f'{period}d_high'])
            market_data[f'is_{period}d_new_low'] = (market_data['low'] <= market_data[f'{period}d_low'])

            # Aggregate by date
            daily_hl = market_data.groupby('date').agg({
                f'is_{period}d_new_high': 'sum',
                f'is_{period}d_new_low': 'sum',
                'ticker': 'count'
            }).reindex(date_index).fillna(0)

            daily_hl.columns = [f'{period_name}_new_highs', f'{period_name}_new_lows', f'total_stocks_hl_{period_name}']

            # Add to results with proper column prefixes
            results[f'{self.column_prefix}_{period_name}_new_highs'] = daily_hl[f'{period_name}_new_highs']
            results[f'{self.column_prefix}_{period_name}_new_lows'] = daily_hl[f'{period_name}_new_lows']
            results[f'{self.column_prefix}_total_stocks_hl_{period_name}'] = daily_hl[f'total_stocks_hl_{period_name}']

            # Calculate derived metrics
            results[f'{self.column_prefix}_net_{period_name}_new_highs'] = (
                results[f'{self.column_prefix}_{period_name}_new_highs'] -
                results[f'{self.column_prefix}_{period_name}_new_lows']
            )

            results[f'{self.column_prefix}_hl_ratio_{period_name}'] = (
                results[f'{self.column_prefix}_{period_name}_new_highs'] /
                results[f'{self.column_prefix}_{period_name}_new_lows'].replace(0, np.nan)
            ).round(2)

            results[f'{self.column_prefix}_{period_name}_new_highs_pct'] = (
                (results[f'{self.column_prefix}_{period_name}_new_highs'] /
                 results[f'{self.column_prefix}_total_stocks_hl_{period_name}'].replace(0, np.nan)) * 100
            ).round(2)

            results[f'{self.column_prefix}_{period_name}_new_lows_pct'] = (
                (results[f'{self.column_prefix}_{period_name}_new_lows'] /
                 results[f'{self.column_prefix}_total_stocks_hl_{period_name}'].replace(0, np.nan)) * 100
            ).round(2)

            results[f'{self.column_prefix}_net_hl_line_{period_name}'] = results[f'{self.column_prefix}_net_{period_name}_new_highs'].cumsum()

            logger.debug(f"Calculated {len([c for c in results.columns if period_name in c])} {period}-{period_name} new highs/lows indicators")

        except Exception as e:
            logger.error(f"Error calculating {period}-{period_name} new highs/lows indicators: {e}")

        return results

    def _calculate_252day_new_highs_lows_indicators(self, market_data: pd.DataFrame,
                                                  date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate 252-day new highs/lows indicators with 252-day lookback period.
        
        Args:
            market_data: Combined market data for all tickers
            date_index: Date index for results
            
        Returns:
            DataFrame with 252-day new highs/lows indicators
        """
        results = pd.DataFrame(index=date_index)
        
        try:
            logger.debug("Calculating 252-day new highs/lows indicators")
            
            # Sort data for rolling calculations
            market_data = market_data.sort_values(['ticker', 'date'])
            
            # Calculate 252-day rolling highs and lows
            market_data['252d_high'] = market_data.groupby('ticker')['high'].rolling(
                window=252, min_periods=50
            ).max().reset_index(0, drop=True)
            
            market_data['252d_low'] = market_data.groupby('ticker')['low'].rolling(
                window=252, min_periods=50
            ).min().reset_index(0, drop=True)
            
            # Identify new highs and new lows
            market_data['is_252day_new_high'] = (market_data['high'] >= market_data['252d_high'])
            market_data['is_252day_new_low'] = (market_data['low'] <= market_data['252d_low'])
            
            # Group by date to calculate daily new highs/lows
            daily_hl = market_data.groupby('date').agg({
                'is_252day_new_high': 'sum',
                'is_252day_new_low': 'sum',
                'ticker': 'count'
            })
            
            daily_hl.columns = ['252day_new_highs', '252day_new_lows', 'total_stocks_hl_252day']
            daily_hl = daily_hl.reindex(date_index).fillna(0)
            
            # Calculate core 252-day new highs/lows metrics
            results[f'{self.column_prefix}_252day_new_highs'] = daily_hl['252day_new_highs']
            results[f'{self.column_prefix}_252day_new_lows'] = daily_hl['252day_new_lows']
            results[f'{self.column_prefix}_total_stocks_hl_252day'] = daily_hl['total_stocks_hl_252day']
            
            # Calculate net 252-day new highs
            results[f'{self.column_prefix}_net_252day_new_highs'] = (
                results[f'{self.column_prefix}_252day_new_highs'] - 
                results[f'{self.column_prefix}_252day_new_lows']
            )
            
            # Calculate 252-day new highs/lows ratio
            results[f'{self.column_prefix}_hl_ratio_252day'] = (
                results[f'{self.column_prefix}_252day_new_highs'] / 
                results[f'{self.column_prefix}_252day_new_lows'].replace(0, np.nan)
            ).round(4)
            
            # Calculate 252-day new highs percentage
            results[f'{self.column_prefix}_252day_new_highs_pct'] = (
                (results[f'{self.column_prefix}_252day_new_highs'] / 
                 results[f'{self.column_prefix}_total_stocks_hl_252day'].replace(0, np.nan)) * 100
            ).round(2)
            
            # Calculate 252-day new lows percentage
            results[f'{self.column_prefix}_252day_new_lows_pct'] = (
                (results[f'{self.column_prefix}_252day_new_lows'] / 
                 results[f'{self.column_prefix}_total_stocks_hl_252day'].replace(0, np.nan)) * 100
            ).round(2)
            
            # Calculate cumulative net 252-day new highs line
            results[f'{self.column_prefix}_net_hl_line_252day'] = results[f'{self.column_prefix}_net_252day_new_highs'].cumsum()
            
            logger.debug(f"Calculated {len([c for c in results.columns if '252day' in c])} 252-day new highs/lows indicators")
            
        except Exception as e:
            logger.error(f"Error calculating 252-day new highs/lows indicators: {e}")
        
        return results
    
    def _calculate_20day_new_highs_lows_indicators(self, market_data: pd.DataFrame, 
                                                  date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate 20-day new highs/lows indicators with 20-day lookback period.
        
        Args:
            market_data: Combined market data for all tickers
            date_index: Date index for results
            
        Returns:
            DataFrame with 20-day new highs/lows indicators
        """
        results = pd.DataFrame(index=date_index)
        
        try:
            logger.debug("Calculating 20-day new highs/lows indicators")
            
            # Sort data for rolling calculations
            market_data = market_data.sort_values(['ticker', 'date'])
            
            # Calculate 20-day rolling highs and lows
            market_data['20d_high'] = market_data.groupby('ticker')['high'].rolling(
                window=20, min_periods=10
            ).max().reset_index(0, drop=True)
            
            market_data['20d_low'] = market_data.groupby('ticker')['low'].rolling(
                window=20, min_periods=10
            ).min().reset_index(0, drop=True)
            
            # Identify new highs and new lows
            market_data['is_20day_new_high'] = (market_data['high'] >= market_data['20d_high'])
            market_data['is_20day_new_low'] = (market_data['low'] <= market_data['20d_low'])
            
            # Group by date to calculate daily new highs/lows
            daily_hl_20day = market_data.groupby('date').agg({
                'is_20day_new_high': 'sum',
                'is_20day_new_low': 'sum',
                'ticker': 'count'
            })
            
            daily_hl_20day.columns = ['20day_new_highs', '20day_new_lows', 'total_stocks_hl_20day']
            daily_hl_20day = daily_hl_20day.reindex(date_index).fillna(0)
            
            # Calculate core 20-day new highs/lows metrics
            results[f'{self.column_prefix}_20day_new_highs'] = daily_hl_20day['20day_new_highs']
            results[f'{self.column_prefix}_20day_new_lows'] = daily_hl_20day['20day_new_lows']
            results[f'{self.column_prefix}_total_stocks_hl_20day'] = daily_hl_20day['total_stocks_hl_20day']
            
            # Calculate net 20-day new highs
            results[f'{self.column_prefix}_net_20day_new_highs'] = (
                results[f'{self.column_prefix}_20day_new_highs'] - 
                results[f'{self.column_prefix}_20day_new_lows']
            )
            
            # Calculate 20-day new highs/lows ratio
            results[f'{self.column_prefix}_hl_ratio_20day'] = (
                results[f'{self.column_prefix}_20day_new_highs'] / 
                results[f'{self.column_prefix}_20day_new_lows'].replace(0, np.nan)
            ).round(4)
            
            # Calculate 20-day new highs percentage
            results[f'{self.column_prefix}_20day_new_highs_pct'] = (
                (results[f'{self.column_prefix}_20day_new_highs'] / 
                 results[f'{self.column_prefix}_total_stocks_hl_20day'].replace(0, np.nan)) * 100
            ).round(2)
            
            # Calculate 20-day new lows percentage
            results[f'{self.column_prefix}_20day_new_lows_pct'] = (
                (results[f'{self.column_prefix}_20day_new_lows'] / 
                 results[f'{self.column_prefix}_total_stocks_hl_20day'].replace(0, np.nan)) * 100
            ).round(2)
            
            # Calculate cumulative net 20-day new highs line
            results[f'{self.column_prefix}_net_hl_line_20day'] = results[f'{self.column_prefix}_net_20day_new_highs'].cumsum()
            
            logger.debug(f"Calculated {len([c for c in results.columns if '20day' in c])} 20-day new highs/lows indicators")
            
        except Exception as e:
            logger.error(f"Error calculating 20-day new highs/lows indicators: {e}")
        
        return results
    
    def _calculate_63day_new_highs_lows_indicators(self, market_data: pd.DataFrame, 
                                                  date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate 63-day new highs/lows indicators with 63-day lookback period (~3 months).
        
        Args:
            market_data: Combined market data for all tickers
            date_index: Date index for results
            
        Returns:
            DataFrame with 63-day new highs/lows indicators
        """
        results = pd.DataFrame(index=date_index)
        
        try:
            logger.debug("Calculating 63-day new highs/lows indicators")
            
            # Sort data for rolling calculations
            market_data = market_data.sort_values(['ticker', 'date'])
            
            # Calculate 63-day rolling highs and lows (~3 months of trading)
            market_data['63d_high'] = market_data.groupby('ticker')['high'].rolling(
                window=63, min_periods=30
            ).max().reset_index(0, drop=True)
            
            market_data['63d_low'] = market_data.groupby('ticker')['low'].rolling(
                window=63, min_periods=30
            ).min().reset_index(0, drop=True)
            
            # Identify new highs and new lows
            market_data['is_63day_new_high'] = (market_data['high'] >= market_data['63d_high'])
            market_data['is_63day_new_low'] = (market_data['low'] <= market_data['63d_low'])
            
            # Group by date to calculate daily new highs/lows
            daily_hl_63day = market_data.groupby('date').agg({
                'is_63day_new_high': 'sum',
                'is_63day_new_low': 'sum',
                'ticker': 'count'
            })
            
            daily_hl_63day.columns = ['63day_new_highs', '63day_new_lows', 'total_stocks_hl_63day']
            daily_hl_63day = daily_hl_63day.reindex(date_index).fillna(0)
            
            # Calculate core 63-day new highs/lows metrics
            results[f'{self.column_prefix}_63day_new_highs'] = daily_hl_63day['63day_new_highs']
            results[f'{self.column_prefix}_63day_new_lows'] = daily_hl_63day['63day_new_lows']
            results[f'{self.column_prefix}_total_stocks_hl_63day'] = daily_hl_63day['total_stocks_hl_63day']
            
            # Calculate net 63-day new highs
            results[f'{self.column_prefix}_net_63day_new_highs'] = (
                results[f'{self.column_prefix}_63day_new_highs'] - 
                results[f'{self.column_prefix}_63day_new_lows']
            )
            
            # Calculate 63-day new highs/lows ratio
            results[f'{self.column_prefix}_hl_ratio_63day'] = (
                results[f'{self.column_prefix}_63day_new_highs'] / 
                results[f'{self.column_prefix}_63day_new_lows'].replace(0, np.nan)
            ).round(4)
            
            # Calculate 63-day new highs percentage
            results[f'{self.column_prefix}_63day_new_highs_pct'] = (
                (results[f'{self.column_prefix}_63day_new_highs'] / 
                 results[f'{self.column_prefix}_total_stocks_hl_63day'].replace(0, np.nan)) * 100
            ).round(2)
            
            # Calculate 63-day new lows percentage
            results[f'{self.column_prefix}_63day_new_lows_pct'] = (
                (results[f'{self.column_prefix}_63day_new_lows'] / 
                 results[f'{self.column_prefix}_total_stocks_hl_63day'].replace(0, np.nan)) * 100
            ).round(2)
            
            # Calculate cumulative net 63-day new highs line
            results[f'{self.column_prefix}_net_hl_line_63day'] = results[f'{self.column_prefix}_net_63day_new_highs'].cumsum()
            
            logger.debug(f"Calculated {len([c for c in results.columns if '63day' in c])} 63-day new highs/lows indicators")
            
        except Exception as e:
            logger.error(f"Error calculating 63-day new highs/lows indicators: {e}")
        
        return results
    
    def _calculate_moving_average_breadth(self, market_data: pd.DataFrame,
                                        date_index: pd.DatetimeIndex, ma_periods: list = None) -> pd.DataFrame:
        """
        Calculate moving average breadth indicators using dynamic periods.

        Args:
            market_data: Combined market data for all tickers
            date_index: Date index for results
            ma_periods: List of MA periods to calculate (defaults to [20, 50, 200])

        Returns:
            DataFrame with moving average breadth indicators
        """
        results = pd.DataFrame(index=date_index)

        try:
            # Use provided periods or defaults
            if ma_periods is None:
                ma_periods = [20, 50, 200]

            logger.debug(f"Calculating moving average breadth indicators for periods: {ma_periods}")
            
            for period in ma_periods:
                # Calculate moving averages for each ticker
                market_data = market_data.sort_values(['ticker', 'date'])
                ma_col = f'ma_{period}'
                market_data[ma_col] = market_data.groupby('ticker')['close'].rolling(
                    window=period, min_periods=period//2
                ).mean().reset_index(0, drop=True)
                
                # Identify stocks above their moving average
                above_ma_col = f'above_ma_{period}'
                market_data[above_ma_col] = market_data['close'] > market_data[ma_col]
                
                # Group by date to calculate breadth
                daily_ma_breadth = market_data.groupby('date').agg({
                    above_ma_col: 'sum',
                    'ticker': 'count'
                })
                
                daily_ma_breadth.columns = [f'stocks_above_ma_{period}', f'total_stocks_ma_{period}']
                daily_ma_breadth = daily_ma_breadth.reindex(date_index).fillna(0)
                
                # Store results
                results[f'{self.column_prefix}_stocks_above_ma_{period}'] = daily_ma_breadth[f'stocks_above_ma_{period}']
                results[f'{self.column_prefix}_total_stocks_ma_{period}'] = daily_ma_breadth[f'total_stocks_ma_{period}']
                
                # Calculate percentage above moving average
                results[f'{self.column_prefix}_pct_above_ma_{period}'] = (
                    (results[f'{self.column_prefix}_stocks_above_ma_{period}'] / 
                     results[f'{self.column_prefix}_total_stocks_ma_{period}'].replace(0, np.nan)) * 100
                ).round(2)
            
            logger.debug(f"Calculated moving average breadth for periods: {ma_periods}")
            
        except Exception as e:
            logger.error(f"Error calculating moving average breadth: {e}")
        
        return results
    
    def _calculate_breadth_thresholds(self, results_df: pd.DataFrame, user_config=None) -> pd.DataFrame:
        """
        Calculate specialized breadth threshold indicators using generic position-based configuration.

        Args:
            results_df: DataFrame with previously calculated indicators
            user_config: User configuration object with threshold settings

        Returns:
            DataFrame with threshold-based indicators
        """
        threshold_results = pd.DataFrame(index=results_df.index)

        try:
            logger.debug("Calculating breadth threshold indicators")

            # Get generic threshold configuration
            new_highs_thresholds = {
                'long': getattr(user_config, 'market_breadth_new_highs_threshold_long', 100),
                'medium': getattr(user_config, 'market_breadth_new_highs_threshold_medium', 100),
                'short': getattr(user_config, 'market_breadth_new_highs_threshold_short', 100)
            }

            success_window_pcts = {
                'long': getattr(user_config, 'market_breadth_success_window_pct_long', 25),
                'medium': getattr(user_config, 'market_breadth_success_window_pct_medium', 50),
                'short': getattr(user_config, 'market_breadth_success_window_pct_short', 50)
            }

            success_threshold_pcts = {
                'long': getattr(user_config, 'market_breadth_success_threshold_pct_long', 30),
                'medium': getattr(user_config, 'market_breadth_success_threshold_pct_medium', 30),
                'short': getattr(user_config, 'market_breadth_success_threshold_pct_short', 50)
            }

            # Universal thresholds (same across timeframes)
            strong_ad_ratio = getattr(user_config, 'market_breadth_strong_ad_ratio_threshold', 2.0)
            weak_ad_ratio = getattr(user_config, 'market_breadth_weak_ad_ratio_threshold', 0.5)
            strong_advance = getattr(user_config, 'market_breadth_strong_advance_threshold', 70.0)
            weak_advance = getattr(user_config, 'market_breadth_weak_advance_threshold', 30.0)
            strong_ma_breadth = getattr(user_config, 'market_breadth_strong_ma_breadth_threshold', 80.0)
            weak_ma_breadth = getattr(user_config, 'market_breadth_weak_ma_breadth_threshold', 20.0)

            logger.debug(f"Using generic thresholds - long: {new_highs_thresholds['long']}, medium: {new_highs_thresholds['medium']}, short: {new_highs_thresholds['short']}")

            # Calculate new highs threshold conditions for each period (long, medium, short)
            for period_name in ['long', 'medium', 'short']:
                nh_col = f'{self.column_prefix}_{period_name}_new_highs'

                if nh_col in results_df.columns:
                    threshold = new_highs_thresholds[period_name]

                    # Basic threshold condition
                    nh_gt_threshold = (results_df[nh_col] > threshold).astype(int)
                    threshold_results[f'{self.column_prefix}_{period_name}_new_highs_gt_{threshold}'] = nh_gt_threshold

                    # Calculate success window for this period
                    # Get period length from actual column data if available
                    hl_periods_col = f'{self.column_prefix}_hl_periods'
                    if hl_periods_col in results_df.columns:
                        # Extract period from metadata
                        periods_str = results_df[hl_periods_col].iloc[0] if not results_df.empty else '252;63;20'
                        periods = [int(p) for p in periods_str.split(';')]
                        period_lengths = {'long': periods[0], 'medium': periods[1] if len(periods) > 1 else 63, 'short': periods[2] if len(periods) > 2 else 20}
                    else:
                        # Default fallback
                        period_lengths = {'long': 252, 'medium': 63, 'short': 20}

                    period_length = period_lengths.get(period_name, 252)
                    window_size = max(1, int(period_length * success_window_pcts[period_name] / 100))
                    success_threshold = max(1, int(window_size * success_threshold_pcts[period_name] / 100))

                    # Calculate success condition if we have enough data
                    if len(results_df) >= window_size:
                        success_count = nh_gt_threshold.rolling(window=window_size, min_periods=max(1, window_size//2)).sum()
                        success_condition = (success_count >= success_threshold).astype(int)
                        threshold_results[f'{self.column_prefix}_{period_name}_success_condition'] = success_condition
                    else:
                        threshold_results[f'{self.column_prefix}_{period_name}_success_condition'] = 0
            
            # Calculate advance/decline ratio thresholds
            if f'{self.column_prefix}_ad_ratio' in results_df.columns:
                ad_ratio_gt_strong = (results_df[f'{self.column_prefix}_ad_ratio'] > strong_ad_ratio).astype(int)
                threshold_results[f'{self.column_prefix}_ad_ratio_gt_{str(strong_ad_ratio).replace(".", "_")}'] = ad_ratio_gt_strong
                
                ad_ratio_lt_weak = (results_df[f'{self.column_prefix}_ad_ratio'] < weak_ad_ratio).astype(int)
                threshold_results[f'{self.column_prefix}_ad_ratio_lt_{str(weak_ad_ratio).replace(".", "_")}'] = ad_ratio_lt_weak
            
            # Calculate breadth momentum (advance percentage thresholds)
            if f'{self.column_prefix}_advance_pct' in results_df.columns:
                strong_breadth = (results_df[f'{self.column_prefix}_advance_pct'] > strong_advance).astype(int)
                threshold_results[f'{self.column_prefix}_strong_advance_breadth'] = strong_breadth
                
                weak_breadth = (results_df[f'{self.column_prefix}_advance_pct'] < weak_advance).astype(int)
                threshold_results[f'{self.column_prefix}_weak_advance_breadth'] = weak_breadth
            
            # Calculate moving average breadth thresholds using dynamic MA periods
            ma_periods_col = f'{self.column_prefix}_ma_periods'
            if ma_periods_col in results_df.columns:
                # Extract MA periods from metadata
                ma_periods_str = results_df[ma_periods_col].iloc[0] if not results_df.empty else '20;50;200'
                ma_periods = [int(p) for p in ma_periods_str.split(';')]
            else:
                # Default fallback
                ma_periods = [20, 50, 200]

            for period in ma_periods:
                pct_col = f'{self.column_prefix}_pct_above_ma_{period}'
                if pct_col in results_df.columns:
                    # Strong MA breadth (above threshold% above MA)
                    strong_ma_col = f'{self.column_prefix}_strong_ma_breadth_{period}'
                    threshold_results[strong_ma_col] = (results_df[pct_col] > strong_ma_breadth).astype(int)

                    # Weak MA breadth (below threshold% above MA)
                    weak_ma_col = f'{self.column_prefix}_weak_ma_breadth_{period}'
                    threshold_results[weak_ma_col] = (results_df[pct_col] < weak_ma_breadth).astype(int)
            
            logger.debug(f"Calculated {len(threshold_results.columns)} threshold indicators")
            
        except Exception as e:
            logger.error(f"Error calculating breadth thresholds: {e}")
        
        return threshold_results
    
    def save_breadth_calculations(self, results_df: pd.DataFrame, universe_name: str, 
                                 file_suffix: str = "") -> str:
        """
        Save market breadth calculations to CSV file following naming conventions.
        
        Args:
            results_df: DataFrame with breadth calculations
            universe_name: Name of the universe analyzed
            file_suffix: Optional suffix for filename
            
        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if file_suffix:
                filename = f"market_breadth_{universe_name}_{file_suffix}_{timestamp}.csv"
            else:
                filename = f"market_breadth_{universe_name}_{timestamp}.csv"
            
            output_file = self.results_dir / 'market_breadth' / filename
            
            # Ensure the output directory exists
            os.makedirs(output_file.parent, exist_ok=True)
            
            # Reset index to include date as a column
            save_df = results_df.reset_index()
            
            # Save to CSV
            save_df.to_csv(output_file, index=False)
            
            logger.info(f"Saved market breadth calculations to: {output_file}")
            logger.info(f"File contains {len(save_df)} rows and {len(save_df.columns)} columns")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving breadth calculations: {e}")
            return ""
    
    def _load_combined_universe_data(self, universes: List[str]) -> pd.DataFrame:
        """
        Load and combine market data from multiple universes.
        
        Args:
            universes: List of universe names to combine
            
        Returns:
            Combined DataFrame with all tickers from all universes (deduplicated)
        """
        try:
            all_data = []
            total_tickers = 0
            
            for universe in universes:
                logger.info(f"Loading data for universe: {universe}")
                universe_data = self._load_universe_market_data(universe)
                
                if not universe_data.empty:
                    universe_ticker_count = universe_data['ticker'].nunique()
                    total_tickers += universe_ticker_count
                    all_data.append(universe_data)
                    logger.info(f"Loaded {universe_ticker_count} tickers from {universe}")
                else:
                    logger.warning(f"No data loaded for universe: {universe}")
            
            if all_data:
                # Combine all DataFrames
                combined = pd.concat(all_data, ignore_index=True)
                
                # Remove duplicates (tickers that appear in multiple universes)
                initial_rows = len(combined)
                combined = combined.drop_duplicates(subset=['ticker', 'date'])
                final_rows = len(combined)
                
                unique_tickers = combined['ticker'].nunique()
                
                logger.info(f"Combined universe data: {unique_tickers} unique tickers, {final_rows} total rows")
                if initial_rows != final_rows:
                    logger.info(f"Removed {initial_rows - final_rows} duplicate ticker-date combinations")
                
                return combined
            
            logger.warning("No data found for any universe in combined processing")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading combined universe data: {e}")
            return pd.DataFrame()
    
    def _calculate_breadth_from_combined_data(self, combined_data: pd.DataFrame, timeframe: str = "daily", periods_config=None, user_config=None) -> pd.DataFrame:
        """
        Calculate market breadth indicators from pre-combined universe data.

        Args:
            combined_data: Combined market data from multiple universes
            timeframe: Data timeframe (daily, weekly, monthly)
            periods_config: Configuration for new highs/lows and MA periods
            user_config: User configuration for thresholds

        Returns:
            DataFrame with breadth calculations indexed by date
        """
        try:
            if combined_data.empty:
                logger.warning("No combined data provided for breadth calculation")
                return pd.DataFrame()
            
            # Get unique dates and create date index
            dates = sorted(combined_data['date'].unique())
            results_df = pd.DataFrame(index=pd.DatetimeIndex(dates, name='date'))
            
            logger.info(f"Calculating breadth for combined data: {combined_data['ticker'].nunique()} tickers, {len(dates)} trading days")
            
            # Calculate advance/decline indicators
            ad_results = self._calculate_advance_decline_indicators(combined_data, results_df.index)
            results_df = pd.concat([results_df, ad_results], axis=1)
            
            # Get periods from configuration or fallback to defaults
            if periods_config is None:
                periods_config = self._get_timeframe_periods(timeframe, user_config)

            hl_periods = periods_config['new_high_lows_periods']
            ma_periods = periods_config['ma_periods']

            # Calculate new highs/lows indicators using dynamic periods (long, medium, short)
            for i, period in enumerate(hl_periods):
                period_name = ['long', 'medium', 'short'][i] if i < 3 else f'period_{i+1}'
                hl_results = self._calculate_new_highs_lows_indicators(
                    combined_data, results_df.index, period, period_name
                )
                results_df = pd.concat([results_df, hl_results], axis=1)

            # Calculate moving average breadth indicators using dynamic periods
            ma_results = self._calculate_moving_average_breadth(
                combined_data, results_df.index, ma_periods
            )
            results_df = pd.concat([results_df, ma_results], axis=1)
            
            # Calculate specialized breadth thresholds
            threshold_results = self._calculate_breadth_thresholds(results_df, user_config)
            results_df = pd.concat([results_df, threshold_results], axis=1)
            
            # Add metadata columns for combined universe
            combined_universe_name = "+".join([name for name in combined_data['ticker'].unique()[:5]]) + "..." if combined_data['ticker'].nunique() > 5 else "combined"
            results_df[f'{self.column_prefix}_universe'] = "combined"
            results_df[f'{self.column_prefix}_calculation_date'] = datetime.now().strftime('%Y-%m-%d')
            results_df[f'{self.column_prefix}_hl_periods'] = ';'.join(map(str, hl_periods))
            results_df[f'{self.column_prefix}_ma_periods'] = ';'.join(map(str, ma_periods))
            
            logger.info(f"Calculated {len([c for c in results_df.columns if self.column_prefix in c])} breadth indicators for combined universe")
            
            # Add advanced analysis columns
            analysis_results = self._add_analysis_columns(results_df)
            results_df = pd.concat([results_df, analysis_results], axis=1)

            return results_df

        except Exception as e:
            logger.error(f"Error calculating breadth from combined data: {e}")
            return pd.DataFrame()

    def _add_analysis_columns(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced analysis columns including momentum, signals, and ratings.

        Args:
            results_df: DataFrame with basic breadth calculations

        Returns:
            DataFrame with additional analysis columns
        """
        try:
            if results_df.empty:
                return pd.DataFrame()

            analysis_df = pd.DataFrame(index=results_df.index)

            # Calculate momentum indicators
            momentum_results = self._calculate_momentum_indicators(results_df)
            analysis_df = pd.concat([analysis_df, momentum_results], axis=1)

            # Calculate breadth signals
            signal_results = self._calculate_breadth_signals(results_df)
            analysis_df = pd.concat([analysis_df, signal_results], axis=1)

            # Calculate breadth ratings and scores
            rating_results = self._calculate_breadth_ratings(results_df)
            analysis_df = pd.concat([analysis_df, rating_results], axis=1)

            # Calculate consecutive day tracking
            consecutive_results = self._calculate_consecutive_indicators(signal_results)
            analysis_df = pd.concat([analysis_df, consecutive_results], axis=1)

            logger.debug(f"Added {len(analysis_df.columns)} analysis columns to breadth data")
            return analysis_df

        except Exception as e:
            logger.error(f"Error adding analysis columns: {e}")
            return pd.DataFrame()

    def _calculate_momentum_indicators(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators with configurable windows."""
        try:
            momentum_df = pd.DataFrame(index=results_df.index)

            # Get key breadth metrics for momentum calculation
            total_bullish_base = (
                results_df.get(f'{self.column_prefix}_strong_advance_breadth', 0) +
                results_df.get(f'{self.column_prefix}_strong_ma_breadth_20', 0) +
                results_df.get(f'{self.column_prefix}_strong_ma_breadth_50', 0) +
                results_df.get(f'{self.column_prefix}_strong_ma_breadth_200', 0) +
                results_df.get(f'{self.column_prefix}_long_success_condition', 0) +
                results_df.get(f'{self.column_prefix}_medium_success_condition', 0)
            )

            total_bearish_base = (
                results_df.get(f'{self.column_prefix}_weak_advance_breadth', 0) +
                results_df.get(f'{self.column_prefix}_weak_ma_breadth_20', 0) +
                results_df.get(f'{self.column_prefix}_weak_ma_breadth_50', 0) +
                results_df.get(f'{self.column_prefix}_weak_ma_breadth_200', 0)
            )

            # Calculate momentum for different windows
            for window in [5, 10, 20]:
                # Bullish momentum
                momentum_df[f'{self.column_prefix}_bullish_momentum_{window}d'] = total_bullish_base.rolling(
                    window=window, min_periods=window//2
                ).mean().round(2)

                # Bearish momentum
                momentum_df[f'{self.column_prefix}_bearish_momentum_{window}d'] = total_bearish_base.rolling(
                    window=window, min_periods=window//2
                ).mean().round(2)

                # Net momentum
                net_momentum = (total_bullish_base - total_bearish_base).rolling(
                    window=window, min_periods=window//2
                ).mean().round(2)
                momentum_df[f'{self.column_prefix}_net_momentum_{window}d'] = net_momentum

            return momentum_df

        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return pd.DataFrame()

    def _calculate_breadth_signals(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate breadth thrust, deterioration, and signal scores."""
        try:
            signals_df = pd.DataFrame(index=results_df.index)

            # Configurable thresholds (using defaults if not configured)
            thrust_threshold = 80
            deterioration_threshold = 20
            new_highs_threshold = 50
            ad_thrust_threshold = 2.0

            # Breadth thrust (strong MA breadth across timeframes)
            avg_ma_breadth = (
                results_df.get(f'{self.column_prefix}_pct_above_ma_20', 0) +
                results_df.get(f'{self.column_prefix}_pct_above_ma_50', 0) +
                results_df.get(f'{self.column_prefix}_pct_above_ma_200', 0)
            ) / 3
            signals_df[f'{self.column_prefix}_breadth_thrust'] = (avg_ma_breadth > thrust_threshold).astype(int)

            # Breadth deterioration
            signals_df[f'{self.column_prefix}_breadth_deterioration'] = (avg_ma_breadth < deterioration_threshold).astype(int)

            # A/D thrust
            signals_df[f'{self.column_prefix}_ad_thrust'] = (results_df.get(f'{self.column_prefix}_ad_ratio', 0) > ad_thrust_threshold).astype(int)

            # New highs expansion (using long period)
            signals_df[f'{self.column_prefix}_new_highs_expansion'] = (
                results_df.get(f'{self.column_prefix}_long_new_highs', 0) > new_highs_threshold
            ).astype(int)

            # New lows expansion
            signals_df[f'{self.column_prefix}_new_lows_expansion'] = (
                results_df.get(f'{self.column_prefix}_long_new_lows', 0) > new_highs_threshold
            ).astype(int)

            # Total bullish signals
            bullish_signals = (
                signals_df[f'{self.column_prefix}_breadth_thrust'] +
                signals_df[f'{self.column_prefix}_ad_thrust'] +
                signals_df[f'{self.column_prefix}_new_highs_expansion'] +
                results_df.get(f'{self.column_prefix}_strong_advance_breadth', 0) +
                results_df.get(f'{self.column_prefix}_long_success_condition', 0) +
                results_df.get(f'{self.column_prefix}_medium_success_condition', 0)
            )
            signals_df[f'{self.column_prefix}_total_bullish_signals'] = bullish_signals

            # Total bearish signals
            bearish_signals = (
                signals_df[f'{self.column_prefix}_breadth_deterioration'] +
                signals_df[f'{self.column_prefix}_new_lows_expansion'] +
                results_df.get(f'{self.column_prefix}_weak_advance_breadth', 0) +
                results_df.get(f'{self.column_prefix}_weak_ma_breadth_20', 0) +
                results_df.get(f'{self.column_prefix}_weak_ma_breadth_50', 0) +
                results_df.get(f'{self.column_prefix}_weak_ma_breadth_200', 0)
            )
            signals_df[f'{self.column_prefix}_total_bearish_signals'] = bearish_signals

            # Net signal score
            signals_df[f'{self.column_prefix}_net_signal_score'] = bullish_signals - bearish_signals

            return signals_df

        except Exception as e:
            logger.error(f"Error calculating breadth signals: {e}")
            return pd.DataFrame()

    def _calculate_breadth_ratings(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate breadth ratings and overall scores."""
        try:
            ratings_df = pd.DataFrame(index=results_df.index)

            # Overall breadth score (0-100 scale)
            ma20_pct = results_df.get(f'{self.column_prefix}_pct_above_ma_20', 0)
            ma50_pct = results_df.get(f'{self.column_prefix}_pct_above_ma_50', 0)
            ma200_pct = results_df.get(f'{self.column_prefix}_pct_above_ma_200', 0)
            advance_pct = results_df.get(f'{self.column_prefix}_advance_pct', 0)

            overall_score = (ma20_pct * 0.3 + ma50_pct * 0.25 + ma200_pct * 0.25 + advance_pct * 0.2)
            ratings_df[f'{self.column_prefix}_overall_breadth_score'] = overall_score.round(2)

            # Breadth rating categories
            def categorize_breadth(score):
                if score >= 80:
                    return 'Excellent'
                elif score >= 60:
                    return 'Good'
                elif score >= 40:
                    return 'Fair'
                elif score >= 20:
                    return 'Poor'
                else:
                    return 'Very Poor'

            ratings_df[f'{self.column_prefix}_breadth_rating'] = overall_score.apply(categorize_breadth)

            # Signal strength classification
            def classify_signal_strength(bullish_signals, bearish_signals):
                net_signals = bullish_signals - bearish_signals
                if net_signals >= 3:
                    return 'Strong Bullish'
                elif net_signals >= 1:
                    return 'Moderate Bullish'
                elif net_signals <= -3:
                    return 'Strong Bearish'
                elif net_signals <= -1:
                    return 'Moderate Bearish'
                else:
                    return 'Neutral'

            # Calculate signal strength from boolean threshold columns
            def calculate_signal_strength_from_booleans(row):
                # Count bullish signals from boolean columns
                bullish_signals = sum([
                    row.get(f'{self.column_prefix}_strong_ma_breadth_20', 0),
                    row.get(f'{self.column_prefix}_strong_ma_breadth_50', 0),
                    row.get(f'{self.column_prefix}_strong_ma_breadth_200', 0),
                    row.get(f'{self.column_prefix}_ad_ratio_gt_2_0', 0),
                    row.get(f'{self.column_prefix}_long_new_highs_gt_100', 0)
                ])

                # Count bearish signals from boolean columns
                bearish_signals = sum([
                    row.get(f'{self.column_prefix}_weak_ma_breadth_20', 0),
                    row.get(f'{self.column_prefix}_weak_ma_breadth_50', 0),
                    row.get(f'{self.column_prefix}_weak_ma_breadth_200', 0),
                    row.get(f'{self.column_prefix}_ad_ratio_lt_0_5', 0)
                ])

                net_score = bullish_signals - bearish_signals

                # Apply signal strength classification matching your table
                if bullish_signals >= 3 and net_score >= 2:
                    return 'Strong Bullish'
                elif bearish_signals >= 3 and net_score <= -2:
                    return 'Strong Bearish'
                elif bullish_signals >= 2 and net_score >= 1:
                    return 'Moderate Bullish'
                elif bearish_signals >= 2 and net_score <= -1:
                    return 'Moderate Bearish'
                elif bullish_signals > 0 or bearish_signals > 0:
                    return 'Weak Signal (Bullish)' if net_score > 0 else 'Weak Signal (Bearish)'
                else:
                    return 'Neutral'

            # Apply signal strength calculation to each row
            ratings_df[f'{self.column_prefix}_signal_strength'] = results_df.apply(
                calculate_signal_strength_from_booleans, axis=1
            )

            # Calculate threshold classification from boolean columns
            def calculate_threshold_classification(row):
                """Map active boolean signals to threshold table classifications."""
                active_classifications = []

                if any([row.get(f'{self.column_prefix}_strong_ma_breadth_20', 0),
                        row.get(f'{self.column_prefix}_strong_ma_breadth_50', 0),
                        row.get(f'{self.column_prefix}_strong_ma_breadth_200', 0)]):
                    active_classifications.append("Breadth Thrust (Bullish)")

                if row.get(f'{self.column_prefix}_ad_ratio_gt_2_0', 0):
                    active_classifications.append("Advance/Decline Thrust (Bullish)")

                if row.get(f'{self.column_prefix}_long_new_highs_gt_100', 0):
                    active_classifications.append("New Highs Expansion (Bullish)")

                if any([row.get(f'{self.column_prefix}_weak_ma_breadth_20', 0),
                        row.get(f'{self.column_prefix}_weak_ma_breadth_50', 0),
                        row.get(f'{self.column_prefix}_weak_ma_breadth_200', 0)]):
                    active_classifications.append("Breadth Deterioration (Bearish)")

                if not active_classifications:
                    return "Neutral"
                elif len(active_classifications) == 1:
                    return active_classifications[0]
                else:
                    return "; ".join(active_classifications)

            # Apply threshold classification to each row
            ratings_df[f'{self.column_prefix}_threshold_classification'] = results_df.apply(
                calculate_threshold_classification, axis=1
            )

            # Market participation (percentage of advancing stocks)
            market_participation = results_df.get(f'{self.column_prefix}_advance_pct', 0)
            ratings_df[f'{self.column_prefix}_market_participation'] = market_participation

            return ratings_df

        except Exception as e:
            logger.error(f"Error calculating breadth ratings: {e}")
            return pd.DataFrame()

    def _calculate_consecutive_indicators(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate consecutive bullish/bearish day counts."""
        try:
            consecutive_df = pd.DataFrame(index=signals_df.index)

            if f'{self.column_prefix}_total_bullish_signals' in signals_df.columns:
                bullish_higher = (
                    signals_df[f'{self.column_prefix}_total_bullish_signals'] >
                    signals_df[f'{self.column_prefix}_total_bearish_signals']
                )
                bearish_higher = (
                    signals_df[f'{self.column_prefix}_total_bearish_signals'] >
                    signals_df[f'{self.column_prefix}_total_bullish_signals']
                )

                consecutive_df[f'{self.column_prefix}_consecutive_bullish_days'] = self._calculate_consecutive_signals(bullish_higher)
                consecutive_df[f'{self.column_prefix}_consecutive_bearish_days'] = self._calculate_consecutive_signals(bearish_higher)

            return consecutive_df

        except Exception as e:
            logger.error(f"Error calculating consecutive indicators: {e}")
            return pd.DataFrame()

    def _calculate_consecutive_signals(self, condition_series: pd.Series) -> pd.Series:
        """Calculate consecutive days where condition is True."""
        try:
            consecutive = pd.Series(index=condition_series.index, dtype=int)
            count = 0

            for i, value in enumerate(condition_series):
                if value:
                    count += 1
                else:
                    count = 0
                consecutive.iloc[i] = count

            return consecutive

        except Exception as e:
            logger.error(f"Error calculating consecutive signals: {e}")
            return pd.Series(index=condition_series.index, dtype=int)

    def get_breadth_summary(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate a summary of market breadth calculations.
        
        Args:
            results_df: DataFrame with breadth calculations
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            if results_df.empty:
                return {"error": "No data to summarize"}
            
            summary = {
                "calculation_period": {
                    "start_date": results_df.index.min().strftime('%Y-%m-%d'),
                    "end_date": results_df.index.max().strftime('%Y-%m-%d'),
                    "total_days": len(results_df)
                },
                "breadth_indicators": {
                    "total_indicators": len([c for c in results_df.columns if self.column_prefix in c]),
                    "advance_decline_indicators": len([c for c in results_df.columns if 'ad_' in c or 'advance' in c]),
                    "252day_new_highs_lows_indicators": len([c for c in results_df.columns if '252day' in c and ('high' in c or 'low' in c)]),
                    "20day_new_highs_lows_indicators": len([c for c in results_df.columns if '20day' in c and ('high' in c or 'low' in c)]),
                    "63day_new_highs_lows_indicators": len([c for c in results_df.columns if '63day' in c and ('high' in c or 'low' in c)]),
                    "ma_breadth_indicators": len([c for c in results_df.columns if 'ma_' in c]),
                    "threshold_indicators": len([c for c in results_df.columns if '_gt_' in c or '_lt_' in c])
                }
            }
            
            # Add recent readings for key indicators
            if len(results_df) > 0:
                latest_data = results_df.iloc[-1]
                summary["latest_readings"] = {}
                
                key_indicators = [
                    f'{self.column_prefix}_advance_pct',
                    f'{self.column_prefix}_252day_new_highs',
                    f'{self.column_prefix}_252day_new_lows',
                    f'{self.column_prefix}_20day_new_highs',
                    f'{self.column_prefix}_20day_new_lows',
                    f'{self.column_prefix}_63day_new_highs',
                    f'{self.column_prefix}_63day_new_lows',
                    f'{self.column_prefix}_pct_above_ma_50',
                    f'{self.column_prefix}_daily_252day_new_highs_gt_100',
                    f'{self.column_prefix}_daily_20day_new_highs_gt_100',
                    f'{self.column_prefix}_daily_63day_new_highs_gt_100'
                ]
                
                for indicator in key_indicators:
                    if indicator in latest_data.index:
                        summary["latest_readings"][indicator] = latest_data[indicator]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating breadth summary: {e}")
            return {"error": str(e)}


# Module-level variable to accumulate results across batches (following basic_calculations pattern)
class MarketBreadthAccumulator:
    """Accumulator class to store market breadth results across all batches."""
    def __init__(self):
        self.all_results = {}
        
market_breadth_calculation = MarketBreadthAccumulator()

def save_market_breadth_matrix(config, user_config, timeframe):
    """
    Save the accumulated market breadth calculations as matrix files.
    Handles single, separate, and combined universe configurations.
    Called after all batches are processed for a timeframe.
    
    Args:
        config: Config object with directory paths
        user_config: User configuration object
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
    
    Returns:
        list: List of dictionaries containing output file paths, data dates, and metadata
    """
    global market_breadth_calculation
    
    if not hasattr(market_breadth_calculation, 'all_results'):
        print(f"⚠️  No market breadth calculation results to save for {timeframe}")
        return []
        
    if timeframe not in market_breadth_calculation.all_results:
        print(f"⚠️  No {timeframe} market breadth calculation results to save")
        return []
    
    results_data = market_breadth_calculation.all_results[timeframe]
    
    if not results_data:
        print(f"⚠️  No valid market breadth calculation data for {timeframe}")
        return []
    
    # Get universe configuration
    universe_config = getattr(user_config, 'market_breadth_universe', {
        'type': 'single', 'universes': ['all'], 'display_name': 'all'
    })
    
    # Get user choice from configuration
    safe_user_choice = str(getattr(user_config, 'ticker_choice', 'unknown')).replace('/', '_')
    
    output_dir = config.directories.get('RESULTS_DIR', Path('results')) / 'market_breadth'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Process each universe's results
    for universe_name, matrix_df in results_data.items():
        try:
            if matrix_df is None or matrix_df.empty:
                print(f"⚠️  No data for universe: {universe_name}")
                continue
            
            # Create a copy for processing
            processed_df = matrix_df.copy()
            
            # Reset index to make date a column (following basic_calculations pattern)
            if processed_df.index.name == 'date' or 'date' in str(processed_df.index.name):
                processed_df = processed_df.reset_index()
            
            # Ensure date column exists and is properly formatted
            if 'date' not in processed_df.columns:
                print(f"⚠️  No date column found in results for universe: {universe_name}")
                continue
            
            # Extract data date from the matrix (latest date in the data)
            try:
                # Get the latest (maximum) date from the data
                latest_date = processed_df['date'].max()
                
                # Handle both string dates and Timestamp objects
                if isinstance(latest_date, str):
                    data_date = latest_date.replace('-', '')  # Convert 2025-08-29 to 20250829
                else:
                    data_date = latest_date.strftime('%Y%m%d')  # Handle pandas Timestamp
            except Exception as e:
                # Fallback to file generation date if date extraction fails
                data_date = datetime.now().strftime("%Y%m%d")
                print(f"⚠️  Error extracting data date for {universe_name}, using file generation date: {data_date}")
            
            # Create output filename
            safe_universe_name = universe_name.replace('/', '_').replace(' ', '_')
            output_file = output_dir / f'market_breadth_{safe_universe_name}_{safe_user_choice}_{timeframe}_{data_date}.csv'
            
            # COLUMN TRANSFORMATIONS
            # =====================
            
            # Step 1: Remove market_breadth_calculation_date column
            processed_df = processed_df.drop(columns=[f'market_breadth_calculation_date'], errors='ignore')
            
            # Step 2: Add timeframe column
            processed_df['timeframe'] = timeframe
            
            # Step 3: Shorten column prefix from 'market_breadth_' to 'mb_'
            shortened_columns = {}
            for col in processed_df.columns:
                if col.startswith('market_breadth_'):
                    shortened_columns[col] = col.replace('market_breadth_', 'mb_')
            processed_df = processed_df.rename(columns=shortened_columns)
            
            # Step 4: Add timeframe prefix to all columns except 'date' and 'timeframe'
            timeframe_prefixed_columns = {}
            for col in processed_df.columns:
                if col not in ['date', 'timeframe']:
                    timeframe_prefixed_columns[col] = f'{timeframe}_{col}'
            processed_df = processed_df.rename(columns=timeframe_prefixed_columns)
            
            # Step 5: Ensure proper column order: date, timeframe, then all others
            ordered_columns = ['date', 'timeframe'] + [col for col in processed_df.columns if col not in ['date', 'timeframe']]
            processed_df = processed_df[ordered_columns]
            
            # Save matrix to CSV
            processed_df.to_csv(output_file, index=False, float_format='%.2f')
            
            # Print results summary
            print(f"✅ Market Breadth Matrix Saved ({timeframe.upper()} DATA):")
            print(f"  • Output file: {output_file}")
            print(f"  • Universe: {universe_name}")
            print(f"  • Total trading days: {len(processed_df)}")
            print(f"  • Total indicators: {len(processed_df.columns)}")
            print(f"  • Data date: {data_date}")
            
            # Show key indicators calculated
            key_breadth_indicators = [col for col in processed_df.columns if f'{timeframe}_mb_' in col and any(
                indicator in col for indicator in ['advance', '252day_new_highs', '252day_new_lows', '20day_new_highs', '20day_new_lows', '63day_new_highs', '63day_new_lows', 'pct_above_ma']
            )]
            
            if key_breadth_indicators:
                print(f"  • Key breadth indicators: {len(key_breadth_indicators)} (advance/decline, highs/lows, MA breadth)")
            
            # Show date range
            if 'date' in processed_df.columns and len(processed_df) > 0:
                start_date = processed_df['date'].min()
                end_date = processed_df['date'].max()
                print(f"  • Date range: {start_date} to {end_date}")
            
            logger.info(f"Market breadth matrix saved: {output_file} with {len(processed_df)} rows and {len(processed_df.columns)} columns")
            
            # Add to results
            saved_files.append({
                'output_file': str(output_file),
                'data_date': data_date,
                'formatted_date': data_date[:4] + '-' + data_date[4:6] + '-' + data_date[6:8] if len(data_date) >= 8 else data_date,
                'timeframe': timeframe,
                'universe': universe_name
            })
            
        except Exception as e:
            logger.error(f"Error saving matrix for universe {universe_name}: {e}")
            print(f"❌ Error saving matrix for universe {universe_name}: {e}")
    
    # Print summary
    if saved_files:
        print(f"\n📊 Market Breadth Matrix Generation Summary ({timeframe.upper()}):")
        print(f"  • Configuration type: {universe_config['type']}")
        print(f"  • Files generated: {len(saved_files)}")
        print(f"  • Universes processed: {[f['universe'] for f in saved_files]}")
        
        if universe_config['type'] == 'separate':
            print(f"  • Separate analysis: Each universe analyzed independently")
        elif universe_config['type'] == 'combined':
            print(f"  • Combined analysis: All universes merged into single dataset")
        else:
            print(f"  • Single universe: Standard analysis (backwards compatible)")
    
    return saved_files