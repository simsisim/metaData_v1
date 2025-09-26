"""
Sustainability Ratios Overview Values Module
============================================

Calculates overview values for indexes, sectors, and industries.
Provides short-term gains/losses analysis for market timing.

This module:
- Loads ticker lists from configuration (direct tickers or CSV files)
- Calculates percentage changes over specified periods
- Generates overview data for Latest N periods
- Supports configurable history periods for calculations
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ...config import Config
from ...data_reader import DataReader
from ...user_defined_data import UserConfiguration

logger = logging.getLogger(__name__)


class OverviewValuesProcessor:
    """
    Processes overview values for SR analysis.
    Calculates short-term percentage changes for indexes, sectors, and industries.
    """

    def __init__(self, config: Config, user_config: UserConfiguration, timeframe: str):
        """
        Initialize overview values processor.

        Args:
            config: System configuration
            user_config: User configuration
            timeframe: Processing timeframe ('daily', 'weekly', 'monthly')
        """
        self.config = config
        self.user_config = user_config
        self.timeframe = timeframe
        self.data_reader = DataReader(config, timeframe, user_config.batch_size)
        self.results = {}

    def load_ticker_list(self, ticker_config: str, category: str) -> List[str]:
        """
        Load ticker list from configuration (direct tickers or CSV file).

        Args:
            ticker_config: Configuration string (e.g., 'SPY;IWM' or 'my_file.csv')
            category: Category name for logging ('indexes', 'sectors', 'industries')

        Returns:
            List of ticker symbols
        """
        try:
            # Check if it's a CSV file path
            if ticker_config.endswith('.csv'):
                csv_path = Path(self.config.base_dir) / ticker_config
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    if 'ticker' in df.columns:
                        tickers = df['ticker'].tolist()
                    elif len(df.columns) > 0:
                        tickers = df.iloc[:, 0].tolist()  # Use first column
                    else:
                        tickers = []
                    logger.info(f"{category}: Loaded {len(tickers)} tickers from CSV file: {csv_path}")
                    return [str(t).strip().upper() for t in tickers if pd.notna(t)]
                else:
                    logger.warning(f"{category}: CSV file not found: {csv_path}")
                    return []
            else:
                # Direct ticker list (semicolon separated)
                tickers = [t.strip().upper() for t in ticker_config.split(';') if t.strip()]
                logger.info(f"{category}: Loaded {len(tickers)} tickers from direct config: {tickers}")
                return tickers

        except Exception as e:
            logger.error(f"Error loading {category} ticker list: {e}")
            return []

    def parse_timeframe_config(self, timeframe_config: str) -> List[Dict[str, Any]]:
        """
        Parse SR_overview_values_timeframe configuration.

        Args:
            timeframe_config: Configuration string (e.g., "latest;5;latest_Wednesday")

        Returns:
            List of calculation definitions
        """
        calculations = []

        if ';' in timeframe_config:
            values = [v.strip() for v in timeframe_config.split(';') if v.strip()]
        else:
            values = [timeframe_config.strip()] if timeframe_config.strip() else []

        for value in values:
            if value == 'latest':
                calculations.append({
                    'type': 'current_value',
                    'description': 'Show current/latest value only'
                })
            elif value.startswith('latest_'):
                day_name = value.split('_')[1]
                calculations.append({
                    'type': 'day_change',
                    'day': day_name,
                    'description': f'% change since latest {day_name}'
                })
            elif value.isdigit():
                periods = int(value)
                calculations.append({
                    'type': 'period_change',
                    'periods': periods,
                    'description': f'% change since {periods} time units back'
                })
            else:
                logger.warning(f"Unknown timeframe format: {value}")
                calculations.append({
                    'type': 'unknown',
                    'value': value,
                    'description': f'UNKNOWN FORMAT: {value}'
                })

        return calculations

    def get_current_value(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get current/latest value for ticker.

        Args:
            ticker_data: DataFrame with ticker data

        Returns:
            Dictionary with current value information
        """
        if ticker_data.empty:
            return {'error': 'No data available'}

        latest_row = ticker_data.iloc[-1]
        return {
            'latest_close': round(latest_row['Close'], 2),
            'latest_date': ticker_data.index[-1].strftime('%Y-%m-%d')
        }

    def calculate_period_change(self, ticker_data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """
        Calculate percentage change over specified number of periods.

        Args:
            ticker_data: DataFrame with ticker data
            periods: Number of periods to look back

        Returns:
            Dictionary with period change information
        """
        if len(ticker_data) < periods + 1:
            return {'error': f'Insufficient data: need {periods + 1} periods, have {len(ticker_data)}'}

        # Get the required data points
        latest_close = ticker_data.iloc[-1]['Close']
        base_close = ticker_data.iloc[-(periods + 1)]['Close']

        # Calculate percentage change
        pct_change = ((latest_close - base_close) / base_close) * 100

        return {
            f'{periods}_periods_change': round(pct_change, 2),
            f'{periods}_periods_base_date': ticker_data.index[-(periods + 1)].strftime('%Y-%m-%d'),
            f'{periods}_periods_base_close': round(base_close, 2)
        }

    def calculate_day_change(self, ticker_data: pd.DataFrame, day_name: str) -> Dict[str, Any]:
        """
        Calculate percentage change between latest occurrence of specified day and previous occurrence.
        Example: latest_Wednesday compares latest Wednesday vs previous Wednesday.

        Args:
            ticker_data: DataFrame with ticker data
            day_name: Name of day (Monday, Tuesday, etc.)

        Returns:
            Dictionary with day change information
        """
        try:
            # Map day names to weekday numbers (Monday=0, Sunday=6)
            day_mapping = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }

            target_weekday = day_mapping.get(day_name.lower())
            if target_weekday is None:
                return {'error': f'Invalid day name: {day_name}'}

            # Find all occurrences of the specified day
            matching_days = []
            for i in range(len(ticker_data) - 1, -1, -1):
                date = ticker_data.index[i]
                if date.weekday() == target_weekday:
                    matching_days.append({
                        'index': i,
                        'date': date,
                        'close': ticker_data.iloc[i]['Close']
                    })

            # Need at least 2 occurrences to calculate change
            if len(matching_days) < 2:
                return {'error': f'Need at least 2 {day_name}s in data, found {len(matching_days)}'}

            # Get latest and previous occurrences
            latest_day = matching_days[0]  # Most recent (first in reversed list)
            previous_day = matching_days[1]  # Previous occurrence (second in reversed list)

            # Calculate percentage change: (latest - previous) / previous
            pct_change = ((latest_day['close'] - previous_day['close']) / previous_day['close']) * 100

            return {
                f'latest_{day_name.lower()}_change': round(pct_change, 2),
                f'latest_{day_name.lower()}_date': latest_day['date'].strftime('%Y-%m-%d'),
                f'latest_{day_name.lower()}_close': round(latest_day['close'], 2),
                f'previous_{day_name.lower()}_date': previous_day['date'].strftime('%Y-%m-%d'),
                f'previous_{day_name.lower()}_close': round(previous_day['close'], 2)
            }

        except Exception as e:
            logger.error(f"Error calculating {day_name} change: {e}")
            return {'error': f'Error calculating {day_name} change: {str(e)}'}

    def calculate_multi_value_changes(self, ticker: str, calculations: List[Dict[str, Any]], history_periods: int) -> Dict[str, Any]:
        """
        Calculate multiple values based on configuration for a single ticker.

        Args:
            ticker: Ticker symbol
            calculations: List of calculation definitions
            history_periods: Number of historical periods to load

        Returns:
            Dictionary with all calculated values and metadata
        """
        try:
            # Load ticker data
            ticker_data = self.data_reader.read_stock_data(ticker)
            if ticker_data is None or ticker_data.empty:
                logger.warning(f"No data available for {ticker}")
                return {
                    'ticker': ticker,
                    'error': 'No data available'
                }

            # Get the most recent data (limit to history_periods)
            recent_data = ticker_data.tail(history_periods).copy()

            if len(recent_data) < 2:
                logger.warning(f"Insufficient data for {ticker}: only {len(recent_data)} periods")
                return {
                    'ticker': ticker,
                    'error': f'Insufficient data: {len(recent_data)} periods'
                }

            # Initialize result dictionary
            result = {
                'ticker': ticker,
                'timeframe': self.timeframe,
                'data_points': len(recent_data)
            }

            # Process each calculation
            for calc in calculations:
                if calc['type'] == 'current_value':
                    current_values = self.get_current_value(recent_data)
                    result.update(current_values)

                elif calc['type'] == 'period_change':
                    period_values = self.calculate_period_change(recent_data, calc['periods'])
                    result.update(period_values)

                elif calc['type'] == 'day_change':
                    day_values = self.calculate_day_change(recent_data, calc['day'])
                    result.update(day_values)

                else:
                    logger.warning(f"Unknown calculation type: {calc['type']}")

            return result

        except Exception as e:
            logger.error(f"Error calculating multi-value changes for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e)
            }

    def process_category(self, category_name: str, ticker_config: str, history_periods: int, timeframe_config: str) -> Dict[str, Any]:
        """
        Process a category of tickers (indexes, sectors, industries).

        Args:
            category_name: Name of the category
            ticker_config: Configuration string for tickers
            history_periods: Historical periods to load
            timeframe_config: Timeframe configuration (e.g., "latest;5")

        Returns:
            Dictionary with category results
        """
        logger.info(f"Processing {category_name} overview values...")

        # Parse timeframe configuration
        calculations = self.parse_timeframe_config(timeframe_config)
        if not calculations:
            logger.warning(f"No valid calculations found in timeframe config: {timeframe_config}")
            return {
                'category': category_name,
                'tickers_processed': 0,
                'results': [],
                'summary': {'error': 'No valid calculations found'}
            }

        logger.info(f"Parsed {len(calculations)} calculations: {[c['description'] for c in calculations]}")

        # Load ticker list
        tickers = self.load_ticker_list(ticker_config, category_name)

        if not tickers:
            logger.warning(f"No tickers found for {category_name}")
            return {
                'category': category_name,
                'tickers_processed': 0,
                'results': [],
                'summary': {'error': 'No tickers found'}
            }

        # Process each ticker with multi-value calculations
        category_results = []
        successful_tickers = 0

        for ticker in tickers:
            logger.debug(f"Processing {ticker} for {category_name}...")
            ticker_result = self.calculate_multi_value_changes(ticker, calculations, history_periods)
            category_results.append(ticker_result)

            if 'error' not in ticker_result:
                successful_tickers += 1

        # Generate category summary
        if successful_tickers > 0:
            summary = {
                'total_tickers': len(tickers),
                'successful_tickers': successful_tickers,
                'timeframe': self.timeframe,
                'calculations': [c['description'] for c in calculations],
                'history_periods': history_periods
            }
        else:
            summary = {
                'total_tickers': len(tickers),
                'successful_tickers': 0,
                'error': 'No successful ticker processing'
            }

        logger.info(f"{category_name}: Processed {successful_tickers}/{len(tickers)} tickers successfully")

        return {
            'category': category_name,
            'tickers_processed': successful_tickers,
            'results': category_results,
            'summary': summary,
            'calculations': calculations
        }

    def run_overview_values_analysis(self) -> Dict[str, Any]:
        """
        Run complete overview values analysis for all categories.

        Returns:
            Dictionary with results for all categories
        """
        try:
            logger.info(f"Starting overview values analysis for {self.timeframe} timeframe...")

            # Get configuration parameters
            history_periods = getattr(self.user_config, 'sr_overview_values_history', 10)
            timeframe_config = getattr(self.user_config, 'sr_overview_values_timeframe', 'latest;5')

            logger.info(f"Using history_periods={history_periods}, timeframe_config='{timeframe_config}'")

            # Process each category
            results = {}

            # Indexes
            if hasattr(self.user_config, 'sr_overview_values_indexes'):
                indexes_config = self.user_config.sr_overview_values_indexes
                results['indexes'] = self.process_category('indexes', indexes_config, history_periods, timeframe_config)

            # Sectors
            if hasattr(self.user_config, 'sr_overview_values_sectors'):
                sectors_config = self.user_config.sr_overview_values_sectors
                results['sectors'] = self.process_category('sectors', sectors_config, history_periods, timeframe_config)

            # Industries
            if hasattr(self.user_config, 'sr_overview_values_industries'):
                industries_config = self.user_config.sr_overview_values_industries
                results['industries'] = self.process_category('industries', industries_config, history_periods, timeframe_config)

            # Store results
            self.results = results

            total_processed = sum(r.get('tickers_processed', 0) for r in results.values())
            logger.info(f"Overview values analysis completed: {total_processed} total tickers processed")

            return results

        except Exception as e:
            logger.error(f"Error in overview values analysis: {e}")
            return {'error': str(e)}

    def save_results_to_csv(self, output_dir: Path, filename_prefix: str) -> List[str]:
        """
        Save overview values results to CSV files with dynamic column structure.

        Args:
            output_dir: Output directory path
            filename_prefix: Filename prefix for output files

        Returns:
            List of generated file paths
        """
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            generated_files = []
            timestamp = datetime.now().strftime("%Y%m%d")

            for category, category_data in self.results.items():
                if category == 'error':
                    continue

                # Create detailed results DataFrame with dynamic columns
                rows = []
                for ticker_result in category_data['results']:
                    if 'error' in ticker_result:
                        # Create error row with basic structure
                        row = {
                            'ticker': ticker_result['ticker'],
                            'error': ticker_result['error']
                        }
                    else:
                        # Create row with all calculated values
                        row = {}
                        for key, value in ticker_result.items():
                            if key not in ['error']:  # Include all non-error keys
                                row[key] = value

                    rows.append(row)

                if rows:
                    df = pd.DataFrame(rows)

                    # Reorder columns to put ticker first, then basic info, then calculated values
                    column_order = ['ticker']

                    # Add basic columns if they exist
                    basic_columns = ['timeframe', 'data_points', 'latest_close', 'latest_date']
                    for col in basic_columns:
                        if col in df.columns:
                            column_order.append(col)

                    # Add all calculated value columns (period changes, day changes, etc.)
                    calc_columns = [col for col in df.columns if col not in column_order + ['error']]
                    calc_columns.sort()  # Sort for consistency
                    column_order.extend(calc_columns)

                    # Add error column last if it exists
                    if 'error' in df.columns:
                        column_order.append('error')

                    # Reorder DataFrame columns
                    df = df.reindex(columns=[col for col in column_order if col in df.columns])

                    # Generate filename
                    filename = f"{filename_prefix}_{category}_{self.user_config.ticker_choice}_{self.timeframe}_{timestamp}.csv"
                    filepath = output_dir / filename

                    # Save to CSV
                    df.to_csv(filepath, index=False)
                    generated_files.append(str(filepath))

                    logger.info(f"Saved {category} results to: {filepath} with {len(df.columns)} columns")
                    logger.debug(f"Columns: {list(df.columns)}")

            # Save summary file with calculation info
            summary_rows = []
            for category, category_data in self.results.items():
                if category != 'error' and 'summary' in category_data:
                    summary_data = category_data['summary'].copy()
                    summary_data['category'] = category

                    # Add calculation info to summary
                    if 'calculations' in category_data:
                        calc_descriptions = [calc['description'] for calc in category_data['calculations']]
                        summary_data['calculations'] = '; '.join(calc_descriptions)

                    summary_rows.append(summary_data)

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_filename = f"{filename_prefix}_summary_{self.user_config.ticker_choice}_{self.timeframe}_{timestamp}.csv"
                summary_filepath = output_dir / summary_filename

                summary_df.to_csv(summary_filepath, index=False)
                generated_files.append(str(summary_filepath))

                logger.info(f"Saved overview summary to: {summary_filepath}")

            return generated_files

        except Exception as e:
            logger.error(f"Error saving overview values results: {e}")
            return []