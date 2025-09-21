"""
Base Relative Strength (RS) Framework
====================================

Provides abstract base classes and interfaces for implementing different
RS calculation methods (IBD-style, Moving Average-style, etc.).

This modular design allows for easy extension with new RS methodologies
while maintaining consistent interfaces and shared functionality.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RSCalculatorBase(ABC):
    """
    Abstract base class for all Relative Strength calculators.
    
    Defines the common interface that all RS calculation methods must implement,
    ensuring consistency across different RS methodologies.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize RS calculator with configuration.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object with RS settings
        """
        self.config = config
        self.user_config = user_config
        self.benchmark_data = None
        self.universe_data = None
        self.ticker_universe_data = None
        self.results = {}
    
    def _extract_data_date_from_dataframe(self, df: pd.DataFrame) -> str:
        """
        Extract data date from DataFrame.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            Date string in YYYYMMDD format from the data, not file generation
        """
        try:
            if df is not None and not df.empty and 'date' in df.columns:
                # Get the date from the DataFrame data
                data_date = df['date'].iloc[0]
                if isinstance(data_date, str):
                    # Handle string dates like '2025-09-05'
                    return data_date.replace('-', '')
                else:
                    # Handle pandas Timestamp
                    return data_date.strftime('%Y%m%d')
            
            # Fallback to current date if no data found
            logger.warning("No date column found in RS DataFrame, using file generation date as fallback")
            return datetime.now().strftime('%Y%m%d')
            
        except Exception as e:
            logger.error(f"Error extracting data date from RS DataFrame: {e}")
            return datetime.now().strftime('%Y%m%d')

    def _load_universe_data(self) -> pd.DataFrame:
        """
        Load ticker universe data for column enrichment.

        Returns:
            DataFrame with universe data excluding 'description' column
        """
        try:
            if self.ticker_universe_data is not None:
                return self.ticker_universe_data

            # Construct path to ticker_universe_all.csv
            universe_path = self.config.directories['RESULTS_DIR'] / 'ticker_universes' / 'ticker_universe_all.csv'

            if not universe_path.exists():
                logger.warning(f"Ticker universe file not found: {universe_path}")
                return pd.DataFrame()

            # Load universe data
            universe_df = pd.read_csv(universe_path)

            # Exclude 'description' column as specified
            universe_columns = [col for col in universe_df.columns if col.lower() != 'description']
            self.ticker_universe_data = universe_df[universe_columns]

            logger.info(f"Loaded universe data: {len(self.ticker_universe_data)} tickers, {len(universe_columns)} columns")
            return self.ticker_universe_data

        except Exception as e:
            logger.error(f"Error loading ticker universe data: {e}")
            return pd.DataFrame()

    @abstractmethod
    def calculate_stock_rs(self, price_data, benchmark_data, periods):
        """
        Calculate RS values for individual stocks.
        
        Args:
            price_data: DataFrame with stock price data (dates x tickers)
            benchmark_data: Series with benchmark price data
            periods: List of periods to calculate RS for
            
        Returns:
            Dictionary of {period: DataFrame} with RS values
        """
        pass
    
    @abstractmethod
    def calculate_composite_rs(self, composite_indices, benchmark_data, periods):
        """
        Calculate RS values for composite indices (sectors/industries).
        
        Args:
            composite_indices: Dictionary of {group_name: Series} with composite price data
            benchmark_data: Series with benchmark price data  
            periods: List of periods to calculate RS for
            
        Returns:
            Dictionary of {period: DataFrame} with RS values for composites
        """
        pass
    
    def calculate_percentile_rankings(self, rs_data):
        """
        Convert RS values to percentile rankings.
        
        Args:
            rs_data: DataFrame with RS values
            
        Returns:
            DataFrame with percentile rankings (1-99 scale)
        """
        # Use pandas rank with percentile method
        percentiles = rs_data.rank(pct=True, axis=0, method='min')
        
        # Convert to 1-99 scale (IBD style)
        rankings = percentiles * 98 + 1
        
        return rankings.round(0).astype(int)
    
    def load_benchmark_data(self, benchmark_ticker, timeframe='daily'):
        """
        Load benchmark data for RS calculations.

        Args:
            benchmark_ticker: Ticker symbol for benchmark (e.g., 'SPY')
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')

        Returns:
            Series with benchmark price data
        """
        try:
            # Use environment-aware path resolution
            data_dir = self.config.get_market_data_dir(timeframe)
            benchmark_file = data_dir / f"{benchmark_ticker}.csv"

            if benchmark_file.exists():
                df = pd.read_csv(benchmark_file, index_col=0, parse_dates=True)
                return df['Close'].copy()
            else:
                logger.warning(f"Benchmark file not found: {benchmark_file}")
                return None

        except Exception as e:
            logger.error(f"Error loading benchmark data for {benchmark_ticker}: {e}")
            return None

    def _enhance_with_universe_data(self, rs_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge RS results with universe data columns.

        Args:
            rs_data: DataFrame with RS calculation results

        Returns:
            Enhanced DataFrame with RS columns + universe columns
        """
        try:
            # Load universe data
            universe_data = self._load_universe_data()

            if universe_data.empty:
                logger.warning("No universe data available, returning original RS data")
                return rs_data

            # CRITICAL FIX: Convert ticker from index to column if needed
            if rs_data.index.name == 'ticker':
                rs_data_with_ticker = rs_data.reset_index()
                logger.debug("Converted ticker from index to column for merge")
            else:
                rs_data_with_ticker = rs_data.copy()

            # Merge on ticker column
            enhanced_data = pd.merge(rs_data_with_ticker, universe_data, on='ticker', how='left')

            # Ensure column order: ticker, date, other RS columns, then universe columns
            rs_columns = rs_data_with_ticker.columns.tolist()
            universe_columns = [col for col in universe_data.columns if col != 'ticker']

            # Build final column order to match basic_calculation pattern
            final_columns = []

            # 1. ticker (first column)
            if 'ticker' in rs_columns:
                final_columns.append('ticker')

            # 2. date (second column - critical for basic_calculation compatibility)
            if 'date' in rs_columns:
                final_columns.append('date')

            # 3. Other RS columns (excluding ticker and date)
            other_rs_columns = [col for col in rs_columns if col not in ['ticker', 'date']]
            final_columns.extend(other_rs_columns)

            # 4. Universe columns
            final_columns.extend(universe_columns)

            enhanced_data = enhanced_data[final_columns]

            logger.info(f"Enhanced RS data: {len(rs_data.columns)} -> {len(enhanced_data.columns)} columns")
            return enhanced_data

        except Exception as e:
            logger.error(f"Error enhancing RS data with universe columns: {e}")
            logger.warning("Returning original RS data without enhancement")
            return rs_data

    def save_results(self, results_data, level, method, timeframe, choice, benchmark_ticker, date_str=None):
        """
        Save RS calculation results to CSV file with benchmark ticker in filename.

        Args:
            results_data: DataFrame with results to save
            level: Analysis level ('stocks', 'sectors', 'industries')
            method: RS method ('ibd', 'ma')
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            choice: User ticker choice number
            benchmark_ticker: Benchmark ticker symbol (e.g., 'SPY', 'QQQ')
            date_str: Optional date string, defaults to extracted data date

        Returns:
            Path to saved file
        """
        if date_str is None:
            # Extract data date from DataFrame instead of using generation date
            date_str = self._extract_data_date_from_dataframe(results_data)

        # Create output directory using user-configurable path
        rs_dir = self.config.directories['RS_DIR']
        rs_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with benchmark ticker
        filename = f"rs_{benchmark_ticker}_{method}_{level}_{timeframe}_{choice}_{date_str}.csv"
        output_file = rs_dir / filename

        # ENHANCEMENT: Add universe data columns before saving
        results_data_enhanced = self._enhance_with_universe_data(results_data)

        # Save enhanced data to CSV
        results_data_enhanced.to_csv(output_file, index=False, float_format='%.4f')
        logger.info(f"Enhanced RS results saved: {output_file} ({len(results_data_enhanced.columns)} columns)")

        return output_file
    
    def get_analysis_periods(self, timeframe):
        """
        Get comprehensive RS analysis periods for a specific timeframe using basic_calculations convention.
        
        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            
        Returns:
            List of tuples: [(period_value, period_category, column_suffix), ...]
            Example: [(1, 'daily', 'daily_daily_daily_1d'), (7, 'weekly', 'daily_daily_weekly_7d')]
        """
        if timeframe == 'daily':
            periods = []
            
            # Daily periods (from daily_daily_periods)
            daily_periods_str = getattr(self.user_config, 'daily_daily_periods', '1;3;5')
            for period in self._parse_periods(daily_periods_str):
                periods.append((period, 'daily', f'daily_daily_daily_{period}d'))
            
            # Weekly periods in days (from daily_weekly_periods)  
            weekly_periods_str = getattr(self.user_config, 'daily_weekly_periods', '7;14')
            for period in self._parse_periods(weekly_periods_str):
                periods.append((period, 'weekly', f'daily_daily_weekly_{period}d'))
            
            # Monthly periods in days (from daily_monthly_periods)
            monthly_periods_str = getattr(self.user_config, 'daily_monthly_periods', '22;44')
            for period in self._parse_periods(monthly_periods_str):
                periods.append((period, 'monthly', f'daily_daily_monthly_{period}d'))
            
            # Quarterly periods in days (from daily_quarterly_periods)
            quarterly_periods_str = getattr(self.user_config, 'daily_quarterly_periods', '66;132')
            for period in self._parse_periods(quarterly_periods_str):
                periods.append((period, 'quarterly', f'daily_daily_quarterly_{period}d'))
            
            # Yearly periods in days (from daily_yearly_periods)
            yearly_periods_str = getattr(self.user_config, 'daily_yearly_periods', '252')
            for period in self._parse_periods(yearly_periods_str):
                periods.append((period, 'yearly', f'daily_daily_yearly_{period}d'))
            
            return periods
            
        elif timeframe == 'weekly':
            periods = []
            
            # Weekly periods (from weekly_weekly_periods)
            weekly_periods_str = getattr(self.user_config, 'weekly_weekly_periods', '1;3;5;10;15')
            for period in self._parse_periods(weekly_periods_str):
                periods.append((period, 'weekly', f'weekly_weekly_weekly_{period}w'))
            
            # Monthly periods in weeks (from weekly_monthly_periods)
            monthly_periods_str = getattr(self.user_config, 'weekly_monthly_periods', '')
            for period in self._parse_periods(monthly_periods_str):
                periods.append((period, 'monthly', f'weekly_weekly_monthly_{period}w'))
            
            return periods
            
        elif timeframe == 'monthly':
            periods = []
            
            # Monthly periods (from RS_monthly_periods)
            monthly_periods_str = getattr(self.user_config, 'RS_monthly_periods', '1;3;6')
            for period in self._parse_periods(monthly_periods_str):
                periods.append((period, 'monthly', f'monthly_monthly_monthly_{period}m'))
            
            return periods
            
        else:
            # Default fallback
            return [(1, 'daily', 'daily_daily_daily_1d'), (3, 'daily', 'daily_daily_daily_3d'), (5, 'daily', 'daily_daily_daily_5d')]
    
    def _parse_periods(self, periods_str):
        """
        Parse semicolon-separated period string into list of integers.
        
        Args:
            periods_str: String like "1;3;5" or "22;44"
            
        Returns:
            List of integers
        """
        if not periods_str or periods_str.strip() == '' or str(periods_str).lower() in ['nan', 'none']:
            return []
        
        try:
            return [int(p.strip()) for p in str(periods_str).split(';') if p.strip() and p.strip().lower() not in ['nan', 'none']]
        except ValueError as e:
            logger.warning(f"Error parsing periods '{periods_str}': {e}")
            return []

    def _load_price_data(self, ticker_list, timeframe):
        """
        Load historical price data for a list of tickers.

        Args:
            ticker_list: List of ticker symbols
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')

        Returns:
            DataFrame with price data (dates x tickers)
        """
        data_dir = self.config.get_market_data_dir(timeframe)
        price_data = {}

        logger.info(f"Loading {timeframe} price data for {len(ticker_list)} tickers")

        loaded_count = 0
        for ticker in ticker_list:
            try:
                ticker_file = data_dir / f"{ticker}.csv"
                if ticker_file.exists():
                    df = pd.read_csv(ticker_file, index_col=0, parse_dates=True)
                    if 'Close' in df.columns and len(df) > 0:
                        price_data[ticker] = df['Close']
                        loaded_count += 1
            except Exception as e:
                logger.debug(f"Could not load data for {ticker}: {e}")
                continue

        logger.info(f"Successfully loaded price data for {loaded_count} tickers")

        if price_data:
            # Convert to DataFrame and forward fill missing values
            df = pd.DataFrame(price_data)
            df = df.ffill().dropna(how='all')
            return df
        else:
            return pd.DataFrame()

    def _load_benchmark_data(self, benchmark_ticker, timeframe):
        """
        Load benchmark ticker data.

        Args:
            benchmark_ticker: Benchmark ticker symbol
            timeframe: Data timeframe

        Returns:
            Series with benchmark price data
        """
        price_data = self._load_price_data([benchmark_ticker], timeframe)
        if not price_data.empty and benchmark_ticker in price_data.columns:
            return price_data[benchmark_ticker]
        else:
            logger.warning(f"Could not load benchmark data for {benchmark_ticker}")
            return pd.Series()

    def _combine_rs_periods(self, rs_results, date_index):
        """
        Combine RS results from multiple periods into single DataFrame.

        Args:
            rs_results: Dictionary of {column_suffix: DataFrame} with RS values
            date_index: Date index for the data

        Returns:
            Combined DataFrame with all RS periods
        """
        if not rs_results:
            return pd.DataFrame()

        # Combine all period results
        combined_dfs = []
        for period_name, period_df in rs_results.items():
            if period_df is not None and not period_df.empty:
                combined_dfs.append(period_df)

        if combined_dfs:
            # Concatenate along columns (different periods)
            return pd.concat(combined_dfs, axis=1)
        else:
            return pd.DataFrame()

    def _prepare_final_dataset(self, rs_data, entity_type):
        """
        Prepare final dataset with universe data and metadata.

        Args:
            rs_data: DataFrame with RS calculation results
            entity_type: Type of entity ('stocks', 'sectors', 'industries')

        Returns:
            DataFrame with universe data merged
        """
        if rs_data is None or rs_data.empty:
            return pd.DataFrame()

        try:
            # Load universe data
            universe_data = self._load_universe_data()

            if universe_data is not None and not universe_data.empty:
                # Reset index to make ticker a column for merging
                rs_data_reset = rs_data.reset_index()

                # Merge with universe data
                merged_data = pd.merge(rs_data_reset, universe_data,
                                     left_on='ticker', right_on='ticker',
                                     how='left')

                # Restore ticker as index
                merged_data = merged_data.set_index('ticker')

                return merged_data
            else:
                logger.warning("No universe data available for enrichment")
                return rs_data

        except Exception as e:
            logger.error(f"Error preparing final dataset: {e}")
            return rs_data

    def _save_results_to_file(self, data, output_path):
        """
        Save results DataFrame to CSV file.

        Args:
            data: DataFrame to save
            output_path: Path object for output file

        Returns:
            Boolean indicating success
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            data.to_csv(output_path)

            return True
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            return False


class RSResults:
    """
    Container for RS calculation results with metadata.
    """
    
    def __init__(self, method, level, timeframe):
        """
        Initialize results container.
        
        Args:
            method: RS calculation method ('ibd', 'ma')
            level: Analysis level ('stocks', 'sectors', 'industries')
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        """
        self.method = method
        self.level = level
        self.timeframe = timeframe
        self.rs_values = {}  # {period: DataFrame}
        self.percentile_rankings = {}  # {period: DataFrame}
        self.metadata = {
            'calculation_date': datetime.now().strftime('%Y-%m-%d'),  # Format as YYYY-MM-DD
            'universe_size': 0,
            'benchmark_ticker': None,
            'periods_calculated': []
        }
    
    def add_rs_data(self, period, rs_data, percentile_data=None):
        """
        Add RS data for a specific period.
        
        Args:
            period: Analysis period (e.g., 5, 20, 63)
            rs_data: DataFrame with RS values
            percentile_data: Optional DataFrame with percentile rankings
        """
        self.rs_values[period] = rs_data
        if percentile_data is not None:
            self.percentile_rankings[period] = percentile_data
        self.metadata['periods_calculated'] = list(self.rs_values.keys())
    
    def get_summary_stats(self):
        """
        Get summary statistics for the RS calculations.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.rs_values:
            return {}
            
        # Get stats from first period's data
        first_period = list(self.rs_values.keys())[0]
        sample_data = self.rs_values[first_period]
        
        return {
            'universe_size': len(sample_data),
            'periods_calculated': len(self.rs_values),
            'period_list': sorted(self.rs_values.keys()),
            'calculation_method': self.method,
            'analysis_level': self.level,
            'timeframe': self.timeframe
        }