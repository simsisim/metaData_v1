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
    
    def save_results(self, results_data, level, method, timeframe, choice, date_str=None):
        """
        Save RS calculation results to CSV file.
        
        Args:
            results_data: DataFrame with results to save
            level: Analysis level ('stocks', 'sectors', 'industries')  
            method: RS method ('ibd', 'ma')
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            choice: User ticker choice number
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
        
        # Generate filename
        filename = f"rs_{method}_{level}_{timeframe}_{choice}_{date_str}.csv"
        output_file = rs_dir / filename
        
        # Save to CSV
        results_data.to_csv(output_file, index=True, float_format='%.4f')
        logger.info(f"RS results saved: {output_file}")
        
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