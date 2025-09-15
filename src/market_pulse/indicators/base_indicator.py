"""
Base Indicator Class
===================

Base class for all market pulse indicators providing common interface and utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseIndicator:
    """
    Base class for all market pulse indicators.
    
    Provides common interface and utilities for indicator calculations.
    """
    
    def __init__(self, symbol: str, config, user_config=None):
        """
        Initialize base indicator.
        
        Args:
            symbol: Market symbol to analyze (SPY, QQQ, IWM, ^DJI)
            config: System configuration object
            user_config: User configuration object (optional)
        """
        self.symbol = symbol
        self.config = config
        self.user_config = user_config
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{symbol}")
        
        # Common data paths
        self.data_paths = {
            'daily_data': self.config.get_market_data_dir('daily'),
            'basic_calculations': self.config.directories['BASIC_CALCULATION_DIR'],
            'results': self.config.directories['RESULTS_DIR']
        }
        
    def load_ticker_data(self, timeframe: str = 'daily') -> Optional[pd.DataFrame]:
        """
        Load market data for the symbol.
        
        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            data_dir = self.config.get_market_data_dir(timeframe)
                
            # Handle special case for ^DJI (might have different file naming)
            symbol_file = self.symbol.replace('^', '')
            data_file = data_dir / f"{symbol_file}.csv"
            
            if not data_file.exists():
                # Try with original symbol name
                data_file = data_dir / f"{self.symbol}.csv"
                
            if not data_file.exists():
                self.logger.warning(f"Data file not found for {self.symbol} in {timeframe}")
                return None
                
            df = pd.read_csv(data_file)
            
            # Ensure Date column is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
                df = df.sort_values('Date')
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {self.symbol}: {e}")
            return None
    
    def load_universe_data(self, universe_name: str) -> Optional[List[str]]:
        """
        Load ticker universe for filtering.
        
        Args:
            universe_name: Universe name (e.g., 'SP500', 'NASDAQ100')
            
        Returns:
            List of ticker symbols or None if not found
        """
        try:
            universe_file = self.config.directories['RESULTS_DIR'] / 'ticker_universes' / f'ticker_universe_{universe_name}.csv'
            
            if not universe_file.exists():
                self.logger.warning(f"Universe file not found: {universe_file}")
                return None
                
            universe_df = pd.read_csv(universe_file)
            
            if 'ticker' in universe_df.columns:
                return universe_df['ticker'].tolist()
            else:
                self.logger.error(f"No 'ticker' column in universe file: {universe_file}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading universe {universe_name}: {e}")
            return None
    
    def get_output_filename(self, indicator_name: str, timeframe: str, data_date: str = None) -> str:
        """
        Generate standardized output filename.
        
        Args:
            indicator_name: Name of the indicator
            timeframe: Data timeframe
            data_date: Date from dataframe (not file creation date)
            
        Returns:
            Standardized filename following convention: name_user_choice_timeframe_date.csv
        """
        if self.user_config:
            safe_user_choice = str(self.user_config.ticker_choice).replace('-', '_')
        else:
            safe_user_choice = 'unknown'
            
        date_str = data_date or datetime.now().strftime('%Y%m%d')
        
        return f"{indicator_name}_{safe_user_choice}_{timeframe}_{date_str}.csv"
    
    def run_analysis(self, timeframe: str = 'daily', data_date: str = None) -> Dict[str, Any]:
        """
        Run indicator analysis. Override in subclasses.
        
        Args:
            timeframe: Data timeframe to analyze
            data_date: Date from dataframe for output naming
            
        Returns:
            Dictionary containing analysis results
        """
        raise NotImplementedError("Subclasses must implement run_analysis method")
    
    def save_results(self, results: Dict[str, Any], output_path: Path, 
                    indicator_name: str, timeframe: str, data_date: str = None) -> Optional[str]:
        """
        Save indicator results to file.
        
        Args:
            results: Analysis results dictionary
            output_path: Output directory path
            indicator_name: Name of the indicator
            timeframe: Data timeframe
            data_date: Date from dataframe
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = self.get_output_filename(indicator_name, timeframe, data_date)
            output_file = output_path / filename
            
            # Convert results to DataFrame if it's a dict
            if isinstance(results, dict):
                # Create a simple results DataFrame
                results_df = pd.DataFrame([{
                    'symbol': self.symbol,
                    'timeframe': timeframe,
                    'analysis_date': data_date or datetime.now().strftime('%Y-%m-%d'),
                    **results
                }])
            else:
                results_df = results
                
            results_df.to_csv(output_file, index=False, float_format='%.4f')
            
            self.logger.info(f"Results saved: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return None