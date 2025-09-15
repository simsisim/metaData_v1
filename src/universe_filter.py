"""
Universal Universe Filtering System
===================================

Provides consistent universe filtering across all modules using the same
ticker_universe files as PER/RS analysis. This ensures data consistency
and provides a reusable filtering utility for future modules.

Usage:
    from src.universe_filter import filter_dataframe_by_universe, get_universe_members
    
    # Filter DataFrame to SP500 members only
    sp500_df = filter_dataframe_by_universe(df, 'SP500', config)
    
    # Get list of universe members
    nasdaq100_tickers = get_universe_members('NASDAQ100', config)
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Set, Dict

logger = logging.getLogger(__name__)


class UniverseFilter:
    """
    Universal filtering system for ticker universes.
    Uses the same ticker_universe files as PER/RS analysis.
    """
    
    def __init__(self, config):
        """
        Initialize UniverseFilter with configuration.
        
        Args:
            config: Config object with directories
        """
        self.config = config
        self.universe_dir = config.directories['RESULTS_DIR'] / 'ticker_universes'
        self._universe_cache = {}  # Cache loaded universes for performance
        
    def get_available_universes(self) -> List[str]:
        """
        Get list of all available universe names.
        
        Returns:
            List of universe names (e.g., ['SP500', 'NASDAQ100', 'Russell1000'])
        """
        available_universes = []
        
        if not self.universe_dir.exists():
            logger.warning(f"Universe directory not found: {self.universe_dir}")
            return available_universes
            
        # Look for ticker_universe_*.csv files
        for file_path in self.universe_dir.glob('ticker_universe_*.csv'):
            # Extract universe name from filename: ticker_universe_SP500.csv -> SP500
            universe_name = file_path.stem.replace('ticker_universe_', '')
            available_universes.append(universe_name)
            
        return sorted(available_universes)
        
    def validate_universe_exists(self, universe_name: str) -> bool:
        """
        Check if universe file exists.
        
        Args:
            universe_name: Universe name (e.g., 'SP500', 'NASDAQ100')
            
        Returns:
            True if universe file exists, False otherwise
        """
        universe_file = self.universe_dir / f'ticker_universe_{universe_name}.csv'
        return universe_file.exists()
        
    def get_universe_members(self, universe_name: str) -> Set[str]:
        """
        Get set of tickers in specified universe.
        
        Args:
            universe_name: Universe name (e.g., 'SP500', 'NASDAQ100')
            
        Returns:
            Set of ticker symbols in the universe
        """
        # Check cache first
        if universe_name in self._universe_cache:
            return self._universe_cache[universe_name]
            
        universe_file = self.universe_dir / f'ticker_universe_{universe_name}.csv'
        
        if not universe_file.exists():
            logger.warning(f"Universe file not found: {universe_file}")
            return set()
            
        try:
            # Load universe file
            universe_df = pd.read_csv(universe_file)
            
            if universe_df.empty:
                logger.warning(f"Universe file is empty: {universe_file}")
                return set()
                
            # Get ticker column (should be first column or named 'ticker')
            if 'ticker' in universe_df.columns:
                tickers = set(universe_df['ticker'].dropna().astype(str))
            else:
                # Use first column if 'ticker' column not found
                tickers = set(universe_df.iloc[:, 0].dropna().astype(str))
                
            # Cache the result
            self._universe_cache[universe_name] = tickers
            
            logger.debug(f"Loaded universe {universe_name}: {len(tickers)} tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Error loading universe {universe_name}: {e}")
            return set()
            
    def filter_dataframe_by_universe(self, df: pd.DataFrame, universe_name: str, 
                                   ticker_column: str = 'ticker') -> pd.DataFrame:
        """
        Filter DataFrame to only include tickers from specified universe.
        
        Args:
            df: DataFrame with ticker data to filter
            universe_name: Universe name (e.g., 'SP500', 'NASDAQ100')
            ticker_column: Name of ticker column in df (default: 'ticker')
            
        Returns:
            Filtered DataFrame containing only universe members
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df.copy()
            
        if ticker_column not in df.columns:
            logger.error(f"Ticker column '{ticker_column}' not found in DataFrame")
            return df.copy()
            
        # Get universe members
        universe_members = self.get_universe_members(universe_name)
        
        if not universe_members:
            logger.warning(f"No members found for universe {universe_name}")
            return pd.DataFrame(columns=df.columns)  # Return empty DataFrame with same columns
            
        # Filter DataFrame
        filtered_df = df[df[ticker_column].isin(universe_members)].copy()
        
        logger.info(f"Filtered {len(df)} rows to {len(filtered_df)} rows using universe {universe_name}")
        
        return filtered_df
        
    def filter_dataframe_by_multiple_universes(self, df: pd.DataFrame, universe_names: List[str],
                                             ticker_column: str = 'ticker') -> Dict[str, pd.DataFrame]:
        """
        Filter DataFrame by multiple universes simultaneously.
        
        Args:
            df: DataFrame with ticker data to filter
            universe_names: List of universe names to filter by
            ticker_column: Name of ticker column in df
            
        Returns:
            Dictionary mapping universe_name -> filtered DataFrame
        """
        results = {}
        
        for universe_name in universe_names:
            filtered_df = self.filter_dataframe_by_universe(df, universe_name, ticker_column)
            results[universe_name] = filtered_df
            
        return results
        
    def get_universe_stats(self, universe_name: str) -> Dict[str, any]:
        """
        Get statistics about a universe.
        
        Args:
            universe_name: Universe name
            
        Returns:
            Dictionary with universe statistics
        """
        members = self.get_universe_members(universe_name)
        
        return {
            'universe_name': universe_name,
            'member_count': len(members),
            'file_exists': self.validate_universe_exists(universe_name),
            'sample_tickers': sorted(list(members))[:10] if members else []
        }


# Convenience functions for easy importing
def filter_dataframe_by_universe(df: pd.DataFrame, universe_name: str, config, 
                                ticker_column: str = 'ticker') -> pd.DataFrame:
    """
    Convenience function to filter DataFrame by universe.
    
    Args:
        df: DataFrame to filter
        universe_name: Universe name (e.g., 'SP500', 'NASDAQ100')
        config: Config object
        ticker_column: Name of ticker column
        
    Returns:
        Filtered DataFrame
    """
    filter_obj = UniverseFilter(config)
    return filter_obj.filter_dataframe_by_universe(df, universe_name, ticker_column)


def get_universe_members(universe_name: str, config) -> Set[str]:
    """
    Convenience function to get universe members.
    
    Args:
        universe_name: Universe name
        config: Config object
        
    Returns:
        Set of ticker symbols
    """
    filter_obj = UniverseFilter(config)
    return filter_obj.get_universe_members(universe_name)


def get_available_universes(config) -> List[str]:
    """
    Convenience function to get available universes.
    
    Args:
        config: Config object
        
    Returns:
        List of available universe names
    """
    filter_obj = UniverseFilter(config)
    return filter_obj.get_available_universes()


def validate_universe_exists(universe_name: str, config) -> bool:
    """
    Convenience function to validate universe exists.
    
    Args:
        universe_name: Universe name
        config: Config object
        
    Returns:
        True if universe exists
    """
    filter_obj = UniverseFilter(config)
    return filter_obj.validate_universe_exists(universe_name)