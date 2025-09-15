"""
Simple Universe Generator
========================

Generates ticker universe files based on tradingview_universe_bool.csv.
Creates universe files for all boolean index columns, sectors, industries, and market cap sizes.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import os
import re

logger = logging.getLogger(__name__)


class SimpleUniverseGenerator:
    """
    Simple universe generator that creates all possible universe files
    from tradingview_universe_bool.csv boolean columns and categorical data.
    """
    
    def __init__(self, config):
        """
        Initialize the simple universe generator.
        
        Args:
            config: Configuration object with directory paths
        """
        self.config = config
        self.universe_dir = config.directories['RESULTS_DIR'] / 'ticker_universes'
        self.source_file = config.directories['TICKERS_DIR'] / 'tradingview_universe_bool.csv'
        
        # Ensure output directory exists
        os.makedirs(self.universe_dir, exist_ok=True)
    
    def generate_all_universes(self) -> Dict[str, int]:
        """
        Generate all possible universe files from the boolean data.
        
        Returns:
            Dict mapping universe name to ticker count
        """
        results = {}
        
        try:
            # Load the boolean universe data
            if not self.source_file.exists():
                logger.error(f"Source file not found: {self.source_file}")
                return results
            
            logger.info(f"Loading universe data from: {self.source_file}")
            df = pd.read_csv(self.source_file)
            
            if df.empty:
                logger.warning("Universe data file is empty")
                return results
            
            logger.info(f"Loaded {len(df)} tickers from universe data")
            
            # Generate master universe file (all tickers) first
            all_results = self._generate_master_universe(df)
            results.update(all_results)
            
            # Generate index universe files from boolean columns
            index_results = self._generate_index_universes(df)
            results.update(index_results)
            
            # Generate sector universe files
            sector_results = self._generate_sector_universes(df)
            results.update(sector_results)
            
            # Generate industry universe files
            industry_results = self._generate_industry_universes(df)
            results.update(industry_results)
            
            # Generate market cap universe files
            market_cap_results = self._generate_market_cap_universes(df)
            results.update(market_cap_results)
            
            logger.info(f"Generated {len(results)} universe files")
            
        except Exception as e:
            logger.error(f"Error generating universes: {e}")
        
        return results
    
    def _generate_master_universe(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generate the master universe file containing all tickers before any filtering.
        This serves as the source of truth and audit trail for all universe generation.
        
        Args:
            df: DataFrame containing all ticker data
            
        Returns:
            Dict mapping universe name to ticker count
        """
        results = {}
        
        try:
            # Create the master universe with all tickers
            all_tickers = df['ticker'].tolist()
            universe_name = "ticker_universe_all"
            
            # Create the master universe file with all data
            self._create_universe_file(universe_name, all_tickers, df, "all_tickers")
            results[universe_name] = len(all_tickers)
            
            logger.info(f"Created master universe {universe_name}.csv with {len(all_tickers)} tickers")
            
        except Exception as e:
            logger.error(f"Error generating master universe: {e}")
        
        return results
    
    def _generate_index_universes(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generate universe files for all boolean index columns.
        
        Args:
            df: DataFrame with boolean index columns
            
        Returns:
            Dict mapping universe name to ticker count
        """
        results = {}
        
        # Identify boolean columns (exclude metadata columns)
        metadata_columns = {'ticker', 'description', 'market_cap', 'market_cap_currency', 
                          'sector', 'industry', 'exchange', 'analyst rating', 
                          'upcoming earnings date', 'recent earnings date', 'index'}
        
        boolean_columns = []
        for col in df.columns:
            if col not in metadata_columns:
                # Check if column contains boolean values
                if df[col].dtype == bool or df[col].isin([True, False, 'True', 'False']).all():
                    boolean_columns.append(col)
        
        logger.info(f"Found {len(boolean_columns)} boolean index columns")
        
        for col in boolean_columns:
            try:
                # Filter tickers where this index is True
                universe_tickers = df[df[col] == True]['ticker'].tolist()
                
                if universe_tickers:
                    # Clean column name for filename
                    clean_name = self._clean_filename(col)
                    universe_name = f"ticker_universe_{clean_name}"
                    
                    # Create universe file
                    self._create_universe_file(universe_name, universe_tickers, df, col)
                    results[universe_name] = len(universe_tickers)
                    
                    logger.info(f"Created {universe_name}.csv with {len(universe_tickers)} tickers")
                
            except Exception as e:
                logger.error(f"Error processing boolean column {col}: {e}")
        
        return results
    
    def _generate_sector_universes(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generate universe files for all sectors.
        
        Args:
            df: DataFrame with sector information
            
        Returns:
            Dict mapping universe name to ticker count
        """
        results = {}
        
        if 'sector' not in df.columns:
            logger.warning("No sector column found")
            return results
        
        # Get all unique sectors (excluding empty/null values)
        sectors = df['sector'].dropna().unique()
        logger.info(f"Found {len(sectors)} sectors")
        
        for sector in sectors:
            try:
                if pd.isna(sector) or sector == '':
                    continue
                
                # Filter tickers for this sector
                sector_tickers = df[df['sector'] == sector]['ticker'].tolist()
                
                if sector_tickers:
                    # Clean sector name for filename
                    clean_name = self._clean_filename(sector)
                    universe_name = f"ticker_universe_sectors_{clean_name}"
                    
                    # Create universe file
                    self._create_universe_file(universe_name, sector_tickers, df, f"sector={sector}")
                    results[universe_name] = len(sector_tickers)
                    
                    logger.info(f"Created {universe_name}.csv with {len(sector_tickers)} tickers")
            
            except Exception as e:
                logger.error(f"Error processing sector {sector}: {e}")
        
        return results
    
    def _generate_industry_universes(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generate universe files for all industries.
        
        Args:
            df: DataFrame with industry information
            
        Returns:
            Dict mapping universe name to ticker count
        """
        results = {}
        
        if 'industry' not in df.columns:
            logger.warning("No industry column found")
            return results
        
        # Get all unique industries (excluding empty/null values)
        industries = df['industry'].dropna().unique()
        logger.info(f"Found {len(industries)} industries")
        
        for industry in industries:
            try:
                if pd.isna(industry) or industry == '':
                    continue
                
                # Filter tickers for this industry
                industry_tickers = df[df['industry'] == industry]['ticker'].tolist()
                
                if industry_tickers:
                    # Clean industry name for filename
                    clean_name = self._clean_filename(industry)
                    universe_name = f"ticker_universe_industry_{clean_name}"
                    
                    # Create universe file
                    self._create_universe_file(universe_name, industry_tickers, df, f"industry={industry}")
                    results[universe_name] = len(industry_tickers)
                    
                    logger.info(f"Created {universe_name}.csv with {len(industry_tickers)} tickers")
            
            except Exception as e:
                logger.error(f"Error processing industry {industry}: {e}")
        
        return results
    
    def _generate_market_cap_universes(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generate universe files for different market cap sizes.
        
        Args:
            df: DataFrame with market cap information
            
        Returns:
            Dict mapping universe name to ticker count
        """
        results = {}
        
        if 'market_cap' not in df.columns:
            logger.warning("No market_cap column found")
            return results
        
        # Convert market_cap to numeric, handling any string values
        df_copy = df.copy()
        df_copy['market_cap'] = pd.to_numeric(df_copy['market_cap'], errors='coerce')
        
        # Filter out rows with invalid market cap
        valid_market_cap_df = df_copy.dropna(subset=['market_cap'])
        
        if valid_market_cap_df.empty:
            logger.warning("No valid market cap data found")
            return results
        
        logger.info(f"Found {len(valid_market_cap_df)} tickers with valid market cap data")
        
        # Define market cap categories (in USD)
        market_cap_categories = {
            'mega_cap': (200_000_000_000, float('inf')),      # $200B+
            'large_cap': (10_000_000_000, 200_000_000_000),   # $10B-$200B
            'mid_cap': (2_000_000_000, 10_000_000_000),       # $2B-$10B
            'small_cap': (300_000_000, 2_000_000_000),        # $300M-$2B
            'micro_cap': (50_000_000, 300_000_000),           # $50M-$300M
            'nano_cap': (0, 50_000_000)                       # Under $50M
        }
        
        for category, (min_cap, max_cap) in market_cap_categories.items():
            try:
                # Filter tickers for this market cap range
                if max_cap == float('inf'):
                    cap_tickers = valid_market_cap_df[valid_market_cap_df['market_cap'] >= min_cap]['ticker'].tolist()
                else:
                    cap_tickers = valid_market_cap_df[
                        (valid_market_cap_df['market_cap'] >= min_cap) & 
                        (valid_market_cap_df['market_cap'] < max_cap)
                    ]['ticker'].tolist()
                
                if cap_tickers:
                    universe_name = f"ticker_universe_market_cap_{category}"
                    
                    # Create universe file
                    self._create_universe_file(universe_name, cap_tickers, df, f"market_cap={category}")
                    results[universe_name] = len(cap_tickers)
                    
                    logger.info(f"Created {universe_name}.csv with {len(cap_tickers)} tickers")
            
            except Exception as e:
                logger.error(f"Error processing market cap category {category}: {e}")
        
        return results
    
    def _create_universe_file(self, universe_name: str, ticker_list: List[str], 
                            full_df: pd.DataFrame, filter_description: str):
        """
        Create a universe CSV file with the specified tickers.
        
        Args:
            universe_name: Name of the universe (without .csv extension)
            ticker_list: List of tickers to include
            full_df: Full DataFrame to get ticker details from
            filter_description: Description of the filter used
        """
        try:
            # Filter the full DataFrame to get complete ticker information
            universe_df = full_df[full_df['ticker'].isin(ticker_list)].copy()
            
            # Add metadata columns
            universe_df['filter_applied'] = filter_description
            universe_df['universe_name'] = universe_name
            universe_df['generation_source'] = 'simple_universe_generator'
            universe_df['generation_date'] = pd.Timestamp.now().isoformat()
            
            # Save to CSV
            output_file = self.universe_dir / f"{universe_name}.csv"
            universe_df.to_csv(output_file, index=False)
            
            logger.debug(f"Saved universe file: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating universe file {universe_name}: {e}")
    
    def _clean_filename(self, name: str) -> str:
        """
        Clean a name to be safe for use in filenames.
        
        Args:
            name: Original name
            
        Returns:
            Cleaned name safe for filenames
        """
        # Replace spaces and special characters with underscores
        cleaned = re.sub(r'[^\w\-_.]', '_', str(name))
        
        # Remove multiple consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def get_generation_summary(self, results: Dict[str, int]) -> str:
        """
        Generate a summary of the universe generation results.
        
        Args:
            results: Results dictionary from generate_all_universes
            
        Returns:
            Summary string
        """
        if not results:
            return "No universe files generated"
        
        # Count by type
        index_count = len([k for k in results.keys() if k.startswith('ticker_universe_') and not any(x in k for x in ['sectors_', 'industry_', 'market_cap_'])])
        sector_count = len([k for k in results.keys() if 'sectors_' in k])
        industry_count = len([k for k in results.keys() if 'industry_' in k])
        market_cap_count = len([k for k in results.keys() if 'market_cap_' in k])
        
        total_tickers = sum(results.values())
        
        summary = f"""Universe Generation Summary:
================================
Total files generated: {len(results)}
- Index universes: {index_count}
- Sector universes: {sector_count}
- Industry universes: {industry_count}
- Market cap universes: {market_cap_count}

Total ticker entries: {total_tickers}
Output directory: {self.universe_dir}
"""
        
        return summary