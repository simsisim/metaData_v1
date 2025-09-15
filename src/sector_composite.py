"""
Sector and Industry Composite Index Builder
==========================================

Creates composite price indices for sectors and industries using either
equal-weighted or market capitalization-weighted aggregation methods.

These composite indices serve as representative price series for entire
sectors/industries, enabling group-level Relative Strength analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SectorCompositeBuilder:
    """
    Builder for creating sector and industry composite indices from constituent stocks.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize composite builder.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object with settings
        """
        self.config = config
        self.user_config = user_config
        self.sector_map = {}
        self.industry_map = {}
        self.market_cap_data = {}
    
    def load_universe_info(self, ticker_choice=0):
        """
        Load universe information including sector and industry classifications.
        
        Args:
            ticker_choice: User ticker choice number
            
        Returns:
            DataFrame with ticker info (ticker, sector, industry, market_cap, etc.)
        """
        try:
            # Load ticker info file
            info_file = self.config.directories['TICKERS_DIR'] / f'combined_info_tickers_clean_{ticker_choice}.csv'
            
            if info_file.exists():
                info_df = pd.read_csv(info_file)
                logger.info(f"Loaded universe info: {len(info_df)} tickers")
                
                # Filter out tickers without sector/industry classification
                before_filter = len(info_df)
                info_df = info_df.dropna(subset=['sector', 'industry'])
                after_filter = len(info_df)
                
                logger.info(f"Filtered universe: {after_filter} tickers with sector/industry info "
                           f"({before_filter - after_filter} excluded)")
                
                # Create sector and industry mappings
                self._create_group_mappings(info_df)
                
                return info_df
            else:
                logger.error(f"Universe info file not found: {info_file}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading universe info: {e}")
            return pd.DataFrame()
    
    def _create_group_mappings(self, info_df):
        """
        Create mappings of sectors and industries to their constituent tickers.
        
        Args:
            info_df: DataFrame with ticker information
        """
        # Create sector mapping: {sector_name: [ticker_list]}
        self.sector_map = info_df.groupby('sector')['ticker'].apply(list).to_dict()
        
        # Create industry mapping: {industry_name: [ticker_list]}
        self.industry_map = info_df.groupby('industry')['ticker'].apply(list).to_dict()
        
        # Store market cap data if available (try different column name variations)
        market_cap_column = None
        for col_name in ['market_cap', 'market_capitalization', 'Market_Capitalization']:
            if col_name in info_df.columns:
                market_cap_column = col_name
                break
        
        if market_cap_column:
            # Convert market cap to numeric, handling any non-numeric values
            info_df[market_cap_column] = pd.to_numeric(info_df[market_cap_column], errors='coerce')
            self.market_cap_data = info_df.set_index('ticker')[market_cap_column].to_dict()
            # Remove any tickers with NaN market cap
            self.market_cap_data = {k: v for k, v in self.market_cap_data.items() if not pd.isna(v) and v > 0}
            logger.info(f"Loaded market cap data for {len(self.market_cap_data)} tickers from column '{market_cap_column}'")
        else:
            logger.warning("No market cap column found - falling back to equal weighting for all composites")
        
        logger.info(f"Created mappings: {len(self.sector_map)} sectors, {len(self.industry_map)} industries")
    
    def build_composite_indices(self, price_data, method='equal_weighted'):
        """
        Build composite indices for all sectors and industries.
        
        Args:
            price_data: DataFrame with stock price data (dates x tickers)
            method: Aggregation method ('equal_weighted' or 'market_cap_weighted')
            
        Returns:
            Dictionary with composite indices: {'sectors': {name: Series}, 'industries': {name: Series}}
        """
        logger.info(f"Building composite indices using {method} method")
        
        composite_indices = {
            'sectors': {},
            'industries': {}
        }
        
        # Build sector composites
        for sector_name, ticker_list in self.sector_map.items():
            composite = self._build_single_composite(
                price_data, ticker_list, sector_name, method
            )
            if composite is not None:
                composite_indices['sectors'][sector_name] = composite
        
        # Build industry composites  
        for industry_name, ticker_list in self.industry_map.items():
            composite = self._build_single_composite(
                price_data, ticker_list, industry_name, method
            )
            if composite is not None:
                composite_indices['industries'][industry_name] = composite
        
        logger.info(f"Built {len(composite_indices['sectors'])} sector and "
                   f"{len(composite_indices['industries'])} industry composites")
        
        return composite_indices
    
    def _build_single_composite(self, price_data, ticker_list, group_name, method):
        """
        Build a single composite index for a group of tickers.
        
        Args:
            price_data: DataFrame with price data
            ticker_list: List of tickers in this group
            group_name: Name of the group (for logging)
            method: Aggregation method
            
        Returns:
            Series with composite index values, or None if insufficient data
        """
        # Filter to available tickers
        available_tickers = [t for t in ticker_list if t in price_data.columns]
        
        # Use configurable minimum group size from user config
        min_size = getattr(self.user_config, 'rs_min_group_size', 2)
        if len(available_tickers) < min_size:
            logger.debug(f"Skipping {group_name}: insufficient tickers ({len(available_tickers)} < {min_size})")
            return None
        
        try:
            # Get price data for group tickers
            group_data = price_data[available_tickers].copy()
            
            # Remove rows where all values are NaN
            group_data = group_data.dropna(how='all')
            
            if group_data.empty:
                logger.debug(f"Skipping {group_name}: no valid price data")
                return None
            
            # Calculate composite based on method
            if method == 'equal_weighted':
                composite = self._equal_weighted_composite(group_data)
            elif method == 'market_cap_weighted':
                composite = self._market_cap_weighted_composite(group_data, available_tickers)
            else:
                logger.warning(f"Unknown method {method}, using equal_weighted")
                composite = self._equal_weighted_composite(group_data)
            
            logger.debug(f"Created {group_name} composite: {len(composite)} data points, "
                        f"{len(available_tickers)} constituents")
            
            return composite
            
        except Exception as e:
            logger.error(f"Error building composite for {group_name}: {e}")
            return None
    
    def _equal_weighted_composite(self, group_data):
        """
        Calculate equal-weighted composite index.
        
        Args:
            group_data: DataFrame with price data for group
            
        Returns:
            Series with composite index values
        """
        # Simple mean across all tickers for each date
        # Forward fill missing values before averaging
        group_data_filled = group_data.ffill()
        composite = group_data_filled.mean(axis=1)
        
        return composite
    
    def _market_cap_weighted_composite(self, group_data, ticker_list):
        """
        Calculate market capitalization-weighted composite index.
        
        Args:
            group_data: DataFrame with price data for group
            ticker_list: List of tickers in group
            
        Returns:
            Series with weighted composite index values
        """
        # Get market cap weights for available tickers
        weights = {}
        for ticker in ticker_list:
            if ticker in self.market_cap_data:
                weights[ticker] = self.market_cap_data[ticker]
            else:
                weights[ticker] = 1.0  # Equal weight fallback
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Fallback to equal weights
            weights = {ticker: 1.0/len(ticker_list) for ticker in ticker_list}
        
        # Calculate weighted composite for each date
        composite_values = []
        for date_idx in group_data.index:
            date_values = group_data.loc[date_idx]
            
            # Calculate weighted sum for this date
            weighted_sum = 0.0
            total_weight_this_date = 0.0
            
            for ticker in ticker_list:
                if ticker in date_values and not pd.isna(date_values[ticker]):
                    weight = weights.get(ticker, 0.0)
                    weighted_sum += date_values[ticker] * weight
                    total_weight_this_date += weight
            
            # Normalize by actual weights used (handle missing data)
            if total_weight_this_date > 0:
                composite_value = weighted_sum / total_weight_this_date
            else:
                composite_value = np.nan
                
            composite_values.append(composite_value)
        
        composite = pd.Series(composite_values, index=group_data.index)
        return composite
    
    def save_composite_indices(self, composite_indices, ticker_choice=0, timeframe='daily'):
        """
        Save composite indices to CSV files.
        
        Args:
            composite_indices: Dictionary with composite index data
            ticker_choice: User ticker choice number
            timeframe: Data timeframe
            
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}
        
        # Extract data date from composite indices instead of using file generation date
        data_date = None
        for group_type, indices in composite_indices.items():
            if indices:
                # Get the latest date from the first composite index
                first_composite = next(iter(indices.values()))
                if hasattr(first_composite, 'index') and len(first_composite.index) > 0:
                    latest_date = first_composite.index[-1]
                    if hasattr(latest_date, 'strftime'):
                        data_date = latest_date.strftime('%Y%m%d')
                    else:
                        # Handle string dates
                        data_date = str(latest_date).replace('-', '')[:8]
                    break
        
        # Fallback to file generation date if no data date found
        if not data_date:
            data_date = datetime.now().strftime('%Y%m%d')
            logger.warning(f"Using file generation date as fallback for {timeframe} composite indices: {data_date}")
        else:
            logger.info(f"Using data date for {timeframe} composite indices filename: {data_date}")
        
        # Create RS output directory
        rs_dir = self.config.directories['RESULTS_DIR'] / 'rs'
        rs_dir.mkdir(parents=True, exist_ok=True)
        
        for group_type, indices in composite_indices.items():
            if indices:
                # Create DataFrame with all composite indices for this group type
                composite_df = pd.DataFrame(indices)
                
                # Save to CSV
                filename = f"composite_indices_{group_type}_{ticker_choice}_{timeframe}_{data_date}.csv"
                output_file = rs_dir / filename
                composite_df.to_csv(output_file, float_format='%.4f')
                
                saved_files[group_type] = output_file
                logger.info(f"Saved {group_type} composite indices: {output_file}")
        
        return saved_files
    
    def get_group_summary(self):
        """
        Get summary statistics for sector and industry groups.
        
        Returns:
            Dictionary with group statistics
        """
        sector_stats = {}
        for sector, tickers in self.sector_map.items():
            sector_stats[sector] = len(tickers)
        
        industry_stats = {}
        for industry, tickers in self.industry_map.items():
            industry_stats[industry] = len(tickers)
        
        return {
            'sector_count': len(self.sector_map),
            'industry_count': len(self.industry_map),
            'sector_sizes': sector_stats,
            'industry_sizes': industry_stats,
            'avg_sector_size': np.mean(list(sector_stats.values())) if sector_stats else 0,
            'avg_industry_size': np.mean(list(industry_stats.values())) if industry_stats else 0
        }