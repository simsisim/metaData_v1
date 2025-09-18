"""
Percentile (PER) Processing Engine
================================

Handles percentile ranking calculations for RS values across different universe configurations.
Separate from RS calculations to maintain clean file organization and support multiple universe types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PERProcessor:
    """
    Main processor for Percentile calculations across different universe configurations.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize PER processor.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object with percentile settings
        """
        self.config = config
        self.user_config = user_config
    
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
            logger.warning("No date column found in PER DataFrame, using file generation date as fallback")
            return datetime.now().strftime('%Y%m%d')
            
        except Exception as e:
            logger.error(f"Error extracting data date from PER DataFrame: {e}")
            return datetime.now().strftime('%Y%m%d')
        
    def process_per_analysis(self, ticker_choice=0):
        """
        Run complete PER analysis by reading RS files and generating percentile rankings.
        
        Args:
            ticker_choice: User ticker choice number
            
        Returns:
            Dictionary with results summary
        """
        logger.info(f"Starting PER analysis for ticker choice {ticker_choice}")
        
        # Parse benchmark tickers
        benchmark_tickers = self._parse_benchmark_tickers()
        
        results_summary = {
            'ticker_choice': ticker_choice,
            'benchmark_tickers': ';'.join(benchmark_tickers),
            'timeframes_processed': [],
            'files_created': [],
            'errors': []
        }
        
        # Determine which timeframes to process based on RS-specific config flags
        timeframes = []
        if getattr(self.user_config, 'rs_daily_enable', True):
            timeframes.append('daily')
        if getattr(self.user_config, 'rs_weekly_enable', False):
            timeframes.append('weekly')
        if getattr(self.user_config, 'rs_monthly_enable', False):
            timeframes.append('monthly')
            
        if not timeframes:
            logger.warning("No timeframes enabled for PER processing")
            return results_summary
        
        logger.info(f"Processing PER for timeframes: {timeframes}")
        
        # Process each timeframe
        for timeframe in timeframes:
            try:
                timeframe_results = self._process_timeframe(timeframe, ticker_choice, benchmark_tickers)
                results_summary['timeframes_processed'].append(timeframe)
                results_summary['files_created'].extend(timeframe_results['files_created'])
                
            except Exception as e:
                error_msg = f"Error processing {timeframe}: {e}"
                logger.error(error_msg)
                results_summary['errors'].append(error_msg)
        
        logger.info(f"PER analysis completed. Files created: {len(results_summary['files_created'])}")
        return results_summary
    
    def _parse_benchmark_tickers(self):
        """Parse benchmark tickers from configuration, supporting multiple benchmarks."""
        # Try RS_benchmark_tickers (multiple) first
        if hasattr(self.user_config, 'rs_benchmark_tickers'):
            tickers_str = str(self.user_config.rs_benchmark_tickers).strip()
            if tickers_str:
                return [t.strip() for t in tickers_str.split(';') if t.strip()]
        
        # Fallback to legacy rs_benchmark_ticker (single)
        if hasattr(self.user_config, 'rs_benchmark_ticker'):
            return [self.user_config.rs_benchmark_ticker]
        
        # Default fallback
        return ['SPY']
    
    def _process_timeframe(self, timeframe, ticker_choice, benchmark_tickers):
        """
        Process PER analysis for a specific timeframe.
        
        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            ticker_choice: User ticker choice number
            benchmark_tickers: List of benchmark tickers
            
        Returns:
            Dictionary with timeframe results
        """
        logger.info(f"Processing {timeframe} PER analysis...")
        
        timeframe_results = {
            'timeframe': timeframe,
            'files_created': []
        }
        
        # Process each asset level (stocks, sectors, industries)
        asset_levels = []
        if getattr(self.user_config, 'rs_enable_stocks', True):
            asset_levels.append('stocks')
        if getattr(self.user_config, 'rs_enable_sectors', False):
            asset_levels.append('sectors')
        if getattr(self.user_config, 'rs_enable_industries', False):
            asset_levels.append('industries')
        
        for asset_level in asset_levels:
            try:
                logger.info(f"Processing {asset_level} percentiles...")
                per_results = self._process_asset_level_per(asset_level, timeframe, ticker_choice, benchmark_tickers)
                if per_results:
                    timeframe_results['files_created'].extend(per_results)
                    
            except Exception as e:
                logger.error(f"Error processing {asset_level} PER: {e}")
        
        return timeframe_results
    
    def _process_asset_level_per(self, asset_level, timeframe, ticker_choice, benchmark_tickers):
        """
        Process percentile calculations for a specific asset level.
        
        Args:
            asset_level: Asset level ('stocks', 'sectors', 'industries')
            timeframe: Data timeframe 
            ticker_choice: User ticker choice number
            benchmark_tickers: List of benchmark tickers
            
        Returns:
            List of created file paths
        """
        # Load corresponding RS file
        rs_file_path = self._find_rs_file(asset_level, timeframe, ticker_choice)
        if not rs_file_path or not rs_file_path.exists():
            logger.warning(f"RS file not found for {asset_level} {timeframe}: {rs_file_path}")
            return []
        
        logger.info(f"Loading RS data from: {rs_file_path}")
        rs_df = pd.read_csv(rs_file_path, index_col=0)
        
        if rs_df.empty:
            logger.warning(f"Empty RS file for {asset_level} {timeframe}")
            return []
        
        # Get universe configurations for this asset level
        universe_configs = self._get_universe_configs(asset_level)
        
        # Calculate percentiles for each universe configuration
        per_results = {}
        
        for universe_config in universe_configs:
            logger.info(f"Calculating percentiles for {asset_level} universe: {universe_config}")
            
            # Get universe data based on configuration
            universe_data = self._load_universe_data(universe_config, asset_level, ticker_choice)
            if universe_data is None and universe_config.lower() != 'all':
                logger.warning(f"Could not load universe data for {universe_config}")
                continue
            
            # Calculate percentiles for all RS columns in this universe
            universe_percentiles = self._calculate_universe_percentiles(
                rs_df, universe_data, universe_config, benchmark_tickers
            )
            
            per_results[universe_config] = universe_percentiles
        
        # Combine all universe results into single DataFrame
        if per_results:
            combined_per_df = self._combine_universe_percentiles(per_results, rs_df)
            
            # Save PER results
            per_file_path = self._save_per_results(combined_per_df, asset_level, timeframe, ticker_choice)
            return [per_file_path]
        
        return []
    
    def _find_rs_file(self, asset_level, timeframe, ticker_choice):
        """Find the corresponding RS file for an asset level and timeframe."""
        rs_dir = self.config.directories['RESULTS_DIR'] / 'rs'
        
        # Look for recent RS file - try today's date first for quick lookup
        today_date_str = datetime.now().strftime('%Y%m%d')
        rs_filename = f"rs_ibd_{asset_level}_{timeframe}_{ticker_choice}_{today_date_str}.csv"
        rs_file_path = rs_dir / rs_filename
        
        if rs_file_path.exists():
            return rs_file_path
        
        # Fallback: look for any RS file matching the pattern (most recent)
        pattern = f"rs_ibd_{asset_level}_{timeframe}_{ticker_choice}_*.csv"
        matching_files = list(rs_dir.glob(pattern))
        if matching_files:
            return max(matching_files, key=lambda f: f.stat().st_mtime)
        
        # Additional fallback: look for combined ticker choice patterns (e.g., 0-5)
        combined_patterns = [
            f"rs_ibd_{asset_level}_{timeframe}_*-{ticker_choice}_*.csv",
            f"rs_ibd_{asset_level}_{timeframe}_{ticker_choice}-*_*.csv",
            f"rs_ibd_{asset_level}_{timeframe}_*{ticker_choice}*_*.csv"
        ]
        
        for pattern in combined_patterns:
            matching_files = list(rs_dir.glob(pattern))
            if matching_files:
                return max(matching_files, key=lambda f: f.stat().st_mtime)
        
        return None
    
    def _get_universe_configs(self, asset_level):
        """Get universe configurations for an asset level."""
        config_key = f'rs_percentile_universe_{asset_level}'
        universe_str = getattr(self.user_config, config_key, 'ticker_choice')
        
        if isinstance(universe_str, str) and ';' in universe_str:
            # Multiple universe configurations (e.g., "0;2")
            return [u.strip() for u in universe_str.split(';') if u.strip()]
        else:
            # Single universe configuration
            return [str(universe_str).strip()]
    
    def _load_universe_data(self, universe_config, asset_level, ticker_choice):
        """
        Load universe data from the new ticker_universe files.
        
        Args:
            universe_config: Universe configuration ('all', 'ticker_choice', 'NASDAQ100', 'SP500', etc.)
            asset_level: Asset level for context
            ticker_choice: Current ticker choice
            
        Returns:
            DataFrame with universe data or None if not available
        """
        if universe_config.lower() == 'all':
            # Use all available data - return None to indicate full universe
            logger.info(f"Universe config 'all' detected - returning None for full universe")
            return None
        
        elif universe_config.lower() == 'ticker_choice':
            # Use current ticker choice - fall back to old combined files for backward compatibility
            try:
                ticker_file = self.config.directories['TICKERS_DIR'] / f'combined_info_tickers_clean_{ticker_choice}.csv'
                logger.info(f"Loading ticker_choice universe from: {ticker_file}")
                if ticker_file.exists():
                    df = pd.read_csv(ticker_file)
                    logger.info(f"Loaded ticker_choice data: {len(df)} rows")
                    
                    # Transform the universe based on asset level
                    if asset_level == 'sectors' and 'sector' in df.columns:
                        # For sectors, extract unique sectors and create sector universe
                        unique_sectors = df['sector'].dropna().unique()
                        sector_df = pd.DataFrame({'ticker': unique_sectors})
                        logger.info(f"Transformed to {len(unique_sectors)} unique sectors")
                        return sector_df
                    elif asset_level == 'industries' and 'industry' in df.columns:
                        # For industries, extract unique industries and create industry universe  
                        unique_industries = df['industry'].dropna().unique()
                        industry_df = pd.DataFrame({'ticker': unique_industries})
                        logger.info(f"Transformed to {len(unique_industries)} unique industries")
                        return industry_df
                    else:
                        # For stocks, use as-is
                        return df
                else:
                    logger.warning(f"Ticker choice file not found: {ticker_file}")
                    return None
            except Exception as e:
                logger.error(f"Error loading ticker choice data: {e}")
                return None
        
        # Try to load from new universe files directory first
        try:
            universe_file = self.config.directories['RESULTS_DIR'] / 'ticker_universes' / f'ticker_universe_{universe_config}.csv'
            if universe_file.exists():
                logger.info(f"Loading universe data from: {universe_file}")
                df = pd.read_csv(universe_file)
                
                # Transform the universe based on asset level
                if asset_level == 'sectors' and 'sector' in df.columns:
                    # For sectors, extract unique sectors and create sector universe
                    unique_sectors = df['sector'].dropna().unique()
                    sector_df = pd.DataFrame({'ticker': unique_sectors})
                    logger.info(f"Transformed {universe_config} to {len(unique_sectors)} unique sectors")
                    return sector_df
                elif asset_level == 'industries' and 'industry' in df.columns:
                    # For industries, extract unique industries and create industry universe  
                    unique_industries = df['industry'].dropna().unique()
                    industry_df = pd.DataFrame({'ticker': unique_industries})
                    logger.info(f"Transformed {universe_config} to {len(unique_industries)} unique industries")
                    return industry_df
                else:
                    # For stocks, use as-is
                    return df
            else:
                logger.warning(f"Universe file not found: {universe_file}")
        except Exception as e:
            logger.error(f"Error loading universe data from {universe_config}: {e}")
        
        # Fallback: try old combined files for numeric universe configs (backward compatibility)
        if universe_config.isdigit():
            try:
                ticker_file = self.config.directories['TICKERS_DIR'] / f'combined_info_tickers_clean_{universe_config}.csv'
                if ticker_file.exists():
                    logger.info(f"Using fallback ticker file: {ticker_file}")
                    return pd.read_csv(ticker_file)
                else:
                    logger.warning(f"Fallback ticker file not found: {ticker_file}")
            except Exception as e:
                logger.error(f"Error loading fallback ticker data: {e}")
        
        logger.error(f"Could not load universe data for configuration: {universe_config}")
        return None
    
    def _calculate_universe_percentiles(self, rs_df, universe_data, universe_config, benchmark_tickers):
        """
        Calculate percentile rankings for RS columns within a specific universe.
        
        Args:
            rs_df: DataFrame with RS values
            universe_data: DataFrame with universe ticker data (or None for all)
            universe_config: Universe configuration name
            benchmark_tickers: List of benchmark tickers
            
        Returns:
            DataFrame with percentile columns
        """
        percentile_df = pd.DataFrame(index=rs_df.index)
        
        # Filter RS data to universe if specified
        if universe_data is not None:
            universe_tickers = set(universe_data['ticker'].tolist()) if 'ticker' in universe_data.columns else set()
            universe_mask = rs_df.index.isin(universe_tickers)
            universe_rs_df = rs_df[universe_mask]
        else:
            universe_rs_df = rs_df
        
        if universe_rs_df.empty:
            logger.warning(f"No data in universe {universe_config} after filtering")
            return percentile_df
        
        # Find all RS columns and calculate percentiles
        for col in rs_df.columns:
            if '_rs_' in col and any(benchmark in col for benchmark in benchmark_tickers):
                # Extract period and benchmark info
                rs_values = universe_rs_df[col].dropna()
                
                if len(rs_values) > 0:
                    # Calculate percentile rankings (1-99 IBD scale)
                    percentiles = rs_values.rank(pct=True, na_option='keep')
                    percentiles = (percentiles * 98) + 1
                    percentiles = percentiles.round().astype('Int64')
                    
                    # Create percentile column name  
                    per_col_name = f"{col}_per_{universe_config}"
                    
                    # Map back to full DataFrame index
                    percentile_df[per_col_name] = percentiles.reindex(rs_df.index)
        
        return percentile_df
    
    def _combine_universe_percentiles(self, per_results, rs_df):
        """
        Combine percentile results from different universes into single DataFrame.
        
        Args:
            per_results: Dict of {universe_config: percentile_DataFrame}
            rs_df: Original RS DataFrame for structure reference
            
        Returns:
            Combined DataFrame with all universe percentiles
        """
        # Start with basic structure from RS DataFrame (keep date, timeframe columns)
        combined_df = pd.DataFrame(index=rs_df.index)
        
        # Add non-RS columns from original (date, timeframe, etc.)
        # Exclude stock_return and benchmark_return columns
        for col in rs_df.columns:
            if (not '_rs_' in col and 
                not col.endswith('_percentile') and
                not '_stock_return' in col and 
                not 'benchmark_return' in col):
                combined_df[col] = rs_df[col]
        
        # Add percentile columns from all universes
        for universe_config, percentile_df in per_results.items():
            for col in percentile_df.columns:
                combined_df[col] = percentile_df[col]
        
        return combined_df
    
    def _save_per_results(self, per_df, asset_level, timeframe, ticker_choice):
        """
        Save percentile results to CSV file.
        
        Args:
            per_df: DataFrame with percentile results
            asset_level: Asset level ('stocks', 'sectors', 'industries')
            timeframe: Data timeframe
            ticker_choice: User ticker choice number
            
        Returns:
            Path to saved file
        """
        # Create output directory using user-configurable path
        per_dir = self.config.directories['PER_DIR']
        per_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename using data date from DataFrame
        date_str = self._extract_data_date_from_dataframe(per_df)
        filename = f"per_ibd_{asset_level}_{timeframe}_{ticker_choice}_{date_str}.csv"
        output_file = per_dir / filename
        
        # Save to CSV
        per_df.to_csv(output_file, index=True, float_format='%.0f')
        logger.info(f"PER results saved: {output_file}")
        
        return output_file


def run_per_analysis(config, user_config, ticker_choice=0):
    """
    Standalone function to run complete PER analysis.
    
    Args:
        config: Config object
        user_config: User configuration object
        ticker_choice: User ticker choice number
        
    Returns:
        Results summary dictionary
    """
    logger.info("Starting PER analysis pipeline...")
    
    # Check if any RS analysis is enabled (PER depends on RS)
    rs_enabled = (getattr(user_config, 'rs_enable_stocks', True) or 
                  getattr(user_config, 'rs_enable_sectors', False) or 
                  getattr(user_config, 'rs_enable_industries', False))
    
    if not rs_enabled:
        logger.info("PER analysis skipped - no RS analysis enabled")
        return {'status': 'skipped', 'reason': 'No RS analysis enabled (PER depends on RS)'}
    
    # Create PER processor and run analysis
    per_processor = PERProcessor(config, user_config)
    results = per_processor.process_per_analysis(ticker_choice)
    
    logger.info("PER analysis pipeline completed")
    return results