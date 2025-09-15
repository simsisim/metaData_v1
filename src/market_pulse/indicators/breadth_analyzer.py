"""
Market Breadth Analyzer
======================

Implements market breadth analysis including Net New Highs/Lows analysis.
Uses ticker universe filtering to analyze market-wide participation and sentiment.

Features:
- Net New Highs vs New Lows analysis
- Advance/Decline line calculation
- Market breadth momentum indicators
- Sector participation analysis
- Universe-filtered breadth metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .base_indicator import BaseIndicator
from .market_breadth_visualizer import MarketBreadthVisualizer

logger = logging.getLogger(__name__)


class BreadthAnalyzer(BaseIndicator):
    """
    Analyzer for market breadth indicators using universe-filtered data.
    
    Analyzes market-wide participation rather than individual index performance.
    Uses ticker universe files for consistent filtering.
    """
    
    def __init__(self, target_indexes: List[str], config, user_config=None):
        """
        Initialize Breadth analyzer.
        
        Args:
            target_indexes: List of market indexes for context ['SPY', 'QQQ', 'IWM', '^DJI']
            config: System configuration
            user_config: User configuration (optional)
        """
        # Use first index as primary symbol for base class
        super().__init__(target_indexes[0] if target_indexes else 'SPY', config, user_config)
        
        self.target_indexes = target_indexes
        
        # Load configurable thresholds from user_config
        self.breadth_thrust_threshold = getattr(user_config, 'market_breadth_strong_ma_breadth_threshold', 80)
        self.breadth_deterioration_threshold = getattr(user_config, 'market_breadth_weak_ma_breadth_threshold', 20)
        self.new_highs_expansion_threshold = getattr(user_config, 'market_breadth_daily_252day_new_highs_threshold', 50)
        self.ad_thrust_threshold = getattr(user_config, 'market_breadth_strong_ad_ratio_threshold', 3)
        
    def run_analysis(self, timeframe: str = 'daily', data_date: str = None) -> Dict[str, Any]:
        """
        Run complete market breadth analysis using pre-calculated breadth files.
        
        Args:
            timeframe: Data timeframe ('daily' recommended for breadth)
            data_date: Date from dataframe for output naming
            
        Returns:
            Dictionary containing breadth analysis results
        """
        try:
            # Load breadth data for all configured universes
            breadth_data = self._load_breadth_data(timeframe, data_date)
            if not breadth_data:
                return {
                    'success': False,
                    'error': 'No breadth data files available',
                    'timeframe': timeframe
                }
            
            # Analyze each universe using its pre-calculated data
            universe_results = {}
            for universe_name, universe_df in breadth_data.items():
                try:
                    logger.debug(f"Processing universe {universe_name} with {len(universe_df) if not universe_df.empty else 0} rows")
                    universe_metrics = self._extract_universe_metrics(universe_df, universe_name, timeframe)
                    if universe_metrics is not None and not universe_metrics.empty:
                        universe_results[universe_name] = universe_metrics
                        logger.debug(f"Added {universe_name} to universe_results: type={type(universe_metrics)}, rows={len(universe_metrics)}")
                    else:
                        logger.warning(f"No metrics extracted for universe {universe_name}")
                except Exception as e:
                    logger.error(f"Error processing universe {universe_name}: {e}")
            
            logger.debug(f"universe_results final type: {type(universe_results)}, keys: {list(universe_results.keys()) if isinstance(universe_results, dict) else 'NOT_DICT'}")
            
            # Calculate combined market breadth across all universes (historical)
            try:  
                combined_breadth = self._calculate_combined_historical_breadth(universe_results)
                logger.debug(f"Combined breadth calculation completed: {len(combined_breadth)} rows")
            except Exception as e:
                logger.error(f"Error in _calculate_combined_historical_breadth: {e}")
                raise
            
            # Extract historical highs/lows analysis (combine data from all universes)
            try:
                highs_lows_analysis = self._extract_combined_historical_highs_lows(breadth_data, timeframe)
                logger.debug(f"Highs/lows analysis completed: {len(highs_lows_analysis)} rows")
            except Exception as e:
                logger.error(f"Error in _extract_combined_historical_highs_lows: {e}")
                raise
            
            # Extract historical advance/decline metrics (combine data from all universes)
            try:
                advance_decline = self._extract_combined_historical_advance_decline(breadth_data, timeframe)
                logger.debug(f"Advance/decline analysis completed: {len(advance_decline)} rows")
            except Exception as e:
                logger.error(f"Error in _extract_combined_historical_advance_decline: {e}")
                raise
            
            # Generate historical market breadth signals using configurable thresholds
            try:
                breadth_signals = self._generate_historical_breadth_signals(combined_breadth, highs_lows_analysis, advance_decline)
                logger.debug(f"Breadth signals generated: {len(breadth_signals)} rows")
            except Exception as e:
                logger.error(f"Error in _generate_historical_breadth_signals: {e}")
                raise
            
            # Generate historical summary
            try:
                summary = self._generate_historical_summary(universe_results, combined_breadth, breadth_signals)
                logger.debug("Historical summary generated successfully")
            except Exception as e:
                logger.error(f"Error in _generate_historical_summary: {e}")
                raise
            
            # Extract latest data date from breadth analysis (not generation date)
            latest_data_date = self._extract_latest_data_date(breadth_data)
            
            # Log historical analysis summary
            self._log_historical_analysis_summary(universe_results, combined_breadth, breadth_signals)
            
            results = {
                'success': True,
                'timeframe': timeframe,
                'analysis_date': data_date or datetime.now().strftime('%Y-%m-%d'),
                'latest_data_date': latest_data_date,
                'universe_results': universe_results,
                'combined_breadth': combined_breadth,
                'highs_lows_analysis': highs_lows_analysis,
                'advance_decline': advance_decline,
                'breadth_signals': breadth_signals,
                'summary': summary,
                'universes_analyzed': list(universe_results.keys()),
                'breadth_files_processed': list(breadth_data.keys())
            }
            
            # Save historical breadth analysis results to output file(s)
            output_files = self._save_historical_breadth_analysis(results, timeframe, latest_data_date)
            if output_files:
                results['output_files'] = output_files
                logger.info(f"Historical breadth analysis results saved to {len(output_files)} file(s): {output_files}")
                
                # Generate PNG charts for each output file
                chart_files = self._generate_breadth_charts(output_files, universe_results)
                if chart_files:
                    results['chart_files'] = chart_files
                    logger.info(f"Generated {len(chart_files)} breadth chart(s): {chart_files}")
            
            return results
            
        except Exception as e:
            import traceback
            logger.error(f"Breadth analysis failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'timeframe': timeframe
            }
    
    def _parse_universe_config(self, universe_config: str) -> List[str]:
        """
        Parse universe configuration string into list of universe names.
        
        Supports formats: 'all', 'SP500+NASDAQ100', 'SP500;NASDAQ100', etc.
        """
        if not universe_config:
            return ['all']
        
        # Handle different separators
        if '+' in universe_config:
            universes = universe_config.split('+')
        elif ';' in universe_config:
            universes = universe_config.split(';')  
        elif ',' in universe_config:
            universes = universe_config.split(',')
        else:
            universes = [universe_config]
        
        # Clean and normalize universe names to match market_breadth file naming (UPPERCASE)
        return [u.strip().upper() for u in universes if u.strip()]

    def _load_breadth_data(self, timeframe: str, data_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load market breadth data for all configured universes.
        
        Returns:
            Dict[universe_name, DataFrame] - One DataFrame per universe
        """
        try:
            # Get universe configuration from user_data.csv
            universe_config_obj = getattr(self.user_config, 'market_breadth_universe', {'raw_config': 'all'})
            
            # Extract the raw configuration string or use universes list
            if isinstance(universe_config_obj, dict):
                if 'raw_config' in universe_config_obj:
                    universe_config = universe_config_obj['raw_config']
                elif 'universes' in universe_config_obj:
                    universe_config = ';'.join(universe_config_obj['universes'])
                else:
                    universe_config = 'all'
            else:
                universe_config = str(universe_config_obj) if universe_config_obj else 'all'
            
            # Parse universe configuration
            universes = self._parse_universe_config(universe_config)
            
            # Get ticker choice for file naming
            if self.user_config:
                ticker_choice = str(self.user_config.ticker_choice).replace('-', '-')  # Keep original format
            else:
                ticker_choice = '0-5'  # Default fallback
            
            breadth_data = {}
            
            # Loop through each universe
            for universe in universes:
                # Construct file path
                filename = f"market_breadth_{universe}_{ticker_choice}_{timeframe}_{data_date}.csv"
                file_path = self.config.directories['RESULTS_DIR'] / 'market_breadth' / filename
                
                # Load universe breadth file
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    breadth_data[universe.upper()] = df
                    logger.info(f"Loaded breadth data for {universe.upper()}: {len(df)} records from {filename}")
                else:
                    logger.warning(f"Breadth file not found: {file_path}")
                    # Try to find latest file if date not specified
                    if not data_date:
                        pattern = f"market_breadth_{universe}_{ticker_choice}_{timeframe}_*.csv"
                        matching_files = list((self.config.directories['RESULTS_DIR'] / 'market_breadth').glob(pattern))
                        if matching_files:
                            latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                            df = pd.read_csv(latest_file)
                            breadth_data[universe.upper()] = df
                            logger.info(f"Loaded latest breadth data for {universe.upper()}: {latest_file.name}")
            
            return breadth_data
            
        except Exception as e:
            logger.error(f"Error loading breadth data: {e}")
            return {}
    
    def _extract_universe_metrics(self, universe_df: pd.DataFrame, universe_name: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Extract historical breadth metrics from pre-calculated universe breadth file.
        
        Args:
            universe_df: DataFrame from market_breadth_{universe}_{choice}_{timeframe}_{date}.csv
            universe_name: Universe name (ALL, SP500, NASDAQ100, etc.)
            timeframe: Data timeframe for column prefix
            
        Returns:
            DataFrame with historical universe breadth metrics
        """
        try:
            # Process entire historical DataFrame instead of latest row only
            if universe_df.empty:
                logger.warning(f"Empty breadth data for universe {universe_name}")
                return None
                
            # Extract pre-calculated metrics based on timeframe
            timeframe_prefix = f"{timeframe}_mb_"  # e.g., "daily_mb_"
            
            # Create historical metrics DataFrame
            historical_metrics = pd.DataFrame(index=universe_df.index)
            historical_metrics['date'] = universe_df['date']
            historical_metrics['universe'] = universe_name
            
            # Extract all historical data for key breadth metrics
            historical_metrics['total_members'] = universe_df[f'{timeframe_prefix}total_stocks']
            historical_metrics['advancing_stocks'] = universe_df[f'{timeframe_prefix}advancing_stocks']
            historical_metrics['declining_stocks'] = universe_df[f'{timeframe_prefix}declining_stocks']
            historical_metrics['unchanged_stocks'] = universe_df[f'{timeframe_prefix}unchanged_stocks']
            historical_metrics['advance_decline_ratio'] = universe_df[f'{timeframe_prefix}ad_ratio']
            historical_metrics['pct_advancing'] = universe_df[f'{timeframe_prefix}advance_pct']
            historical_metrics['pct_declining'] = universe_df[f'{timeframe_prefix}decline_pct']
            historical_metrics['pct_above_ma20'] = universe_df[f'{timeframe_prefix}pct_above_ma_20']
            historical_metrics['pct_above_ma50'] = universe_df[f'{timeframe_prefix}pct_above_ma_50']
            historical_metrics['pct_above_ma200'] = universe_df[f'{timeframe_prefix}pct_above_ma_200']
            historical_metrics['pct_at_20day_high'] = universe_df[f'{timeframe_prefix}20day_new_highs_pct']
            historical_metrics['pct_at_52week_high'] = universe_df[f'{timeframe_prefix}252day_new_highs_pct']
            historical_metrics['net_20day_highs'] = universe_df[f'{timeframe_prefix}net_20day_new_highs']
            historical_metrics['net_52week_highs'] = universe_df[f'{timeframe_prefix}net_252day_new_highs']
            historical_metrics['net_advances'] = universe_df[f'{timeframe_prefix}net_advances']
            
            # Calculate historical breadth momentum for each date
            historical_metrics['breadth_momentum'] = historical_metrics.apply(
                lambda row: self._calculate_breadth_momentum(row['pct_above_ma20'], row['pct_above_ma50']), axis=1
            )
            
            # Add new highs/lows data
            historical_metrics['new_20day_highs'] = universe_df[f'{timeframe_prefix}20day_new_highs']
            historical_metrics['new_20day_lows'] = universe_df[f'{timeframe_prefix}20day_new_lows']
            historical_metrics['new_52week_highs'] = universe_df[f'{timeframe_prefix}252day_new_highs']
            historical_metrics['new_52week_lows'] = universe_df[f'{timeframe_prefix}252day_new_lows']
            
            # Add threshold indicators if they exist
            threshold_cols = [col for col in universe_df.columns if '_gt_' in col or '_lt_' in col or 'successful' in col]
            for col in threshold_cols:
                # Remove timeframe prefix for cleaner column names
                clean_col = col.replace(f'{timeframe_prefix}', '')
                historical_metrics[clean_col] = universe_df[col]
            
            logger.info(f"Extracted historical metrics for {universe_name}: {len(historical_metrics)} dates, {len(historical_metrics.columns)} indicators")
            
            return historical_metrics
            
        except Exception as e:
            logger.error(f"Error extracting historical metrics for universe {universe_name}: {e}")
            return None
    
    def _calculate_breadth_momentum(self, pct_above_ma20: float, pct_above_ma50: float) -> str:
        """Calculate breadth momentum based on MA percentages."""
        avg_breadth = (pct_above_ma20 + pct_above_ma50) / 2
        
        if avg_breadth >= 80:
            return 'Very Strong'
        elif avg_breadth >= 60:
            return 'Strong'
        elif avg_breadth >= 40:
            return 'Moderate'
        elif avg_breadth >= 20:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _calculate_combined_historical_breadth(self, universe_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate combined historical market breadth across all universes."""
        logger.debug(f"_calculate_combined_historical_breadth called with type: {type(universe_results)}")
        
        if not universe_results:
            return pd.DataFrame()
        
        # Handle case where universe_results might be a list instead of dict
        if isinstance(universe_results, list):
            logger.error(f"CRITICAL ERROR: universe_results is a list instead of dict!")
            logger.error(f"List length: {len(universe_results)}")
            logger.error(f"List contents types: {[type(x) for x in universe_results[:3]]}")
            logger.error(f"List content sample: {str(universe_results)[:500]}")
            
            # Convert list to dict if possible
            if universe_results:
                try:
                    # Check if list contains DataFrames
                    if hasattr(universe_results[0], 'empty'):
                        original_type = type(universe_results)
                        universe_results = {f'universe_{i}': df for i, df in enumerate(universe_results)}
                        logger.error(f"CONVERTED from {original_type} to {type(universe_results)} with {len(universe_results)} universes")
                        # Verify conversion worked
                        if not isinstance(universe_results, dict):
                            logger.error(f"CONVERSION FAILED! Still type: {type(universe_results)}")
                            return pd.DataFrame()
                    else:
                        logger.error(f"List contains {type(universe_results[0])}, expected DataFrame")
                        return pd.DataFrame()
                except (IndexError, AttributeError) as e:
                    logger.error(f"Cannot convert list to dict format: {e}")
                    return pd.DataFrame()
            else:
                logger.error("Empty list provided for universe_results")
                return pd.DataFrame()
        
        # Ensure universe_results is a dictionary
        if not isinstance(universe_results, dict):
            logger.error(f"universe_results is {type(universe_results)}, expected dict")
            return pd.DataFrame()
        
        # Get all unique dates from all universes
        all_dates = set()
        # Final safety check before using .values()
        if not isinstance(universe_results, dict):
            logger.error(f"CRITICAL: universe_results is {type(universe_results)} in _calculate_combined_historical_breadth line 356")
            logger.error(f"Content sample: {str(universe_results)[:200]}...")
            return pd.DataFrame()
        
        for universe_df in universe_results.values():
            if hasattr(universe_df, 'empty') and not universe_df.empty:
                all_dates.update(universe_df['date'].unique())
        all_dates = sorted(list(all_dates))
        
        # Create combined DataFrame with all dates
        combined_df = pd.DataFrame({'date': all_dates})
        
        # Weight universes by size (SP500 gets highest weight)
        weights = {'SP500': 0.5, 'NASDAQ100': 0.3, 'RUSSELL1000': 0.2}
        
        # Calculate weighted metrics for each date
        combined_df['combined_pct_above_ma20'] = 0.0
        combined_df['combined_pct_above_ma50'] = 0.0
        combined_df['combined_pct_above_ma200'] = 0.0
        combined_df['combined_advance_decline_ratio'] = 1.0
        
        for date in all_dates:
            weighted_metrics = {}
            total_weight = 0
            
            for universe, universe_df in universe_results.items():
                # Get data for this specific date
                date_data = universe_df[universe_df['date'] == date]
                if date_data.empty:
                    continue
                    
                date_row = date_data.iloc[0]
                weight = weights.get(universe, 0.1)  # Default small weight for unknown universes
                total_weight += weight
                
                for metric in ['pct_above_ma20', 'pct_above_ma50', 'pct_above_ma200', 'advance_decline_ratio']:
                    if metric not in weighted_metrics:
                        weighted_metrics[metric] = 0
                    weighted_metrics[metric] += date_row.get(metric, 0) * weight
            
            # Normalize by total weight
            if total_weight > 0:
                for metric in weighted_metrics:
                    weighted_metrics[metric] /= total_weight
            
            # Store weighted results for this date
            date_idx = combined_df[combined_df['date'] == date].index[0]
            combined_df.loc[date_idx, 'combined_pct_above_ma20'] = weighted_metrics.get('pct_above_ma20', 0)
            combined_df.loc[date_idx, 'combined_pct_above_ma50'] = weighted_metrics.get('pct_above_ma50', 0)
            combined_df.loc[date_idx, 'combined_pct_above_ma200'] = weighted_metrics.get('pct_above_ma200', 0)
            combined_df.loc[date_idx, 'combined_advance_decline_ratio'] = weighted_metrics.get('advance_decline_ratio', 1)
        
        # Calculate overall breadth score for each date
        combined_df['overall_breadth_score'] = (
            combined_df['combined_pct_above_ma20'] * 0.3 +
            combined_df['combined_pct_above_ma50'] * 0.4 +
            combined_df['combined_pct_above_ma200'] * 0.3
        )
        
        # Calculate breadth rating for each date
        combined_df['breadth_rating'] = combined_df['overall_breadth_score'].apply(self._rate_breadth_score)
        
        # Add metadata
        combined_df['universes_included'] = ';'.join(list(universe_results.keys()))
        
        return combined_df
    
    def _rate_breadth_score(self, score: float) -> str:
        """Rate overall breadth score."""
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Neutral'
        elif score >= 20:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _extract_combined_historical_highs_lows(self, breadth_data: Dict[str, pd.DataFrame], timeframe: str) -> pd.DataFrame:
        """
        Extract combined historical new highs vs new lows analysis from all universe breadth files.
        
        Args:
            breadth_data: Dictionary of universe DataFrames
            timeframe: Data timeframe for column prefix
            
        Returns:
            DataFrame with historical combined highs/lows metrics
        """
        try:
            timeframe_prefix = f"{timeframe}_mb_"
            
            # Handle case where breadth_data might be a list instead of dict
            if isinstance(breadth_data, list):
                logger.warning("breadth_data is a list instead of dict, converting...")
                # Convert list to dict if possible
                if breadth_data and hasattr(breadth_data[0], 'empty'):
                    breadth_data = {f'universe_{i}': df for i, df in enumerate(breadth_data)}
                else:
                    logger.error("Cannot convert breadth_data list to dict format")
                    return pd.DataFrame()
            
            # Ensure breadth_data is a dictionary
            if not isinstance(breadth_data, dict):
                logger.error(f"breadth_data is {type(breadth_data)}, expected dict")
                return pd.DataFrame()
            
            # Get all unique dates from all universes
            all_dates = set()
            # Final safety check before using breadth_data.values() in highs_lows
            if not isinstance(breadth_data, dict):
                logger.error(f"CRITICAL: breadth_data is {type(breadth_data)} in _extract_combined_historical_highs_lows line 501")
                logger.error(f"Content sample: {str(breadth_data)[:200]}...")
                return pd.DataFrame()
            
            for universe_df in breadth_data.values():
                all_dates.update(universe_df['date'].unique())
            all_dates = sorted(list(all_dates))
            
            # Create historical highs/lows DataFrame
            highs_lows_df = pd.DataFrame({'date': all_dates})
            
            # Calculate combined metrics for each date
            for date in all_dates:
                total_20day_highs = 0
                total_20day_lows = 0
                total_52week_highs = 0
                total_52week_lows = 0
                total_stocks = 0
                
                # Aggregate across all universes for this date
                for universe_name, universe_df in breadth_data.items():
                    if universe_df.empty:
                        continue
                        
                    date_data = universe_df[universe_df['date'] == date]
                    if date_data.empty:
                        continue
                        
                    date_row = date_data.iloc[0]
                    
                    total_20day_highs += date_row[f'{timeframe_prefix}20day_new_highs']
                    total_20day_lows += date_row[f'{timeframe_prefix}20day_new_lows']
                    total_52week_highs += date_row[f'{timeframe_prefix}252day_new_highs']
                    total_52week_lows += date_row[f'{timeframe_prefix}252day_new_lows']
                    total_stocks += date_row[f'{timeframe_prefix}total_stocks']
                
                # Store results for this date
                date_idx = highs_lows_df[highs_lows_df['date'] == date].index[0]
                highs_lows_df.loc[date_idx, 'new_20day_highs'] = total_20day_highs
                highs_lows_df.loc[date_idx, 'new_20day_lows'] = total_20day_lows
                highs_lows_df.loc[date_idx, 'new_52week_highs'] = total_52week_highs
                highs_lows_df.loc[date_idx, 'new_52week_lows'] = total_52week_lows
                highs_lows_df.loc[date_idx, 'net_20day_highs'] = total_20day_highs - total_20day_lows
                highs_lows_df.loc[date_idx, 'net_52week_highs'] = total_52week_highs - total_52week_lows
                highs_lows_df.loc[date_idx, 'highs_lows_ratio_20d'] = total_20day_highs / max(total_20day_lows, 1)
                highs_lows_df.loc[date_idx, 'highs_lows_ratio_52w'] = total_52week_highs / max(total_52week_lows, 1)
                highs_lows_df.loc[date_idx, 'pct_at_20day_highs'] = (total_20day_highs / total_stocks * 100) if total_stocks > 0 else 0
                highs_lows_df.loc[date_idx, 'pct_at_52week_highs'] = (total_52week_highs / total_stocks * 100) if total_stocks > 0 else 0
                highs_lows_df.loc[date_idx, 'total_stocks_analyzed'] = total_stocks
            
            return highs_lows_df
            
        except Exception as e:
            logger.error(f"Error extracting combined historical highs/lows: {e}")
            return pd.DataFrame()
    
    def _extract_combined_historical_advance_decline(self, breadth_data: Dict[str, pd.DataFrame], timeframe: str) -> pd.DataFrame:
        """
        Extract combined historical advance/decline metrics from all universe breadth files.
        
        Args:
            breadth_data: Dictionary of universe DataFrames
            timeframe: Data timeframe for column prefix
            
        Returns:
            DataFrame with historical combined advance/decline metrics
        """
        try:
            timeframe_prefix = f"{timeframe}_mb_"
            
            # Handle case where breadth_data might be a list instead of dict
            if isinstance(breadth_data, list):
                logger.warning("breadth_data is a list instead of dict, converting...")
                # Convert list to dict if possible
                if breadth_data and hasattr(breadth_data[0], 'empty'):
                    breadth_data = {f'universe_{i}': df for i, df in enumerate(breadth_data)}
                else:
                    logger.error("Cannot convert breadth_data list to dict format")
                    return pd.DataFrame()
            
            # Ensure breadth_data is a dictionary
            if not isinstance(breadth_data, dict):
                logger.error(f"breadth_data is {type(breadth_data)}, expected dict")
                return pd.DataFrame()
            
            # Get all unique dates from all universes
            all_dates = set()
            # Final safety check before using breadth_data.values() in advance_decline
            if not isinstance(breadth_data, dict):
                logger.error(f"CRITICAL: breadth_data is {type(breadth_data)} in _extract_combined_historical_advance_decline line 584")
                logger.error(f"Content sample: {str(breadth_data)[:200]}...")
                return pd.DataFrame()
            
            for universe_df in breadth_data.values():
                all_dates.update(universe_df['date'].unique())
            all_dates = sorted(list(all_dates))
            
            # Create historical advance/decline DataFrame
            ad_df = pd.DataFrame({'date': all_dates})
            
            # Calculate combined metrics for each date
            for date in all_dates:
                total_advancing = 0
                total_declining = 0
                total_unchanged = 0
                total_stocks = 0
                
                # Aggregate across all universes for this date
                for universe_name, universe_df in breadth_data.items():
                    if universe_df.empty:
                        continue
                        
                    date_data = universe_df[universe_df['date'] == date]
                    if date_data.empty:
                        continue
                        
                    date_row = date_data.iloc[0]
                    
                    total_advancing += date_row[f'{timeframe_prefix}advancing_stocks']
                    total_declining += date_row[f'{timeframe_prefix}declining_stocks']
                    total_unchanged += date_row[f'{timeframe_prefix}unchanged_stocks']
                    total_stocks += date_row[f'{timeframe_prefix}total_stocks']
                
                # Store results for this date
                date_idx = ad_df[ad_df['date'] == date].index[0]
                ad_df.loc[date_idx, 'advancing_stocks'] = total_advancing
                ad_df.loc[date_idx, 'declining_stocks'] = total_declining
                ad_df.loc[date_idx, 'unchanged_stocks'] = total_unchanged
                ad_df.loc[date_idx, 'total_stocks'] = total_stocks
                ad_df.loc[date_idx, 'advance_decline_ratio'] = total_advancing / max(total_declining, 1)
                ad_df.loc[date_idx, 'pct_advancing'] = (total_advancing / total_stocks * 100) if total_stocks > 0 else 0
                ad_df.loc[date_idx, 'pct_declining'] = (total_declining / total_stocks * 100) if total_stocks > 0 else 0
                ad_df.loc[date_idx, 'net_advances'] = total_advancing - total_declining
                ad_df.loc[date_idx, 'market_participation'] = ((total_advancing + total_declining) / total_stocks * 100) if total_stocks > 0 else 0
            
            return ad_df
            
        except Exception as e:
            logger.error(f"Error extracting combined historical advance/decline: {e}")
            return pd.DataFrame()
    
    def _generate_historical_breadth_signals(self, combined_breadth: pd.DataFrame, highs_lows: pd.DataFrame, advance_decline: pd.DataFrame) -> pd.DataFrame:
        """Generate historical market breadth signals and alerts using configurable thresholds."""
        try:
            if combined_breadth.empty:
                return pd.DataFrame()
            
            # Create signals DataFrame with same dates as combined_breadth
            signals_df = pd.DataFrame({'date': combined_breadth['date']})
            
            # Initialize signal columns
            signals_df['breadth_thrust'] = 0
            signals_df['breadth_deterioration'] = 0
            signals_df['new_highs_expansion'] = 0
            signals_df['new_lows_expansion'] = 0
            signals_df['ad_thrust'] = 0
            signals_df['total_bullish_signals'] = 0
            signals_df['total_bearish_signals'] = 0
            signals_df['net_signal_score'] = 0
            
            # Process each date for historical signal generation
            for idx, date in enumerate(combined_breadth['date']):
                bullish_count = 0
                bearish_count = 0
                
                # Get data for this date from each DataFrame
                cb_row = combined_breadth[combined_breadth['date'] == date]
                hl_row = highs_lows[highs_lows['date'] == date] if not highs_lows.empty else pd.DataFrame()
                ad_row = advance_decline[advance_decline['date'] == date] if not advance_decline.empty else pd.DataFrame()
                
                if cb_row.empty:
                    continue
                    
                cb_data = cb_row.iloc[0]
                hl_data = hl_row.iloc[0] if not hl_row.empty else {}
                ad_data = ad_row.iloc[0] if not ad_row.empty else {}
                
                # Breadth thrust signal (using configurable threshold)
                if cb_data.get('overall_breadth_score', 0) > self.breadth_thrust_threshold:
                    signals_df.loc[signals_df['date'] == date, 'breadth_thrust'] = 1
                    bullish_count += 1
                
                # Breadth deterioration (using configurable threshold)
                elif cb_data.get('overall_breadth_score', 0) < self.breadth_deterioration_threshold:
                    signals_df.loc[signals_df['date'] == date, 'breadth_deterioration'] = 1
                    bearish_count += 1
                
                # New highs expansion (using configurable threshold)
                if hl_data.get('net_52week_highs', 0) > self.new_highs_expansion_threshold:
                    signals_df.loc[signals_df['date'] == date, 'new_highs_expansion'] = 1
                    bullish_count += 1
                
                # New lows expansion
                elif hl_data.get('net_52week_highs', 0) < -self.new_highs_expansion_threshold:
                    signals_df.loc[signals_df['date'] == date, 'new_lows_expansion'] = 1
                    bearish_count += 1
                
                # Advance/Decline thrust (using configurable threshold)
                if ad_data.get('advance_decline_ratio', 1) > self.ad_thrust_threshold:
                    signals_df.loc[signals_df['date'] == date, 'ad_thrust'] = 1
                    bullish_count += 1
                
                # Store signal counts for this date
                signals_df.loc[signals_df['date'] == date, 'total_bullish_signals'] = bullish_count
                signals_df.loc[signals_df['date'] == date, 'total_bearish_signals'] = bearish_count
                signals_df.loc[signals_df['date'] == date, 'net_signal_score'] = bullish_count - bearish_count
            
            # Add trend analysis - signal momentum over rolling periods
            self._add_signal_trend_analysis(signals_df)
            
            return signals_df
            
        except Exception as e:
            logger.error(f"Error generating historical breadth signals: {e}")
            return pd.DataFrame()
    
    def _add_signal_trend_analysis(self, signals_df: pd.DataFrame) -> None:
        """
        Add trend analysis to historical signals DataFrame.
        
        Args:
            signals_df: DataFrame with historical breadth signals
        """
        try:
            # Calculate rolling signal momentum (5-day, 10-day, 20-day windows)
            for window in [5, 10, 20]:
                # Rolling average of bullish signals
                signals_df[f'bullish_momentum_{window}d'] = signals_df['total_bullish_signals'].rolling(
                    window=window, min_periods=window//2
                ).mean().round(2)
                
                # Rolling average of bearish signals
                signals_df[f'bearish_momentum_{window}d'] = signals_df['total_bearish_signals'].rolling(
                    window=window, min_periods=window//2
                ).mean().round(2)
                
                # Rolling net signal score
                signals_df[f'net_momentum_{window}d'] = signals_df['net_signal_score'].rolling(
                    window=window, min_periods=window//2
                ).mean().round(2)
            
            # Signal persistence - count consecutive days with same signal type
            signals_df['consecutive_bullish_days'] = self._calculate_consecutive_signals(
                signals_df['total_bullish_signals'] > signals_df['total_bearish_signals']
            )
            signals_df['consecutive_bearish_days'] = self._calculate_consecutive_signals(
                signals_df['total_bearish_signals'] > signals_df['total_bullish_signals']
            )
            
            # Signal strength classification
            signals_df['signal_strength'] = signals_df.apply(self._classify_signal_strength, axis=1)
            
        except Exception as e:
            logger.error(f"Error adding signal trend analysis: {e}")
    
    def _calculate_consecutive_signals(self, condition_series: pd.Series) -> pd.Series:
        """
        Calculate consecutive days where a condition is True.
        
        Args:
            condition_series: Boolean series indicating condition
            
        Returns:
            Series with consecutive count for each date
        """
        # Create groups for consecutive True values
        groups = (condition_series != condition_series.shift()).cumsum()
        
        # Count consecutive occurrences within each group
        consecutive_counts = condition_series.groupby(groups).cumsum()
        
        # Set count to 0 where condition is False
        consecutive_counts[~condition_series] = 0
        
        return consecutive_counts
    
    def _classify_signal_strength(self, row: pd.Series) -> str:
        """
        Classify signal strength based on multiple factors.
        
        Args:
            row: Row from signals DataFrame
            
        Returns:
            String classification of signal strength
        """
        bullish = row['total_bullish_signals']
        bearish = row['total_bearish_signals']
        net_score = row['net_signal_score']
        
        # Strong signals: 3+ signals of same type with net score >= 2
        if bullish >= 3 and net_score >= 2:
            return 'Strong Bullish'
        elif bearish >= 3 and net_score <= -2:
            return 'Strong Bearish'
        
        # Moderate signals: 2+ signals with net score >= 1
        elif bullish >= 2 and net_score >= 1:
            return 'Moderate Bullish'
        elif bearish >= 2 and net_score <= -1:
            return 'Moderate Bearish'
        
        # Weak signals: Any signal but low net score
        elif bullish > 0 or bearish > 0:
            return 'Weak Signal'
        
        # No signals
        else:
            return 'Neutral'
    
    def _log_historical_analysis_summary(self, universe_results: Dict, combined_breadth: pd.DataFrame, breadth_signals: pd.DataFrame) -> None:
        """
        Log summary of historical analysis for debugging and monitoring.
        
        Args:
            universe_results: Dictionary of universe DataFrames
            combined_breadth: Combined historical breadth DataFrame
            breadth_signals: Historical signals DataFrame
        """
        try:
            total_universes = len(universe_results)
            total_days = len(combined_breadth) if not combined_breadth.empty else 0
            signal_days = len(breadth_signals) if not breadth_signals.empty else 0
            
            logger.info(f"Historical Breadth Analysis Summary:")
            logger.info(f"  • Universes processed: {total_universes}")
            logger.info(f"  • Historical trading days: {total_days}")
            logger.info(f"  • Signal analysis days: {signal_days}")
            
            if not combined_breadth.empty:
                date_range = f"{combined_breadth['date'].min()} to {combined_breadth['date'].max()}"
                avg_breadth = combined_breadth['overall_breadth_score'].mean()
                logger.info(f"  • Date range: {date_range}")
                logger.info(f"  • Average breadth score: {avg_breadth:.1f}")
            
            if not breadth_signals.empty and 'signal_strength' in breadth_signals.columns:
                signal_distribution = breadth_signals['signal_strength'].value_counts().to_dict()
                logger.info(f"  • Signal distribution: {signal_distribution}")
                
            # Log universe data quality
            for universe, df in universe_results.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info(f"  • {universe}: {len(df)} days of data")
                    
        except Exception as e:
            logger.error(f"Error logging historical analysis summary: {e}")
    
    def _extract_latest_data_date(self, breadth_data: Dict[str, pd.DataFrame]) -> str:
        """
        Extract the latest data date from breadth analysis files.
        
        Args:
            breadth_data: Dictionary of universe DataFrames
            
        Returns:
            Latest data date as string (YYYYMMDD format)
        """
        latest_date = None
        
        try:
            for universe_name, universe_df in breadth_data.items():
                if universe_df.empty:
                    continue
                    
                # Get the latest date from this universe
                universe_latest = universe_df.iloc[-1]['date']
                
                # Convert to datetime if it's a string
                if isinstance(universe_latest, str):
                    universe_date = pd.to_datetime(universe_latest)
                else:
                    universe_date = universe_latest
                    
                # Update latest_date if this is newer
                if latest_date is None or universe_date > latest_date:
                    latest_date = universe_date
            
            # Return in YYYYMMDD format
            if latest_date:
                return latest_date.strftime('%Y%m%d')
            else:
                return datetime.now().strftime('%Y%m%d')
                
        except Exception as e:
            logger.error(f"Error extracting latest data date: {e}")
            return datetime.now().strftime('%Y%m%d')
    
    def _save_historical_breadth_analysis(self, results: Dict[str, Any], timeframe: str, latest_data_date: str) -> list:
        """
        Save historical breadth analysis results to CSV file(s).
        
        Args:
            results: Historical analysis results dictionary
            timeframe: Data timeframe 
            latest_data_date: Latest data date (YYYYMMDD format)
            
        Returns:
            List of paths to saved files (multiple files for semicolon separator, single file for plus separator)
        """
        try:
            # Get universe configuration for filename
            universe_config_obj = getattr(self.user_config, 'market_breadth_universe', {'raw_config': 'all'})
            
            if isinstance(universe_config_obj, dict):
                if 'raw_config' in universe_config_obj:
                    universe_str = universe_config_obj['raw_config']
                elif 'universes' in universe_config_obj:
                    # Clean universes list and preserve original separator intent
                    clean_universes = [u.strip() for u in universe_config_obj['universes'] if u.strip()]
                    
                    # Check if there's separator info stored in the config
                    if 'separator' in universe_config_obj:
                        separator = universe_config_obj['separator']
                    else:
                        # Default to semicolon for multiple universes (separate files)
                        # This preserves the original intent when raw_config was parsed
                        separator = ';' if len(clean_universes) > 1 else ''
                    
                    universe_str = separator.join(clean_universes) if separator else clean_universes[0]
                else:
                    universe_str = 'all'
            else:
                universe_str = str(universe_config_obj)
            
            # Get user choice for filename
            user_choice = str(self.user_config.ticker_choice) if self.user_config else '0-5'
            
            # Ensure results directory exists - use market_pulse for consistency
            results_dir = self.config.directories['RESULTS_DIR'] / 'market_pulse'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Detect separator type to determine file generation strategy
            has_semicolon = ';' in universe_str
            has_plus = '+' in universe_str
            has_comma = ',' in universe_str
            
            saved_files = []
            
            if has_semicolon or has_comma:
                # Generate separate files for each universe (semicolon/comma = separate files)
                universes_analyzed = results.get('universes_analyzed', [])
                
                logger.info(f"Generating separate files for universes: {universes_analyzed}")
                
                for universe in universes_analyzed:
                    # Create individual universe results
                    individual_results = self._create_individual_universe_results(results, universe)
                    
                    # Generate filename for individual universe
                    filename = f"ba_historical_{universe}_{user_choice}_{timeframe}_{latest_data_date}.csv"
                    output_path = results_dir / filename
                    
                    # Prepare historical data for this specific universe
                    output_data = self._prepare_historical_csv_output(individual_results, timeframe, latest_data_date, universe, user_choice)
                    
                    # Save to CSV
                    output_data.to_csv(output_path, index=False)
                    
                    saved_files.append(str(output_path))
                    logger.info(f"Individual universe analysis saved: {universe} -> {filename} ({len(output_data)} rows)")
            
            else:
                # Generate combined file (plus separator or single universe)
                universe_filename = universe_str.upper()
                
                filename = f"ba_historical_{universe_filename}_{user_choice}_{timeframe}_{latest_data_date}.csv"
                output_path = results_dir / filename
                
                # Prepare historical data for CSV output
                output_data = self._prepare_historical_csv_output(results, timeframe, latest_data_date, universe_str, user_choice)
                
                # Save to CSV
                output_data.to_csv(output_path, index=False)
                
                saved_files.append(str(output_path))
                logger.info(f"Combined breadth analysis saved: {filename} ({len(output_data)} rows, {len(output_data.columns)} columns)")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving historical breadth analysis results: {e}")
            return []
    
    def _create_individual_universe_results(self, results: Dict[str, Any], target_universe: str) -> Dict[str, Any]:
        """
        Create individual universe results for separate file generation.
        
        Args:
            results: Combined analysis results dictionary
            target_universe: Universe name to extract results for
            
        Returns:
            Dictionary with results filtered for the target universe
        """
        try:
            # Create filtered results for specific universe
            individual_results = results.copy()
            
            # Filter universe_results to only include target universe
            universe_results = results.get('universe_results', {})
            if target_universe in universe_results:
                individual_results['universe_results'] = {target_universe: universe_results[target_universe]}
            else:
                individual_results['universe_results'] = {}
            
            # Filter combined_breadth, highs_lows_analysis, advance_decline, breadth_signals 
            # to only include data for target universe (if they contain universe-specific data)
            # For now, keep the combined data as it represents the overall analysis
            
            # Update universes_analyzed to only include target universe
            individual_results['universes_analyzed'] = [target_universe]
            
            return individual_results
            
        except Exception as e:
            logger.error(f"Error creating individual universe results for {target_universe}: {e}")
            return results.copy()  # Return original results as fallback
    
    # Preserve original _save_breadth_analysis method for backwards compatibility
    def _save_breadth_analysis(self, results: Dict[str, Any], timeframe: str, latest_data_date: str) -> str:
        """Save breadth analysis results to CSV file (backwards compatibility)."""
        # Redirect to historical method which handles both formats
        files = self._save_historical_breadth_analysis(results, timeframe, latest_data_date)
        return files[0] if files else None
    
    def _generate_breadth_charts(self, csv_files: List[str], universe_results: Dict) -> List[str]:
        """
        Generate PNG charts for breadth analysis CSV files.
        
        Args:
            csv_files: List of CSV file paths
            universe_results: Dictionary of universe results for additional context
            
        Returns:
            List of generated PNG chart file paths
        """
        try:
            chart_files = []
            
            # Create visualizer instance
            visualizer = MarketBreadthVisualizer(self.config)
            
            for csv_file in csv_files:
                try:
                    # Load breadth data from CSV
                    breadth_data = pd.read_csv(csv_file)
                    
                    # Create PNG path (same name as CSV but with .png extension)
                    csv_path = Path(csv_file)
                    png_path = csv_path.with_suffix('.png')
                    
                    # Extract universe name from filename
                    filename_parts = csv_path.stem.split('_')
                    if len(filename_parts) >= 3:
                        universe_name = filename_parts[2]  # Extract from ba_historical_SP500_...
                    else:
                        universe_name = 'Market'
                    
                    # Try to load corresponding index price data
                    index_data = self._load_index_price_data(universe_name)
                    
                    # Generate chart
                    chart_path = visualizer.create_breadth_chart(
                        breadth_data=breadth_data,
                        index_data=index_data,
                        output_path=png_path,
                        universe_name=universe_name
                    )
                    
                    if chart_path:
                        chart_files.append(chart_path)
                        logger.info(f"Generated breadth chart: {chart_path}")
                    else:
                        logger.warning(f"Failed to generate chart for: {csv_file}")
                        
                except Exception as e:
                    logger.error(f"Error generating chart for {csv_file}: {e}")
                    continue
            
            return chart_files
            
        except Exception as e:
            logger.error(f"Error generating breadth charts: {e}")
            return []
    
    def _load_index_price_data(self, universe_name: str) -> pd.DataFrame:
        """
        Load index price data for chart generation.
        
        Args:
            universe_name: Universe name to determine appropriate index
            
        Returns:
            DataFrame with OHLCV price data or None
        """
        try:
            # Map universe names to index symbols
            index_mapping = {
                'SP500': 'SPY',
                'NASDAQ100': 'QQQ',
                'RUSSELL1000': 'IWM',
                'ALL': 'SPY',  # Default to SPY for combined analysis
                'COMBINED': 'SPY'
            }
            
            index_symbol = index_mapping.get(universe_name.upper(), 'SPY')
            
            # Try to load daily price data
            daily_data_dir = self.config.get_market_data_dir('daily')
            index_file = daily_data_dir / f"{index_symbol}.csv"

            if index_file.exists():
                index_data = pd.read_csv(index_file)
                logger.debug(f"Loaded price data for {index_symbol}: {len(index_data)} records")
                return index_data
            else:
                logger.debug(f"Index file not found: {index_file}")
            
            # Fallback: try to load from market_data directory using environment-aware path
            try:
                market_data_dir = self.config.get_market_data_dir('daily')
                index_file = Path(market_data_dir) / f"{index_symbol}.csv"

                if index_file.exists():
                    index_data = pd.read_csv(index_file)
                    logger.debug(f"Loaded price data from market_data for {index_symbol}: {len(index_data)} records")
                    return index_data
            except Exception as e:
                logger.debug(f"Error accessing market data directory: {e}")

            logger.debug(f"No price data found for universe {universe_name} (tried {index_symbol})")
            return None
            
        except Exception as e:
            logger.error(f"Error loading index price data for {universe_name}: {e}")
            return None
    
    def _prepare_historical_csv_output(self, results: Dict[str, Any], timeframe: str, latest_data_date: str, universe_str: str, user_choice: str) -> pd.DataFrame:
        """
        Prepare historical breadth analysis results for CSV output.
        
        Returns:
            DataFrame with complete historical breadth analysis results
        """
        try:
            # Get the historical DataFrames from results
            combined_breadth = results.get('combined_breadth', pd.DataFrame())
            highs_lows_analysis = results.get('highs_lows_analysis', pd.DataFrame())
            advance_decline = results.get('advance_decline', pd.DataFrame())
            breadth_signals = results.get('breadth_signals', pd.DataFrame())
            
            if combined_breadth.empty:
                logger.warning("No combined breadth data available for CSV output")
                return pd.DataFrame()
            
            # Start with combined breadth as base DataFrame
            output_df = combined_breadth.copy()
            
            # Add metadata columns
            output_df['timeframe'] = timeframe
            output_df['universe_config'] = universe_str
            output_df['user_choice'] = user_choice
            output_df['universes_analyzed'] = ';'.join(results.get('universes_analyzed', []))
            
            # Merge highs/lows analysis
            if not highs_lows_analysis.empty:
                # Merge on date, adding 'hl_' prefix to avoid column conflicts
                hl_cols_to_merge = [col for col in highs_lows_analysis.columns if col != 'date']
                hl_renamed = highs_lows_analysis[['date'] + hl_cols_to_merge].copy()
                for col in hl_cols_to_merge:
                    if col in output_df.columns:
                        # Rename duplicate columns with hl_ prefix
                        hl_renamed = hl_renamed.rename(columns={col: f'hl_{col}'})
                
                output_df = output_df.merge(hl_renamed, on='date', how='left')
            
            # Merge advance/decline analysis
            if not advance_decline.empty:
                # Merge on date, adding 'ad_' prefix to avoid column conflicts
                ad_cols_to_merge = [col for col in advance_decline.columns if col != 'date']
                ad_renamed = advance_decline[['date'] + ad_cols_to_merge].copy()
                for col in ad_cols_to_merge:
                    if col in output_df.columns:
                        # Rename duplicate columns with ad_ prefix
                        ad_renamed = ad_renamed.rename(columns={col: f'ad_{col}'})
                
                output_df = output_df.merge(ad_renamed, on='date', how='left')
            
            # Merge breadth signals
            if not breadth_signals.empty:
                # Merge on date, all signal columns are unique
                output_df = output_df.merge(breadth_signals, on='date', how='left')
            
            # Add universe-specific data (latest values for each universe)
            universe_results = results.get('universe_results', {})
            for i, (universe, metrics_df) in enumerate(universe_results.items()):
                if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
                    prefix = f'univ_{i+1}_'
                    
                    # Merge universe-specific data on date
                    univ_cols = ['date', 'total_members', 'advancing_stocks', 'pct_advancing', 
                               'pct_above_ma20', 'pct_above_ma50', 'pct_above_ma200', 'breadth_momentum']
                    univ_data = metrics_df[['date'] + [col for col in univ_cols[1:] if col in metrics_df.columns]].copy()
                    
                    # Rename columns with universe prefix
                    rename_dict = {col: f'{prefix}{col}' for col in univ_data.columns if col != 'date'}
                    univ_data = univ_data.rename(columns=rename_dict)
                    univ_data[f'{prefix}name'] = universe
                    
                    output_df = output_df.merge(univ_data, on='date', how='left')
            
            # Ensure proper column order: date first, then metadata, then all metrics
            meta_cols = ['date', 'timeframe', 'universe_config', 'user_choice', 'universes_analyzed']
            other_cols = [col for col in output_df.columns if col not in meta_cols]
            output_df = output_df[meta_cols + sorted(other_cols)]
            
            logger.info(f"Prepared historical CSV output: {len(output_df)} rows, {len(output_df.columns)} columns")
            
            return output_df
            
        except Exception as e:
            logger.error(f"Error preparing historical CSV output: {e}")
            return pd.DataFrame()
    
    def _generate_historical_summary(self, universe_results: Dict, combined_breadth: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of historical breadth analysis."""
        try:
            if combined_breadth.empty or signals.empty:
                return {'error': 'No data available for summary'}
            
            # Get latest data for current readings
            latest_breadth = combined_breadth.iloc[-1] if not combined_breadth.empty else {}
            latest_signals = signals.iloc[-1] if not signals.empty else {}
            
            # Calculate historical statistics
            total_days = len(combined_breadth)
            
            # Signal frequency analysis
            bullish_days = (signals['total_bullish_signals'] > 0).sum()
            bearish_days = (signals['total_bearish_signals'] > 0).sum()
            strong_bullish_days = (signals['signal_strength'].str.contains('Strong Bullish', na=False)).sum()
            strong_bearish_days = (signals['signal_strength'].str.contains('Strong Bearish', na=False)).sum()
            
            # Breadth score statistics
            avg_breadth_score = combined_breadth['overall_breadth_score'].mean()
            max_breadth_score = combined_breadth['overall_breadth_score'].max()
            min_breadth_score = combined_breadth['overall_breadth_score'].min()
            
            return {
                'analysis_period': {
                    'total_trading_days': total_days,
                    'start_date': combined_breadth['date'].min(),
                    'end_date': combined_breadth['date'].max()
                },
                'latest_readings': {
                    'overall_breadth_rating': latest_breadth.get('breadth_rating', 'Unknown'),
                    'breadth_score': latest_breadth.get('overall_breadth_score', 0),
                    'total_bullish_signals': latest_signals.get('total_bullish_signals', 0),
                    'total_bearish_signals': latest_signals.get('total_bearish_signals', 0),
                    'signal_strength': latest_signals.get('signal_strength', 'Unknown'),
                    'market_participation': {
                        'above_ma20': f"{latest_breadth.get('combined_pct_above_ma20', 0):.1f}%",
                        'above_ma50': f"{latest_breadth.get('combined_pct_above_ma50', 0):.1f}%",
                        'above_ma200': f"{latest_breadth.get('combined_pct_above_ma200', 0):.1f}%"
                    }
                },
                'historical_statistics': {
                    'breadth_score': {
                        'average': round(avg_breadth_score, 2),
                        'maximum': round(max_breadth_score, 2),
                        'minimum': round(min_breadth_score, 2)
                    },
                    'signal_frequency': {
                        'bullish_days': bullish_days,
                        'bearish_days': bearish_days,
                        'strong_bullish_days': strong_bullish_days,
                        'strong_bearish_days': strong_bearish_days,
                        'bullish_percentage': round((bullish_days / total_days) * 100, 1),
                        'bearish_percentage': round((bearish_days / total_days) * 100, 1)
                    }
                },
                'universes_analyzed': len(universe_results),
                'data_quality': {
                    'universes_with_data': len([k for k, v in universe_results.items() if not v.empty]) if isinstance(universe_results, dict) else 0,
                    'total_data_points': sum(len(v) for v in universe_results.values()) if isinstance(universe_results, dict) else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating historical summary: {e}")
            return {'error': str(e)}