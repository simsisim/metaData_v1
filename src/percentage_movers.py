"""
Percentage Movers Analysis Module

This module analyzes significant percentage price movements in stocks from the TradingView universe.
It identifies daily and weekly movers above user-defined thresholds and provides comprehensive reporting.

Author: Claude Code Assistant
Created: 2025-08-29
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path


class PercentageMoversAnalyzer:
    """
    Analyzes percentage price movements for stocks in the TradingView universe.
    
    Features:
    - Daily and weekly percentage change calculations
    - Configurable thresholds for movers detection
    - Volume filtering and metadata integration
    - Parallel processing for large datasets
    - Comprehensive reporting and CSV output
    """
    
    def __init__(self, config, user_config):
        """
        Initialize the PercentageMoversAnalyzer.
        
        Args:
            config: System configuration object
            user_config: User configuration from user_data.csv
        """
        self.config = config
        self.user_config = user_config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.daily_threshold = user_config.daily_pct_threshold
        self.weekly_threshold = user_config.weekly_pct_threshold
        self.min_volume = user_config.movers_min_volume
        self.top_n = user_config.movers_top_n
        self.output_dir = user_config.movers_output_dir
        
        # Data paths
        self.daily_data_dir = user_config.yf_daily_data_files
        self.weekly_data_dir = user_config.yf_weekly_data_files
        self.universe_file = os.path.join(
            config.directories['TICKERS_DIR'], 
            user_config.ticker_info_TW_file
        )
        
        # Results containers
        self.daily_movers = []
        self.weekly_movers = []
        self.universe_df = None
        
        # Setup output directory
        self._setup_output_directory()
        
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory ready: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise
    
    def load_universe_metadata(self) -> pd.DataFrame:
        """
        Load TradingView universe metadata for ticker enrichment.
        
        Returns:
            DataFrame with universe metadata
        """
        try:
            self.universe_df = pd.read_csv(self.universe_file)
            self.logger.info(f"Loaded universe metadata: {len(self.universe_df)} tickers")
            return self.universe_df
        except Exception as e:
            self.logger.error(f"Failed to load universe metadata from {self.universe_file}: {e}")
            return pd.DataFrame()
    
    def _load_ticker_data(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load historical price data for a specific ticker and timeframe.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: 'daily' or 'weekly'
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            if timeframe == 'daily':
                file_path = os.path.join(self.daily_data_dir, f"{ticker}.csv")
            elif timeframe == 'weekly':
                file_path = os.path.join(self.weekly_data_dir, f"{ticker}.csv")
            else:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            if not os.path.exists(file_path):
                return None
            
            df = pd.read_csv(file_path)
            
            # Standardize column names
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df = df.sort_values('Date')
            
            # Validate required columns
            required_cols = ['Date', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning(f"Missing required columns in {file_path}")
                return None
                
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to load data for {ticker} ({timeframe}): {e}")
            return None
    
    def _calculate_percentage_change(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate percentage change from the latest available data.
        
        Args:
            df: DataFrame with OHLCV data sorted by Date
            
        Returns:
            Dictionary with price change information or None
        """
        try:
            if len(df) < 2:
                return None
            
            # Get latest two data points
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            current_price = latest['Close']
            previous_price = previous['Close']
            current_volume = latest['Volume']
            
            # Calculate percentage change
            pct_change = ((current_price - previous_price) / previous_price) * 100
            
            return {
                'current_price': current_price,
                'previous_price': previous_price,
                'pct_change': pct_change,
                'abs_pct_change': abs(pct_change),
                'volume': current_volume,
                'current_date': latest['Date'],
                'previous_date': previous['Date']
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate percentage change: {e}")
            return None
    
    def _process_ticker(self, ticker: str, timeframe: str, threshold: float) -> Optional[Dict]:
        """
        Process a single ticker for percentage change analysis.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: 'daily' or 'weekly'
            threshold: Percentage threshold for significance
            
        Returns:
            Dictionary with mover information or None
        """
        df = self._load_ticker_data(ticker, timeframe)
        if df is None:
            return None
        
        change_info = self._calculate_percentage_change(df)
        if change_info is None:
            return None
        
        # Check if meets threshold and volume requirements
        if (change_info['abs_pct_change'] >= threshold and 
            change_info['volume'] >= self.min_volume):
            
            return {
                'ticker': ticker,
                'timeframe': timeframe,
                **change_info
            }
        
        return None
    
    def calculate_daily_movers(self, tickers_list: List[str]) -> List[Dict]:
        """
        Calculate daily percentage movers for the given ticker list.
        
        Args:
            tickers_list: List of ticker symbols to analyze
            
        Returns:
            List of dictionaries with daily mover information
        """
        self.logger.info(f"Analyzing daily movers for {len(tickers_list)} tickers...")
        
        daily_movers = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks
            future_to_ticker = {
                executor.submit(self._process_ticker, ticker, 'daily', self.daily_threshold): ticker
                for ticker in tickers_list
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        daily_movers.append(result)
                    
                    completed += 1
                    if completed % 100 == 0:
                        self.logger.info(f"Processed {completed}/{len(tickers_list)} tickers for daily analysis")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {ticker} for daily analysis: {e}")
        
        # Sort by absolute percentage change (descending)
        daily_movers.sort(key=lambda x: x['abs_pct_change'], reverse=True)
        
        self.logger.info(f"Found {len(daily_movers)} daily movers above {self.daily_threshold}% threshold")
        return daily_movers
    
    def calculate_weekly_movers(self, tickers_list: List[str]) -> List[Dict]:
        """
        Calculate weekly percentage movers for the given ticker list.
        
        Args:
            tickers_list: List of ticker symbols to analyze
            
        Returns:
            List of dictionaries with weekly mover information
        """
        self.logger.info(f"Analyzing weekly movers for {len(tickers_list)} tickers...")
        
        weekly_movers = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks
            future_to_ticker = {
                executor.submit(self._process_ticker, ticker, 'weekly', self.weekly_threshold): ticker
                for ticker in tickers_list
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        weekly_movers.append(result)
                    
                    completed += 1
                    if completed % 100 == 0:
                        self.logger.info(f"Processed {completed}/{len(tickers_list)} tickers for weekly analysis")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {ticker} for weekly analysis: {e}")
        
        # Sort by absolute percentage change (descending)
        weekly_movers.sort(key=lambda x: x['abs_pct_change'], reverse=True)
        
        self.logger.info(f"Found {len(weekly_movers)} weekly movers above {self.weekly_threshold}% threshold")
        return weekly_movers
    
    def merge_with_metadata(self, movers_data: List[Dict]) -> pd.DataFrame:
        """
        Merge movers data with TradingView universe metadata.
        
        Args:
            movers_data: List of mover dictionaries
            
        Returns:
            DataFrame with enriched mover information
        """
        if not movers_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        movers_df = pd.DataFrame(movers_data)
        
        # Merge with universe metadata if available
        if self.universe_df is not None and not self.universe_df.empty:
            merged_df = movers_df.merge(
                self.universe_df,
                on='ticker',
                how='left'
            )
            self.logger.info(f"Merged {len(merged_df)} movers with metadata")
        else:
            merged_df = movers_df
            self.logger.warning("No universe metadata available for merging")
        
        return merged_df
    
    def generate_reports(self, daily_movers: List[Dict], weekly_movers: List[Dict], all_moves: List[Dict] = None) -> Dict[str, str]:
        """
        Generate comprehensive reports for daily and weekly movers.
        
        Args:
            daily_movers: List of daily mover dictionaries
            weekly_movers: List of weekly mover dictionaries
            
        Returns:
            Dictionary with output file paths
        """
        # Extract data date from movers data instead of using file generation timestamp
        data_date = None
        movers_for_date = daily_movers or weekly_movers or all_moves
        
        if movers_for_date:
            # Get the current_date from the first mover
            current_date = movers_for_date[0].get('current_date')
            if current_date:
                if hasattr(current_date, 'strftime'):
                    data_date = current_date.strftime('%Y%m%d')
                else:
                    # Handle string dates
                    data_date = str(current_date)[:10].replace('-', '')
        
        # Fallback to file generation timestamp if no data date found
        if not data_date:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.logger.warning(f"Using file generation timestamp as fallback for movers reports: {timestamp}")
        else:
            timestamp = f"{data_date}_{datetime.now().strftime('%H%M%S')}"
            self.logger.info(f"Using data date for movers reports filename: {data_date}")
        
        output_files = {}
        
        try:
            # Generate daily movers report
            if daily_movers:
                daily_df = self.merge_with_metadata(daily_movers)
                daily_file = os.path.join(self.output_dir, f"daily_movers_{timestamp}.csv")
                daily_df.to_csv(daily_file, index=False)
                output_files['daily'] = daily_file
                self.logger.info(f"Daily movers report saved: {daily_file}")
            
            # Generate weekly movers report
            if weekly_movers:
                weekly_df = self.merge_with_metadata(weekly_movers)
                weekly_file = os.path.join(self.output_dir, f"weekly_movers_{timestamp}.csv")
                weekly_df.to_csv(weekly_file, index=False)
                output_files['weekly'] = weekly_file
                self.logger.info(f"Weekly movers report saved: {weekly_file}")
            
            # Generate combined report
            if daily_movers or weekly_movers:
                combined_data = []
                for mover in daily_movers[:self.top_n]:
                    combined_data.append({**mover, 'analysis_type': 'Daily'})
                for mover in weekly_movers[:self.top_n]:
                    combined_data.append({**mover, 'analysis_type': 'Weekly'})
                
                if combined_data:
                    combined_df = self.merge_with_metadata(combined_data)
                    combined_file = os.path.join(self.output_dir, f"combined_movers_{timestamp}.csv")
                    combined_df.to_csv(combined_file, index=False)
                    output_files['combined'] = combined_file
                    self.logger.info(f"Combined movers report saved: {combined_file}")
            
            # Generate comprehensive moves report (all stocks, regardless of threshold)
            if all_moves:
                all_moves_df = self.merge_with_metadata(all_moves)
                all_moves_file = os.path.join(self.output_dir, f"all_moves_{timestamp}.csv")
                all_moves_df.to_csv(all_moves_file, index=False)
                output_files['all_moves'] = all_moves_file
                self.logger.info(f"Comprehensive moves report saved: {all_moves_file}")
            
            # Generate summary report
            summary_data = {
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'daily_threshold_pct': self.daily_threshold,
                'weekly_threshold_pct': self.weekly_threshold,
                'min_volume': self.min_volume,
                'daily_movers_found': len(daily_movers),
                'weekly_movers_found': len(weekly_movers),
                'all_moves_calculated': len(all_moves) if all_moves else 0,
                'top_daily_mover': daily_movers[0]['ticker'] if daily_movers else 'None',
                'top_daily_change_pct': f"{daily_movers[0]['pct_change']:.2f}%" if daily_movers else 'N/A',
                'top_weekly_mover': weekly_movers[0]['ticker'] if weekly_movers else 'None',
                'top_weekly_change_pct': f"{weekly_movers[0]['pct_change']:.2f}%" if weekly_movers else 'N/A'
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_file = os.path.join(self.output_dir, f"movers_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            output_files['summary'] = summary_file
            self.logger.info(f"Summary report saved: {summary_file}")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
            return {}
    
    def calculate_all_moves(self, tickers_list: List[str]) -> List[Dict]:
        """
        Calculate daily and weekly percentage moves for ALL stocks (not just those meeting thresholds).
        
        Args:
            tickers_list: List of ticker symbols to analyze
            
        Returns:
            List of dictionaries with move information for all stocks
        """
        self.logger.info(f"Calculating percentage moves for all {len(tickers_list)} tickers...")
        
        all_moves = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks for both daily and weekly moves
            future_to_ticker_timeframe = {}
            
            if getattr(self.user_config, 'yf_daily_data', True):
                for ticker in tickers_list:
                    future = executor.submit(self._process_ticker_all_moves, ticker, 'daily')
                    future_to_ticker_timeframe[future] = (ticker, 'daily')
            
            if getattr(self.user_config, 'yf_weekly_data', True):
                for ticker in tickers_list:
                    future = executor.submit(self._process_ticker_all_moves, ticker, 'weekly')
                    future_to_ticker_timeframe[future] = (ticker, 'weekly')
            
            # Process completed tasks
            completed = 0
            total_tasks = len(future_to_ticker_timeframe)
            
            for future in as_completed(future_to_ticker_timeframe):
                ticker, timeframe = future_to_ticker_timeframe[future]
                try:
                    result = future.result()
                    if result:
                        all_moves.append(result)
                    
                    completed += 1
                    if completed % 100 == 0:
                        self.logger.info(f"Processed {completed}/{total_tasks} ticker-timeframe combinations...")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing {ticker} ({timeframe}): {e}")
        
        self.logger.info(f"Calculated moves for {len(all_moves)} ticker-timeframe combinations")
        return all_moves
    
    def _process_ticker_all_moves(self, ticker: str, timeframe: str) -> Optional[Dict]:
        """
        Process a single ticker for percentage change analysis (no threshold filtering).
        
        Args:
            ticker: Stock ticker symbol
            timeframe: 'daily' or 'weekly'
            
        Returns:
            Dictionary with move information or None
        """
        df = self._load_ticker_data(ticker, timeframe)
        if df is None:
            return None
        
        change_info = self._calculate_percentage_change(df)
        if change_info is None:
            return None
        
        # Return all moves (no threshold filtering)
        return {
            'ticker': ticker,
            'timeframe': timeframe,
            **change_info
        }

    def analyze_movers(self, tickers_list: List[str]) -> Dict[str, str]:
        """
        Main analysis function to identify and report percentage movers.
        
        Args:
            tickers_list: List of ticker symbols to analyze
            
        Returns:
            Dictionary with paths to generated report files
        """
        self.logger.info("Starting percentage movers analysis...")
        
        # Load universe metadata
        self.load_universe_metadata()
        
        # Calculate daily movers if enabled
        daily_movers = []
        if getattr(self.user_config, 'yf_daily_data', True):
            daily_movers = self.calculate_daily_movers(tickers_list)
            self.daily_movers = daily_movers
        
        # Calculate weekly movers if enabled
        weekly_movers = []
        if getattr(self.user_config, 'yf_weekly_data', True):
            weekly_movers = self.calculate_weekly_movers(tickers_list)
            self.weekly_movers = weekly_movers
        
        # Calculate ALL moves (including those not meeting thresholds)
        all_moves = self.calculate_all_moves(tickers_list)
        self.all_moves = all_moves
        
        # Generate reports (including comprehensive moves file)
        output_files = self.generate_reports(daily_movers, weekly_movers, all_moves)
        
        # Log analysis summary
        self.logger.info(f"Percentage movers analysis completed:")
        self.logger.info(f"  • Daily movers found: {len(daily_movers)}")
        self.logger.info(f"  • Weekly movers found: {len(weekly_movers)}")
        self.logger.info(f"  • All moves calculated: {len(all_moves)}")
        self.logger.info(f"  • Report files generated: {len(output_files)}")
        
        return output_files


def run_movers_analysis(tickers_list: List[str], user_config, config) -> Dict[str, str]:
    """
    Convenience function to run percentage movers analysis.
    
    Args:
        tickers_list: List of ticker symbols to analyze
        user_config: User configuration object
        config: System configuration object
        
    Returns:
        Dictionary with paths to generated report files
    """
    if not user_config.enable_movers_analysis:
        logging.getLogger(__name__).info("Percentage movers analysis disabled")
        return {}
    
    analyzer = PercentageMoversAnalyzer(config, user_config)
    return analyzer.analyze_movers(tickers_list)