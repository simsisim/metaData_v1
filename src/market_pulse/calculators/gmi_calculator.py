#!/usr/bin/env python3
"""
GMI (General Market Index) Calculator - Clean R1-R6 Requirements Model

Implements a 6-requirement market timing methodology:

Requirements (Total: 0-6 points):
- R1: 10-day success rate of 252-day new highs > 100 (1 point) - from market breadth
- R2: Daily new highs > 100 (1 point) - from market breadth
- R3: Index1 short-term analysis (1 point: 0.5 momentum + 0.5 trend)
- R4: Index2 short-term analysis (1 point: 0.5 momentum + 0.5 trend)  
- R5: Index2 long-term analysis (1 point: 0.5 momentum + 0.5 trend)
- R6: MF momentum analysis (0.5 points - momentum only)

Signal Logic: 
- GREEN: GMI > threshold for confirmation_days
- RED: GMI < threshold for confirmation_days
- NEUTRAL: Otherwise
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GMICalculator:
    """
    GMI Calculator implementing clean R1-R6 requirements model.
    """
    
    def __init__(self, paths: Dict[str, str] = None, 
                 threshold: int = 3, confirmation_days: int = 2, user_config=None):
        """
        Initialize GMI Calculator
        
        Args:
            paths: Dictionary containing data paths
            threshold: GMI score threshold for signals (default: 3, range: 0-6)
            confirmation_days: Days required for signal confirmation (default: 2)
            user_config: User configuration object
        """
        self.paths = paths or {}
        self.threshold = threshold
        self.confirmation_days = confirmation_days
        self.user_config = user_config
        
        # Get index configuration from user config
        self.index1 = str(getattr(user_config, 'market_pulse_gmi_index1', 'SPY')) if user_config else 'SPY'  # For R3
        self.index2 = str(getattr(user_config, 'market_pulse_gmi_index2', 'QQQ')) if user_config else 'QQQ'  # For R4, R5
        self.mf_index = str(getattr(user_config, 'market_pulse_gmi_mf_index', 'SPY')) if user_config else 'SPY'  # For R6
        
        # Moving average periods
        self.short_term_sma = int(getattr(user_config, 'market_pulse_gmi_short_term_sma', 50)) if user_config else 50
        self.long_term_sma = int(getattr(user_config, 'market_pulse_gmi_long_term_sma', 150)) if user_config else 150
        
        # Get user choice for file naming
        ticker_choice = getattr(user_config, 'ticker_choice', '0-5') if user_config else '0-5'
        self.safe_user_choice = str(ticker_choice).replace('-', '_')  # For GMI output files
        self.breadth_user_choice = str(ticker_choice)  # Keep original format for breadth files
        
        # Store last results for save_results method
        self._last_results = None
        
    def load_market_data(self, timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
        """Load market data for all required indexes"""
        try:
            market_data = {}
            
            # Load Index1 data (for R3)
            market_data['index1'] = self._load_single_index_data(self.index1)
            
            # Load Index2 data (for R4 and R5) - reuse if same as Index1
            if self.index2 == self.index1:
                market_data['index2'] = market_data['index1'].copy()
            else:
                market_data['index2'] = self._load_single_index_data(self.index2)
            
            # Load MF data (for R6) - reuse if same as existing indexes
            if self.mf_index == self.index1:
                market_data['mf'] = market_data['index1'].copy()
            elif self.mf_index == self.index2:
                market_data['mf'] = market_data['index2'].copy()
            else:
                market_data['mf'] = self._load_single_index_data(self.mf_index)
                
            # Try to load market breadth data (for R1 and R2)
            market_data['breadth'] = self._load_market_breadth_data(timeframe=timeframe)
            
                
            return market_data
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            raise
    
    def _load_single_index_data(self, index_symbol: str) -> pd.DataFrame:
        """Load data for a single market index with moving averages"""
        try:
            file_path = os.path.join(
                self.paths.get('source_market_data', ''),
                f"{index_symbol}.csv"
            )
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Market data file not found: {file_path}")
                
            df = pd.read_csv(file_path, index_col='Date', parse_dates=False)
            
            # Clean and standardize date index
            df.index = df.index.str.split(' ').str[0]
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Ensure required columns exist
            required_columns = ['Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in {index_symbol}: {missing_columns}")
            
            # Calculate required moving averages
            df[f'SMA_{self.short_term_sma}'] = df['Close'].rolling(
                window=self.short_term_sma, min_periods=self.short_term_sma
            ).mean()
            df[f'SMA_{self.long_term_sma}'] = df['Close'].rolling(
                window=self.long_term_sma, min_periods=self.long_term_sma
            ).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {index_symbol}: {e}")
            raise
    
    def _load_market_breadth_data(self, timeframe: str = 'daily', universe: str = 'all') -> Optional[pd.DataFrame]:
        """
        Load market breadth data for R1 and R2 requirements with configuration-driven date selection
        
        Args:
            timeframe: Data timeframe (daily, weekly, monthly)
            universe: Market universe (all, SP500, NASDAQ100)
            
        Returns:
            Single-row DataFrame with latest or specific date data, or None if not found
        """
        try:
            import glob
            
            # Get breadth file suffix configuration
            breath_suffix = getattr(self.user_config, 'market_pulse_gmi_breath_file_suffix', 'latest') if self.user_config else 'latest'
            
            # Set breadth directory
            breadth_dir = os.path.join('results', 'market_breadth')
            if not os.path.exists(breadth_dir):
                breadth_dir = os.path.join(self.paths.get('results', ''), 'market_breadth')
            
            # Try different universe fallbacks
            universes = [universe, 'all', 'SP500', 'NASDAQ100']
            
            for target_universe in universes:
                target_file = None
                
                if breath_suffix == "latest":
                    # Find most recent file by pattern
                    pattern = f"market_breadth_{target_universe}_{self.breadth_user_choice}_{timeframe}_*.csv"
                    files = glob.glob(os.path.join(breadth_dir, pattern))
                    if files:
                        target_file = sorted(files)[-1]  # Most recent by filename
                else:
                    # Specific date - convert YYYY-MM-DD to YYYYMMDD
                    try:
                        formatted_date = breath_suffix.replace('-', '')
                        filename = f"market_breadth_{target_universe}_{self.breadth_user_choice}_{timeframe}_{formatted_date}.csv"
                        potential_file = os.path.join(breadth_dir, filename)
                        if os.path.exists(potential_file):
                            target_file = potential_file
                    except Exception as date_error:
                        logger.warning(f"Invalid date format in breath_suffix '{breath_suffix}': {date_error}")
                        continue
                
                if target_file and os.path.exists(target_file):
                    try:
                        # Load the file
                        df = pd.read_csv(target_file)
                        
                        if df.empty:
                            logger.warning(f"Empty market breadth file: {target_file}")
                            continue
                            
                        # Ensure date column exists and is properly formatted
                        if 'date' not in df.columns:
                            logger.warning(f"No 'date' column in {target_file}")
                            continue
                            
                        # Select target row based on configuration
                        target_row = None
                        
                        if breath_suffix == "latest":
                            # Use last row (most recent data)
                            target_row = df.iloc[-1:].copy()
                            logger.info(f"Using latest data from {os.path.basename(target_file)}, date: {target_row['date'].iloc[0]}")
                        else:
                            # Look for specific date
                            target_date_rows = df[df['date'] == breath_suffix]
                            if not target_date_rows.empty:
                                target_row = target_date_rows.iloc[-1:].copy()
                                logger.info(f"Found exact date {breath_suffix} in {os.path.basename(target_file)}")
                            else:
                                # Fallback: closest previous date
                                df['date'] = pd.to_datetime(df['date'])
                                target_datetime = pd.to_datetime(breath_suffix)
                                previous_dates = df[df['date'] <= target_datetime]
                                if not previous_dates.empty:
                                    target_row = previous_dates.iloc[-1:].copy()
                                    actual_date = target_row['date'].iloc[0].strftime('%Y-%m-%d')
                                    logger.info(f"Using closest previous date {actual_date} for target {breath_suffix} in {os.path.basename(target_file)}")
                                else:
                                    logger.warning(f"No data on or before {breath_suffix} in {os.path.basename(target_file)}")
                                    continue
                        
                        if target_row is not None and not target_row.empty:
                            # Log available variables for R1 and R2
                            required_vars = {
                                'r1_10day_success': f'{timeframe}_mb_10day_successful_252day_new_highs_gt_100',
                                'r2_new_highs': f'{timeframe}_mb_252day_new_highs',
                                'r2_net_advances': f'{timeframe}_mb_net_advances'
                            }
                            
                            available_vars = []
                            for key, col_name in required_vars.items():
                                if col_name in target_row.columns:
                                    value = target_row[col_name].iloc[0]
                                    available_vars.append(f"{key}={value}")
                            
                            logger.info(f"Market breadth variables: {', '.join(available_vars)}")
                            return target_row
                            
                    except Exception as file_error:
                        logger.warning(f"Error processing {target_file}: {file_error}")
                        continue
            
            # If we get here, no suitable file was found
            logger.warning("No market breadth data found. R1 and R2 will return 0.")
            logger.info(f"Searched for: timeframe={timeframe}, universe={universe}, suffix={breath_suffix}, user_choice={self.breadth_user_choice}")
            return None
            
        except Exception as e:
            logger.warning(f"Error loading market breadth data: {e}. R1 and R2 will return 0.")
            return None
    
    def calculate_gmi_requirements(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate all GMI requirements for the latest date"""
        try:
            # Handle different data structures: market data (indexed) vs breadth data (single-row)
            date_ranges = []
            breadth_date = None
            
            for key, df in market_data.items():
                if df is not None and not df.empty:
                    if key == 'breadth':
                        # Breadth data is single-row with 'date' column, extract the date
                        if 'date' in df.columns:
                            breadth_date = pd.to_datetime(df['date'].iloc[0])
                            logger.debug(f"Breadth data target date: {breadth_date}")
                        else:
                            logger.warning("Breadth data missing 'date' column")
                    else:
                        # Market index data should have Date as index
                        if df.index.name == 'Date' or hasattr(df.index, 'to_pydatetime'):
                            date_ranges.append(df.index)
                        else:
                            logger.warning(f"{key} data doesn't have proper date index")
            
            # Determine target date for analysis
            if breadth_date is not None:
                # Use breadth data date as the target date
                latest_date = breadth_date
                logger.info(f"Using breadth data date as target: {latest_date}")
            elif date_ranges:
                # Fallback: use latest common date from market data
                common_dates = date_ranges[0]
                for date_range in date_ranges[1:]:
                    if date_range is not None:
                        common_dates = common_dates.intersection(date_range)
                        
                if len(common_dates) == 0:
                    raise ValueError("No common dates found across market data")
                    
                latest_date = common_dates.max()
                logger.info(f"Using market data common date: {latest_date}")
            else:
                raise ValueError("No valid date information found in market data")
            
            # Calculate each requirement for latest date
            results = {
                'date': latest_date,
                'R1': self.requirement_r1_10day_252day_highs(market_data.get('breadth'), latest_date),
                'R2': self.requirement_r2_daily_highs(market_data.get('breadth'), latest_date),
                'R3': self.requirement_r3_index1_analysis(market_data.get('index1'), latest_date),
                'R4': self.requirement_r4_index2_short_term(market_data.get('index2'), latest_date),
                'R5': self.requirement_r5_index2_long_term(market_data.get('index2'), latest_date),
                'R6': self.requirement_r6_mf_momentum(market_data.get('mf'), latest_date)
            }
            
            # Calculate total GMI score
            results['total_score'] = sum([results['R1'], results['R2'], results['R3'], 
                                        results['R4'], results['R5'], results['R6']])
            
            # Generate signal
            results['signal'] = self._generate_signal(results['total_score'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating GMI requirements: {e}")
            raise
    
    def requirement_r1_10day_252day_highs(self, breadth_data: Optional[pd.DataFrame], 
                                         latest_date: pd.Timestamp) -> float:
        """R1: 10-day success rate of 252-day new highs > 100 (1 point)"""
        if breadth_data is None or breadth_data.empty:
            return 0.0
            
        try:
            # Get timeframe from breadth data (should be single row)
            timeframe = breadth_data['timeframe'].iloc[0]
            target_column = f'{timeframe}_mb_10day_successful_252day_new_highs_gt_100'
            
            if target_column in breadth_data.columns:
                value = breadth_data[target_column].iloc[0]
                logger.debug(f"R1: {target_column}={value}")
                return 1.0 if value > 0 else 0.0
            else:
                logger.debug(f"R1: Column {target_column} not found in breadth data")
                return 0.0
                
        except Exception as e:
            logger.warning(f"R1 calculation error: {e}")
            return 0.0
    
    def requirement_r2_daily_highs(self, breadth_data: Optional[pd.DataFrame], 
                                  latest_date: pd.Timestamp) -> float:
        """R2: Daily new highs > 100 (1 point)"""
        if breadth_data is None or breadth_data.empty:
            return 0.0
            
        try:
            # Get timeframe from breadth data (should be single row)
            timeframe = breadth_data['timeframe'].iloc[0]
            
            # Try multiple possible column names for R2
            target_columns = [
                f'{timeframe}_mb_252day_new_highs',     # 252-day new highs > 100
                f'{timeframe}_mb_net_advances'          # Net advances > 100 (alternative)
            ]
            
            for col in target_columns:
                if col in breadth_data.columns:
                    value = breadth_data[col].iloc[0]
                    result = 1.0 if value > 100 else 0.0
                    logger.debug(f"R2: {col}={value} -> {result}")
                    return result
                    
            logger.debug(f"R2: No suitable columns found in breadth data")
            return 0.0
            
        except Exception as e:
            logger.warning(f"R2 calculation error: {e}")
            return 0.0
    
    def requirement_r3_index1_analysis(self, index1_data: Optional[pd.DataFrame], 
                                      latest_date: pd.Timestamp) -> float:
        """R3: Index1 short-term analysis (1 point: 0.5 momentum + 0.5 trend)"""
        if index1_data is None or index1_data.empty:
            return 0.0
            
        try:
            score = 0.0
            sma_col = f'SMA_{self.short_term_sma}'
            
            # R3 momentum: Index1 price > short-term SMA (0.5 points)
            if index1_data.loc[latest_date, 'Close'] > index1_data.loc[latest_date, sma_col]:
                score += 0.5
                
            # R3 trend: Short-term SMA trending positive (0.5 points)
            previous_date = index1_data.index[index1_data.index < latest_date][-1]
            if index1_data.loc[latest_date, sma_col] > index1_data.loc[previous_date, sma_col]:
                score += 0.5
                
            return score
            
        except Exception as e:
            logger.warning(f"R3 calculation error: {e}")
            return 0.0
    
    def requirement_r4_index2_short_term(self, index2_data: Optional[pd.DataFrame], 
                                        latest_date: pd.Timestamp) -> float:
        """R4: Index2 short-term analysis (1 point: 0.5 momentum + 0.5 trend)"""
        if index2_data is None or index2_data.empty:
            return 0.0
            
        try:
            score = 0.0
            sma_col = f'SMA_{self.short_term_sma}'
            
            # R4 momentum: Index2 price > short-term SMA (0.5 points)
            if index2_data.loc[latest_date, 'Close'] > index2_data.loc[latest_date, sma_col]:
                score += 0.5
                
            # R4 trend: Short-term SMA trending positive (0.5 points)
            previous_date = index2_data.index[index2_data.index < latest_date][-1]
            if index2_data.loc[latest_date, sma_col] > index2_data.loc[previous_date, sma_col]:
                score += 0.5
                
            return score
            
        except Exception as e:
            logger.warning(f"R4 calculation error: {e}")
            return 0.0
    
    def requirement_r5_index2_long_term(self, index2_data: Optional[pd.DataFrame], 
                                       latest_date: pd.Timestamp) -> float:
        """R5: Index2 long-term analysis (1 point: 0.5 momentum + 0.5 trend)"""
        if index2_data is None or index2_data.empty:
            return 0.0
            
        try:
            score = 0.0
            sma_col = f'SMA_{self.long_term_sma}'
            
            # R5 momentum: Index2 price > long-term SMA (0.5 points)
            if index2_data.loc[latest_date, 'Close'] > index2_data.loc[latest_date, sma_col]:
                score += 0.5
                
            # R5 trend: Long-term SMA trending positive (0.5 points)
            previous_date = index2_data.index[index2_data.index < latest_date][-1]
            if index2_data.loc[latest_date, sma_col] > index2_data.loc[previous_date, sma_col]:
                score += 0.5
                
            return score
            
        except Exception as e:
            logger.warning(f"R5 calculation error: {e}")
            return 0.0
    
    def requirement_r6_mf_momentum(self, mf_data: Optional[pd.DataFrame], 
                                  latest_date: pd.Timestamp) -> float:
        """R6: MF momentum analysis (0.5 points - momentum only)"""
        if mf_data is None or mf_data.empty:
            return 0.0
            
        try:
            sma_col = f'SMA_{self.short_term_sma}'
            
            # R6 momentum: MF price > short-term SMA (0.5 points)
            if mf_data.loc[latest_date, 'Close'] > mf_data.loc[latest_date, sma_col]:
                return 1
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"R6 calculation error: {e}")
            return 0.0
    
    def _generate_signal(self, gmi_score: float) -> str:
        """Generate GMI signal based on threshold"""
        if gmi_score >= self.threshold:
            return 'GREEN'
        elif gmi_score < self.threshold:
            return 'RED'
        else:
            return 'NEUTRAL'
    
    def generate_output_dataframe(self, results: Dict, timeframe: str, 
                                 user_choice: str, date_str: str) -> pd.DataFrame:
        """Generate standardized output dataframe with required columns"""
        
        row = {
            'index': 1,
            'date': results['date'].strftime('%Y-%m-%d'),
            'timeframe': timeframe,
            f'{timeframe}_gmi_signal': results['signal'],
            f'{timeframe}_gmi_score': results['total_score'],
            f'{timeframe}_gmi_max_score': 6,
            f'{timeframe}_gmi_threshold': self.threshold,
            f'{timeframe}_gmi_confirmation_days': self.confirmation_days,
            f'{timeframe}_gmi_r1': results['R1'],
            f'{timeframe}_gmi_r2': results['R2'],
            f'{timeframe}_gmi_r3': results['R3'],
            f'{timeframe}_gmi_r4': results['R4'],
            f'{timeframe}_gmi_r5': results['R5'],
            f'{timeframe}_gmi_r6': results['R6']
        }
        
        return pd.DataFrame([row])
    
    def save_gmi_results(self, results_df: pd.DataFrame, timeframe: str, 
                        user_choice: str, date_str: str) -> str:
        """Save GMI results using existing filename format"""
        try:
            # Use existing pattern: gmi_{INDEX}_{USER_CHOICE}_{timeframe}_{DATE}.csv
            primary_index = self.index1  # Use index1 as primary for filename
            filename = f"gmi_{primary_index}_{user_choice}_{timeframe}_{date_str}.csv"
            
            output_dir = self.paths.get('results', 'results')
            market_pulse_dir = os.path.join(output_dir, 'market_pulse')
            os.makedirs(market_pulse_dir, exist_ok=True)
            
            output_path = os.path.join(market_pulse_dir, filename)
            results_df.to_csv(output_path, index=False)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving GMI results: {e}")
            raise
    
    def run_gmi_analysis(self, timeframe: str = 'daily', user_choice: str = '0-5', 
                        date_str: str = None) -> Dict:
        """Run complete GMI analysis with standardized output"""
        try:
            if date_str is None:
                date_str = datetime.now().strftime('%Y%m%d')
                
            logger.info(f"Starting GMI R1-R6 analysis for {timeframe} timeframe")
            
            # Load all market data
            market_data = self.load_market_data(timeframe=timeframe)
            
            # Calculate GMI requirements
            results = self.calculate_gmi_requirements(market_data)
            
            # Extract actual data date from results for filename (not generation date)
            actual_data_date = results['date'].strftime('%Y%m%d')
            
            # Generate standardized output dataframe
            results_df = self.generate_output_dataframe(results, timeframe, user_choice, actual_data_date)
            
            # Save results with existing filename format using actual data date
            output_path = self.save_gmi_results(results_df, timeframe, user_choice, actual_data_date)
            
            # Store results for compatibility
            self._last_results = {
                'success': True,
                'current_signal': results['signal'],
                'current_score': results['total_score'],
                'max_score': 6,
                'current_date': results['date'].strftime('%Y-%m-%d'),
                'threshold': self.threshold,
                'confirmation_days': self.confirmation_days,
                'components': {
                    'R1': results['R1'],
                    'R2': results['R2'], 
                    'R3': results['R3'],
                    'R4': results['R4'],
                    'R5': results['R5'],
                    'R6': results['R6']
                }
            }
            
            return {
                'success': True,
                'timeframe': timeframe,
                'output_file': output_path,
                'results_dataframe': results_df,
                'latest_score': results['total_score'],
                'latest_signal': results['signal'],
                'requirements_breakdown': {
                    'R1': results['R1'],
                    'R2': results['R2'],
                    'R3': results['R3'],
                    'R4': results['R4'],
                    'R5': results['R5'],
                    'R6': results['R6']
                },
                'configuration': {
                    'index1': self.index1,
                    'index2': self.index2,
                    'mf_index': self.mf_index,
                    'threshold': self.threshold,
                    'confirmation_days': self.confirmation_days
                }
            }
            
        except Exception as e:
            logger.error(f"GMI analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timeframe': timeframe
            }