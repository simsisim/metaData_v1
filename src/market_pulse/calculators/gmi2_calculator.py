#!/usr/bin/env python3
"""
GMI2 (General Market Index 2) Calculator - Multi-SMA Requirements Model

Implements a 9-requirement market timing methodology based on multiple SMAs:

Requirements (Total: 0-9 points):
- R1: Daily new highs > 100 (1 point) - from market breadth - PLACEHOLDER: returns 0
- R2: Price > SMA3 with trend (0.5) + momentum (0.5) = 1 point
- R3: Price > SMA2 with trend (0.5) + momentum (0.5) = 1 point  
- R4: Price > SMA1 with trend (0.5) + momentum (0.5) = 1 point
- R5: SMA alignment (SMA2 > SMA3 > SMA4) with momentum = 1 point
- R6: To be added (0 points for now)
- R7: To be added (0 points for now)
- R8: To be added (0 points for now)
- R9: To be added (0 points for now)

Configuration:
- MARKET_PULSE_gmi2_index: SPY;QQQ (multiple indexes supported)
- MARKET_PULSE_gmi2_sma: 10;20;50;150 (expects exactly 4 values)

Signal Logic: 
- GREEN: GMI > threshold for confirmation_days
- RED: GMI < threshold for confirmation_days
- NEUTRAL: Otherwise
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GMI2Calculator:
    """
    GMI2 Calculator implementing multi-SMA requirements model.
    """
    
    def __init__(self, paths: Dict[str, str] = None, 
                 threshold: int = 5, confirmation_days: int = 2, user_config=None):
        """
        Initialize GMI2 Calculator
        
        Args:
            paths: Dictionary containing data paths
            threshold: GMI2 score threshold for signals (default: 5, range: 0-9)
            confirmation_days: Days required for signal confirmation (default: 2)
            user_config: User configuration object
        """
        self.paths = paths or {}
        self.threshold = threshold
        self.confirmation_days = confirmation_days
        self.user_config = user_config
        
        # Parse GMI2-specific configuration
        self.indexes = self._parse_indexes(
            getattr(user_config, 'market_pulse_gmi2_index', 'SPY;QQQ')
        )
        
        self.sma_periods = self._parse_sma_periods(
            getattr(user_config, 'market_pulse_gmi2_sma', '10;20;50;150')
        )
        
        # Stochastic configuration for R6, R7, R8, R9
        self.stoch_1_threshold = float(getattr(user_config, 'market_pulse_gmi2_index_stochastic_1_thresholds', 20))
        self.stoch_1_lookback = int(getattr(user_config, 'market_pulse_gmi2_index_stochastic_1_lookback', 10))
        self.stoch_1_smoothness = int(getattr(user_config, 'market_pulse_gmi2_index_stochastic_1_smoothness', 4))
        
        self.stoch_2_threshold = float(getattr(user_config, 'market_pulse_gmi2_index_stochastic_2_thresholds', 20))
        self.stoch_2_lookback = int(getattr(user_config, 'market_pulse_gmi2_index_stochastic_2_lookback', 10))
        self.stoch_2_smoothness = int(getattr(user_config, 'market_pulse_gmi2_index_stochastic_2_smoothness', 4))
        
        self.stoch_3_threshold = float(getattr(user_config, 'market_pulse_gmi2_index_stochastic_3_thresholds', 20))
        self.stoch_3_lookback = int(getattr(user_config, 'market_pulse_gmi2_index_stochastic_3_lookback', 10))
        self.stoch_3_smoothness = int(getattr(user_config, 'market_pulse_gmi2_index_stochastic_3_smoothness', 4))
        
        # Market breadth configuration for GMI2
        self.breadth_user_choice = str(getattr(user_config, 'ticker_choice', '0-5'))
        
        # Store last results for compatibility
        self._last_results = {}
        
        logger.info(f"GMI2 Calculator initialized with indexes: {self.indexes}")
        logger.info(f"GMI2 SMA periods: {self.sma_periods}")
        logger.info(f"GMI2 threshold: {self.threshold}, confirmation days: {self.confirmation_days}")
    
    def _parse_indexes(self, index_config: str) -> List[str]:
        """Parse index configuration string into list"""
        if isinstance(index_config, str):
            return [idx.strip() for idx in index_config.split(';') if idx.strip()]
        elif isinstance(index_config, list):
            return index_config
        else:
            return ['SPY', 'QQQ']  # Default fallback
    
    def _parse_sma_periods(self, sma_config: str) -> List[int]:
        """Parse SMA periods configuration string into list of integers"""
        try:
            if isinstance(sma_config, str):
                periods = [int(p.strip()) for p in sma_config.split(';') if p.strip()]
            elif isinstance(sma_config, list):
                periods = [int(p) for p in sma_config]
            else:
                periods = [10, 20, 50, 150]  # Default
                
            if len(periods) != 4:
                logger.warning(f"GMI2 expects exactly 4 SMA periods, got {len(periods)}. Using defaults.")
                periods = [10, 20, 50, 150]
                
            return sorted(periods)  # Sort to ensure proper ordering
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing SMA periods: {e}. Using defaults.")
            return [10, 20, 50, 150]
    
    def load_market_data(self, timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
        """
        Load market data for GMI2 analysis
        
        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary containing market data for each index
        """
        market_data = {}
        
        try:
            # Load data for each configured index
            for index_symbol in self.indexes:
                try:
                    index_data = self._load_single_index_data(index_symbol, timeframe)
                    if index_data is not None and not index_data.empty:
                        # Calculate required SMAs
                        index_data = self._calculate_smas(index_data)
                        market_data[index_symbol] = index_data
                        logger.debug(f"Loaded GMI2 data for {index_symbol}: {len(index_data)} rows")
                    else:
                        logger.warning(f"No data loaded for GMI2 index {index_symbol}")
                        
                except Exception as e:
                    logger.error(f"Error loading GMI2 data for {index_symbol}: {e}")
                    continue
                    
            # Load market breadth data for R1 (placeholder)
            try:
                breadth_data = self._load_market_breadth_data(timeframe)
                if breadth_data is not None:
                    market_data['breadth'] = breadth_data
                    logger.debug("Loaded market breadth data for GMI2 R1")
            except Exception as e:
                logger.warning(f"Error loading market breadth data for GMI2: {e}")
                
            return market_data
            
        except Exception as e:
            logger.error(f"Error loading GMI2 market data: {e}")
            return {}
    
    def _load_single_index_data(self, index_symbol: str, timeframe: str = 'daily') -> pd.DataFrame:
        """Load and prepare data for a single index"""
        try:
            # Get base market data directory and construct path for specific timeframe
            base_market_data_dir = self.paths.get('source_market_data', 'data/market_data')
            
            # Handle timeframe-specific paths: replace any existing timeframe with the requested one
            # This handles cases where base_market_data_dir might be /data/market_data/daily
            if '/daily' in base_market_data_dir or '/weekly' in base_market_data_dir or '/monthly' in base_market_data_dir:
                # Replace existing timeframe with requested timeframe
                parent_dir = base_market_data_dir.rsplit('/', 1)[0]  # Remove last directory
                data_path = os.path.join(parent_dir, timeframe.lower(), f"{index_symbol}.csv")
            else:
                # Add timeframe directory to base path
                data_path = os.path.join(base_market_data_dir, timeframe.lower(), f"{index_symbol}.csv")
            
            if not os.path.exists(data_path):
                logger.warning(f"GMI2 data file not found: {data_path}")
                return None
                
            df = pd.read_csv(data_path)
            
            # Ensure Date column and set as index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df.set_index('Date', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading GMI2 index data for {index_symbol}: {e}")
            return None
    
    def _calculate_smas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required SMAs for the dataframe"""
        try:
            df_copy = df.copy()
            
            for period in self.sma_periods:
                sma_col = f'SMA_{period}'
                df_copy[sma_col] = df_copy['Close'].rolling(window=period).mean()
                
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating SMAs: {e}")
            return df
    
    def _load_market_breadth_data(self, timeframe: str = 'daily', universe: str = 'all') -> Optional[pd.DataFrame]:
        """
        Load market breadth data for GMI2 R1 requirement with configuration-driven date selection
        
        Args:
            timeframe: Data timeframe (daily, weekly, monthly)
            universe: Market universe (all, SP500, NASDAQ100)
            
        Returns:
            Single-row DataFrame with latest or specific date data, or None if not found
        """
        try:
            import glob
            
            # Get breadth file suffix configuration (GMI2-specific or fallback to GMI)
            breath_suffix = getattr(self.user_config, 'market_pulse_gmi2_breath_file_suffix', 
                                  getattr(self.user_config, 'market_pulse_gmi_breath_file_suffix', 'latest')) if self.user_config else 'latest'
            
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
                            logger.info(f"GMI2 using latest data from {os.path.basename(target_file)}, date: {target_row['date'].iloc[0]}")
                        else:
                            # Look for specific date
                            target_date_rows = df[df['date'] == breath_suffix]
                            if not target_date_rows.empty:
                                target_row = target_date_rows.iloc[-1:].copy()
                                logger.info(f"GMI2 found exact date {breath_suffix} in {os.path.basename(target_file)}")
                            else:
                                # Fallback: closest previous date
                                df['date'] = pd.to_datetime(df['date'])
                                target_datetime = pd.to_datetime(breath_suffix)
                                previous_dates = df[df['date'] <= target_datetime]
                                if not previous_dates.empty:
                                    target_row = previous_dates.iloc[-1:].copy()
                                    actual_date = target_row['date'].iloc[0].strftime('%Y-%m-%d')
                                    logger.info(f"GMI2 using closest previous date {actual_date} for target {breath_suffix} in {os.path.basename(target_file)}")
                                else:
                                    logger.warning(f"No data on or before {breath_suffix} in {os.path.basename(target_file)}")
                                    continue
                        
                        if target_row is not None and not target_row.empty:
                            # Log available variables for GMI2 R1
                            required_vars = {
                                'r1_10day_success': f'{timeframe}_mb_10day_successful_long_new_highs_gt_100',
                                'r1_new_highs': f'{timeframe}_mb_long_new_highs',
                                'r1_net_advances': f'{timeframe}_mb_net_advances'
                            }
                            
                            available_vars = []
                            for key, col_name in required_vars.items():
                                if col_name in target_row.columns:
                                    value = target_row[col_name].iloc[0]
                                    available_vars.append(f"{key}={value}")
                            
                            logger.info(f"GMI2 market breadth variables: {', '.join(available_vars)}")
                            return target_row
                            
                    except Exception as file_error:
                        logger.warning(f"Error processing {target_file}: {file_error}")
                        continue
            
            # If we get here, no suitable file was found
            logger.warning("No market breadth data found for GMI2. R1 will return 0.")
            logger.info(f"GMI2 searched for: timeframe={timeframe}, universe={universe}, suffix={breath_suffix}, user_choice={self.breadth_user_choice}")
            return None
            
        except Exception as e:
            logger.warning(f"Error loading GMI2 market breadth data: {e}. R1 will return 0.")
            return None
    
    def calculate_gmi2_requirements_for_index(self, market_data: Dict[str, pd.DataFrame], 
                                             target_index: str) -> pd.DataFrame:
        """Calculate GMI2 requirements for a specific index"""
        try:
            # Find latest common date for the target index
            target_data = market_data.get(target_index)
            if target_data is None or target_data.empty:
                raise ValueError(f"No market data available for GMI2 index {target_index}")
            
            latest_date = target_data.index[-1]
            logger.debug(f"GMI2 using latest date for {target_index}: {latest_date}")
            
            # Calculate requirements for this specific index
            requirements = {
                'R1': self.requirement_r1_daily_highs(market_data.get('breadth'), latest_date),
                'R2': self.requirement_r2_sma3_analysis_for_index(target_data, latest_date),
                'R3': self.requirement_r3_sma2_analysis_for_index(target_data, latest_date),
                'R4': self.requirement_r4_sma1_analysis_for_index(target_data, latest_date),
                'R5': self.requirement_r5_sma_alignment_for_index(target_data, latest_date),
                'R6': self.requirement_r6_stochastic_analysis_for_index(target_data, latest_date),
                'R7': 0.0,  # To be implemented
                'R8': self.requirement_r8_dual_stochastic_analysis_for_index(target_data, latest_date),
                'R9': self.requirement_r9_stochastic_analysis_for_index(target_data, latest_date)
            }
            
            # Calculate totals
            total_score = sum(requirements.values())
            max_score = 9  # Total possible score
            
            # Determine signal
            signal = self._determine_signal(total_score)
            
            return {
                'date': latest_date,
                'signal': signal,
                'total_score': total_score,
                'max_score': max_score,
                'target_index': target_index,
                **requirements
            }
            
        except Exception as e:
            logger.error(f"Error calculating GMI2 requirements for {target_index}: {e}")
            raise

    def calculate_gmi2_requirements(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate all GMI2 requirements for the latest date"""
        try:
            # Find latest common date across all data
            date_ranges = []
            for key, df in market_data.items():
                if df is not None and not df.empty and key != 'breadth':
                    date_ranges.append(df.index)
                    
            if not date_ranges:
                raise ValueError("No market data available for GMI2 analysis")
                
            # Find latest common date
            common_dates = date_ranges[0]
            for date_range in date_ranges[1:]:
                common_dates = common_dates.intersection(date_range)
                
            if len(common_dates) == 0:
                raise ValueError("No common dates found across GMI2 market data")
                
            latest_date = common_dates[-1]
            logger.debug(f"GMI2 using latest date: {latest_date}")
            
            # Calculate each requirement
            requirements = {
                'R1': self.requirement_r1_daily_highs(market_data.get('breadth'), latest_date),
                'R2': self.requirement_r2_sma3_analysis(market_data, latest_date),
                'R3': self.requirement_r3_sma2_analysis(market_data, latest_date),
                'R4': self.requirement_r4_sma1_analysis(market_data, latest_date),
                'R5': self.requirement_r5_sma_alignment(market_data, latest_date),
                'R6': 0.0,  # To be implemented
                'R7': 0.0,  # To be implemented
                'R8': 0.0,  # To be implemented
                'R9': 0.0   # To be implemented
            }
            
            # Calculate totals
            total_score = sum(requirements.values())
            max_score = 9  # Total possible score
            
            # Determine signal
            signal = self._determine_signal(total_score)
            
            return {
                'date': latest_date,
                'signal': signal,
                'total_score': total_score,
                'max_score': max_score,
                **requirements
            }
            
        except Exception as e:
            logger.error(f"Error calculating GMI2 requirements: {e}")
            raise
    
    def requirement_r1_daily_highs(self, breadth_data: Optional[pd.DataFrame], 
                                  latest_date: pd.Timestamp) -> float:
        """R1: If net advances > 0 award 1 point"""
        if breadth_data is None or breadth_data.empty:
            return 0.0
            
        try:
            # Get timeframe from breadth data (should be single row)
            timeframe = breadth_data['timeframe'].iloc[0]
            target_column = f'{timeframe}_mb_net_advances'
            
            if target_column in breadth_data.columns:
                value = breadth_data[target_column].iloc[0]
                result = 1.0 if value > 0 else 0.0
                logger.debug(f"GMI2 R1: {target_column}={value} -> {result}")
                return result
            else:
                logger.debug(f"GMI2 R1: Column {target_column} not found in breadth data")
                return 0.0
                
        except Exception as e:
            logger.warning(f"GMI2 R1 calculation error: {e}")
            return 0.0
    
    def requirement_r2_sma3_analysis_for_index(self, index_data: pd.DataFrame, 
                                              latest_date: pd.Timestamp) -> float:
        """R2: Price > SMA3 with trend (0.5) + momentum (0.5) = 1 point for specific index"""
        return self._calculate_sma_requirement_for_index(index_data, latest_date, self.sma_periods[2])  # SMA3
    
    def requirement_r3_sma2_analysis_for_index(self, index_data: pd.DataFrame, 
                                              latest_date: pd.Timestamp) -> float:
        """R3: Price > SMA2 with trend (0.5) + momentum (0.5) = 1 point for specific index"""
        return self._calculate_sma_requirement_for_index(index_data, latest_date, self.sma_periods[1])  # SMA2
    
    def requirement_r4_sma1_analysis_for_index(self, index_data: pd.DataFrame, 
                                              latest_date: pd.Timestamp) -> float:
        """R4: Price > SMA1 with trend (0.5) + momentum (0.5) = 1 point for specific index"""
        return self._calculate_sma_requirement_for_index(index_data, latest_date, self.sma_periods[0])  # SMA1
    
    def requirement_r5_sma_alignment_for_index(self, index_data: pd.DataFrame, 
                                              latest_date: pd.Timestamp) -> float:
        """R5: SMA alignment (SMA2 > SMA3 > SMA4) for specific index"""
        try:
            # Use periods 2, 3, 4 (index 1, 2, 3 in sorted list)
            sma_cols = [f'SMA_{self.sma_periods[i]}' for i in [1, 2, 3]]
            
            if latest_date not in index_data.index:
                return 0.0
                
            # Check if all required SMA columns exist
            if not all(col in index_data.columns for col in sma_cols):
                return 0.0
                
            try:
                sma2 = index_data.loc[latest_date, sma_cols[0]]  # SMA2
                sma3 = index_data.loc[latest_date, sma_cols[1]]  # SMA3
                sma4 = index_data.loc[latest_date, sma_cols[2]]  # SMA4
                
                # Check alignment: SMA2 > SMA3 > SMA4
                if sma2 > sma3 > sma4:
                    return 1.0  # Full point for proper alignment
                else:
                    return 0.0
            except:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating SMA alignment requirement: {e}")
            return 0.0

    def requirement_r6_stochastic_analysis_for_index(self, index_data: pd.DataFrame, 
                                                    latest_date: pd.Timestamp) -> float:
        """R6: Stochastic < threshold for specific index (1 point if oversold)"""
        try:
            if latest_date not in index_data.index:
                return 0.0
                
            # Check if required OHLC columns exist
            required_columns = ['High', 'Low', 'Close']
            if not all(col in index_data.columns for col in required_columns):
                logger.debug(f"Missing required columns for stochastic calculation: {required_columns}")
                return 0.0
            
            # Extract OHLC data
            highs = index_data['High']
            lows = index_data['Low']
            closes = index_data['Close']
            
            # Calculate stochastic value for R6 (single smoothing)
            stoch_value = self.calculate_stochastic_value(
                highs=highs,
                lows=lows, 
                closes=closes,
                lookback=self.stoch_1_lookback,
                smoothness=self.stoch_1_smoothness,
                double_smoothness=None,  # Single smoothing
                check_index=latest_date
            )
            
            # R6: Award 1.0 if stochastic < threshold (oversold)
            result = 1.0 if stoch_value < self.stoch_1_threshold else 0.0
            
            return result
            
        except Exception as e:
            logger.warning(f"Error calculating R6 stochastic requirement: {e}")
            return 0.0

    def requirement_r8_dual_stochastic_analysis_for_index(self, index_data: pd.DataFrame, 
                                                         latest_date: pd.Timestamp) -> float:
        """R8: Dual stochastic comparison - Fast > Slow OR Fast > 80 (1 point if condition met)"""
        return self.calculate_dual_stochastic_comparison(index_data, latest_date)

    def requirement_r9_stochastic_analysis_for_index(self, index_data: pd.DataFrame, 
                                                    latest_date: pd.Timestamp) -> float:
        """R9: Simple stochastic ≤ 20 using 10.1 parameters (1 point if oversold)"""
        try:
            if latest_date not in index_data.index:
                return 0.0
                
            # Check if required OHLC columns exist
            required_columns = ['High', 'Low', 'Close']
            if not all(col in index_data.columns for col in required_columns):
                logger.debug(f"Missing required columns for R9 stochastic calculation: {required_columns}")
                return 0.0
            
            # Extract OHLC data
            highs = index_data['High']
            lows = index_data['Low']
            closes = index_data['Close']
            
            # Calculate stochastic value for R9 using stochastic_3 parameters (10.1 - single smoothing)
            stoch_value = self.calculate_stochastic_value(
                highs=highs,
                lows=lows, 
                closes=closes,
                lookback=self.stoch_3_lookback,
                smoothness=self.stoch_3_smoothness,
                double_smoothness=None,  # Single smoothing
                check_index=latest_date
            )
            
            # R9: Award 1.0 if stochastic ≤ threshold (oversold)
            result = 1.0 if stoch_value <= self.stoch_3_threshold else 0.0
            
            logger.debug(f"R9 stochastic: value={stoch_value:.2f}, threshold={self.stoch_3_threshold}, result={result}")
            return result
            
        except Exception as e:
            logger.warning(f"Error calculating R9 stochastic requirement: {e}")
            return 0.0

    def requirement_r2_sma3_analysis(self, market_data: Dict[str, pd.DataFrame], 
                                    latest_date: pd.Timestamp) -> float:
        """R2: Price > SMA3 with trend (0.5) + momentum (0.5) = 1 point"""
        return self._calculate_sma_requirement(market_data, latest_date, self.sma_periods[2])  # SMA3 (3rd period)
    
    def requirement_r3_sma2_analysis(self, market_data: Dict[str, pd.DataFrame], 
                                    latest_date: pd.Timestamp) -> float:
        """R3: Price > SMA2 with trend (0.5) + momentum (0.5) = 1 point"""
        return self._calculate_sma_requirement(market_data, latest_date, self.sma_periods[1])  # SMA2 (2nd period)
    
    def requirement_r4_sma1_analysis(self, market_data: Dict[str, pd.DataFrame], 
                                    latest_date: pd.Timestamp) -> float:
        """R4: Price > SMA1 with trend (0.5) + momentum (0.5) = 1 point"""
        return self._calculate_sma_requirement(market_data, latest_date, self.sma_periods[0])  # SMA1 (1st period)
    
    def _calculate_sma_requirement(self, market_data: Dict[str, pd.DataFrame], 
                                  latest_date: pd.Timestamp, sma_period: int) -> float:
        """Helper method to calculate SMA-based requirements"""
        try:
            total_score = 0.0
            valid_indexes = 0
            sma_col = f'SMA_{sma_period}'
            
            for index_symbol in self.indexes:
                index_data = market_data.get(index_symbol)
                if index_data is None or index_data.empty:
                    continue
                    
                if latest_date not in index_data.index or sma_col not in index_data.columns:
                    continue
                    
                valid_indexes += 1
                score = 0.0
                
                # Momentum: Price > SMA (0.5 points)
                if index_data.loc[latest_date, 'Close'] > index_data.loc[latest_date, sma_col]:
                    score += 0.5
                    
                # Trend: SMA trending positive (0.5 points)
                try:
                    previous_dates = index_data.index[index_data.index < latest_date]
                    if len(previous_dates) > 0:
                        previous_date = previous_dates[-1]
                        if index_data.loc[latest_date, sma_col] > index_data.loc[previous_date, sma_col]:
                            score += 0.5
                except:
                    pass
                    
                total_score += score
                
            # Average across valid indexes
            if valid_indexes > 0:
                return total_score / valid_indexes
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating SMA requirement for period {sma_period}: {e}")
            return 0.0
    
    def requirement_r5_sma_alignment(self, market_data: Dict[str, pd.DataFrame], 
                                    latest_date: pd.Timestamp) -> float:
        """R5: SMA alignment (SMA2 > SMA3 > SMA4) with momentum = 1 point"""
        try:
            total_score = 0.0
            valid_indexes = 0
            
            # Use periods 2, 3, 4 (index 1, 2, 3 in sorted list)
            sma_cols = [f'SMA_{self.sma_periods[i]}' for i in [1, 2, 3]]
            
            for index_symbol in self.indexes:
                index_data = market_data.get(index_symbol)
                if index_data is None or index_data.empty:
                    continue
                    
                if latest_date not in index_data.index:
                    continue
                    
                # Check if all required SMA columns exist
                if not all(col in index_data.columns for col in sma_cols):
                    continue
                    
                valid_indexes += 1
                score = 0.0
                
                try:
                    sma2 = index_data.loc[latest_date, sma_cols[0]]  # SMA2
                    sma3 = index_data.loc[latest_date, sma_cols[1]]  # SMA3
                    sma4 = index_data.loc[latest_date, sma_cols[2]]  # SMA4
                    
                    # Check alignment: SMA2 > SMA3 > SMA4
                    if sma2 > sma3 > sma4:
                        score = 1.0  # Full point for proper alignment
                        
                except:
                    pass
                    
                total_score += score
                
            # Average across valid indexes
            if valid_indexes > 0:
                return total_score / valid_indexes
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating SMA alignment requirement: {e}")
            return 0.0
    
    def _determine_signal(self, score: float) -> str:
        """Determine GMI2 signal based on score"""
        # Simplified signal logic for now
        if score >= self.threshold:
            return "GREEN"
        elif score <= (self.threshold - 2):
            return "RED"
        else:
            return "NEUTRAL"
    
    def generate_output_dataframe(self, results: Dict, timeframe: str, 
                                 user_choice: str, date_str: str) -> pd.DataFrame:
        """Generate standardized output dataframe with required columns"""
        
        row = {
            'index': 1,
            'date': results['date'].strftime('%Y-%m-%d'),
            'timeframe': timeframe,
            f'{timeframe}_gmi2_signal': results['signal'],
            f'{timeframe}_gmi2_score': results['total_score'],
            f'{timeframe}_gmi2_max_score': results['max_score'],
            f'{timeframe}_gmi2_threshold': self.threshold,
            f'{timeframe}_gmi2_confirmation_days': self.confirmation_days,
            f'{timeframe}_gmi2_r1': results['R1'],
            f'{timeframe}_gmi2_r2': results['R2'],
            f'{timeframe}_gmi2_r3': results['R3'],
            f'{timeframe}_gmi2_r4': results['R4'],
            f'{timeframe}_gmi2_r5': results['R5'],
            f'{timeframe}_gmi2_r6': results['R6'],
            f'{timeframe}_gmi2_r7': results['R7'],
            f'{timeframe}_gmi2_r8': results['R8'],
            f'{timeframe}_gmi2_r9': results['R9']
        }
        
        return pd.DataFrame([row])
    
    def save_gmi2_results(self, results_df: pd.DataFrame, timeframe: str, 
                         user_choice: str, date_str: str) -> str:
        """Save GMI2 results using existing filename format"""
        try:
            # Use existing pattern: gmi2_{INDEXES}_{USER_CHOICE}_{timeframe}_{DATE}.csv
            indexes_str = "_".join(self.indexes)
            filename = f"gmi2_{indexes_str}_{user_choice}_{timeframe}_{date_str}.csv"
            
            output_dir = self.paths.get('results', 'results')
            market_pulse_dir = os.path.join(output_dir, 'market_pulse')
            os.makedirs(market_pulse_dir, exist_ok=True)
            
            output_path = os.path.join(market_pulse_dir, filename)
            results_df.to_csv(output_path, index=False)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving GMI2 results: {e}")
            raise
    
    def run_gmi2_analysis(self, timeframe: str = 'daily', user_choice: str = '0-5', 
                         date_str: str = None) -> Dict:
        """Run complete GMI2 analysis with separate results for each index"""
        try:
            if date_str is None:
                date_str = datetime.now().strftime('%Y%m%d')
                
            logger.info(f"Starting GMI2 multi-SMA analysis for {timeframe} timeframe")
            
            # Load all market data
            market_data = self.load_market_data(timeframe=timeframe)
            
            # Generate separate results for each index
            all_results = {}
            output_files = []
            
            for index_symbol in self.indexes:
                logger.info(f"Calculating GMI2 for {index_symbol}")
                
                # Calculate GMI2 requirements for this specific index
                index_results = self.calculate_gmi2_requirements_for_index(market_data, index_symbol)
                
                # Extract actual data date from results for filename (not generation date)
                actual_data_date = index_results['date'].strftime('%Y%m%d')
                
                # Generate standardized output dataframe for this index
                results_df = self.generate_output_dataframe(index_results, timeframe, user_choice, actual_data_date)
                
                # Save results with individual index filename format
                output_path = self.save_gmi2_results_for_index(results_df, timeframe, user_choice, actual_data_date, index_symbol)
                output_files.append(output_path)
                
                # Store results for this index
                all_results[index_symbol] = {
                    'success': True,
                    'timeframe': timeframe,
                    'output_file': output_path,
                    'results_dataframe': results_df,
                    'latest_score': index_results['total_score'],
                    'latest_signal': index_results['signal'],
                    'requirements_breakdown': {f'R{i}': index_results[f'R{i}'] for i in range(1, 10)},
                    'index_symbol': index_symbol
                }
            
            # For backward compatibility, use the first index's results as primary
            primary_index = self.indexes[0]
            primary_results = all_results[primary_index]
            
            # Store results for compatibility
            self._last_results = {
                'success': True,
                'current_signal': primary_results['latest_signal'],
                'current_score': primary_results['latest_score'],
                'max_score': 9,  # GMI2 max score
                'current_date': actual_data_date,
                'threshold': self.threshold,
                'confirmation_days': self.confirmation_days,
                'components': primary_results['requirements_breakdown']
            }
            
            return {
                'success': True,
                'timeframe': timeframe,
                'output_files': output_files,  # List of all output files
                'output_file': output_files[0],  # Primary file for backward compatibility  
                'results_by_index': all_results,  # Individual results per index
                'latest_score': primary_results['latest_score'],  # Primary index score
                'latest_signal': primary_results['latest_signal'],  # Primary index signal
                'requirements_breakdown': primary_results['requirements_breakdown'],  # Primary index breakdown
                'configuration': {
                    'indexes': self.indexes,
                    'sma_periods': self.sma_periods,
                    'threshold': self.threshold,
                    'confirmation_days': self.confirmation_days
                }
            }
            
        except Exception as e:
            logger.error(f"GMI2 analysis failed for {timeframe}: {e}")
            return {
                'success': False,
                'error': str(e),
                'timeframe': timeframe
            }
    
    def _calculate_sma_requirement_for_index(self, index_data: pd.DataFrame, 
                                           latest_date: pd.Timestamp, sma_period: int) -> float:
        """Helper method to calculate SMA-based requirements for a specific index"""
        try:
            sma_col = f'SMA_{sma_period}'
            
            if latest_date not in index_data.index or sma_col not in index_data.columns:
                return 0.0
                
            score = 0.0
            
            # Momentum: Price > SMA (0.5 points)
            if index_data.loc[latest_date, 'Close'] > index_data.loc[latest_date, sma_col]:
                score += 0.5
                
            # Trend: SMA trending positive (0.5 points)
            try:
                previous_dates = index_data.index[index_data.index < latest_date]
                if len(previous_dates) > 0:
                    previous_date = previous_dates[-1]
                    if index_data.loc[latest_date, sma_col] > index_data.loc[previous_date, sma_col]:
                        score += 0.5
            except:
                pass
                
            return score
                
        except Exception as e:
            logger.warning(f"Error calculating SMA requirement for period {sma_period}: {e}")
            return 0.0
    
    def calculate_stochastic_and_check(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                                     lookback: int = 10, smoothness: int = 4, threshold: float = 20.0, 
                                     check_index=None) -> float:
        """
        Calculate stochastic oscillator and check if below threshold.
        
        Args:
            highs: High prices series
            lows: Low prices series  
            closes: Close prices series
            lookback: Lookback period for stochastic calculation
            smoothness: Smoothness factor for %K smoothing
            threshold: Threshold value for comparison
            check_index: Specific index to check, if None returns latest value
            
        Returns:
            1.0 if stochastic < threshold, 0.0 otherwise
        """
        try:
            # Ensure we have enough data
            if len(closes) < max(lookback, smoothness):
                logger.debug(f"Insufficient data for stochastic calculation: {len(closes)} bars")
                return 0.0
            
            # Calculate rolling highest high and lowest low
            lowest_low = lows.rolling(window=lookback, min_periods=lookback).min()
            highest_high = highs.rolling(window=lookback, min_periods=lookback).max()
            
            # Calculate denominator and handle division by zero
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, np.nan)
            
            # Calculate raw %K
            stoch_k = 100 * (closes - lowest_low) / denominator
            stoch_k = stoch_k.fillna(0)
            
            # Apply smoothing to %K
            stoch_k_smooth = stoch_k.rolling(window=smoothness, min_periods=smoothness).mean().fillna(0)
            
            # Get value to check
            if check_index is not None and check_index in stoch_k_smooth.index:
                stoch_value = stoch_k_smooth.loc[check_index]
            else:
                # Use latest available value
                stoch_value = stoch_k_smooth.iloc[-1] if len(stoch_k_smooth) > 0 else 0.0
            
            # Return 1.0 if below threshold (oversold condition), 0.0 otherwise
            result = 1.0 if stoch_value < threshold else 0.0
            
            logger.debug(f"Stochastic calculation: value={stoch_value:.2f}, threshold={threshold}, result={result}")
            return result
            
        except Exception as e:
            logger.warning(f"Error calculating stochastic: {e}")
            return 0.0
    
    def calculate_stochastic_value(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                                 lookback: int = 10, smoothness: int = 4, double_smoothness: int = None, check_index=None) -> float:
        """
        Calculate stochastic oscillator value (returns actual numeric value).
        
        Args:
            highs: High prices series
            lows: Low prices series  
            closes: Close prices series
            lookback: Lookback period for stochastic calculation
            smoothness: First smoothness factor for %K smoothing
            double_smoothness: Second smoothness factor for double smoothing (None = single smoothing)
            check_index: Specific index to check, if None returns latest value
            
        Returns:
            Actual stochastic value (0-100)
        """
        try:
            # Ensure we have enough data
            if len(closes) < max(lookback, smoothness):
                logger.debug(f"Insufficient data for stochastic calculation: {len(closes)} bars")
                return 0.0
            
            # Calculate rolling highest high and lowest low
            lowest_low = lows.rolling(window=lookback, min_periods=lookback).min()
            highest_high = highs.rolling(window=lookback, min_periods=lookback).max()
            
            # Calculate denominator and handle division by zero
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, np.nan)
            
            # Calculate raw %K
            stoch_k = 100 * (closes - lowest_low) / denominator
            stoch_k = stoch_k.fillna(0)
            
            # Apply first smoothing to %K
            stoch_k_smooth = stoch_k.rolling(window=smoothness, min_periods=smoothness).mean().fillna(0)
            
            # Apply second smoothing if specified (for fast stochastic 10.4.4)
            if double_smoothness is not None:
                stoch_k_smooth = stoch_k_smooth.rolling(window=double_smoothness, min_periods=double_smoothness).mean().fillna(0)
            
            # Get value to return
            if check_index is not None and check_index in stoch_k_smooth.index:
                stoch_value = stoch_k_smooth.loc[check_index]
            else:
                # Use latest available value
                stoch_value = stoch_k_smooth.iloc[-1] if len(stoch_k_smooth) > 0 else 0.0
            
            return float(stoch_value)
            
        except Exception as e:
            logger.warning(f"Error calculating stochastic value: {e}")
            return 0.0
    
    def calculate_dual_stochastic_comparison(self, index_data: pd.DataFrame, latest_date: pd.Timestamp) -> float:
        """
        Calculate dual stochastic comparison for R8.
        Returns 1.0 if (Fast Stochastic > Slow Stochastic) OR (Fast Stochastic > 80)
        
        Args:
            index_data: OHLC data for specific index
            latest_date: Date to calculate for
            
        Returns:
            1.0 if conditions met, 0.0 otherwise
        """
        try:
            if latest_date not in index_data.index:
                return 0.0
                
            # Check if required OHLC columns exist
            required_columns = ['High', 'Low', 'Close']
            if not all(col in index_data.columns for col in required_columns):
                logger.debug(f"Missing required columns for dual stochastic calculation: {required_columns}")
                return 0.0
            
            # Extract OHLC data
            highs = index_data['High']
            lows = index_data['Low']
            closes = index_data['Close']
            
            # Calculate slow stochastic (10.4 - single smoothing) using stochastic_1 parameters
            slow_stoch = self.calculate_stochastic_value(
                highs=highs,
                lows=lows, 
                closes=closes,
                lookback=self.stoch_1_lookback,
                smoothness=self.stoch_1_smoothness,
                double_smoothness=None,  # Single smoothing
                check_index=latest_date
            )
            
            # Calculate fast stochastic (10.4.4 - double smoothing) using stochastic_2 parameters  
            fast_stoch = self.calculate_stochastic_value(
                highs=highs,
                lows=lows, 
                closes=closes,
                lookback=self.stoch_2_lookback,
                smoothness=self.stoch_2_smoothness,
                double_smoothness=self.stoch_2_smoothness,  # Double smoothing (same value)
                check_index=latest_date
            )
            
            # R8 Logic: Fast > Slow OR Fast > 80
            overbought_threshold = 80.0
            result = 1.0 if (fast_stoch > slow_stoch or fast_stoch > overbought_threshold) else 0.0
            
            logger.debug(f"R8 dual stochastic: Fast={fast_stoch:.2f}, Slow={slow_stoch:.2f}, Result={result}")
            return result
            
        except Exception as e:
            logger.warning(f"Error calculating dual stochastic comparison: {e}")
            return 0.0
    
    def save_gmi2_results_for_index(self, results_df: pd.DataFrame, timeframe: str, 
                                   user_choice: str, date_str: str, index_symbol: str) -> str:
        """Save GMI2 results for a specific index to individual file"""
        try:
            # Create filename for individual index
            filename = f"gmi2_{index_symbol}_{user_choice}_{timeframe}_{date_str}.csv"
            output_path = Path(self.paths['results']) / 'market_pulse' / filename
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the results
            results_df.to_csv(output_path, index=False)
            
            logger.info(f"GMI2 results for {index_symbol} saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving GMI2 results for {index_symbol}: {e}")
            raise