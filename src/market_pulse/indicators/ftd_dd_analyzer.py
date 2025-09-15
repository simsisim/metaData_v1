"""
Follow-Through Day & Distribution Day Analyzer
==============================================

Implements William J. O'Neil's Follow-Through Day and Distribution Day analysis
for market timing and trend confirmation.

Follow-Through Days (FTD):
- Strong volume increase on market index
- Price advance after market correction
- Confirms market uptrend resumption

Distribution Days (DD):
- Heavy selling on increased volume
- Price decline on market index
- Indicates institutional selling pressure

Configuration:
- ftd_dd_indexes: Target indexes (SPY;QQQ;IWM)
- ftd_dd_ftd_price_threshold: FTD price gain threshold (1.5%)
- ftd_dd_ftd_volume_threshold: FTD volume ratio (1.25)
- ftd_dd_dd_price_threshold: DD price decline threshold (-0.2%)
- ftd_dd_dd_volume_threshold: DD volume ratio (1.20)
- ftd_dd_analysis_period: Lookback period (50 days)
- ftd_dd_recent_activity_period: Recent activity window (25 days)
- ftd_dd_charts: Chart generation indexes
- ftd_dd_charts_timeframe: Chart timeframe (150 days)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import json

from .base_indicator import BaseIndicator

logger = logging.getLogger(__name__)


class FTDDistributionAnalyzer(BaseIndicator):
    """
    Analyzer for Follow-Through Days and Distribution Days in market indexes.
    
    Based on William J. O'Neil's market timing methodology from 
    "How to Make Money in Stocks" and IBD analysis.
    
    Follows the established pattern from ChillaxMAS and MACyclesAnalyzer with:
    - Multi-index support via semicolon-separated configuration
    - Chart generation with signal visualization
    - Proper output file naming with latest data date
    - Comprehensive error handling and logging
    """
    
    def __init__(self, paths: Dict[str, str] = None, user_config=None):
        """
        Initialize FTD/DD analyzer.
        
        Args:
            paths: Dictionary containing data paths
            user_config: User configuration object
        """
        from src.config import Config
        
        # Initialize with dummy symbol and proper config
        config = Config()
        super().__init__('SPY', config, user_config)
        self.paths = paths or {}
        
        # Load configuration from user_config
        self.indexes = self._parse_indexes_config()
        self.ftd_price_threshold = self._get_config_float('ftd_dd_ftd_price_threshold', 1.5)
        self.ftd_volume_threshold = self._get_config_float('ftd_dd_ftd_volume_threshold', 1.25)
        self.dd_price_threshold = self._get_config_float('ftd_dd_dd_price_threshold', -0.2)
        self.dd_volume_threshold = self._get_config_float('ftd_dd_dd_volume_threshold', 1.20)
        self.analysis_period = self._get_config_int('ftd_dd_analysis_period', 50)
        self.recent_activity_period = self._get_config_int('ftd_dd_recent_activity_period', 25)
        self.optimal_days_min = self._get_config_int('ftd_dd_optimal_days_min', 4)
        self.optimal_days_max = self._get_config_int('ftd_dd_optimal_days_max', 7)
        
        # Chart configuration
        self.chart_indexes = self._parse_chart_indexes_config()
        self.chart_timeframe = self._get_config_int('ftd_dd_charts_timeframe', 150)
        
        logger.info(f"FTD/DD Analyzer initialized with indexes: {self.indexes}")
        logger.info(f"Chart generation for: {self.chart_indexes}")
    
    def _parse_indexes_config(self) -> List[str]:
        """Parse ftd_dd_indexes configuration."""
        if not self.user_config:
            return ['SPY', 'QQQ', 'IWM']
        
        indexes_config = getattr(self.user_config, 'ftd_dd_indexes', 'SPY;QQQ;IWM')
        return [idx.strip() for idx in str(indexes_config).split(';') if idx.strip()]
    
    def _parse_chart_indexes_config(self) -> List[str]:
        """Parse ftd_dd_charts configuration."""
        if not self.user_config:
            return ['SPY', 'QQQ']
        
        chart_config = getattr(self.user_config, 'ftd_dd_charts', 'SPY;QQQ')
        if not chart_config or str(chart_config).strip() == '':
            return self.indexes  # Default to all indexes if empty
        
        return [idx.strip() for idx in str(chart_config).split(';') if idx.strip()]
    
    def _get_config_float(self, key: str, default: float) -> float:
        """Get float configuration value."""
        if not self.user_config:
            return default
        return float(getattr(self.user_config, key, default))
    
    def _get_config_int(self, key: str, default: int) -> int:
        """Get integer configuration value."""
        if not self.user_config:
            return default
        return int(getattr(self.user_config, key, default))
        
    def run_ftd_dd_analysis(self, timeframe: str = 'daily') -> Dict[str, Any]:
        """
        Run complete FTD/DD analysis for all configured indexes.
        
        Args:
            timeframe: Data timeframe ('daily' recommended for FTD/DD)
            
        Returns:
            Dictionary containing comprehensive FTD/DD analysis results
        """
        try:
            logger.info(f"Starting FTD/DD analysis for {len(self.indexes)} indexes: {self.indexes}")
            
            # Load market data for all indexes
            market_data = self.load_market_data(timeframe)
            if not market_data:
                return {
                    'success': False,
                    'error': 'Failed to load market data for FTD/DD analysis',
                    'indexes': self.indexes
                }
            
            # Get latest data date for output naming
            latest_data_date = self._get_latest_data_date(market_data)
            
            # Process each index
            results_by_index = {}
            output_files = []
            signals_files = []
            chart_files = []
            
            for index in self.indexes:
                try:
                    logger.info(f"Processing FTD/DD analysis for {index}")
                    index_results = self._analyze_ftd_dd_for_index(market_data, index)
                    
                    if index_results['success']:
                        # Generate output dataframe
                        results_df = self._generate_output_dataframe(
                            index_results, timeframe, self.user_config.ticker_choice, latest_data_date
                        )
                        
                        # Save results
                        output_file = self._save_ftd_dd_results_for_index(
                            results_df, timeframe, self.user_config.ticker_choice, 
                            latest_data_date, index
                        )
                        output_files.append(output_file)
                        
                        # Export detailed signal data
                        signals_file = self._export_detailed_signals(
                            index_results, timeframe, self.user_config.ticker_choice,
                            latest_data_date, index
                        )
                        if signals_file:
                            index_results['signals_file'] = signals_file
                            signals_files.append(signals_file)

                        # Generate chart if configured
                        if index in self.chart_indexes:
                            chart_file = self._generate_ftd_dd_chart(
                                market_data[index], index_results, index, 
                                timeframe, latest_data_date
                            )
                            if chart_file:
                                index_results['chart_file'] = chart_file
                                chart_files.append(chart_file)
                        
                        results_by_index[index] = index_results
                    else:
                        logger.warning(f"FTD/DD analysis failed for {index}: {index_results.get('error', 'Unknown error')}")
                        results_by_index[index] = index_results
                        
                except Exception as e:
                    logger.error(f"Error processing FTD/DD analysis for {index}: {e}")
                    results_by_index[index] = {
                        'success': False,
                        'error': str(e),
                        'index': index
                    }
            
            return {
                'success': True,
                'timeframe': timeframe,
                'analysis_date': latest_data_date,
                'indexes_processed': list(results_by_index.keys()),
                'results_by_index': results_by_index,
                'output_files': output_files,
                'signals_files': signals_files,
                'chart_files': chart_files,
                'configuration': {
                    'ftd_price_threshold': self.ftd_price_threshold,
                    'ftd_volume_threshold': self.ftd_volume_threshold,
                    'dd_price_threshold': self.dd_price_threshold,
                    'dd_volume_threshold': self.dd_volume_threshold,
                    'analysis_period': self.analysis_period,
                    'recent_activity_period': self.recent_activity_period
                }
            }
            
        except Exception as e:
            logger.error(f"FTD/DD analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'indexes': self.indexes
            }
    
    def _analyze_ftd_dd_for_index(self, market_data: Dict[str, pd.DataFrame], index: str) -> Dict[str, Any]:
        """
        Analyze FTD/DD signals for a specific index.
        
        Args:
            market_data: Dictionary containing market data for all indexes
            index: Index symbol to analyze
            
        Returns:
            Dictionary containing FTD/DD analysis results for the index
        """
        try:
            index_data = market_data.get(index)
            if index_data is None or index_data.empty:
                return {
                    'success': False,
                    'error': f'No market data available for {index}',
                    'index': index
                }
            
            # Calculate volume moving averages for comparison
            index_data = self._calculate_volume_metrics(index_data)
            
            # Identify Follow-Through Days
            ftd_results = self._identify_follow_through_days(index_data)
            
            # Identify Distribution Days  
            dd_results = self._identify_distribution_days(index_data)
            
            # Calculate current market state
            market_state = self._determine_market_state(index_data, ftd_results, dd_results)
            
            # Generate summary metrics
            summary = self._generate_summary(ftd_results, dd_results, market_state)
            
            # Get latest values for quick reference
            latest_data = index_data.iloc[-1]
            
            return {
                'success': True,
                'index': index,
                'date': latest_data['Date'],
                'latest_price': latest_data['Close'],
                'follow_through_days': ftd_results,
                'distribution_days': dd_results,
                'market_state': market_state,
                'summary': summary,
                'data_points': len(index_data),
                'latest_signal': self._get_latest_signal(ftd_results, dd_results),
                'signal_strength': self._calculate_signal_strength(market_state, ftd_results, dd_results)
            }
            
        except Exception as e:
            logger.error(f"FTD/DD analysis failed for {index}: {e}")
            return {
                'success': False,
                'error': str(e),
                'index': index
            }
    
    def load_market_data(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Load market data for all configured indexes.
        
        Args:
            timeframe: Data timeframe to load
            
        Returns:
            Dictionary mapping index symbols to their DataFrames
        """
        market_data = {}
        
        for index in self.indexes:
            try:
                # Set current symbol for data loading
                self.symbol = index
                data = self.load_ticker_data(timeframe)
                
                if data is not None and not data.empty:
                    # Ensure Date column is datetime
                    if 'Date' in data.columns:
                        data['Date'] = pd.to_datetime(data['Date'])
                    market_data[index] = data
                    logger.info(f"Loaded {len(data)} data points for {index}")
                else:
                    logger.warning(f"No data available for {index}")
                    
            except Exception as e:
                logger.error(f"Failed to load data for {index}: {e}")
        
        return market_data
    
    def _get_latest_data_date(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """
        Get the latest data date across all loaded market data.
        
        Args:
            market_data: Dictionary of market data
            
        Returns:
            Latest date string in YYYYMMDD format
        """
        latest_date = None
        
        for index, data in market_data.items():
            if data is not None and not data.empty and 'Date' in data.columns:
                index_latest = data['Date'].max()
                if latest_date is None or index_latest > latest_date:
                    latest_date = index_latest
        
        if latest_date is not None:
            return latest_date.strftime('%Y%m%d')
        else:
            return datetime.now().strftime('%Y%m%d')
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based metrics for FTD/DD identification."""
        df = df.copy()
        
        # 50-day average volume
        df['volume_50ma'] = df['Volume'].rolling(window=50, min_periods=10).mean()
        
        # Volume ratio (current vs average)
        df['volume_ratio'] = df['Volume'] / df['volume_50ma']
        
        # Price change percentage
        df['price_change_pct'] = df['Close'].pct_change() * 100
        
        return df
    
    def _identify_follow_through_days(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify Follow-Through Days in the market data.
        
        FTD Criteria:
        1. Price increase of at least 1.0%
        2. Volume increase of at least 25% above 50-day average
        3. Occurs after market correction/downtrend
        """
        ftd_days = []
        
        # Look for FTDs in recent data
        recent_data = df.tail(self.analysis_period).copy()
        
        for idx, row in recent_data.iterrows():
            if pd.isna(row['volume_ratio']) or pd.isna(row['price_change_pct']):
                continue
                
            # Check FTD criteria
            volume_surge = row['volume_ratio'] >= self.ftd_volume_threshold
            price_advance = row['price_change_pct'] >= self.ftd_price_threshold
            
            if volume_surge and price_advance:
                ftd_day = {
                    'date': row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else str(idx),
                    'price_change_pct': row['price_change_pct'],
                    'volume_ratio': row['volume_ratio'],
                    'close_price': row['Close'],
                    'volume': row['Volume'],
                    'strength': self._calculate_ftd_strength(row)
                }
                ftd_days.append(ftd_day)
        
        # Sort by date (most recent first)
        ftd_days.sort(key=lambda x: x['date'], reverse=True)
        
        return ftd_days
    
    def _identify_distribution_days(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify Distribution Days in the market data.
        
        DD Criteria:
        1. Price decline of at least 0.2%
        2. Volume increase of at least 20% above 50-day average
        3. Indicates institutional selling
        """
        dd_days = []
        
        # Look for DDs in recent data
        recent_data = df.tail(self.analysis_period).copy()
        
        for idx, row in recent_data.iterrows():
            if pd.isna(row['volume_ratio']) or pd.isna(row['price_change_pct']):
                continue
                
            # Check DD criteria
            volume_surge = row['volume_ratio'] >= self.dd_volume_threshold
            price_decline = row['price_change_pct'] <= self.dd_price_threshold
            
            if volume_surge and price_decline:
                dd_day = {
                    'date': row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else str(idx),
                    'price_change_pct': round(row['price_change_pct'], 2),
                    'volume_ratio': round(row['volume_ratio'], 2),
                    'close_price': round(row['Close'], 2),
                    'volume': int(row['Volume']),
                    'severity': self._calculate_dd_severity(row),
                    'signal_type': 'DD'
                }
                dd_days.append(dd_day)
        
        # Sort by date (most recent first)
        dd_days.sort(key=lambda x: x['date'], reverse=True)
        
        return dd_days
    
    def _calculate_ftd_strength(self, row: pd.Series) -> str:
        """Calculate the strength of a Follow-Through Day."""
        volume_strength = row['volume_ratio']
        price_strength = row['price_change_pct']
        
        # Strong FTD: >2.0% price gain + >50% volume increase
        if price_strength >= 2.0 and volume_strength >= 1.5:
            return 'Strong'
        # Moderate FTD: >1.5% price gain + >35% volume increase  
        elif price_strength >= 1.5 and volume_strength >= 1.35:
            return 'Moderate'
        else:
            return 'Weak'
    
    def _calculate_dd_severity(self, row: pd.Series) -> str:
        """Calculate the severity of a Distribution Day."""
        volume_intensity = row['volume_ratio']
        price_decline = abs(row['price_change_pct'])
        
        # Severe DD: >1.0% decline + >40% volume increase
        if price_decline >= 1.0 and volume_intensity >= 1.4:
            return 'Severe'
        # Moderate DD: >0.5% decline + >30% volume increase
        elif price_decline >= 0.5 and volume_intensity >= 1.3:
            return 'Moderate'
        else:
            return 'Mild'
    
    def _determine_market_state(self, df: pd.DataFrame, ftd_days: List, dd_days: List) -> Dict[str, Any]:
        """
        Determine current market state based on recent FTD/DD activity.
        
        Market States:
        - Confirmed Uptrend: Recent FTDs, few DDs
        - Under Pressure: Recent DDs accumulating
        - Correction: Heavy DD activity, no recent FTDs
        - Emerging: Potential FTD after correction
        """
        # Analyze recent activity based on configured period
        recent_ftds = [ftd for ftd in ftd_days if self._is_recent(ftd['date'], self.recent_activity_period)]
        recent_dds = [dd for dd in dd_days if self._is_recent(dd['date'], self.recent_activity_period)]
        
        # Count recent activity
        ftd_count = len(recent_ftds)
        dd_count = len(recent_dds)
        
        # Determine market state based on O'Neil methodology
        if ftd_count >= 2 and dd_count <= 1:
            state = 'Confirmed Uptrend'
            confidence = 'High'
        elif ftd_count >= 1 and dd_count <= 2:
            state = 'Uptrend'
            confidence = 'Moderate'
        elif dd_count >= 3 and ftd_count == 0:
            state = 'Under Pressure'
            confidence = 'High'
        elif dd_count >= 5:
            state = 'Correction'
            confidence = 'High'
        else:
            state = 'Neutral'
            confidence = 'Low'
        
        return {
            'current_state': state,
            'confidence': confidence,
            'recent_ftd_count': ftd_count,
            'recent_dd_count': dd_count,
            'ftd_dd_ratio': round(ftd_count / max(dd_count, 1), 2),
            'last_ftd_date': recent_ftds[0]['date'] if recent_ftds else None,
            'last_dd_date': recent_dds[0]['date'] if recent_dds else None,
            'analysis_period_days': self.recent_activity_period
        }
    
    def _is_recent(self, date_str: str, days_back: int) -> bool:
        """Check if a date is within the specified number of days."""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            cutoff_date = datetime.now() - timedelta(days=days_back)
            return date_obj >= cutoff_date
        except Exception as e:
            logger.warning(f"Error parsing date {date_str}: {e}")
            return False
    
    def _generate_summary(self, ftd_days: List, dd_days: List, market_state: Dict) -> Dict[str, Any]:
        """Generate summary statistics for FTD/DD analysis."""
        return {
            'total_ftd_count': len(ftd_days),
            'total_dd_count': len(dd_days),
            'market_state': market_state['current_state'],
            'state_confidence': market_state['confidence'],
            'recent_activity': {
                f'ftds_last_{self.recent_activity_period}_days': market_state['recent_ftd_count'],
                f'dds_last_{self.recent_activity_period}_days': market_state['recent_dd_count']
            },
            'latest_signals': {
                'last_ftd': ftd_days[0]['date'] if ftd_days else 'None',
                'last_dd': dd_days[0]['date'] if dd_days else 'None'
            },
            'ftd_dd_ratio': market_state['ftd_dd_ratio']
        }
    
    def _get_latest_signal(self, ftd_days: List, dd_days: List) -> str:
        """Get the most recent signal type."""
        if not ftd_days and not dd_days:
            return 'None'
        
        latest_ftd_date = ftd_days[0]['date'] if ftd_days else '1900-01-01'
        latest_dd_date = dd_days[0]['date'] if dd_days else '1900-01-01'
        
        if latest_ftd_date > latest_dd_date:
            return f"FTD ({latest_ftd_date})"
        elif latest_dd_date > latest_ftd_date:
            return f"DD ({latest_dd_date})"
        else:
            return 'None'
    
    def _calculate_signal_strength(self, market_state: Dict, ftd_days: List, dd_days: List) -> str:
        """Calculate overall signal strength based on market state and recent activity."""
        confidence = market_state['confidence']
        ftd_count = market_state['recent_ftd_count']
        dd_count = market_state['recent_dd_count']
        
        if confidence == 'High':
            if ftd_count >= 2 and dd_count <= 1:
                return 'Strong Bullish'
            elif dd_count >= 3:
                return 'Strong Bearish'
        elif confidence == 'Moderate':
            if ftd_count >= 1:
                return 'Moderate Bullish'
        
        return 'Neutral'
    
    def _generate_output_dataframe(self, index_results: Dict, timeframe: str, ticker_choice: str, data_date: str) -> pd.DataFrame:
        """
        Generate output dataframe for FTD/DD analysis results.
        
        Args:
            index_results: Results from FTD/DD analysis
            timeframe: Data timeframe
            ticker_choice: User's ticker choice
            data_date: Data date for file naming
            
        Returns:
            DataFrame with FTD/DD analysis results
        """
        try:
            # Create base record
            base_data = {
                'date': data_date,
                'timeframe': timeframe,
                'ticker_choice': ticker_choice,
                'index': index_results['index'],
                'latest_price': index_results['latest_price'],
                'market_state': index_results['market_state']['current_state'],
                'confidence': index_results['market_state']['confidence'],
                'recent_ftd_count': index_results['market_state']['recent_ftd_count'],
                'recent_dd_count': index_results['market_state']['recent_dd_count'],
                'ftd_dd_ratio': index_results['market_state']['ftd_dd_ratio'],
                'total_ftd_count': len(index_results['follow_through_days']),
                'total_dd_count': len(index_results['distribution_days']),
                'latest_signal': index_results['latest_signal'],
                'signal_strength': index_results['signal_strength'],
                'last_ftd_date': index_results['market_state']['last_ftd_date'] or '',
                'last_dd_date': index_results['market_state']['last_dd_date'] or '',
                'analysis_period': self.analysis_period,
                'recent_activity_period': self.recent_activity_period,
                'ftd_price_threshold': self.ftd_price_threshold,
                'ftd_volume_threshold': self.ftd_volume_threshold,
                'dd_price_threshold': self.dd_price_threshold,
                'dd_volume_threshold': self.dd_volume_threshold
            }
            
            # Create DataFrame
            df = pd.DataFrame([base_data])
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating output dataframe: {e}")
            raise
    
    def _save_ftd_dd_results_for_index(self, results_df: pd.DataFrame, timeframe: str, ticker_choice: str, data_date: str, index: str) -> str:
        """
        Save FTD/DD results for a specific index.
        
        Args:
            results_df: Results DataFrame
            timeframe: Data timeframe
            ticker_choice: User's ticker choice
            data_date: Data date for file naming
            index: Index symbol
            
        Returns:
            Path to saved file
        """
        try:
            # Create output directory - ensure it goes to market_pulse subdirectory
            base_results_dir = Path(self.paths.get('results', 'results'))
            output_dir = base_results_dir / 'market_pulse'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename: ftd_dd_{index}_{ticker_choice}_{timeframe}_{data_date}.csv
            filename = f"ftd_dd_{index}_{ticker_choice}_{timeframe}_{data_date}.csv"
            output_path = output_dir / filename
            
            # Save to CSV
            results_df.to_csv(output_path, index=False)
            logger.info(f"FTD/DD results for {index} saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving FTD/DD results for {index}: {e}")
            raise
    
    def _generate_ftd_dd_chart(self, index_data: pd.DataFrame, index_results: Dict, index: str, timeframe: str, data_date: str) -> Optional[str]:
        """
        Generate professional FTD/DD chart with candlesticks, volume, and volume ratio.
        
        Args:
            index_data: Price and volume data for the index
            index_results: FTD/DD analysis results
            index: Index symbol
            timeframe: Data timeframe
            data_date: Data date for file naming
            
        Returns:
            Path to saved chart file or None if chart generation fails
        """
        try:
            # Prepare data for charting (limit to chart timeframe)
            chart_data = index_data.tail(self.chart_timeframe).copy()
            
            if chart_data.empty:
                logger.warning(f"No data available for {index} chart generation")
                return None
            
            # Calculate volume metrics for chart data
            chart_data = self._calculate_volume_metrics(chart_data)
            
            # Convert Date to datetime if not already
            if 'Date' in chart_data.columns:
                chart_data['Date'] = pd.to_datetime(chart_data['Date'])
                dates = chart_data['Date']
            else:
                dates = pd.to_datetime(chart_data.index)
                chart_data['Date'] = dates
            
            # Create figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), 
                                              gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.15})
            fig.patch.set_facecolor('white')
            
            # Set white background for all subplots
            for ax in [ax1, ax2, ax3]:
                ax.set_facecolor('white')
                ax.grid(True, color='#CCCCCC', alpha=0.3)
                ax.tick_params(colors='black')
            
            # 1. TOP SUBPLOT: Candlestick Chart with FTD/DD markers
            self._create_candlestick_subplot(ax1, chart_data, index_results, index)
            
            # 2. MIDDLE SUBPLOT: Volume bars with 50-day MA
            self._create_volume_subplot(ax2, chart_data)
            
            # 3. BOTTOM SUBPLOT: Volume ratio with thresholds
            self._create_volume_ratio_subplot(ax3, chart_data)
            
            # Format x-axis (only on bottom subplot)
            self._format_chart_axes(ax1, ax2, ax3, dates)
            
            # Add metrics table
            self._add_metrics_table(fig, index_results)
            
            # Save chart - ensure it goes to market_pulse subdirectory
            base_results_dir = Path(self.paths.get('results', 'results'))
            output_dir = base_results_dir / 'market_pulse'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            chart_filename = f"ftd_dd_chart_{index}_{self.user_config.ticker_choice}_{timeframe}_{data_date}.png"
            chart_path = output_dir / chart_filename
            
            # Use subplots_adjust instead of tight_layout to avoid warnings
            plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.15, hspace=0.3)
            plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Enhanced FTD/DD chart for {index} saved to: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error generating enhanced FTD/DD chart for {index}: {e}")
            return None

    def _create_candlestick_subplot(self, ax, chart_data: pd.DataFrame, index_results: Dict, index: str):
        """Create candlestick chart with FTD/DD markers (top subplot)."""
        try:
            # Draw candlesticks
            for i, (date, row) in enumerate(chart_data.iterrows()):
                open_price = row['Open']
                high_price = row['High']
                low_price = row['Low']
                close_price = row['Close']
                
                # Determine candle color
                candle_color = '#26A69A' if close_price >= open_price else '#EF5350'
                
                # Draw the high-low line (wick)
                ax.plot([i, i], [low_price, high_price], color=candle_color, linewidth=1)
                
                # Draw the open-close body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                # Create candlestick body rectangle
                from matplotlib.patches import Rectangle
                body = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                               facecolor=candle_color, edgecolor=candle_color, alpha=0.8)
                ax.add_patch(body)
            
            # Add FTD and DD markers (latest 5 only)
            self._add_enhanced_signal_markers(ax, chart_data, index_results)
            
            # Formatting
            ax.set_title(f'{index} - Follow-Through Days & Distribution Days Analysis', 
                        fontsize=16, fontweight='bold', color='black')
            ax.set_ylabel('Price ($)', fontsize=12, color='black')
            
            # Set x-axis to show only indices for now
            ax.set_xlim(-0.5, len(chart_data) - 0.5)
            ax.set_xticklabels([])  # Remove x-labels (shared with bottom)
            
        except Exception as e:
            logger.error(f"Error creating candlestick subplot: {e}")
    
    def _create_volume_subplot(self, ax, chart_data: pd.DataFrame):
        """Create volume bars with moving average (middle subplot)."""
        try:
            # Create volume bars colored by price movement
            volume_colors = ['#EF5350' if change < 0 else '#26A69A' for change in chart_data['price_change_pct']]
            
            bars = ax.bar(range(len(chart_data)), chart_data['Volume'], 
                         color=volume_colors, alpha=0.7, width=0.8)
            
            # Add 50-day volume moving average
            ax.plot(range(len(chart_data)), chart_data['volume_50ma'], 
                   color='#FF9800', linewidth=2, label='50-day Volume MA')
            
            # Add volume threshold lines
            ftd_threshold_volume = chart_data['volume_50ma'] * self.ftd_volume_threshold
            dd_threshold_volume = chart_data['volume_50ma'] * self.dd_volume_threshold
            
            ax.plot(range(len(chart_data)), ftd_threshold_volume, 
                   color='green', linestyle='--', alpha=0.7, label=f'FTD Threshold ({self.ftd_volume_threshold}x)')
            ax.plot(range(len(chart_data)), dd_threshold_volume, 
                   color='red', linestyle='--', alpha=0.7, label=f'DD Threshold ({self.dd_volume_threshold}x)')
            
            # Formatting
            ax.set_ylabel('Volume', fontsize=12, color='black')
            ax.legend(loc='upper left', fontsize=9)
            ax.set_xticklabels([])  # Remove x-labels (shared with bottom)
            
            # Format volume numbers (millions/billions)
            def volume_formatter(x, pos):
                if x >= 1e9:
                    return f'{x/1e9:.1f}B'
                elif x >= 1e6:
                    return f'{x/1e6:.1f}M'
                else:
                    return f'{x/1e3:.0f}K'
            
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(volume_formatter))
            
        except Exception as e:
            logger.error(f"Error creating volume subplot: {e}")
    
    def _create_volume_ratio_subplot(self, ax, chart_data: pd.DataFrame):
        """Create volume ratio line chart with threshold lines (bottom subplot)."""
        try:
            # Plot volume ratio line
            ax.plot(range(len(chart_data)), chart_data['volume_ratio'], 
                   color='#2196F3', linewidth=2, label='Volume Ratio (Vol/50MA)')
            
            # Add threshold lines
            ax.axhline(y=self.ftd_volume_threshold, color='green', linestyle='--', 
                      alpha=0.7, label=f'FTD Threshold ({self.ftd_volume_threshold})')
            ax.axhline(y=self.dd_volume_threshold, color='red', linestyle='--', 
                      alpha=0.7, label=f'DD Threshold ({self.dd_volume_threshold})')
            ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5, label='Baseline (1.0)')
            
            # Color fill areas above thresholds
            ax.fill_between(range(len(chart_data)), chart_data['volume_ratio'], self.ftd_volume_threshold, 
                           where=(chart_data['volume_ratio'] >= self.ftd_volume_threshold), 
                           color='green', alpha=0.2, interpolate=True)
            ax.fill_between(range(len(chart_data)), chart_data['volume_ratio'], self.dd_volume_threshold, 
                           where=((chart_data['volume_ratio'] >= self.dd_volume_threshold) & 
                                  (chart_data['volume_ratio'] < self.ftd_volume_threshold)), 
                           color='orange', alpha=0.2, interpolate=True)
            
            # Formatting
            ax.set_ylabel('Volume Ratio', fontsize=12, color='black')
            ax.set_xlabel('Trading Days', fontsize=12, color='black')
            ax.legend(loc='upper left', fontsize=9)
            ax.set_ylim(0, max(3.0, chart_data['volume_ratio'].max() * 1.1))
            
        except Exception as e:
            logger.error(f"Error creating volume ratio subplot: {e}")
    
    def _format_chart_axes(self, ax1, ax2, ax3, dates):
        """Format axes for all subplots."""
        try:
            # Set x-axis limits for all subplots
            xlim = (-0.5, len(dates) - 0.5)
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)
            ax3.set_xlim(xlim)
            
            # Format bottom subplot x-axis with dates
            step = max(1, len(dates) // 10)  # Show ~10 date labels
            tick_positions = range(0, len(dates), step)
            tick_labels = [dates.iloc[i].strftime('%Y-%m-%d') for i in tick_positions]
            
            ax3.set_xticks(tick_positions)
            ax3.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
            
        except Exception as e:
            logger.error(f"Error formatting chart axes: {e}")
    
    def _add_enhanced_signal_markers(self, ax, chart_data: pd.DataFrame, index_results: Dict):
        """Add enhanced FTD and DD markers showing latest 5 points only."""
        try:
            # Get FTD and DD data
            ftd_days = index_results.get('follow_through_days', [])
            dd_days = index_results.get('distribution_days', [])
            
            logger.info(f"Processing signals: {len(ftd_days)} FTDs, {len(dd_days)} DDs")
            logger.info(f"Chart data date range: {chart_data['Date'].min()} to {chart_data['Date'].max()}")
            
            # Sort by date and get latest 5 of each type
            latest_ftd = sorted(ftd_days, key=lambda x: x['date'], reverse=True)[:5]
            latest_dd = sorted(dd_days, key=lambda x: x['date'], reverse=True)[:5]
            
            logger.info(f"Latest 5 FTDs: {[f['date'] for f in latest_ftd]}")
            logger.info(f"Latest 5 DDs: {[d['date'] for d in latest_dd]}")
            
            # Normalize chart data dates for easier matching
            chart_data['Date_normalized'] = pd.to_datetime(chart_data['Date']).dt.strftime('%Y-%m-%d')
            
            # Process FTD markers
            ftd_markers_added = 0
            for i, ftd in enumerate(latest_ftd):
                try:
                    # Normalize FTD date - handle both YYYY-MM-DD and YYYYMMDD formats
                    if isinstance(ftd['date'], str):
                        if len(ftd['date']) == 8:  # YYYYMMDD format
                            ftd_date_str = f"{ftd['date'][:4]}-{ftd['date'][4:6]}-{ftd['date'][6:8]}"
                        else:  # Assume YYYY-MM-DD format
                            ftd_date_str = ftd['date']
                    else:
                        ftd_date_str = pd.to_datetime(ftd['date']).strftime('%Y-%m-%d')
                    
                    # Find matching data point
                    matching_data = chart_data[chart_data['Date_normalized'] == ftd_date_str]
                    
                    if not matching_data.empty:
                        # Get position in chart (use reset index for consistent positioning)
                        chart_position = matching_data.index[0]
                        row_position = chart_data.reset_index().index[chart_data.index == chart_position][0]
                        price = matching_data['High'].iloc[0] + (matching_data['High'].iloc[0] * 0.01)  # Position above high
                        
                        # Marker size based on strength (larger for more recent)
                        base_size = 200 - (i * 30)  # Decreasing size: 200, 170, 140, 110, 80
                        strength_multiplier = {'Strong': 1.2, 'Moderate': 1.0, 'Weak': 0.8}
                        final_size = base_size * strength_multiplier.get(ftd.get('strength', 'Moderate'), 1.0)
                        
                        # Add upward arrow marker
                        ax.scatter(row_position, price, color='limegreen', marker='^', 
                                 s=final_size, zorder=15, edgecolor='darkgreen', linewidth=2, alpha=0.9)
                        
                        # Add number annotation
                        ax.annotate(str(i + 1), (row_position, price), 
                                  xytext=(0, 12), textcoords='offset points',
                                  ha='center', va='bottom', fontsize=11, fontweight='bold',
                                  color='white', 
                                  bbox=dict(boxstyle='circle,pad=0.3', facecolor='darkgreen', alpha=0.9, edgecolor='white'))
                        
                        ftd_markers_added += 1
                        logger.info(f"Added FTD marker #{i+1} at position {row_position}, price {price:.2f} for date {ftd_date_str}")
                        
                    else:
                        logger.warning(f"No matching data for FTD date {ftd_date_str}")
                        
                except Exception as e:
                    logger.error(f"Error adding FTD marker for {ftd.get('date', 'unknown')}: {e}")
            
            # Process DD markers  
            dd_markers_added = 0
            for i, dd in enumerate(latest_dd):
                try:
                    # Normalize DD date - handle both YYYY-MM-DD and YYYYMMDD formats
                    if isinstance(dd['date'], str):
                        if len(dd['date']) == 8:  # YYYYMMDD format
                            dd_date_str = f"{dd['date'][:4]}-{dd['date'][4:6]}-{dd['date'][6:8]}"
                        else:  # Assume YYYY-MM-DD format
                            dd_date_str = dd['date']
                    else:
                        dd_date_str = pd.to_datetime(dd['date']).strftime('%Y-%m-%d')
                    
                    # Find matching data point
                    matching_data = chart_data[chart_data['Date_normalized'] == dd_date_str]
                    
                    if not matching_data.empty:
                        # Get position in chart
                        chart_position = matching_data.index[0]
                        row_position = chart_data.reset_index().index[chart_data.index == chart_position][0]
                        price = matching_data['Low'].iloc[0] - (matching_data['Low'].iloc[0] * 0.01)  # Position below low
                        
                        # Marker size based on severity (larger for more recent)
                        base_size = 200 - (i * 30)  # Decreasing size: 200, 170, 140, 110, 80
                        severity_multiplier = {'Severe': 1.2, 'Moderate': 1.0, 'Mild': 0.8}
                        final_size = base_size * severity_multiplier.get(dd.get('severity', 'Moderate'), 1.0)
                        
                        # Add downward arrow marker
                        ax.scatter(row_position, price, color='red', marker='v', 
                                 s=final_size, zorder=15, edgecolor='darkred', linewidth=2, alpha=0.9)
                        
                        # Add number annotation
                        ax.annotate(str(i + 1), (row_position, price), 
                                  xytext=(0, -15), textcoords='offset points',
                                  ha='center', va='top', fontsize=11, fontweight='bold',
                                  color='white',
                                  bbox=dict(boxstyle='circle,pad=0.3', facecolor='darkred', alpha=0.9, edgecolor='white'))
                        
                        dd_markers_added += 1
                        logger.info(f"Added DD marker #{i+1} at position {row_position}, price {price:.2f} for date {dd_date_str}")
                        
                    else:
                        logger.warning(f"No matching data for DD date {dd_date_str}")
                        
                except Exception as e:
                    logger.error(f"Error adding DD marker for {dd.get('date', 'unknown')}: {e}")
            
            # Add legend for markers with counts
            if latest_ftd or latest_dd:
                legend_elements = []
                if latest_ftd:
                    legend_elements.append(plt.scatter([], [], color='limegreen', marker='^', s=120, 
                                                     label=f'Latest 5 FTD (Shown: {ftd_markers_added})', 
                                                     edgecolor='darkgreen', linewidth=1))
                if latest_dd:
                    legend_elements.append(plt.scatter([], [], color='red', marker='v', s=120, 
                                                     label=f'Latest 5 DD (Shown: {dd_markers_added})', 
                                                     edgecolor='darkred', linewidth=1))
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                         fancybox=True, shadow=True)
            
            logger.info(f"Signal markers summary: {ftd_markers_added} FTD markers, {dd_markers_added} DD markers added")
            
        except Exception as e:
            logger.error(f"Error adding enhanced signal markers: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _export_detailed_signals(self, index_results: Dict, timeframe: str, ticker_choice: str, data_date: str, index: str) -> Optional[str]:
        """
        Export detailed FTD/DD signals to JSON file with complete metadata.
        
        Args:
            index_results: Analysis results containing signal data
            timeframe: Data timeframe (daily, weekly, monthly)  
            ticker_choice: User ticker choice
            data_date: Data date for file naming
            index: Index symbol
            
        Returns:
            Path to saved signals file or None if export fails
        """
        try:
            # Get signal data
            ftd_days = index_results.get('follow_through_days', [])
            dd_days = index_results.get('distribution_days', [])
            
            if not ftd_days and not dd_days:
                logger.info(f"No signals to export for {index}")
                return None
            
            # Sort signals by date (most recent first) and get latest 20 of each type
            latest_20_ftd = sorted(ftd_days, key=lambda x: x['date'], reverse=True)[:20]
            latest_20_dd = sorted(dd_days, key=lambda x: x['date'], reverse=True)[:20]
            
            # Create comprehensive signal export data
            signal_export = {
                'metadata': {
                    'index': index,
                    'timeframe': timeframe,
                    'ticker_choice': ticker_choice,
                    'data_date': data_date,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'analysis_period_days': self.analysis_period,
                    'recent_activity_period_days': self.recent_activity_period,
                    'ftd_price_threshold_pct': self.ftd_price_threshold,
                    'ftd_volume_threshold': self.ftd_volume_threshold,
                    'dd_price_threshold_pct': self.dd_price_threshold,
                    'dd_volume_threshold': self.dd_volume_threshold
                },
                'summary_statistics': {
                    'total_ftd_signals': len(ftd_days),
                    'total_dd_signals': len(dd_days),
                    'latest_20_ftd_count': len(latest_20_ftd),
                    'latest_20_dd_count': len(latest_20_dd),
                    'market_state': index_results['market_state']['current_state'],
                    'confidence': index_results['market_state']['confidence'],
                    'latest_signal': index_results['latest_signal'],
                    'signal_strength': index_results['signal_strength'],
                    'ftd_dd_ratio': index_results['market_state']['ftd_dd_ratio']
                },
                'follow_through_days': {
                    'description': 'Latest 20 Follow-Through Days (FTDs) - bullish signals with strong volume',
                    'criteria': {
                        'min_price_increase_pct': self.ftd_price_threshold,
                        'min_volume_ratio': self.ftd_volume_threshold,
                        'lookback_period_days': self.analysis_period
                    },
                    'signals': []
                },
                'distribution_days': {
                    'description': 'Latest 20 Distribution Days (DDs) - bearish signals with heavy selling',
                    'criteria': {
                        'max_price_decline_pct': self.dd_price_threshold,
                        'min_volume_ratio': self.dd_volume_threshold,
                        'lookback_period_days': self.analysis_period
                    },
                    'signals': []
                }
            }
            
            # Add detailed FTD signal data
            for i, ftd in enumerate(latest_20_ftd):
                signal_detail = {
                    'sequence_number': i + 1,
                    'date': ftd['date'],
                    'signal_type': 'FTD',
                    'price_change_pct': ftd.get('price_change_pct', 0),
                    'volume_ratio': ftd.get('volume_ratio', 0),
                    'strength': ftd.get('strength', 'Unknown'),
                    'close_price': ftd.get('close_price', 0),
                    'volume': ftd.get('volume', 0),
                    'volume_50ma': ftd.get('volume_50ma', 0),
                    'days_ago': self._calculate_days_ago(ftd['date'], data_date)
                }
                signal_export['follow_through_days']['signals'].append(signal_detail)
            
            # Add detailed DD signal data  
            for i, dd in enumerate(latest_20_dd):
                signal_detail = {
                    'sequence_number': i + 1,
                    'date': dd['date'],
                    'signal_type': 'DD',
                    'price_change_pct': dd.get('price_change_pct', 0),
                    'volume_ratio': dd.get('volume_ratio', 0),
                    'severity': dd.get('severity', 'Unknown'),
                    'close_price': dd.get('close_price', 0),
                    'volume': dd.get('volume', 0),
                    'volume_50ma': dd.get('volume_50ma', 0),
                    'days_ago': self._calculate_days_ago(dd['date'], data_date)
                }
                signal_export['distribution_days']['signals'].append(signal_detail)
            
            # Save to JSON file
            base_results_dir = Path(self.paths.get('results', 'results'))
            output_dir = base_results_dir / 'market_pulse'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            signals_filename = f"ftd_dd_signals_{index}_{ticker_choice}_{timeframe}_{data_date}.json"
            signals_path = output_dir / signals_filename
            
            with open(signals_path, 'w') as f:
                json.dump(signal_export, f, indent=2, default=str)
            
            logger.info(f"Detailed FTD/DD signals for {index} exported to: {signals_path}")
            logger.info(f"Export summary - FTDs: {len(latest_20_ftd)}, DDs: {len(latest_20_dd)}")
            
            return str(signals_path)
            
        except Exception as e:
            logger.error(f"Error exporting detailed signals for {index}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _calculate_days_ago(self, signal_date: str, reference_date: str) -> int:
        """Calculate how many days ago a signal occurred."""
        try:
            # Handle different date formats
            if isinstance(signal_date, str):
                if len(signal_date) == 8:  # YYYYMMDD
                    signal_dt = datetime.strptime(signal_date, '%Y%m%d')
                else:  # YYYY-MM-DD
                    signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
            else:
                signal_dt = pd.to_datetime(signal_date)
                
            if isinstance(reference_date, str):
                if len(reference_date) == 8:  # YYYYMMDD
                    reference_dt = datetime.strptime(reference_date, '%Y%m%d')
                else:  # YYYY-MM-DD
                    reference_dt = datetime.strptime(reference_date, '%Y-%m-%d')
            else:
                reference_dt = pd.to_datetime(reference_date)
                
            delta = (reference_dt - signal_dt).days
            return max(0, delta)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Error calculating days ago for {signal_date}: {e}")
            return 0
    
    def _add_signal_markers(self, ax, chart_data: pd.DataFrame, index_results: Dict, dates):
        """
        Add FTD and DD signal markers to the price chart.
        
        Args:
            ax: Matplotlib axis
            chart_data: Chart data with dates and prices
            index_results: Analysis results containing signals
            dates: Date series for the chart
        """
        try:
            # Get chart date range for filtering signals
            chart_start_date = dates.min().strftime('%Y-%m-%d')
            
            # Add FTD markers (green up arrows)
            ftd_days = index_results.get('follow_through_days', [])
            for ftd in ftd_days:
                ftd_date = pd.to_datetime(ftd['date'])
                if ftd_date >= pd.to_datetime(chart_start_date):
                    # Find matching data point
                    matching_data = chart_data[chart_data['Date'] == ftd_date]
                    if not matching_data.empty:
                        price = matching_data['Close'].iloc[0]
                        ax.scatter(ftd_date, price, color='green', marker='^', s=100, 
                                 label='FTD' if ftd == ftd_days[0] else '', zorder=5)
                        # Add strength annotation
                        ax.annotate(f"FTD\\n{ftd['strength']}", (ftd_date, price), 
                                  xytext=(5, 10), textcoords='offset points',
                                  fontsize=8, ha='left', 
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            
            # Add DD markers (red down arrows)
            dd_days = index_results.get('distribution_days', [])
            for dd in dd_days:
                dd_date = pd.to_datetime(dd['date'])
                if dd_date >= pd.to_datetime(chart_start_date):
                    # Find matching data point
                    matching_data = chart_data[chart_data['Date'] == dd_date]
                    if not matching_data.empty:
                        price = matching_data['Close'].iloc[0]
                        ax.scatter(dd_date, price, color='red', marker='v', s=100, 
                                 label='DD' if dd == dd_days[0] else '', zorder=5)
                        # Add severity annotation
                        ax.annotate(f"DD\\n{dd['severity']}", (dd_date, price), 
                                  xytext=(5, -15), textcoords='offset points',
                                  fontsize=8, ha='left',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
                        
        except Exception as e:
            logger.error(f"Error adding signal markers: {e}")
    
    def _add_metrics_table(self, fig, index_results: Dict):
        """
        Add enhanced metrics table to the chart with latest 5 signals.
        
        Args:
            fig: Matplotlib figure
            index_results: Analysis results
        """
        try:
            market_state = index_results['market_state']
            summary = index_results['summary']
            
            # Get latest 5 FTD and DD dates for display
            ftd_days = index_results.get('follow_through_days', [])
            dd_days = index_results.get('distribution_days', [])
            
            latest_5_ftd = sorted(ftd_days, key=lambda x: x['date'], reverse=True)[:5]
            latest_5_dd = sorted(dd_days, key=lambda x: x['date'], reverse=True)[:5]
            
            # Format dates for display
            def format_signal_dates(signals):
                if not signals:
                    return "None"
                dates = []
                for i, signal in enumerate(signals):
                    date_str = signal['date']
                    # Convert YYYYMMDD to MM/DD format if needed
                    if len(date_str) == 8:
                        date_display = f"{date_str[4:6]}/{date_str[6:8]}"
                    elif len(date_str) == 10:  # YYYY-MM-DD
                        date_display = f"{date_str[5:7]}/{date_str[8:10]}"
                    else:
                        date_display = date_str[-5:]  # Last 5 chars
                    dates.append(f"{i+1}:{date_display}")
                return ", ".join(dates)
            
            # Create enhanced metrics data
            metrics_data = [
                ['Market State', f"{market_state['current_state']}\\nConfidence: {market_state['confidence']}"],
                ['Recent FTDs', f"{market_state['recent_ftd_count']} (last {self.recent_activity_period} days)\\nRecent DDs: {market_state['recent_dd_count']} (last {self.recent_activity_period} days)"],
                ['FTD/DD Ratio', f"{market_state['ftd_dd_ratio']:.2f}\\nSignal Strength: {index_results['signal_strength']}"],
                ['Latest Signal', f"{index_results['latest_signal']}\\nTotal FTDs: {summary['total_ftd_count']}, Total DDs: {summary['total_dd_count']}"],
                ['Latest 5 FTDs', format_signal_dates(latest_5_ftd)],
                ['Latest 5 DDs', format_signal_dates(latest_5_dd)]
            ]
            
            # Create metrics text
            metrics_text = '\\n\\n'.join([f'{label}: {value}' for label, value in metrics_data])
            
            # Add enhanced table to figure
            fig.text(0.02, 0.02, metrics_text,
                    fontsize=8, verticalalignment='bottom', 
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
            
        except Exception as e:
            logger.error(f"Error adding enhanced metrics table: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")