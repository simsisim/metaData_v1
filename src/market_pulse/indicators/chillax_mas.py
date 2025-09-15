"""
Chillax MA Analysis with Risk Zone Coloring
==========================================

Implements Chillax Moving Average analysis with risk-on/risk-off zone visualization:

Workflow:
1. Data Loading: Load historical OHLC data for specified indexes
2. MA Calculations: Calculate 10-period and 20-period SMAs 
3. Trend Analysis: Determine trend direction using 5-day comparison
4. Risk Zones: Define risk-on (light green) and risk-off (light red) zones
5. Visualization: Generate candlestick charts with colored background zones
6. Output: Save analysis results and charts with latest data date naming

Features:
- Risk-On Zone: SMA_10 > SMA_20 and both trending up or SMA_10 trending up alone
- Risk-Off Zone: SMA_10 < SMA_20 or weakening trend
- Configurable indexes, SMA periods, and chart timeframes
- Latest 3 months data restriction (configurable chart timeframe)
- Output files named with latest data date (not generation date)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
import os

# Chart libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import mplfinance as mpf
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    logging.warning("Chart libraries not available. Install matplotlib and mplfinance for chart generation.")

from .base_indicator import BaseIndicator

logger = logging.getLogger(__name__)


class ChillaxMAS(BaseIndicator):
    """
    Chillax Moving Average Analysis with Risk Zone Coloring.
    
    Analyzes market indexes using dual SMA system with risk zone identification
    and generates visualization charts with dynamic background coloring.
    """
    
    def __init__(self, target_indexes: List[str], config, user_config=None):
        """
        Initialize Chillax MA analyzer.
        
        Args:
            target_indexes: List of market indexes for analysis
            config: System configuration
            user_config: User configuration (optional)
        """
        # Use first index as primary symbol for base class
        super().__init__(target_indexes[0] if target_indexes else 'SPY', config, user_config)
        
        self.target_indexes = target_indexes
        
        # Load Chillax MA configuration from user_data.csv
        self.chillax_indexes = self._parse_chillax_indexes(
            getattr(user_config, 'chillax_mas_indexes', 'SPY;QQQ;IWM')
        )
        
        self.sma_periods = self._parse_sma_periods(
            getattr(user_config, 'chillax_mas_sma', '10;20')
        )
        
        self.chart_indexes = self._parse_chart_indexes(
            getattr(user_config, 'chillax_mas_charts', ''),
            self.chillax_indexes
        )
        
        self.chart_timeframe = int(getattr(user_config, 'chillax_mas_charts_timeframe', 150))
        self.trend_days = int(getattr(user_config, 'market_pulse_chillax_trend_days', 5))
        
        # Ensure we have exactly 2 SMA periods
        if len(self.sma_periods) != 2:
            logger.warning(f"Expected 2 SMA periods, got {len(self.sma_periods)}. Using default [10, 20]")
            self.sma_periods = [10, 20]
        
        self.fast_sma = min(self.sma_periods)
        self.slow_sma = max(self.sma_periods)
        
        logger.info(f"ChillaxMAS initialized: indexes={self.chillax_indexes}, SMAs=[{self.fast_sma}, {self.slow_sma}], chart_timeframe={self.chart_timeframe}")
    
    def _parse_chillax_indexes(self, indexes_str: str) -> List[str]:
        """Parse chillax_mas_indexes configuration string."""
        if not indexes_str:
            return ['SPY', 'QQQ', 'IWM']
        return [idx.strip().upper() for idx in indexes_str.split(';') if idx.strip()]
    
    def _parse_sma_periods(self, sma_str: str) -> List[int]:
        """Parse chillax_mas_sma configuration string."""
        if not sma_str:
            return [10, 20]
        try:
            return [int(period.strip()) for period in sma_str.split(';') if period.strip()]
        except ValueError as e:
            logger.error(f"Error parsing SMA periods '{sma_str}': {e}")
            return [10, 20]
    
    def _parse_chart_indexes(self, charts_str: str, all_indexes: List[str]) -> List[str]:
        """Parse chillax_mas_charts configuration string."""
        if not charts_str:
            return all_indexes  # Create charts for all indexes if not specified
        return [idx.strip().upper() for idx in charts_str.split(';') if idx.strip()]
    
    def run_analysis(self, timeframe: str = 'daily', data_date: str = None) -> Dict[str, Any]:
        """
        Run complete Chillax MA analysis with risk zone coloring.
        
        Args:
            timeframe: Data timeframe ('daily' recommended)
            data_date: Date for output naming (uses latest data date if None)
            
        Returns:
            Dictionary containing analysis results and file paths
        """
        try:
            # Load market data for all configured indexes
            market_data = self._load_market_data()
            if not market_data:
                return {
                    'success': False,
                    'error': 'No market data available for any configured index',
                    'timeframe': timeframe
                }
            
            # Run analysis for each index
            analysis_results = {}
            latest_data_date = None
            
            for index in self.chillax_indexes:
                if index not in market_data:
                    logger.warning(f"No data available for index {index}")
                    continue
                    
                # Analyze index with risk zone determination
                index_analysis = self._analyze_index(market_data[index], index)
                if index_analysis['success']:
                    analysis_results[index] = index_analysis
                    
                    # Track latest data date
                    if latest_data_date is None or index_analysis['latest_date'] > latest_data_date:
                        latest_data_date = index_analysis['latest_date']
            
            if not analysis_results:
                return {
                    'success': False,
                    'error': 'No successful analysis results for any index',
                    'timeframe': timeframe
                }
            
            # Generate charts for specified indexes
            chart_files = {}
            if CHARTS_AVAILABLE:
                for index in self.chart_indexes:
                    if index in market_data and index in analysis_results:
                        chart_path = self._generate_chart(
                            market_data[index], 
                            analysis_results[index], 
                            index, 
                            timeframe, 
                            latest_data_date
                        )
                        if chart_path:
                            chart_files[index] = chart_path
            else:
                logger.warning("Chart generation skipped - matplotlib/mplfinance not available")
            
            # Save analysis results to CSV
            output_file = self._save_analysis_results(
                analysis_results, 
                timeframe, 
                latest_data_date
            )
            
            # Generate summary
            summary = self._generate_summary(analysis_results)
            
            return {
                'success': True,
                'timeframe': timeframe,
                'analysis_date': data_date or datetime.now().strftime('%Y-%m-%d'),
                'latest_data_date': latest_data_date,
                'indexes_analyzed': list(analysis_results.keys()),
                'chart_indexes': list(chart_files.keys()),
                'analysis_results': analysis_results,
                'chart_files': chart_files,
                'output_file': output_file,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Chillax MA analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timeframe': timeframe
            }
    
    def _load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for all configured indexes."""
        market_data = {}
        
        for index in self.chillax_indexes:
            try:
                # Load index data using environment-aware path resolution
                file_path = self.config.get_market_data_dir('daily') / f"{index}.csv"
                
                if not file_path.exists():
                    logger.warning(f"Market data file not found: {file_path}")
                    continue
                    
                df = pd.read_csv(file_path, index_col='Date', parse_dates=False)
                
                # Clean and standardize date index
                df.index = df.index.str.split(' ').str[0]
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Ensure required columns exist
                required_columns = ['Open', 'High', 'Low', 'Close']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.warning(f"Missing columns in {index}: {missing_columns}")
                    continue
                
                # Restrict to chart timeframe (e.g., last 150 trading days)
                if len(df) > self.chart_timeframe:
                    df = df.tail(self.chart_timeframe)
                
                market_data[index] = df
                logger.info(f"Loaded {len(df)} records for {index}")
                
            except Exception as e:
                logger.error(f"Error loading data for {index}: {e}")
                continue
        
        return market_data
    
    def _analyze_index(self, data: pd.DataFrame, index: str) -> Dict[str, Any]:
        """
        Analyze single index with moving averages and risk zone determination.
        
        Args:
            data: Price data DataFrame with OHLC columns
            index: Index symbol
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Calculate moving averages
            df[f'SMA_{self.fast_sma}'] = df['Close'].rolling(window=self.fast_sma).mean()
            df[f'SMA_{self.slow_sma}'] = df['Close'].rolling(window=self.slow_sma).mean()
            
            # Calculate trend conditions (5-day comparison)
            df['fast_trending_up'] = df[f'SMA_{self.fast_sma}'] > df[f'SMA_{self.fast_sma}'].shift(self.trend_days)
            df['slow_trending_up'] = df[f'SMA_{self.slow_sma}'] > df[f'SMA_{self.slow_sma}'].shift(self.trend_days)
            df['fast_above_slow'] = df[f'SMA_{self.fast_sma}'] > df[f'SMA_{self.slow_sma}']
            
            # Determine risk zones
            df['risk_zone'] = df.apply(self._determine_risk_zone, axis=1)
            
            # Get latest values
            latest = df.iloc[-1]
            latest_date = df.index[-1].strftime('%Y%m%d')
            
            # Calculate zone statistics
            zone_stats = self._calculate_zone_statistics(df)
            
            return {
                'success': True,
                'index': index,
                'latest_date': latest_date,
                'current_price': round(latest['Close'], 2),
                'fast_sma': round(latest[f'SMA_{self.fast_sma}'], 2),
                'slow_sma': round(latest[f'SMA_{self.slow_sma}'], 2),
                'fast_trending_up': bool(latest['fast_trending_up']),
                'slow_trending_up': bool(latest['slow_trending_up']),
                'fast_above_slow': bool(latest['fast_above_slow']),
                'current_risk_zone': latest['risk_zone'],
                'zone_statistics': zone_stats,
                'data': df  # Include processed data for charting
            }
            
        except Exception as e:
            logger.error(f"Error analyzing index {index}: {e}")
            return {
                'success': False,
                'index': index,
                'error': str(e)
            }
    
    def _determine_risk_zone(self, row) -> str:
        """
        Determine Qullamaggie color state based on MA conditions.
        
        Qullamaggie Color System:
        - Dark Green: short_ma > long_ma AND both_trending_up
        - Light Green: short_ma > long_ma AND only_short_ma_trending_up  
        - Yellow: short_ma > long_ma AND neither_trending_up
        - Light Red: short_ma < long_ma AND short_ma_trending_up
        - Dark Red: short_ma < long_ma AND neither_trending_up
        """
        short_above_long = row['fast_above_slow']  # 10 > 20
        short_trending_up = row['fast_trending_up'] 
        long_trending_up = row['slow_trending_up']
        
        if short_above_long:
            if short_trending_up and long_trending_up:
                return 'DARK_GREEN'      # Both trending up - strongest bullish
            elif short_trending_up:
                return 'LIGHT_GREEN'     # Only short trending up - moderate bullish  
            else:
                return 'YELLOW'          # Neither trending up - weakening bullish
        else:  # short < long
            if short_trending_up:
                return 'LIGHT_RED'       # Short trying to recover - potential reversal
            else:
                return 'DARK_RED'        # Both declining/flat - strongest bearish
    
    def _calculate_zone_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about Qullamaggie color zones."""
        try:
            total_days = len(df.dropna())
            zone_counts = df['risk_zone'].value_counts()
            
            return {
                'total_days': total_days,
                'dark_green_days': zone_counts.get('DARK_GREEN', 0),
                'light_green_days': zone_counts.get('LIGHT_GREEN', 0),
                'yellow_days': zone_counts.get('YELLOW', 0),
                'light_red_days': zone_counts.get('LIGHT_RED', 0),
                'dark_red_days': zone_counts.get('DARK_RED', 0),
                'bullish_percentage': round((zone_counts.get('DARK_GREEN', 0) + zone_counts.get('LIGHT_GREEN', 0) + zone_counts.get('YELLOW', 0)) / total_days * 100, 1),
                'bearish_percentage': round((zone_counts.get('LIGHT_RED', 0) + zone_counts.get('DARK_RED', 0)) / total_days * 100, 1),
                'strong_trend_percentage': round((zone_counts.get('DARK_GREEN', 0) + zone_counts.get('DARK_RED', 0)) / total_days * 100, 1)
            }
        except Exception as e:
            logger.error(f"Error calculating zone statistics: {e}")
            return {}
    
    def _generate_chart(self, data: pd.DataFrame, analysis: Dict[str, Any], 
                       index: str, timeframe: str, latest_date: str) -> Optional[str]:
        """
        Generate candlestick chart with Qullamaggie coloring system.
        
        Args:
            data: Price data with analysis results
            analysis: Analysis results dictionary  
            index: Index symbol
            timeframe: Data timeframe
            latest_date: Latest data date for filename
            
        Returns:
            Path to generated chart file or None if failed
        """
        try:
            if not CHARTS_AVAILABLE:
                return None
                
            # Get processed data from analysis
            df = analysis['data'].copy()
            df = df.dropna()
            
            if len(df) == 0:
                logger.warning(f"No valid data for charting {index}")
                return None
            
            # Define Qullamaggie color mapping
            qullamaggie_colors = {
                'DARK_GREEN': '#006400',    # Dark green - strongest bullish
                'LIGHT_GREEN': '#90EE90',   # Light green - moderate bullish  
                'YELLOW': '#FFD700',        # Yellow - weakening bullish
                'LIGHT_RED': '#FFB6C1',     # Light red - potential reversal
                'DARK_RED': '#8B0000'       # Dark red - strongest bearish
            }
            
            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
            ax.set_facecolor('white')
            
            # Draw candlesticks with Qullamaggie coloring
            for i, (date, row) in enumerate(df.iterrows()):
                open_price = row['Open']
                high_price = row['High'] 
                low_price = row['Low']
                close_price = row['Close']
                color_state = row.get('risk_zone', 'YELLOW')
                
                # Get color for this candlestick
                candle_color = qullamaggie_colors.get(color_state, '#FFD700')
                
                # Draw the high-low line (wick)
                ax.plot([i, i], [low_price, high_price], color=candle_color, linewidth=1, alpha=0.8)
                
                # Draw the open-close body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                # Create candlestick body rectangle
                body = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                               facecolor=candle_color, edgecolor=candle_color, 
                               alpha=0.8 if close_price >= open_price else 0.6)
                ax.add_patch(body)
            
            # Add moving averages with dynamic coloring based on overall trend
            fast_ma_values = df[f'SMA_{self.fast_sma}']
            slow_ma_values = df[f'SMA_{self.slow_sma}']
            
            # Color MAs based on current state
            current_state = df.iloc[-1]['risk_zone'] if len(df) > 0 else 'YELLOW'
            ma_color_map = {
                'DARK_GREEN': '#006400',
                'LIGHT_GREEN': '#32CD32', 
                'YELLOW': '#FFA500',
                'LIGHT_RED': '#FF6347',
                'DARK_RED': '#DC143C'
            }
            
            ma_color = ma_color_map.get(current_state, '#FFA500')
            
            # Plot moving averages
            ax.plot(range(len(df)), fast_ma_values, color=ma_color, linewidth=2.5, 
                   label=f'SMA {self.fast_sma}', alpha=0.9)
            ax.plot(range(len(df)), slow_ma_values, color=ma_color, linewidth=2, 
                   label=f'SMA {self.slow_sma}', alpha=0.7, linestyle='--')
            
            # Formatting
            ax.set_title(f'{index} - Qullamaggie Moving Average Coloring System', 
                        fontsize=16, fontweight='bold', color='black')
            ax.set_ylabel('Price', fontsize=12, color='black')
            ax.set_xlabel('Trading Days', fontsize=12, color='black')
            
            # Set x-axis labels with dates (sample every 10th date)
            date_labels = df.index.strftime('%Y-%m-%d')
            step = max(1, len(date_labels) // 10)
            ax.set_xticks(range(0, len(date_labels), step))
            ax.set_xticklabels(date_labels[::step], rotation=45, ha='right')
            
            # Grid
            ax.grid(True, alpha=0.3, color='#CCCCCC')
            ax.tick_params(colors='black')
            
            # Legend for Qullamaggie colors
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='#006400', alpha=0.8, label='Dark Green (Strong Bull)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', alpha=0.8, label='Light Green (Moderate Bull)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#FFD700', alpha=0.8, label='Yellow (Weak Bull)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#FFB6C1', alpha=0.8, label='Light Red (Potential Reversal)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#8B0000', alpha=0.8, label='Dark Red (Strong Bear)')
            ]
            
            # Add MA lines to legend
            legend_elements.extend([
                plt.Line2D([0], [0], color=ma_color, linewidth=2.5, label=f'SMA {self.fast_sma}'),
                plt.Line2D([0], [0], color=ma_color, linewidth=2, linestyle='--', label=f'SMA {self.slow_sma}')
            ])
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)
            
            # Create output filename
            user_choice = str(self.user_config.ticker_choice) if self.user_config else '0-5'
            chart_filename = f"chillax_mas_{index.lower()}_{user_choice}_{timeframe}_{latest_date}.png"
            
            # Ensure output directory exists
            output_dir = self.config.directories['RESULTS_DIR'] / 'market_pulse'
            output_dir.mkdir(parents=True, exist_ok=True)
            chart_path = output_dir / chart_filename
            
            # Save the chart
            plt.tight_layout()
            plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close to free memory
            
            logger.info(f"Qullamaggie chart generated: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error generating Qullamaggie chart for {index}: {e}")
            return None
    
    def _save_analysis_results(self, analysis_results: Dict[str, Dict], 
                             timeframe: str, latest_date: str) -> Optional[str]:
        """
        Save analysis results to CSV file.
        
        Args:
            analysis_results: Dictionary of analysis results per index
            timeframe: Data timeframe
            latest_date: Latest data date for filename
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Create output data
            output_rows = []
            
            for index, results in analysis_results.items():
                row = {
                    'date': latest_date,
                    'timeframe': timeframe,
                    'index': index,
                    'current_price': results['current_price'],
                    'fast_sma': results['fast_sma'],
                    'slow_sma': results['slow_sma'],
                    'fast_sma_period': self.fast_sma,
                    'slow_sma_period': self.slow_sma,
                    'fast_trending_up': results['fast_trending_up'],
                    'slow_trending_up': results['slow_trending_up'],
                    'fast_above_slow': results['fast_above_slow'],
                    'current_risk_zone': results['current_risk_zone'],
                    'trend_lookback_days': self.trend_days,
                    'chart_timeframe': self.chart_timeframe
                }
                
                # Add Qullamaggie zone statistics
                zone_stats = results.get('zone_statistics', {})
                row.update({
                    'total_analysis_days': zone_stats.get('total_days', 0),
                    'dark_green_days': zone_stats.get('dark_green_days', 0),
                    'light_green_days': zone_stats.get('light_green_days', 0),
                    'yellow_days': zone_stats.get('yellow_days', 0),
                    'light_red_days': zone_stats.get('light_red_days', 0),
                    'dark_red_days': zone_stats.get('dark_red_days', 0),
                    'bullish_percentage': zone_stats.get('bullish_percentage', 0),
                    'bearish_percentage': zone_stats.get('bearish_percentage', 0),
                    'strong_trend_percentage': zone_stats.get('strong_trend_percentage', 0)
                })
                
                output_rows.append(row)
            
            # Create DataFrame
            df_output = pd.DataFrame(output_rows)
            
            # Create output filename
            indexes_str = '+'.join([idx.lower() for idx in sorted(analysis_results.keys())])
            user_choice = str(self.user_config.ticker_choice) if self.user_config else '0-5'
            output_filename = f"chillax_mas_{indexes_str}_{user_choice}_{timeframe}_{latest_date}.csv"
            
            # Ensure output directory exists
            output_dir = self.config.directories['RESULTS_DIR'] / 'market_pulse'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename
            
            # Save to CSV
            df_output.to_csv(output_path, index=False)
            
            logger.info(f"Analysis results saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            return None
    
    def _generate_summary(self, analysis_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary of Qullamaggie MA analysis."""
        try:
            total_indexes = len(analysis_results)
            
            # Count indexes by Qullamaggie color states
            bullish_indexes = sum(1 for r in analysis_results.values() 
                                if r['current_risk_zone'] in ['DARK_GREEN', 'LIGHT_GREEN', 'YELLOW'])
            bearish_indexes = sum(1 for r in analysis_results.values() 
                                if r['current_risk_zone'] in ['LIGHT_RED', 'DARK_RED'])
            strong_trend_indexes = sum(1 for r in analysis_results.values() 
                                     if r['current_risk_zone'] in ['DARK_GREEN', 'DARK_RED'])
            
            # Calculate average zone percentages
            avg_bullish_pct = np.mean([
                r.get('zone_statistics', {}).get('bullish_percentage', 0) 
                for r in analysis_results.values()
            ])
            
            avg_bearish_pct = np.mean([
                r.get('zone_statistics', {}).get('bearish_percentage', 0) 
                for r in analysis_results.values()
            ])
            
            avg_strong_trend_pct = np.mean([
                r.get('zone_statistics', {}).get('strong_trend_percentage', 0) 
                for r in analysis_results.values()
            ])
            
            return {
                'total_indexes_analyzed': total_indexes,
                'bullish_indexes': bullish_indexes,
                'bearish_indexes': bearish_indexes,
                'strong_trend_indexes': strong_trend_indexes,
                'bullish_percentage': round(bullish_indexes / total_indexes * 100, 1) if total_indexes > 0 else 0,
                'bearish_percentage': round(bearish_indexes / total_indexes * 100, 1) if total_indexes > 0 else 0,
                'strong_trend_percentage': round(strong_trend_indexes / total_indexes * 100, 1) if total_indexes > 0 else 0,
                'average_bullish_days_percentage': round(avg_bullish_pct, 1),
                'average_bearish_days_percentage': round(avg_bearish_pct, 1),
                'average_strong_trend_days_percentage': round(avg_strong_trend_pct, 1),
                'fast_sma_period': self.fast_sma,
                'slow_sma_period': self.slow_sma,
                'trend_confirmation_days': self.trend_days,
                'analysis_timeframe_days': self.chart_timeframe,
                'qullamaggie_color_system': True
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {'error': str(e)}