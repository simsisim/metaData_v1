"""
Moving Average Cycles Analyzer
==============================

Implements Moving Average Cycles analysis with comprehensive cycle tracking and visualization:

Features:
- Bull/Bear cycle detection based on price-MA relationship
- Sharp and Smoothed cycle detection modes
- Comprehensive cycle statistics (length, distance, averages, maximums)
- Dual histogram visualization with metrics table overlay
- Configurable MA period and detection parameters
- Latest data date naming convention

Algorithm:
1. Calculate configurable SMA (default 50-period)
2. Track price-MA crossovers to identify cycle boundaries
3. Monitor cycle length (candles) and maximum % distance from MA
4. Maintain historical statistics for bull and bear cycles
5. Generate charts with price/MA overlay and cycle histogram panel
6. Display metrics table with current and historical cycle data

Based on TradingView Pine Script: Moving Average Cycles by TradingView
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
    import matplotlib.gridspec as gridspec
    import mplfinance as mpf
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    logging.warning("Chart libraries not available. Install matplotlib and mplfinance for chart generation.")

from .base_indicator import BaseIndicator

logger = logging.getLogger(__name__)


class MACyclesAnalyzer(BaseIndicator):
    """
    Moving Average Cycles Analyzer with comprehensive cycle tracking.
    
    Analyzes market indexes using MA crossovers to identify bull/bear cycles
    and generates detailed cycle statistics with visualization.
    """
    
    def __init__(self, target_indexes: List[str], config, user_config=None):
        """
        Initialize MA Cycles analyzer.
        
        Args:
            target_indexes: List of market indexes for analysis
            config: System configuration
            user_config: User configuration (optional)
        """
        # Use first index as primary symbol for base class
        super().__init__(target_indexes[0] if target_indexes else 'SPY', config, user_config)
        
        self.target_indexes = target_indexes
        
        # Load MA Cycles configuration from user_data.csv
        self.ma_cycles_indexes = self._parse_cycles_indexes(
            getattr(user_config, 'ma_cycles_indexes', 'SPY;QQQ;IWM')
        )
        
        # Parse MA periods configuration
        ma_periods_str = getattr(user_config, 'market_pulse_ma_cycles_ma_period', '20;50')
        self.ma_periods = self._parse_ma_periods(ma_periods_str)
        # Use first period as primary for backward compatibility
        self.ma_period = self.ma_periods[0] if self.ma_periods else 50
        
        self.chart_indexes = self._parse_chart_indexes(
            getattr(user_config, 'ma_cycles_charts', ''),
            self.ma_cycles_indexes
        )
        
        self.chart_timeframe = int(getattr(user_config, 'ma_cycles_charts_timeframe', 200))
        self.cycle_mode = getattr(user_config, 'ma_cycles_cycle_mode', 'Sharp')
        self.smoothed_candles = int(getattr(user_config, 'ma_cycles_smoothed_candles', 3))
        
        logger.info(f"MACyclesAnalyzer initialized: indexes={self.ma_cycles_indexes}, MA={self.ma_period}, mode={self.cycle_mode}, timeframe={self.chart_timeframe}")
    
    def _parse_cycles_indexes(self, indexes_str: str) -> List[str]:
        """Parse ma_cycles_indexes configuration string."""
        if not indexes_str:
            return ['SPY', 'QQQ', 'IWM']
        return [idx.strip().upper() for idx in indexes_str.split(';') if idx.strip()]

    def _parse_ma_periods(self, ma_str: str) -> List[int]:
        """Parse ma_cycles_ma_period configuration string."""
        if not ma_str:
            return [20, 50]
        try:
            periods = [int(period.strip()) for period in ma_str.split(';') if period.strip()]
            return sorted(periods) if periods else [20, 50]
        except ValueError as e:
            logger.error(f"Error parsing MA periods '{ma_str}': {e}")
            return [20, 50]
    
    def _parse_chart_indexes(self, charts_str: str, all_indexes: List[str]) -> List[str]:
        """Parse ma_cycles_charts configuration string."""
        if not charts_str:
            return all_indexes  # Create charts for all indexes if not specified
        return [idx.strip().upper() for idx in charts_str.split(';') if idx.strip()]
    
    def run_analysis(self, timeframe: str = 'daily', data_date: str = None) -> Dict[str, Any]:
        """
        Run complete MA Cycles analysis.
        
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
            
            for index in self.ma_cycles_indexes:
                if index not in market_data:
                    logger.warning(f"No data available for index {index}")
                    continue
                    
                # Analyze index cycles
                index_analysis = self._analyze_cycles(market_data[index], index)
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
            logger.error(f"MA Cycles analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timeframe': timeframe
            }
    
    def _load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for all configured indexes."""
        market_data = {}
        
        for index in self.ma_cycles_indexes:
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
                
                # Restrict to chart timeframe (e.g., last 200 trading days)
                if len(df) > self.chart_timeframe:
                    df = df.tail(self.chart_timeframe)
                
                market_data[index] = df
                logger.info(f"Loaded {len(df)} records for {index}")
                
            except Exception as e:
                logger.error(f"Error loading data for {index}: {e}")
                continue
        
        return market_data
    
    def _analyze_cycles(self, data: pd.DataFrame, index: str) -> Dict[str, Any]:
        """
        Analyze MA cycles for single index.
        
        Args:
            data: Price data DataFrame with OHLCV columns
            index: Index symbol
            
        Returns:
            Analysis results dictionary with cycle metrics
        """
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Calculate moving average
            df[f'MA_{self.ma_period}'] = df['Close'].rolling(window=self.ma_period).mean()
            
            # Drop rows where MA is NaN
            df = df.dropna()
            
            if len(df) < self.ma_period + 10:  # Need sufficient data
                return {
                    'success': False,
                    'index': index,
                    'error': 'Insufficient data for cycle analysis'
                }
            
            # Detect cycles based on mode
            cycle_data = self._detect_cycle_transitions(df)
            
            # Calculate comprehensive cycle metrics
            cycle_metrics = self._calculate_cycle_metrics(cycle_data)
            
            # Get latest values
            latest = cycle_data.iloc[-1]
            latest_date = df.index[-1].strftime('%Y%m%d')
            
            return {
                'success': True,
                'index': index,
                'latest_date': latest_date,
                'current_price': round(latest['Close'], 2),
                'current_ma': round(latest[f'MA_{self.ma_period}'], 2),
                'current_cycle_type': latest['cycle_type'],
                'current_candles': int(latest['cycle_candles']),
                'current_max_dist': round(latest['current_max_dist'], 2),
                'cycle_metrics': cycle_metrics,
                'data': cycle_data  # Include processed data for charting
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cycles for index {index}: {e}")
            return {
                'success': False,
                'index': index,
                'error': str(e)
            }
    
    def _detect_cycle_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect MA cycle transitions using Sharp or Smoothed mode.
        
        Args:
            df: DataFrame with Close and MA columns
            
        Returns:
            DataFrame with cycle information added
        """
        # Initialize cycle tracking variables
        cycle_data = df.copy()
        
        # Calculate price-MA relationship
        cycle_data['above_ma'] = cycle_data['Close'] > cycle_data[f'MA_{self.ma_period}']
        cycle_data['price_ma_sign'] = cycle_data['above_ma'].astype(int) * 2 - 1  # 1 for bull, -1 for bear
        
        # Initialize cycle tracking columns
        cycle_data['cycle_type'] = 'bull'  # 'bull' or 'bear'
        cycle_data['cycle_candles'] = 0
        cycle_data['ma_crossover_price'] = np.nan
        cycle_data['cycle_peak'] = np.nan
        cycle_data['cycle_trough'] = np.nan
        cycle_data['current_max_dist'] = 0.0
        cycle_data['percent_distance'] = 0.0
        
        if self.cycle_mode == 'Sharp':
            # Sharp mode: immediate cycle change on crossover
            cycle_data = self._detect_sharp_cycles(cycle_data)
        else:
            # Smoothed mode: requires confirmation
            cycle_data = self._detect_smoothed_cycles(cycle_data)
        
        return cycle_data
    
    def _detect_sharp_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect cycles using Sharp mode (immediate crossover detection)."""
        current_cycle = 1 if df['above_ma'].iloc[0] else -1
        cycle_candles = 0
        ma_crossover_price = df[f'MA_{self.ma_period}'].iloc[0]
        cycle_peak = df['High'].iloc[0] if current_cycle > 0 else np.nan
        cycle_trough = df['Low'].iloc[0] if current_cycle < 0 else np.nan
        current_max_dist = 0.0
        
        for i in range(len(df)):
            # Check for crossover
            if i > 0:
                prev_sign = 1 if df['above_ma'].iloc[i-1] else -1
                curr_sign = 1 if df['above_ma'].iloc[i] else -1
                
                if prev_sign != curr_sign:  # Crossover detected
                    # Reset cycle
                    current_cycle = curr_sign
                    cycle_candles = 1
                    ma_crossover_price = df[f'MA_{self.ma_period}'].iloc[i]
                    cycle_peak = df['High'].iloc[i] if current_cycle > 0 else np.nan
                    cycle_trough = df['Low'].iloc[i] if current_cycle < 0 else np.nan
                    current_max_dist = 0.0
                else:
                    cycle_candles += 1
            
            # Update cycle metrics
            if current_cycle > 0:  # Bull cycle
                if not np.isnan(cycle_peak):
                    cycle_peak = max(cycle_peak, df['High'].iloc[i])
                else:
                    cycle_peak = df['High'].iloc[i]
                    
                if not np.isnan(ma_crossover_price) and ma_crossover_price > 0:
                    dist = (cycle_peak - ma_crossover_price) / ma_crossover_price * 100
                    current_max_dist = max(current_max_dist, dist)
                    percent_distance = (df['High'].iloc[i] - ma_crossover_price) / ma_crossover_price * 100
                else:
                    percent_distance = 0.0
            else:  # Bear cycle
                if not np.isnan(cycle_trough):
                    cycle_trough = min(cycle_trough, df['Low'].iloc[i])
                else:
                    cycle_trough = df['Low'].iloc[i]
                    
                if not np.isnan(ma_crossover_price) and ma_crossover_price > 0:
                    dist = (ma_crossover_price - cycle_trough) / ma_crossover_price * 100
                    current_max_dist = max(current_max_dist, dist)
                    percent_distance = (ma_crossover_price - df['Low'].iloc[i]) / ma_crossover_price * 100
                else:
                    percent_distance = 0.0
            
            # Store values
            df.iloc[i, df.columns.get_loc('cycle_type')] = 'bull' if current_cycle > 0 else 'bear'
            df.iloc[i, df.columns.get_loc('cycle_candles')] = cycle_candles
            df.iloc[i, df.columns.get_loc('ma_crossover_price')] = ma_crossover_price
            df.iloc[i, df.columns.get_loc('cycle_peak')] = cycle_peak
            df.iloc[i, df.columns.get_loc('cycle_trough')] = cycle_trough
            df.iloc[i, df.columns.get_loc('current_max_dist')] = current_max_dist
            df.iloc[i, df.columns.get_loc('percent_distance')] = percent_distance
        
        return df
    
    def _detect_smoothed_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect cycles using Smoothed mode (requires confirmation)."""
        # For simplicity, implement smoothed as a filtered version of sharp
        # In practice, this would require more sophisticated confirmation logic
        df_sharp = self._detect_sharp_cycles(df.copy())
        
        # Apply smoothing by requiring N consecutive confirmations
        # This is a simplified version - full implementation would be more complex
        smoothed_df = df_sharp.copy()
        
        # Apply basic smoothing filter
        for i in range(self.smoothed_candles, len(smoothed_df)):
            # Check if we have N consecutive bars confirming the cycle
            recent_types = smoothed_df['cycle_type'].iloc[i-self.smoothed_candles:i+1]
            if len(set(recent_types)) == 1:  # All same type
                continue  # Keep the cycle
            else:
                # Maintain previous cycle until confirmed
                if i > 0:
                    smoothed_df.iloc[i, smoothed_df.columns.get_loc('cycle_type')] = smoothed_df.iloc[i-1]['cycle_type']
        
        return smoothed_df
    
    def _calculate_cycle_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive cycle statistics.
        
        Args:
            df: DataFrame with cycle information
            
        Returns:
            Dictionary with cycle metrics
        """
        try:
            # Separate bull and bear cycles
            bull_cycles = df[df['cycle_type'] == 'bull']
            bear_cycles = df[df['cycle_type'] == 'bear']
            
            # Find cycle boundaries to calculate individual cycle stats
            cycle_changes = df['cycle_type'].ne(df['cycle_type'].shift(1))
            cycle_boundaries = df[cycle_changes].index.tolist()
            
            if len(cycle_boundaries) == 0:
                cycle_boundaries = [df.index[0]]
            
            bull_cycle_lengths = []
            bear_cycle_lengths = []
            bull_max_dists = []
            bear_max_dists = []
            
            # Calculate individual cycle statistics
            for i in range(len(cycle_boundaries)):
                start_idx = cycle_boundaries[i]
                end_idx = cycle_boundaries[i + 1] if i + 1 < len(cycle_boundaries) else df.index[-1]
                
                cycle_segment = df.loc[start_idx:end_idx]
                if len(cycle_segment) == 0:
                    continue
                
                cycle_type = cycle_segment['cycle_type'].iloc[0]
                cycle_length = len(cycle_segment)
                max_dist = cycle_segment['current_max_dist'].max()
                
                if cycle_type == 'bull':
                    bull_cycle_lengths.append(cycle_length)
                    bull_max_dists.append(max_dist)
                else:
                    bear_cycle_lengths.append(cycle_length)
                    bear_max_dists.append(max_dist)
            
            # Calculate statistics
            metrics = {
                # Current cycle info
                'current_cycle_type': df['cycle_type'].iloc[-1],
                'current_candles': int(df['cycle_candles'].iloc[-1]),
                'current_max_dist': df['current_max_dist'].iloc[-1],
                
                # Bull cycle statistics
                'max_bull_candles': max(bull_cycle_lengths) if bull_cycle_lengths else 0,
                'avg_bull_candles': np.mean(bull_cycle_lengths) if bull_cycle_lengths else 0,
                'max_bull_dist': max(bull_max_dists) if bull_max_dists else 0,
                'avg_bull_dist': np.mean(bull_max_dists) if bull_max_dists else 0,
                'total_bull_cycles': len(bull_cycle_lengths),
                
                # Bear cycle statistics  
                'max_bear_candles': max(bear_cycle_lengths) if bear_cycle_lengths else 0,
                'avg_bear_candles': np.mean(bear_cycle_lengths) if bear_cycle_lengths else 0,
                'max_bear_dist': max(bear_max_dists) if bear_max_dists else 0,
                'avg_bear_dist': np.mean(bear_max_dists) if bear_max_dists else 0,
                'total_bear_cycles': len(bear_cycle_lengths),
                
                # Overall statistics
                'total_cycles': len(bull_cycle_lengths) + len(bear_cycle_lengths),
                'bull_cycle_percentage': len(bull_cycles) / len(df) * 100 if len(df) > 0 else 0,
                'bear_cycle_percentage': len(bear_cycles) / len(df) * 100 if len(df) > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating cycle metrics: {e}")
            return {}
    
    def _generate_chart(self, data: pd.DataFrame, analysis: Dict[str, Any], 
                       index: str, timeframe: str, latest_date: str) -> Optional[str]:
        """
        Generate chart with price/MA overlay and cycle histogram with metrics table.
        
        Args:
            data: Original price data
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
            metrics = analysis['cycle_metrics']
            
            # Prepare data for visualization
            df_chart = df[['Open', 'High', 'Low', 'Close', f'MA_{self.ma_period}']].copy()
            df_chart = df_chart.dropna()
            
            if len(df_chart) == 0:
                logger.warning(f"No valid data for charting {index}")
                return None
            
            # Create figure with subplots - increase table space
            fig = plt.figure(figsize=(16, 14))
            gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1.2, 0.8], hspace=0.4)
            
            # Upper panel: Price chart with MA
            ax1 = fig.add_subplot(gs[0])
            
            # Create date-based x-axis
            dates_numeric = range(len(df_chart))
            date_labels = df_chart.index

            # Plot candlesticks manually (simplified)
            for i, (date, row) in enumerate(df_chart.iterrows()):
                color = 'green' if row['Close'] > row['Open'] else 'red'
                ax1.plot([i, i], [row['Low'], row['High']], color='black', linewidth=0.5)
                ax1.plot([i, i], [row['Open'], row['Close']], color=color, linewidth=2)

            # Plot MA
            ax1.plot(dates_numeric, df_chart[f'MA_{self.ma_period}'],
                    color='blue', linewidth=2, label=f'MA {self.ma_period}')

            # Set x-axis with proper date labels
            step = max(1, len(date_labels) // 10)  # Show ~10 date labels
            ax1.set_xticks(dates_numeric[::step])
            ax1.set_xticklabels([d.strftime('%Y-%m-%d') for d in date_labels[::step]], rotation=45, ha='right')
            
            ax1.set_title(f'{index} - Moving Average Cycles Analysis (MA {self.ma_period})', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Lower panel: Cycle histogram
            ax2 = fig.add_subplot(gs[1])
            
            # Prepare histogram data
            cycle_candles = []
            max_dists = []
            colors = []
            
            for _, row in df.iterrows():
                candles = row['cycle_candles'] if row['cycle_type'] == 'bull' else -row['cycle_candles']
                max_dist = row['current_max_dist'] if row['cycle_type'] == 'bull' else -row['current_max_dist']
                
                cycle_candles.append(candles)
                max_dists.append(max_dist)
                colors.append('green' if row['cycle_type'] == 'bull' else 'red')
            
            # Plot cycle candles as bars with date alignment
            bars = ax2.bar(dates_numeric, cycle_candles, color=colors, alpha=0.7, width=1.0)

            # Plot max distance as line
            ax2_twin = ax2.twinx()
            ax2_twin.plot(dates_numeric, max_dists, color='white', linewidth=1, alpha=0.8)

            # Set x-axis labels for histogram panel
            ax2.set_xticks(dates_numeric[::step])
            ax2.set_xticklabels([d.strftime('%Y-%m-%d') for d in date_labels[::step]], rotation=45, ha='right')

            ax2.set_ylabel('Cycle Candles', fontsize=12)
            ax2_twin.set_ylabel('Max % Distance', fontsize=12)
            ax2.set_xlabel('Trading Days', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add current value reference lines
            current_candles = metrics['current_candles']
            current_max_dist = metrics['current_max_dist']
            
            if metrics['current_cycle_type'] == 'bull':
                ax2.axhline(y=current_candles, color='green', linestyle='--', alpha=0.7)
                ax2_twin.axhline(y=current_max_dist, color='white', linestyle='--', alpha=0.7)
            else:
                ax2.axhline(y=-current_candles, color='red', linestyle='--', alpha=0.7)
                ax2_twin.axhline(y=-current_max_dist, color='white', linestyle='--', alpha=0.7)
            
            # Add metrics table overlay
            self._create_metrics_table(fig, analysis, gs[2])
            
            # Create output filename
            user_choice = str(self.user_config.ticker_choice) if self.user_config else '0-5'
            chart_filename = f"ma_cycles_{index}_{user_choice}_{timeframe}_{latest_date}.png"
            
            # Ensure output directory exists
            output_dir = self.config.directories['RESULTS_DIR'] / 'market_pulse'
            output_dir.mkdir(parents=True, exist_ok=True)
            chart_path = output_dir / chart_filename
            
            # Save chart
            plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            logger.info(f"MA Cycles chart generated: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error generating chart for {index}: {e}")
            return None
    
    def _create_metrics_table(self, fig, analysis: Dict[str, Any], gs_pos) -> None:
        """
        Create metrics table overlay similar to TradingView Pine script.
        
        Args:
            fig: Matplotlib figure
            analysis: Analysis results with metrics
            gs_pos: GridSpec position for table
        """
        try:
            metrics = analysis['cycle_metrics']
            
            # Create table subplot
            ax_table = fig.add_subplot(gs_pos)
            ax_table.axis('off')
            
            # Prepare table data in column format for better space utilization
            current_cycle = metrics['current_cycle_type']
            current_color = 'green' if current_cycle == 'bull' else 'red'

            # Organize data in columns: Current, Bull Stats, Bear Stats
            headers = ['Metric', 'Current', 'Bull Max', 'Bull Avg', 'Bear Max', 'Bear Avg']

            table_data = [
                headers,
                ['Candles', f"{metrics['current_candles']}",
                 f"{metrics['max_bull_candles']}", f"{metrics['avg_bull_candles']:.1f}",
                 f"{metrics['max_bear_candles']}", f"{metrics['avg_bear_candles']:.1f}"],
                ['Max % Dist', f"{metrics['current_max_dist']:.1f}%",
                 f"{metrics['max_bull_dist']:.1f}%", f"{metrics['avg_bull_dist']:.1f}%",
                 f"{metrics['max_bear_dist']:.1f}%", f"{metrics['avg_bear_dist']:.1f}%"]
            ]

            # Create table with column-based layout
            table = ax_table.table(
                cellText=table_data,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )

            # Style table with improved readability
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)  # Increase row height significantly

            # Style header row
            for j in range(len(headers)):
                table[(0, j)].set_facecolor('darkgray')
                table[(0, j)].set_text_props(weight='bold', color='white')
                table[(0, j)].set_edgecolor('white')
                table[(0, j)].set_linewidth(2)

            # Style data rows
            for i in range(1, len(table_data)):
                for j in range(len(headers)):
                    if j == 0:  # Metric names
                        table[(i, j)].set_facecolor('lightgray')
                        table[(i, j)].set_text_props(weight='bold', color='black')
                    elif j == 1:  # Current values
                        table[(i, j)].set_facecolor('black')
                        table[(i, j)].set_text_props(weight='bold', color=current_color)
                    elif j in [2, 3]:  # Bull stats
                        color = 'lightgreen' if j == 3 else 'green'
                        table[(i, j)].set_facecolor('black')
                        table[(i, j)].set_text_props(weight='bold', color=color)
                    else:  # Bear stats
                        color = 'lightcoral' if j == 5 else 'red'
                        table[(i, j)].set_facecolor('black')
                        table[(i, j)].set_text_props(weight='bold', color=color)

                    table[(i, j)].set_edgecolor('gray')
                    table[(i, j)].set_linewidth(1)
            
        except Exception as e:
            logger.error(f"Error creating metrics table: {e}")
    
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
                metrics = results['cycle_metrics']
                
                row = {
                    'date': latest_date,
                    'timeframe': timeframe,
                    'index': index,
                    'ma_period': self.ma_period,
                    'cycle_mode': self.cycle_mode,
                    'current_cycle_type': metrics['current_cycle_type'],
                    'current_candles': metrics['current_candles'],
                    'current_max_dist_pct': metrics['current_max_dist'],
                    'max_bull_candles': metrics['max_bull_candles'],
                    'max_bull_dist_pct': metrics['max_bull_dist'],
                    'avg_bull_candles': metrics['avg_bull_candles'],
                    'avg_bull_dist_pct': metrics['avg_bull_dist'],
                    'max_bear_candles': metrics['max_bear_candles'],
                    'max_bear_dist_pct': metrics['max_bear_dist'],
                    'avg_bear_candles': metrics['avg_bear_candles'],
                    'avg_bear_dist_pct': metrics['avg_bear_dist'],
                    'total_bull_cycles': metrics['total_bull_cycles'],
                    'total_bear_cycles': metrics['total_bear_cycles'],
                    'current_price': results['current_price'],
                    'current_ma': results['current_ma'],
                    'chart_timeframe': self.chart_timeframe
                }
                
                output_rows.append(row)
            
            # Create DataFrame
            df_output = pd.DataFrame(output_rows)
            
            # Create output filename
            indexes_str = '+'.join([idx.lower() for idx in sorted(analysis_results.keys())])
            user_choice = str(self.user_config.ticker_choice) if self.user_config else '0-5'
            output_filename = f"ma_cycles_{indexes_str}_{user_choice}_{timeframe}_{latest_date}.csv"
            
            # Ensure output directory exists
            output_dir = self.config.directories['RESULTS_DIR'] / 'market_pulse'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename
            
            # Save to CSV
            df_output.to_csv(output_path, index=False)
            
            logger.info(f"MA Cycles results saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving MA Cycles results: {e}")
            return None
    
    def _generate_summary(self, analysis_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary of MA Cycles analysis."""
        try:
            total_indexes = len(analysis_results)
            bull_cycle_indexes = sum(1 for r in analysis_results.values() 
                                   if r['cycle_metrics']['current_cycle_type'] == 'bull')
            bear_cycle_indexes = total_indexes - bull_cycle_indexes
            
            # Calculate average statistics
            avg_bull_candles = np.mean([
                r['cycle_metrics']['avg_bull_candles'] 
                for r in analysis_results.values()
            ])
            
            avg_bear_candles = np.mean([
                r['cycle_metrics']['avg_bear_candles'] 
                for r in analysis_results.values()
            ])
            
            avg_bull_dist = np.mean([
                r['cycle_metrics']['avg_bull_dist'] 
                for r in analysis_results.values()
            ])
            
            avg_bear_dist = np.mean([
                r['cycle_metrics']['avg_bear_dist'] 
                for r in analysis_results.values()
            ])
            
            return {
                'total_indexes_analyzed': total_indexes,
                'bull_cycle_indexes': bull_cycle_indexes,
                'bear_cycle_indexes': bear_cycle_indexes,
                'bull_cycle_percentage': round(bull_cycle_indexes / total_indexes * 100, 1),
                'average_bull_cycle_length': round(avg_bull_candles, 1),
                'average_bear_cycle_length': round(avg_bear_candles, 1),
                'average_bull_distance': round(avg_bull_dist, 1),
                'average_bear_distance': round(avg_bear_dist, 1),
                'ma_period': self.ma_period,
                'cycle_detection_mode': self.cycle_mode,
                'analysis_timeframe_days': self.chart_timeframe
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {'error': str(e)}