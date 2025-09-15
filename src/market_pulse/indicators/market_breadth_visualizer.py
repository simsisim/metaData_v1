"""
Market Breadth Visualizer
========================

Creates TradingView-style stacked chart visualizations for market breadth analysis.
Generates PNG charts with same filename as the breadth analysis CSV files.

Features:
- 3-layer stacked visualization with shared date axis
- Top: Candlestick chart with index price data
- Middle: Line/area chart for above-MA breadth percentages
- Bottom: Histogram for new highs/lows analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = logging.getLogger(__name__)

# Module-level caching for universe mapping
_universe_mapping_cache = None


def _load_universe_mapping() -> Dict[str, str]:
    """
    Load universe-to-ticker mapping from CSV file with caching.
    
    Returns:
        Dict mapping universe names to index symbols (e.g., {'SP500': 'SPY'})
    """
    global _universe_mapping_cache
    
    if _universe_mapping_cache is not None:
        return _universe_mapping_cache
    
    try:
        # Try to find mapping_tickers.csv in current directory
        mapping_path = Path('mapping_tickers.csv')
        if not mapping_path.exists():
            # Try relative to this module's directory
            module_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
            mapping_path = module_dir / 'mapping_tickers.csv'
        
        if mapping_path.exists():
            mapping_df = pd.read_csv(mapping_path)
            # Strip whitespace from column names and values
            mapping_df.columns = mapping_df.columns.str.strip()
            mapping_df['universe'] = mapping_df['universe'].str.strip()
            mapping_df['index_symbol'] = mapping_df['index_symbol'].str.strip()
            # Convert to dict: {'SP500': 'SPY', 'NASDAQ100': 'QQQ', ...}
            _universe_mapping_cache = dict(zip(mapping_df['universe'], mapping_df['index_symbol']))
            logger.debug(f"Loaded universe mapping from {mapping_path}: {_universe_mapping_cache}")
        else:
            logger.warning(f"mapping_tickers.csv not found at {mapping_path}")
            _universe_mapping_cache = {}
            
    except Exception as e:
        logger.error(f"Error loading universe mapping: {e}")
        _universe_mapping_cache = {}
    
    return _universe_mapping_cache


def _get_index_symbol_from_mapping(universe_name: str) -> str:
    """
    Get index symbol for universe with fallback chain.
    
    Args:
        universe_name: Universe name (e.g., 'SP500', 'NASDAQ100')
        
    Returns:
        Index symbol (e.g., 'SPY', 'QQQ') with fallback to 'SPY'
    """
    mapping = _load_universe_mapping()
    
    # Try exact match first
    if universe_name in mapping:
        return mapping[universe_name]
    
    # Try uppercase
    universe_upper = universe_name.upper()
    if universe_upper in mapping:
        return mapping[universe_upper]
    
    # Try common variations
    universe_variations = {
        'SP500': 'SPY',
        'S&P500': 'SPY', 
        'S&P_500': 'SPY',
        'NASDAQ100': 'QQQ',
        'NASDAQ_100': 'QQQ',
        'RUSSELL1000': 'IWM',
        'RUSSELL_1000': 'IWM',
        'ALL': 'SPY',
        'COMBINED': 'SPY'
    }
    
    if universe_upper in universe_variations:
        return universe_variations[universe_upper]
    
    # Default fallback
    logger.debug(f"No mapping found for universe '{universe_name}', defaulting to SPY")
    return 'SPY'


class MarketBreadthVisualizer:
    """
    Creates comprehensive market breadth visualizations with TradingView-style layout.
    """
    
    def __init__(self, config=None):
        """
        Initialize the market breadth visualizer.
        
        Args:
            config: Configuration object with directory paths
        """
        self.config = config
        
        # Chart styling parameters
        self.style_config = {
            'figure_size': (16, 12),  # Wide format like TradingView
            'dpi': 150,
            'background_color': '#ffffff',  # White theme
            'grid_color': '#cccccc',
            'text_color': '#000000',
            'bullish_color': '#26a69a',  # Green for bullish
            'bearish_color': '#ef5350',   # Red for bearish
            'volume_color': '#64b5f6',   # Blue for volume
            'ma_colors': {
                'ma20': '#ffa726',   # Orange
                'ma50': '#42a5f5',   # Blue  
                'ma200': '#ab47bc'   # Purple
            },
            'breadth_colors': {
                'above_ma20': '#4caf50',    # Green
                'above_ma50': '#2196f3',    # Blue
                'above_ma200': '#9c27b0',   # Purple
                'breadth_score': '#ff9800'  # Orange
            },
            'highs_lows_colors': {
                'new_highs': '#4caf50',     # Green
                'new_lows': '#f44336',      # Red
                'net_highs': '#ff9800'      # Orange
            }
        }
    
    def create_breadth_chart(self, breadth_data: pd.DataFrame, index_data: pd.DataFrame = None, 
                           output_path: str = None, universe_name: str = 'Market') -> str:
        """
        Create comprehensive market breadth chart with 3 stacked subplots.
        
        Args:
            breadth_data: DataFrame with historical market breadth data
            index_data: Optional DataFrame with index price data (OHLCV)
            output_path: Path to save the PNG chart
            universe_name: Universe name for chart title
            
        Returns:
            Path to saved chart file
        """
        try:
            # Prepare data
            chart_data = self._prepare_chart_data(breadth_data, index_data)
            
            if chart_data is None or chart_data.empty:
                logger.warning("No data available for chart generation")
                return None
            
            # Create the figure with subplots
            fig = self._create_figure_layout()
            
            # Generate each subplot
            self._create_candlestick_subplot(fig, chart_data, universe_name)
            self._create_breadth_subplot(fig, chart_data, universe_name)
            self._create_highs_lows_subplot(fig, chart_data, universe_name)
            
            # Apply final styling and save
            self._finalize_chart(fig, universe_name)
            
            # Save the chart
            if output_path:
                chart_path = self._save_chart(fig, output_path)
                plt.close(fig)  # Free memory
                return chart_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating breadth chart: {e}")
            return None
    
    def _prepare_chart_data(self, breadth_data: pd.DataFrame, index_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare and combine data for charting.
        
        Args:
            breadth_data: Historical market breadth data
            index_data: Optional index price data
            
        Returns:
            Combined DataFrame ready for charting
        """
        try:
            # Start with breadth data as base
            chart_data = breadth_data.copy()
            
            # Ensure date column is datetime and normalize to YYYY-MM-DD only
            if 'date' in chart_data.columns:
                chart_data['date'] = pd.to_datetime(chart_data['date'], errors='coerce').dt.date
                chart_data.set_index('date', inplace=True)
            
            # Add index price data if available
            if index_data is not None and not index_data.empty:
                # Ensure index_data has datetime index and normalize to YYYY-MM-DD only
                if 'date' in index_data.columns:
                    index_data['date'] = pd.to_datetime(index_data['date'], errors='coerce').dt.date
                    index_data.set_index('date', inplace=True)
                elif 'Date' in index_data.columns:
                    index_data['Date'] = pd.to_datetime(index_data['Date'], utc=True, errors='coerce').dt.date
                    index_data.set_index('Date', inplace=True)
                
                # Normalize column names to lowercase for OHLCV data
                column_mapping = {
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                }
                for old_col, new_col in column_mapping.items():
                    if old_col in index_data.columns:
                        index_data[new_col] = index_data[old_col]
                
                # Merge with breadth data - use lowercase column names
                available_columns = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in index_data.columns]
                if available_columns:
                    chart_data = chart_data.join(index_data[available_columns], how='left')
            
            # Fill missing price data with synthetic data if needed
            if 'close' not in chart_data.columns or chart_data['close'].isna().all():
                logger.info("No index price data available, creating synthetic price based on breadth score")
                chart_data = self._create_synthetic_price_data(chart_data)
            
            # Sort by date
            chart_data = chart_data.sort_index()
            
            logger.info(f"Prepared chart data: {len(chart_data)} rows, {len(chart_data.columns)} columns")
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            return None
    
    def _create_synthetic_price_data(self, breadth_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic price data based on breadth score when real price data is unavailable.
        
        Args:
            breadth_data: DataFrame with breadth metrics
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        try:
            # Use overall_breadth_score as base for synthetic price
            if 'overall_breadth_score' in breadth_data.columns:
                base_price = 100 + (breadth_data['overall_breadth_score'] - 50)  # Center around 100
            else:
                # Fallback: use combined_pct_above_ma50
                if 'combined_pct_above_ma50' in breadth_data.columns:
                    base_price = 100 + (breadth_data['combined_pct_above_ma50'] - 50)
                else:
                    base_price = pd.Series([100] * len(breadth_data), index=breadth_data.index)
            
            # Create OHLC from base price with some variance
            noise = np.random.normal(0, 0.5, len(breadth_data))
            
            breadth_data['close'] = base_price + noise
            breadth_data['open'] = breadth_data['close'].shift(1).fillna(breadth_data['close'])
            breadth_data['high'] = breadth_data[['open', 'close']].max(axis=1) + abs(noise) * 0.5
            breadth_data['low'] = breadth_data[['open', 'close']].min(axis=1) - abs(noise) * 0.5
            breadth_data['volume'] = abs(breadth_data.get('net_advances', 0)) * 1000 + 100000
            
            logger.info("Created synthetic price data based on breadth metrics")
            
            return breadth_data
            
        except Exception as e:
            logger.error(f"Error creating synthetic price data: {e}")
            return breadth_data
    
    def _create_figure_layout(self) -> plt.Figure:
        """
        Create the main figure with 3 stacked subplots layout.
        
        Returns:
            Matplotlib figure object
        """
        # Set style to default (white background)
        plt.style.use('default')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.style_config['figure_size'],
                                           dpi=self.style_config['dpi'],
                                           gridspec_kw={'height_ratios': [3, 1.5, 1],  # 60%, 25%, 15%
                                                       'hspace': 0.1})
        
        # Store axes in figure for later access
        fig.ax1 = ax1  # Candlestick chart
        fig.ax2 = ax2  # Breadth chart
        fig.ax3 = ax3  # Highs/Lows chart
        
        # Set white background for figure and all subplots
        fig.patch.set_facecolor(self.style_config['background_color'])
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor(self.style_config['background_color'])
            ax.grid(True, color=self.style_config['grid_color'], alpha=0.3)
            ax.tick_params(colors=self.style_config['text_color'])
        
        return fig
    
    def _create_candlestick_subplot(self, fig: plt.Figure, chart_data: pd.DataFrame, universe_name: str):
        """
        Create the top candlestick chart subplot.
        
        Args:
            fig: Matplotlib figure
            chart_data: Prepared chart data
            universe_name: Universe name for labeling
        """
        ax = fig.ax1
        
        try:
            # Prepare OHLC data
            dates = chart_data.index
            opens = chart_data['open']
            highs = chart_data['high']
            lows = chart_data['low']
            closes = chart_data['close']
            volumes = chart_data.get('volume', pd.Series([0] * len(chart_data)))
            
            # Create candlestick chart
            for i, (date, o, h, l, c) in enumerate(zip(dates, opens, highs, lows, closes)):
                if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
                    continue
                
                # Determine color
                color = self.style_config['bullish_color'] if c >= o else self.style_config['bearish_color']
                
                # Draw high-low line
                ax.plot([date, date], [l, h], color=color, linewidth=0.8, alpha=0.8)
                
                # Draw body rectangle
                body_height = abs(c - o)
                body_bottom = min(o, c)
                
                if body_height > 0:
                    rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 0.6, body_height,
                                   facecolor=color, edgecolor=color, alpha=0.8)
                    ax.add_patch(rect)
            
            # Add volume bars (scaled and positioned at bottom)
            if not volumes.isna().all():
                volume_max = volumes.max()
                volume_scale = (highs.max() - lows.min()) * 0.1  # 10% of price range
                
                volume_bars = (volumes / volume_max) * volume_scale
                volume_bottom = lows.min() - volume_scale * 1.2
                
                ax.bar(dates, volume_bars, bottom=volume_bottom, width=0.8,
                      color=self.style_config['volume_color'], alpha=0.6, label='Volume')
            
            # Styling
            ax.set_title(f'{universe_name} Market Breadth Analysis - Price Chart', 
                        color=self.style_config['text_color'], fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, color=self.style_config['grid_color'])
            ax.set_ylabel('Price', color=self.style_config['text_color'])
            
            # Remove x-axis labels (will be shared with bottom subplot)
            ax.set_xticklabels([])
            
        except Exception as e:
            logger.error(f"Error creating candlestick subplot: {e}")
    
    def _create_breadth_subplot(self, fig: plt.Figure, chart_data: pd.DataFrame, universe_name: str):
        """
        Create the middle breadth analysis subplot.
        
        Args:
            fig: Matplotlib figure
            chart_data: Prepared chart data
            universe_name: Universe name for labeling
        """
        ax = fig.ax2
        
        try:
            dates = chart_data.index
            
            # Plot MA breadth percentages
            ma_columns = {
                'combined_pct_above_ma20': ('Above MA20', self.style_config['breadth_colors']['above_ma20']),
                'combined_pct_above_ma50': ('Above MA50', self.style_config['breadth_colors']['above_ma50']),
                'combined_pct_above_ma200': ('Above MA200', self.style_config['breadth_colors']['above_ma200'])
            }
            
            for col, (label, color) in ma_columns.items():
                if col in chart_data.columns:
                    ax.plot(dates, chart_data[col], color=color, linewidth=2, label=label, alpha=0.8)
                    
                    # Add area fill
                    ax.fill_between(dates, chart_data[col], alpha=0.2, color=color)
            
            # Plot overall breadth score
            if 'overall_breadth_score' in chart_data.columns:
                ax.plot(dates, chart_data['overall_breadth_score'], 
                       color=self.style_config['breadth_colors']['breadth_score'], 
                       linewidth=3, label='Breadth Score', alpha=0.9)
            
            # Add horizontal reference lines
            ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=50, color='yellow', linestyle='-', alpha=0.5, linewidth=1)
            ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Styling
            ax.set_title('Market Breadth Indicators (% Above Moving Averages)', 
                        color=self.style_config['text_color'], fontsize=12)
            ax.set_ylabel('Percentage (%)', color=self.style_config['text_color'])
            ax.grid(True, alpha=0.3, color=self.style_config['grid_color'])
            ax.legend(loc='upper left', framealpha=0.8)
            ax.set_ylim(0, 100)
            
            # Remove x-axis labels (shared with bottom subplot)
            ax.set_xticklabels([])
            
        except Exception as e:
            logger.error(f"Error creating breadth subplot: {e}")
    
    def _create_highs_lows_subplot(self, fig: plt.Figure, chart_data: pd.DataFrame, universe_name: str):
        """
        Create the bottom new highs/lows histogram subplot.
        
        Args:
            fig: Matplotlib figure
            chart_data: Prepared chart data
            universe_name: Universe name for labeling
        """
        ax = fig.ax3
        
        try:
            dates = chart_data.index
            
            # Plot new highs and lows as histograms
            if 'new_52week_highs' in chart_data.columns and 'new_52week_lows' in chart_data.columns:
                new_highs = chart_data['new_52week_highs']
                new_lows = chart_data['new_52week_lows']
                
                # Create histogram bars
                width = 0.8
                ax.bar(dates, new_highs, width=width, color=self.style_config['highs_lows_colors']['new_highs'], 
                      alpha=0.7, label='New 52W Highs')
                ax.bar(dates, -new_lows, width=width, color=self.style_config['highs_lows_colors']['new_lows'], 
                      alpha=0.7, label='New 52W Lows')
            
            # Plot net new highs as line
            if 'net_52week_highs' in chart_data.columns:
                ax.plot(dates, chart_data['net_52week_highs'], 
                       color=self.style_config['highs_lows_colors']['net_highs'], 
                       linewidth=2, label='Net New Highs', alpha=0.9)
            
            # Add zero line
            ax.axhline(y=0, color='white', linestyle='-', alpha=0.8, linewidth=1)
            
            # Styling
            ax.set_title('New Highs vs New Lows (52-Week)', 
                        color=self.style_config['text_color'], fontsize=12)
            ax.set_ylabel('Count', color=self.style_config['text_color'])
            ax.set_xlabel('Date', color=self.style_config['text_color'])
            ax.grid(True, alpha=0.3, color=self.style_config['grid_color'])
            ax.legend(loc='upper left', framealpha=0.8)
            
            # Format x-axis dates
            # Convert date objects back to datetime for proper formatting
            if hasattr(dates, 'to_list') and len(dates) > 0:
                import datetime as dt
                if isinstance(dates.to_list()[0], dt.date):
                    # Convert date objects to datetime for matplotlib
                    datetime_dates = pd.to_datetime(dates)
                    ax.set_xlim(datetime_dates.min(), datetime_dates.max())
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        except Exception as e:
            logger.error(f"Error creating highs/lows subplot: {e}")
    
    def _finalize_chart(self, fig: plt.Figure, universe_name: str):
        """
        Apply final styling and formatting to the chart.
        
        Args:
            fig: Matplotlib figure
            universe_name: Universe name for title
        """
        # Set overall title
        fig.suptitle(f'Market Breadth Analysis - {universe_name}', 
                    color=self.style_config['text_color'], fontsize=16, fontweight='bold')
        
        # Align all x-axes
        fig.ax1.sharex(fig.ax3)
        fig.ax2.sharex(fig.ax3)
        
        # Use subplots_adjust instead of tight_layout to avoid warnings
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12, hspace=0.15)
    
    def _save_chart(self, fig: plt.Figure, output_path: str) -> str:
        """
        Save the chart to PNG file.
        
        Args:
            fig: Matplotlib figure
            output_path: Path to save the chart
            
        Returns:
            Path to saved chart file
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high DPI
            fig.savefig(output_path, dpi=self.style_config['dpi'], 
                       bbox_inches='tight', facecolor=self.style_config['background_color'],
                       edgecolor='none', pad_inches=0.1)
            
            logger.info(f"Market breadth chart saved: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            return None


def generate_breadth_chart_from_csv(csv_path: str, config=None, index_symbol: str = None) -> str:
    """
    Convenience function to generate breadth chart directly from CSV file.
    
    Args:
        csv_path: Path to breadth analysis CSV file
        config: Configuration object
        index_symbol: Index symbol for price data (optional, auto-detected from universe if None)
        
    Returns:
        Path to generated PNG chart
    """
    try:
        # Load breadth data
        breadth_data = pd.read_csv(csv_path)
        
        # Create output path (same name as CSV but with .png extension)
        csv_path = Path(csv_path)
        png_path = csv_path.with_suffix('.png')
        
        # Extract universe name from filename
        filename_parts = csv_path.stem.split('_')
        if len(filename_parts) >= 3:
            universe_name = filename_parts[2]  # Extract universe from ba_historical_SP500_...
        else:
            universe_name = 'Market'
        
        # Create visualizer and generate chart
        visualizer = MarketBreadthVisualizer(config)
        
        # Determine index symbol from mapping or use provided one
        if index_symbol is None:
            index_symbol = _get_index_symbol_from_mapping(universe_name)
        
        # Try to load index price data (optional)
        index_data = None
        if config and index_symbol:
            try:
                daily_data_dir = config.directories.get('DAILY_DATA_DIR')
                if daily_data_dir:
                    index_file = Path(daily_data_dir) / f"{index_symbol}.csv"
                    if index_file.exists():
                        index_data = pd.read_csv(index_file)
                        logger.debug(f"Loaded index data for {universe_name} -> {index_symbol}")
                    else:
                        logger.debug(f"Index file not found: {index_file}")
            except Exception as e:
                logger.debug(f"Could not load index data for {index_symbol}: {e}")
        
        # Generate chart
        chart_path = visualizer.create_breadth_chart(
            breadth_data=breadth_data,
            index_data=index_data,
            output_path=png_path,
            universe_name=universe_name
        )
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Error generating breadth chart from CSV {csv_path}: {e}")
        return None