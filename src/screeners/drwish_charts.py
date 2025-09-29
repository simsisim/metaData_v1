"""
Dr. Wish Suite Chart Generator
==============================

Generates charts for Dr. Wish trading signals including:
- GLB (Green Line Breakout) with pivot levels
- Blue Dot oversold bounce signals  
- Black Dot oversold bounce signals

Optimized for batch processing with matplotlib backend.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DrWishChartGenerator:
    """
    Generate Dr. Wish suite charts with all visual elements
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_charts = config.get('enable_charts', False)
        self.chart_output_dir = config.get('chart_output_dir', 'results/screeners/drwish/charts')
        
        # Chart appearance settings
        self.glb_color = '#00ff00'  # Green
        self.blue_dot_color = '#0000ff'  # Blue
        self.black_dot_color = '#000000'  # Black
        self.breakout_color = '#ff0000'  # Red
        
        logger.info(f"DrWish chart generator initialized (enabled: {self.enable_charts})")

    def create_drwish_chart(self, ticker: str, df: pd.DataFrame,
                           glb_signals: pd.DataFrame = None,
                           blue_signals: pd.DataFrame = None,
                           black_signals: pd.DataFrame = None,
                           timeframe: str = 'daily',
                           parameter_set: str = None) -> Optional[str]:
        """
        Create comprehensive Dr. Wish chart for a ticker
        
        Args:
            ticker: Stock ticker symbol
            df: OHLCV price data
            glb_signals: GLB breakout signals DataFrame
            blue_signals: Blue Dot signals DataFrame
            black_signals: Black Dot signals DataFrame
            
        Returns:
            str: Path to saved chart file, or None if not enabled
        """
        if not self.enable_charts:
            return None
            
        if df is None or df.empty:
            logger.warning(f"No data available for {ticker} chart")
            return None
        
        try:
            # Prepare data
            df = df.copy()
            df = df.tail(252)  # Last trading year for chart clarity
            
            # Force timezone-naive index for all date operations
            try:
                # Force conversion to timezone-naive dates
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            except:
                try:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                except:
                    # Final fallback: string conversion
                    df.index = pd.to_datetime(df.index.astype(str))
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price data
            self._plot_price_data(ax1, df, ticker)
            
            # Plot GLB signals
            if glb_signals is not None and not glb_signals.empty:
                self._plot_glb_signals(ax1, df, glb_signals)
            
            # Plot Blue/Black Dot signals
            if blue_signals is not None and not blue_signals.empty:
                self._plot_blue_dot_signals(ax1, df, blue_signals)
                
            if black_signals is not None and not black_signals.empty:
                self._plot_black_dot_signals(ax1, df, black_signals)
            
            # Plot volume
            self._plot_volume(ax2, df)
            
            # Add signal summary
            self._add_signal_summary(ax1, glb_signals, blue_signals, black_signals)
            
            # Format chart
            self._format_chart(ax1, ax2, ticker, timeframe)
            
            # Save chart
            chart_path = self._save_chart(fig, ticker, timeframe, parameter_set)
            plt.close(fig)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating chart for {ticker}: {e}")
            return None

    def _plot_price_data(self, ax: plt.Axes, df: pd.DataFrame, ticker: str):
        """Plot candlestick price data"""
        dates = df.index
        opens = df['Open']
        highs = df['High'] 
        lows = df['Low']
        closes = df['Close']
        
        # Plot high-low wicks
        ax.vlines(dates, lows, highs, colors='black', linewidth=0.5)
        
        # Plot candlestick bodies
        up_days = closes >= opens
        down_days = ~up_days
        
        # Green candles (up days)
        ax.vlines(dates[up_days], opens[up_days], closes[up_days], 
                 colors='green', linewidth=3)
        
        # Red candles (down days)  
        ax.vlines(dates[down_days], opens[down_days], closes[down_days],
                 colors='red', linewidth=3)
        
        # Plot close line for trend visibility
        ax.plot(dates, closes, color='gray', linewidth=1, alpha=0.7, label='Close')

    def _plot_glb_signals(self, ax: plt.Axes, df: pd.DataFrame, glb_signals: pd.DataFrame):
        """Plot historical GLB levels from detection to breakout (read from individual files)"""
        if glb_signals.empty:
            return
            
        # Extract ticker from first signal to read individual GLB file
        first_signal = glb_signals.iloc[0]
        ticker = first_signal.get('ticker', 'UNKNOWN')
        
        # Read individual GLB results file directly
        try:
            glb_file_path = Path(f"results/screeners/drwish/individual/{ticker}_glb_results.csv")
            if glb_file_path.exists():
                glb_df = pd.read_csv(glb_file_path)
                logger.info(f"Found {len(glb_df)} GLB levels for {ticker}")
                
                # Get chart date range for filtering
                chart_start_date = df.index[0]
                chart_end_date = df.index[-1]
                
                for idx, row in glb_df.iterrows():
                    glb_level = float(row['level'])
                    detection_date = pd.to_datetime(row['detection_date'])
                    breakout_date = pd.to_datetime(row['breakout_date']) if pd.notna(row['breakout_date']) else None
                    is_confirmed = bool(row['is_confirmed'])
                    is_broken = bool(row['is_broken'])
                    
                    # Ensure timezone-naive dates
                    try:
                        if hasattr(detection_date, 'tz_convert') and detection_date.tz is not None:
                            detection_date = detection_date.tz_convert(None)
                    except:
                        pass
                    
                    # Skip GLB levels that are outside the chart date range
                    if detection_date < chart_start_date:
                        logger.info(f"Skipping GLB level {glb_level} - detection date {detection_date} before chart range")
                        continue
                    
                    # Determine line end point and appearance
                    if is_broken and breakout_date:
                        try:
                            if hasattr(breakout_date, 'tz_convert') and breakout_date.tz is not None:
                                breakout_date = breakout_date.tz_convert(None)
                        except:
                            pass
                        end_date = min(breakout_date, chart_end_date)
                        line_color = 'gray'  # Broken GLB
                        line_style = '--'
                        line_width = 2
                        alpha = 0.6
                    else:
                        end_date = chart_end_date
                        line_color = self.glb_color  # Active GLB
                        line_style = '-'
                        line_width = 3
                        alpha = 0.9
                    
                    # Plot GLB horizontal line from detection to end
                    try:
                        logger.info(f"Plotting GLB level {glb_level} from {detection_date} to {end_date}")
                        ax.plot([detection_date, end_date], [glb_level, glb_level],
                               color=line_color, linestyle=line_style, linewidth=line_width,
                               label=f'GLB {glb_level:.1f}' if idx == 0 else '', alpha=alpha, zorder=10)
                        
                        # Add GLB level text label at the start of the line
                        if self.config.get('show_breakout_labels', True):
                            ax.text(detection_date, glb_level, f'  GLB {glb_level:.1f}',
                                   verticalalignment='center', horizontalalignment='left',
                                   fontsize=9, color=line_color, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                        
                        # Mark detection point
                        ax.scatter(detection_date, glb_level, color=line_color, 
                                  s=80, marker='s', edgecolors='black', linewidth=1,
                                  label='GLB Detection' if idx == 0 else '', zorder=11)
                        
                        # Mark breakout point if broken
                        if is_broken and breakout_date and breakout_date <= chart_end_date:
                            ax.scatter(breakout_date, glb_level, color=self.breakout_color, 
                                      s=120, marker='^', edgecolors='black', linewidth=1,
                                      label='GLB Breakout' if idx == 0 else '', zorder=12)
                                      
                    except Exception as e:
                        logger.error(f"Error plotting GLB level {glb_level}: {e}")
                        
            else:
                logger.warning(f"GLB file not found: {glb_file_path}")
                        
        except Exception as e:
            logger.error(f"Could not read individual GLB file for {ticker}: {e}")
        
        # Mark current GLB breakout signals from screener results
        for idx, signal in glb_signals.iterrows():
            try:
                signal_date = pd.to_datetime(signal.get('signal_date', signal.get('date')))
                try:
                    if hasattr(signal_date, 'tz_convert') and signal_date.tz is not None:
                        signal_date = signal_date.tz_convert(None)
                except:
                    pass
                    
                breakout_high = signal.get('breakout_high', signal.get('price', 0))
                
                # Mark current breakout point
                ax.scatter(signal_date, breakout_high, color=self.breakout_color, 
                          s=140, marker='*', edgecolors='black', linewidth=2,
                          label='Recent Breakout' if idx == 0 else '', zorder=13)
                          
            except Exception as e:
                logger.warning(f"Error plotting current breakout: {e}")

    def _plot_blue_dot_signals(self, ax: plt.Axes, df: pd.DataFrame, blue_signals: pd.DataFrame):
        """Plot Blue Dot oversold bounce signals"""
        for _, signal in blue_signals.iterrows():
            signal_date = pd.to_datetime(signal['signal_date'])
            if hasattr(signal_date, 'tz_localize'):
                signal_date = signal_date.tz_localize(None)
            
            # Find closest price data
            try:
                signal_idx = df.index.get_indexer([signal_date], method='nearest')[0]
                signal_low = df['Low'].iloc[signal_idx]
                
                # Plot blue dot below the bar
                dot_y = signal_low * 0.995
                ax.scatter(signal_date, dot_y, color=self.blue_dot_color,
                          s=80, marker='o', label='Blue Dot', zorder=5)
                
                # Add label
                ax.text(signal_date, dot_y * 0.992, 'BD',
                       horizontalalignment='center', verticalalignment='top',
                       fontsize=8, color='white', weight='bold')
                       
            except Exception as e:
                logger.warning(f"Error plotting blue dot for {signal_date}: {e}")

    def _plot_black_dot_signals(self, ax: plt.Axes, df: pd.DataFrame, black_signals: pd.DataFrame):
        """Plot Black Dot oversold bounce signals"""
        for _, signal in black_signals.iterrows():
            signal_date = pd.to_datetime(signal['signal_date'])
            if hasattr(signal_date, 'tz_localize'):
                signal_date = signal_date.tz_localize(None)
            
            # Find closest price data
            try:
                signal_idx = df.index.get_indexer([signal_date], method='nearest')[0]
                signal_low = df['Low'].iloc[signal_idx]
                
                # Plot black dot below the bar
                dot_y = signal_low * 0.993
                ax.scatter(signal_date, dot_y, color=self.black_dot_color,
                          s=100, marker='o', edgecolors='gray', linewidth=1,
                          label='Black Dot', zorder=5)
                
                # Add label
                ax.text(signal_date, dot_y * 0.990, 'BD',
                       horizontalalignment='center', verticalalignment='top',
                       fontsize=8, color='white', weight='bold')
                       
            except Exception as e:
                logger.warning(f"Error plotting black dot for {signal_date}: {e}")

    def _plot_volume(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot volume bars"""
        if 'Volume' not in df.columns:
            ax.text(0.5, 0.5, 'Volume data not available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
            
        dates = df.index
        volumes = df['Volume']
        closes = df['Close']
        
        # Color bars based on price movement
        colors = ['green' if closes.iloc[i] >= closes.iloc[i-1] else 'red' 
                 for i in range(1, len(closes))]
        colors.insert(0, 'gray')  # First bar
        
        ax.bar(dates, volumes, color=colors, alpha=0.6, width=0.8)
        ax.set_ylabel('Volume')
        ax.set_title('Volume')

    def _add_signal_summary(self, ax: plt.Axes, glb_signals, blue_signals, black_signals):
        """Add signal summary text box"""
        glb_count = len(glb_signals) if glb_signals is not None else 0
        blue_count = len(blue_signals) if blue_signals is not None else 0  
        black_count = len(black_signals) if black_signals is not None else 0
        
        summary_text = [
            "Dr. Wish Signals:",
            f"GLB Breakouts: {glb_count}",
            f"Blue Dots: {blue_count}",
            f"Black Dots: {black_count}",
            f"Total Signals: {glb_count + blue_count + black_count}"
        ]
        
        ax.text(0.02, 0.98, '\n'.join(summary_text),
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    def _format_chart(self, ax1: plt.Axes, ax2: plt.Axes, ticker: str, timeframe: str = 'daily'):
        """Format chart appearance"""
        # Main chart formatting
        ax1.set_title(f'{ticker} - Dr. Wish Analysis ({timeframe.title()}) - {datetime.now().strftime("%Y-%m-%d")}',
                     fontsize=16, weight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Create custom legend to avoid GLB label overwriting
        handles, labels = ax1.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        seen_labels = set()
        
        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
                seen_labels.add(label)
        
        ax1.legend(unique_handles, unique_labels, loc='upper left', framealpha=0.9)
        
        # Volume chart formatting  
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()

    def _save_chart(self, fig: plt.Figure, ticker: str, timeframe: str = 'daily', parameter_set: str = None) -> str:
        """Save chart to file with parameter set support"""
        # Create output directory with parameter set subdirectory if specified
        output_dir = Path(self.chart_output_dir)
        if parameter_set:
            output_dir = output_dir / parameter_set
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timeframe and parameter set
        timestamp = datetime.now().strftime('%Y%m%d')
        if parameter_set:
            filename = f"{ticker}_drwish_{parameter_set}_{timeframe}_{timestamp}.png"
        else:
            filename = f"{ticker}_drwish_{timeframe}_{timestamp}.png"
        chart_path = output_dir / filename
        
        # Save with high quality
        fig.savefig(chart_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        logger.info(f"DrWish chart saved: {chart_path}")
        return str(chart_path)


def generate_drwish_charts(screener_results: List[Dict], batch_data: Dict, 
                          config: Dict[str, Any]) -> List[str]:
    """
    Generate charts for top Dr. Wish signals
    
    Args:
        screener_results: List of screening results
        batch_data: Dictionary of ticker data
        config: Configuration parameters
        
    Returns:
        List of generated chart file paths
    """
    if not config.get('enable_charts', False):
        return []
    
    chart_generator = DrWishChartGenerator(config)
    chart_paths = []
    
    # Group results by ticker
    ticker_results = {}
    for result in screener_results:
        ticker = result['ticker']
        if ticker not in ticker_results:
            ticker_results[ticker] = {'glb': [], 'blue_dot': [], 'black_dot': []}
        
        screen_type = result['screen_type']
        if 'glb' in screen_type:
            ticker_results[ticker]['glb'].append(result)
        elif 'blue_dot' in screen_type:
            ticker_results[ticker]['blue_dot'].append(result)
        elif 'black_dot' in screen_type:
            ticker_results[ticker]['black_dot'].append(result)
    
    # Generate charts for all tickers with signals (up to 50 to avoid too many files)
    top_tickers = list(ticker_results.keys())[:50]  # Top 50 tickers
    
    for ticker in top_tickers:
        if ticker in batch_data:
            try:
                # Convert signal results to DataFrames
                glb_df = pd.DataFrame(ticker_results[ticker]['glb']) if ticker_results[ticker]['glb'] else None
                blue_df = pd.DataFrame(ticker_results[ticker]['blue_dot']) if ticker_results[ticker]['blue_dot'] else None
                black_df = pd.DataFrame(ticker_results[ticker]['black_dot']) if ticker_results[ticker]['black_dot'] else None
                
                # Generate chart
                timeframe = config.get('timeframe', 'daily')
                parameter_set = config.get('parameter_set_name', None)
                chart_path = chart_generator.create_drwish_chart(
                    ticker, batch_data[ticker], glb_df, blue_df, black_df, timeframe, parameter_set
                )
                
                if chart_path:
                    chart_paths.append(chart_path)
                    
            except Exception as e:
                logger.warning(f"Error generating chart for {ticker}: {e}")
                continue
    
    logger.info(f"Generated {len(chart_paths)} DrWish charts")
    return chart_paths


# Module interface
__all__ = ['DrWishChartGenerator', 'generate_drwish_charts']