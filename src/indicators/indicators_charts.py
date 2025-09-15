"""
Technical Indicators Charting Module
===================================

Comprehensive charting and visualization functionality for technical indicators.
Creates multi-panel charts similar to TradingView with all major indicators.

Features:
- Multi-panel chart layout with price, volume, and multiple indicator panels
- Configurable indicator selection per ticker
- Professional styling and formatting
- Export to PNG format
- Support for all indicators in indicators_calculation module
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from .indicators_calculation import (
    calculate_kurutoga, calculate_tsi, calculate_macd, calculate_mfi,
    calculate_cog, calculate_momentum, calculate_rsi, calculate_ma_crosses,
    calculate_easy_trade, analyze_combined_signals, calculate_all_indicators
)

logger = logging.getLogger(__name__)


class IndicatorChartGenerator:
    """
    Generate comprehensive technical indicator charts.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory to save chart files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart styling similar to TradingView
        plt.style.use('dark_background')
        self.colors = {
            'bullish': '#00ff00',
            'bearish': '#ff0000', 
            'neutral': '#ffff00',
            'line': '#2196F3',
            'signal': '#FF9800',
            'volume': '#607D8B',
            'ma_fast': '#2196F3',
            'ma_slow': '#F44336',
            'background': '#1E1E1E'
        }
    
    def create_comprehensive_chart(self, data: pd.DataFrame, ticker: str, 
                                 indicators_config: Dict, timeframe: str = '1d') -> str:
        """
        Create comprehensive multi-panel chart with selected indicators.
        
        Args:
            data: DataFrame with OHLCV data
            ticker: Ticker symbol
            indicators_config: Dict specifying which indicators to include
            timeframe: Chart timeframe (default '1d')
            
        Returns:
            str: Path to saved chart file
        """
        # Calculate all requested indicators
        chart_data = self._calculate_chart_indicators(data, indicators_config)
        
        # Determine number of panels needed
        panel_count = self._count_panels(indicators_config)
        
        # Create figure with TradingView-style layout
        fig = plt.figure(figsize=(16, 20), facecolor=self.colors['background'])
        gs = fig.add_gridspec(panel_count, 1, height_ratios=self._get_height_ratios(panel_count, indicators_config))
        axes = [fig.add_subplot(gs[i]) for i in range(panel_count)]
        
        if panel_count == 1:
            axes = [axes]
        
        panel_idx = 0
        
        # Panel 1: Price chart with moving averages
        self._plot_price_panel(axes[panel_idx], chart_data, ticker, indicators_config)
        panel_idx += 1
        
        # Panel 2: Volume (if space allows)
        if panel_count > 2:
            self._plot_volume_panel(axes[panel_idx], chart_data)
            panel_idx += 1
        
        # Additional indicator panels
        if indicators_config.get('tsi', False):
            self._plot_tsi_panel(axes[panel_idx], chart_data)
            panel_idx += 1
            
        if indicators_config.get('macd', False):
            self._plot_macd_panel(axes[panel_idx], chart_data)
            panel_idx += 1
            
        if indicators_config.get('mfi', False):
            self._plot_mfi_panel(axes[panel_idx], chart_data)
            panel_idx += 1
            
        if indicators_config.get('cog', False):
            self._plot_cog_panel(axes[panel_idx], chart_data)
            panel_idx += 1
            
        if indicators_config.get('kurutoga', False):
            self._plot_kurutoga_panel(axes[panel_idx], chart_data)
            panel_idx += 1
            
        if indicators_config.get('momentum', False) or indicators_config.get('rsi', False):
            self._plot_momentum_rsi_panel(axes[panel_idx], chart_data, indicators_config)
            panel_idx += 1
        
        # Format and save
        output_file = self.output_dir / f"{ticker}_{timeframe}.png"
        self._format_and_save_chart(fig, axes, ticker, timeframe, output_file)
        
        return str(output_file)
    
    def _calculate_chart_indicators(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Calculate all indicators needed for the chart."""
        result = data.copy()
        
        if config.get('kurutoga', False):
            kurutoga_data = calculate_kurutoga(data)
            result = pd.concat([result, kurutoga_data], axis=1)
            
        if config.get('tsi', False):
            tsi_data = calculate_tsi(data)
            result = pd.concat([result, tsi_data], axis=1)
            
        if config.get('macd', False):
            macd_data = calculate_macd(data)
            result = pd.concat([result, macd_data], axis=1)
            
        if config.get('mfi', False):
            mfi_data = calculate_mfi(data)
            result = pd.concat([result, mfi_data], axis=1)
            
        if config.get('cog', False):
            cog_data = calculate_cog(data)
            result = pd.concat([result, cog_data], axis=1)
            
        if config.get('momentum', False):
            mom_data = calculate_momentum(data)
            result = pd.concat([result, mom_data], axis=1)
            
        if config.get('rsi', False):
            result['RSI'] = calculate_rsi(data)
            
        if config.get('ma_crosses', False):
            ma_data = calculate_ma_crosses(data)
            result = pd.concat([result, ma_data], axis=1)
            
        if config.get('easy_trade', False):
            easy_data = calculate_easy_trade(data)
            result = pd.concat([result, easy_data], axis=1)
        
        return result
    
    def _count_panels(self, config: Dict) -> int:
        """Count how many chart panels are needed."""
        panel_count = 1  # Price panel always included
        
        if any(config.get(indicator, False) for indicator in ['tsi', 'macd', 'mfi', 'cog', 'kurutoga']):
            panel_count += 1  # Volume panel
            
        panel_count += sum([
            config.get('tsi', False),
            config.get('macd', False), 
            config.get('mfi', False),
            config.get('cog', False),
            config.get('kurutoga', False),
            config.get('momentum', False) or config.get('rsi', False)
        ])
        
        return min(panel_count, 10)  # Maximum 10 panels
    
    def _get_height_ratios(self, panel_count: int, config: Dict) -> List[float]:
        """Get height ratios for chart panels - price panel gets more space."""
        ratios = [3]  # Price panel gets 3x height
        
        # Other panels get standard height
        for i in range(1, panel_count):
            ratios.append(1)
            
        return ratios
    
    def _plot_price_panel(self, ax, data: pd.DataFrame, ticker: str, config: Dict):
        """Plot price panel with candlesticks and moving averages."""
        # Plot candlestick chart using mplfinance style
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        
        # Plot price line instead of candlesticks for simplicity
        ax.plot(data.index, data['Close'], label='Close', linewidth=2, color='white')
        
        # Add moving averages if calculated
        if 'MA_50' in data.columns:
            ax.plot(data.index, data['MA_50'], label='MA 50', linewidth=1.2, color=self.colors['ma_fast'])
        if 'MA_200' in data.columns:
            ax.plot(data.index, data['MA_200'], label='MA 200', linewidth=1.2, color=self.colors['ma_slow'])
            
        # Mark crossover points
        if 'Golden_Cross' in data.columns:
            golden_points = data[data['Golden_Cross']]
            if not golden_points.empty:
                ax.scatter(golden_points.index, golden_points['Close'], 
                         color=self.colors['bullish'], marker='^', s=120, 
                         label='Golden Cross', zorder=5, edgecolors='white')
                         
        if 'Death_Cross' in data.columns:
            death_points = data[data['Death_Cross']]
            if not death_points.empty:
                ax.scatter(death_points.index, death_points['Close'],
                         color=self.colors['bearish'], marker='v', s=120,
                         label='Death Cross', zorder=5, edgecolors='white')
        
        ax.set_title(f'{ticker} Stock Analysis', fontsize=16, fontweight='bold', color='white')
        ax.legend(loc='upper left', facecolor='black', edgecolor='white')
        ax.grid(True, alpha=0.2, color='gray')
        ax.set_facecolor(self.colors['background'])
    
    def _plot_volume_panel(self, ax, data: pd.DataFrame):
        """Plot volume panel."""
        # Color volume bars based on price movement
        colors = ['green' if close > open_price else 'red' 
                 for close, open_price in zip(data['Close'], data['Open'])]
        
        ax.bar(data.index, data['Volume'], color=colors, alpha=0.6, width=1)
        ax.set_title('Volume', fontsize=12)
        ax.ticklabel_format(style='plain', axis='y')
        ax.grid(True, alpha=0.3)
    
    def _plot_tsi_panel(self, ax, data: pd.DataFrame):
        """Plot TSI indicator panel."""
        if 'TSI' in data.columns:
            ax.plot(data.index, data['TSI'], label='TSI', color=self.colors['line'])
        if 'TSI_Signal' in data.columns:
            ax.plot(data.index, data['TSI_Signal'], label='TSI Signal', color=self.colors['signal'])
            
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-25, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('TSI', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_macd_panel(self, ax, data: pd.DataFrame):
        """Plot MACD indicator panel."""
        if 'MACD' in data.columns:
            ax.plot(data.index, data['MACD'], label='MACD', color=self.colors['line'])
        if 'MACD_Signal' in data.columns:
            ax.plot(data.index, data['MACD_Signal'], label='Signal', color=self.colors['signal'])
        if 'MACD_Hist' in data.columns:
            # Color histogram bars based on value
            colors = ['green' if val > 0 else 'red' for val in data['MACD_Hist']]
            ax.bar(data.index, data['MACD_Hist'], color=colors, alpha=0.7, width=1)
            
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('MACD', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_mfi_panel(self, ax, data: pd.DataFrame):
        """Plot MFI indicator panel."""
        if 'MFI' in data.columns:
            ax.plot(data.index, data['MFI'], label='MFI', color=self.colors['line'])
        if 'MFI_Signal' in data.columns:
            ax.plot(data.index, data['MFI_Signal'], label='MFI Signal', color=self.colors['signal'])
            
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_title('MFI', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cog_panel(self, ax, data: pd.DataFrame):
        """Plot COG indicator panel."""
        if 'COG' in data.columns:
            ax.plot(data.index, data['COG'], label='COG', color=self.colors['line'])
        if 'COG_ALMA' in data.columns:
            ax.plot(data.index, data['COG_ALMA'], label='COG ALMA', color='purple', alpha=0.7)
        if 'COG_LSMA' in data.columns:
            ax.plot(data.index, data['COG_LSMA'], label='COG LSMA', color='orange', alpha=0.7)
            
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('COG', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_kurutoga_panel(self, ax, data: pd.DataFrame):
        """Plot Kurutoga histogram panel with TradingView-style bars."""
        def plot_kurutoga_bars(values, color_pos, color_neg, alpha, width=0.8):
            """Plot histogram bars with different colors for positive/negative values."""
            positive_mask = values >= 0
            negative_mask = values < 0
            
            # Plot positive bars
            if positive_mask.any():
                ax.bar(data.index[positive_mask], values[positive_mask], 
                      color=color_pos, alpha=alpha, width=width, label=None)
            
            # Plot negative bars  
            if negative_mask.any():
                ax.bar(data.index[negative_mask], values[negative_mask],
                      color=color_neg, alpha=alpha, width=width, label=None)
        
        # Plot bars in reverse order (4x, 2x, 1x) so current timeframe is most visible
        if 'Kurutoga_4x' in data.columns:
            plot_kurutoga_bars(data['Kurutoga_4x'], 'darkgreen', 'darkred', 0.3)
        if 'Kurutoga_2x' in data.columns:
            plot_kurutoga_bars(data['Kurutoga_2x'], 'limegreen', 'red', 0.5)
        if 'Kurutoga_Current' in data.columns:
            plot_kurutoga_bars(data['Kurutoga_Current'], 'lightgreen', 'lightcoral', 0.8)
            
        ax.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.8)
        ax.set_title('Kurutoga Histogram', fontsize=12, color='white')
        ax.legend(['4x', '2x', 'Current'], loc='upper left', facecolor='black', edgecolor='white')
        ax.grid(True, alpha=0.2, color='gray')
        ax.set_facecolor(self.colors['background'])
    
    def _plot_momentum_rsi_panel(self, ax, data: pd.DataFrame, config: Dict):
        """Plot momentum and RSI indicators panel with TradingView-style filled areas."""
        # Plot momentum with filled areas
        if config.get('momentum', False) and 'MOM_Norm' in data.columns:
            ax.plot(data.index, data['MOM_Norm'], label='MOM', color='aqua', linewidth=1)
            ax.fill_between(data.index, data['MOM_Norm'], 50, 
                           where=(data['MOM_Norm'] >= 50), color='aqua', alpha=0.2)
            ax.fill_between(data.index, data['MOM_Norm'], 50, 
                           where=(data['MOM_Norm'] < 50), color='#00bcd4', alpha=0.2)
            
        if config.get('rsi', False) and 'RSI' in data.columns:
            ax.plot(data.index, data['RSI'], label='RSI', color='purple', linewidth=1)
            ax.fill_between(data.index, data['RSI'], 50,
                           where=(data['RSI'] >= 50), color='purple', alpha=0.2)
            ax.fill_between(data.index, data['RSI'], 50,
                           where=(data['RSI'] < 50), color='#9c27b0', alpha=0.2)
            
            # RSI levels
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        
        # Add MFI if available
        if 'MFI' in data.columns:
            ax.plot(data.index, data['MFI'], label='MFI', color='lime', linewidth=1)
            ax.fill_between(data.index, data['MFI'], 50,
                           where=(data['MFI'] >= 50), color='lime', alpha=0.2)
            ax.fill_between(data.index, data['MFI'], 50,
                           where=(data['MFI'] < 50), color='#00e676', alpha=0.2)
        
        ax.axhline(y=50, color='white', linestyle='-', linewidth=0.5, alpha=0.8)
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=75, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_title('MOM, MFI, RSI', fontsize=12, color='white')
        ax.legend(loc='upper left', facecolor='black', edgecolor='white')
        ax.grid(True, alpha=0.2, color='gray')
        ax.set_facecolor(self.colors['background'])
    
    def _format_and_save_chart(self, fig, axes: List, ticker: str, timeframe: str, output_file: Path):
        """Format chart and save to file."""
        # Format x-axis for all panels
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Only show x-axis labels on bottom panel
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        
        plt.suptitle(f'{ticker} Technical Analysis - {timeframe.upper()}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.97)
        
        # Save chart
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Chart saved: {output_file}")


def create_indicator_chart(data: pd.DataFrame, ticker: str, output_dir: Union[str, Path],
                          indicators_config: Dict, timeframe: str = '1d') -> str:
    """
    Create technical indicator chart for a single ticker.
    
    Args:
        data: DataFrame with OHLCV data
        ticker: Ticker symbol
        output_dir: Directory to save chart
        indicators_config: Dict specifying which indicators to include
        timeframe: Chart timeframe (default '1d')
        
    Returns:
        str: Path to saved chart file
    """
    chart_generator = IndicatorChartGenerator(output_dir)
    return chart_generator.create_comprehensive_chart(data, ticker, indicators_config, timeframe)


def create_charts_from_config_file(market_data: Dict[str, pd.DataFrame], 
                                  config_file: Union[str, Path],
                                  output_dir: Union[str, Path],
                                  timeframe: str = '1d') -> List[str]:
    """
    Create charts for multiple tickers based on configuration CSV file.
    
    Args:
        market_data: Dict of {ticker: DataFrame} with market data
        config_file: Path to CSV file with ticker and indicator configurations
        output_dir: Directory to save charts
        timeframe: Chart timeframe (default '1d')
        
    Returns:
        List of paths to created chart files
    """
    # Load configuration file
    config_df = pd.read_csv(config_file)
    
    # Ensure ticker column exists
    if 'ticker' not in config_df.columns:
        raise ValueError("Configuration file must have 'ticker' column")
    
    # Get indicator columns (all columns except 'ticker')
    indicator_columns = [col for col in config_df.columns if col != 'ticker']
    
    created_charts = []
    chart_generator = IndicatorChartGenerator(output_dir)
    
    for _, row in config_df.iterrows():
        ticker = row['ticker']
        
        # Skip if no market data available
        if ticker not in market_data:
            logger.warning(f"No market data available for {ticker}")
            continue
        
        # Build indicator configuration for this ticker
        indicators_config = {}
        for indicator in indicator_columns:
            # Convert string TRUE/FALSE to boolean
            value = row[indicator]
            if isinstance(value, str):
                indicators_config[indicator.lower()] = value.upper() == 'TRUE'
            else:
                indicators_config[indicator.lower()] = bool(value)
        
        # Skip if no indicators enabled
        if not any(indicators_config.values()):
            logger.info(f"No indicators enabled for {ticker}")
            continue
        
        # Create chart
        try:
            chart_path = chart_generator.create_comprehensive_chart(
                market_data[ticker], ticker, indicators_config, timeframe
            )
            created_charts.append(chart_path)
            logger.info(f"Chart created for {ticker}: {chart_path}")
            
        except Exception as e:
            logger.error(f"Error creating chart for {ticker}: {e}")
            continue
    
    return created_charts


def analyze_indicator_signals(market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze signals from all tickers and return summary.
    
    Args:
        market_data: Dict of {ticker: DataFrame} with market data
        
    Returns:
        DataFrame with signal analysis for each ticker
    """
    results = []
    
    for ticker, data in market_data.items():
        try:
            # Calculate indicators with default config
            indicators_data = calculate_all_indicators(data)
            
            # Analyze signals
            signals = analyze_combined_signals(indicators_data)
            
            # Add ticker and latest values
            result_row = {'ticker': ticker}
            result_row.update(signals)
            
            # Add latest indicator values
            latest = indicators_data.iloc[-1]
            if 'RSI' in indicators_data.columns:
                result_row['rsi_latest'] = latest['RSI']
            if 'MFI' in indicators_data.columns:
                result_row['mfi_latest'] = latest['MFI']
            if 'TSI' in indicators_data.columns:
                result_row['tsi_latest'] = latest['TSI']
            
            results.append(result_row)
            
        except Exception as e:
            logger.error(f"Error analyzing signals for {ticker}: {e}")
            continue
    
    return pd.DataFrame(results)


def create_sample_indicator_config(tickers: List[str], output_file: Union[str, Path]) -> str:
    """
    Create a sample indicator configuration CSV file.
    
    Args:
        tickers: List of ticker symbols
        output_file: Path for output CSV file
        
    Returns:
        str: Path to created file
    """
    # Define indicators and default settings
    indicators = ['kurutoga', 'tsi', 'macd', 'mfi', 'cog', 'momentum', 'rsi', 'ma_crosses', 'easy_trade']
    
    # Create sample configuration
    config_data = []
    for ticker in tickers:
        row = {'ticker': ticker}
        
        # Set some indicators to TRUE, others to FALSE for variety
        ticker_hash = hash(ticker) % 100
        for i, indicator in enumerate(indicators):
            # Create some variety based on ticker hash
            row[indicator] = 'TRUE' if (ticker_hash + i) % 3 != 0 else 'FALSE'
        
        config_data.append(row)
    
    # Create DataFrame and save
    config_df = pd.DataFrame(config_data)
    config_df.to_csv(output_file, index=False)
    
    logger.info(f"Sample indicator config created: {output_file}")
    return str(output_file)