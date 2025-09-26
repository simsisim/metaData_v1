"""
SR Dashboard Generator Module
============================

Generate multi-panel charts and dashboards for sustainability ratios analysis.
Creates visualizations based on CSV panel configuration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

from .sr_output_manager import get_sr_output_manager

from ..indicators.indicator_parser import calculate_indicator, get_indicator_chart_type

logger = logging.getLogger(__name__)

# Set style for consistent charts
plt.style.use('default')
sns.set_palette("husl")


def plot_candlestick_chart(ax, result: Dict, x_positions: range, main_label: str):
    """
    Plot candlestick chart using OHLCV data from result dict.

    Args:
        ax: Matplotlib axes to plot on
        result: Dictionary containing OHLCV data
        x_positions: Range object for x-axis positions (for weekend gap removal)
        main_label: Label for the chart

    Returns:
        None (plots directly on ax) or list of plot objects for compatibility
    """
    try:
        # Extract OHLCV data from result dict
        open_data = result.get('Open')
        high_data = result.get('High')
        low_data = result.get('Low')
        close_data = result.get('Close')

        # Check if we have proper OHLCV data structure
        missing_data = []
        if not isinstance(open_data, pd.Series): missing_data.append('Open')
        if not isinstance(high_data, pd.Series): missing_data.append('High')
        if not isinstance(low_data, pd.Series): missing_data.append('Low')
        if not isinstance(close_data, pd.Series): missing_data.append('Close')

        if missing_data:
            logger.warning(f"❌ Missing OHLCV data for candlestick chart: {missing_data}")
            logger.info(f"   Available result keys: {list(result.keys())}")

            # Try to find Close data in result for fallback line chart
            close_data = result.get('Close') or result.get('close') or result.get('Close_SPY') or result.get('Close_QQQ')
            if close_data is not None and hasattr(close_data, 'values'):
                logger.info("✅ Found Close data for line chart fallback")
                return ax.plot(x_positions, close_data.values, label=main_label, linewidth=1.5, alpha=0.8, color='blue')
            else:
                # Try to use main data series if available
                for key in result.keys():
                    if hasattr(result[key], 'values') and len(result[key]) > 0:
                        logger.info(f"✅ Using {key} data for line chart fallback")
                        return ax.plot(x_positions, result[key].values, label=main_label, linewidth=1.5, alpha=0.8, color='blue')

                logger.error("❌ No suitable data available for fallback line chart")
                return None

        # Prepare candlestick data
        opens = open_data.values
        highs = high_data.values
        lows = low_data.values
        closes = close_data.values

        # Plot candlestick chart manually (compatible with existing weekend gap removal)
        for i, x_pos in enumerate(x_positions):
            if i >= len(opens):
                break

            open_val = opens[i]
            high_val = highs[i]
            low_val = lows[i]
            close_val = closes[i]

            # Skip if any value is NaN
            if pd.isna(open_val) or pd.isna(high_val) or pd.isna(low_val) or pd.isna(close_val):
                continue

            # Determine color (green for up, red for down)
            color = 'green' if close_val >= open_val else 'red'
            edge_color = 'darkgreen' if close_val >= open_val else 'darkred'

            # Draw high-low line
            ax.plot([x_pos, x_pos], [low_val, high_val], color=edge_color, linewidth=1, alpha=0.8)

            # Draw candlestick body
            body_height = abs(close_val - open_val)
            body_bottom = min(open_val, close_val)

            if body_height > 0:
                # Draw filled rectangle for body
                ax.bar(x_pos, body_height, bottom=body_bottom, width=0.8,
                       color=color, edgecolor=edge_color, alpha=0.7)
            else:
                # Draw thin line for doji (open == close)
                ax.plot([x_pos - 0.4, x_pos + 0.4], [close_val, close_val],
                       color=edge_color, linewidth=2)

        logger.info(f"✅ Candlestick chart plotted for {len(x_positions)} periods")

        # Add label to legend (create invisible line for legend compatibility)
        ax.plot([], [], color='blue', label=main_label, linewidth=0, marker='s', markersize=8, alpha=0.7)

        return None  # Return None for compatibility, actual plotting is done directly on ax

    except Exception as e:
        logger.error(f"❌ Failed to plot candlestick chart: {e}")
        # Fallback to line chart
        close_data = result.get('Close')
        if isinstance(close_data, pd.Series):
            logger.info("🔄 Falling back to line chart")
            return ax.plot(x_positions, close_data.values, label=main_label, linewidth=1.5, alpha=0.8, color='blue')
        else:
            logger.error("❌ No Close data available for fallback")
            return None


def is_bundled_format(data_source: str, panel_data: Dict[str, Any] = None) -> bool:
    """
    Check if data source is bundled format.

    Args:
        data_source: Data source string
        panel_data: Panel data dict (optional)

    Returns:
        True if bundled format
    """
    if panel_data and panel_data.get('is_bundled', False):
        return True

    return '+' in data_source and ('(' in data_source and ')' in data_source)


def extract_base_ticker_from_bundled(data_source: str) -> str:
    """
    Extract base ticker from bundled format.

    Args:
        data_source: Bundled format string like "QQQ + EMA(QQQ,10)"

    Returns:
        Base ticker string like "QQQ"
    """
    if '+' in data_source:
        # Split by + and take first part
        parts = data_source.split('+')
        base_part = parts[0].strip()

        # Remove any prefix if present
        if base_part.startswith(('A_', 'B_')):
            base_part = base_part[2:]

        return base_part

    return data_source


def extract_overlay_info_from_bundled(data_source: str) -> List[str]:
    """
    Extract overlay indicator names from bundled format.

    Args:
        data_source: Bundled format string like "QQQ + EMA(QQQ,10)"

    Returns:
        List of overlay indicator names like ["EMA(10)"]
    """
    overlays = []

    if '+' in data_source:
        parts = data_source.split('+')[1:]  # Skip first part (base ticker)

        for part in parts:
            part = part.strip()
            if '(' in part and ')' in part:
                # Extract indicator name and clean parameters
                if ',' in part:
                    # Format: EMA(QQQ,10) -> EMA(10)
                    indicator_name = part.split('(')[0]
                    params_part = part.split('(')[1].split(')')[0]
                    # Remove ticker from parameters
                    params = [p.strip() for p in params_part.split(',')]
                    # Filter out ticker-like parameters (alphabetic)
                    numeric_params = [p for p in params if p.replace('.', '').isdigit()]

                    if numeric_params:
                        clean_indicator = f"{indicator_name}({','.join(numeric_params)})"
                    else:
                        clean_indicator = indicator_name
                else:
                    clean_indicator = part

                overlays.append(clean_indicator)

    return overlays


def is_indicator_column(column_name: str, expected_indicators: List[str] = None) -> bool:
    """
    Check if column name represents an indicator.

    Args:
        column_name: Column name to check
        expected_indicators: Optional list of expected indicators

    Returns:
        True if column represents an indicator
    """
    # Standard OHLCV columns
    if column_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
        return False

    # Enhanced indicator patterns
    indicator_patterns = [
        'ema_', 'sma_', 'ma_',  # Moving averages
        'ppo_', 'macd_',        # Oscillators
        'rsi_', 'stoch_',       # Momentum
        'bb_', 'bands_',        # Bands
        'ratio_'                # Ratios
    ]

    column_lower = column_name.lower()

    # Check for enhanced patterns
    for pattern in indicator_patterns:
        if column_lower.startswith(pattern):
            return True

    # Legacy patterns
    legacy_patterns = ['ma', 'ema', 'sma', 'ppo', 'rsi', 'macd', 'ratio']
    for pattern in legacy_patterns:
        if pattern in column_lower:
            return True

    return False


def format_trading_days_axis(ax, dates: pd.DatetimeIndex, data_length: int):
    """
    Format x-axis for trading days only (no weekend gaps).

    Args:
        ax: Matplotlib axis
        dates: Original datetime index
        data_length: Number of data points to display
    """
    try:
        if data_length <= 0:
            return

        # Set x-axis limits to index positions (0 to data_length-1)
        ax.set_xlim(0, data_length - 1)

        # Create custom tick positions every ~10-15 trading days
        tick_interval = max(1, data_length // 8)  # Aim for ~8 ticks maximum
        tick_positions = list(range(0, data_length, tick_interval))

        # Always include first and last positions
        if 0 not in tick_positions:
            tick_positions.insert(0, 0)
        if (data_length - 1) not in tick_positions:
            tick_positions.append(data_length - 1)

        # Remove duplicates and sort
        tick_positions = sorted(list(set(tick_positions)))

        # Create labels from actual dates at those positions
        tick_labels = []
        for pos in tick_positions:
            if pos < len(dates):
                date_str = dates[pos].strftime('%Y-%m-%d')
                tick_labels.append(date_str)
            else:
                tick_labels.append('')

        # Apply custom ticks
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        logger.debug(f"Trading days axis formatting applied: {len(tick_positions)} ticks for {data_length} data points")

    except Exception as e:
        logger.warning(f"Failed to format trading days axis: {e}")
        # Fallback to default formatting
        ax.tick_params(axis='x', rotation=45)


def generate_overlay_label(column_name: str, expected_overlays: List[str], indicator: str = None) -> str:
    """
    Generate intelligent label for overlay indicators.

    Args:
        column_name: Column name like "EMA_ema"
        expected_overlays: Expected overlay names like ["EMA(10)"]
        indicator: Indicator name from panel info

    Returns:
        Clean label for legend
    """
    # Try to match with expected overlays first
    for expected in expected_overlays:
        if expected.lower().startswith(column_name.split('_')[0].lower()):
            return expected

    # Generate from column name
    parts = column_name.split('_')
    if len(parts) >= 2:
        indicator_type = parts[0].upper()

        # Common indicator mappings
        if indicator_type == 'EMA':
            return f"EMA"  # Will be enhanced with parameters if available
        elif indicator_type == 'SMA':
            return f"SMA"
        elif indicator_type == 'PPO':
            return f"PPO"
        elif indicator_type == 'RSI':
            return f"RSI"
        else:
            return indicator_type

    # Fallback to cleaned column name
    return column_name.replace('_', ' ').title()


def create_multi_panel_chart(panel_results: Dict[str, Any],
                           output_path: str,
                           chart_title: str = "Sustainability Ratios Dashboard",
                           user_config = None) -> str:
    """
    Create multi-panel chart based on panel results.

    Args:
        panel_results: Dict with panel calculation results
        output_path: Path to save chart
        chart_title: Chart title

    Returns:
        Path to saved chart file
    """
    # DEBUG: Function call path verification
    # print("🎯 ENHANCED CHART GENERATION CALLED!")  # Debug output
    logger.info(f"🚀 FUNCTION CALL: create_multi_panel_chart()")
    logger.info(f"   panel_results keys: {list(panel_results.keys()) if panel_results else 'None'}")
    logger.info(f"   output_path: {output_path}")
    logger.info(f"   chart_title: {chart_title}")
    # print(f"📊 Panel results keys: {list(panel_results.keys()) if panel_results else 'None'}")  # Debug output

    try:
        # Count panels with data
        valid_panels = {k: v for k, v in panel_results.items()
                       if v.get('result') and not v['result'].get('metadata', {}).get('error')}

        if not valid_panels:
            logger.warning("No valid panels for chart generation")
            return ""

        # Sort panels by stacking_order if available
        sorted_panels = sorted(valid_panels.items(),
                             key=lambda x: x[1].get('result', {}).get('metadata', {}).get('stacking_order', 999))

        # Determine chart layout - use pure vertical stacking for positioned panels
        num_panels = len(valid_panels)

        # Check if any panels have positioning (above/below) - if so, use vertical stacking only
        has_positioned_panels = any(
            panel_data.get('result', {}).get('metadata', {}).get('position') in ['above', 'below']
            for panel_data in valid_panels.values()
        )

        if has_positioned_panels:
            # Pure vertical stacking for positioned panels (PPO above Panel_1, etc.)
            rows, cols = num_panels, 1
        else:
            # Traditional grid layout for regular panels
            if num_panels <= 2:
                rows, cols = 2, 1
            elif num_panels <= 4:
                rows, cols = 2, 2
            elif num_panels <= 6:
                rows, cols = 3, 2
            else:
                rows, cols = 3, 3

        # Create figure and subplots with increased height for better panel visibility
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

        panel_idx = 0
        for panel_name, panel_data in sorted_panels:
            if panel_idx >= rows * cols:
                break

            row = panel_idx // cols
            col = panel_idx % cols

            ax = fig.add_subplot(gs[row, col])

            # Plot panel data with position info
            logger.info(f"🚀 CALLING plot_panel for '{panel_name}' (panel {panel_idx + 1}/{len(valid_panels)})")
            plot_panel(ax, panel_data, panel_name, user_config)
            panel_idx += 1

        # Add main title
        fig.suptitle(chart_title, fontsize=16, fontweight='bold', y=0.95)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.99, 0.01, f"Generated: {timestamp}", ha='right', va='bottom',
                fontsize=8, alpha=0.7)

        # Save chart
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Created multi-panel chart: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating multi-panel chart: {e}")
        plt.close('all')  # Clean up on error
        return ""


def plot_panel(ax, panel_data: Dict[str, Any], panel_name: str, user_config = None):
    """
    Plot data for a single panel.

    Args:
        ax: Matplotlib axis
        panel_data: Panel data dict
        panel_name: Panel name for title
    """
    try:
        # DEBUG: Log function entry
        logger.info(f"\n🎯 plot_panel CALLED:")
        logger.info(f"   panel_name: '{panel_name}'")

        data_source = panel_data.get('data_source', 'Unknown')
        indicator = panel_data.get('indicator', '')
        result = panel_data.get('result', {})

        logger.info(f"   data_source: '{data_source}'")
        logger.info(f"   indicator: '{indicator}'")
        logger.info(f"   panel_data keys: {list(panel_data.keys())}")
        logger.info(f"   result keys: {list(result.keys()) if result else 'None'}")
        logger.info(f"   is_bundled in panel_data: {panel_data.get('is_bundled', False)}")

        # DEBUG: Show what type of data we actually receive
        # print(f"🔍 PLOT_PANEL RECEIVED:")  # Debug output
        # print(f"   result type: {type(result)}")  # Debug output
        # if result:  # Debug output
        #     for key, value in result.items():  # Debug output
        #         print(f"   {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")  # Debug output

        if not result:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{panel_name}: {data_source}")
            return

        # 🔧 NEW: Apply chart display range filtering
        # Get chart display setting from user config (default: show all data)
        try:
            chart_display_limit = getattr(user_config, 'sr_chart_display', None) if user_config else None

            if chart_display_limit and chart_display_limit > 0:
                # Filter all pandas Series in result to show only the last N data points
                filtered_result = {}
                for key, value in result.items():
                    if hasattr(value, 'tail') and hasattr(value, '__len__'):  # pandas Series/DataFrame
                        if len(value) > chart_display_limit:
                            filtered_result[key] = value.tail(chart_display_limit)
                            logger.info(f"📊 Chart display filtering applied to '{key}': {len(value)} → {len(filtered_result[key])} data points")
                        else:
                            filtered_result[key] = value
                    else:
                        filtered_result[key] = value  # Keep non-series data as-is (metadata, etc.)

                result = filtered_result
                logger.info(f"📊 Chart display limit applied: showing last {chart_display_limit} data points")
            else:
                logger.info(f"📊 No chart display limit configured: showing all available data")

        except Exception as e:
            logger.warning(f"⚠️ Failed to apply chart display filtering: {e}")
            # Continue with original result if filtering fails

        metadata = result.get('metadata', {})
        indicator_chart_type = metadata.get('chart_type', 'overlay')

        # Extract new chart_type for main series display (candle, line, no_drawing)
        main_chart_type = panel_data.get('chart_type', 'candle')

        # DEBUG: Log chart type detection
        logger.info(f"🎯 CHART TYPE DETECTION:")
        logger.info(f"   metadata: {metadata}")
        logger.info(f"   indicator_chart_type: '{indicator_chart_type}' (for indicator display)")
        logger.info(f"   main_chart_type: '{main_chart_type}' (for main series display)")
        logger.info(f"   indicator: '{indicator}'")

        # Special handling for oscillator indicators (PPO, RSI, MACD)
        if indicator_chart_type == 'oscillator' and indicator:
            # print(f"🎯 OSCILLATOR DETECTED - Routing to plot_indicator_chart")  # Debug output
            logger.info(f"🎯 OSCILLATOR DETECTED - Routing directly to plot_indicator_chart")
            plot_indicator_chart(ax, result, None, data_source, indicator, False)

            # Set title for oscillator
            title = f"{panel_name}: {indicator}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            return

        # Check if this is bundled format
        is_bundled = is_bundled_format(data_source, panel_data)
        logger.info(f"📊 BUNDLED FORMAT CHECK:")
        logger.info(f"   is_bundled_format() result: {is_bundled}")

        # Get the main data series to plot
        main_data_key = None
        logger.info(f"🔍 MAIN SERIES DETECTION:")

        if is_bundled:
            logger.info(f"   Using BUNDLED format logic")
            # For bundled format, prioritize Close price of base ticker
            base_ticker = extract_base_ticker_from_bundled(data_source)
            logger.info(f"   base_ticker: '{base_ticker}'")

            # Look for Close price first (primary chart data)
            for key in result.keys():
                if key not in ['metadata'] and isinstance(result[key], pd.Series):
                    if key == 'Close':
                        main_data_key = key
                        logger.info(f"   ✅ Found Close price: '{key}'")
                        break

            # Fallback to base ticker name or price-related keys
            if main_data_key is None:
                logger.info(f"   Close not found, trying fallback...")
                for key in result.keys():
                    if key not in ['metadata'] and isinstance(result[key], pd.Series):
                        if key == base_ticker or 'price' in key.lower():
                            main_data_key = key
                            logger.info(f"   ✅ Found fallback: '{key}'")
                            break
        else:
            logger.info(f"   Using STANDARD format logic")
            # Standard format - original logic
            for key in result.keys():
                if key not in ['metadata'] and isinstance(result[key], pd.Series):
                    if 'price' in key.lower() or 'close' in key.lower() or key == data_source:
                        main_data_key = key
                        logger.info(f"   ✅ Found standard main series: '{key}'")
                        break

        if main_data_key is None:
            logger.info(f"   No specific main series found, using first available...")
            # Use first available series
            for key in result.keys():
                if key != 'metadata' and isinstance(result[key], pd.Series):
                    main_data_key = key
                    logger.info(f"   ✅ Using first available: '{key}'")
                    break

        if main_data_key is None:
            ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{panel_name}: {data_source}")
            return

        main_series = result[main_data_key]

        if indicator_chart_type == 'overlay' or not indicator:
            # Plot price data with indicator overlay
            logger.info(f"🚀 CALLING plot_overlay_chart (main_chart_type='{main_chart_type}', indicator='{indicator}')")
            ax = plot_overlay_chart(ax, result, main_series, data_source, indicator, is_bundled, main_chart_type)
        else:
            # Plot indicator in subplot style
            logger.info(f"🚀 CALLING plot_indicator_chart (indicator_chart_type='{indicator_chart_type}', indicator='{indicator}')")
            plot_indicator_chart(ax, result, main_series, data_source, indicator, is_bundled)

        # Set title (only if ax is still valid)
        if ax is not None:
            title = f"{panel_name}: {data_source}"
            if indicator:
                title += f" + {indicator.split('(')[0]}"
            ax.set_title(title, fontsize=10, fontweight='bold')

            # Date formatting is now handled in plot_overlay_chart and plot_indicator_chart

    except Exception as e:
        import traceback
        logger.error(f"❌ CRITICAL ERROR in plot_panel for '{panel_name}':")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception message: {str(e)}")
        logger.error(f"   data_source: '{data_source}'")
        logger.error(f"   indicator: '{indicator}'")
        logger.error(f"   chart_type: {chart_type}")
        logger.error(f"   is_bundled: {is_bundled}")
        logger.error(f"   Full traceback: {traceback.format_exc()}")

        # Still display error on chart for user visibility
        ax.text(0.5, 0.5, f'Plot Error: {str(e)[:50]}...', ha='center', va='center',
               transform=ax.transAxes, fontsize=8, color='red')
        ax.set_title(f"ERROR: {panel_name}", color='red')


def plot_overlay_chart(ax, result: Dict, main_series: pd.Series, data_source: str, indicator: str, is_bundled: bool = False, chart_type: str = 'candle'):
    """Plot chart with indicator overlaid on price data."""
    try:
        # DEBUG: Log function entry
        # print(f"🎯 PLOT_OVERLAY_CHART: data_source='{data_source}', result_keys={list(result.keys())}")  # Debug output
        logger.info(f"🎯 plot_overlay_chart CALLED:")
        logger.info(f"   data_source: '{data_source}'")
        logger.info(f"   indicator: '{indicator}'")
        logger.info(f"   is_bundled: {is_bundled}")
        logger.info(f"   result keys: {list(result.keys())}")
        logger.info(f"   main_series.name: {main_series.name}")
        logger.info(f"   main_series length: {len(main_series)}")

        # 🔧 NEW: Check if Volume data is available for automatic volume display
        has_volume = 'Volume' in result and isinstance(result['Volume'], pd.Series) and not result['Volume'].empty

        if has_volume:
            # Create two subplots: top for price, bottom for volume
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt

            # Get the position and figure reference BEFORE removing axis
            pos = ax.get_position()
            fig = ax.figure

            # Remove the original axis
            ax.remove()
            gs = gridspec.GridSpec(2, 1, height_ratios=[8, 2],
                                 left=pos.x0, right=pos.x1,
                                 bottom=pos.y0, top=pos.y1,
                                 hspace=0.1)

            # Create price subplot (top 70%)
            ax_price = fig.add_subplot(gs[0])
            # Create volume subplot (bottom 30%)
            ax_volume = fig.add_subplot(gs[1], sharex=ax_price)

            # Use price axis for main plotting
            ax = ax_price

            # print(f"📊 VOLUME DISPLAY: Created price + volume subplots")  # Debug output
        else:
            # No volume data, use single plot as before
            ax_volume = None
            # print(f"📊 SINGLE PLOT: No volume data available")  # Debug output

        # Determine main series label
        if is_bundled:
            logger.info(f"📊 BUNDLED FORMAT DETECTED - Using enhanced logic")
            base_ticker = extract_base_ticker_from_bundled(data_source)
            main_label = base_ticker
            expected_overlays = extract_overlay_info_from_bundled(data_source)
            logger.info(f"   base_ticker: '{base_ticker}'")
            logger.info(f"   main_label: '{main_label}'")
            logger.info(f"   expected_overlays: {expected_overlays}")
        else:
            logger.info(f"📊 STANDARD FORMAT - Using original logic")
            main_label = data_source
            expected_overlays = []

        # Plot main data (price)
        logger.info(f"🎨 PLOTTING MAIN SERIES:")
        logger.info(f"   label: '{main_label}'")
        logger.info(f"   color: blue (default)")
        logger.info(f"   data type: {type(main_series)}")
        logger.info(f"   data length: {len(main_series)}")
        logger.info(f"   data range: {main_series.min():.2f} to {main_series.max():.2f}")
        logger.info(f"   data index type: {type(main_series.index)}")
        logger.info(f"   first 3 values: {main_series.head(3).tolist()}")
        logger.info(f"   last 3 values: {main_series.tail(3).tolist()}")
        logger.info(f"   has NaN values: {main_series.isna().any()}")
        logger.info(f"   NaN count: {main_series.isna().sum()}")

        try:
            # Use index positions instead of dates for no weekend gaps
            x_positions = range(len(main_series))

            # Chart type routing logic
            if chart_type == 'no_drawing':
                # Skip plotting main series entirely
                logger.info(f"🚫 Skipping main series plot (chart_type='no_drawing')")
                main_line = None
            elif chart_type == 'candle':
                # Plot candlestick chart
                main_line = plot_candlestick_chart(ax, result, x_positions, main_label)
                logger.info(f"✅ Candlestick chart plotted")
            else:  # chart_type == 'line' (default)
                # Current line chart logic
                main_line = ax.plot(x_positions, main_series.values,
                                   label=main_label, linewidth=1.5, alpha=0.8, color='blue')
                logger.info(f"✅ Line chart plotted: {len(main_line)} lines created")
        except Exception as e:
            logger.error(f"❌ FAILED to plot main series: {e}")
            logger.error(f"   chart_type: {chart_type}")
            logger.error(f"   main_series type: {type(main_series)}")
            logger.error(f"   main_series.index type: {type(main_series.index)}")
            logger.error(f"   main_series.values type: {type(main_series.values)}")
            raise e

        # Plot overlays
        overlay_count = 0
        overlay_colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']

        logger.info(f"🔍 OVERLAY DETECTION STARTING:")
        logger.info(f"   is_bundled: {is_bundled}")

        if is_bundled:
            logger.info(f"📊 ENHANCED OVERLAY DETECTION (Bundled Format)")
            # Enhanced overlay detection for bundled format
            plotted_indicators = set()  # Track plotted indicators to avoid duplicates

            for key, series in result.items():
                logger.info(f"\n   🔎 Checking column: '{key}'")

                if key not in ['metadata'] and isinstance(series, pd.Series):
                    is_indicator = is_indicator_column(key)
                    is_not_main = key != main_series.name

                    logger.info(f"      is_metadata: {key in ['metadata']}")
                    logger.info(f"      is_series: {isinstance(series, pd.Series)}")
                    logger.info(f"      is_indicator_column: {is_indicator}")
                    logger.info(f"      key != main_series.name: {is_not_main} ('{key}' != '{main_series.name}')")

                    if is_indicator and is_not_main:
                        logger.info(f"      ✅ Passed basic checks")

                        # Generate intelligent label for overlay
                        overlay_label = generate_overlay_label(key, expected_overlays, indicator)
                        logger.info(f"      overlay_label: '{overlay_label}'")

                        # Only plot primary indicator columns (avoid duplicates)
                        indicator_type = key.split('_')[0].upper()
                        logger.info(f"      indicator_type: '{indicator_type}'")
                        logger.info(f"      plotted_indicators: {plotted_indicators}")

                        should_plot = False

                        if indicator_type not in plotted_indicators:
                            # Check if this is a primary indicator column
                            ends_with_ema = key.endswith('_ema')
                            ends_with_sma = key.endswith('_sma')
                            ends_with_ppo = key.endswith('_ppo')
                            ends_with_rsi = key.endswith('_rsi')
                            ends_with_macd = key.endswith('_macd')
                            equals_type = key == indicator_type

                            logger.info(f"      ends_with_ema: {ends_with_ema}")
                            logger.info(f"      ends_with_sma: {ends_with_sma}")
                            logger.info(f"      ends_with_ppo: {ends_with_ppo}")
                            logger.info(f"      ends_with_rsi: {ends_with_rsi}")
                            logger.info(f"      ends_with_macd: {ends_with_macd}")
                            logger.info(f"      equals_type: {equals_type}")

                            if (ends_with_ema or ends_with_sma or ends_with_ppo or
                                ends_with_rsi or ends_with_macd or equals_type):
                                should_plot = True
                                plotted_indicators.add(indicator_type)
                                logger.info(f"      ✅ PRIMARY INDICATOR - WILL PLOT")
                            else:
                                logger.info(f"      ❌ Not a primary indicator column")
                        else:
                            logger.info(f"      ❌ Already plotted indicator type '{indicator_type}'")

                        if should_plot:
                            # Use different color for each overlay
                            color = overlay_colors[overlay_count % len(overlay_colors)]
                            logger.info(f"      🎨 PLOTTING OVERLAY:")
                            logger.info(f"         label: '{overlay_label}'")
                            logger.info(f"         color: {color}")
                            logger.info(f"         data type: {type(series)}")
                            logger.info(f"         data length: {len(series)}")
                            logger.info(f"         data range: {series.min():.2f} to {series.max():.2f}")
                            logger.info(f"         data index type: {type(series.index)}")
                            logger.info(f"         first 3 values: {series.head(3).tolist()}")
                            logger.info(f"         last 3 values: {series.tail(3).tolist()}")
                            logger.info(f"         has NaN values: {series.isna().any()}")
                            logger.info(f"         NaN count: {series.isna().sum()}")
                            logger.info(f"         overlay #{overlay_count + 1}")

                            try:
                                # Use index positions for overlay plotting (no weekend gaps)
                                overlay_x_positions = range(len(series))
                                overlay_line = ax.plot(overlay_x_positions, series.values,
                                                     label=overlay_label,
                                                     linewidth=1,
                                                     alpha=0.7,
                                                     color=color)
                                overlay_count += 1
                                logger.info(f"         ✅ Overlay plotted: {len(overlay_line)} lines created")
                            except Exception as e:
                                logger.error(f"         ❌ FAILED to plot overlay '{overlay_label}': {e}")
                                logger.error(f"            series type: {type(series)}")
                                logger.error(f"            series.index type: {type(series.index)}")
                                logger.error(f"            series.values type: {type(series.values)}")
                                logger.error(f"            color: {color}")
                                # Continue with next overlay instead of failing completely
                    else:
                        logger.info(f"      ❌ Failed basic checks")
                else:
                    logger.info(f"      ❌ Skipped: metadata or not series")

        elif indicator:
            logger.info(f"📊 STANDARD OVERLAY DETECTION (Non-bundled Format)")
            # Standard overlay detection for non-bundled format
            for key, series in result.items():
                logger.info(f"   🔎 Checking column: '{key}'")
                if key not in ['metadata', 'price'] and isinstance(series, pd.Series):
                    has_ma = 'ma' in key.lower()
                    has_ema = 'ema' in key.lower()
                    has_sma = 'sma' in key.lower()
                    logger.info(f"      has_ma: {has_ma}, has_ema: {has_ema}, has_sma: {has_sma}")

                    if has_ma or has_ema or has_sma:
                        logger.info(f"      🎨 PLOTTING STANDARD OVERLAY: '{key}'")
                        # Use index positions for standard overlays (no weekend gaps)
                        std_overlay_x_positions = range(len(series))
                        overlay_line = ax.plot(std_overlay_x_positions, series.values, label=key, linewidth=1, alpha=0.7)
                        overlay_count += 1
                        logger.info(f"         ✅ Standard overlay plotted: {len(overlay_line)} lines created")
        else:
            logger.info(f"📊 NO OVERLAY DETECTION - No indicator specified and not bundled")

        # 🔧 NEW: Plot volume data if available
        if has_volume and 'ax_volume' in locals() and ax_volume is not None:
            try:
                volume_series = result['Volume']
                logger.info(f"📊 PLOTTING VOLUME:")
                logger.info(f"   Volume data length: {len(volume_series)}")
                logger.info(f"   Volume data range: {volume_series.min():.0f} to {volume_series.max():.0f}")

                # Plot volume bars using index positions (no weekend gaps)
                volume_x_positions = range(len(volume_series))
                ax_volume.bar(volume_x_positions, volume_series.values,
                             alpha=0.7, color='gray', width=0.8)
                ax_volume.set_ylabel('Volume', fontsize=8)
                ax_volume.tick_params(axis='y', labelsize=8)
                ax_volume.grid(True, alpha=0.3)

                # Format volume y-axis to show in millions/billions
                ax_volume.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x < 1e9 else f'{x/1e9:.1f}B'))

                logger.info(f"✅ Volume subplot created successfully")
            except Exception as e:
                logger.error(f"❌ FAILED to plot volume: {e}")

        logger.info(f"\n🎨 FINAL CHART ASSEMBLY:")
        logger.info(f"   Total overlays plotted: {overlay_count}")

        # Set up legend and grid
        try:
            ax.legend(fontsize=8)
            logger.info(f"✅ Legend created successfully")
        except Exception as e:
            logger.error(f"❌ FAILED to create legend: {e}")

        try:
            ax.grid(True, alpha=0.3)
            logger.info(f"✅ Grid enabled successfully")
        except Exception as e:
            logger.error(f"❌ FAILED to enable grid: {e}")

        # Get final chart info
        try:
            lines = ax.get_lines()
            legend = ax.get_legend()
            legend_labels = [text.get_text() for text in legend.get_texts()] if legend else []
            logger.info(f"✅ Chart info retrieval successful")
        except Exception as e:
            logger.error(f"❌ FAILED to retrieve chart info: {e}")
            lines = []
            legend = None
            legend_labels = []

        logger.info(f"📊 FINAL CHART RESULTS:")
        logger.info(f"   Total lines on chart: {len(lines)}")
        logger.info(f"   Legend labels: {legend_labels}")
        logger.info(f"   Legend visible: {legend.get_visible() if legend else 'No legend'}")
        logger.info(f"   Legend entries count: {len(legend_labels)}")

        for i, line in enumerate(lines):
            color = line.get_color()
            label = line.get_label()
            linewidth = line.get_linewidth()
            alpha = line.get_alpha()
            visible = line.get_visible()
            data_count = len(line.get_xdata())
            logger.info(f"   Line {i+1}: '{label}'")
            logger.info(f"     color: {color}")
            logger.info(f"     linewidth: {linewidth}")
            logger.info(f"     alpha: {alpha}")
            logger.info(f"     visible: {visible}")
            logger.info(f"     data points: {data_count}")

        # 🔧 NEW: Plot volume data if available
        if has_volume and ax_volume is not None:
            try:
                volume_data = result['Volume']
                logger.info(f"📊 PLOTTING VOLUME DATA:")
                logger.info(f"   Volume series length: {len(volume_data)}")
                logger.info(f"   Volume range: {volume_data.min():,.0f} to {volume_data.max():,.0f}")

                # Plot volume as bars using index positions (no weekend gaps)
                volume_x_pos = range(len(volume_data))
                bars = ax_volume.bar(volume_x_pos, volume_data.values,
                                   width=0.8, alpha=0.6, color='gray', label='Volume')

                # Format volume axis
                ax_volume.set_ylabel('Volume', fontsize=9)
                ax_volume.tick_params(axis='y', labelsize=8)
                ax_volume.tick_params(axis='x', labelsize=8)
                ax_volume.grid(True, alpha=0.3)

                # Format volume numbers (e.g., 1M, 2.5K)
                def format_volume(x, p):
                    if x >= 1e9:
                        return f'{x/1e9:.1f}B'
                    elif x >= 1e6:
                        return f'{x/1e6:.1f}M'
                    elif x >= 1e3:
                        return f'{x/1e3:.1f}K'
                    else:
                        return f'{x:.0f}'

                from matplotlib.ticker import FuncFormatter
                ax_volume.yaxis.set_major_formatter(FuncFormatter(format_volume))

                # Remove x-axis labels from price chart (volume chart will show them)
                ax.tick_params(axis='x', labelbottom=False)

                # Ensure shared x-axis
                ax_volume.sharex(ax)

                print(f"✅ VOLUME CHART: Volume bars plotted successfully")  # Keep - user needs this
                logger.info(f"✅ Volume bars plotted successfully")

            except Exception as e:
                logger.error(f"❌ FAILED to plot volume: {e}")
                print(f"❌ VOLUME ERROR: {e}")  # Keep - error reporting

        # Apply custom date formatting for trading days (no weekend gaps)
        try:
            format_trading_days_axis(ax, main_series.index, len(main_series))
            logger.info(f"✅ Trading days axis formatting applied")
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply trading days formatting: {e}")

        logger.info(f"✅ plot_overlay_chart COMPLETED SUCCESSFULLY")

        # Return the axis that should be used for title and formatting
        return ax

    except Exception as e:
        logger.error(f"❌ ERROR in plot_overlay_chart: {e}")
        logger.error(f"   data_source: '{data_source}'")
        logger.error(f"   is_bundled: {is_bundled}")
        logger.error(f"   indicator: '{indicator}'")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        # Return None to indicate error
        return None


def plot_indicator_chart(ax, result: Dict, main_series: pd.Series, data_source: str, indicator: str, is_bundled: bool = False):
    """Plot indicator chart (oscillator style)."""
    try:
        # DEBUG: Log function entry for PPO debugging
        # print(f"🎯 plot_indicator_chart CALLED: indicator='{indicator}', keys={list(result.keys())}")  # Debug output
        logger.info(f"🎯 plot_indicator_chart CALLED:")
        logger.info(f"   indicator: '{indicator}'")
        logger.info(f"   result keys: {list(result.keys())}")
        logger.info(f"   data_source: '{data_source}'")

        indicator_name = indicator.split('(')[0].lower()
        metadata = result.get('metadata', {})

        # Check if this is a multi-component indicator like PPO or MACD
        has_ppo = 'ppo' in result and isinstance(result['ppo'], pd.Series)
        has_macd = 'macd' in result and isinstance(result['macd'], pd.Series)
        has_signal = 'signal' in result and isinstance(result['signal'], pd.Series)
        has_histogram = 'histogram' in result and isinstance(result['histogram'], pd.Series)

        logger.info(f"   Multi-component check:")
        logger.info(f"      has_ppo: {has_ppo}")
        logger.info(f"      has_macd: {has_macd}")
        logger.info(f"      has_signal: {has_signal}")
        logger.info(f"      has_histogram: {has_histogram}")

        if (has_ppo or has_macd) and has_signal and has_histogram:
            # Multi-component oscillator indicator (PPO or MACD)
            indicator_type = 'PPO' if has_ppo else 'MACD'
            main_line_key = 'ppo' if has_ppo else 'macd'
            # print(f"   📊 PLOTTING {indicator_type} MULTI-COMPONENT - has_{main_line_key.lower()}={has_ppo or has_macd}, has_signal={has_signal}, has_histogram={has_histogram}")  # Debug output
            logger.info(f"   📊 PLOTTING {indicator_type} MULTI-COMPONENT")

            # Get color scheme from metadata
            color_scheme = metadata.get('color_scheme', {})
            main_line_color = color_scheme.get(main_line_key, 'blue')  # 'ppo' or 'macd' color
            signal_color = color_scheme.get('signal', 'red')
            hist_pos_color = color_scheme.get('histogram_positive', 'green')
            hist_neg_color = color_scheme.get('histogram_negative', 'red')

            # Plot main line (PPO or MACD) using index positions (no weekend gaps)
            main_line_data = result[main_line_key]
            main_x_positions = range(len(main_line_data))
            ax.plot(main_x_positions, main_line_data.values,
                   label=indicator_type, linewidth=1.5, color=main_line_color)
            logger.info(f"      ✅ {indicator_type} line plotted: {len(main_line_data)} points")

            # Plot Signal line using index positions (no weekend gaps)
            signal_data = result['signal']
            signal_x_positions = range(len(signal_data))
            ax.plot(signal_x_positions, signal_data.values,
                   label='Signal', linewidth=1, color=signal_color)
            logger.info(f"      ✅ Signal line plotted: {len(signal_data)} points")

            # Plot Histogram as bars
            histogram_data = result['histogram']

            # Create positive and negative histogram data for different colors
            positive_hist = histogram_data.where(histogram_data >= 0, 0)
            negative_hist = histogram_data.where(histogram_data < 0, 0)

            # Plot histogram bars
            bar_width = (histogram_data.index[1] - histogram_data.index[0]).days if len(histogram_data) > 1 else 1

            # print(f"      📊 Plotting histogram bars: bar_width={bar_width}, pos_values={positive_hist.sum():.3f}, neg_values={negative_hist.sum():.3f}")  # Debug output

            # Plot histogram bars using index positions (no weekend gaps)
            hist_x_positions = range(len(histogram_data))
            ax.bar(hist_x_positions, positive_hist.values,
                   width=0.8, color=hist_pos_color, alpha=0.6, label='Histogram+')
            ax.bar(hist_x_positions, negative_hist.values,
                   width=0.8, color=hist_neg_color, alpha=0.6, label='Histogram-')

            # print(f"      ✅ Histogram bars plotted successfully")  # Debug output
            logger.info(f"      ✅ Histogram bars plotted: {len(histogram_data)} bars")
            logger.info(f"         Positive values: {(histogram_data >= 0).sum()}")
            logger.info(f"         Negative values: {(histogram_data < 0).sum()}")
            logger.info(f"         Bar width: {bar_width}")

        else:
            # Single component indicator - original logic
            logger.info(f"   📊 PLOTTING SINGLE COMPONENT INDICATOR")
            indicator_data = None

            for key, series in result.items():
                if key != 'metadata' and isinstance(series, pd.Series):
                    if indicator_name in key.lower():
                        indicator_data = series
                        logger.info(f"      Found indicator data: '{key}'")
                        break

            if indicator_data is not None:
                # Plot indicator using index positions (no weekend gaps)
                indicator_x_positions = range(len(indicator_data))
                ax.plot(indicator_x_positions, indicator_data.values,
                       label=indicator, linewidth=1.5, color='blue')
                logger.info(f"      ✅ Single indicator plotted: {len(indicator_data)} points")
            else:
                # Fallback to main series using index positions (no weekend gaps)
                main_x_positions = range(len(main_series))
                ax.plot(main_x_positions, main_series.values, label=data_source, linewidth=1.5)
                logger.info(f"      ✅ Fallback to main series: {len(main_series)} points")

        # Add threshold lines if available
        if 'overbought' in metadata:
            ax.axhline(y=metadata['overbought'], color='red', linestyle='--', alpha=0.5, label='Overbought')
            logger.info(f"      ✅ Overbought line added: {metadata['overbought']}")
        if 'oversold' in metadata:
            ax.axhline(y=metadata['oversold'], color='green', linestyle='--', alpha=0.5, label='Oversold')
            logger.info(f"      ✅ Oversold line added: {metadata['oversold']}")

        # Add zero line for oscillators
        if metadata.get('zero_line'):
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            logger.info(f"      ✅ Zero line added")

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Apply custom date formatting for trading days (no weekend gaps)
        try:
            # Get any series with datetime index for date formatting
            datetime_series = None
            for key, series in result.items():
                if isinstance(series, pd.Series) and hasattr(series.index, 'to_pydatetime'):
                    datetime_series = series
                    break

            if datetime_series is not None:
                format_trading_days_axis(ax, datetime_series.index, len(datetime_series))
                logger.info(f"      ✅ Trading days axis formatting applied to indicator chart")
        except Exception as e:
            logger.warning(f"      ⚠️ Failed to apply trading days formatting: {e}")

        logger.info(f"   ✅ plot_indicator_chart completed successfully")

    except Exception as e:
        logger.error(f"❌ Error plotting indicator chart: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")


def create_ratio_dashboard(ratio_results: pd.DataFrame, output_path: str) -> str:
    """
    Create dashboard for intermarket ratios.

    Args:
        ratio_results: DataFrame with ratio calculations
        output_path: Path to save dashboard

    Returns:
        Path to saved file
    """
    try:
        if ratio_results.empty:
            logger.warning("No ratio data for dashboard")
            return ""

        # Create figure with subplots - increased height for better panel visibility
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Intermarket Ratios Dashboard', fontsize=16, fontweight='bold')

        # Plot ratios from data (no hardcoded ratios)
        ratio_plots = []

        # Extract available ratio columns from data
        if not ratio_results.empty:
            for col in ratio_results.columns:
                if '_' in col and not col.endswith('_SMA20') and not col.endswith('_SMA50'):
                    ratio_plots.append((col, col.replace('_', ' vs ')))

        # Limit to 4 ratios maximum
        ratio_plots = ratio_plots[:4]

        for idx, (ratio_col, title) in enumerate(ratio_plots):
            if idx >= 4:
                break

            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            if ratio_col in ratio_results.columns:
                ratio_data = ratio_results[ratio_col].dropna()

                # Plot ratio using index positions (no weekend gaps)
                ratio_x_positions = range(len(ratio_data))
                ax.plot(ratio_x_positions, ratio_data.values, linewidth=2, label=ratio_col)

                # Add moving averages if available
                sma20_col = f'{ratio_col}_SMA20'
                sma50_col = f'{ratio_col}_SMA50'

                if sma20_col in ratio_results.columns:
                    sma20_data = ratio_results[sma20_col].dropna()
                    sma20_x_positions = range(len(sma20_data))
                    ax.plot(sma20_x_positions, sma20_data.values,
                           linewidth=1, alpha=0.7, label='SMA20')

                if sma50_col in ratio_results.columns:
                    sma50_data = ratio_results[sma50_col].dropna()
                    sma50_x_positions = range(len(sma50_data))
                    ax.plot(sma50_x_positions, sma50_data.values,
                           linewidth=1, alpha=0.7, label='SMA50')

                ax.set_title(title, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Apply trading days formatting
                format_trading_days_axis(ax, ratio_data.index, len(ratio_data))

            else:
                ax.text(0.5, 0.5, f'No Data for {ratio_col}', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Created ratio dashboard: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating ratio dashboard: {e}")
        plt.close('all')
        return ""


def create_breadth_dashboard(breadth_results: Dict[str, pd.Series], output_path: str) -> str:
    """
    Create dashboard for market breadth indicators.

    Args:
        breadth_results: Dict with breadth calculations
        output_path: Path to save dashboard

    Returns:
        Path to saved file
    """
    try:
        if not breadth_results:
            logger.warning("No breadth data for dashboard")
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Market Breadth Dashboard', fontsize=16, fontweight='bold')

        # Plot key breadth indicators
        breadth_plots = [
            ('AD_Ratio', 'Advance/Decline Ratio'),
            ('Advance_Pct', 'Advance Percentage'),
            ('Above_SMA50_Pct', '% Above SMA50'),
            ('McClellan_Osc', 'McClellan Oscillator')
        ]

        for idx, (indicator, title) in enumerate(breadth_plots):
            if idx >= 4:
                break

            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            if indicator in breadth_results:
                data = breadth_results[indicator].dropna()

                # Plot using index positions (no weekend gaps)
                data_x_positions = range(len(data))
                ax.plot(data_x_positions, data.values, linewidth=2, color='blue')

                # Add reference lines
                if 'ratio' in indicator.lower():
                    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Neutral')
                elif 'pct' in indicator.lower():
                    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
                elif 'oscillator' in indicator.lower():
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Apply trading days formatting
                format_trading_days_axis(ax, data.index, len(data))

            else:
                ax.text(0.5, 0.5, f'No Data for {indicator}', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Created breadth dashboard: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating breadth dashboard: {e}")
        plt.close('all')
        return ""


def generate_sr_dashboard(results: Dict[str, Any], output_dir: Path, panel_config: Dict, user_config = None) -> Dict[str, str]:
    """
    Generate complete SR dashboard with all charts.

    Args:
        results: SR analysis results
        output_dir: Output directory (will be organized into submodules)
        panel_config: Panel configuration

    Returns:
        Dict with chart file paths
    """
    # DEBUG: Function call path verification
    logger.info(f"🚀 FUNCTION CALL: generate_sr_dashboard()")
    logger.info(f"   results keys: {list(results.keys()) if results else 'None'}")
    logger.info(f"   output_dir: {output_dir}")
    logger.info(f"   panel_config keys: {list(panel_config.keys()) if panel_config else 'None'}")

    try:
        # Initialize SR output manager with organized submodule structure
        output_manager = get_sr_output_manager(str(output_dir))
        logger.info(f"📁 SR Output Manager initialized: {output_manager}")

        # Migrate any legacy files to new structure
        output_manager.migrate_legacy_files()

        chart_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Primary: Multi-panel chart from panel indicators (goes to 'panels' submodule)
        if 'panel_indicators' in results and panel_config:
            panels_dir = output_manager.get_submodule_dir('panels')
            multi_panel_path = panels_dir / f"sr_multi_panel_{timestamp}.png"
            chart_path = create_multi_panel_chart(
                results['panel_indicators'],
                str(multi_panel_path),
                "Sustainability Ratios Dashboard",
                user_config
            )
            if chart_path:
                chart_paths['multi_panel'] = chart_path
                logger.info("Generated panel-based dashboard")

        # Fallback: Only generate hardcoded dashboards if no panel config
        elif not panel_config:
            logger.info("No panel configuration - generating fallback dashboards")

            # Intermarket ratios dashboard (fallback only) - goes to 'ratios' submodule
            if 'intermarket_ratios' in results:
                ratios_dir = output_manager.get_submodule_dir('ratios')
                ratios_path = ratios_dir / f"sr_ratios_{timestamp}.png"
                chart_path = create_ratio_dashboard(
                    results['intermarket_ratios'],
                    str(ratios_path)
                )
                if chart_path:
                    chart_paths['ratios'] = chart_path

            # Market breadth dashboard (fallback only) - goes to 'breadth' submodule
            if 'market_breadth' in results:
                breadth_dir = output_manager.get_submodule_dir('breadth')
                breadth_path = breadth_dir / f"sr_breadth_{timestamp}.png"
                chart_path = create_breadth_dashboard(
                    results['market_breadth'],
                    str(breadth_path)
                )
                if chart_path:
                    chart_paths['breadth'] = chart_path

            # Summary overview (fallback only) - goes to 'overview' submodule
            overview_dir = output_manager.get_submodule_dir('overview')
            overview_path = overview_dir / f"sr_overview_{timestamp}.png"
            chart_path = create_sr_overview(results, str(overview_path))
            if chart_path:
                chart_paths['overview'] = chart_path

        logger.info(f"Generated {len(chart_paths)} SR dashboard charts")
        return chart_paths

    except Exception as e:
        logger.error(f"Error generating SR dashboard: {e}")
        return {}


def create_sr_overview(results: Dict[str, Any], output_path: str) -> str:
    """
    Create SR overview chart similar to market_timing_dashboard.py but with real data.

    Args:
        results: SR analysis results
        output_path: Output file path

    Returns:
        Path to saved file
    """
    try:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('off')

        # Title
        fig.suptitle('Sustainability Ratios Market Timing Overview',
                    fontsize=18, fontweight='bold', y=0.95)

        # Define sections and positions
        sections = ["Intermarket Ratios", "Market Breadth", "Panel Indicators"]
        y_positions = [0.8, 0.5, 0.2]
        section_height = 0.2

        for sec_idx, (section, y_pos) in enumerate(zip(sections, y_positions)):
            # Section header
            ax.text(0.05, y_pos + 0.15, section, fontsize=16, fontweight='bold', color='black')

            # Section content
            if section == "Intermarket Ratios" and 'intermarket_ratios' in results:
                plot_ratio_summary(ax, results['intermarket_ratios'], y_pos)
            elif section == "Market Breadth" and 'market_breadth' in results:
                plot_breadth_summary(ax, results['market_breadth'], y_pos)
            elif section == "Panel Indicators" and 'panel_indicators' in results:
                plot_panel_summary(ax, results['panel_indicators'], y_pos)
            else:
                ax.text(0.1, y_pos + 0.05, "No data available", fontsize=12, style='italic', alpha=0.7)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        ax.text(0.95, 0.02, f"Generated: {timestamp}", ha='right', va='bottom',
               fontsize=10, alpha=0.7)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Created SR overview: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating SR overview: {e}")
        plt.close('all')
        return ""


def plot_ratio_summary(ax, ratio_data: pd.DataFrame, y_pos: float):
    """Plot summary of ratio signals."""
    try:
        from .sr_ratios import get_ratio_signals
        signals = get_ratio_signals(ratio_data)

        x_start = 0.1
        x_spacing = 0.15

        for idx, (ratio, signal) in enumerate(signals.items()):
            if idx >= 5:  # Limit to 5 ratios
                break

            x = x_start + idx * x_spacing

            # Color based on signal
            color = {'Strong Bullish': 'darkgreen', 'Bullish': 'green',
                    'Neutral': 'orange', 'Bearish': 'red', 'Strong Bearish': 'darkred'}.get(signal, 'gray')

            # Draw box
            rect = plt.Rectangle((x, y_pos), 0.12, 0.08, color=color, alpha=0.3)
            ax.add_patch(rect)

            # Add text
            ax.text(x + 0.01, y_pos + 0.05, ratio, fontsize=10, fontweight='bold')
            ax.text(x + 0.01, y_pos + 0.02, signal, fontsize=9)

    except Exception as e:
        logger.warning(f"Error plotting ratio summary: {e}")


def plot_breadth_summary(ax, breadth_data: Dict, y_pos: float):
    """Plot summary of breadth indicators."""
    try:
        breadth_indicators = ['AD_Ratio', 'Advance_Pct', 'Above_SMA50_Pct']
        x_start = 0.1
        x_spacing = 0.15

        for idx, indicator in enumerate(breadth_indicators):
            if idx >= 5 or indicator not in breadth_data:
                continue

            x = x_start + idx * x_spacing
            data = breadth_data[indicator]

            if not data.empty:
                latest_value = data.iloc[-1]

                # Determine signal color
                if 'ratio' in indicator.lower():
                    color = 'green' if latest_value > 1.5 else 'red' if latest_value < 0.7 else 'orange'
                elif 'pct' in indicator.lower():
                    color = 'green' if latest_value > 60 else 'red' if latest_value < 40 else 'orange'
                else:
                    color = 'blue'

                # Draw box
                rect = plt.Rectangle((x, y_pos), 0.12, 0.08, color=color, alpha=0.3)
                ax.add_patch(rect)

                # Add text
                ax.text(x + 0.01, y_pos + 0.05, indicator.replace('_', ' '), fontsize=10, fontweight='bold')
                ax.text(x + 0.01, y_pos + 0.02, f"{latest_value:.2f}", fontsize=9)

    except Exception as e:
        logger.warning(f"Error plotting breadth summary: {e}")


def plot_panel_summary(ax, panel_data: Dict, y_pos: float):
    """Plot summary of panel indicators."""
    try:
        x_start = 0.1
        x_spacing = 0.15

        panel_count = 0
        for panel_name, panel_info in panel_data.items():
            if panel_count >= 5:
                break

            x = x_start + panel_count * x_spacing

            # Get panel status
            has_data = bool(panel_info.get('result'))
            indicator = panel_info.get('indicator', 'None')
            data_source = panel_info.get('data_source', 'Unknown')

            color = 'green' if has_data else 'red'

            # Draw box
            rect = plt.Rectangle((x, y_pos), 0.12, 0.08, color=color, alpha=0.3)
            ax.add_patch(rect)

            # Add text
            ax.text(x + 0.01, y_pos + 0.05, data_source[:8], fontsize=10, fontweight='bold')
            indicator_name = indicator.split('(')[0] if indicator else 'None'
            ax.text(x + 0.01, y_pos + 0.02, indicator_name[:8], fontsize=9)

            panel_count += 1

    except Exception as e:
        logger.warning(f"Error plotting panel summary: {e}")