"""
Market Breadth Tornado Chart Generator
=====================================

Creates tornado charts for market breadth analysis using time-series data from
market_breadth CSV files. Adapted from the main tornado_chart module to work
specifically with breadth metrics and temporal data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# Key market breadth metrics for tornado visualization
BREADTH_TORNADO_METRICS = [
    'daily_mb_advance_pct',
    'daily_mb_decline_pct',
    'daily_mb_ad_ratio',
    'daily_mb_net_advances',
    'daily_mb_pct_above_ma_20',
    'daily_mb_pct_above_ma_50',
    'daily_mb_pct_above_ma_200',
    'daily_mb_long_new_highs_pct',
    'daily_mb_medium_new_highs_pct',
    'daily_mb_short_new_highs_pct',
    'daily_mb_bullish_momentum_5d',
    'daily_mb_bullish_momentum_10d',
    'daily_mb_bullish_momentum_20d',
    'daily_mb_net_momentum_5d',
    'daily_mb_net_momentum_10d',
    'daily_mb_net_momentum_20d',
    'daily_mb_overall_breadth_score',
    'daily_mb_total_bullish_signals',
    'daily_mb_total_bearish_signals',
    'daily_mb_net_signal_score'
]


def prepare_breadth_tornado_data(data: pd.DataFrame,
                                columns: List[str],
                                date_column: str = 'date',
                                days_lookback: int = 30) -> pd.DataFrame:
    """
    Prepare market breadth data for tornado chart visualization.

    Args:
        data: DataFrame with breadth metrics over time
        columns: List of column names to include in tornado chart
        date_column: Name of the date column
        days_lookback: Number of recent days to include

    Returns:
        DataFrame prepared for tornado chart
    """
    if data.empty or date_column not in data.columns:
        return pd.DataFrame()

    # Filter to only requested columns that exist
    available_columns = [col for col in columns if col in data.columns]
    if not available_columns:
        logger.warning(f"None of the requested columns {columns} found in data")
        return pd.DataFrame()

    # Sort by date and take last N days
    data_sorted = data.sort_values(date_column)
    recent_data = data_sorted.tail(days_lookback)

    # Select data
    tornado_data = recent_data[[date_column] + available_columns].copy()

    # Remove rows with all NaN values
    tornado_data = tornado_data.dropna(how='all', subset=available_columns)

    # Fill remaining NaN with 0 for visualization
    tornado_data[available_columns] = tornado_data[available_columns].fillna(0)

    # Format date column for display
    if tornado_data[date_column].dtype == 'object':
        try:
            tornado_data[date_column] = pd.to_datetime(tornado_data[date_column])
        except:
            pass

    # Create display dates (MM-DD format for readability)
    tornado_data['display_date'] = tornado_data[date_column].dt.strftime('%m-%d')

    return tornado_data


def create_breadth_tornado_chart(data: pd.DataFrame,
                                columns: List[str],
                                title: str = "Market Breadth Tornado Chart",
                                output_path: Optional[Path] = None,
                                figsize: Optional[Tuple[int, int]] = None,
                                date_column: str = 'display_date',
                                universe: str = '',
                                days_lookback: int = 30) -> str:
    """
    Create a tornado chart for market breadth evolution over time.

    Args:
        data: DataFrame with breadth metrics
        columns: List of column names to visualize
        title: Chart title
        output_path: Path to save the chart image
        figsize: Figure size (width, height)
        date_column: Name of the date column for labels
        universe: Universe name for filename
        days_lookback: Number of days included

    Returns:
        str: Path to saved chart image
    """
    # Prepare data
    tornado_data = prepare_breadth_tornado_data(data, columns, 'date', days_lookback)

    if tornado_data.empty:
        logger.warning("No data available for breadth tornado chart")
        return ""

    # Get dates and metrics
    dates = tornado_data[date_column].tolist()
    metrics = [col for col in columns if col in tornado_data.columns]

    # Auto-adjust figure size based on number of metrics and dates
    if figsize is None:
        width = max(16, len(dates) * 0.8)  # Wider for more dates
        height = max(12, len(metrics) * 0.5)  # Taller for more metrics
        figsize = (width, height)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette - use breadth-specific colors
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(dates)))  # Red to Green gradient

    # Calculate bar positions
    y_pos = np.arange(len(metrics))
    bar_height = 0.8 / len(dates) if len(dates) > 0 else 0.8

    # Create bars for each date
    for i, date_val in enumerate(dates):
        values = []
        for metric in metrics:
            val = tornado_data[tornado_data[date_column] == date_val][metric].iloc[0]
            values.append(val)

        # Create horizontal bars
        y_positions = y_pos + (i - len(dates)/2 + 0.5) * bar_height
        bars = ax.barh(y_positions, values, bar_height,
                      label=date_val, color=colors[i], alpha=0.7,
                      edgecolor='black', linewidth=0.2)

        # Add value labels on bars (only for significant values)
        for j, (bar, val) in enumerate(zip(bars, values)):
            width = bar.get_width()
            if abs(width) > 0.001:  # Only show label if bar is visible
                label_x = width + (max(values) * 0.01 if width > 0 else min(values) * 0.01)
                fontsize = max(5, min(7, 80 // len(metrics)))  # Adaptive font size
                ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                       ha='left' if width > 0 else 'right', va='center',
                       fontsize=fontsize, alpha=0.8)

    # Customize chart
    ax.set_yticks(y_pos)

    # Create cleaner metric labels
    metric_labels = []
    for col in metrics:
        # Remove 'daily_mb_' prefix and replace underscores
        label = col.replace('daily_mb_', '').replace('_', ' ')
        # Capitalize appropriately
        if 'pct' in label:
            label = label.replace('pct', '%')
        metric_labels.append(label.title())

    ax.set_yticklabels(metric_labels, fontsize=max(7, min(9, 100 // len(metrics))))
    ax.invert_yaxis()  # Top to bottom

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Labels and title
    ax.set_xlabel('Breadth Metric Value', fontsize=11)
    full_title = f'{title}\n{universe} - Last {days_lookback} Trading Days'
    ax.set_title(full_title, fontsize=13, fontweight='bold', pad=20)

    # Legend - limit to avoid overcrowding
    if len(dates) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    else:
        # For many dates, show only every nth date in legend
        step = len(dates) // 10 + 1
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::step], labels[::step], bbox_to_anchor=(1.05, 1),
                 loc='upper left', fontsize=8, ncol=1)

    # Grid for better readability
    ax.grid(True, alpha=0.3, axis='x')

    # Tight layout
    plt.tight_layout()

    # Save chart
    if output_path:
        from datetime import datetime

        # Generate filename with universe and date
        data_date = datetime.now().strftime("%Y%m%d")
        if not data.empty and 'date' in data.columns:
            try:
                latest_date = pd.to_datetime(data['date']).max()
                data_date = latest_date.strftime("%Y%m%d")
            except:
                pass

        universe_clean = universe.replace(' ', '_') if universe else 'breadth'
        chart_file = output_path / f'market_breadth_tornado_{universe_clean}_daily_{data_date}.png'

        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Breadth tornado chart saved: {chart_file}")
        return str(chart_file)
    else:
        plt.show()
        return ""


def generate_breadth_tornado_charts(timeframe: str,
                                  data_date: str,
                                  user_config,
                                  config) -> List[str]:
    """
    Generate tornado charts for market breadth analysis.

    Args:
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        data_date: Data date string (YYYYMMDD)
        user_config: User configuration
        config: System configuration

    Returns:
        List of generated chart file paths
    """
    if not getattr(user_config, 'market_breadth_tornado_chart', False):
        logger.debug("Market breadth tornado charts disabled in configuration")
        return []

    if timeframe != 'daily':
        logger.debug(f"Breadth tornado charts only supported for daily timeframe, got: {timeframe}")
        return []

    generated_files = []

    try:
        # Get market breadth output directory
        breadth_output_dir = Path(getattr(user_config, 'market_breadth_output_dir', 'results/market_breadth'))

        # Get universes from user config
        breadth_config = getattr(user_config, 'market_breadth_universe', ['SP500'])

        if isinstance(breadth_config, dict):
            # Extract universes from parsed dict structure
            universes = breadth_config.get('universes', ['SP500'])
        elif isinstance(breadth_config, str):
            # Fallback for string format
            universes = [u.strip() for u in breadth_config.split(';') if u.strip()]
        else:
            # Fallback for list format
            universes = breadth_config if isinstance(breadth_config, list) else ['SP500']

        # Process each universe
        for universe in universes:
            try:
                # Construct breadth CSV filename
                ticker_choice = getattr(config, 'ticker_choice', '2-5')
                breadth_file = breadth_output_dir / f'market_breadth_{universe}_{ticker_choice}_{timeframe}_{data_date}.csv'

                if not breadth_file.exists():
                    logger.warning(f"Breadth file not found: {breadth_file}")
                    continue

                # Load breadth data
                breadth_df = pd.read_csv(breadth_file)
                if breadth_df.empty:
                    logger.warning(f"Empty breadth data in {breadth_file}")
                    continue

                # Filter to available breadth metrics
                available_metrics = [col for col in BREADTH_TORNADO_METRICS if col in breadth_df.columns]
                if not available_metrics:
                    logger.warning(f"No tornado metrics found in {breadth_file}")
                    continue

                # Get configurable days lookback
                days_lookback = getattr(user_config, 'market_breadth_tornado_chart_display_units_time', 30)

                print(f"📊 Generating breadth tornado chart for {universe}...")
                print(f"  • Using {len(available_metrics)} breadth metrics")
                print(f"  • Data points: {len(breadth_df)} trading days")
                print(f"  • Chart display: Last {days_lookback} trading days")

                # Generate tornado chart
                chart_file = create_breadth_tornado_chart(
                    data=breadth_df,
                    columns=available_metrics,
                    title=f"Market Breadth Evolution",
                    output_path=breadth_output_dir,
                    universe=universe,
                    days_lookback=days_lookback
                )

                if chart_file:
                    generated_files.append(chart_file)
                    print(f"  ✅ Generated: {Path(chart_file).name}")

            except Exception as e:
                logger.error(f"Error generating tornado chart for {universe}: {e}")
                print(f"  ❌ Failed to generate tornado chart for {universe}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in breadth tornado chart generation: {e}")
        print(f"❌ Breadth tornado chart generation failed: {e}")

    return generated_files