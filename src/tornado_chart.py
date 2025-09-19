"""
Tornado Chart Generator
=======================

Creates tornado charts for comparing multiple metrics across different assets.
Tornado charts are ideal for sensitivity analysis and comparative visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import logging
import re
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


def extract_data_date_from_filename(filename: str) -> str:
    """
    Extract data date from CSV filename pattern like 'index_counts_daily_20250829.csv'
    
    Args:
        filename: Path or filename string containing date pattern
        
    Returns:
        str: Date string in YYYYMMDD format, or current date if not found
    """
    from datetime import datetime
    
    # Convert Path to string if needed
    filename_str = str(filename)
    
    # Look for date pattern YYYYMMDD in filename
    date_pattern = r'(\d{8})'
    match = re.search(date_pattern, filename_str)
    
    if match:
        return match.group(1)
    else:
        # Fallback to current date if no date found in filename
        return datetime.now().strftime("%Y%m%d")


def prepare_tornado_data(data: pd.DataFrame, 
                        columns: List[str], 
                        ticker_column: str = 'ticker') -> pd.DataFrame:
    """
    Prepare data for tornado chart visualization.
    
    Args:
        data: DataFrame with metrics for each ticker
        columns: List of column names to include in tornado chart
        ticker_column: Name of the column containing ticker symbols
        
    Returns:
        DataFrame prepared for tornado chart
    """
    if data.empty or ticker_column not in data.columns:
        return pd.DataFrame()
    
    # Filter to only requested columns
    available_columns = [col for col in columns if col in data.columns]
    if not available_columns:
        logger.warning(f"None of the requested columns {columns} found in data")
        return pd.DataFrame()
    
    # Select data
    tornado_data = data[[ticker_column] + available_columns].copy()
    
    # Remove rows with all NaN values
    tornado_data = tornado_data.dropna(how='all', subset=available_columns)
    
    # Fill remaining NaN with 0 for visualization
    tornado_data[available_columns] = tornado_data[available_columns].fillna(0)
    
    return tornado_data


def create_tornado_chart(data: pd.DataFrame, 
                        columns: List[str],
                        title: str = "Tornado Chart",
                        output_path: Optional[Path] = None,
                        figsize: Optional[Tuple[int, int]] = None,
                        ticker_column: str = 'ticker',
                        data_date: Optional[str] = None,
                        source_filename: Optional[str] = None) -> str:
    """
    Create a tornado chart comparing multiple metrics across tickers.
    
    Args:
        data: DataFrame with metrics for each ticker
        columns: List of column names to visualize
        title: Chart title
        output_path: Path to save the chart image
        figsize: Figure size (width, height)
        ticker_column: Name of the column containing ticker symbols
        data_date: Optional date string (YYYYMMDD) for filename, uses current date if None
        source_filename: Optional source CSV filename to base tornado filename on
        
    Returns:
        str: Path to saved chart image
    """
    # Prepare data
    tornado_data = prepare_tornado_data(data, columns, ticker_column)
    
    if tornado_data.empty:
        logger.warning("No data available for tornado chart")
        return ""
    
    # Get tickers and metrics
    tickers = tornado_data[ticker_column].tolist()
    metrics = [col for col in columns if col in tornado_data.columns]
    
    # Auto-adjust figure size based on number of metrics
    if figsize is None:
        width = max(16, len(tickers) * 3)  # Wider for more tickers
        height = max(12, len(metrics) * 0.4)  # Taller for more metrics
        figsize = (width, height)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
    
    # Calculate bar positions
    y_pos = np.arange(len(metrics))
    bar_height = 0.8 / len(tickers) if len(tickers) > 0 else 0.8
    
    # Create bars for each ticker
    for i, ticker in enumerate(tickers):
        values = []
        for metric in metrics:
            val = tornado_data[tornado_data[ticker_column] == ticker][metric].iloc[0]
            values.append(val)
        
        # Use values as-is (no complex normalization)
        # Simple scaling for percentage columns
        display_values = []
        for j, val in enumerate(values):
            if 'pct_change' in metrics[j] or 'pct' in metrics[j]:
                # Already in percentage or ratio form
                display_values.append(val)
            else:
                # Use raw values
                display_values.append(val)
        
        # Create horizontal bars
        y_positions = y_pos + (i - len(tickers)/2 + 0.5) * bar_height
        bars = ax.barh(y_positions, display_values, bar_height, 
                      label=ticker, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.3)
        
        # Add value labels on bars (only for reasonable bar widths)
        for j, (bar, val) in enumerate(zip(bars, display_values)):
            width = bar.get_width()
            if abs(width) > 0.001:  # Only show label if bar is visible
                label_x = width + (max(display_values) * 0.02 if width > 0 else min(display_values) * 0.02)
                fontsize = max(6, min(8, 100 // len(metrics)))  # Adaptive font size
                ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                       ha='left' if width > 0 else 'right', va='center', fontsize=fontsize)
    
    # Customize chart
    ax.set_yticks(y_pos)
    
    # Create cleaner metric labels (preserve original column names but make readable)
    metric_labels = []
    for col in metrics:
        # Keep original column name but make it more readable
        label = col.replace('_', ' ')
        # Don't title case - keep original casing for technical terms
        metric_labels.append(label)
    
    ax.set_yticklabels(metric_labels, fontsize=max(6, min(10, 120 // len(metrics))))
    ax.invert_yaxis()  # Top to bottom (maintains original order)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title(f'{title} - All {len(metrics)} Metrics', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Grid for better readability
    ax.grid(True, alpha=0.3, axis='x')
    
    # Tight layout
    plt.tight_layout()
    
    # Save chart
    if output_path:
        from datetime import datetime
        
        if source_filename:
            # Use source filename pattern: original_name_tornado.png
            source_path = Path(source_filename)
            base_name = source_path.stem  # Gets filename without extension
            chart_file = output_path / f'{base_name}_tornado.png'
        else:
            # Fallback to old naming convention
            timestamp = data_date if data_date else datetime.now().strftime("%Y%m%d")
            chart_file = output_path / f'index_tornado_daily_{timestamp}.png'
            
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Tornado chart saved: {chart_file}")
        return str(chart_file)
    else:
        plt.show()
        return ""


def create_comparative_tornado_chart(data: pd.DataFrame,
                                   columns: List[str],
                                   title: str = "Comparative Analysis",
                                   output_path: Optional[Path] = None,
                                   ticker_column: str = 'ticker') -> str:
    """
    Create a tornado chart with positive/negative comparison structure.
    
    Args:
        data: DataFrame with metrics for each ticker
        columns: List of column names to visualize
        title: Chart title
        output_path: Path to save the chart image
        ticker_column: Name of the column containing ticker symbols
        
    Returns:
        str: Path to saved chart image
    """
    # Prepare data
    tornado_data = prepare_tornado_data(data, columns, ticker_column)
    
    if tornado_data.empty or len(tornado_data) < 2:
        logger.warning("Need at least 2 tickers for comparative tornado chart")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get metrics
    metrics = [col for col in columns if col in tornado_data.columns]
    tickers = tornado_data[ticker_column].tolist()
    
    # Calculate differences from mean for tornado effect
    tornado_values = []
    metric_labels = []
    
    for metric in metrics:
        values = tornado_data[metric].values
        mean_val = np.mean(values)
        
        for i, ticker in enumerate(tickers):
            diff = values[i] - mean_val
            tornado_values.append({
                'metric': metric.replace('_', ' ').title(),
                'ticker': ticker,
                'value': diff,
                'abs_value': abs(diff)
            })
    
    # Convert to DataFrame and sort by absolute value
    tornado_df = pd.DataFrame(tornado_values)
    tornado_df = tornado_df.sort_values(['metric', 'abs_value'], ascending=[True, False])
    
    # Create the tornado chart
    y_pos = 0
    metric_positions = {}
    colors = {'positive': '#2E8B57', 'negative': '#DC143C'}  # Green for positive, Red for negative
    
    current_metric = None
    for _, row in tornado_df.iterrows():
        if row['metric'] != current_metric:
            current_metric = row['metric']
            metric_positions[current_metric] = y_pos
            y_pos += 1
        
        # Determine color based on positive/negative
        color = colors['positive'] if row['value'] >= 0 else colors['negative']
        
        # Create horizontal bar
        bar = ax.barh(metric_positions[current_metric], row['value'], 0.6,
                     color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add ticker label
        label_x = row['value'] + (0.1 if row['value'] >= 0 else -0.1)
        ax.text(label_x, metric_positions[current_metric], row['ticker'],
               ha='left' if row['value'] >= 0 else 'right', va='center', fontweight='bold')
    
    # Customize chart
    ax.set_yticks(list(metric_positions.values()))
    ax.set_yticklabels(list(metric_positions.keys()))
    ax.invert_yaxis()
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Labels and title
    ax.set_xlabel('Deviation from Mean')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    positive_patch = patches.Rectangle((0, 0), 1, 1, facecolor=colors['positive'], alpha=0.7, label='Above Average')
    negative_patch = patches.Rectangle((0, 0), 1, 1, facecolor=colors['negative'], alpha=0.7, label='Below Average')
    ax.legend(handles=[positive_patch, negative_patch], loc='upper right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save chart
    if output_path:
        chart_file = output_path / f'comparative_tornado_chart.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparative tornado chart saved: {chart_file}")
        return str(chart_file)
    else:
        plt.show()
        return ""


def generate_tornado_charts(data: pd.DataFrame, 
                          user_config, 
                          output_path: Path,
                          ticker_column: str = 'ticker',
                          source_filename: Optional[str] = None) -> List[str]:
    """
    Generate tornado charts based on user configuration.
    
    Args:
        data: DataFrame with metrics
        user_config: User configuration with tornado chart settings
        output_path: Path to save charts
        ticker_column: Name of ticker column
        source_filename: Optional source filename to extract data date from
        
    Returns:
        List of generated chart file paths
    """
    # NOTE: index_overview module removed from BASIC calculations
    # Tornado charts now use fallback data only
    if not user_config:
        return []
    
    # Get ALL available numeric columns in their original order (exclude ticker, date, and text columns)
    exclude_columns = ['ticker', 'date', 'timeframe', 'daily_candle_type']  # Non-numeric columns to exclude
    
    # Get all columns except excluded ones
    all_columns = [col for col in data.columns if col not in exclude_columns]
    
    # Filter to numeric columns only
    numeric_columns = []
    for col in all_columns:
        if col in data.columns:
            try:
                # Try to convert to numeric, if it works, include it
                pd.to_numeric(data[col], errors='coerce')
                numeric_columns.append(col)
            except:
                continue
    
    columns = numeric_columns
    
    print(f"📊 Tornado chart will display ALL {len(columns)} numeric columns in original order")
    print(f"  • First 10 columns: {columns[:10]}")
    if len(columns) > 10:
        print(f"  • Last 10 columns: {columns[-10:]}")
    
    # Extract data date from source filename if provided
    data_date = None
    if source_filename:
        data_date = extract_data_date_from_filename(source_filename)
    
    generated_files = []
    
    try:
        # Standard tornado chart
        chart_file = create_tornado_chart(
            data=data,
            columns=columns,
            title="Index Overview Metrics",
            output_path=output_path,
            ticker_column=ticker_column,
            data_date=data_date,
            source_filename=source_filename
        )
        if chart_file:
            generated_files.append(chart_file)
        
        # Comparative tornado chart (if we have multiple tickers)
        if len(data) > 1:
            comp_chart_file = create_comparative_tornado_chart(
                data=data,
                columns=columns,
                title="Comparative Index Analysis",
                output_path=output_path,
                ticker_column=ticker_column
            )
            if comp_chart_file:
                generated_files.append(comp_chart_file)
        
        if generated_files:
            print(f"📊 Generated {len(generated_files)} tornado chart(s)")
            for file in generated_files:
                print(f"  • {Path(file).name}")
        
    except Exception as e:
        logger.error(f"Error generating tornado charts: {e}")
        print(f"❌ Error generating tornado charts: {e}")
    
    return generated_files