"""
Tornado Chart Generator for Index Overview Analysis
==================================================

Creates tornado charts from index percentage analysis data showing
market breadth metrics in a horizontal bar chart format.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys


def load_latest_analysis_data():
    """Load the most recent index percentage analysis CSV file."""
    results_dir = Path(__file__).parent / 'results' / 'overview'
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    # Find latest analysis file
    pattern = "*Percentage_Analysis*.csv"
    analysis_files = list(results_dir.glob(pattern))
    
    if not analysis_files:
        print(f"❌ No index analysis files found in {results_dir}")
        return None
    
    # Get most recent file
    latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading latest analysis: {latest_file.name}")
    
    try:
        df = pd.read_csv(latest_file)
        print(f"✅ Loaded {len(df)} metrics from analysis file")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


# Load data from latest CSV file
df = load_latest_analysis_data()
if df is None:
    # Fallback to hardcoded data if no file found
    print("⚠️ Using fallback data - run main pipeline to generate fresh data")
    data = {
        'Metric': [
            "Day Change (%)", "Month Change (%)", "Quarter Change (%)", "Half Year Change (%)",
            "Price vs SMA 10", "Price vs EMA 20", "Price vs SMA 50",
            "SMA 10 vs SMA 20", "SMA 20 vs SMA 50", "SMA 50 vs SMA 200",
            "Trend Strength", "Perfect Bull Alignment",
            "20-Day Position", "20-Day Low Position", "RSI Momentum"
        ],
        'SP500_Above%': [0.00,0.00,0.00,0.00,71.74,78.26,80.43,76.09,69.57,52.17,43.48,36.96,19.57,0.00,34.78],
        'SP500_Below%': [0.00,0.00,0.00,0.00,28.26,21.74,19.57,23.91,30.43,2.17,15.22,17.39,0.00,0.00,2.17],
        'NASDAQ100_Above%': [0.00,0.00,0.00,0.00,68.18,81.82,81.82,86.36,68.18,0.00,45.45,0.00,13.64,0.00,31.82],
        'NASDAQ100_Below%': [0.00,0.00,0.00,0.00,31.82,18.18,18.18,13.64,31.82,0.00,22.73,0.00,0.00,0.00,4.55]
    }
    df = pd.DataFrame(data)

def create_tornado_chart_from_data(df):
    """Create tornado chart from loaded DataFrame."""
    # Identify available indexes
    available_indexes = []
    for col in df.columns:
        if col.endswith('_Above%'):
            index_name = col.replace('_Above%', '')
            if f'{index_name}_Below%' in df.columns:
                available_indexes.append(index_name)
    
    if not available_indexes:
        print("❌ No valid index data found for tornado chart")
        return None
    
    print(f"📊 Creating tornado chart for indexes: {', '.join(available_indexes)}")
    
    # Filter out rows with all zero values
    mask = pd.Series([False] * len(df))
    for index_name in available_indexes:
        above_col = f'{index_name}_Above%'
        below_col = f'{index_name}_Below%'
        mask |= (df[above_col] > 0) | (df[below_col] > 0)
    
    chart_df = df[mask].copy()
    
    if chart_df.empty:
        print("❌ No non-zero data for tornado chart")
        return None
    
    metrics = chart_df['Metric'].tolist()
    y = np.arange(len(metrics))
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(14, max(8, len(metrics) * 0.6)))
    
    # Colors: colorblind-safe palette
    colors = {
        'SP500': {'above': "#377eb8", 'below': "#e41a1c"},      # Blue/Red
        'NASDAQ100': {'above': "#4daf4a", 'below': "#ff7f00"}, # Green/Orange
        'Russell1000': {'above': "#984ea3", 'below': "#ffff33"}, # Purple/Yellow
        'Russell2000': {'above': "#a65628", 'below': "#f781bf"}  # Brown/Pink
    }
    
    # Calculate bar positioning
    bar_height = 0.35 / len(available_indexes)  # Adjust based on number of indexes
    offsets = np.linspace(-0.15, 0.15, len(available_indexes))
    
    # Create bars for each index
    for i, index_name in enumerate(available_indexes):
        above_col = f'{index_name}_Above%'
        below_col = f'{index_name}_Below%'
        
        above_color = colors.get(index_name, {'above': '#1f77b4'})['above']
        below_color = colors.get(index_name, {'below': '#d62728'})['below']
        
        # Plot bars
        ax.barh(y + offsets[i], chart_df[above_col], height=bar_height, 
               color=above_color, label=f'{index_name} Above%', alpha=0.85, 
               edgecolor='black', linewidth=0.5)
        ax.barh(y + offsets[i], -chart_df[below_col], height=bar_height, 
               color=below_color, label=f'{index_name} Below%', alpha=0.85,
               edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for j in range(len(metrics)):
            above_val = chart_df[above_col].iloc[j]
            below_val = chart_df[below_col].iloc[j]
            
            if above_val > 2:  # Only show labels for values > 2%
                ax.text(above_val + 1, y[j] + offsets[i], f"{above_val:.0f}%", 
                       va='center', ha='left', fontsize=8, color=above_color, fontweight='bold')
            if below_val > 2:
                ax.text(-below_val - 1, y[j] + offsets[i], f"{below_val:.0f}%", 
                       va='center', ha='right', fontsize=8, color=below_color, fontweight='bold')
    
    # Format chart
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.legend(loc='lower right', fontsize=9, frameon=True, ncol=2)
    ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    
    # Dynamic title based on available indexes
    title = f"{' vs '.join(available_indexes)}: Market Breadth Analysis\n(Tornado Chart)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Invert y-axis to show most significant metrics at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = f"Tornado_Chart_{timestamp}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    print(f"✅ Tornado chart saved: {chart_file}")
    plt.show()
    
    return chart_file


# Generate the tornado chart
if df is not None:
    chart_file = create_tornado_chart_from_data(df)
else:
    print("❌ Unable to create tornado chart - no data available")

