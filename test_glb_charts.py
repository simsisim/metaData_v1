#!/usr/bin/env python3
"""
Standalone test to verify GLB lines appear on charts
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def test_glb_chart_generation():
    """Test GLB line generation using individual files"""
    
    ticker = 'TSLA'
    print(f"Testing GLB chart generation for {ticker}")
    
    # Load price data
    price_file = Path(f'data/market_data/daily/{ticker}.csv')
    if not price_file.exists():
        print(f"Price file not found: {price_file}")
        return
        
    df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    df = df.tail(252)  # Last year for clarity
    
    # Force timezone-naive
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    
    print(f"Loaded {len(df)} price bars for {ticker}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Load GLB individual file
    glb_file = Path(f'results/screeners/drwish/individual/{ticker}_glb_results.csv')
    if not glb_file.exists():
        print(f"GLB file not found: {glb_file}")
        return
        
    glb_df = pd.read_csv(glb_file)
    print(f"Found {len(glb_df)} GLB levels")
    print("GLB levels:", glb_df['level'].tolist())
    
    # Create chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot price
    ax.plot(df.index, df['Close'], color='black', linewidth=1, label='Close Price')
    
    # Plot each GLB level
    glb_lines_plotted = 0
    for idx, row in glb_df.iterrows():
        glb_level = float(row['level'])
        detection_date = pd.to_datetime(row['detection_date'])
        breakout_date = pd.to_datetime(row['breakout_date']) if pd.notna(row['breakout_date']) else df.index[-1]
        is_broken = bool(row['is_broken'])
        
        # Ensure timezone-naive
        if hasattr(detection_date, 'tz_convert') and detection_date.tz is not None:
            detection_date = detection_date.tz_convert(None)
        if breakout_date and hasattr(breakout_date, 'tz_convert') and breakout_date.tz is not None:
            breakout_date = breakout_date.tz_convert(None)
            
        # Choose colors and style
        if is_broken:
            color = 'gray'
            style = '--'
            width = 1
            alpha = 0.6
        else:
            color = 'green'
            style = '-'
            width = 3
            alpha = 0.9
            
        # Plot GLB line
        ax.plot([detection_date, breakout_date], [glb_level, glb_level],
               color=color, linestyle=style, linewidth=width, alpha=alpha,
               label=f'GLB {glb_level:.2f}' if idx < 3 else '')
               
        # Add level text
        mid_date = detection_date + (breakout_date - detection_date) / 2
        ax.text(mid_date, glb_level, f'{glb_level:.2f}',
               verticalalignment='bottom', horizontalalignment='center',
               fontsize=9, color=color, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
               
        glb_lines_plotted += 1
        
        print(f"GLB {idx+1}: Level={glb_level:.2f}, Detection={detection_date.date()}, Breakout={breakout_date.date() if breakout_date else 'None'}, Broken={is_broken}")
    
    # Format chart
    ax.set_title(f'{ticker} - GLB Test Chart - {datetime.now().strftime("%Y-%m-%d")}', fontsize=14, weight='bold')
    ax.set_ylabel('Price ($)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save test chart
    test_file = Path('results/screeners/drwish/charts/TEST_GLB_VERIFICATION.png')
    fig.savefig(test_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nTest complete!")
    print(f"GLB lines plotted: {glb_lines_plotted}")
    print(f"Test chart saved: {test_file}")
    
    if glb_lines_plotted > 0:
        print("✅ GLB lines should now be visible on charts!")
    else:
        print("❌ No GLB lines were plotted")

if __name__ == "__main__":
    test_glb_chart_generation()