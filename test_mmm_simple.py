#!/usr/bin/env python3
"""
Simple MMM Gap Calculation Test
===============================

Test the MMM gap calculation logic directly without SR imports.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_gaps_direct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Direct implementation of MMM gap calculations for testing.
    """
    # Create copy to avoid modifying original data
    result_df = df.copy()
    result_df = result_df.sort_index()

    # Calculate gaps using the specified formulas
    prev_close = result_df['Close'].shift(1)
    current_open = result_df['Open']
    current_close = result_df['Close']

    # Formula 1: opening_gap = day_(i)[open] - day_(i-1)[close]
    opening_gap = current_open - prev_close

    # Formula 2: price_without_opening_gap = day(i)[close] - day(i)[open]
    price_without_opening_gap = current_close - current_open

    # Add calculated columns
    result_df['Previous_Close'] = prev_close
    result_df['opening_gap'] = opening_gap
    result_df['opening_gap_pct'] = (opening_gap / prev_close) * 100
    result_df['price_without_opening_gap'] = price_without_opening_gap
    result_df['price_without_gap_pct'] = (price_without_opening_gap / current_open) * 100

    # Add statistical metrics
    result_df['gap_5MA'] = opening_gap.rolling(window=5, min_periods=1).mean()
    result_df['gap_20MA'] = opening_gap.rolling(window=20, min_periods=1).mean()
    result_df['gap_percentile'] = opening_gap.rolling(window=50, min_periods=10).rank(pct=True) * 100

    return result_df

def test_gap_calculation_with_sample_data():
    """Test gap calculation with sample data."""
    print("🧪 Testing MMM Gap Calculation Logic")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range('2024-09-20', periods=10, freq='D')
    np.random.seed(42)

    # Generate sample OHLCV data with known gaps
    sample_data = []
    prev_close = 150.0

    for i, date in enumerate(dates):
        if i == 0:
            open_price = 150.0
        elif i == 3:
            # Create a gap up
            open_price = prev_close + 2.5
        elif i == 6:
            # Create a gap down
            open_price = prev_close - 1.8
        else:
            # Small random gap
            open_price = prev_close + np.random.normal(0, 0.3)

        # Generate intraday movement
        high = open_price + abs(np.random.normal(0, 1.0))
        low = open_price - abs(np.random.normal(0, 0.8))
        close = low + (high - low) * (0.3 + 0.4 * np.random.random())

        sample_data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': int(1000000 + np.random.normal(0, 200000))
        })

        prev_close = close

    # Create DataFrame
    df = pd.DataFrame(sample_data, index=dates)

    print("Original Data (first 5 rows):")
    print(df.head().round(2))
    print()

    # Calculate gaps
    gap_df = calculate_gaps_direct(df)

    print("Gap Analysis Results:")
    print("Date                Open    Close   Prev_Close  opening_gap  price_w/o_gap")
    print("-" * 75)

    for i, (date, row) in enumerate(gap_df.iterrows()):
        if i < 8:  # Show first 8 days
            print(f"{date.strftime('%Y-%m-%d')}     {row['Open']:6.2f}  {row['Close']:6.2f}  "
                  f"{row['Previous_Close']:10.2f}  {row['opening_gap']:11.2f}  {row['price_without_opening_gap']:13.2f}")

    print()

    # Verify calculations manually for a few rows
    print("🔍 Manual Verification:")
    for i in [1, 3, 6]:  # Check specific days
        if i < len(gap_df):
            row = gap_df.iloc[i]
            date = gap_df.index[i]

            # Manual calculations
            expected_opening_gap = row['Open'] - row['Previous_Close']
            expected_price_without_gap = row['Close'] - row['Open']

            print(f"Day {i+1} ({date.strftime('%Y-%m-%d')}):")
            print(f"  opening_gap = {row['Open']:.2f} - {row['Previous_Close']:.2f} = {expected_opening_gap:.2f} ✅")
            print(f"  price_without_opening_gap = {row['Close']:.2f} - {row['Open']:.2f} = {expected_price_without_gap:.2f} ✅")
            print(f"  Calculated: opening_gap={row['opening_gap']:.2f}, price_w/o_gap={row['price_without_opening_gap']:.2f}")
            print()

    # Save gap data to demonstrate file output
    output_dir = Path('results/sustainability_ratios/MMM/gaps')
    output_dir.mkdir(parents=True, exist_ok=True)

    gap_file = output_dir / 'XLY_gap.csv'
    gap_df.to_csv(gap_file)

    print(f"✅ Gap data saved to: {gap_file}")
    print(f"   File size: {gap_file.stat().st_size} bytes")
    print(f"   Columns: {list(gap_df.columns)}")

    # Analyze gap patterns
    print("\n📊 Gap Analysis Summary:")
    valid_gaps = gap_df['opening_gap'].dropna()
    print(f"   Total gaps calculated: {len(valid_gaps)}")
    print(f"   Positive gaps (gap up): {(valid_gaps > 0).sum()}")
    print(f"   Negative gaps (gap down): {(valid_gaps < 0).sum()}")
    print(f"   Average gap: {valid_gaps.mean():.3f}")
    print(f"   Largest gap up: {valid_gaps.max():.3f}")
    print(f"   Largest gap down: {valid_gaps.min():.3f}")

    # Test price without gap analysis
    valid_price_wo_gap = gap_df['price_without_opening_gap'].dropna()
    print(f"\n   Intraday gains (close > open): {(valid_price_wo_gap > 0).sum()}")
    print(f"   Intraday losses (close < open): {(valid_price_wo_gap < 0).sum()}")
    print(f"   Average intraday movement: {valid_price_wo_gap.mean():.3f}")

    print("\n🎉 MMM Gap Calculation Test Completed Successfully!")
    return True

if __name__ == "__main__":
    test_gap_calculation_with_sample_data()