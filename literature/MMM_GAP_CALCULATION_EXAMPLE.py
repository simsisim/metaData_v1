#!/usr/bin/env python3
"""
MMM Gap Calculation Example
===========================

Demonstrates the gap calculation logic for the MMM submodule.

Key Formulas:
1. opening_gap = day_(i)[open] - day_(i-1)[close]
2. price_without_opening_gap = day(i)[close] - day(i)[open]

This example shows how to calculate these metrics from OHLCV data.
"""

import pandas as pd
import numpy as np

def calculate_mmm_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MMM gap analysis metrics.

    Args:
        df: DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume)

    Returns:
        DataFrame with gap analysis columns added
    """
    # Ensure data is sorted by date
    df = df.copy().sort_index()

    # Shift previous close by 1 day
    prev_close = df['Close'].shift(1)
    current_open = df['Open']
    current_close = df['Close']

    # Calculate opening gap: day_(i)[open] - day_(i-1)[close]
    opening_gap = current_open - prev_close
    opening_gap_pct = (opening_gap / prev_close) * 100

    # Calculate price without opening gap: day(i)[close] - day(i)[open]
    price_without_opening_gap = current_close - current_open
    price_without_gap_pct = (price_without_opening_gap / current_open) * 100

    # Create result DataFrame
    result_df = df.copy()
    result_df['Previous_Close'] = prev_close
    result_df['opening_gap'] = opening_gap
    result_df['opening_gap_pct'] = opening_gap_pct
    result_df['price_without_opening_gap'] = price_without_opening_gap
    result_df['price_without_gap_pct'] = price_without_gap_pct

    # Add statistical metrics
    result_df['gap_5MA'] = opening_gap.rolling(window=5).mean()
    result_df['gap_20MA'] = opening_gap.rolling(window=20).mean()
    result_df['gap_percentile'] = opening_gap.rolling(window=50).rank(pct=True) * 100

    return result_df

# Example usage with sample data
if __name__ == "__main__":
    # Sample OHLCV data for XLY
    sample_data = {
        'Date': pd.date_range('2024-09-20', periods=5, freq='D'),
        'Open': [150.75, 151.50, 150.25, 152.00, 150.80],
        'High': [152.10, 152.25, 150.75, 153.50, 151.25],
        'Low': [150.90, 150.80, 149.20, 151.75, 150.20],
        'Close': [150.75, 151.85, 149.80, 153.25, 150.95],
        'Volume': [1000000, 1100000, 1250000, 980000, 1150000]
    }

    # Create DataFrame
    df = pd.DataFrame(sample_data)
    df.set_index('Date', inplace=True)

    print("Original OHLCV Data:")
    print(df.round(2))
    print("\n" + "="*80 + "\n")

    # Calculate MMM gaps
    gap_df = calculate_mmm_gaps(df)

    print("MMM Gap Analysis Results:")
    print(gap_df.round(2))

    print("\n" + "="*80 + "\n")
    print("Key Calculations Explained:")
    print("Day 2 (2024-09-21):")
    print(f"  opening_gap = {gap_df.loc['2024-09-21', 'Open']:.2f} - {gap_df.loc['2024-09-21', 'Previous_Close']:.2f} = {gap_df.loc['2024-09-21', 'opening_gap']:.2f}")
    print(f"  price_without_opening_gap = {gap_df.loc['2024-09-21', 'Close']:.2f} - {gap_df.loc['2024-09-21', 'Open']:.2f} = {gap_df.loc['2024-09-21', 'price_without_opening_gap']:.2f}")

    print("\nDay 3 (2024-09-22):")
    print(f"  opening_gap = {gap_df.loc['2024-09-22', 'Open']:.2f} - {gap_df.loc['2024-09-22', 'Previous_Close']:.2f} = {gap_df.loc['2024-09-22', 'opening_gap']:.2f}")
    print(f"  price_without_opening_gap = {gap_df.loc['2024-09-22', 'Close']:.2f} - {gap_df.loc['2024-09-22', 'Open']:.2f} = {gap_df.loc['2024-09-22', 'price_without_opening_gap']:.2f}")

    print("\n" + "="*80 + "\n")
    print("MMM Analysis Insights:")
    print("- opening_gap: Shows gap between yesterday's close and today's open")
    print("- price_without_opening_gap: Shows intraday performance excluding gap")
    print("- Positive opening_gap = Gap up (opening higher)")
    print("- Negative opening_gap = Gap down (opening lower)")
    print("- price_without_opening_gap reveals true trading session performance")