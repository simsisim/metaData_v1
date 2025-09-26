#!/usr/bin/env python3
"""
Research Column Renaming Approach for Gap Analysis
=================================================

Research the difference between original Close and AdjustClose_woGap
and test the feasibility of renaming AdjustClose_woGap to Close.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def research_column_renaming_approach():
    """Research AdjustClose_woGap vs Close and test column renaming approach."""

    print("=" * 70)
    print("RESEARCH: Column Renaming Approach for Gap Analysis")
    print("=" * 70)

    # Load both XLY files for comparison
    data_dir = Path("../downloadData_v1/data/market_data/daily")
    regular_file = data_dir / "XLY.csv"
    gap_file = data_dir / "XLY_gap.csv"

    if not regular_file.exists() or not gap_file.exists():
        print("❌ Required data files not found")
        return

    # Load regular XLY data
    print("\n1. Loading original XLY data...")
    regular_df = pd.read_csv(regular_file, index_col='Date', parse_dates=True)
    regular_df.index = regular_df.index.tz_localize(None) if hasattr(regular_df.index, 'tz') and regular_df.index.tz is not None else regular_df.index
    regular_df = regular_df.sort_index()

    # Load gap XLY data
    print("2. Loading XLY_gap data...")
    gap_df = pd.read_csv(gap_file, index_col='Date', parse_dates=False)
    gap_df.index = gap_df.index.str.split(' ').str[0]
    gap_df.index = pd.to_datetime(gap_df.index)
    gap_df.index = gap_df.index.tz_localize(None) if hasattr(gap_df.index, 'tz') and gap_df.index.tz is not None else gap_df.index
    gap_df = gap_df.sort_index()

    print(f"   Regular XLY: {len(regular_df)} rows, columns: {list(regular_df.columns)}")
    print(f"   Gap XLY: {len(gap_df)} rows, columns: {list(gap_df.columns)}")

    # Research the mathematical relationship
    print("\n3. Understanding AdjustClose_woGap formula...")
    print("   From MMM gap calculations:")
    print("   gap = Open[i] - Close[i-1]")
    print("   AdjustClose_woGap = Close[i] - (Close[i-1] - Open[i])")
    print("                     = Close[i] - Close[i-1] + Open[i]")
    print("                     = Close[i] + gap")

    # Verify the mathematical relationship
    print("\n4. Verifying mathematical relationship...")
    common_dates = regular_df.index.intersection(gap_df.index)
    if len(common_dates) > 10:
        common_regular = regular_df.loc[common_dates].copy()
        common_gap = gap_df.loc[common_dates].copy()

        # Test the formula: AdjustClose_woGap = Close + gap
        calculated_adjust = common_gap['Close'] + common_gap['gap']
        actual_adjust = common_gap['AdjustClose_woGap']

        # Compare
        difference = np.abs(calculated_adjust - actual_adjust)
        max_diff = difference.max()
        mean_diff = difference.mean()

        print(f"   Formula verification:")
        print(f"   Max difference: {max_diff:.8f}")
        print(f"   Mean difference: {mean_diff:.8f}")

        if max_diff < 1e-6:
            print("   ✅ Formula verified: AdjustClose_woGap = Close + gap")
        else:
            print("   ❌ Formula mismatch - needs investigation")

    # Compare Close values
    print("\n5. Comparing Close prices...")
    recent_dates = common_dates[-10:]

    print("   Recent data comparison (last 10 days):")
    print("   Date                Original_Close    Gap        AdjustClose_woGap")
    print("   " + "-" * 70)

    for date in recent_dates:
        orig_close = common_regular.loc[date, 'Close']
        gap_val = common_gap.loc[date, 'gap'] if pd.notna(common_gap.loc[date, 'gap']) else 0.0
        adj_close = common_gap.loc[date, 'AdjustClose_woGap']

        print(f"   {date.strftime('%Y-%m-%d')}    {orig_close:>10.2f}    {gap_val:>6.2f}    {adj_close:>10.2f}")

    # Analyze the impact of column renaming
    print("\n6. Analyzing column renaming impact...")

    # Calculate statistics
    close_stats = {
        'original_close_mean': common_regular['Close'].mean(),
        'original_close_std': common_regular['Close'].std(),
        'adjusted_close_mean': common_gap['AdjustClose_woGap'].mean(),
        'adjusted_close_std': common_gap['AdjustClose_woGap'].std()
    }

    print(f"   Original Close - Mean: {close_stats['original_close_mean']:.2f}, Std: {close_stats['original_close_std']:.2f}")
    print(f"   AdjustClose_woGap - Mean: {close_stats['adjusted_close_mean']:.2f}, Std: {close_stats['adjusted_close_std']:.2f}")

    # Calculate typical gap impact
    gap_impact = common_gap['gap'].abs().mean()
    print(f"   Average absolute gap: {gap_impact:.2f}")

    # Test visualization implications
    print("\n7. Testing visualization implications...")

    # Simulate what would happen with column renaming
    gap_df_renamed = common_gap.copy()
    gap_df_renamed['Close_original'] = gap_df_renamed['Close']
    gap_df_renamed['Close'] = gap_df_renamed['AdjustClose_woGap']  # Rename

    print("   After renaming AdjustClose_woGap -> Close:")
    print("   - Charts would display gap-adjusted closing prices")
    print("   - Technical indicators would use gap-adjusted data")
    print("   - Original close prices preserved as Close_original")

    # Test DataReader simulation
    print("\n8. Testing DataReader filtering simulation...")

    # Standard DataReader columns
    standard_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Current behavior (original Close)
    current_filtered = common_gap[standard_columns]
    print("   Current DataReader output (first 3 rows):")
    print("   ", current_filtered.head(3)['Close'].values)

    # With renaming (AdjustClose_woGap as Close)
    renamed_filtered = gap_df_renamed[standard_columns]
    print("   With renaming approach (first 3 rows):")
    print("   ", renamed_filtered.head(3)['Close'].values)

    print("\n" + "=" * 70)
    print("RESEARCH CONCLUSIONS")
    print("=" * 70)

    print("📊 MATHEMATICAL RELATIONSHIP:")
    print("   AdjustClose_woGap = Close + gap")
    print("   This represents closing price adjusted for opening gap impact")

    print("\n🎯 VISUALIZATION IMPACT:")
    print("   Original Close: Shows actual market closing prices")
    print("   AdjustClose_woGap: Shows gap-adjusted closing prices for analysis")

    print("\n✅ COLUMN RENAMING FEASIBILITY:")
    print("   - Mathematically sound: AdjustClose_woGap contains meaningful gap analysis data")
    print("   - DataReader compatible: Would pass through as 'Close' column")
    print("   - Preserves original data: XLY.csv retains actual closing prices")
    print("   - Purpose-built: Gap files show gap-adjusted prices for analysis")

    print("\n🔍 APPROACH COMPARISON:")
    print("   Gap-aware DataReader:")
    print("   + Preserves all columns")
    print("   + More complex, flexible")
    print("   - Requires SR system changes")
    print("   - More code complexity")
    print()
    print("   Column Renaming:")
    print("   + Simple, elegant solution")
    print("   + Works with existing DataReader")
    print("   + Gap files purpose-built for gap analysis")
    print("   + No SR system changes needed")
    print("   - Loses original Close in gap files (but available in XLY.csv)")

    print("\n💡 RECOMMENDATION:")
    print("   Column renaming approach is SUPERIOR:")
    print("   - Simpler implementation")
    print("   - Works with existing infrastructure")
    print("   - Gap files become truly gap-analysis focused")
    print("   - Original data preserved in separate files")

if __name__ == "__main__":
    research_column_renaming_approach()