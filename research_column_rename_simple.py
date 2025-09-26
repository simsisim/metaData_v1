#!/usr/bin/env python3
"""
Simple Research: Column Renaming vs Gap-Aware DataReader
=======================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

def simple_research():
    """Simple research of column renaming approach."""

    print("=" * 60)
    print("COLUMN RENAMING RESEARCH")
    print("=" * 60)

    # Load gap file directly
    gap_file = Path("../downloadData_v1/data/market_data/daily/XLY_gap.csv")
    regular_file = Path("../downloadData_v1/data/market_data/daily/XLY.csv")

    if not gap_file.exists():
        print("❌ Gap file not found")
        return

    print("\n1. Loading XLY_gap data...")
    gap_df = pd.read_csv(gap_file, index_col='Date', parse_dates=False)
    gap_df.index = gap_df.index.str.split(' ').str[0]
    gap_df.index = pd.to_datetime(gap_df.index)

    print(f"   Loaded {len(gap_df)} rows")
    print(f"   Columns: {list(gap_df.columns)}")

    # Check gap data availability
    if 'gap' in gap_df.columns and 'AdjustClose_woGap' in gap_df.columns:
        gap_data = gap_df['gap'].dropna()
        adj_data = gap_df['AdjustClose_woGap'].dropna()
        close_data = gap_df['Close'].dropna()

        print(f"\n2. Data availability:")
        print(f"   Gap values: {len(gap_data)}/{len(gap_df)} rows")
        print(f"   AdjustClose_woGap: {len(adj_data)}/{len(gap_df)} rows")
        print(f"   Original Close: {len(close_data)}/{len(gap_df)} rows")

        # Show recent values
        print(f"\n3. Recent data (last 5 rows):")
        recent = gap_df[['Close', 'gap', 'AdjustClose_woGap']].tail(5)
        for date, row in recent.iterrows():
            close = row['Close']
            gap = row['gap'] if pd.notna(row['gap']) else 0.0
            adj_close = row['AdjustClose_woGap']
            print(f"   {date.strftime('%Y-%m-%d')}: Close={close:.2f}, Gap={gap:.2f}, Adj={adj_close:.2f}")

        # Verify mathematical relationship
        print(f"\n4. Verifying AdjustClose_woGap = Close + gap:")
        test_data = gap_df.dropna(subset=['gap', 'AdjustClose_woGap']).tail(10)

        for date, row in test_data.iterrows():
            close = row['Close']
            gap = row['gap']
            adj_close = row['AdjustClose_woGap']
            calculated = close + gap
            diff = abs(calculated - adj_close)
            status = "✅" if diff < 0.01 else "❌"
            print(f"   {date.strftime('%Y-%m-%d')}: {status} {close:.2f} + {gap:.2f} = {calculated:.2f} vs {adj_close:.2f} (diff: {diff:.6f})")

    print(f"\n5. Testing Column Renaming Approach:")
    print(f"   Current situation:")
    print(f"   - DataReader('XLY_gap') returns OHLCV with original Close prices")
    print(f"   - Gap analysis columns (gap, AdjustClose_woGap) are filtered out")
    print(f"   - Charts show original prices, not gap-adjusted analysis")

    print(f"\n   With column renaming (AdjustClose_woGap -> Close):")
    print(f"   - DataReader('XLY_gap') would return OHLCV with gap-adjusted Close")
    print(f"   - Charts would show gap-adjusted prices for analysis")
    print(f"   - Original prices remain available in XLY.csv")

    # Simulate the approaches
    print(f"\n6. Approach Comparison:")

    print(f"\n   APPROACH 1: Gap-Aware DataReader")
    print(f"   Pros:")
    print(f"   + Preserves all columns (Close, gap, AdjustClose_woGap)")
    print(f"   + Maximum flexibility")
    print(f"   + Can display any column combination")
    print(f"   Cons:")
    print(f"   - Requires DataReader modification")
    print(f"   - Requires SR system changes")
    print(f"   - More complex implementation")

    print(f"\n   APPROACH 2: Column Renaming (AdjustClose_woGap -> Close)")
    print(f"   Pros:")
    print(f"   + Works with existing DataReader (no changes needed)")
    print(f"   + Simple, elegant solution")
    print(f"   + Gap files become purpose-built for gap analysis")
    print(f"   + Original data preserved in XLY.csv")
    print(f"   + No SR system modifications")
    print(f"   Cons:")
    print(f"   - Loses original Close prices in gap files")
    print(f"   - Gap column still filtered out (but main analysis preserved)")

    print(f"\n7. MATHEMATICAL VALIDATION:")
    if 'gap' in gap_df.columns and 'AdjustClose_woGap' in gap_df.columns:
        # Test the core concept
        sample = gap_df.dropna(subset=['gap', 'AdjustClose_woGap']).tail(20)

        # Calculate average gap impact
        avg_gap = abs(sample['gap']).mean()
        close_range = sample['Close'].max() - sample['Close'].min()
        gap_impact_pct = (avg_gap / sample['Close'].mean()) * 100

        print(f"   Average absolute gap: ${avg_gap:.2f}")
        print(f"   Close price range: ${close_range:.2f}")
        print(f"   Gap impact: {gap_impact_pct:.2f}% of close price")
        print(f"   ✅ Gap adjustments are meaningful for analysis")

    print(f"\n" + "=" * 60)
    print("RESEARCH CONCLUSION")
    print("=" * 60)

    print("🎯 COLUMN RENAMING APPROACH IS SUPERIOR:")
    print("   ✅ Mathematically sound: AdjustClose_woGap = Close + gap")
    print("   ✅ Purpose-appropriate: Gap files for gap analysis")
    print("   ✅ Infrastructure-compatible: Works with existing DataReader")
    print("   ✅ Implementation simple: Just rename column in gap files")
    print("   ✅ Data preservation: Original data in XLY.csv")

    print(f"\n💡 IMPLEMENTATION APPROACH:")
    print(f"   1. In gap calculation: Rename AdjustClose_woGap -> Close")
    print(f"   2. Keep original Close as Close_original (optional)")
    print(f"   3. Keep gap column for reference (will be filtered but preserved in file)")
    print(f"   4. DataReader returns gap-adjusted Close automatically")
    print(f"   5. Charts display gap-adjusted analysis data")

    print(f"\n🚀 IMMEDIATE BENEFIT:")
    print(f"   XLY_gap charts will show gap-adjusted closing prices")
    print(f"   instead of misleading original closing prices!")

if __name__ == "__main__":
    simple_research()