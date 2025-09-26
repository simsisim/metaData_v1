#!/usr/bin/env python3
"""
Test Gap Data Resolution Issue
============================

Test how the SR data reader handles gap files vs regular files.
"""

import sys
import os
sys.path.append('src')

from data_reader import DataReader
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_gap_data_resolution():
    """Test gap data resolution patterns."""

    print("=" * 60)
    print("GAP DATA RESOLUTION TEST")
    print("=" * 60)

    # Initialize config and data reader
    config = Config()
    data_reader = DataReader(config, timeframe='daily')

    print(f"\nData directory: {data_reader.market_data_dir}")

    # Test 1: Regular XLY file
    print("\n1. Testing regular XLY file:")
    regular_data = data_reader.read_stock_data('XLY')
    if regular_data is not None:
        print(f"   ✅ Regular XLY data loaded: {len(regular_data)} rows")
        print(f"   Columns: {list(regular_data.columns)}")
    else:
        print("   ❌ Regular XLY data not found")

    # Test 2: XLY_gap file (will fail with current data reader)
    print("\n2. Testing XLY_gap file:")
    gap_data = data_reader.read_stock_data('XLY_gap')
    if gap_data is not None:
        print(f"   ✅ XLY_gap data loaded: {len(gap_data)} rows")
        print(f"   Columns: {list(gap_data.columns)}")
        if 'gap' in gap_data.columns:
            print(f"   Gap column found: {gap_data['gap'].notna().sum()} non-null values")
    else:
        print("   ❌ XLY_gap data not found")

    # Test 3: Check if XLY_gap.csv file actually exists
    print("\n3. Testing file existence:")
    gap_file = data_reader.market_data_dir / "XLY_gap.csv"
    regular_file = data_reader.market_data_dir / "XLY.csv"

    print(f"   Regular file exists: {regular_file.exists()}")
    print(f"   Gap file exists: {gap_file.exists()}")

    if gap_file.exists():
        print(f"   Gap file path: {gap_file}")
        # Try reading manually
        import pandas as pd
        try:
            manual_gap_data = pd.read_csv(gap_file, index_col=0, parse_dates=True)
            print(f"   ✅ Manual read successful: {len(manual_gap_data)} rows")
            print(f"   Manual columns: {list(manual_gap_data.columns)}")
            if 'gap' in manual_gap_data.columns:
                non_null_gaps = manual_gap_data['gap'].notna().sum()
                total_rows = len(manual_gap_data)
                print(f"   Gap data: {non_null_gaps}/{total_rows} rows have gap values")
        except Exception as e:
            print(f"   ❌ Manual read failed: {e}")

    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)

    if gap_file.exists() and gap_data is None:
        print("🔍 ISSUE IDENTIFIED:")
        print("   - XLY_gap.csv file exists and contains gap data")
        print("   - DataReader.read_stock_data('XLY_gap') returns None")
        print("   - This means the SR system cannot resolve 'XLY_gap' data sources")
        print("\n🚨 ROOT CAUSE:")
        print("   The DataReader looks for files as '{ticker}.csv'")
        print("   But gap files are stored as '{ticker}_gap.csv'")
        print("   So 'XLY_gap' looks for 'XLY_gap.csv' - which exists!")
        print("   But the SR system may have different expectations.")

        # Check the exact file pattern the data reader expects
        expected_path = data_reader.market_data_dir / "XLY_gap.csv"
        print(f"\n   Expected path: {expected_path}")
        print(f"   File exists: {expected_path.exists()}")

        if expected_path.exists() and gap_data is None:
            print("\n🔍 DEEPER ANALYSIS NEEDED:")
            print("   The file exists at expected location but still fails to load")
            print("   This suggests an issue in the data reading process itself")

if __name__ == "__main__":
    test_gap_data_resolution()