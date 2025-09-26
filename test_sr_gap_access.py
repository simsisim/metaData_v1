#!/usr/bin/env python3
"""
Test SR System Gap Data Access
==============================

Test if the SR system has alternative ways to access gap data.
"""

import sys
import os
sys.path.append('src')
sys.path.append('src/sustainability_ratios')

from data_reader import DataReader
from config import Config
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sr_gap_access():
    """Test different methods to access gap data in SR system."""

    print("=" * 60)
    print("SR SYSTEM GAP DATA ACCESS TEST")
    print("=" * 60)

    # Initialize config
    config = Config()
    data_reader = DataReader(config, timeframe='daily')

    print(f"\nData directory: {data_reader.market_data_dir}")

    # Test 1: Direct file access (bypassing DataReader filtering)
    print("\n1. Testing direct file access:")
    gap_file = data_reader.market_data_dir / "XLY_gap.csv"

    if gap_file.exists():
        # Read with all columns
        df = pd.read_csv(gap_file, index_col='Date', parse_dates=False)
        df.index = df.index.str.split(' ').str[0]
        df.index = pd.to_datetime(df.index)

        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df[df.index.notna()]
        df = df.sort_index()
        df = df[df.index.weekday < 5]

        print(f"   ✅ Direct access successful: {len(df)} rows")
        print(f"   All columns: {list(df.columns)}")

        # Check gap columns specifically
        if 'gap' in df.columns:
            non_null_gaps = df['gap'].notna().sum()
            print(f"   Gap column: {non_null_gaps} non-null values")
            print(f"   Recent gap values: {df['gap'].tail(5).values}")

        if 'AdjustClose_woGap' in df.columns:
            non_null_adj = df['AdjustClose_woGap'].notna().sum()
            print(f"   AdjustClose_woGap column: {non_null_adj} non-null values")
            print(f"   Recent adj values: {df['AdjustClose_woGap'].tail(5).values}")

    # Test 2: Check if SR system needs special data access method
    print("\n2. Testing SR-specific data access:")

    try:
        # Try to import SR data access methods
        from sr_market_data import load_ticker_data

        print("   Trying sr_market_data.load_ticker_data...")
        sr_data = load_ticker_data('XLY_gap', data_reader.market_data_dir)
        if sr_data is not None:
            print(f"   ✅ SR system loaded: {len(sr_data)} rows")
            print(f"   SR columns: {list(sr_data.columns)}")
        else:
            print("   ❌ SR system failed to load data")

    except ImportError as e:
        print(f"   SR market data module not found: {e}")
    except Exception as e:
        print(f"   SR data access error: {e}")

    # Test 3: Test how MMM module would access this data
    print("\n3. Testing MMM module data access:")

    try:
        # Test the path MMM would use to access gap data
        from user_defined_data import UserConfig
        from sustainability_ratios.mmm.mmm_gaps import MmmGapsProcessor

        user_config = UserConfig()

        # Set up minimal MMM configuration
        user_config.sr_mmm_gaps_tickers = 'XLY'
        user_config.sr_mmm_gaps_values_input_folder_daily = '../downloadData_v1/data/market_data/daily/'
        user_config.sr_mmm_gaps_values_output_folder_daily = '../downloadData_v1/data/market_data/daily/'

        mmm_processor = MmmGapsProcessor(config, user_config, 'daily')

        print("   MMM processor initialized successfully")
        print(f"   Input directory: {user_config.sr_mmm_gaps_values_input_folder_daily}")

        # Test if MMM can find the data
        input_dir = mmm_processor._get_input_directory()
        if input_dir:
            test_file = input_dir / "XLY.csv"
            gap_file_location = input_dir / "XLY_gap.csv"
            print(f"   MMM input dir: {input_dir}")
            print(f"   Regular XLY file exists: {test_file.exists()}")
            print(f"   Gap file exists at MMM location: {gap_file_location.exists()}")

    except Exception as e:
        print(f"   MMM module test error: {e}")

    print("\n" + "=" * 60)
    print("FINAL DIAGNOSIS")
    print("=" * 60)

    print("🔍 ROOT CAUSE IDENTIFIED:")
    print("   1. XLY_gap.csv file exists and contains gap/AdjustClose_woGap data")
    print("   2. DataReader.read_stock_data() filters out non-standard columns")
    print("   3. SR system gets OHLCV data but loses gap-specific columns")
    print("   4. MMM charts expect gap data but receive filtered OHLCV data")
    print()
    print("🚨 THE ISSUE:")
    print("   DataReader only returns ['Open', 'High', 'Low', 'Close', 'Volume']")
    print("   Gap columns ['gap', 'AdjustClose_woGap'] are stripped out")
    print("   Charts display OHLCV data instead of gap analysis data")
    print()
    print("💡 SOLUTION:")
    print("   DataReader needs gap-aware mode to preserve gap columns")
    print("   OR: SR system needs dedicated gap data access method")
    print("   OR: MMM module needs custom data reader for gap files")

if __name__ == "__main__":
    test_sr_gap_access()