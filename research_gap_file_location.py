#!/usr/bin/env python3

"""
Research where gap files are actually stored and how data_reader should access them.
Check the gap file output locations and data_reader directory paths.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path

def research_gap_file_location():
    """Research where gap files are stored and how to access them."""

    print("🧪 RESEARCHING GAP FILE LOCATIONS")
    print("=" * 40)

    try:
        from src.config import Config
        from src.user_defined_data import read_user_data

        config = Config()
        user_config = read_user_data()

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Research 1: MMM output configuration
        print(f"🔍 RESEARCH 1: MMM OUTPUT CONFIGURATION")
        print(f"=" * 41)

        print(f"  🔧 MMM gap output folders:")
        daily_output = getattr(user_config, 'sr_mmm_gaps_values_output_folder_daily', '')
        weekly_output = getattr(user_config, 'sr_mmm_gaps_values_output_folder_weekly', '')
        monthly_output = getattr(user_config, 'sr_mmm_gaps_values_output_folder_monthly', '')

        print(f"    Daily:   {daily_output}")
        print(f"    Weekly:  {weekly_output}")
        print(f"    Monthly: {monthly_output}")

        # Research 2: Data reader input configuration
        print(f"\n🔍 RESEARCH 2: DATA READER INPUT CONFIGURATION")
        print(f"=" * 48)

        print(f"  📂 Data reader input folders:")
        print(f"    Daily:   {user_config.yf_daily_data_files}")
        print(f"    Weekly:  {user_config.yf_weekly_data_files}")
        print(f"    Monthly: {user_config.yf_monthly_data_files}")

        print()

        # Research 3: Compare paths
        print(f"🔍 RESEARCH 3: PATH COMPARISON")
        print(f"=" * 32)

        daily_output_path = Path(config.base_dir) / daily_output if daily_output else None
        daily_input_path = Path(config.base_dir) / user_config.yf_daily_data_files

        print(f"  📊 Daily paths:")
        print(f"    Gap output:   {daily_output_path}")
        print(f"    Data input:   {daily_input_path}")
        print(f"    Same location: {daily_output_path == daily_input_path if daily_output_path else 'N/A'}")

        print()

        # Research 4: Check actual gap file locations
        print(f"🔍 RESEARCH 4: ACTUAL GAP FILE LOCATIONS")
        print(f"=" * 42)

        # Check configured output location
        if daily_output_path and daily_output_path.exists():
            print(f"  📂 Checking gap output directory: {daily_output_path}")
            gap_files = list(daily_output_path.glob('*_gap.csv'))
            print(f"    Gap files found: {len(gap_files)}")
            for gap_file in sorted(gap_files)[:5]:
                print(f"      - {gap_file.name}")

        # Check data reader location
        if daily_input_path.exists():
            print(f"\n  📂 Checking data reader directory: {daily_input_path}")
            gap_files_in_data = list(daily_input_path.glob('*_gap.csv'))
            print(f"    Gap files found: {len(gap_files_in_data)}")
            for gap_file in sorted(gap_files_in_data)[:5]:
                print(f"      - {gap_file.name}")

        # Check relative path resolution
        relative_output_path = Path(daily_output) if daily_output else None
        if relative_output_path:
            print(f"\n  📂 Checking relative path: {relative_output_path}")
            if relative_output_path.exists():
                gap_files_relative = list(relative_output_path.glob('*_gap.csv'))
                print(f"    Gap files found: {len(gap_files_relative)}")
                for gap_file in sorted(gap_files_relative)[:5]:
                    print(f"      - {gap_file.name}")

        print()

        # Research 5: Test data_reader with correct path
        print(f"🔍 RESEARCH 5: DATA_READER PATH TESTING")
        print(f"=" * 41)

        try:
            from src.data_reader import DataReader

            # Test current data_reader
            data_reader = DataReader(config)
            print(f"  📊 Current data_reader test:")

            for ticker in ['XLY_gap', 'XLC_gap']:
                data = data_reader.read_stock_data(ticker)
                if data is not None and not data.empty:
                    print(f"    ✅ {ticker}: Found {len(data)} rows")
                else:
                    print(f"    ❌ {ticker}: Not found")

        except Exception as e:
            print(f"  ❌ Error testing data_reader: {e}")

        print()

        # Research 6: Analysis and recommendations
        print(f"🎯 RESEARCH 6: ANALYSIS AND RECOMMENDATIONS")
        print(f"=" * 45)

        print(f"  📋 Key Findings:")
        print(f"    1. Gap files are output to: {daily_output}")
        print(f"    2. Data reader looks in: {user_config.yf_daily_data_files}")
        print(f"    3. These paths {'match' if daily_output == user_config.yf_daily_data_files else 'differ'}")

        if daily_output == user_config.yf_daily_data_files:
            print(f"\n  ✅ GOOD: Gap files are in same location as market data")
            print(f"     - data_reader can find gap files")
            print(f"     - 'SPY_gap + EMA(20)' should work")
        else:
            print(f"\n  ❌ ISSUE: Gap files not in data_reader location")
            print(f"     - data_reader won't find gap files")
            print(f"     - Need to configure data_reader for gap files")

        print(f"\n  🔧 MMM Chart Configuration Strategy:")
        if daily_output == user_config.yf_daily_data_files:
            print(f"    ✅ Use standard sr_market_data approach")
            print(f"    ✅ 'SPY + EMA(20)' → SPY.csv")
            print(f"    ✅ 'XLY_gap + EMA(20)' → XLY_gap.csv")
        else:
            print(f"    📝 Options:")
            print(f"    1. Configure MMM to output gaps to data directory")
            print(f"    2. Configure data_reader to also search gap directory")
            print(f"    3. Custom MMM data resolution with multiple search paths")

        return True

    except Exception as e:
        print(f"❌ Error researching gap file location: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = research_gap_file_location()
    if success:
        print(f"\n🎉 Gap file location research completed!")
    else:
        print(f"\n💥 Gap file location research failed!")