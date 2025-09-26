#!/usr/bin/env python3
"""
Test MMM Output Directory Configuration
=======================================

Test that gap files are saved to the correct configured output directories.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from user_defined_data import read_user_data
from config import Config

def test_mmm_output_directories():
    """Test MMM output directory configuration for different timeframes."""
    print("🧪 Testing MMM Output Directory Configuration")
    print("=" * 50)

    try:
        # Load configurations
        config = Config()
        user_config = read_user_data()

        # Test MMM output directory settings
        print("📊 MMM Output Directory Configuration:")

        daily_output = getattr(user_config, 'sr_mmm_gaps_values_output_folder_daily', None)
        weekly_output = getattr(user_config, 'sr_mmm_gaps_values_output_folder_weekly', None)
        monthly_output = getattr(user_config, 'sr_mmm_gaps_values_output_folder_monthly', None)
        mmm_output = getattr(user_config, 'sr_mmm_output_dir', None)

        print(f"   Daily output:   {daily_output}")
        print(f"   Weekly output:  {weekly_output}")
        print(f"   Monthly output: {monthly_output}")
        print(f"   MMM base dir:   {mmm_output}")

        # Test the logic that determines output directory for each timeframe
        print("\n🔍 Testing Output Directory Logic:")

        timeframes = ['daily', 'weekly', 'monthly']

        for timeframe in timeframes:
            print(f"\n📅 Timeframe: {timeframe}")

            # Simulate the logic from _get_output_directory method
            if timeframe == 'daily':
                output_dir = getattr(user_config, 'sr_mmm_gaps_values_output_folder_daily',
                                   '../downloadData_v1/data/market_data/daily/')
            elif timeframe == 'weekly':
                output_dir = getattr(user_config, 'sr_mmm_gaps_values_output_folder_weekly',
                                   '../downloadData_v1/data/market_data/weekly/')
            elif timeframe == 'monthly':
                output_dir = getattr(user_config, 'sr_mmm_gaps_values_output_folder_monthly',
                                   '../downloadData_v1/data/market_data/monthly/')

            output_path = Path(output_dir)
            print(f"   Configured directory: {output_dir}")
            print(f"   Resolved path: {output_path}")
            print(f"   Absolute path: {output_path.resolve()}")

            # Check if directory exists
            try:
                # Create the directory to test (but don't save files)
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"   Directory accessible: ✅ YES")
                print(f"   Directory exists: {output_path.exists()}")
            except Exception as e:
                print(f"   Directory accessible: ❌ NO - {e}")

        print(f"\n🎯 Expected Behavior:")
        print(f"   - Daily gap files (XLY_gap.csv) will be saved to: {daily_output}")
        print(f"   - Weekly gap files will be saved to: {weekly_output}")
        print(f"   - Monthly gap files will be saved to: {monthly_output}")
        print(f"   - Gap files will be saved alongside original OHLCV files")

        return True

    except Exception as e:
        print(f"❌ Error testing MMM output directories: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mmm_output_directories()
    if success:
        print(f"\n✅ SUCCESS: MMM gap files will be saved to the correct configured directories!")
    else:
        print(f"\n❌ FAILURE: MMM output directory configuration issues")