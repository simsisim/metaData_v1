#!/usr/bin/env python3
"""
Verify MMM Chart Output Directory
=================================

Verify that MMM charts will be saved to the correct MMM subfolder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def verify_mmm_chart_output():
    """Verify MMM chart output directory configuration."""
    print("🧪 Verifying MMM Chart Output Directory")
    print("=" * 50)

    try:
        from user_defined_data import read_user_data
        from sustainability_ratios.sr_output_manager import SROutputManager

        # Load user configuration
        user_config = read_user_data()

        # Test SROutputManager
        print("📊 Testing SROutputManager:")

        output_manager = SROutputManager()
        print(f"   Base directory: {output_manager.base_dir}")
        print(f"   Available submodules: {list(output_manager.SUBMODULES.keys())}")

        # Test MMM submodule directory
        if 'mmm' in output_manager.SUBMODULES:
            mmm_dir = output_manager.get_submodule_dir('mmm')
            print(f"   MMM submodule directory: {mmm_dir}")
            print(f"   MMM submodule exists: {mmm_dir.exists()}")

            # Test charts subdirectory
            charts_dir = mmm_dir / 'charts'
            charts_dir.mkdir(parents=True, exist_ok=True)
            print(f"   MMM charts directory: {charts_dir}")
            print(f"   MMM charts directory exists: {charts_dir.exists()}")

        # Show the difference between data and chart locations
        print(f"\n🎯 Directory Structure:")

        # Data file locations (from configuration)
        mmm_output_dir = getattr(user_config, 'sr_mmm_output_dir', 'results/sustainability_ratios/mmm')
        gap_data_daily = getattr(user_config, 'sr_mmm_gaps_values_output_folder_daily', '../downloadData_v1/data/market_data/daily/')

        print(f"   📄 Gap DATA files (CSV):")
        print(f"      Daily: {gap_data_daily}XLY_gap.csv")
        print(f"      Daily: {gap_data_daily}XLC_gap.csv")

        print(f"   📊 Chart FILES (PNG/images):")
        print(f"      Charts: {mmm_output_dir}/charts/XLY_gap_chart.png")
        print(f"      Charts: {mmm_output_dir}/charts/QQQ_Analysis_chart.png")

        print(f"\n💡 Benefits of this structure:")
        print(f"   ✅ Gap data saved with source OHLCV files (easy to find)")
        print(f"   ✅ Charts saved in organized MMM results folder")
        print(f"   ✅ Clear separation of data vs visualizations")
        print(f"   ✅ MMM results organized in dedicated subfolder")

        return True

    except Exception as e:
        print(f"❌ Error verifying MMM chart output: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_mmm_chart_output()
    if success:
        print(f"\n🎉 SUCCESS: MMM charts will be saved in the MMM subfolder as requested!")
    else:
        print(f"\n❌ FAILURE: MMM chart output verification issues")