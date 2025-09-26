#!/usr/bin/env python3
"""
Final MMM Configuration Test
===========================

Test MMM submodule with corrected configuration loading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from user_defined_data import read_user_data  # Use read_user_data instead of UserConfiguration
from config import Config

def test_mmm_final():
    """Test MMM submodule with proper configuration loading."""
    print("🧪 Final MMM Configuration Test")
    print("=" * 50)

    try:
        # Load configurations correctly
        config = Config()
        user_config = read_user_data()  # This is the key - use read_user_data()

        # Test MMM-specific configuration flags
        mmm_enable = getattr(user_config, 'sr_mmm_enable', None)
        mmm_daily_enable = getattr(user_config, 'sr_mmm_daily_enable', None)
        mmm_gaps_values = getattr(user_config, 'sr_mmm_gaps_values', None)
        mmm_output_dir = getattr(user_config, 'sr_mmm_output_dir', None)

        print("📊 MMM Configuration Status:")
        print(f"   sr_mmm_enable: {mmm_enable} (type: {type(mmm_enable).__name__})")
        print(f"   sr_mmm_daily_enable: {mmm_daily_enable} (type: {type(mmm_daily_enable).__name__})")
        print(f"   sr_mmm_gaps_values: {mmm_gaps_values} (type: {type(mmm_gaps_values).__name__})")
        print(f"   sr_mmm_output_dir: {mmm_output_dir}")
        print()

        # Simulate the check used in sr_calculations.py
        mmm_enabled_check = getattr(user_config, 'sr_mmm_enable', False)
        print(f"🔍 MMM Enablement Check (same as sr_calculations.py):")
        print(f"   getattr(user_config, 'sr_mmm_enable', False) = {mmm_enabled_check}")
        print(f"   Type: {type(mmm_enabled_check).__name__}")
        print(f"   Boolean evaluation: {bool(mmm_enabled_check)}")
        print()

        if mmm_enabled_check:
            print("✅ MMM submodule should be ENABLED")

            # Test MMM processor import
            try:
                print("🔍 Testing MMM processor import...")
                from sustainability_ratios.mmm import MmmProcessor
                print("✅ MMM processor import successful")

                # Test MMM processor initialization
                print("🔍 Testing MMM processor initialization...")
                mmm_processor = MmmProcessor(config, user_config, 'daily')
                print("✅ MMM processor initialization successful")

                print(f"   MMM processor timeframe: {mmm_processor.timeframe}")

            except Exception as e:
                print(f"❌ Error with MMM processor: {e}")
                import traceback
                traceback.print_exc()

            # Test directory creation with correct case
            try:
                print("🔍 Testing output directory creation...")
                from sustainability_ratios.sr_output_manager import get_sr_output_manager

                output_manager = get_sr_output_manager()
                mmm_path = output_manager.get_submodule_path('mmm')  # lowercase
                print(f"   MMM path: {mmm_path}")
                print(f"   MMM path exists: {mmm_path.exists()}")

                gaps_dir = mmm_path / 'gaps'
                gaps_dir.mkdir(parents=True, exist_ok=True)
                print(f"   Gaps directory: {gaps_dir}")
                print(f"   Gaps directory created: {gaps_dir.exists()}")
                print("✅ Directory structure working correctly")

            except Exception as e:
                print(f"❌ Error with directory creation: {e}")

        else:
            print("❌ MMM submodule will be DISABLED")

        return mmm_enabled_check

    except Exception as e:
        print(f"❌ Error in final MMM test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    enabled = test_mmm_final()
    print(f"\n🎯 Final Result: MMM submodule is {'ENABLED' if enabled else 'DISABLED'}")