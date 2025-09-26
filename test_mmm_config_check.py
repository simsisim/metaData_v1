#!/usr/bin/env python3
"""
Test MMM Configuration Loading
=============================

Check if MMM submodule configuration is being read correctly after case fixes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from user_defined_data import UserConfiguration
from config import Config

def test_mmm_configuration():
    """Test if MMM configuration is being read correctly."""
    print("🧪 Testing MMM Configuration Loading")
    print("=" * 50)

    try:
        # Load configurations
        config = Config()
        user_config = UserConfiguration()

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

            # Test directory creation with new case
            from sustainability_ratios.sr_output_manager import get_sr_output_manager

            output_manager = get_sr_output_manager()
            print(f"📁 Testing directory structure:")
            print(f"   Base directory: {output_manager.base_dir}")

            # Check if mmm subdirectory can be accessed
            try:
                mmm_path = output_manager.get_submodule_path('mmm')
                print(f"   MMM path: {mmm_path}")
                print(f"   MMM path exists: {mmm_path.exists()}")

                # Test gaps subdirectory
                gaps_dir = mmm_path / 'gaps'
                gaps_dir.mkdir(parents=True, exist_ok=True)
                print(f"   Gaps directory: {gaps_dir}")
                print(f"   Gaps directory exists: {gaps_dir.exists()}")

            except Exception as e:
                print(f"   ❌ Error accessing MMM directories: {e}")

        else:
            print("❌ MMM submodule will be DISABLED")

        # Additional diagnostics
        print("\n🔎 Additional Diagnostics:")

        # Check all sr_mmm attributes
        mmm_attrs = [attr for attr in dir(user_config) if attr.startswith('sr_mmm')]
        print(f"   All sr_mmm attributes found: {len(mmm_attrs)}")
        for attr in sorted(mmm_attrs)[:10]:  # Show first 10
            value = getattr(user_config, attr)
            print(f"     {attr}: {value}")

        if len(mmm_attrs) > 10:
            print(f"     ... and {len(mmm_attrs) - 10} more")

        return mmm_enabled_check

    except Exception as e:
        print(f"❌ Error testing MMM configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    enabled = test_mmm_configuration()
    print(f"\n🎯 Final Result: MMM submodule is {'ENABLED' if enabled else 'DISABLED'}")