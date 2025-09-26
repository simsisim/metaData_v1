#!/usr/bin/env python3

"""
Test MMM configuration resolution logic to verify current behavior.
Check what config file is actually being used by MMM module.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pathlib import Path

def test_mmm_config_resolution():
    """Test current MMM configuration file resolution behavior."""

    print("🧪 TESTING MMM CONFIGURATION RESOLUTION")
    print("=" * 50)

    try:
        from src.sustainability_ratios.mmm.mmm_charts import MmmChartsProcessor
        from src.config import Config
        from src.user_defined_data import UserConfiguration

        # Create test configuration
        config = Config()
        user_config = UserConfiguration()

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Initialize MMM processor
        processor = MmmChartsProcessor(config, user_config, 'daily')

        print("✅ MmmChartsProcessor initialized successfully")
        print()

        # Test 1: Get the default configuration setting
        chart_config_file = getattr(user_config, 'sr_mmm_gaps_charts_display_panel', 'gaps_display.csv')
        print(f"🔧 CONFIGURATION SETTING:")
        print(f"  sr_mmm_gaps_charts_display_panel = '{chart_config_file}'")
        print(f"  Default fallback: 'gaps_display.csv'")
        print()

        # Test 2: Test path resolution with current method
        resolved_path = processor._resolve_config_path(chart_config_file)
        print(f"📍 PATH RESOLUTION (CURRENT METHOD):")
        print(f"  Input: '{chart_config_file}'")
        print(f"  Resolved: {resolved_path}")
        print(f"  Exists: {resolved_path.exists()}")
        print()

        # Test 3: Check what files exist in base directory
        print(f"📂 FILES IN BASE DIRECTORY ({config.base_dir}):")
        base_path = Path(config.base_dir)
        if base_path.exists():
            csv_files = list(base_path.glob('*.csv'))
            for csv_file in sorted(csv_files):
                if 'user_data' in csv_file.name or 'gap' in csv_file.name or 'mmm' in csv_file.name:
                    print(f"  ✅ {csv_file.name}")
        print()

        # Test 4: Test with expected user_data_sr_mmm.csv
        expected_file = 'user_data_sr_mmm.csv'
        expected_path = processor._resolve_config_path(expected_file)
        print(f"📍 EXPECTED FILE RESOLUTION:")
        print(f"  Input: '{expected_file}'")
        print(f"  Resolved: {expected_path}")
        print(f"  Exists: {expected_path.exists()}")
        print()

        # Test 5: Compare with other modules' defaults
        sr_panel_config = getattr(user_config, 'sr_panel_config_file', 'user_data_panel.csv')
        sr_overview_config = getattr(user_config, 'sr_overview_charts_display_panel', 'user_charts_display.csv')

        print(f"🔄 COMPARISON WITH OTHER MODULES:")
        print(f"  sr_panel_config_file = '{sr_panel_config}' (default: 'user_data_panel.csv')")
        print(f"  sr_overview_charts_display_panel = '{sr_overview_config}' (default: 'user_charts_display.csv')")
        print(f"  sr_mmm_gaps_charts_display_panel = '{chart_config_file}' (default: 'gaps_display.csv')")
        print()

        # Test 6: Check if user_data_sr_mmm.csv has content
        if expected_path.exists():
            print(f"📋 CONTENT OF user_data_sr_mmm.csv:")
            try:
                with open(expected_path, 'r') as f:
                    content = f.read()
                    print(content)
            except Exception as e:
                print(f"  Error reading file: {e}")
        else:
            print(f"❌ user_data_sr_mmm.csv does not exist at expected location")
        print()

        # Test 7: Test validation method
        print(f"🔍 VALIDATION RESULTS:")
        current_valid = processor._validate_chart_config(chart_config_file)
        expected_valid = processor._validate_chart_config(expected_file)

        print(f"  Current config ({chart_config_file}): Valid = {current_valid}")
        print(f"  Expected config ({expected_file}): Valid = {expected_valid}")
        print()

        # Summary
        print(f"📊 SUMMARY:")
        print(f"  Current MMM uses: '{chart_config_file}'")
        print(f"  Expected MMM should use: '{expected_file}'")
        print(f"  Follows same pattern as other modules: {chart_config_file.startswith('user_data_')}")
        print(f"  Configuration file exists: {expected_valid}")

        return {
            'current_config': chart_config_file,
            'expected_config': expected_file,
            'current_path': resolved_path,
            'expected_path': expected_path,
            'current_exists': current_valid,
            'expected_exists': expected_valid,
            'follows_pattern': chart_config_file.startswith('user_data_')
        }

    except Exception as e:
        print(f"❌ Error testing MMM configuration resolution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_mmm_config_resolution()
    if result:
        if not result['follows_pattern']:
            print(f"\n❌ MMM module does NOT follow same methodology as other SR modules")
        elif not result['expected_exists']:
            print(f"\n⚠️  MMM would follow correct pattern but user_data_sr_mmm.csv doesn't exist")
        else:
            print(f"\n✅ MMM configuration resolution test completed")
    else:
        print(f"\n💥 MMM configuration test failed!")