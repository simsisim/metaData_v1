#!/usr/bin/env python3

"""
Test if MMM is reading the updated user_data_sr_mmm.csv correctly.
Verify the exact content and parsing behavior.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pathlib import Path

def test_mmm_config_reading():
    """Test if MMM correctly reads the updated user_data_sr_mmm.csv file."""

    print("🧪 TESTING MMM CONFIG FILE READING")
    print("=" * 50)

    try:
        from src.sustainability_ratios.mmm.mmm_charts import MmmChartsProcessor
        from src.config import Config
        from src.user_defined_data import UserConfiguration

        # Create test configuration
        config = Config()
        user_config = UserConfiguration()
        processor = MmmChartsProcessor(config, user_config, 'daily')

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Test 1: Get configuration file name
        chart_config_file = getattr(user_config, 'sr_mmm_gaps_charts_display_panel', 'user_data_sr_mmm.csv')
        resolved_path = processor._resolve_config_path(chart_config_file)

        print(f"🔧 CONFIGURATION:")
        print(f"  Config key: sr_mmm_gaps_charts_display_panel")
        print(f"  Config value: '{chart_config_file}'")
        print(f"  Resolved path: {resolved_path}")
        print(f"  File exists: {resolved_path.exists()}")
        print()

        # Test 2: Read raw file content
        print(f"📄 RAW FILE CONTENT:")
        if resolved_path.exists():
            with open(resolved_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    print(f"  {i:2d}: {repr(line)}")  # Use repr to see exact characters
        print()

        # Test 3: Test parsing with sr_config_reader
        print(f"🔍 PARSING TEST WITH SR_CONFIG_READER:")
        try:
            from src.sustainability_ratios.sr_config_reader import parse_panel_config
            panels_config = parse_panel_config(str(resolved_path))

            if panels_config:
                print(f"  ✅ Successfully parsed {len(panels_config)} row configurations")

                for row_idx, row_config in enumerate(panels_config, 1):
                    print(f"  \n  📊 ROW {row_idx} - {len(row_config)} panel configs:")

                    for config_key, panel_config in row_config.items():
                        file_name_id = panel_config.get('file_name_id', 'unknown')
                        chart_type = panel_config.get('chart_type', 'unknown')
                        data_source = panel_config.get('data_source', 'unknown')

                        print(f"    🔸 Key: {config_key}")
                        print(f"      - file_name_id: '{file_name_id}'")
                        print(f"      - chart_type: '{chart_type}'")
                        print(f"      - data_source: '{data_source}'")

                        # Show all panel config details
                        for key, value in panel_config.items():
                            if key not in ['file_name_id', 'chart_type', 'data_source']:
                                print(f"      - {key}: {value}")
            else:
                print(f"  ❌ No panel configurations found")
                return False

        except Exception as e:
            print(f"  ❌ Error parsing config: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test 4: Check for expected file_name_ids
        print(f"\n🎯 EXPECTED vs ACTUAL:")
        expected_file_ids = ['mmm_SPY_vs_IWM_mmm', 'mmm_QQQ_Analysis_mmm']
        found_file_ids = []

        for row_config in panels_config:
            for panel_config in row_config.values():
                file_id = panel_config.get('file_name_id')
                if file_id and file_id not in found_file_ids:
                    found_file_ids.append(file_id)

        print(f"  Expected file_name_ids: {expected_file_ids}")
        print(f"  Found file_name_ids: {found_file_ids}")

        for expected in expected_file_ids:
            if expected in found_file_ids:
                print(f"  ✅ Found: {expected}")
            else:
                print(f"  ❌ Missing: {expected}")

        # Test 5: Check chart types
        print(f"\n📈 CHART TYPES:")
        for row_config in panels_config:
            for config_key, panel_config in row_config.items():
                file_id = panel_config.get('file_name_id')
                chart_type = panel_config.get('chart_type')
                if file_id in expected_file_ids:
                    expected_type = 'candle' if 'SPY_vs_IWM' in file_id else 'line'
                    match = chart_type == expected_type
                    print(f"  {file_id}: {chart_type} (expected: {expected_type}) {'✅' if match else '❌'}")

        return True

    except Exception as e:
        print(f"❌ Error testing MMM config reading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mmm_config_reading()
    if success:
        print(f"\n🎉 MMM config reading test completed!")
    else:
        print(f"\n💥 MMM config reading test failed!")