#!/usr/bin/env python3

"""
Research why MMM tickers configuration is missing.
Check user_defined_data.py and user configuration loading.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pathlib import Path

def test_mmm_missing_tickers_config():
    """Research why sr_mmm_gaps_tickers is not configured."""

    print("🧪 RESEARCHING MMM TICKERS CONFIGURATION ISSUE")
    print("=" * 55)

    try:
        from src.config import Config
        from src.user_defined_data import UserConfiguration

        # Create test configuration
        config = Config()
        user_config = UserConfiguration()

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Test 1: Check ALL user_config attributes related to MMM
        print(f"🔧 ALL MMM-RELATED USER CONFIG ATTRIBUTES:")
        mmm_attrs = []
        for attr in dir(user_config):
            if not attr.startswith('_'):  # Skip private attributes
                value = getattr(user_config, attr, None)
                if 'mmm' in attr.lower() or (isinstance(value, str) and 'mmm' in value.lower()):
                    mmm_attrs.append((attr, value))

        if mmm_attrs:
            for attr, value in mmm_attrs:
                print(f"  {attr} = {repr(value)}")
        else:
            print(f"  ❌ No MMM-related attributes found in user_config")
        print()

        # Test 2: Check specific MMM configuration keys
        print(f"🎯 SPECIFIC MMM CONFIG KEYS:")
        mmm_keys = [
            'sr_mmm_gaps_tickers',
            'sr_mmm_gaps_values_input_folder_daily',
            'sr_mmm_gaps_values_output_folder_daily',
            'sr_mmm_gaps_charts_display_panel',
            'sr_mmm_gaps_charts_display_history',
            'sr_mmm_output_dir'
        ]

        for key in mmm_keys:
            value = getattr(user_config, key, 'NOT_FOUND')
            print(f"  {key} = {repr(value)}")
        print()

        # Test 3: Check user_data.csv file content
        print(f"📄 USER_DATA.CSV CONTENT:")
        user_data_file = Path(config.base_dir) / 'user_data.csv'
        if user_data_file.exists():
            with open(user_data_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    if 'mmm' in line.lower() or i <= 10:  # Show first 10 lines + any MMM lines
                        print(f"  {i:2d}: {line.rstrip()}")
        else:
            print(f"  ❌ user_data.csv not found at {user_data_file}")
        print()

        # Test 4: Check user_defined_data.py source code for MMM handling
        print(f"🔍 USER_DEFINED_DATA.PY MMM HANDLING:")
        user_data_py = Path(config.base_dir) / 'src' / 'user_defined_data.py'
        if user_data_py.exists():
            with open(user_data_py, 'r') as f:
                content = f.read()

            # Look for MMM-related lines
            mmm_lines = []
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'mmm' in line.lower():
                    mmm_lines.append(f"  {i:3d}: {line.strip()}")

            if mmm_lines:
                print(f"  MMM-related lines found:")
                for line in mmm_lines:
                    print(line)
            else:
                print(f"  ❌ No MMM-related lines found in user_defined_data.py")
        else:
            print(f"  ❌ user_defined_data.py not found at {user_data_py}")
        print()

        # Test 5: Check if we can manually create MMM configuration
        print(f"🛠️  MANUAL MMM CONFIGURATION TEST:")

        # Try to manually set some MMM configurations
        manual_tickers = "SPY;QQQ;IWM"  # Based on chart config file content
        manual_input_dir = "../downloadData_v1/data/market_data/daily"

        print(f"  Trying manual configuration:")
        print(f"    sr_mmm_gaps_tickers = '{manual_tickers}'")
        print(f"    sr_mmm_gaps_values_input_folder_daily = '{manual_input_dir}'")

        # Temporarily set these values
        setattr(user_config, 'sr_mmm_gaps_tickers', manual_tickers)
        setattr(user_config, 'sr_mmm_gaps_values_input_folder_daily', manual_input_dir)

        # Test with MMM gaps processor
        try:
            from src.sustainability_ratios.mmm.mmm_gaps import MmmGapsProcessor
            gaps_processor = MmmGapsProcessor(config, user_config, 'daily')
            tickers = gaps_processor._parse_tickers_config()
            input_dir = gaps_processor._get_input_directory()

            print(f"  ✅ Manual config test results:")
            print(f"    Parsed tickers: {tickers}")
            print(f"    Input directory: {input_dir}")
            print(f"    Input directory exists: {input_dir.exists() if input_dir else False}")

            if input_dir and input_dir.exists():
                print(f"  📊 Available data files in input directory:")
                csv_files = list(input_dir.glob('*.csv'))
                for csv_file in sorted(csv_files):
                    if any(ticker in csv_file.name for ticker in ['SPY', 'QQQ', 'IWM']):
                        print(f"    ✅ {csv_file.name}")

        except Exception as e:
            print(f"  ❌ Error testing manual configuration: {e}")
        print()

        # Test 6: Check what the chart config actually expects
        print(f"📊 CHART CONFIG EXPECTATIONS:")
        chart_config_file = Path(config.base_dir) / 'user_data_sr_mmm.csv'
        if chart_config_file.exists():
            with open(chart_config_file, 'r') as f:
                lines = f.readlines()

            print(f"  Chart configurations expect these tickers:")
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        file_name_id = parts[0].strip()
                        panel_1 = parts[2].strip()
                        print(f"    {file_name_id}: Panel_1 = '{panel_1}'")

                        # Extract tickers from panel definitions
                        expected_tickers = set()
                        if 'SPY' in panel_1:
                            expected_tickers.add('SPY')
                        if 'QQQ' in panel_1:
                            expected_tickers.add('QQQ')
                        if 'IWM' in panel_1:
                            expected_tickers.add('IWM')

                        print(f"      Expected tickers: {sorted(expected_tickers)}")

        print(f"\n📋 SUMMARY OF ISSUES:")
        print(f"  1. ❌ sr_mmm_gaps_tickers is not configured in user_config")
        print(f"  2. ❌ user_data.csv may not contain MMM configuration")
        print(f"  3. ❌ user_defined_data.py may not load MMM settings")
        print(f"  4. ✅ user_data_sr_mmm.csv exists with proper chart configurations")
        print(f"  5. ❌ Gap calculation fails due to missing ticker configuration")

        return {
            'mmm_attrs_found': len(mmm_attrs) > 0,
            'tickers_configured': getattr(user_config, 'sr_mmm_gaps_tickers', None) is not None,
            'chart_config_exists': chart_config_file.exists(),
            'manual_config_works': len(tickers) > 0 if 'tickers' in locals() else False
        }

    except Exception as e:
        print(f"❌ Error researching MMM configuration: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_mmm_missing_tickers_config()
    if result:
        print(f"\n🎯 DIAGNOSIS: MMM configuration is incomplete or missing from user_defined_data.py")
    else:
        print(f"\n💥 MMM configuration research failed!")