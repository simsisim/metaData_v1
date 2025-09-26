#!/usr/bin/env python3
"""
Verify MMM Chart Fix
===================

Test if adding panel indices to mmm_test configuration fixes the XLY_gap display issue.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from pathlib import Path

def verify_mmm_fix():
    """Verify that the mmm_test configuration now has panel indices."""

    print("=" * 80)
    print("VERIFY: MMM CHART FIX")
    print("=" * 80)

    # 1. Verify CSV configuration
    print("\n1. Verifying CSV Configuration:")

    config_file = Path("user_data_sr_mmm.csv")
    if config_file.exists():
        try:
            config_df = pd.read_csv(config_file, comment='#')

            # Find mmm_test row
            test_row = config_df[config_df['file_name_id'] == 'mmm_test']
            if not test_row.empty:
                test_config = test_row.iloc[0]

                print(f"   mmm_test configuration:")
                print(f"     file_name_id: {test_config['file_name_id']}")
                print(f"     chart_type: {test_config['chart_type']}")
                print(f"     Panel_1: '{test_config['Panel_1']}'")
                print(f"     Panel_2: '{test_config['Panel_2']}'")
                print(f"     Panel_1_index: '{test_config['Panel_1_index']}'")
                print(f"     Panel_2_index: '{test_config['Panel_2_index']}'")

                # Verify fix
                if pd.isna(test_config['Panel_1_index']) or test_config['Panel_1_index'] == '':
                    print("   ❌ Panel_1_index is still empty!")
                else:
                    print(f"   ✅ Panel_1_index configured: '{test_config['Panel_1_index']}'")

                if pd.isna(test_config['Panel_2_index']) or test_config['Panel_2_index'] == '':
                    print("   ❌ Panel_2_index is still empty!")
                else:
                    print(f"   ✅ Panel_2_index configured: '{test_config['Panel_2_index']}'")

            else:
                print("   ❌ mmm_test configuration not found!")

        except Exception as e:
            print(f"   ❌ Error reading MMM configuration: {e}")
    else:
        print(f"   ❌ Configuration file not found: {config_file}")

    # 2. Compare with working configuration
    print("\n2. Comparing with Working Configuration:")

    if config_file.exists():
        try:
            config_df = pd.read_csv(config_file, comment='#')

            configs_to_compare = ['mmm_QQQ_Analysis_mmm', 'mmm_test']

            for config_name in configs_to_compare:
                row = config_df[config_df['file_name_id'] == config_name]
                if not row.empty:
                    config = row.iloc[0]

                    print(f"\n   {config_name}:")
                    print(f"     Panel_1: '{config['Panel_1']}'")
                    print(f"     Panel_2: '{config['Panel_2']}'")
                    print(f"     Panel_1_index: '{config.get('Panel_1_index', 'N/A')}'")
                    print(f"     Panel_2_index: '{config.get('Panel_2_index', 'N/A')}'")

                    # Check if both panels and indices are configured
                    has_panel_1 = not (pd.isna(config['Panel_1']) or config['Panel_1'] == '')
                    has_panel_2 = not (pd.isna(config['Panel_2']) or config['Panel_2'] == '')
                    has_index_1 = not (pd.isna(config.get('Panel_1_index', '')) or config.get('Panel_1_index', '') == '')
                    has_index_2 = not (pd.isna(config.get('Panel_2_index', '')) or config.get('Panel_2_index', '') == '')

                    print(f"     Status: Panel_1={has_panel_1}, Index_1={has_index_1}")
                    print(f"             Panel_2={has_panel_2}, Index_2={has_index_2}")

                    if has_panel_1 and has_panel_2 and has_index_1 and has_index_2:
                        print(f"     ✅ Complete configuration")
                    else:
                        print(f"     ⚠️ Incomplete configuration")

        except Exception as e:
            print(f"   ❌ Error comparing configurations: {e}")

    print("\n" + "=" * 80)
    print("FIX VERIFICATION SUMMARY")
    print("=" * 80)

    print("\n📋 CHANGES MADE:")
    print("   Added Panel_1_index: A_RSI(14)_for_(XLY)")
    print("   Added Panel_2_index: B_RSI(14)_for_(XLY_gap)")

    print("\n🔧 EXPECTED BEHAVIOR:")
    print("   - XLY_gap should now display in Panel_2")
    print("   - Both panels should show RSI indicators")
    print("   - Chart should match the structure of mmm_QQQ_Analysis_mmm")

    print("\n🧪 TO TEST:")
    print("   Run the MMM module and check if mmm_test now generates:")
    print("   - Panel_1: XLY line chart with RSI indicator")
    print("   - Panel_2: XLY_gap line chart with RSI indicator")

if __name__ == "__main__":
    verify_mmm_fix()