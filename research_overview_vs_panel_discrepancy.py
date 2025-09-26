#!/usr/bin/env python3
"""
Research Overview vs Panel Configuration Discrepancy
==================================================

Research why overview module works correctly (respects file names)
but panel module doesn't create panel_TESTTEST charts.
"""

import pandas as pd
from pathlib import Path

def research_overview_vs_panel():
    """Research the discrepancy between overview and panel implementations."""

    print("=" * 80)
    print("RESEARCH: OVERVIEW VS PANEL CONFIGURATION DISCREPANCY")
    print("=" * 80)

    # 1. Compare CSV configurations
    print("\n1. CSV CONFIGURATION COMPARISON:")

    files_to_examine = {
        'Panel': '/home/imagda/_invest2024/python/metaData_v1/user_data_panel.csv',
        'Overview': '/home/imagda/_invest2024/python/metaData_v1/user_data_sr_overview.csv',
        'MMM': '/home/imagda/_invest2024/python/metaData_v1/user_data_sr_mmm.csv'
    }

    configs = {}

    for module_name, file_path in files_to_examine.items():
        file_obj = Path(file_path)
        if file_obj.exists():
            try:
                df = pd.read_csv(file_path)
                configs[module_name] = df

                print(f"\n   {module_name} ({file_obj.name}):")
                print(f"     Rows: {len(df)}")
                print(f"     Columns: {list(df.columns)}")

                # Look for TESTTEST configurations
                testtest_rows = df[df.iloc[:, 0].astype(str).str.contains('TESTTEST', na=False)]
                if len(testtest_rows) > 0:
                    for idx, row in testtest_rows.iterrows():
                        file_name_id = row.iloc[0]
                        chart_type = row.get('chart_type', 'N/A')
                        panel_1 = row.get('Panel_1', 'N/A')
                        panel_2 = row.get('Panel_2', 'N/A')
                        print(f"     TESTTEST config: {file_name_id}")
                        print(f"       chart_type: {chart_type}")
                        print(f"       Panel_1: '{panel_1}'")
                        print(f"       Panel_2: '{panel_2}'")
                else:
                    print(f"     No TESTTEST configurations found")

            except Exception as e:
                print(f"     ❌ Error reading {file_path}: {e}")
        else:
            print(f"     ❌ File not found: {file_path}")

    print("\n" + "=" * 80)
    print("2. IMPLEMENTATION COMPARISON")
    print("=" * 80)

    print("\n🔍 OVERVIEW MODULE IMPLEMENTATION:")
    print("   From overview_charts.py analysis:")
    print("   1. Uses SRProcessor (same as panel)")
    print("   2. Calls parse_panel_config() (same parser)")
    print("   3. Sets processor.panel_configs = panel_configs")
    print("   4. Calls processor.process_all_row_configurations()")
    print("   ✅ Uses SAME charting code, just different CSV input")

    print("\n🔍 PANEL MODULE IMPLEMENTATION:")
    print("   From sr_calculations.py analysis:")
    print("   1. SRProcessor loads user_data_panel.csv automatically")
    print("   2. Uses same parse_panel_config() function")
    print("   3. Uses same process_all_row_configurations()")
    print("   ❓ Should be IDENTICAL to overview module")

    print("\n🔍 MMM MODULE IMPLEMENTATION:")
    print("   From mmm_charts.py analysis:")
    print("   1. Uses SRProcessor (same as others)")
    print("   2. Uses parse_panel_config() (same parser)")
    print("   3. Sets processor.panel_configs = panel_configs")
    print("   4. Calls processor.process_all_row_configurations()")
    print("   ❓ Should be IDENTICAL to overview module")

    print("\n" + "=" * 80)
    print("3. CONFIGURATION ANALYSIS")
    print("=" * 80)

    print("\n📊 TESTTEST CONFIGURATIONS:")

    print("   Panel (user_data_panel.csv):")
    print("     panel_TESTTEST,line,QQQ,SPY,,,,,,'A_PPO(12,26,9)_for_(SPY)',,,,")
    print("     ✅ Configuration looks valid")

    print("\n   Overview (user_data_sr_overview.csv):")
    print("     overview_TESTTEST,line,QQQ,SPY,,,,,,'A_PPO(12,26,9)_for_(SPY)',,,,")
    print("     ✅ Configuration looks valid (nearly identical)")

    print("\n   MMM (user_data_sr_mmm.csv):")
    print("     mmm_test,line,XLY,XLY_gap,,,,,,,,,,")
    print("     ❌ No indices configured")

    print("\n" + "=" * 80)
    print("4. HYPOTHESIS FORMATION")
    print("=" * 80)

    print("\n🤔 POSSIBLE CAUSES FOR PANEL FAILURE:")

    print("\n   HYPOTHESIS 1: File Location Issue")
    print("     - Overview explicitly specifies CSV path")
    print("     - Panel might be looking in wrong location")
    print("     - SR_EB/user_data_panel.csv vs user_data_panel.csv")

    print("\n   HYPOTHESIS 2: Configuration Loading Order")
    print("     - Panel loads during SRProcessor.__init__()")
    print("     - Overview overrides panel_configs after init")
    print("     - Panel might be using stale/cached configuration")

    print("\n   HYPOTHESIS 3: Module Integration Differences")
    print("     - Overview is called as separate module")
    print("     - Panel might be integrated differently")
    print("     - Different execution contexts")

    print("\n   HYPOTHESIS 4: File Naming Convention")
    print("     - Overview uses overview_TESTTEST (module_name prefix)")
    print("     - Panel uses panel_TESTTEST (module_name prefix)")
    print("     - System might expect different naming patterns")

    print("\n" + "=" * 80)
    print("5. CRITICAL QUESTIONS TO INVESTIGATE")
    print("=" * 80)

    print("\n❓ KEY QUESTIONS:")
    print("   1. WHERE does panel configuration actually get loaded?")
    print("   2. WHICH CSV file is panel module actually reading?")
    print("   3. WHY does overview work but panel doesn't with identical code?")
    print("   4. IS there a different execution path for regular panel vs overview?")
    print("   5. ARE there multiple user_data_panel.csv files in different locations?")

    print("\n🎯 MOST LIKELY ISSUE:")
    print("   The panel module is NOT reading from the expected CSV file.")
    print("   Overview explicitly specifies: user_data_sr_overview.csv")
    print("   Panel might be reading from: SR_EB/user_data_panel.csv (different location)")
    print("   OR there's a file location/naming discrepancy")

    print("\n🔍 TO INVESTIGATE:")
    print("   1. Check if there's a user_data_panel.csv in SR_EB directory")
    print("   2. Trace actual CSV file path in panel processing")
    print("   3. Compare file resolution logic between overview and panel")
    print("   4. Check if panel is using cached/stale configuration")

if __name__ == "__main__":
    research_overview_vs_panel()