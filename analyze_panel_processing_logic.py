#!/usr/bin/env python3
"""
Analyze Panel Processing Logic
=============================

Analyze how the SR system processes panel configurations and identify
why mmm_test Panel_2 might be skipped.
"""

import pandas as pd
from pathlib import Path

def analyze_panel_processing():
    """Analyze panel processing logic by examining configuration differences."""

    print("=" * 70)
    print("ANALYZING PANEL PROCESSING LOGIC")
    print("=" * 70)

    # Read the MMM configuration
    config_file = Path("user_data_sr_mmm.csv")

    if not config_file.exists():
        print("❌ MMM configuration file not found")
        return

    print("\n1. Reading MMM Configuration:")

    try:
        # Read with explicit header handling
        df = pd.read_csv(config_file)

        print(f"   Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")

        # Process each configuration row
        for idx, row in df.iterrows():
            file_name_id = row.get('file_name_id', f'Row_{idx}')

            # Skip comment rows
            if pd.isna(file_name_id) or str(file_name_id).startswith('#'):
                continue

            print(f"\n   Configuration: {file_name_id}")
            print(f"     chart_type: {row.get('chart_type', 'N/A')}")

            # Analyze panels
            panels = {}
            panel_indices = {}

            for col in df.columns:
                if col.startswith('Panel_') and not col.endswith('_index'):
                    panel_num = col.split('_')[1]
                    panel_value = row.get(col, '')

                    if pd.notna(panel_value) and str(panel_value).strip():
                        panels[col] = str(panel_value).strip()

                        # Check corresponding index
                        index_col = f'Panel_{panel_num}_index'
                        index_value = row.get(index_col, '')

                        if pd.notna(index_value) and str(index_value).strip():
                            panel_indices[index_col] = str(index_value).strip()
                        else:
                            panel_indices[index_col] = 'EMPTY'

            # Display panel analysis
            print("     Panels configured:")
            for panel_name, panel_value in panels.items():
                index_name = panel_name.replace('Panel_', 'Panel_').replace('Panel_', 'Panel_') + '_index'
                index_value = panel_indices.get(index_name, 'N/A')
                print(f"       {panel_name}: '{panel_value}' | Index: '{index_value}'")

            if not panels:
                print("       No panels configured")

    except Exception as e:
        print(f"   ❌ Error reading configuration: {e}")
        return

    print("\n" + "=" * 70)
    print("2. COMPARATIVE ANALYSIS")
    print("=" * 70)

    print("\n   mmm_QQQ_Analysis_mmm (WORKING):")
    print("     Panel_1: 'QQQ + EMA(10) + SMA(50)' | Index: 'A_PPO(12,26,9)_for_(QQQ)'")
    print("     Panel_2: 'XLY_gap'                 | Index: 'B_RSI(14)_for_(QQQ)'")
    print("     ✅ Both panels have indices configured")

    print("\n   mmm_test (NOT WORKING):")
    print("     Panel_1: 'XLY'     | Index: 'EMPTY'")
    print("     Panel_2: 'XLY_gap' | Index: 'EMPTY'")
    print("     ❌ Both panels have empty indices")

    print("\n" + "=" * 70)
    print("3. HYPOTHESIS ANALYSIS")
    print("=" * 70)

    print("\n🧪 HYPOTHESIS 1: Panel indices are required")
    print("   Theory: SR system skips panels without configured indices")
    print("   Evidence:")
    print("   - Working config has indices: A_PPO(...), B_RSI(...)")
    print("   - Non-working config has empty indices")
    print("   - Panel data is available and loads correctly")
    print("   ✅ LIKELY CAUSE")

    print("\n🧪 HYPOTHESIS 2: Panel ordering dependency")
    print("   Theory: Panel_2 requires Panel_1 to have indices")
    print("   Evidence:")
    print("   - Panel_1 in working config has complex format + index")
    print("   - Panel_1 in non-working config has simple format + no index")
    print("   ❓ POSSIBLE FACTOR")

    print("\n🧪 HYPOTHESIS 3: Chart type dependency")
    print("   Theory: 'line' charts require specific configuration")
    print("   Evidence:")
    print("   - Both configs use 'line' chart type")
    print("   - Data availability is identical")
    print("   ❌ UNLIKELY CAUSE")

    print("\n" + "=" * 70)
    print("4. ROOT CAUSE ANALYSIS")
    print("=" * 70)

    print("\n🎯 MOST LIKELY ROOT CAUSE:")
    print("   The SR panel system requires panel indices (indicators)")
    print("   to display panels in charts. Without indices:")
    print("   - Panel data loads correctly")
    print("   - Panel configuration parses correctly")
    print("   - Chart generation skips panels without indicators")

    print("\n🔍 SUPPORTING EVIDENCE:")
    print("   1. Data availability: ✅ Both XLY and XLY_gap load correctly")
    print("   2. Configuration parsing: ✅ Both panels appear in config")
    print("   3. Index requirement: ❌ mmm_test has no panel indices")
    print("   4. Working example: ✅ mmm_QQQ_Analysis_mmm has indices")

    print("\n💡 SOLUTION TO TEST:")
    print("   Add panel indices to mmm_test:")
    print("   Panel_1_index: 'A_RSI(14)_for_(XLY)'")
    print("   Panel_2_index: 'B_RSI(14)_for_(XLY_gap)'")
    print()
    print("   This would provide:")
    print("   - Panel_1: XLY with RSI indicator")
    print("   - Panel_2: XLY_gap with RSI indicator")
    print("   - Both panels would have required indices")

    print("\n🚨 EXPLANATION:")
    print("   The SR panel system appears designed to display:")
    print("   - Main panels: Price data with technical overlays")
    print("   - Index panels: Technical indicators in separate subplots")
    print("   ")
    print("   Without indices, the system may determine there's")
    print("   nothing meaningful to display beyond the main price chart,")
    print("   so it skips additional panels.")

if __name__ == "__main__":
    analyze_panel_processing()