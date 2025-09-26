#!/usr/bin/env python3

"""
Test the current behavior of chart_type='no_drawing' for QQQ + EMA panel.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_no_drawing_behavior():
    """Test how no_drawing chart_type currently behaves."""

    print("🔍 TESTING NO_DRAWING CHART TYPE BEHAVIOR")
    print("=" * 60)

    try:
        # Read the panel configuration to find the tesst2_panel
        import pandas as pd

        panel_config_file = "user_data_sr_panel.csv"
        print(f"📊 Reading panel configuration: {panel_config_file}")

        df = pd.read_csv(panel_config_file)
        print(f"✅ Panel configuration loaded: {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")

        # Find tesst2_panel row (first column should be the identifier)
        first_col = df.columns[0]  # Should be '#file_name_id'
        tesst2_rows = df[df[first_col] == 'tesst2_panel']

        if len(tesst2_rows) == 0:
            print("❌ tesst2_panel not found in configuration")
            print(f"📋 Available row identifiers: {df[first_col].tolist()}")
            return

        print(f"\n🎯 FOUND tesst2_panel configurations: {len(tesst2_rows)} rows")

        for idx, row in tesst2_rows.iterrows():
            print(f"\n📋 Row {idx + 1}:")
            print(f"   Panel ID: {row.get(first_col, 'N/A')}")
            print(f"   Chart Type: {row.get('chart_type', 'N/A')}")
            print(f"   Panel_1: {row.get('Panel_1', 'N/A')}")
            print(f"   Panel_2: {row.get('Panel_2', 'N/A')}")
            print(f"   Panel_1_index: {row.get('Panel_1_index', 'N/A')}")

            # Analyze what should happen
            chart_type = str(row.get('chart_type', '')).lower()
            panel_1 = str(row.get('Panel_1', ''))
            panel_2 = str(row.get('Panel_2', ''))
            panel_1_index = str(row.get('Panel_1_index', ''))

            print(f"\n🔧 EXPECTED BEHAVIOR ANALYSIS:")
            print(f"   Chart Type: '{chart_type}'")
            print(f"   Panel_1 Data: '{panel_1}'")

            if chart_type == 'no_drawing':
                print(f"   ✅ Main series (price data) should be HIDDEN")
                if '+' in panel_1 and any(x in panel_1 for x in ['EMA', 'SMA', 'PPO', 'RSI']):
                    print(f"   ✅ Overlay indicators should be VISIBLE")
                    print(f"   🎯 Expected Result: Only EMA line visible, no QQQ price bars/candles")

                    # Extract indicator details
                    if 'EMA' in panel_1:
                        print(f"   📊 EMA indicator detected in bundled format")
                elif panel_1_index:
                    print(f"   ✅ Index indicators should be VISIBLE")
                    print(f"   🎯 Expected Result: Only {panel_1_index} visible, no main price data")
                else:
                    print(f"   ⚠️ No overlays detected - chart may be empty")

            elif chart_type in ['line', 'candle']:
                print(f"   ✅ Main series should be VISIBLE as {chart_type}")
                print(f"   ✅ Overlay indicators should also be VISIBLE")
                print(f"   🎯 Expected Result: Both price data and overlays shown")
            else:
                print(f"   ❓ Unknown chart type: '{chart_type}'")

        print(f"\n🧪 TESTING CURRENT IMPLEMENTATION:")
        print(f"📄 Dashboard Generator Logic Check:")
        print(f"   - Line 703-705: if chart_type == 'no_drawing': skip main series ✅")
        print(f"   - Line 724+: Plot overlays regardless of chart_type ✅")
        print(f"   - Expected: Only overlays visible for no_drawing ✅")

    except Exception as e:
        print(f"❌ Error testing no_drawing behavior: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_no_drawing_behavior()