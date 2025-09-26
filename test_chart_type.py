#!/usr/bin/env python3
"""
Quick test script to verify chart_type implementation in SR module
"""

import sys
import pandas as pd
sys.path.append('src')

def test_chart_type_parsing():
    """Test chart_type parsing functionality"""
    print("🧪 Testing chart_type parsing...")

    try:
        # Test CSV reading
        df = pd.read_csv('user_charts_display.csv')
        print("📊 CSV content:")
        print(df.to_string())

        # Test config reader
        from sustainability_ratios.sr_config_reader import read_panels_config

        print("\n🔧 Testing config reader with chart_type...")
        panels_config = read_panels_config('user_charts_display.csv')

        print(f"✅ Config read successfully! {len(panels_config)} panels found")

        # Check if chart_type is in the configuration
        for config_key, config_data in panels_config.items():
            chart_type = config_data.get('chart_type', 'NOT_FOUND')
            print(f"Panel '{config_key}': chart_type = '{chart_type}'")

        print("\n🎯 Chart type implementation test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chart_type_parsing()