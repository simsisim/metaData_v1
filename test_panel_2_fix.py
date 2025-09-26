#!/usr/bin/env python3

"""
Test script to verify Panel_2 data processing and chart_type handling.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from modules
from src.sustainability_ratios.sr_config_reader import parse_panel_config
from src.config import Config
from src.user_defined_data import UserConfiguration

def test_panel_2_processing():
    """Test Panel_2 data processing with chart_type."""

    print("🧪 Testing Panel_2 data processing and chart_type handling...")

    try:
        # Initialize configuration
        config = Config()
        user_config = UserConfiguration()

        # Parse panel configuration
        panel_csv_file = "user_data_sr_panel.csv"
        print(f"📄 Reading panel configuration from: {panel_csv_file}")

        panel_config = parse_panel_config(panel_csv_file)

        print(f"✅ Parsed {len(panel_config)} panel configurations")

        # Print details for each configuration
        for row_idx, row_config in enumerate(panel_config):
            print(f"\n🔍 Row {row_idx + 1}:")
            for config_key, config_data in row_config.items():
                print(f"   Config Key: {config_key}")
                print(f"   panel_name: {config_data.get('panel_name')}")
                print(f"   data_source: {config_data.get('data_source')}")
                print(f"   indicator: {config_data.get('indicator', 'None')}")
                print(f"   chart_type: {config_data.get('chart_type', 'NOT SET')}")
                print(f"   format_type: {config_data.get('format_type', 'unknown')}")
                print(f"   position: {config_data.get('position', 'main')}")

        # Test Panel_2 SPY specifically
        spy_panels = []
        for row_config in panel_config:
            for config_key, config_data in row_config.items():
                if 'SPY' in config_data.get('data_source', '') or 'Panel_2' in config_data.get('panel_name', ''):
                    spy_panels.append((config_key, config_data))

        if spy_panels:
            print(f"\n🎯 Found SPY/Panel_2 configurations:")
            for spy_key, spy_config in spy_panels:
                print(f"   Key: {spy_key}")
                print(f"   Data Source: {spy_config.get('data_source')}")
                chart_type = spy_config.get('chart_type', 'NOT SET')
                if chart_type and chart_type != 'NOT SET':
                    print(f"   Chart Type: {chart_type} ✅")
                else:
                    print(f"   Chart Type: NOT SET ❌")
        else:
            print("⚠️ No SPY/Panel_2 configurations found")

        print(f"\n✅ Panel configuration test completed successfully!")

    except Exception as e:
        print(f"❌ Error testing panel configuration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_panel_2_processing()