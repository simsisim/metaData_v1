#!/usr/bin/env python3
"""
Test gaps_display.csv Parsing
=============================

Test parsing of the gaps_display.csv configuration file.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_gaps_display_parsing():
    """Test parsing of gaps_display.csv file."""
    print("🧪 Testing gaps_display.csv Parsing")
    print("=" * 50)

    try:
        from sustainability_ratios.sr_config_reader import parse_panel_config

        config_file = "gaps_display.csv"
        print(f"📊 Parsing configuration file: {config_file}")

        # Test the parsing
        panels_config = parse_panel_config(config_file)

        print(f"   Parsing result type: {type(panels_config)}")
        print(f"   Number of configurations: {len(panels_config) if panels_config else 0}")

        if panels_config:
            print(f"\n🔍 Configuration Details:")

            for idx, config in enumerate(panels_config):
                print(f"\n   Configuration {idx + 1}:")
                print(f"   Type: {type(config)}")
                print(f"   Keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")

                if isinstance(config, dict):
                    for panel_key, panel_info in config.items():
                        if isinstance(panel_info, dict):
                            print(f"      Panel: {panel_key}")
                            print(f"      Data source: {panel_info.get('data_source', 'N/A')}")
                            print(f"      Indicator: {panel_info.get('indicator', 'N/A')}")
                            print(f"      Timeframe: {panel_info.get('timeframe', 'N/A')}")
                        else:
                            print(f"      {panel_key}: {panel_info}")

        else:
            print("❌ No panels configuration returned")

        return panels_config is not None and len(panels_config) > 0

    except Exception as e:
        print(f"❌ Error parsing gaps_display.csv: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gaps_display_parsing()
    if success:
        print(f"\n✅ SUCCESS: gaps_display.csv parsing working!")
    else:
        print(f"\n❌ FAILURE: gaps_display.csv parsing issues")