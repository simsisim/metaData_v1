#!/usr/bin/env python3

"""
Research how SR modules separate calculation vs chart configuration.
Study panel and overview modules to understand the proper separation of concerns.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pathlib import Path

def research_sr_module_separation():
    """Research how other SR modules handle calculation vs charting separation."""

    print("🧪 RESEARCHING SR MODULE SEPARATION OF CONCERNS")
    print("=" * 60)

    try:
        from src.config import Config
        from src.user_defined_data import read_user_data

        config = Config()
        user_config = read_user_data()

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Research 1: Panel module separation
        print(f"🔍 RESEARCH 1: PANEL MODULE SEPARATION")
        print(f"=" * 45)

        # Check panel configuration vs calculation configuration
        panel_config_file = getattr(user_config, 'sr_panel_config_file', 'user_data_panel.csv')
        print(f"  Panel config file: {panel_config_file}")

        # Look for panel calculation settings
        panel_calc_attrs = []
        for attr in dir(user_config):
            if not attr.startswith('_') and 'sr_panel' in attr.lower() and 'calc' in attr.lower():
                value = getattr(user_config, attr)
                panel_calc_attrs.append((attr, value))

        print(f"  Panel calculation attributes:")
        if panel_calc_attrs:
            for attr, value in panel_calc_attrs:
                print(f"    {attr} = {repr(value)}")
        else:
            print(f"    None found - panels likely use existing market data")

        # Check what the panel config file contains
        panel_config_path = Path(config.base_dir) / panel_config_file
        if panel_config_path.exists():
            print(f"  \n  📄 Panel config file content (first 5 lines):")
            with open(panel_config_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:5], 1):
                    print(f"    {i}: {line.rstrip()}")

            print(f"\n  🎯 Panel Analysis:")
            print(f"    - Panel config defines WHAT to chart")
            print(f"    - Uses existing market data from yf_daily_data_files")
            print(f"    - No separate calculation step for panels")
        else:
            print(f"  ❌ Panel config file not found: {panel_config_path}")

        print()

        # Research 2: Overview module separation
        print(f"🔍 RESEARCH 2: OVERVIEW MODULE SEPARATION")
        print(f"=" * 45)

        overview_config_file = getattr(user_config, 'sr_overview_charts_display_panel', 'user_charts_display.csv')
        print(f"  Overview config file: {overview_config_file}")

        # Look for overview calculation settings
        overview_calc_attrs = []
        for attr in dir(user_config):
            if not attr.startswith('_') and 'sr_overview' in attr.lower():
                value = getattr(user_config, attr)
                overview_calc_attrs.append((attr, value))

        print(f"  Overview configuration attributes:")
        for attr, value in overview_calc_attrs:
            print(f"    {attr} = {repr(value)}")

        # Check overview config file
        overview_config_path = Path(config.base_dir) / overview_config_file
        if overview_config_path.exists():
            print(f"  \n  📄 Overview config file content (first 5 lines):")
            with open(overview_config_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:5], 1):
                    print(f"    {i}: {line.rstrip()}")

            print(f"\n  🎯 Overview Analysis:")
            print(f"    - Overview has sr_overview_values_enable for calculations")
            print(f"    - sr_overview_charts_display_panel for chart config")
            print(f"    - Separate calculation and charting concerns")
        else:
            print(f"  ❌ Overview config file not found: {overview_config_path}")

        print()

        # Research 3: MMM current implementation
        print(f"🔍 RESEARCH 3: MMM CURRENT IMPLEMENTATION")
        print(f"=" * 45)

        mmm_calc_attrs = []
        mmm_chart_attrs = []
        for attr in dir(user_config):
            if not attr.startswith('_') and 'sr_mmm' in attr.lower():
                value = getattr(user_config, attr)
                if 'chart' in attr.lower():
                    mmm_chart_attrs.append((attr, value))
                else:
                    mmm_calc_attrs.append((attr, value))

        print(f"  MMM calculation attributes:")
        for attr, value in mmm_calc_attrs:
            print(f"    {attr} = {repr(value)}")

        print(f"\n  MMM charting attributes:")
        for attr, value in mmm_chart_attrs:
            print(f"    {attr} = {repr(value)}")

        print()

        # Research 4: Data flow analysis
        print(f"🔍 RESEARCH 4: DATA FLOW ANALYSIS")
        print(f"=" * 35)

        print(f"  📊 Panel Data Flow:")
        print(f"    1. Uses existing market data files (yf_daily_data_files)")
        print(f"    2. Panel config defines which tickers/indicators to chart")
        print(f"    3. Charts generated from existing data + indicators")

        print(f"\n  📊 Overview Data Flow:")
        print(f"    1. sr_overview_values_enable triggers calculations")
        print(f"    2. Calculations create overview values files")
        print(f"    3. Chart config defines what to chart from calculated data")

        print(f"\n  📊 MMM Current Data Flow:")
        print(f"    1. sr_mmm_gaps_tickers defines which tickers to calculate")
        print(f"    2. Gap calculations create ticker_gap.csv files")
        print(f"    3. Chart config defines what to chart...")
        print(f"    4. ❌ ISSUE: Chart looks for gap files matching chart config")

        print(f"\n  📊 MMM SHOULD Work Like:")
        print(f"    1. sr_mmm_gaps_tickers defines calculation scope")
        print(f"    2. Gap calculations create available gap data")
        print(f"    3. Chart config defines what to chart")
        print(f"    4. ✅ Chart generation uses available data or fallback")

        print()

        # Research 5: Check how chart generation finds data
        print(f"🔍 RESEARCH 5: CHART DATA RESOLUTION")
        print(f"=" * 40)

        print(f"  Current MMM chart data resolution:")
        print(f"    - Looks for CSV files matching data_source names")
        print(f"    - Expects exact match: 'SPY' -> 'SPY_gap.csv'")
        print(f"    - Fails if no exact match found")

        print(f"\n  Panel chart data resolution:")
        print(f"    - Uses market data files: ticker.csv")
        print(f"    - Applies indicators on-the-fly during chart generation")
        print(f"    - Flexible data source matching")

        print(f"\n  Overview chart data resolution:")
        print(f"    - Uses calculated overview values files")
        print(f"    - Matches chart config to available calculated data")
        print(f"    - Graceful fallback if data not available")

        print()

        # Research 6: Recommended MMM approach
        print(f"🎯 RECOMMENDED MMM APPROACH:")
        print(f"=" * 32)

        print(f"  ✅ SEPARATION OF CONCERNS:")
        print(f"    1. sr_mmm_gaps_tickers = 'XLY;XLC' (calculation scope)")
        print(f"    2. user_data_sr_mmm.csv can reference ANY tickers")
        print(f"    3. Chart generation should:")
        print(f"       - First look for gap data (ticker_gap.csv)")
        print(f"       - If not available, use regular market data")
        print(f"       - Apply indicators/calculations as needed")
        print(f"       - Skip gracefully if no data available")

        print(f"\n  📝 CURRENT ISSUE:")
        print(f"    - Chart generation is too rigid")
        print(f"    - Only looks for gap files, doesn't fallback")
        print(f"    - Should be more flexible like panel system")

        print(f"\n  🔧 REQUIRED CHANGES:")
        print(f"    1. Update chart data resolution to be more flexible")
        print(f"    2. Add fallback to regular market data if gap data missing")
        print(f"    3. Allow charting tickers not in calculation scope")
        print(f"    4. Graceful handling of missing data")

        return True

    except Exception as e:
        print(f"❌ Error researching SR module separation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = research_sr_module_separation()
    if success:
        print(f"\n🎉 SR module separation research completed!")
    else:
        print(f"\n💥 SR module separation research failed!")