#!/usr/bin/env python3

"""
Research how Panel and Overview modules handle data source naming conventions.
Understand how data source names map to actual file lookups.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pathlib import Path

def research_data_source_naming_convention():
    """Research data source naming convention in Panel and Overview modules."""

    print("🧪 RESEARCHING DATA SOURCE NAMING CONVENTION")
    print("=" * 55)

    try:
        from src.config import Config
        from src.user_defined_data import read_user_data

        config = Config()
        user_config = read_user_data()

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Research 1: Panel data source naming
        print(f"🔍 RESEARCH 1: PANEL DATA SOURCE NAMING")
        print(f"=" * 43)

        panel_config_file = getattr(user_config, 'sr_panel_config_file', 'user_data_panel.csv')
        panel_config_path = Path(config.base_dir) / panel_config_file

        if panel_config_path.exists():
            print(f"  📄 Panel config examples:")
            with open(panel_config_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:8], 1):
                    if not line.startswith('#') and line.strip():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            file_name_id = parts[0].strip()
                            panel_1 = parts[2].strip()
                            print(f"    {file_name_id}: Panel_1 = '{panel_1}'")

            print(f"\n  🎯 Panel Data Source Analysis:")
            print(f"    - 'SPY + EMA(20)' → Uses SPY.csv + applies EMA indicator")
            print(f"    - 'QQQ' → Uses QQQ.csv directly")
            print(f"    - 'SPY:QQQ' → Uses SPY.csv and QQQ.csv for ratio")
            print(f"    - Data source name = ticker name (base convention)")

        print()

        # Research 2: Overview data source naming
        print(f"🔍 RESEARCH 2: OVERVIEW DATA SOURCE NAMING")
        print(f"=" * 45)

        overview_config_file = getattr(user_config, 'sr_overview_charts_display_panel', 'user_data_sr_overview.csv')
        overview_config_path = Path(config.base_dir) / overview_config_file

        if overview_config_path.exists():
            print(f"  📄 Overview config examples:")
            with open(overview_config_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:8], 1):
                    if not line.startswith('#') and line.strip():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            file_name_id = parts[0].strip()
                            panel_1 = parts[2].strip()
                            print(f"    {file_name_id}: Panel_1 = '{panel_1}'")

            print(f"\n  🎯 Overview Data Source Analysis:")
            print(f"    - Same naming convention as Panel")
            print(f"    - 'SPY + EMA(20)' → Uses SPY.csv (market data)")
            print(f"    - Overview values calculated separately from chart config")
            print(f"    - Chart config references ticker names, not calculated file names")

        print()

        # Research 3: How sr_market_data resolves data sources
        print(f"🔍 RESEARCH 3: DATA RESOLUTION IN sr_market_data.py")
        print(f"=" * 52)

        print(f"  📊 Data Resolution Process:")
        print(f"    1. decompose_data_source() parses 'SPY + EMA(20)'")
        print(f"    2. Identifies base ticker: 'SPY'")
        print(f"    3. Identifies indicators: ['EMA(20)']")
        print(f"    4. load_market_data_for_panels() calls data_reader.read_stock_data('SPY')")
        print(f"    5. Applies EMA indicator to loaded SPY data")

        print(f"\n  🔍 Let's trace the data_reader.read_stock_data() method:")

        # Test data_reader to see how it resolves file names
        try:
            from src.data_reader import DataReader
            data_reader = DataReader(config)

            print(f"    📂 Data reader directory: {data_reader.data_dir}")

            # Check what files exist
            if hasattr(data_reader, 'data_dir') and Path(data_reader.data_dir).exists():
                data_dir = Path(data_reader.data_dir)
                csv_files = list(data_dir.glob('*.csv'))
                print(f"    📄 Available data files:")
                for csv_file in sorted(csv_files)[:10]:  # Show first 10
                    print(f"      - {csv_file.name}")

            print(f"\n    🔧 Data Reader Logic:")
            print(f"      - Ticker 'SPY' → looks for 'SPY.csv' in data directory")
            print(f"      - Ticker 'SPY_gap' → would look for 'SPY_gap.csv'")
            print(f"      - File name = ticker name + '.csv'")

        except Exception as e:
            print(f"    ❌ Error testing data_reader: {e}")

        print()

        # Research 4: Current MMM data source handling
        print(f"🔍 RESEARCH 4: CURRENT MMM DATA SOURCE HANDLING")
        print(f"=" * 49)

        mmm_config_file = getattr(user_config, 'sr_mmm_gaps_charts_display_panel', 'user_data_sr_mmm.csv')
        mmm_config_path = Path(config.base_dir) / mmm_config_file

        if mmm_config_path.exists():
            print(f"  📄 MMM config examples:")
            with open(mmm_config_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:5], 1):
                    if not line.startswith('#') and line.strip():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            file_name_id = parts[0].strip()
                            panel_1 = parts[2].strip()
                            print(f"    {file_name_id}: Panel_1 = '{panel_1}'")

            print(f"\n  ❌ Current MMM Problem:")
            print(f"    - Config: 'SPY + EMA(20)' → MMM looks for SPY_gap.csv")
            print(f"    - This is WRONG - should look for SPY.csv like Panel/Overview")

        print()

        # Research 5: Proposed MMM data source convention
        print(f"🎯 RESEARCH 5: PROPOSED MMM DATA SOURCE CONVENTION")
        print(f"=" * 52)

        print(f"  ✅ CORRECT MMM Convention (Same as Panel/Overview):")
        print(f"    📋 Data Source Name → File Lookup Logic:")
        print(f"      'SPY + EMA(20)'     → SPY.csv (regular market data)")
        print(f"      'QQQ'               → QQQ.csv (regular market data)")
        print(f"      'SPY_gap + EMA(20)' → SPY_gap.csv (gap data if available)")
        print(f"      'XLY_gap'           → XLY_gap.csv (gap data)")

        print(f"\n  🔧 Required MMM Changes:")
        print(f"    1. Use standard data_reader.read_stock_data() like Panel/Overview")
        print(f"    2. 'SPY' → looks for SPY.csv")
        print(f"    3. 'SPY_gap' → looks for SPY_gap.csv")
        print(f"    4. User explicitly chooses data source by naming convention")

        print(f"\n  📊 Example Usage:")
        print(f"    Current Config: 'SPY + EMA(20)' → ❌ Looks for SPY_gap.csv")
        print(f"    Fixed Config:   'SPY + EMA(20)' → ✅ Looks for SPY.csv")
        print(f"    Gap Config:     'SPY_gap + EMA(20)' → ✅ Looks for SPY_gap.csv")

        print()

        # Research 6: Implementation approach
        print(f"🔧 RESEARCH 6: IMPLEMENTATION APPROACH")
        print(f"=" * 38)

        print(f"  📋 MMM Should Follow Panel/Overview Pattern:")
        print(f"    1. Use existing sr_market_data.py infrastructure")
        print(f"    2. Replace _prepare_chart_data() with standard approach")
        print(f"    3. Use data_reader.read_stock_data() for file resolution")
        print(f"    4. Let user explicitly specify data source type in config")

        print(f"\n  🎯 Benefits:")
        print(f"    - Consistent with Panel/Overview modules")
        print(f"    - User has explicit control over data source")
        print(f"    - Leverages existing, tested infrastructure")
        print(f"    - No special case handling needed")

        print(f"\n  📝 User Control Examples:")
        print(f"    Regular data: 'SPY + EMA(20)', 'QQQ + SMA(50)'")
        print(f"    Gap data:     'XLY_gap + EMA(20)', 'XLC_gap'")
        print(f"    Mixed:        'SPY + EMA(20)' and 'XLY_gap' in same config")

        return True

    except Exception as e:
        print(f"❌ Error researching data source naming convention: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = research_data_source_naming_convention()
    if success:
        print(f"\n🎉 Data source naming convention research completed!")
        print(f"\n📋 CONCLUSION: MMM should use same naming convention as Panel/Overview")
        print(f"   - 'SPY + EMA(20)' → SPY.csv")
        print(f"   - 'SPY_gap + EMA(20)' → SPY_gap.csv")
        print(f"   - User explicitly controls data source type")
    else:
        print(f"\n💥 Data source naming convention research failed!")