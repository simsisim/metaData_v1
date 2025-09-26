#!/usr/bin/env python3

"""
Research why XLY_gap is not displaying in MMM Panel_2.
The configuration shows 'XLY_gap' in Panel_2 but the chart doesn't display it.
This suggests a gap data source resolution issue similar to today's Panel/Overview problems.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path

def research_xly_gap_missing():
    """Research why XLY_gap is missing from MMM Panel_2 display."""

    print("🔍 RESEARCH: WHY XLY_GAP IS MISSING FROM MMM PANEL_2")
    print("=" * 55)

    try:
        from src.config import Config
        from src.user_defined_data import read_user_data
        from src.data_reader import DataReader

        config = Config()
        user_config = read_user_data()
        data_reader = DataReader(config)

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Research 1: MMM Configuration Analysis
        print(f"🔍 RESEARCH 1: MMM CONFIGURATION ANALYSIS")
        print(f"=" * 44)

        config_path = Path(config.base_dir) / 'user_data_sr_mmm.csv'
        print(f"  📄 MMM config file: {config_path}")

        with open(config_path, 'r') as f:
            lines = f.readlines()

        print(f"  📋 Configuration content:")
        for i, line in enumerate(lines, 1):
            if not line.startswith('#'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    file_name_id = parts[0]
                    chart_type = parts[1]
                    panel_1 = parts[2]
                    panel_2 = parts[3] if parts[3] else 'EMPTY'
                    print(f"    Row {i}: {file_name_id}")
                    print(f"      Chart type: {chart_type}")
                    print(f"      Panel_1: '{panel_1}'")
                    print(f"      Panel_2: '{panel_2}'")

        print()

        # Research 2: Test XLY_gap data availability
        print(f"🔍 RESEARCH 2: XLY_GAP DATA AVAILABILITY TEST")
        print(f"=" * 46)

        print(f"  🧪 Testing direct data_reader access:")
        try:
            xly_gap_data = data_reader.read_stock_data('XLY_gap')
            if xly_gap_data is not None and not xly_gap_data.empty:
                print(f"    ✅ XLY_gap data found: {len(xly_gap_data)} rows")
                print(f"    📊 Columns: {list(xly_gap_data.columns)[:8]}...")

                # Check for gap-specific columns
                gap_columns = [col for col in xly_gap_data.columns if 'gap' in col.lower()]
                if gap_columns:
                    print(f"    📈 Gap columns: {gap_columns}")
                else:
                    print(f"    ⚠️  No gap-specific columns found")
            else:
                print(f"    ❌ XLY_gap data NOT found")
        except Exception as e:
            print(f"    ❌ Error reading XLY_gap: {e}")

        print()

        # Research 3: Check where gap files are stored
        print(f"🔍 RESEARCH 3: GAP FILES LOCATION CHECK")
        print(f"=" * 40)

        # Check data directory
        data_dir = Path(config.base_dir) / user_config.yf_daily_data_files
        print(f"  📂 Data directory: {data_dir}")
        print(f"  Directory exists: {data_dir.exists()}")

        if data_dir.exists():
            xly_gap_files = list(data_dir.glob('XLY_gap*'))
            print(f"  📄 XLY_gap files found: {len(xly_gap_files)}")
            for gap_file in xly_gap_files:
                print(f"    - {gap_file.name}")

        # Check MMM output directory
        mmm_output = getattr(user_config, 'sr_mmm_gaps_values_output_folder_daily', '')
        if mmm_output:
            mmm_output_dir = Path(config.base_dir) / mmm_output
            print(f"  📂 MMM output directory: {mmm_output_dir}")
            print(f"  Directory exists: {mmm_output_dir.exists()}")

            if mmm_output_dir.exists():
                xly_gap_files = list(mmm_output_dir.glob('XLY_gap*'))
                print(f"  📄 XLY_gap files in MMM output: {len(xly_gap_files)}")
                for gap_file in xly_gap_files:
                    print(f"    - {gap_file.name}")

        print()

        # Research 4: Test sr_market_data resolution
        print(f"🔍 RESEARCH 4: SR_MARKET_DATA RESOLUTION TEST")
        print(f"=" * 47)

        from src.sustainability_ratios.sr_config_reader import parse_panel_config
        from src.sustainability_ratios.sr_market_data import load_market_data_for_panels

        panels_config = parse_panel_config(str(config_path))
        if panels_config and len(panels_config) >= 2:
            # Test the second row which should have XLY_gap
            second_row = panels_config[1]
            print(f"  📊 Testing second row configuration:")
            for panel_key, panel_info in second_row.items():
                print(f"    {panel_key}: '{panel_info['data_source']}'")

            print(f"  🧪 Loading market data for second row...")
            market_data = load_market_data_for_panels(second_row, data_reader)

            print(f"  📈 Market data results:")
            for data_source, data in market_data.items():
                if data is not None and not data.empty:
                    print(f"    ✅ {data_source}: SUCCESS ({len(data)} rows)")
                else:
                    print(f"    ❌ {data_source}: FAILED")

        print()

        # Research 5: Compare with working data sources
        print(f"🔍 RESEARCH 5: COMPARE WITH WORKING DATA SOURCES")
        print(f"=" * 50)

        working_tickers = ['QQQ', 'SPY', 'IWM']
        problematic_tickers = ['XLY_gap', 'XLC_gap']

        print(f"  🧪 Testing working vs problematic tickers:")

        for ticker in working_tickers + problematic_tickers:
            try:
                data = data_reader.read_stock_data(ticker)
                if data is not None and not data.empty:
                    print(f"    ✅ {ticker}: Found ({len(data)} rows)")
                else:
                    print(f"    ❌ {ticker}: Not found")
            except Exception as e:
                print(f"    ❌ {ticker}: Error - {e}")

        print()

        # Research 6: Check recent gap calculation
        print(f"🔍 RESEARCH 6: RECENT GAP CALCULATION CHECK")
        print(f"=" * 43)

        print(f"  🔄 Testing MMM gap calculation to ensure data exists...")
        try:
            from src.sustainability_ratios.mmm.mmm_gaps import MmmGapsProcessor
            gaps_processor = MmmGapsProcessor(config, user_config, 'daily')
            gap_results = gaps_processor.run_gap_analysis()

            if 'error' not in gap_results:
                print(f"    ✅ Gap calculation successful:")
                print(f"      Processed tickers: {gap_results.get('tickers_processed', [])}")
                print(f"      CSV files created: {len(gap_results.get('csv_files', []))}")

                for csv_file in gap_results.get('csv_files', []):
                    file_path = Path(csv_file)
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        print(f"        - {file_path.name} ({file_size:,} bytes)")
            else:
                print(f"    ❌ Gap calculation failed: {gap_results.get('error')}")

        except Exception as e:
            print(f"    ❌ Error in gap calculation test: {e}")

        print()

        # Research 7: Analysis and hypothesis
        print(f"🎯 RESEARCH 7: ANALYSIS AND HYPOTHESIS")
        print(f"=" * 38)

        print(f"  📋 Potential Causes:")
        print(f"    1. Gap files created in different directory than data_reader searches")
        print(f"    2. Gap files not created or corrupted")
        print(f"    3. sr_market_data doesn't handle '_gap' suffix correctly")
        print(f"    4. Panel rendering skips empty/failed data sources")
        print(f"    5. Configuration parsing issue with Panel_2 position")

        print(f"  🔧 Investigation Priority:")
        print(f"    1. HIGH: Verify gap files exist and are readable")
        print(f"    2. HIGH: Test data_reader path resolution for gap files")
        print(f"    3. MEDIUM: Test sr_market_data with gap data sources")
        print(f"    4. MEDIUM: Check panel rendering behavior with missing data")

        return True

    except Exception as e:
        print(f"❌ Error researching XLY_gap missing issue: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = research_xly_gap_missing()
    if success:
        print(f"\n🎉 XLY_gap missing research completed!")
        print(f"📋 Next: Identify specific cause and plan solution")
    else:
        print(f"\n💥 XLY_gap missing research failed!"))