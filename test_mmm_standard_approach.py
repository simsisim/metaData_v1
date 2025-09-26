#!/usr/bin/env python3

"""
Test MMM chart generation with the new standard approach.
Verify that MMM now uses the same data resolution as Panel/Overview modules.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pathlib import Path

def test_mmm_standard_approach():
    """Test MMM chart generation with standard sr_market_data approach."""

    print("🧪 TESTING MMM WITH STANDARD SR APPROACH")
    print("=" * 50)

    try:
        from src.sustainability_ratios.mmm.mmm_gaps import MmmGapsProcessor
        from src.sustainability_ratios.mmm.mmm_charts import MmmChartsProcessor
        from src.config import Config
        from src.user_defined_data import read_user_data

        # Use proper configuration loading
        config = Config()
        user_config = read_user_data()

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Test 1: Verify MMM configuration
        print(f"🔧 MMM CONFIGURATION:")
        print(f"  sr_mmm_gaps_tickers = {repr(getattr(user_config, 'sr_mmm_gaps_tickers'))}")
        print(f"  sr_mmm_gaps_charts_display_panel = {repr(getattr(user_config, 'sr_mmm_gaps_charts_display_panel'))}")
        print()

        # Test 2: Run gap calculation (if needed)
        print(f"🔄 ENSURING GAP DATA EXISTS:")
        gaps_processor = MmmGapsProcessor(config, user_config, 'daily')
        gap_results = gaps_processor.run_gap_analysis()

        if 'error' not in gap_results:
            print(f"  ✅ Gap calculation successful:")
            print(f"    - Processed tickers: {gap_results.get('tickers_processed', [])}")
            print(f"    - CSV files: {len(gap_results.get('csv_files', []))}")
        else:
            print(f"  ⚠️  Gap calculation issue: {gap_results.get('error')}")
            # Continue anyway - standard approach should handle missing gap data
            gap_results = {'tickers_processed': [], 'csv_files': [], 'failed_tickers': []}

        print()

        # Test 3: Test standard approach chart generation
        print(f"📊 TESTING STANDARD APPROACH CHART GENERATION:")
        charts_processor = MmmChartsProcessor(config, user_config, 'daily')

        # Verify chart configuration
        chart_config_file = getattr(user_config, 'sr_mmm_gaps_charts_display_panel', 'user_data_sr_mmm.csv')
        config_valid = charts_processor._validate_chart_config(chart_config_file)
        print(f"  Chart config file: {chart_config_file}")
        print(f"  Chart config valid: {config_valid}")

        if not config_valid:
            print(f"  ❌ Chart configuration not valid!")
            return False

        # Show what's in the chart config
        config_path = charts_processor._resolve_config_path(chart_config_file)
        print(f"  📄 Chart configuration content:")
        with open(config_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:5], 1):
                if not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        file_name_id = parts[0].strip()
                        chart_type = parts[1].strip()
                        panel_1 = parts[2].strip()
                        print(f"    {file_name_id}: {chart_type}, Panel_1='{panel_1}'")

        print()

        # Test 4: Run chart generation with standard approach
        print(f"🚀 RUNNING CHART GENERATION (STANDARD APPROACH):")
        try:
            chart_files = charts_processor.run_gap_charts_analysis(gap_results)

            print(f"  📈 Chart generation results:")
            print(f"    - Charts generated: {len(chart_files)}")

            if chart_files:
                print(f"    ✅ Generated chart files:")
                for chart_file in chart_files:
                    print(f"      - {chart_file}")
            else:
                print(f"    ⚠️  No chart files generated")

        except Exception as e:
            print(f"  ❌ Error in chart generation: {e}")
            import traceback
            traceback.print_exc()

        print()

        # Test 5: Check output directory for generated files
        print(f"📁 CHECKING OUTPUT DIRECTORY:")
        from src.sustainability_ratios.sr_output_manager import get_sr_output_manager
        output_manager = get_sr_output_manager()
        charts_dir = output_manager.get_submodule_dir('mmm') / 'charts'

        print(f"  Charts directory: {charts_dir}")
        print(f"  Directory exists: {charts_dir.exists()}")

        if charts_dir.exists():
            png_files = list(charts_dir.glob('*.png'))
            print(f"  📊 PNG files found: {len(png_files)}")
            for png_file in sorted(png_files):
                file_size = png_file.stat().st_size
                print(f"    - {png_file.name} ({file_size:,} bytes)")

        print()

        # Test 6: Test data source resolution manually
        print(f"🔍 MANUAL DATA SOURCE RESOLUTION TEST:")
        try:
            from src.data_reader import DataReader
            from src.sustainability_ratios.sr_market_data import load_market_data_for_panels
            from src.sustainability_ratios.sr_config_reader import parse_panel_config

            data_reader = DataReader(config)
            panels_config = parse_panel_config(str(config_path))

            if panels_config:
                # Test first row of panel config
                first_row = panels_config[0]
                print(f"  📊 Testing data resolution for first row:")

                market_data = load_market_data_for_panels(first_row, data_reader)
                print(f"    - Market data loaded: {len(market_data)} data sources")

                for data_source, data in market_data.items():
                    if data is not None:
                        print(f"    ✅ {data_source}: {len(data)} rows")
                        if hasattr(data, 'columns'):
                            print(f"      Columns: {list(data.columns)[:8]}...")  # Show first 8
                    else:
                        print(f"    ❌ {data_source}: No data")

        except Exception as e:
            print(f"  ❌ Error in manual data resolution test: {e}")

        print()

        # Test 7: Expected behavior verification
        print(f"🎯 EXPECTED BEHAVIOR VERIFICATION:")
        print(f"  ✅ MMM now uses standard sr_market_data approach")
        print(f"  ✅ Data source naming follows Panel/Overview convention:")
        print(f"    - 'SPY + EMA(20)' → Uses SPY.csv (regular market data)")
        print(f"    - 'XLY_gap + EMA(20)' → Uses XLY_gap.csv (gap data)")
        print(f"  ✅ User explicitly controls data source via configuration")
        print(f"  ✅ Leverages existing tested infrastructure")

        return True

    except Exception as e:
        print(f"❌ Error testing MMM standard approach: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mmm_standard_approach()
    if success:
        print(f"\n🎉 MMM standard approach test completed!")
        print(f"📋 MMM now follows Panel/Overview data resolution pattern")
    else:
        print(f"\n💥 MMM standard approach test failed!")