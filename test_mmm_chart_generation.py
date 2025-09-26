#!/usr/bin/env python3

"""
Test MMM chart generation process to find why charts are not being generated.
Simulate the complete MMM workflow from gap calculation to chart generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pathlib import Path

def test_mmm_chart_generation():
    """Test complete MMM chart generation workflow."""

    print("🧪 TESTING MMM CHART GENERATION PROCESS")
    print("=" * 50)

    try:
        from src.sustainability_ratios.mmm.mmm_gaps import MmmGapsProcessor
        from src.sustainability_ratios.mmm.mmm_charts import MmmChartsProcessor
        from src.config import Config
        from src.user_defined_data import UserConfiguration

        # Create test configuration
        config = Config()
        user_config = UserConfiguration()

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Test 1: Check if gap data files exist
        print(f"🔍 CHECKING MMM GAP DATA FILES:")

        # Get MMM tickers from configuration
        gaps_processor = MmmGapsProcessor(config, user_config, 'daily')
        tickers = gaps_processor._parse_tickers_config()
        print(f"  Configured tickers: {tickers}")

        if not tickers:
            print(f"  ❌ No tickers configured in sr_mmm_gaps_tickers")
            print(f"  📋 USER CONFIG CHECK:")

            # Check all sr_mmm related configurations
            for attr in dir(user_config):
                if 'sr_mmm' in attr.lower():
                    value = getattr(user_config, attr, None)
                    print(f"    {attr} = {value}")
            print()

        # Get input/output directories
        input_dir = gaps_processor._get_input_directory()
        output_dir = gaps_processor._get_output_directory()

        print(f"  Input directory: {input_dir}")
        print(f"  Output directory: {output_dir}")
        print()

        if tickers and input_dir:
            # Check for input files (market data)
            print(f"📊 MARKET DATA INPUT FILES:")
            for ticker in tickers:
                input_file = input_dir / f"{ticker}.csv"
                print(f"  {ticker}.csv: {'✅ EXISTS' if input_file.exists() else '❌ MISSING'} ({input_file})")
            print()

            # Check for gap output files
            print(f"📈 GAP OUTPUT FILES:")
            filename_suffix = getattr(user_config, 'sr_mmm_gaps_values_filename_suffix', '_gap')
            for ticker in tickers:
                gap_file = output_dir / f"{ticker}{filename_suffix}.csv"
                print(f"  {ticker}{filename_suffix}.csv: {'✅ EXISTS' if gap_file.exists() else '❌ MISSING'} ({gap_file})")
            print()

        # Test 2: Run gap calculation if needed
        print(f"🔄 GAP CALCULATION TEST:")
        try:
            gap_results = gaps_processor.run_gap_analysis()
            print(f"  Gap analysis result: {gap_results}")

            if 'error' in gap_results:
                print(f"  ❌ Gap calculation error: {gap_results['error']}")
                return False
            else:
                print(f"  ✅ Gap calculation successful:")
                print(f"    - Processed tickers: {gap_results.get('tickers_processed', [])}")
                print(f"    - CSV files: {len(gap_results.get('csv_files', []))}")
                print(f"    - Failed tickers: {gap_results.get('failed_tickers', [])}")

        except Exception as e:
            print(f"  ❌ Error running gap calculation: {e}")
            import traceback
            traceback.print_exc()
            gap_results = {'error': str(e)}
        print()

        # Test 3: Test chart generation
        print(f"📊 CHART GENERATION TEST:")
        charts_processor = MmmChartsProcessor(config, user_config, 'daily')

        try:
            # Create mock gap_results if needed
            if 'error' in gap_results:
                print(f"  Using mock gap results due to calculation error")
                gap_results = {
                    'tickers_processed': tickers if tickers else ['SPY', 'QQQ', 'IWM'],
                    'csv_files': [],
                    'failed_tickers': [],
                    'total_gaps_calculated': 0
                }

            # Test chart configuration reading
            chart_config_file = getattr(user_config, 'sr_mmm_gaps_charts_display_panel', 'user_data_sr_mmm.csv')
            config_valid = charts_processor._validate_chart_config(chart_config_file)
            print(f"  Chart config file valid: {config_valid} ({chart_config_file})")

            if config_valid:
                # Test chart generation
                chart_files = charts_processor.run_gap_charts_analysis(gap_results)
                print(f"  Chart generation result: {len(chart_files)} files generated")

                if chart_files:
                    print(f"  ✅ Generated charts:")
                    for chart_file in chart_files:
                        print(f"    - {chart_file}")
                else:
                    print(f"  ❌ No charts generated")

                # Check charts output directory
                from src.sustainability_ratios.sr_output_manager import get_sr_output_manager
                output_manager = get_sr_output_manager()
                charts_dir = output_manager.get_submodule_dir('mmm') / 'charts'
                print(f"  Charts directory: {charts_dir}")
                print(f"  Charts directory exists: {charts_dir.exists()}")

                if charts_dir.exists():
                    chart_files_found = list(charts_dir.glob('*.png'))
                    print(f"  PNG files found: {len(chart_files_found)}")
                    for chart_file in chart_files_found:
                        print(f"    - {chart_file.name}")

            else:
                print(f"  ❌ Chart configuration file not valid")

        except Exception as e:
            print(f"  ❌ Error in chart generation test: {e}")
            import traceback
            traceback.print_exc()
        print()

        # Test 4: Check data preparation for charts
        print(f"🎯 DATA PREPARATION TEST:")
        if config_valid:
            try:
                from src.sustainability_ratios.sr_config_reader import parse_panel_config
                resolved_path = charts_processor._resolve_config_path(chart_config_file)
                panels_config = parse_panel_config(str(resolved_path))

                if panels_config:
                    # Test data preparation for first panel config
                    row_config = panels_config[0]
                    panel_config = list(row_config.values())[0]

                    print(f"  Testing data preparation for: {panel_config.get('file_name_id')}")
                    print(f"  Data source: {panel_config.get('data_source')}")

                    chart_data = charts_processor._prepare_chart_data(panel_config, gap_results)

                    if chart_data:
                        print(f"  ✅ Chart data prepared successfully")
                        print(f"    - Data source: {chart_data.get('data_source')}")
                        result = chart_data.get('result', {})
                        print(f"    - Available columns: {list(result.keys()) if result else 'None'}")
                    else:
                        print(f"  ❌ Chart data preparation failed")

                        # Check if gap CSV files exist
                        csv_files = gap_results.get('csv_files', [])
                        print(f"    Available CSV files: {csv_files}")

                        for csv_file in csv_files:
                            if Path(csv_file).exists():
                                print(f"    ✅ CSV exists: {csv_file}")
                                # Try to read it
                                try:
                                    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                                    print(f"      Columns: {list(df.columns)}")
                                    print(f"      Rows: {len(df)}")
                                except Exception as e:
                                    print(f"      Error reading: {e}")
                            else:
                                print(f"    ❌ CSV missing: {csv_file}")

            except Exception as e:
                print(f"  ❌ Error in data preparation test: {e}")
                import traceback
                traceback.print_exc()

        return True

    except Exception as e:
        print(f"❌ Error in MMM chart generation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mmm_chart_generation()
    if success:
        print(f"\n🎉 MMM chart generation test completed!")
    else:
        print(f"\n💥 MMM chart generation test failed!")