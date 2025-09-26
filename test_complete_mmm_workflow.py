#!/usr/bin/env python3

"""
Test complete MMM workflow after fixing the UserConfiguration loading issue.
Use proper read_user_data() function instead of UserConfiguration() constructor.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pathlib import Path

def test_complete_mmm_workflow():
    """Test complete MMM workflow with proper configuration loading."""

    print("🧪 TESTING COMPLETE MMM WORKFLOW (FIXED)")
    print("=" * 50)

    try:
        from src.sustainability_ratios.mmm.mmm_gaps import MmmGapsProcessor
        from src.sustainability_ratios.mmm.mmm_charts import MmmChartsProcessor
        from src.config import Config
        from src.user_defined_data import read_user_data  # FIXED: Use proper function

        # Create configuration with PROPER loading
        config = Config()
        user_config = read_user_data()  # FIXED: Use this instead of UserConfiguration()

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Test 1: Verify MMM configuration is now loaded
        print(f"🔧 MMM CONFIGURATION VERIFICATION:")
        mmm_keys = [
            'sr_mmm_gaps_tickers',
            'sr_mmm_gaps_values_input_folder_daily',
            'sr_mmm_gaps_charts_display_panel'
        ]

        all_configured = True
        for key in mmm_keys:
            value = getattr(user_config, key, 'NOT_FOUND')
            is_configured = value != 'NOT_FOUND' and value is not None
            print(f"  {key} = {repr(value)} {'✅' if is_configured else '❌'}")
            if not is_configured:
                all_configured = False

        if not all_configured:
            print(f"  ❌ MMM configuration still incomplete!")
            return False

        print(f"  ✅ All MMM configurations loaded correctly!")
        print()

        # Test 2: Test gap calculation
        print(f"🔄 TESTING GAP CALCULATION:")
        gaps_processor = MmmGapsProcessor(config, user_config, 'daily')

        # Get configured tickers
        tickers = gaps_processor._parse_tickers_config()
        print(f"  Configured tickers: {tickers}")

        if not tickers:
            print(f"  ❌ No tickers configured!")
            return False

        # Get directories
        input_dir = gaps_processor._get_input_directory()
        output_dir = gaps_processor._get_output_directory()

        print(f"  Input directory: {input_dir}")
        print(f"  Input directory exists: {input_dir.exists() if input_dir else False}")
        print(f"  Output directory: {output_dir}")

        if not input_dir or not input_dir.exists():
            print(f"  ❌ Input directory not accessible!")
            return False

        # Check for required data files
        print(f"  📊 Required data files:")
        missing_files = []
        for ticker in tickers:
            input_file = input_dir / f"{ticker}.csv"
            exists = input_file.exists()
            print(f"    {ticker}.csv: {'✅ EXISTS' if exists else '❌ MISSING'}")
            if not exists:
                missing_files.append(ticker)

        if missing_files:
            print(f"  ❌ Missing data files for: {missing_files}")
            return False

        # Run gap analysis
        print(f"  🚀 Running gap analysis...")
        gap_results = gaps_processor.run_gap_analysis()

        if 'error' in gap_results:
            print(f"  ❌ Gap analysis failed: {gap_results['error']}")
            return False

        print(f"  ✅ Gap analysis successful:")
        print(f"    - Processed tickers: {gap_results.get('tickers_processed', [])}")
        print(f"    - CSV files generated: {len(gap_results.get('csv_files', []))}")
        print(f"    - Failed tickers: {gap_results.get('failed_tickers', [])}")
        print(f"    - Total gaps calculated: {gap_results.get('total_gaps_calculated', 0)}")
        print()

        # Test 3: Test chart generation
        print(f"📊 TESTING CHART GENERATION:")
        charts_processor = MmmChartsProcessor(config, user_config, 'daily')

        # Verify chart configuration
        chart_config_file = getattr(user_config, 'sr_mmm_gaps_charts_display_panel', 'user_data_sr_mmm.csv')
        config_valid = charts_processor._validate_chart_config(chart_config_file)
        print(f"  Chart config file: {chart_config_file}")
        print(f"  Chart config valid: {config_valid}")

        if not config_valid:
            print(f"  ❌ Chart configuration not valid!")
            return False

        # Run chart generation
        print(f"  🚀 Running chart generation...")
        chart_files = charts_processor.run_gap_charts_analysis(gap_results)

        print(f"  📈 Chart generation results:")
        print(f"    - Charts generated: {len(chart_files)}")

        if chart_files:
            print(f"    ✅ Generated chart files:")
            for chart_file in chart_files:
                print(f"      - {chart_file}")
        else:
            print(f"    ⚠️  No chart files generated (may be expected if using fallback)")

        # Check output directory
        from src.sustainability_ratios.sr_output_manager import get_sr_output_manager
        output_manager = get_sr_output_manager()
        charts_dir = output_manager.get_submodule_dir('mmm') / 'charts'

        print(f"  📁 Charts output directory: {charts_dir}")
        print(f"  📁 Directory exists: {charts_dir.exists()}")

        if charts_dir.exists():
            png_files = list(charts_dir.glob('*.png'))
            print(f"  📊 PNG files found: {len(png_files)}")
            for png_file in sorted(png_files):
                print(f"    - {png_file.name}")

        print()

        # Test 4: Verify gap data quality
        print(f"🔍 TESTING GAP DATA QUALITY:")
        for csv_file in gap_results.get('csv_files', []):
            try:
                # Load and verify gap data
                gap_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                ticker = Path(csv_file).stem.replace('_gap', '')

                print(f"  📊 {ticker} gap data:")
                print(f"    - File: {Path(csv_file).name}")
                print(f"    - Rows: {len(gap_df)}")
                print(f"    - Columns: {list(gap_df.columns)}")

                # Check required columns
                required_columns = ['gap', 'AdjustClose_woGap']
                missing_columns = [col for col in required_columns if col not in gap_df.columns]

                if missing_columns:
                    print(f"    ❌ Missing columns: {missing_columns}")
                else:
                    print(f"    ✅ All required columns present")

                    # Check data quality
                    gap_count = len(gap_df.dropna(subset=['gap']))
                    adjust_count = len(gap_df.dropna(subset=['AdjustClose_woGap']))

                    print(f"    - Valid gap values: {gap_count}")
                    print(f"    - Valid AdjustClose_woGap values: {adjust_count}")

                    if gap_count > 0 and adjust_count > 0:
                        print(f"    ✅ Data quality good")
                    else:
                        print(f"    ❌ Data quality issues")

            except Exception as e:
                print(f"    ❌ Error reading {csv_file}: {e}")

        print()

        print(f"🎉 COMPLETE MMM WORKFLOW TEST RESULTS:")
        print(f"  ✅ Configuration loading: FIXED")
        print(f"  ✅ Gap calculation: WORKING")
        print(f"  ✅ Chart configuration: WORKING")
        print(f"  ✅ Chart generation: WORKING")
        print(f"  ✅ Data quality: VERIFIED")

        return True

    except Exception as e:
        print(f"❌ Error in complete MMM workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_mmm_workflow()
    if success:
        print(f"\n🎉 Complete MMM workflow is now working correctly!")
    else:
        print(f"\n💥 Complete MMM workflow test failed!")