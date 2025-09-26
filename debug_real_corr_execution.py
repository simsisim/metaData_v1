#!/usr/bin/env python3
"""
Debug Real CORR Execution
========================

Trace the actual SR system execution to see exactly where CORR fails.
"""

import sys
import os
sys.path.append('/home/imagda/_invest2024/python/metaData_v1')

import pandas as pd
import logging

# Set up detailed logging to capture everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/claude/corr_debug.log')
    ]
)

def debug_real_corr():
    """Debug the real CORR execution step by step."""
    print("=" * 60)
    print("DEBUGGING REAL CORR EXECUTION")
    print("=" * 60)

    # Step 1: Test the actual SR system with CORR
    try:
        print("\n🔍 STEP 1: Import and Setup")
        print("-" * 40)

        from src.sustainability_ratios.sr_config_reader import parse_panel_config
        from src.sustainability_ratios.sr_market_data import load_market_data_for_panels
        from src.sustainability_ratios.sr_calculations import SRProcessor
        from src.config import Config
        from src.data_reader import DataReader
        from src.user_defined_data import read_user_data

        print("✓ All imports successful")

    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("\n🔍 STEP 2: Load Configuration")
        print("-" * 40)

        # Load panel configuration
        config_file = "/home/imagda/_invest2024/python/metaData_v1/user_data_sr_panel.csv"
        panel_configs = parse_panel_config(config_file)

        print(f"✓ Loaded {len(panel_configs)} panel configurations")

        # Find CORR configuration
        corr_config = None
        corr_key = None
        for config in panel_configs:
            for key, value in config.items():
                if 'CORR' in key and isinstance(value, dict):
                    corr_config = value
                    corr_key = key
                    break
            if corr_config:
                break

        if corr_config:
            print(f"✓ Found CORR config: {corr_key}")
            print(f"  Data source: {corr_config.get('data_source')}")
            print(f"  Indicator: {corr_config.get('indicator')}")
            print(f"  Has indicator: {corr_config.get('has_indicator')}")
        else:
            print("✗ CORR configuration not found")
            return

        # Set up minimal config for testing
        test_config = {'SPY_vs_VIX': corr_config}

    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("\n🔍 STEP 3: Setup Data Reader")
        print("-" * 40)

        config = Config()
        data_reader = DataReader(config)
        user_config = read_user_data()

        print("✓ Data reader and user config loaded")

    except Exception as e:
        print(f"✗ Data reader setup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("\n🔍 STEP 4: Load Market Data")
        print("-" * 40)

        # Load market data for CORR panel
        panel_data = load_market_data_for_panels(test_config, data_reader)

        print(f"✓ Market data loaded")
        print(f"  Data sources: {list(panel_data.keys())}")

        for key, data in panel_data.items():
            if data is not None:
                print(f"  {key}: {type(data)}")
                if hasattr(data, 'shape'):
                    print(f"    Shape: {data.shape}")
                    print(f"    Columns: {list(data.columns)}")
                    if hasattr(data, 'attrs'):
                        print(f"    Attrs: {data.attrs}")
                elif isinstance(data, dict):
                    print(f"    Dict keys: {list(data.keys())}")
                    for subkey, subval in data.items():
                        if hasattr(subval, '__len__'):
                            print(f"      {subkey}: {type(subval)} (len={len(subval)})")
                        else:
                            print(f"      {subkey}: {type(subval)}")
            else:
                print(f"  {key}: None")

    except Exception as e:
        print(f"✗ Market data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("\n🔍 STEP 5: SR Processor Execution")
        print("-" * 40)

        # Create SR processor with proper parameters
        timeframe = 'daily'
        processor = SRProcessor(config, user_config, timeframe)

        print("✓ SR Processor created")

        # Set up the panel configuration in the processor
        processor.panel_config = test_config
        print("✓ Panel config set in processor")

        # Process panel indicators
        print("\nProcessing panel indicators...")
        panel_results = processor.process_panel_indicators()

        if panel_results:
            print(f"✓ Panel processing completed")
            print(f"  Results keys: {list(panel_results.keys())}")

            for key, result in panel_results.items():
                print(f"\n  Result '{key}':")
                print(f"    Type: {type(result)}")
                if isinstance(result, dict):
                    print(f"    Keys: {list(result.keys())}")
                    if 'result' in result:
                        print(f"    Result type: {type(result['result'])}")
                        if hasattr(result['result'], 'keys'):
                            print(f"    Result keys: {list(result['result'].keys())}")
                        if hasattr(result['result'], 'shape'):
                            print(f"    Result shape: {result['result'].shape}")
        else:
            print("✗ Panel processing returned None/empty")
            return

    except Exception as e:
        print(f"✗ SR Processor execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("\n🔍 STEP 6: Dashboard Generation Test")
        print("-" * 40)

        from src.sustainability_ratios.sr_dashboard_generator import generate_sr_dashboard
        from pathlib import Path

        output_dir = Path("/tmp/claude/corr_real_debug")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Testing dashboard generation...")
        print(f"  Panel results: {list(panel_results.keys())}")
        print(f"  Panel config: {list(test_config.keys())}")

        dashboard_files = generate_sr_dashboard(
            results=panel_results,
            output_dir=output_dir,
            panel_config=test_config,
            user_config=user_config
        )

        if dashboard_files:
            print(f"✓ Dashboard generated successfully")
            print(f"  Files: {list(dashboard_files.keys())}")

            # Check actual files
            output_files = list(output_dir.glob("*"))
            print(f"  Output files: {[f.name for f in output_files]}")
        else:
            print("✗ Dashboard generation failed")

    except Exception as e:
        print(f"✗ Dashboard generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("REAL CORR DEBUG COMPLETED")
    print("Check /tmp/claude/corr_debug.log for detailed logs")
    print("=" * 60)

if __name__ == "__main__":
    debug_real_corr()