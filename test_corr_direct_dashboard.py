#!/usr/bin/env python3
"""
Test CORR Direct Dashboard Generation
====================================

Use the real CORR data to test dashboard generation directly.
"""

import sys
import os
sys.path.append('/home/imagda/_invest2024/python/metaData_v1')

import pandas as pd
from pathlib import Path

def test_corr_direct_dashboard():
    """Test CORR with direct dashboard generation."""
    print("=" * 50)
    print("TESTING CORR DIRECT DASHBOARD GENERATION")
    print("=" * 50)

    try:
        # Step 1: Load the real CORR data
        print("\n🔍 STEP 1: Load Real CORR Data")
        print("-" * 30)

        from src.sustainability_ratios.sr_config_reader import parse_panel_config
        from src.sustainability_ratios.sr_market_data import load_market_data_for_panels
        from src.config import Config
        from src.data_reader import DataReader

        # Load CORR configuration
        config_file = "/home/imagda/_invest2024/python/metaData_v1/user_data_sr_panel.csv"
        panel_configs = parse_panel_config(config_file)

        # Find CORR config
        corr_config = None
        for config in panel_configs:
            for key, value in config.items():
                if 'CORR' in key and isinstance(value, dict):
                    corr_config = value
                    break
            if corr_config:
                break

        # Load data
        config = Config()
        data_reader = DataReader(config)
        test_config = {'SPY_vs_VIX': corr_config}
        panel_data = load_market_data_for_panels(test_config, data_reader)

        corr_data = panel_data['CORR(20)_for_(SPY,QQQ)']
        print(f"✓ Loaded CORR data: {type(corr_data)}")
        print(f"  Keys: {list(corr_data.keys())}")
        print(f"  Correlation points: {len(corr_data['correlation'])}")

    except Exception as e:
        print(f"✗ Failed to load CORR data: {e}")
        return

    try:
        # Step 2: Create proper panel results structure
        print("\n🔍 STEP 2: Create Panel Results Structure")
        print("-" * 30)

        # Convert CORR dict to the format expected by dashboard generator
        # Follow the same pattern as SR processor in sr_calculations.py
        panel_results = {
            'SPY_vs_VIX': {
                'data_source': 'CORR(20)_for_(SPY,QQQ)',
                'indicator': 'CORR(20)',
                'result': corr_data  # The actual CORR calculation result
            }
        }

        print(f"✓ Created panel results structure")
        print(f"  Result type: {type(panel_results['SPY_vs_VIX']['result'])}")

    except Exception as e:
        print(f"✗ Failed to create panel results: {e}")
        return

    try:
        # Step 3: Test dashboard generation
        print("\n🔍 STEP 3: Test Dashboard Generation")
        print("-" * 30)

        from src.sustainability_ratios.sr_dashboard_generator import generate_sr_dashboard
        from src.user_defined_data import read_user_data

        output_dir = Path("/tmp/claude/corr_dashboard_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        user_config = read_user_data()

        print(f"Testing dashboard generation...")
        dashboard_files = generate_sr_dashboard(
            results=panel_results,
            output_dir=output_dir,
            panel_config=test_config,
            user_config=user_config
        )

        if dashboard_files:
            print(f"✓ Dashboard generated successfully!")
            print(f"  Files created: {list(dashboard_files.keys())}")

            # Check actual output files
            output_files = list(output_dir.glob("*"))
            print(f"  Output files: {[f.name for f in output_files]}")

            if output_files:
                print(f"\n🎯 SUCCESS! CORR panel should now be visible in:")
                for file in output_files:
                    print(f"    {file}")
            else:
                print("⚠ No files were actually created")
        else:
            print("✗ Dashboard generation returned empty")

    except Exception as e:
        print(f"✗ Dashboard generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("CORR DIRECT DASHBOARD TEST COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    test_corr_direct_dashboard()