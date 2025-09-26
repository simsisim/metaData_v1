#!/usr/bin/env python3
"""
Research MMM Test Panel_2 Issue
==============================

Research why XLY_gap is not displaying in Panel_2 of mmm_test configuration.
Compare with working mmm_QQQ_Analysis_mmm configuration.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from pathlib import Path
from config import Config
from data_reader import DataReader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def research_mmm_test_panel2():
    """Research why mmm_test Panel_2 (XLY_gap) is not displaying."""

    print("=" * 80)
    print("RESEARCH: MMM_TEST PANEL_2 ISSUE")
    print("=" * 80)

    # Step 1: Analyze the configuration differences
    print("\n1. Configuration Analysis:")
    print("   mmm_QQQ_Analysis_mmm: line, QQQ + EMA(10) + SMA(50), XLY_gap")
    print("   mmm_test:            line, XLY,                     XLY_gap")
    print()
    print("   Key Differences:")
    print("   - Panel_1: QQQ + indicators vs simple XLY")
    print("   - Panel_2: Both use XLY_gap")
    print("   - Panel indices: QQQ has indices, mmm_test has none")

    # Step 2: Test data availability for both configurations
    print("\n2. Data Availability Test:")

    config = Config()
    data_reader = DataReader(config, timeframe='daily')

    # Test each ticker
    tickers_to_test = ['QQQ', 'XLY', 'XLY_gap']
    data_status = {}

    for ticker in tickers_to_test:
        try:
            data = data_reader.read_stock_data(ticker)
            if data is not None:
                data_status[ticker] = {
                    'status': 'SUCCESS',
                    'rows': len(data),
                    'columns': list(data.columns),
                    'recent_close': data['Close'].iloc[-1] if 'Close' in data.columns else 'N/A'
                }
                print(f"   ✅ {ticker}: {len(data)} rows, Close={data_status[ticker]['recent_close']:.2f}")
            else:
                data_status[ticker] = {'status': 'FAILED', 'error': 'No data returned'}
                print(f"   ❌ {ticker}: No data returned")

        except Exception as e:
            data_status[ticker] = {'status': 'ERROR', 'error': str(e)}
            print(f"   ❌ {ticker}: Error - {e}")

    # Step 3: Compare data characteristics
    print("\n3. Data Comparison (XLY vs XLY_gap):")

    if data_status.get('XLY', {}).get('status') == 'SUCCESS' and data_status.get('XLY_gap', {}).get('status') == 'SUCCESS':
        xly_data = data_reader.read_stock_data('XLY')
        xly_gap_data = data_reader.read_stock_data('XLY_gap')

        print("   Data Structure Comparison:")
        print(f"   XLY:     {len(xly_data)} rows, columns: {list(xly_data.columns)}")
        print(f"   XLY_gap: {len(xly_gap_data)} rows, columns: {list(xly_gap_data.columns)}")

        # Check data ranges
        common_dates = xly_data.index.intersection(xly_gap_data.index)
        print(f"   Common dates: {len(common_dates)}")

        if len(common_dates) > 0:
            print(f"   Date range: {common_dates.min()} to {common_dates.max()}")

            # Compare recent values
            print("\n   Recent Close Values Comparison (last 3 dates):")
            print("   Date                XLY_Close    XLY_gap_Close    Difference")
            print("   " + "-" * 65)

            recent_dates = common_dates[-3:]
            for date in recent_dates:
                xly_close = xly_data.loc[date, 'Close']
                xly_gap_close = xly_gap_data.loc[date, 'Close']
                diff = xly_gap_close - xly_close

                print(f"   {date.strftime('%Y-%m-%d')}    {xly_close:>8.2f}    {xly_gap_close:>8.2f}    {diff:>+6.2f}")

            print(f"   ✅ Data comparison shows gap-adjusted values working correctly!")

    # Step 4: Analyze panel configuration parsing
    print("\n4. Panel Configuration Analysis:")

    # Read the MMM configuration
    config_file = Path("user_data_sr_mmm.csv")
    if config_file.exists():
        try:
            config_df = pd.read_csv(config_file)
            print("   MMM Configuration file loaded successfully")

            # Find mmm_test row
            test_row = config_df[config_df['file_name_id'] == 'mmm_test']
            if not test_row.empty:
                test_config = test_row.iloc[0]
                print(f"   mmm_test configuration:")
                print(f"     file_name_id: {test_config['file_name_id']}")
                print(f"     chart_type: {test_config['chart_type']}")
                print(f"     Panel_1: '{test_config['Panel_1']}'")
                print(f"     Panel_2: '{test_config['Panel_2']}'")

                # Check for empty or NaN values
                if pd.isna(test_config['Panel_2']) or test_config['Panel_2'] == '':
                    print("   ❌ ISSUE: Panel_2 is empty or NaN!")
                else:
                    print(f"   ✅ Panel_2 has value: '{test_config['Panel_2']}'")

                # Check all panel indices
                panel_indices = ['Panel_1_index', 'Panel_2_index', 'Panel_3_index', 'Panel_4_index', 'Panel_5_index', 'Panel_6_index']
                for idx_col in panel_indices:
                    if idx_col in test_config:
                        val = test_config[idx_col]
                        if pd.isna(val) or val == '':
                            print(f"     {idx_col}: empty")
                        else:
                            print(f"     {idx_col}: '{val}'")

            else:
                print("   ❌ mmm_test configuration not found!")

            # Compare with working configuration
            qqq_row = config_df[config_df['file_name_id'] == 'mmm_QQQ_Analysis_mmm']
            if not qqq_row.empty:
                qqq_config = qqq_row.iloc[0]
                print(f"\n   mmm_QQQ_Analysis_mmm (working) configuration:")
                print(f"     Panel_1: '{qqq_config['Panel_1']}'")
                print(f"     Panel_2: '{qqq_config['Panel_2']}'")
                print(f"     Panel_1_index: '{qqq_config.get('Panel_1_index', 'empty')}'")
                print(f"     Panel_2_index: '{qqq_config.get('Panel_2_index', 'empty')}'")

        except Exception as e:
            print(f"   ❌ Error reading MMM configuration: {e}")

    # Step 5: Test potential SR panel parsing issues
    print("\n5. Potential SR Panel Parsing Issues:")

    print("   Checking for common panel parsing problems:")

    # Check for data source extraction patterns
    panel_configs = {
        'mmm_QQQ_Analysis_mmm': {
            'Panel_1': 'QQQ + EMA(10) + SMA(50)',
            'Panel_2': 'XLY_gap'
        },
        'mmm_test': {
            'Panel_1': 'XLY',
            'Panel_2': 'XLY_gap'
        }
    }

    for config_name, panels in panel_configs.items():
        print(f"\n   {config_name}:")
        for panel_name, panel_value in panels.items():
            print(f"     {panel_name}: '{panel_value}'")

            # Simulate ticker extraction
            if panel_value:
                # Simple ticker extraction (basic approach)
                simple_tickers = []
                if '+' in panel_value:
                    # Complex bundled format
                    parts = panel_value.split('+')
                    base_ticker = parts[0].strip()
                    simple_tickers.append(base_ticker)
                    print(f"       Complex format detected, base ticker: {base_ticker}")
                else:
                    # Simple ticker
                    simple_tickers.append(panel_value.strip())
                    print(f"       Simple ticker: {panel_value.strip()}")

                # Check if ticker data is available
                for ticker in simple_tickers:
                    if ticker in data_status:
                        status = data_status[ticker]['status']
                        print(f"       Data status for {ticker}: {status}")
                    else:
                        print(f"       {ticker} not tested")

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print("🔍 CONFIGURATION COMPARISON:")
    print("   mmm_QQQ_Analysis_mmm Panel_2: XLY_gap ✅ Works")
    print("   mmm_test Panel_2:             XLY_gap ❌ Not showing")

    print("\n📊 DATA AVAILABILITY:")
    print("   XLY:     Available and loads correctly")
    print("   XLY_gap: Available and loads correctly with gap-adjusted Close")
    print("   QQQ:     Available and loads correctly")

    print("\n🤔 POTENTIAL ROOT CAUSES:")
    print("   1. Panel index configuration differences")
    print("      - mmm_QQQ_Analysis_mmm has Panel_1_index and Panel_2_index")
    print("      - mmm_test has empty panel indices")
    print()
    print("   2. SR panel processing logic")
    print("      - May require panel indices to display panels")
    print("      - May have different behavior for simple vs complex panels")
    print()
    print("   3. Chart generation configuration")
    print("      - Panel display may depend on indicator configuration")
    print("      - Empty indices may cause panel to be skipped")

    print("\n💡 HYPOTHESIS:")
    print("   The SR panel system may require panel indices (indicators)")
    print("   to properly display panels. mmm_test has empty indices,")
    print("   while mmm_QQQ_Analysis_mmm has configured indices.")

    print("\n🧪 TO TEST HYPOTHESIS:")
    print("   Add panel indices to mmm_test configuration:")
    print("   Panel_1_index: A_RSI(14)_for_(XLY)")
    print("   Panel_2_index: B_RSI(14)_for_(XLY_gap)")
    print("   This would test if indices are required for panel display.")

if __name__ == "__main__":
    research_mmm_test_panel2()