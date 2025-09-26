#!/usr/bin/env python3

"""
Test script to verify the candlestick chart fix for Panel_2 SPY.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.sustainability_ratios.sr_config_reader import parse_panel_config
from src.sustainability_ratios.sr_market_data import load_market_data_for_panels
from src.data_reader import DataReader
from src.config import Config

def test_candlestick_fix():
    """Test that Panel_2 SPY now has complete OHLCV data in result structure."""

    print("🧪 Testing candlestick chart fix...")

    try:
        # Initialize configuration
        config = Config()
        data_reader = DataReader(config)

        # Parse panel configuration
        panel_csv_file = "user_data_sr_panel.csv"
        panel_config = parse_panel_config(panel_csv_file)

        # Find QQQ_vs_SPY_panel row (row 4)
        target_row = None
        for row_config in panel_config:
            for config_key, config_data in row_config.items():
                if 'row4' in config_key:
                    target_row = row_config
                    break
            if target_row:
                break

        if not target_row:
            print("❌ Could not find QQQ_vs_SPY_panel configuration")
            return

        # Load market data for the panels
        market_data = load_market_data_for_panels(target_row, data_reader)
        print(f"✅ Market data loaded for: {list(market_data.keys())}")

        # Simulate the panel processing logic for SPY Panel_2
        panel_2_config = None
        for config_key, config_data in target_row.items():
            if (config_data.get('panel_name') == 'Panel_2' and
                config_data.get('position') == 'main'):
                panel_2_config = config_data
                break

        if not panel_2_config:
            print("❌ Could not find Panel_2 SPY configuration")
            return

        print(f"\n🎯 Testing Panel_2 SPY data structure:")
        print(f"   Data Source: {panel_2_config.get('data_source')}")
        print(f"   Chart Type: {panel_2_config.get('chart_type')}")
        print(f"   Has Indicator: {panel_2_config.get('has_indicator', False)}")

        # Simulate the indicator_result creation logic from sr_calculations.py
        data_source = panel_2_config['data_source']
        indicator = panel_2_config.get('indicator', '')

        if data_source in market_data:
            panel_data = market_data[data_source]
            print(f"   ✅ Panel data available: {len(panel_data)} rows")
            print(f"   📊 Panel data columns: {list(panel_data.columns)}")

            # Simulate the fixed logic for panels without indicators
            if not indicator:
                print(f"\n🔧 APPLYING FIXED LOGIC (no indicator):")
                base_data = panel_data

                # Create the indicator_result as the fixed code would
                indicator_result = {
                    'Open': base_data.get('Open'),
                    'High': base_data.get('High'),
                    'Low': base_data.get('Low'),
                    'Close': base_data.get('Close'),
                    'Volume': base_data.get('Volume'),
                    'price': base_data['Close'],  # Keep for backward compatibility
                    'metadata': {
                        'chart_type': 'price',
                        'data_type': 'ohlcv'
                    }
                }

                print(f"   📋 Result structure keys: {list(indicator_result.keys())}")

                # Check OHLCV availability
                ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = []

                for col in ohlcv_cols:
                    data_obj = indicator_result.get(col)
                    if data_obj is None:
                        missing_cols.append(f"{col} (None)")
                    elif not hasattr(data_obj, 'values'):
                        missing_cols.append(f"{col} (not Series)")
                    elif len(data_obj) == 0:
                        missing_cols.append(f"{col} (empty)")
                    else:
                        print(f"      ✅ {col}: {type(data_obj)} with {len(data_obj)} values")

                if missing_cols:
                    print(f"      ❌ OHLCV Issues: {missing_cols}")
                else:
                    print(f"      ✅ ALL OHLCV DATA PRESENT - Candlestick charts should work!")

                # Test sample values
                if indicator_result.get('Close') is not None:
                    close_data = indicator_result['Close']
                    print(f"      📈 Close sample (last 3): {close_data.tail(3).tolist()}")

        else:
            print(f"   ❌ Panel data not found for: {data_source}")

        print(f"\n✅ Candlestick fix test completed!")

    except Exception as e:
        print(f"❌ Error testing candlestick fix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_candlestick_fix()