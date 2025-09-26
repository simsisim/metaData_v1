#!/usr/bin/env python3

"""
Test script to verify OHLCV data loading for candlestick charts.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from modules
from src.data_reader import DataReader
from src.config import Config
from src.sustainability_ratios.sr_market_data import load_market_data_for_panels

def test_ohlcv_data_loading():
    """Test OHLCV data loading for specific tickers."""

    print("🧪 Testing OHLCV data loading...")

    try:
        # Initialize configuration and data reader
        config = Config()
        data_reader = DataReader(config)

        # Test loading SPY data directly
        print(f"\n1. 🔍 Testing direct SPY data loading...")
        spy_data = data_reader.read_stock_data('SPY')

        if spy_data is not None:
            print(f"   ✅ SPY data loaded: {len(spy_data)} rows")
            print(f"   📊 Columns: {list(spy_data.columns)}")

            # Check for OHLCV columns
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_ohlcv = [col for col in ohlcv_columns if col not in spy_data.columns]

            if missing_ohlcv:
                print(f"   ❌ Missing OHLCV columns: {missing_ohlcv}")
            else:
                print(f"   ✅ All OHLCV columns present!")

                # Show sample data
                print(f"   📈 Latest 3 rows:")
                print(spy_data.tail(3).to_string())

        else:
            print(f"   ❌ Failed to load SPY data")

        # Test Panel_2 configuration specifically
        print(f"\n2. 🔍 Testing Panel_2 market data loading...")

        # Simulate Panel_2 configuration
        panel_config = {
            'Panel_2_SPY': {
                'data_source': 'SPY',
                'indicator': '',
                'panel_name': 'Panel_2',
                'chart_type': 'candle'
            }
        }

        market_data = load_market_data_for_panels(panel_config, data_reader)

        if 'SPY' in market_data:
            spy_panel_data = market_data['SPY']
            print(f"   ✅ Panel SPY data loaded: {len(spy_panel_data)} rows")
            print(f"   📊 Columns: {list(spy_panel_data.columns)}")

            # Check for OHLCV columns in panel data
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_ohlcv = [col for col in ohlcv_columns if col not in spy_panel_data.columns]

            if missing_ohlcv:
                print(f"   ❌ Missing OHLCV columns in panel data: {missing_ohlcv}")
            else:
                print(f"   ✅ All OHLCV columns present in panel data!")
        else:
            print(f"   ❌ SPY not found in market data: {list(market_data.keys())}")

        print(f"\n✅ OHLCV data loading test completed!")

    except Exception as e:
        print(f"❌ Error testing OHLCV data loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ohlcv_data_loading()