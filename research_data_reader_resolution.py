#!/usr/bin/env python3

"""
Research how data_reader resolves ticker names to files.
Test the file resolution mechanism used by Panel/Overview modules.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path

def research_data_reader_resolution():
    """Research how data_reader resolves ticker names to actual files."""

    print("🧪 RESEARCHING DATA_READER FILE RESOLUTION")
    print("=" * 50)

    try:
        from src.config import Config
        from src.user_defined_data import read_user_data
        from src.data_reader import DataReader

        config = Config()
        user_config = read_user_data()
        data_reader = DataReader(config)

        print(f"📁 Base directory: {config.base_dir}")
        print()

        # Research 1: Data reader configuration
        print(f"🔍 RESEARCH 1: DATA READER CONFIGURATION")
        print(f"=" * 42)

        print(f"  📂 Data reader attributes:")
        for attr in ['daily_data_dir', 'weekly_data_dir', 'monthly_data_dir']:
            if hasattr(data_reader, attr):
                value = getattr(data_reader, attr)
                print(f"    {attr}: {value}")

        print(f"  📊 User config data directories:")
        print(f"    yf_daily_data_files: {user_config.yf_daily_data_files}")
        print(f"    yf_weekly_data_files: {user_config.yf_weekly_data_files}")
        print(f"    yf_monthly_data_files: {user_config.yf_monthly_data_files}")

        print()

        # Research 2: Test ticker resolution
        print(f"🔍 RESEARCH 2: TICKER RESOLUTION TESTING")
        print(f"=" * 44)

        # Test different ticker variations
        test_tickers = ['SPY', 'QQQ', 'XLY', 'XLC', 'SPY_gap', 'XLY_gap']

        for ticker in test_tickers:
            print(f"  🔍 Testing ticker: '{ticker}'")

            try:
                # Try to read data for this ticker
                data = data_reader.read_stock_data(ticker)

                if data is not None and not data.empty:
                    print(f"    ✅ SUCCESS: Found data ({len(data)} rows)")
                    print(f"    📊 Columns: {list(data.columns)[:5]}...")  # Show first 5 columns
                else:
                    print(f"    ❌ NO DATA: Returned None or empty DataFrame")

            except Exception as e:
                print(f"    ❌ ERROR: {e}")

        print()

        # Research 3: Check actual file existence
        print(f"🔍 RESEARCH 3: ACTUAL FILE EXISTENCE CHECK")
        print(f"=" * 45)

        # Check what files actually exist in the data directory
        daily_data_path = Path(user_config.yf_daily_data_files)
        if daily_data_path.exists():
            print(f"  📂 Daily data directory: {daily_data_path}")

            # Look for our test tickers
            csv_files = list(daily_data_path.glob('*.csv'))
            relevant_files = [f for f in csv_files if any(ticker in f.name for ticker in test_tickers)]

            print(f"  📄 Relevant CSV files found:")
            for csv_file in sorted(relevant_files):
                print(f"    - {csv_file.name}")

            # Check gap files specifically
            gap_files = [f for f in csv_files if '_gap.csv' in f.name]
            print(f"\n  📈 Gap files found:")
            for gap_file in sorted(gap_files):
                print(f"    - {gap_file.name}")

        else:
            print(f"  ❌ Daily data directory not found: {daily_data_path}")

        print()

        # Research 4: Test sr_market_data with different data sources
        print(f"🔍 RESEARCH 4: SR_MARKET_DATA RESOLUTION TEST")
        print(f"=" * 47)

        # Create mock panel configuration to test sr_market_data
        test_panel_config = {
            'test_spy': {
                'data_source': 'SPY + EMA(20)',
                'is_bundled': True,
                'bundled_components': ['SPY', 'EMA(20)']
            },
            'test_spy_gap': {
                'data_source': 'SPY_gap',
                'is_bundled': False
            },
            'test_xly_gap': {
                'data_source': 'XLY_gap + EMA(20)',
                'is_bundled': True,
                'bundled_components': ['XLY_gap', 'EMA(20)']
            }
        }

        try:
            from src.sustainability_ratios.sr_market_data import load_market_data_for_panels

            print(f"  🧪 Testing sr_market_data with different data sources...")

            market_data = load_market_data_for_panels(test_panel_config, data_reader)

            print(f"  📊 Market data results:")
            for data_source, data in market_data.items():
                if data is not None:
                    print(f"    ✅ {data_source}: SUCCESS ({len(data)} rows)")
                else:
                    print(f"    ❌ {data_source}: FAILED")

        except Exception as e:
            print(f"  ❌ Error testing sr_market_data: {e}")

        print()

        # Research 5: File resolution patterns
        print(f"🎯 RESEARCH 5: FILE RESOLUTION PATTERNS")
        print(f"=" * 40)

        print(f"  📋 Current Panel/Overview Pattern:")
        print(f"    'SPY + EMA(20)' → data_reader.read_stock_data('SPY') → SPY.csv")
        print(f"    'QQQ' → data_reader.read_stock_data('QQQ') → QQQ.csv")
        print(f"    'XLY_gap' → data_reader.read_stock_data('XLY_gap') → XLY_gap.csv")

        print(f"\n  🔧 Required MMM Pattern (Same as Panel/Overview):")
        print(f"    'SPY + EMA(20)' → Use sr_market_data → SPY.csv")
        print(f"    'SPY_gap + EMA(20)' → Use sr_market_data → SPY_gap.csv")
        print(f"    'XLY_gap' → Use sr_market_data → XLY_gap.csv")

        print(f"\n  📊 User Configuration Examples:")
        print(f"    Regular market data: 'SPY + EMA(20)', 'QQQ + SMA(50)'")
        print(f"    Gap analysis data:   'XLY_gap + EMA(20)', 'XLC_gap'")
        print(f"    Mixed configuration: Both regular and gap data in same config")

        return True

    except Exception as e:
        print(f"❌ Error researching data_reader resolution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = research_data_reader_resolution()
    if success:
        print(f"\n🎉 Data reader resolution research completed!")
        print(f"\n📋 CONCLUSION: MMM should use existing sr_market_data infrastructure")
        print(f"   - Leverage data_reader.read_stock_data() for all file resolution")
        print(f"   - User controls data source via explicit naming: 'ticker' vs 'ticker_gap'")
    else:
        print(f"\n💥 Data reader resolution research failed!")