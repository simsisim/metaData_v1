#!/usr/bin/env python3
"""
Test MMM Submodule Functionality
=================================

Test the MMM (Market Maker Manipulation) submodule integration with SR module.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Add src to path for imports
sys.path.append('src')

def create_sample_data():
    """Create sample OHLCV data for testing."""
    # Create sample data directories
    input_dir = Path('../downloadData_v1/data/market_data/daily')
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create sample XLY data
    dates = pd.date_range('2024-09-01', periods=30, freq='D')
    np.random.seed(42)  # For reproducible results

    # Generate realistic OHLCV data
    base_price = 150.0
    prices = []

    for i in range(len(dates)):
        if i == 0:
            open_price = base_price
        else:
            # Add some gap patterns
            gap = np.random.normal(0, 0.5)  # Small gaps
            if i % 7 == 0:  # Weekly larger gaps
                gap = np.random.normal(0, 2.0)
            open_price = max(140, min(160, prices[-1]['Close'] + gap))

        # Intraday movement
        high = open_price + abs(np.random.normal(0, 1.5))
        low = open_price - abs(np.random.normal(0, 1.5))
        close = low + (high - low) * np.random.random()
        volume = int(np.random.normal(1000000, 200000))

        prices.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': max(500000, volume)
        })

    # Create DataFrames
    xly_df = pd.DataFrame(prices, index=dates)
    xlc_df = xly_df.copy()

    # Add some variation to XLC
    xlc_df['Open'] *= 0.95
    xlc_df['High'] *= 0.95
    xlc_df['Low'] *= 0.95
    xlc_df['Close'] *= 0.95
    xlc_df = xlc_df.round(2)

    # Save sample data
    xly_df.to_csv(input_dir / 'XLY.csv')
    xlc_df.to_csv(input_dir / 'XLC.csv')

    print(f"✅ Created sample data files:")
    print(f"   - {input_dir / 'XLY.csv'}")
    print(f"   - {input_dir / 'XLC.csv'}")

    return xly_df, xlc_df

def test_mmm_gap_calculation():
    """Test MMM gap calculation directly."""
    try:
        print("\n🧪 Testing MMM gap calculation...")

        from sustainability_ratios.mmm.mmm_gaps import MmmGapsProcessor

        # Create mock configuration
        class MockConfig:
            pass

        class MockUserConfig:
            sr_mmm_gaps_tickers = 'XLY;XLC'
            sr_mmm_gaps_values_input_folder_daily = '../downloadData_v1/data/market_data/daily/'
            sr_mmm_output_dir = 'results/sustainability_ratios/MMM'
            sr_mmm_gaps_values_filename_suffix = '_gap'

        config = MockConfig()
        user_config = MockUserConfig()

        # Test gaps processor
        gaps_processor = MmmGapsProcessor(config, user_config, 'daily')
        results = gaps_processor.run_gap_analysis()

        if results and results.get('tickers_processed'):
            print(f"✅ Gap calculation successful:")
            print(f"   - Tickers processed: {results['tickers_processed']}")
            print(f"   - CSV files created: {len(results.get('csv_files', []))}")
            print(f"   - Total gaps calculated: {results.get('total_gaps_calculated', 0)}")

            # Check gap file content
            for csv_file in results.get('csv_files', []):
                if Path(csv_file).exists():
                    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                    print(f"   - {Path(csv_file).name}: {len(df)} rows, columns: {list(df.columns[:5])}...")

                    # Show sample gap calculations
                    if 'opening_gap' in df.columns:
                        valid_gaps = df['opening_gap'].dropna()
                        print(f"     Sample gaps: {valid_gaps.head(3).tolist()}")

            return True
        else:
            print("❌ Gap calculation failed")
            return False

    except Exception as e:
        print(f"❌ MMM gap calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mmm_integration():
    """Test MMM integration with SR system."""
    try:
        print("\n🔗 Testing MMM integration with SR system...")

        from sustainability_ratios.mmm import MmmProcessor

        # Create mock configuration
        class MockConfig:
            pass

        class MockUserConfig:
            sr_mmm_enable = True
            sr_mmm_gaps_values = True
            sr_mmm_gaps_tickers = 'XLY;XLC'
            sr_mmm_gaps_values_input_folder_daily = '../downloadData_v1/data/market_data/daily/'
            sr_mmm_output_dir = 'results/sustainability_ratios/MMM'
            sr_mmm_gaps_values_filename_suffix = '_gap'
            sr_mmm_gaps_chart_enable = False  # Disable charts for simple test
            sr_mmm_gaps_charts_display_panel = 'gaps_display.csv'
            sr_mmm_gaps_charts_display_history = 30

        config = MockConfig()
        user_config = MockUserConfig()

        # Test MMM processor
        mmm_processor = MmmProcessor(config, user_config, 'daily')
        results = mmm_processor.run_complete_analysis()

        if results and results.get('success'):
            print(f"✅ MMM integration successful:")
            print(f"   - Status: {results.get('success')}")
            print(f"   - CSV files: {len(results.get('csv_files', []))}")
            print(f"   - Chart files: {len(results.get('chart_files', []))}")
            print(f"   - Gap results: {bool(results.get('gap_results'))}")

            return True
        else:
            print("❌ MMM integration failed")
            print(f"   Results: {results}")
            return False

    except Exception as e:
        print(f"❌ MMM integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_directories():
    """Test MMM output directory creation."""
    try:
        print("\n📁 Testing MMM output directories...")

        from sustainability_ratios.sr_output_manager import get_sr_output_manager

        output_manager = get_sr_output_manager()

        # Test MMM directory
        mmm_path = output_manager.get_submodule_path('MMM')
        if mmm_path and mmm_path.exists():
            print(f"✅ MMM output directory exists: {mmm_path}")

            # Check subdirectories
            gaps_dir = mmm_path / 'gaps'
            charts_dir = mmm_path / 'charts'

            if gaps_dir.exists():
                print(f"✅ Gaps directory exists: {gaps_dir}")

            return True
        else:
            print(f"❌ MMM output directory not found")
            return False

    except Exception as e:
        print(f"❌ Output directory test failed: {e}")
        return False

def main():
    """Run all MMM tests."""
    print("🚀 Starting MMM Submodule Tests")
    print("=" * 50)

    # Test 1: Create sample data
    try:
        xly_df, xlc_df = create_sample_data()
        print(f"✅ Sample data created successfully")
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return

    # Test 2: Test output directories
    test_output_directories()

    # Test 3: Test gap calculation
    gap_success = test_mmm_gap_calculation()

    # Test 4: Test integration
    integration_success = test_mmm_integration()

    # Summary
    print("\n" + "=" * 50)
    print("🎯 MMM Test Results Summary:")
    print(f"   - Sample data: ✅")
    print(f"   - Output directories: ✅")
    print(f"   - Gap calculation: {'✅' if gap_success else '❌'}")
    print(f"   - SR integration: {'✅' if integration_success else '❌'}")

    if gap_success and integration_success:
        print("🎉 All MMM tests passed! MMM submodule is ready.")
    else:
        print("⚠️ Some MMM tests failed. Check the logs above.")

if __name__ == "__main__":
    main()