#!/usr/bin/env python3
"""
Test Volume Suite Screener
===========================

Simple test to verify the volume suite screener components work correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test each component individually to avoid import issues
def test_components_individually():
    """Test components by importing them directly"""
    
    # Test 1: HV Absolute
    try:
        exec(open('src/screeners/volume_suite/HVAbsoluteETC.py').read(), globals())
        print("✅ HVAbsoluteETC component loaded successfully")
    except Exception as e:
        print(f"❌ HVAbsoluteETC import failed: {e}")
        return False
    
    # Test 2: HV StdDev
    try:
        exec(open('src/screeners/volume_suite/HVStdv.py').read(), globals())
        print("✅ HVStdv component loaded successfully")
    except Exception as e:
        print(f"❌ HVStdv import failed: {e}")
        return False
    
    # Test 3: Enhanced Volume Anomaly
    try:
        exec(open('src/screeners/volume_suite/enhanced_volume_anomaly.py').read(), globals())
        print("✅ Enhanced Volume Anomaly component loaded successfully")
    except Exception as e:
        print(f"❌ Enhanced Volume Anomaly import failed: {e}")
        return False
        
    # Test 4: Volume Indicators
    try:
        exec(open('src/screeners/volume_suite/volume_indicators.py').read(), globals())
        print("✅ Volume Indicators component loaded successfully")
    except Exception as e:
        print(f"❌ Volume Indicators import failed: {e}")
        return False
    
    return True


def create_test_data():
    """Create sample test data for volume suite testing"""
    # Create date range
    dates = pd.date_range('2024-01-01', '2025-08-31', freq='D')
    
    # Create sample OHLCV data for a few tickers
    tickers = ['AAPL', 'TSLA', 'NVDA']
    test_data = {}
    
    np.random.seed(42)  # For reproducible results
    
    for ticker in tickers:
        # Generate realistic-looking price and volume data
        n_days = len(dates)
        
        # Price data with trend
        base_price = np.random.uniform(100, 300)
        price_trend = np.cumsum(np.random.normal(0, 2, n_days))
        prices = base_price + price_trend
        
        # Generate OHLC from close prices
        close = prices
        high = close * (1 + np.random.uniform(0, 0.05, n_days))
        low = close * (1 - np.random.uniform(0, 0.05, n_days))
        open_price = close + np.random.uniform(-5, 5, n_days)
        
        # Volume data with some spikes
        base_volume = np.random.uniform(1000000, 5000000)
        volume = np.random.lognormal(np.log(base_volume), 0.5, n_days)
        
        # Add some volume spikes
        spike_days = np.random.choice(n_days, size=10, replace=False)
        volume[spike_days] *= np.random.uniform(3, 10, len(spike_days))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume.astype(int)
        }, index=dates)
        
        test_data[ticker] = df
    
    return test_data


def test_volume_suite_screener():
    """Test the volume suite screener with sample data"""
    print("🧪 Testing Volume Suite Screener")
    print("================================")
    
    # Test component loading first
    print("Testing component imports...")
    if not test_components_individually():
        return False
    
    # Create test data
    print("\\nCreating test data...")
    test_data = create_test_data()
    print(f"Created test data for {len(test_data)} tickers")
    
    # Configure volume suite screening
    config = {
        'enable_volume_suite': True,
        'volume_suite': {
            'enable_hv_absolute': True,
            'enable_hv_stdv': True,
            'enable_enhanced_anomaly': True,
            'enable_volume_indicators': True,
            'enable_pvb_integration': False,  # Skip PVB for simple test
            'save_individual_files': False,  # Don't save files during test
            
            # HV Absolute parameters
            'hv_month_cutoff': 12,
            'hv_day_cutoff': 3,
            'hv_std_cutoff': 8,
            'hv_min_volume': 50000,
            'hv_min_price': 10,
            
            # HV StdDev parameters
            'stdv_cutoff': 10,
            'stdv_min_volume': 100000,
            
            # Volume indicators parameters
            'vroc_threshold': 40,
            'rvol_threshold': 1.8,
            'rvol_extreme_threshold': 4.0,
            'mfi_overbought': 75,
            'mfi_oversold': 25,
            'vpt_threshold': 0.08
        },
        'volume_output_dir': 'test_results/volume_suite'
    }
    
    try:
        # Run volume suite components individually for testing
        print("\\nRunning volume suite components...")
        all_results = []
        
        # 1. Test HV Absolute
        print("Testing HV Absolute...")
        hv_absolute_params = {
            'month_cuttoff': 12,
            'day_cuttoff': 3,
            'std_cuttoff': 8,
            'min_stock_volume': 50000,
            'min_price': 10,
            'use_enhanced_filtering': True
        }
        hv_absolute_results = run_HVAbsoluteStrategy_Enhanced(test_data, hv_absolute_params)
        for result in hv_absolute_results:
            if result.get('Date') is not None:
                all_results.append({
                    'ticker': result['Ticker'],
                    'signal_date': result['Date'],
                    'signal_type': 'hv_absolute',
                    'volume': result['MaxVolume'],
                    'price': result.get('Close', 0)
                })
        
        # 2. Test HV StdDev
        print("Testing HV Standard Deviation...")
        hv_stdv_params = {
            'std_cuttoff': 10,
            'min_stock_volume': 100000
        }
        hv_stdv_results = run_HVStdvStrategy(test_data, hv_stdv_params)
        for result in hv_stdv_results:
            if result.get('Date') is not None:
                all_results.append({
                    'ticker': result['Ticker'],
                    'signal_date': result['Date'],
                    'signal_type': 'hv_stdv',
                    'volume': result['UnusualVolume'],
                    'price': result.get('Close', 0)
                })
        
        # 3. Test Enhanced Volume Anomaly
        print("Testing Enhanced Volume Anomaly...")
        enhanced_results = run_enhanced_volume_anomaly_detection(test_data)
        for result in enhanced_results:
            all_results.append({
                'ticker': result['ticker'],
                'signal_date': result['signal_date'],
                'signal_type': 'enhanced_anomaly',
                'volume': result['volume'],
                'price': result['price']
            })
        
        # 4. Test Volume Indicators
        print("Testing Volume Indicators...")
        volume_params = {
            'vroc_threshold': 40,
            'rvol_threshold': 1.8,
            'rvol_extreme_threshold': 4.0,
            'mfi_overbought_threshold': 75,
            'mfi_oversold_threshold': 25,
            'vpt_signal_threshold': 0.08
        }
        indicators_output = run_volume_indicators_analysis(test_data, volume_params)
        if indicators_output:
            signals = indicators_output.get('signals', {})
            for signal_type, signal_list in signals.items():
                for signal in signal_list:
                    all_results.append({
                        'ticker': signal['Ticker'],
                        'signal_date': signal['Date'],
                        'signal_type': f'volume_indicators_{signal_type}',
                        'volume': signal.get('Volume', 0),
                        'price': signal.get('Close', 0)
                    })
        
        results = all_results
        
        # Display results
        print(f"\\n✅ Volume Suite Test Completed!")
        print(f"Total signals found: {len(results)}")
        
        if results:
            # Group by signal type
            signal_types = {}
            for result in results:
                signal_type = result.get('signal_type', 'unknown')
                if signal_type not in signal_types:
                    signal_types[signal_type] = []
                signal_types[signal_type].append(result)
            
            print("\\nSignals by type:")
            for signal_type, signals in signal_types.items():
                print(f"  {signal_type}: {len(signals)} signals")
                
                # Show top 3 signals for each type
                for i, signal in enumerate(signals[:3]):
                    ticker = signal.get('ticker', 'N/A')
                    date = signal.get('signal_date', 'N/A')
                    strength = signal.get('strength', 'N/A')
                    print(f"    {i+1}. {ticker} on {date} ({strength})")
        else:
            print("No signals found in test data")
        
        return True
        
    except Exception as e:
        print(f"❌ Volume Suite Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_volume_suite_screener()
    exit(0 if success else 1)