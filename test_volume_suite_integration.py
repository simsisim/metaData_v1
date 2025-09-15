#!/usr/bin/env python3
"""
Test Volume Suite Integration
=============================

Test that the volume suite screener works when integrated into the main system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

from src.run_screeners import run_screeners
from src.user_defined_data import UserConfiguration


def create_test_data():
    """Create realistic test data"""
    dates = pd.date_range('2024-01-01', '2025-08-31', freq='D')
    
    np.random.seed(42)
    n_days = len(dates)
    
    # Generate realistic TSLA-like data
    base_price = 200
    price_trend = np.cumsum(np.random.normal(0.1, 3, n_days))
    close = base_price + price_trend
    high = close * (1 + np.random.uniform(0, 0.03, n_days))
    low = close * (1 - np.random.uniform(0, 0.03, n_days))
    open_price = close + np.random.normal(0, 2, n_days)
    
    # Volume with realistic spikes
    base_volume = 50000000  # 50M average volume like TSLA
    volume = np.random.lognormal(np.log(base_volume), 0.4, n_days)
    
    # Add some significant volume spikes
    spike_days = np.random.choice(n_days, size=15, replace=False)
    volume[spike_days] *= np.random.uniform(2, 8, len(spike_days))
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(int)
    }, index=dates)
    
    return {'TSLA': df}


def test_volume_suite_integration():
    """Test volume suite integration in main screener system"""
    print("🧪 Testing Volume Suite Integration")
    print("===================================")
    
    # Create test data
    test_data = create_test_data()
    print(f"Created test data for TSLA")
    
    # Create test configuration with volume suite enabled
    config = UserConfiguration()
    config.volume_suite_enable = True
    config.volume_suite_hv_absolute = True
    config.volume_suite_hv_stdv = True
    config.volume_suite_enhanced_anomaly = True
    config.volume_suite_volume_indicators = True
    config.volume_suite_pvb_integration = False  # Skip PVB for simple test
    
    # Relax thresholds for test data
    config.volume_suite_hv_std_cutoff = 2  # Lower threshold
    config.volume_suite_stdv_cutoff = 3   # Lower threshold
    config.volume_suite_vroc_threshold = 20  # Lower threshold
    config.volume_suite_rvol_threshold = 1.5  # Lower threshold
    
    try:
        # Test output directory
        output_path = Path('test_results/volume_suite_integration')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run screeners
        print("\\nRunning integrated screeners with Volume Suite enabled...")
        total_hits = run_screeners(test_data, output_path, 'daily', config)
        
        print(f"\\n✅ Integration test completed!")
        print(f"Total screening hits: {total_hits}")
        
        # Check if volume suite files were created
        volume_suite_files = list(output_path.glob('*volume*'))
        if volume_suite_files:
            print(f"✅ Volume suite files created: {len(volume_suite_files)}")
            for file in volume_suite_files:
                print(f"  - {file.name}")
        else:
            print("ℹ️  No volume suite specific files found")
        
        # Check main screener results
        result_files = list(output_path.glob('screener_results_*.csv'))
        if result_files:
            latest_results = sorted(result_files)[-1]
            results_df = pd.read_csv(latest_results)
            volume_suite_results = results_df[results_df['screen_type'].str.contains('volume', case=False)]
            
            print(f"\\n📊 Volume Suite Results in main file:")
            print(f"  - Total volume-related signals: {len(volume_suite_results)}")
            if len(volume_suite_results) > 0:
                signal_types = volume_suite_results['screen_type'].value_counts()
                for screen_type, count in signal_types.items():
                    print(f"    • {screen_type}: {count} signals")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_volume_suite_integration()
    exit(0 if success else 1)