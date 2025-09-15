#!/usr/bin/env python3
"""
Test Market Pulse Module
========================

Test script for the market_pulse.py module to verify functionality
with existing market data.
"""

import sys
sys.path.insert(0, 'src')

from src.config import Config
from src.user_defined_data import read_user_data
from src.market_pulse import MarketPulseAnalyzer, run_gmi_analysis_for_index
import pandas as pd
from pathlib import Path

def test_gmi_integration():
    """Test GMI integration with market pulse module"""
    print("Testing GMI Integration...")
    
    try:
        config = Config()
        
        # Test standalone GMI function
        result = run_gmi_analysis_for_index('QQQ', config, threshold=3)
        
        if result.get('success'):
            print(f"✅ GMI for QQQ: {result['current_signal']} (Score: {result['current_score']}/{result['max_score']})")
            return True
        else:
            print(f"❌ GMI test failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ GMI integration error: {e}")
        return False

def test_data_loading():
    """Test market data loading for target indexes"""
    print("\nTesting Index Data Loading...")
    
    try:
        config = Config()
        user_config = read_user_data()
        
        analyzer = MarketPulseAnalyzer(config, user_config)
        
        # Test loading each target index
        target_indexes = ['SPY', 'QQQ', 'IWM', '^DJI']
        loaded_indexes = []
        
        for index in target_indexes:
            data = analyzer._load_index_data(index)
            if data is not None:
                print(f"✅ {index}: {len(data)} days of data available")
                loaded_indexes.append(index)
            else:
                print(f"❌ {index}: No data available")
        
        print(f"\nSuccessfully loaded {len(loaded_indexes)}/{len(target_indexes)} target indexes")
        return len(loaded_indexes) > 0
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_universe_data():
    """Test universe data loading for breadth analysis"""
    print("\nTesting Universe Data Loading...")
    
    try:
        config = Config()
        user_config = read_user_data()
        
        analyzer = MarketPulseAnalyzer(config, user_config)
        universe_data = analyzer._load_universe_data()
        
        if universe_data:
            print(f"✅ Universe: {len(universe_data)} tickers loaded")
            
            # Show sample tickers
            sample_tickers = list(universe_data.keys())[:10]
            print(f"Sample tickers: {', '.join(sample_tickers)}")
            
            # Show data range for a sample ticker
            if sample_tickers:
                sample_data = universe_data[sample_tickers[0]]
                start_date = sample_data.index.min().strftime('%Y-%m-%d')
                end_date = sample_data.index.max().strftime('%Y-%m-%d')
                print(f"Data range ({sample_tickers[0]}): {start_date} to {end_date}")
            
            return True
        else:
            print("❌ No universe data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Universe loading error: {e}")
        return False

def test_ftd_dd_calculation():
    """Test FTD & DD calculation with real data"""
    print("\nTesting FTD & DD Calculation...")
    
    try:
        config = Config()
        user_config = read_user_data()
        
        analyzer = MarketPulseAnalyzer(config, user_config)
        
        # Test with SPY data
        ftd_dd_results = analyzer._run_ftd_dd_analysis('SPY')
        
        if ftd_dd_results.get('success'):
            dd_data = ftd_dd_results['distribution_days']
            ftd_data = ftd_dd_results['follow_through_days']
            health = ftd_dd_results['market_health']
            
            print(f"✅ FTD & DD Analysis Successful:")
            print(f"   Distribution Days: {dd_data['current_count']} (Warning: {dd_data['warning_level']})")
            print(f"   Market Health: {health['assessment']} - {health['reason']}")
            
            if ftd_data['latest_ftd']:
                latest_ftd = ftd_data['latest_ftd']
                print(f"   Latest FTD: {latest_ftd['date']} (+{latest_ftd['gain_pct']:.1f}%, Day {latest_ftd['days_from_bottom']})")
            
            return True
        else:
            print(f"❌ FTD & DD test failed: {ftd_dd_results.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ FTD & DD calculation error: {e}")
        return False

def test_net_highs_lows():
    """Test Net New Highs/Lows calculation"""
    print("\nTesting Net New Highs/Lows...")
    
    try:
        config = Config()
        user_config = read_user_data()
        
        analyzer = MarketPulseAnalyzer(config, user_config)
        
        # Test universe-based calculation
        universe_results = analyzer._run_net_highs_lows_analysis()
        
        if universe_results.get('success'):
            print(f"✅ Net Highs/Lows Analysis Successful:")
            print(f"   Universe Size: {universe_results['universe_size']} tickers")
            
            for tf_name, tf_data in universe_results['timeframes'].items():
                net = tf_data['net_new_highs']
                highs = tf_data['new_highs']
                lows = tf_data['new_lows']
                print(f"   {tf_name}: {net:+d} net ({highs} highs, {lows} lows)")
            
            breadth = universe_results['breadth_signal']
            print(f"   Breadth Signal: {breadth['signal']} ({breadth['breadth_percentage']:+.1f}%)")
            
            return True
        else:
            print(f"❌ Net Highs/Lows test failed: {universe_results.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Net Highs/Lows error: {e}")
        return False

def test_full_analysis():
    """Test complete market pulse analysis"""
    print("\nTesting Complete Market Pulse Analysis...")
    
    try:
        config = Config()
        user_config = read_user_data()
        
        analyzer = MarketPulseAnalyzer(config, user_config)
        results = analyzer.run_complete_market_analysis()
        
        if results.get('success'):
            print("✅ Complete Market Pulse Analysis Successful!")
            print(f"   Analyzed {len(results['indexes'])} indexes")
            print(f"   Generated {len(results['alerts'])} alerts")
            
            # Show summary
            summary_text = analyzer.get_market_pulse_summary()
            print("\n" + "="*50)
            print(summary_text)
            print("="*50)
            
            return True
        else:
            print(f"❌ Complete analysis failed: {results.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Complete analysis error: {e}")
        return False

def main():
    """Run all market pulse tests"""
    print("Market Pulse Module Test Suite")
    print("=" * 50)
    
    tests = [
        ("GMI Integration", test_gmi_integration),
        ("Data Loading", test_data_loading),
        ("Universe Data", test_universe_data),
        ("FTD & DD Calculation", test_ftd_dd_calculation),
        ("Net Highs/Lows", test_net_highs_lows),
        ("Full Analysis", test_full_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Market Pulse module is ready.")
    else:
        print(f"⚠️ {total - passed} tests failed. Check implementation.")

if __name__ == "__main__":
    main()