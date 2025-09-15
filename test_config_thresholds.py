#!/usr/bin/env python3
"""
Test Market Breadth Configuration Thresholds
===========================================

Tests that market breadth calculation uses configurable threshold values.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
from dataclasses import dataclass

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.market_breadth_calculation import MarketBreadthCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MockUserConfig:
    """Mock user configuration with custom threshold values."""
    market_breadth_daily_252day_new_highs_threshold: int = 50  # Changed from default 100
    market_breadth_ten_day_success_threshold: int = 3  # Changed from default 5
    market_breadth_daily_20day_new_highs_threshold: int = 75  # Changed from default 100
    market_breadth_twenty_day_success_threshold: int = 15  # Changed from default 10
    market_breadth_daily_63day_new_highs_threshold: int = 25  # Changed from default 100
    market_breadth_sixty_three_day_success_threshold: int = 20  # Changed from default 30
    market_breadth_strong_ad_ratio_threshold: float = 1.5  # Changed from default 2.0
    market_breadth_weak_ad_ratio_threshold: float = 0.7  # Changed from default 0.5
    market_breadth_strong_advance_threshold: float = 80.0  # Changed from default 70.0
    market_breadth_weak_advance_threshold: float = 20.0  # Changed from default 30.0
    market_breadth_strong_ma_breadth_threshold: float = 85.0  # Changed from default 80.0
    market_breadth_weak_ma_breadth_threshold: float = 15.0  # Changed from default 20.0

def test_custom_thresholds():
    """Test market breadth calculation with custom threshold configuration."""
    print("🧪 Testing Custom Configuration Thresholds")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        
        # Create custom configuration with different thresholds
        custom_config = MockUserConfig()
        
        print(f"📊 Custom thresholds:")
        print(f"   • 252-day new highs threshold: {custom_config.market_breadth_daily_252day_new_highs_threshold}")
        print(f"   • 20-day new highs threshold: {custom_config.market_breadth_daily_20day_new_highs_threshold}")
        print(f"   • 63-day new highs threshold: {custom_config.market_breadth_daily_63day_new_highs_threshold}")
        print(f"   • Strong A/D ratio threshold: {custom_config.market_breadth_strong_ad_ratio_threshold}")
        print()
        
        # Initialize calculator
        calculator = MarketBreadthCalculator(config)
        
        # Test with small data subset
        results_df = calculator.calculate_all_breadth_indicators(
            universe_name="all",
            lookback_days=252,
            user_config=custom_config
        )
        
        if results_df.empty:
            print("❌ No breadth calculations were generated")
            return False
        
        print(f"✅ Successfully calculated breadth indicators with custom config!")
        print(f"   • Total indicators: {len(results_df.columns)}")
        
        # Check for expected column names with custom thresholds
        expected_custom_columns = [
            'market_breadth_daily_252day_new_highs_gt_50',  # Should use 50 instead of 100
            'market_breadth_daily_20day_new_highs_gt_75',   # Should use 75 instead of 100
            'market_breadth_daily_63day_new_highs_gt_25',   # Should use 25 instead of 100
            'market_breadth_ad_ratio_gt_1_5',               # Should use 1.5 instead of 2.0
            'market_breadth_ad_ratio_lt_0_7'                # Should use 0.7 instead of 0.5
        ]
        
        print("🔍 Checking custom threshold columns:")
        all_found = True
        for col in expected_custom_columns:
            if col in results_df.columns:
                print(f"   ✅ Found: {col}")
            else:
                print(f"   ❌ Missing: {col}")
                all_found = False
        
        if all_found:
            print("\n🎉 All custom threshold columns found! Configuration system working correctly.")
            
            # Show some sample data
            if len(results_df) > 0:
                latest_data = results_df.iloc[-1]
                print("\n📊 Latest readings with custom thresholds:")
                for col in expected_custom_columns:
                    if col in latest_data.index:
                        value = latest_data[col]
                        print(f"   • {col.replace('market_breadth_', '')}: {value}")
            
            return True
        else:
            print("\n❌ Some expected custom threshold columns are missing!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.exception("Test error details:")
        return False

if __name__ == "__main__":
    print("Market Breadth Custom Configuration Test")
    print("=======================================")
    print()
    
    success = test_custom_thresholds()
    
    if success:
        print("\n✅ Configuration threshold test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Configuration threshold test failed!")
        sys.exit(1)