#!/usr/bin/env python3
"""
Test Market Breadth Calculation Module
=====================================

Tests the MarketBreadthCalculator functionality with real market data.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.market_breadth_calculation import MarketBreadthCalculator
from src.models import MarketBreadthConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_market_breadth_calculator():
    """
    Test the MarketBreadthCalculator with available market data.
    """
    print("🧪 Testing Market Breadth Calculator")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        
        # Create market breadth configuration
        breadth_config = MarketBreadthConfig(
            ma_periods=[20, 50, 200],
            daily_new_highs_threshold=100,
            strong_advance_threshold=70.0,
            default_universe="all"
        )
        
        print(f"📊 Configuration loaded:")
        print(f"   • MA periods: {breadth_config.ma_periods}")
        print(f"   • New highs threshold: {breadth_config.daily_new_highs_threshold}")
        print(f"   • Default universe: {breadth_config.default_universe}")
        print()
        
        # Initialize calculator
        calculator = MarketBreadthCalculator(config)
        
        # Check if ticker universe exists
        results_dir = config.directories.get('RESULTS_DIR', Path('results'))
        universe_file = results_dir / 'ticker_universes' / 'ticker_universe_all.csv'
        
        if not universe_file.exists():
            print(f"⚠️  Master universe file not found: {universe_file}")
            print("   Creating a test universe with available market data...")
            
            # Look for available market data
            market_data_dir = config.directories.get('MARKET_DATA_DIR', Path('data/market_data'))
            daily_dir = market_data_dir / 'daily'
            
            if daily_dir.exists():
                available_tickers = []
                for file in daily_dir.glob('*.csv'):
                    if file.stat().st_size > 0:  # Non-empty files
                        available_tickers.append(file.stem)
                
                if available_tickers:
                    # Create a temporary universe file for testing
                    os.makedirs(universe_file.parent, exist_ok=True)
                    test_universe_df = pd.DataFrame({
                        'ticker': available_tickers[:50],  # Use first 50 tickers for testing
                        'filter_applied': 'test_universe',
                        'universe_name': 'ticker_universe_all',
                        'generation_source': 'test_script'
                    })
                    test_universe_df.to_csv(universe_file, index=False)
                    print(f"   ✅ Created test universe with {len(available_tickers[:50])} tickers")
                else:
                    print("   ❌ No market data files found")
                    return False
            else:
                print(f"   ❌ Market data directory not found: {daily_dir}")
                return False
        
        # Test market breadth calculations
        print("🔍 Running market breadth calculations...")
        
        results_df = calculator.calculate_all_breadth_indicators(
            universe_name="all",
            lookback_days=252
        )
        
        if results_df.empty:
            print("❌ No breadth calculations were generated")
            return False
        
        print(f"✅ Successfully calculated breadth indicators!")
        print(f"   • Date range: {results_df.index.min().strftime('%Y-%m-%d')} to {results_df.index.max().strftime('%Y-%m-%d')}")
        print(f"   • Total trading days: {len(results_df)}")
        print(f"   • Total indicators: {len(results_df.columns)}")
        print()
        
        # Display key indicator categories
        breadth_columns = [col for col in results_df.columns if 'market_breadth' in col]
        
        ad_indicators = [col for col in breadth_columns if any(x in col.lower() for x in ['advance', 'decline', 'ad_'])]
        hl_indicators = [col for col in breadth_columns if any(x in col.lower() for x in ['high', 'low', 'hl_'])]
        ma_indicators = [col for col in breadth_columns if 'ma_' in col.lower()]
        threshold_indicators = [col for col in breadth_columns if any(x in col for x in ['_gt_', '_lt_', 'strong_', 'weak_'])]
        
        print("📈 Indicator Categories:")
        print(f"   • Advance/Decline indicators: {len(ad_indicators)}")
        print(f"   • New Highs/Lows indicators: {len(hl_indicators)}")
        print(f"   • Moving Average breadth: {len(ma_indicators)}")
        print(f"   • Threshold indicators: {len(threshold_indicators)}")
        print()
        
        # Show sample of latest data
        if len(results_df) > 0:
            latest_data = results_df.iloc[-1]
            print("📊 Latest Breadth Readings:")
            
            key_indicators = [
                'market_breadth_advance_pct',
                'market_breadth_252day_new_highs',
                'market_breadth_252day_new_lows',
                'market_breadth_20day_new_highs',
                'market_breadth_20day_new_lows',
                'market_breadth_63day_new_highs',
                'market_breadth_63day_new_lows',
                'market_breadth_pct_above_ma_50',
                'market_breadth_daily_252day_new_highs_gt_100',
                'market_breadth_daily_20day_new_highs_gt_100',
                'market_breadth_daily_63day_new_highs_gt_100'
            ]
            
            for indicator in key_indicators:
                if indicator in latest_data.index:
                    value = latest_data[indicator]
                    print(f"   • {indicator.replace('market_breadth_', '')}: {value}")
            print()
        
        # Test saving functionality
        print("💾 Testing file output...")
        output_file = calculator.save_breadth_calculations(
            results_df, 
            universe_name="all", 
            file_suffix="test"
        )
        
        if output_file and os.path.exists(output_file):
            print(f"✅ Successfully saved results to: {Path(output_file).name}")
            
            # Check file content
            saved_df = pd.read_csv(output_file)
            print(f"   • File contains {len(saved_df)} rows and {len(saved_df.columns)} columns")
        else:
            print("❌ Failed to save results file")
            return False
        
        # Test summary generation
        print("📋 Testing summary generation...")
        summary = calculator.get_breadth_summary(results_df)
        
        if summary and 'error' not in summary:
            print("✅ Successfully generated breadth summary:")
            if 'calculation_period' in summary:
                period = summary['calculation_period']
                print(f"   • Period: {period['start_date']} to {period['end_date']} ({period['total_days']} days)")
            
            if 'breadth_indicators' in summary:
                indicators = summary['breadth_indicators']
                print(f"   • Total indicators: {indicators['total_indicators']}")
                print(f"   • A/D indicators: {indicators['advance_decline_indicators']}")
                print(f"   • 252-day H/L indicators: {indicators.get('252day_new_highs_lows_indicators', 0)}")
                print(f"   • 20-day H/L indicators: {indicators.get('20day_new_highs_lows_indicators', 0)}")
                print(f"   • 63-day H/L indicators: {indicators.get('63day_new_highs_lows_indicators', 0)}")
                print(f"   • MA indicators: {indicators['ma_breadth_indicators']}")
                print(f"   • Threshold indicators: {indicators['threshold_indicators']}")
        else:
            print("❌ Failed to generate summary")
            return False
        
        print()
        print("🎉 All market breadth calculation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.exception("Test error details:")
        return False


if __name__ == "__main__":
    print("Market Breadth Calculation Test")
    print("==============================")
    print()
    
    success = test_market_breadth_calculator()
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)