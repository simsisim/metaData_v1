#!/usr/bin/env python3

import sys
import os
import pandas as pd
sys.path.append('src')

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from src.market_pulse.indicators.breadth_analyzer import BreadthAnalyzer
        from src.config import Config
        from src.user_defined_data import UserConfiguration
        
        # Read user configuration from CSV
        user_data = pd.read_csv('user_data.csv', header=None)
        user_config = UserConfiguration()
        
        print("Setting up configuration...")
        config = Config()
        # user_config already set above
        
        print("Creating BreadthAnalyzer...")
        analyzer = BreadthAnalyzer(['SPY'], config, user_config)
        
        print("Running breadth analysis...")
        result = analyzer.run_analysis('daily', '2025-09-05')
        
        if result.get('success'):
            print("✅ Analysis completed successfully!")
        else:
            print(f"❌ Analysis failed: {result.get('error')}")
            
    except Exception as e:
        import traceback
        print(f"❌ EXCEPTION OCCURRED: {e}")
        print("🔍 FULL TRACEBACK:")
        traceback.print_exc()

if __name__ == "__main__":
    main()