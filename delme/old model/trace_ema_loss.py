#!/usr/bin/env python3
"""
Trace where EMA data gets lost between processing and chart generation.
This script will help identify the exact transformation that removes EMA columns.
"""

import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def inject_panel_result_tracing():
    """Inject tracing into the SR system to catch where EMA data gets lost."""

    print("🔍 INJECTING PANEL RESULT TRACING")
    print("="*60)

    try:
        # We need to patch the function that assembles panel results for chart generation
        # Based on the console output, the issue is between:
        # 1. Data processing: FINAL RESULT_DATA COLUMNS: [..., 'EMA_ema', 'EMA_price', 'EMA_signals']
        # 2. Chart generation: result_keys=['price', 'metadata']

        # The most likely place is where the processed data gets converted into panel results
        # Let's look for this conversion function

        from src.sustainability_ratios import sr_market_data

        # Get the original process_decomposed_data_source function
        original_func = sr_market_data.process_decomposed_data_source

        def traced_process_decomposed_data_source(*args, **kwargs):
            """Traced version to see what gets returned."""
            result = original_func(*args, **kwargs)

            print(f"🔍 PROCESS_DECOMPOSED_DATA_SOURCE RESULT:")
            if hasattr(result, 'columns'):
                print(f"   Result columns: {list(result.columns)}")
            else:
                print(f"   Result type: {type(result)}")
                print(f"   Result: {result}")

            return result

        # Monkey patch the function
        sr_market_data.process_decomposed_data_source = traced_process_decomposed_data_source

        print("✅ Successfully injected tracing into process_decomposed_data_source")

        # Also check if there's a function that converts DataFrame to dict
        print("\n💡 The issue is likely in a function that converts:")
        print("   DataFrame[Open, High, Low, Close, Volume, EMA_ema, EMA_price, EMA_signals]")
        print("   ↓")
        print("   Dict{'price': Series, 'metadata': dict}")
        print("\n🎯 This conversion strips out the EMA columns!")

    except Exception as e:
        print(f"❌ Error during tracing injection: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    inject_panel_result_tracing()

    print("\n🔧 NEXT STEPS:")
    print("1. The issue is a data transformation that converts DataFrame to simplified dict")
    print("2. Find the function that creates {'price': Series, 'metadata': dict} format")
    print("3. Modify it to preserve EMA columns in the result")
    print("4. This will fix the missing EMA overlay issue")