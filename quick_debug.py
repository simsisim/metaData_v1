#!/usr/bin/env python3
"""
Quick debug script to check why the real SR system isn't generating EMA overlays.
This will investigate the actual data being passed to chart generation.
"""

import logging
import sys
from pathlib import Path

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quick_debug_injection.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def inject_debug_into_sr_system():
    """Inject debug logging into the current SR system to see what's happening."""

    print("🔍 QUICK DEBUG INJECTION")
    print("="*60)

    try:
        # Import the SR modules
        from src.sustainability_ratios.sr_dashboard_generator import create_multi_panel_chart
        from src.sustainability_ratios.sr_market_data import calculate_bundled_indicator

        print("✅ Successfully imported enhanced SR modules")

        # Check if we can find the most recent SR results
        results_dir = Path("./results/sustainability_ratios")
        if results_dir.exists():
            print(f"✅ Found results directory: {results_dir}")

            # Look for recent result files
            result_files = list(results_dir.glob("*20250922*.png"))
            print(f"📊 Recent chart files: {[f.name for f in result_files]}")

        # Try to inspect the current SR configuration
        panel_file = Path("./SR_EB/user_data_panel.csv")
        if panel_file.exists():
            print(f"✅ Found panel config: {panel_file}")
            with open(panel_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:10], 1):
                    if 'QQQ + EMA' in line:
                        print(f"📋 Line {i}: {line.strip()}")

        # Check what the issue might be
        print("\n🎯 POTENTIAL ISSUES:")
        print("1. SR system might not be calling calculate_bundled_indicator")
        print("2. Panel parsing might not detect bundled format")
        print("3. Data loading might fail silently")
        print("4. Chart generation might receive incomplete data")

        print("\n💡 RECOMMENDED NEXT STEPS:")
        print("1. Add temporary print() statements to main.py")
        print("2. Check if bundled format is detected during panel parsing")
        print("3. Verify if EMA data is actually loaded")
        print("4. Confirm chart generation receives expected data structure")

    except Exception as e:
        print(f"❌ ERROR during debug injection: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    inject_debug_into_sr_system()