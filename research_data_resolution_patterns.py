#!/usr/bin/env python3

"""
Research how Panel and Overview modules handle data resolution.
Understand the flexible data resolution patterns used in working modules.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def research_data_resolution_patterns():
    """Research data resolution patterns in Panel and Overview modules."""

    print("🧪 RESEARCHING DATA RESOLUTION PATTERNS")
    print("=" * 50)

    try:
        # Research 1: Panel data resolution pattern
        print(f"🔍 RESEARCH 1: PANEL DATA RESOLUTION PATTERN")
        print(f"=" * 47)

        # Look at sr_calculations.py panel processing
        print(f"  📄 Examining sr_calculations.py panel data resolution...")

        # Read key sections from sr_calculations.py
        from pathlib import Path
        sr_calc_file = Path("src/sustainability_ratios/sr_calculations.py")

        if sr_calc_file.exists():
            with open(sr_calc_file, 'r') as f:
                content = f.read()

            # Look for data resolution logic
            print(f"  🔍 Panel data resolution logic:")

            # Find the _prepare_panel_data method or similar
            if "_prepare_panel_data" in content:
                print(f"    ✅ Found _prepare_panel_data method")
            else:
                print(f"    Looking for data preparation patterns...")

            # Look for patterns where panels get their data
            lines = content.split('\n')
            data_patterns = []
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['data_source', 'market_data', 'read_csv', 'load_data']):
                    if not line.strip().startswith('#') and line.strip():
                        data_patterns.append((i+1, line.strip()))

            print(f"    📊 Data access patterns found:")
            for line_num, line in data_patterns[:10]:  # Show first 10
                print(f"      {line_num}: {line}")

        print()

        # Research 2: Overview data resolution pattern
        print(f"🔍 RESEARCH 2: OVERVIEW DATA RESOLUTION PATTERN")
        print(f"=" * 47)

        overview_charts_file = Path("src/sustainability_ratios/overview/overview_charts.py")
        if overview_charts_file.exists():
            print(f"  📄 Examining overview_charts.py data resolution...")

            with open(overview_charts_file, 'r') as f:
                content = f.read()

            # Look for data resolution methods
            print(f"  🔍 Overview data resolution methods:")

            methods = []
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'def ' in line and any(keyword in line.lower() for keyword in ['data', 'load', 'get', 'read', 'prepare']):
                    methods.append((i+1, line.strip()))

            for line_num, method in methods:
                print(f"    {line_num}: {method}")

            # Look for file resolution patterns
            print(f"\n  🔍 File resolution patterns:")
            file_patterns = []
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in ['.csv', 'Path', 'exists', 'glob']) and not line.strip().startswith('#'):
                    file_patterns.append((i+1, line.strip()))

            for line_num, pattern in file_patterns[:8]:  # Show first 8
                print(f"    {line_num}: {pattern}")

        print()

        # Research 3: Current MMM data resolution issue
        print(f"🔍 RESEARCH 3: MMM DATA RESOLUTION ISSUE")
        print(f"=" * 43)

        mmm_charts_file = Path("src/sustainability_ratios/mmm/mmm_charts.py")
        if mmm_charts_file.exists():
            print(f"  📄 Examining mmm_charts.py data resolution...")

            with open(mmm_charts_file, 'r') as f:
                content = f.read()

            # Look for the _prepare_chart_data method
            print(f"  🔍 Current MMM data resolution logic:")

            lines = content.split('\n')
            in_prepare_method = False
            method_lines = []

            for i, line in enumerate(lines):
                if 'def _prepare_chart_data' in line:
                    in_prepare_method = True
                    method_lines.append((i+1, line.strip()))
                elif in_prepare_method:
                    if line.strip().startswith('def ') and 'def _prepare_chart_data' not in line:
                        break
                    if line.strip():
                        method_lines.append((i+1, line.strip()))

            print(f"    📊 Current _prepare_chart_data method:")
            for line_num, line in method_lines[:20]:  # Show first 20 lines
                print(f"      {line_num}: {line}")

        print()

        # Research 4: Compare data source strategies
        print(f"🔍 RESEARCH 4: DATA SOURCE STRATEGIES COMPARISON")
        print(f"=" * 49)

        print(f"  📊 PANEL Strategy:")
        print(f"    1. Uses market data files directly (ticker.csv)")
        print(f"    2. Looks in yf_daily_data_files directory")
        print(f"    3. Applies indicators during chart generation")
        print(f"    4. Flexible ticker matching")
        print(f"    5. No pre-calculation requirements")

        print(f"\n  📊 OVERVIEW Strategy:")
        print(f"    1. Pre-calculates overview values")
        print(f"    2. Stores in sr_overview output directory")
        print(f"    3. Chart generation reads calculated files")
        print(f"    4. Graceful fallback if files missing")
        print(f"    5. Separation: calculation tickers ≠ chart tickers")

        print(f"\n  📊 MMM Current Strategy (PROBLEMATIC):")
        print(f"    1. Pre-calculates gap data (good)")
        print(f"    2. Stores as ticker_gap.csv files (good)")
        print(f"    3. Chart generation ONLY looks for gap files (bad)")
        print(f"    4. No fallback to regular market data (bad)")
        print(f"    5. Rigid: chart tickers = calculation tickers (bad)")

        print()

        # Research 5: Recommended MMM improvements
        print(f"🎯 RESEARCH 5: RECOMMENDED MMM IMPROVEMENTS")
        print(f"=" * 46)

        print(f"  🔧 IMPROVED MMM Strategy Should Be:")
        print(f"    1. Pre-calculate gap data for specified tickers")
        print(f"    2. Store as ticker_gap.csv files")
        print(f"    3. Chart generation tries multiple data sources:")
        print(f"       a) First: Look for gap data (ticker_gap.csv)")
        print(f"       b) Second: Fallback to regular market data (ticker.csv)")
        print(f"       c) Third: Skip gracefully if no data")
        print(f"    4. Allow chart config to reference ANY ticker")
        print(f"    5. Flexible data resolution like Panel/Overview")

        print(f"\n  📝 SPECIFIC CODE CHANGES NEEDED:")
        print(f"    1. Update _prepare_chart_data() in mmm_charts.py")
        print(f"    2. Add fallback logic: gap data → market data → skip")
        print(f"    3. Make data source matching more flexible")
        print(f"    4. Add graceful error handling for missing data")

        print(f"\n  📋 CURRENT vs DESIRED BEHAVIOR:")
        print(f"    Current:")
        print(f"      sr_mmm_gaps_tickers = 'XLY;XLC'")
        print(f"      Chart config: SPY + EMA(20)")
        print(f"      Result: ❌ Error - SPY_gap.csv not found")

        print(f"\n    Desired:")
        print(f"      sr_mmm_gaps_tickers = 'XLY;XLC'")
        print(f"      Chart config: SPY + EMA(20)")
        print(f"      Result: ✅ Uses SPY.csv + applies EMA, or uses XLY_gap.csv if SPY not available")

        print(f"\n    More Flexibility:")
        print(f"      sr_mmm_gaps_tickers = 'XLY;XLC'")
        print(f"      Chart config: XLY + EMA(20) (gap data available)")
        print(f"      Result: ✅ Uses XLY_gap.csv + gap analysis + EMA")

        return True

    except Exception as e:
        print(f"❌ Error researching data resolution patterns: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = research_data_resolution_patterns()
    if success:
        print(f"\n🎉 Data resolution patterns research completed!")
        print(f"\n📋 SUMMARY: MMM needs flexible data resolution like Panel/Overview modules")
    else:
        print(f"\n💥 Data resolution patterns research failed!")