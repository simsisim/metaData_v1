#!/usr/bin/env python3
"""
Test script to analyze CSV parser compatibility with new timeframe-less format.
"""

import pandas as pd
from pathlib import Path
import tempfile
import os

def test_header_detection():
    """Test how the CSV parser handles different header formats."""

    print("🧪 Testing Header Detection Logic")
    print("=" * 50)

    # Current CSV structure (no timeframe)
    current_csv = '''#file_name_id,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
QQQ_vs_SPY,"QQQ + EMA(QQQ, 10)",SPY,SPY:QQQ,,,,"A_PPO(12,26,9)_for_(QQQ)",,,,,'''

    # Old CSV structure (with timeframe)
    old_csv = '''#timeframe,file_name_id,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
daily,QQQ_vs_SPY,"QQQ + EMA(QQQ, 10)",SPY,SPY:QQQ,,,,"A_PPO(12,26,9)_for_(QQQ)",,,,,'''

    print("Current CSV header detection:")
    print("Header line:", current_csv.split('\n')[0])

    # Check what the parser looks for
    header_line = current_csv.split('\n')[0]
    print(f"Starts with '#timeframe': {header_line.startswith('#timeframe')}")
    print(f"Starts with 'timeframe': {header_line.startswith('timeframe')}")
    print(f"Contains 'timeframe': {'timeframe' in header_line}")

    print("\nOld CSV header detection:")
    print("Header line:", old_csv.split('\n')[0])

    old_header_line = old_csv.split('\n')[0]
    print(f"Starts with '#timeframe': {old_header_line.startswith('#timeframe')}")
    print(f"Starts with 'timeframe': {old_header_line.startswith('timeframe')}")

    print("\n❌ ISSUE: Parser only looks for lines starting with '#timeframe' or 'timeframe'")
    print("❌ Current CSV header starts with '#file_name_id' - not detected!")

def test_column_mapping():
    """Test how columns are mapped with the new structure."""

    print("\n🧪 Testing Column Mapping")
    print("=" * 50)

    # Simulate current CSV parsing
    current_data = "QQQ_vs_SPY,\"QQQ + EMA(QQQ, 10)\",SPY,SPY:QQQ,,,,"
    current_columns = ["file_name_id", "Panel_1", "Panel_2", "Panel_3", "Panel_4", "Panel_5", "Panel_6"]

    # Simulate how parser currently works (expects timeframe first)
    parser_expectation = ["timeframe", "file_name_id", "Panel_1", "Panel_2", "Panel_3", "Panel_4", "Panel_5"]

    current_values = current_data.split(',')[:len(current_columns)]

    print("Current CSV structure:")
    for i, (col, val) in enumerate(zip(current_columns, current_values)):
        print(f"  {col}: {val}")

    print("\nWhat parser expects to map:")
    for i, (expected_col, actual_val) in enumerate(zip(parser_expectation, current_values)):
        print(f"  {expected_col}: {actual_val} {'❌ WRONG' if expected_col != current_columns[i] else '✅'}")

    print("\n❌ ISSUE: Parser maps data to wrong columns due to missing timeframe!")

def test_single_row_parser():
    """Test how the single row parser handles the data."""

    print("\n🧪 Testing Single Row Parser Logic")
    print("=" * 50)

    # Simulate the problematic parsing
    print("Current parser logic (line 517 in sr_config_reader.py):")
    print("timeframe = str(row.iloc[0]).strip()  # Gets 'QQQ_vs_SPY' instead of timeframe")

    # Sample row data as parsed
    row_data = ["QQQ_vs_SPY", "\"QQQ + EMA(QQQ, 10)\"", "SPY", "SPY:QQQ", "", "", ""]
    columns = ["file_name_id", "Panel_1", "Panel_2", "Panel_3", "Panel_4", "Panel_5", "Panel_6"]

    print(f"\nRow data: {row_data}")
    print(f"Parser extracts timeframe from row.iloc[0]: '{row_data[0]}'")
    print("❌ ISSUE: Gets 'QQQ_vs_SPY' as timeframe instead of actual timeframe!")

    # Show the confusion this causes
    print(f"\nResulting panel configuration confusion:")
    print(f"  Expected timeframe: 'daily' (from global config)")
    print(f"  Parser gets: '{row_data[0]}' (file_name_id)")
    print(f"  Expected Panel_1: '{row_data[1]}'")
    print(f"  Parser assigns to wrong position due to column shift")

def analyze_required_fixes():
    """Analyze what changes are needed to fix the parser."""

    print("\n🔧 Required Parser Fixes")
    print("=" * 50)

    print("1. Header Detection Update:")
    print("   Current: looks for '#timeframe' or 'timeframe'")
    print("   Needed:  also look for '#file_name_id' or detect new format")
    print()

    print("2. Column Mapping Update:")
    print("   Current: expects [timeframe, file_name_id, Panel_1, Panel_2, ...]")
    print("   Needed:  handle [file_name_id, Panel_1, Panel_2, Panel_3, ...]")
    print()

    print("3. Timeframe Source Change:")
    print("   Current: reads from first column of each row")
    print("   Needed:  get from global SR_timeframe_* settings in user_data.csv")
    print()

    print("4. Single Row Parser Update:")
    print("   Current: timeframe = str(row.iloc[0])")
    print("   Needed:  file_name_id = str(row.iloc[0]), timeframe = get_global_timeframe()")
    print()

    print("5. Backward Compatibility:")
    print("   Needed:  detect old vs new format and handle both")

def create_test_data():
    """Create test data to validate fixes."""

    print("\n📊 Test Data for Validation")
    print("=" * 50)

    # New format (current)
    new_format = {
        'header': '#file_name_id,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index',
        'data': [
            'QQQ_vs_SPY,"QQQ + EMA(QQQ, 10)",SPY,SPY:QQQ,,,,"A_PPO(12,26,9)_for_(QQQ)",,,,,',
            ',PRICE_for_(QQQ),SPY,SPY,,,,,,,,,',
            ',QQQ,SPY,,,,,A_QQQ,,,,,',
        ]
    }

    # Expected parsing result
    expected_result = {
        'row_1': {
            'file_name_id': 'QQQ_vs_SPY',
            'timeframe': 'daily',  # From global config
            'panels': {
                'Panel_1': 'QQQ + EMA(QQQ, 10)',
                'Panel_2': 'SPY',
                'Panel_3': 'SPY:QQQ',
            },
            'indicators': {
                'Panel_1_index': 'A_PPO(12,26,9)_for_(QQQ)',
            }
        },
        'row_2': {
            'file_name_id': '',  # Empty, should generate default
            'timeframe': 'daily',
            'panels': {
                'Panel_1': 'PRICE_for_(QQQ)',
                'Panel_2': 'SPY',
                'Panel_3': 'SPY',
            }
        }
    }

    print("New format structure:")
    print(f"Header: {new_format['header']}")
    print("Data rows:")
    for i, row in enumerate(new_format['data'][:2], 1):
        print(f"  Row {i}: {row}")

    print(f"\nExpected parsing for Row 1:")
    for key, value in expected_result['row_1'].items():
        print(f"  {key}: {value}")

def main():
    """Run all compatibility tests."""

    print("🚀 SR CSV Parser Compatibility Analysis")
    print("=" * 70)

    test_header_detection()
    test_column_mapping()
    test_single_row_parser()
    analyze_required_fixes()
    create_test_data()

    print(f"\n📋 Summary")
    print("=" * 50)
    print("✅ Issue identified: Parser expects old CSV format with timeframe column")
    print("✅ Root cause: Header detection only looks for '#timeframe' pattern")
    print("✅ Impact: Column misalignment causes complete parsing failure")
    print("✅ Solution: Update parser to handle new format and use global timeframe")

    print(f"\n🔧 Implementation Plan:")
    print("1. Update _read_csv_with_comment_headers() header detection")
    print("2. Modify _parse_single_row() to handle new column order")
    print("3. Add global timeframe detection from user_data.csv")
    print("4. Implement backward compatibility for old format")
    print("5. Add validation tests for both formats")

if __name__ == "__main__":
    main()