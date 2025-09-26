#!/usr/bin/env python3
"""
Research Exact CSV Parsing for MMM Configuration
===============================================

Examine how the exact user_data_sr_mmm.csv file is parsed
and why Panel_2 in row 3 (mmm_test) is not being displayed.
"""

import pandas as pd
from pathlib import Path
import csv

def research_csv_parsing():
    """Research the exact CSV parsing behavior."""

    print("=" * 80)
    print("RESEARCH: EXACT CSV PARSING FOR MMM CONFIGURATION")
    print("=" * 80)

    csv_file = Path("/home/imagda/_invest2024/python/metaData_v1/user_data_sr_mmm.csv")

    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return

    print(f"\n1. File Analysis: {csv_file}")

    # Read raw file content
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    print(f"   Total lines: {len(lines)}")
    for i, line in enumerate(lines, 1):
        print(f"   Line {i}: {repr(line.rstrip())}")

    # Parse with pandas - different methods
    print(f"\n2. Pandas Parsing Analysis:")

    # Method 1: Standard read_csv
    try:
        df1 = pd.read_csv(csv_file)
        print(f"   Standard read_csv:")
        print(f"     Shape: {df1.shape}")
        print(f"     Columns: {list(df1.columns)}")
        print(f"     Index: {list(df1.index)}")

        print(f"\n   Row-by-row analysis:")
        for idx, row in df1.iterrows():
            file_name_id = row.get('#file_name_id', row.get('file_name_id', f'Row_{idx}'))
            panel_1 = row.get('Panel_1', 'N/A')
            panel_2 = row.get('Panel_2', 'N/A')

            print(f"     Row {idx}: {file_name_id}")
            print(f"       Panel_1: '{panel_1}'")
            print(f"       Panel_2: '{panel_2}'")
            print(f"       Panel_2 type: {type(panel_2)}")
            print(f"       Panel_2 is NaN: {pd.isna(panel_2)}")
            print(f"       Panel_2 is empty string: {panel_2 == ''}")

    except Exception as e:
        print(f"   ❌ Standard read_csv failed: {e}")

    # Method 2: Skip header comment
    try:
        df2 = pd.read_csv(csv_file, skiprows=0, comment='#')
        print(f"\n   With comment handling:")
        print(f"     Shape: {df2.shape}")
        print(f"     Columns: {list(df2.columns)}")

    except Exception as e:
        print(f"   ❌ Comment handling failed: {e}")

    # Method 3: Manual CSV parsing
    print(f"\n3. Manual CSV Parsing:")
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        print(f"   Total rows: {len(rows)}")
        for i, row in enumerate(rows):
            print(f"   Row {i}: {len(row)} columns")
            if i < len(rows):
                print(f"     {row}")

        # Focus on headers and data
        if len(rows) > 0:
            headers = rows[0]
            print(f"\n   Headers: {headers}")

            # Find Panel_2 position
            if 'Panel_2' in headers:
                panel_2_idx = headers.index('Panel_2')
                print(f"   Panel_2 column index: {panel_2_idx}")

                # Check each data row
                for i, row in enumerate(rows[1:], 1):
                    if len(row) > panel_2_idx:
                        panel_2_value = row[panel_2_idx]
                        print(f"   Row {i} Panel_2: '{panel_2_value}' (length: {len(panel_2_value)})")
                    else:
                        print(f"   Row {i} Panel_2: MISSING (row too short)")

    except Exception as e:
        print(f"   ❌ Manual CSV parsing failed: {e}")

    # Method 4: Character-by-character analysis of mmm_test row
    print(f"\n4. Character Analysis of mmm_test Row:")

    mmm_test_line = None
    with open(csv_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if 'mmm_test' in line:
                mmm_test_line = line.rstrip()
                print(f"   Found mmm_test at line {line_num}")
                break

    if mmm_test_line:
        print(f"   Raw line: {repr(mmm_test_line)}")
        print(f"   Length: {len(mmm_test_line)}")

        # Split by comma
        parts = mmm_test_line.split(',')
        print(f"   Split parts ({len(parts)}):")
        for i, part in enumerate(parts):
            print(f"     [{i}]: '{part}' (len: {len(part)})")

        # Expected positions based on header
        expected_positions = {
            0: 'file_name_id',
            1: 'chart_type',
            2: 'Panel_1',
            3: 'Panel_2',
            4: 'Panel_3',
            5: 'Panel_4',
            6: 'Panel_5',
            7: 'Panel_6',
            8: 'Panel_1_index',
            9: 'Panel_2_index',
            10: 'Panel_3_index',
            11: 'Panel_4_index',
            12: 'Panel_5_index',
            13: 'Panel_6_index'
        }

        print(f"\n   Mapped to expected positions:")
        for i, part in enumerate(parts):
            expected = expected_positions.get(i, f'Unknown_{i}')
            print(f"     {expected}: '{part}'")

    print(f"\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"📋 FILE STRUCTURE:")
    print(f"   - 4 lines total (header + 3 data rows)")
    print(f"   - Header starts with # (comment)")
    print(f"   - Row 4 (mmm_test) has Panel_2 = 'XLY_gap'")

    print(f"\n🔍 POTENTIAL ISSUES TO INVESTIGATE:")
    print(f"   1. Comment header handling (#file_name_id vs file_name_id)")
    print(f"   2. Empty trailing columns in mmm_test row")
    print(f"   3. CSV parsing behavior with comment headers")
    print(f"   4. Panel configuration validation logic")
    print(f"   5. Multi-panel display requirements")

    print(f"\n❓ KEY QUESTIONS:")
    print(f"   - Does SR system handle '#file_name_id' header correctly?")
    print(f"   - Does Panel_2 parsing depend on Panel_1_index/Panel_2_index?")
    print(f"   - Are there minimum requirements for panel display?")
    print(f"   - Does the system require both panels to have non-empty values?")

if __name__ == "__main__":
    research_csv_parsing()