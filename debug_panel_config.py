#!/usr/bin/env python3
"""
Debug Panel Configuration and File ID Extraction
=================================================

This script traces the exact panel configuration structure generated from the CSV
and debugs why file_name_id is not being used in filename generation.
"""

import json
import re
from typing import Dict, Any

def manual_csv_parse(csv_path: str) -> Dict[int, Dict[str, str]]:
    """
    Manually parse the CSV to understand the exact data structure.
    """
    print("🔧 Manual CSV parsing...")

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Find header line
    header_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('#file_name_id'):
            header_line_idx = i
            break

    if header_line_idx is None:
        print("❌ Header line not found")
        return {}

    # Parse header
    header = lines[header_line_idx].strip().lstrip('#').split(',')
    print(f"📋 Header: {header}")

    # Parse data rows
    rows = {}
    row_number = 1
    for i in range(header_line_idx + 1, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            # Simple CSV parsing (handles quoted values)
            values = []
            current_value = ""
            in_quotes = False

            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    values.append(current_value.strip())
                    current_value = ""
                else:
                    current_value += char

            # Add last value
            values.append(current_value.strip())

            # Create row dictionary
            row_data = {}
            for j, value in enumerate(values):
                if j < len(header):
                    row_data[header[j]] = value

            rows[row_number] = row_data
            print(f"📊 Row {row_number}: file_name_id='{row_data.get('file_name_id', 'NOT_FOUND')}'")
            row_number += 1

    return rows

def simulate_file_id_extraction(panel_config: Dict[str, Dict[str, Any]], row_number: int) -> str:
    """
    Simulate the _get_file_id_from_config function logic.
    """
    print(f"\n🔍 Simulating file ID extraction for row {row_number}...")
    print(f"Panel config keys: {list(panel_config.keys())}")

    # Replicate the exact logic from sr_calculations.py
    if panel_config:
        # Try to find file_name_id in configuration
        for panel_name, panel_info in panel_config.items():
            print(f"  🔹 Checking panel '{panel_name}':")
            print(f"      Keys: {list(panel_info.keys())}")

            if 'file_name_id' in panel_info:
                file_name_id = panel_info['file_name_id']
                print(f"      ✅ Found file_name_id: '{file_name_id}'")
                return file_name_id
            else:
                print(f"      ❌ No file_name_id in this panel")

            # Fallback: Use first panel's data_source as file_id
            if 'data_source' in panel_info:
                data_source = panel_info['data_source']
                print(f"      📄 Has data_source: '{data_source}'")

                # Extract meaningful name from data source (simplified logic)
                tickers = extract_tickers_from_data_source_simple(data_source)
                if tickers:
                    fallback_id = '_'.join(tickers[:2])
                    print(f"      ⚠️ Using fallback from tickers: '{fallback_id}'")
                    return fallback_id
                else:
                    fallback_id = data_source.replace('"', '').replace(' ', '_')[:20]
                    print(f"      ⚠️ Using fallback from data_source: '{fallback_id}'")
                    return fallback_id

    # Final fallback
    final_fallback = f"Panel_Row_{row_number}"
    print(f"  ❌ Using final fallback: '{final_fallback}'")
    return final_fallback

def extract_tickers_from_data_source_simple(data_source: str) -> list:
    """
    Simplified version of ticker extraction.
    """
    # Look for ticker-like patterns (2-5 uppercase letters)
    tickers = re.findall(r'\b[A-Z]{2,5}\b', data_source)
    return list(set(tickers))  # Remove duplicates

def create_mock_panel_config(csv_data: Dict[int, Dict[str, str]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Create a mock panel configuration similar to what sr_config_reader generates.
    """
    print("\n🏗️ Creating mock panel configuration...")

    panel_configs = {}

    for row_num, row_data in csv_data.items():
        print(f"\n📋 Processing row {row_num}...")

        file_name_id = row_data.get('file_name_id', '')
        print(f"   file_name_id from CSV: '{file_name_id}'")

        # Create mock panels (simplified)
        panel_config = {}

        # Main panel (Panel_1)
        panel_1_data = row_data.get('Panel_1', '')
        if panel_1_data:
            panel_config['Panel_1_main'] = {
                'data_source': panel_1_data,
                'timeframe': 'daily',
                'config_row': row_num,
                'position': 'main',
                'format_type': 'enhanced',
                'file_name_id': file_name_id  # This should be stored!
            }
            print(f"   ✅ Created Panel_1_main with file_name_id: '{file_name_id}'")

        # Indicator panel (if exists)
        panel_1_index = row_data.get('Panel_1_index', '')
        if panel_1_index:
            panel_config['Panel_1_A_above'] = {
                'data_source': panel_1_index,
                'timeframe': 'daily',
                'config_row': row_num,
                'position': 'above',
                'base_panel': 'Panel_1_main',
                'prefix': 'A_',
                'file_name_id': file_name_id  # This should also be stored!
            }
            print(f"   ✅ Created Panel_1_A_above with file_name_id: '{file_name_id}'")

        # Panel_2 if exists
        panel_2_data = row_data.get('Panel_2', '')
        if panel_2_data:
            panel_config['Panel_2_main'] = {
                'data_source': panel_2_data,
                'timeframe': 'daily',
                'config_row': row_num,
                'position': 'main',
                'format_type': 'simple',
                'file_name_id': file_name_id
            }
            print(f"   ✅ Created Panel_2_main with file_name_id: '{file_name_id}'")

        # Panel_2_index if exists (like B_MACD for SPY_MACD row)
        panel_2_index = row_data.get('Panel_2_index', '')
        if panel_2_index:
            panel_config['Panel_2_B_above'] = {
                'data_source': panel_2_index,
                'timeframe': 'daily',
                'config_row': row_num,
                'position': 'above',
                'base_panel': 'Panel_2_main',
                'prefix': 'B_',
                'file_name_id': file_name_id
            }
            print(f"   ✅ Created Panel_2_B_above with file_name_id: '{file_name_id}'")

        panel_configs[row_num] = panel_config

        print(f"   📊 Row {row_num} panel summary:")
        for panel_name in panel_config.keys():
            print(f"      - {panel_name}")

    return panel_configs

def test_filename_generation(file_id: str, row_number: int) -> str:
    """
    Test the filename generation logic.
    """
    # Simulate the generate_sr_filename function
    user_choice = "2-5"  # Mock user choice
    timeframe = "daily"
    latest_date = "20250905"

    # Clean file_id (remove special characters, spaces)
    clean_file_id = re.sub(r'[^\w\-_]', '_', file_id)
    clean_file_id = re.sub(r'_+', '_', clean_file_id)  # Remove multiple underscores
    clean_file_id = clean_file_id.strip('_')  # Remove leading/trailing underscores

    # Format: sr_{file_id}_row{row_number}_{user_choice}_{timeframe}_{date}.png
    filename = f"sr_{clean_file_id}_row{row_number}_{user_choice}_{timeframe}_{latest_date}.png"

    return filename

def test_real_csv_parsing():
    """
    Test how the real sr_config_reader parses the CSV.
    """
    print("\n" + "=" * 70)
    print("🔧 TESTING REAL CSV PARSING LOGIC")
    print("=" * 70)

    csv_path = 'SR_EB/user_data_panel.csv'

    # Simulate the actual sr_config_reader logic
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    header_line = None
    data_lines = []

    print("📄 Processing CSV lines...")
    for i, line in enumerate(lines):
        line = line.strip()
        print(f"Line {i+1}: '{line}' -> ", end="")

        if not line:
            print("EMPTY - skipped")
            continue

        # Skip quoted comment lines
        if line.startswith('"#'):
            print("QUOTED COMMENT - skipped")
            continue

        # Header line detection
        if (line.startswith('#timeframe') or line.startswith('timeframe') or
            line.startswith('#file_name_id') or line.startswith('file_name_id')):
            if header_line is None:  # Take first header found
                header_line = line.lstrip('#')  # Remove leading #
                print(f"HEADER FOUND: '{header_line}'")
            continue

        # Data line
        if not line.startswith('#') and ',' in line:
            data_lines.append(line)
            print(f"DATA LINE added")
        else:
            print("COMMENT or INVALID - skipped")

    print(f"\n📋 Final parsing results:")
    print(f"   Header line: '{header_line}'")
    print(f"   Data lines: {len(data_lines)}")
    for i, line in enumerate(data_lines):
        print(f"      {i+1}: '{line}'")

    # Create the CSV content
    if header_line and data_lines:
        csv_content = header_line + '\n' + '\n'.join(data_lines)
        print(f"\n📊 CSV content to be parsed:")
        print("---")
        print(csv_content)
        print("---")

        # Try to parse with pandas
        try:
            import pandas as pd
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            print(f"\n✅ Pandas parsing successful!")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Data:")
            print(df)

            # Check file_name_id column
            if 'file_name_id' in df.columns:
                print(f"\n✅ file_name_id column found!")
                print(f"   Values: {df['file_name_id'].tolist()}")
            else:
                print(f"\n❌ file_name_id column NOT found")
                print(f"   Available columns: {list(df.columns)}")

        except Exception as e:
            print(f"\n❌ Pandas parsing failed: {e}")

def main():
    """
    Main debug function.
    """
    print("🐛 DEBUGGING PANEL CONFIG AND FILE_NAME_ID INTEGRATION")
    print("=" * 70)

    csv_path = 'SR_EB/user_data_panel.csv'

    # Step 1: Test real CSV parsing logic
    test_real_csv_parsing()

    # Step 2: Parse CSV manually
    csv_data = manual_csv_parse(csv_path)

    # Step 3: Create mock panel configuration
    panel_configs = create_mock_panel_config(csv_data)

    # Step 4: Test file ID extraction for each row
    print("\n" + "=" * 70)
    print("🧪 TESTING FILE ID EXTRACTION")
    print("=" * 70)

    for row_num, panel_config in panel_configs.items():
        print(f"\n{'='*50}")
        print(f"ROW {row_num} ANALYSIS")
        print(f"{'='*50}")

        # Show panel config structure
        print("📋 Panel Configuration:")
        for panel_name, panel_info in panel_config.items():
            print(f"  {panel_name}:")
            for key, value in panel_info.items():
                print(f"    {key}: {value}")

        # Test file ID extraction
        extracted_file_id = simulate_file_id_extraction(panel_config, row_num)

        # Test filename generation
        expected_filename = test_filename_generation(extracted_file_id, row_num)

        print(f"\n📁 FILENAME GENERATION:")
        print(f"   Extracted file_id: '{extracted_file_id}'")
        print(f"   Generated filename: '{expected_filename}'")

        # Check if it's correct
        expected_file_id = csv_data[row_num].get('file_name_id', '')
        if extracted_file_id == expected_file_id:
            print(f"   ✅ CORRECT: matches CSV file_name_id '{expected_file_id}'")
        else:
            print(f"   ❌ INCORRECT: should be '{expected_file_id}' from CSV")

if __name__ == "__main__":
    main()