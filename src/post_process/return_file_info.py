"""
File Resolution Module
======================

Maps logical filenames to physical file paths and identifies the latest files
based on YYYYMMDD date patterns in filenames.

This module does not modify existing code - it only reads from existing output directories.
"""

import os
import re
import glob
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_logical_file_mapping() -> Dict[str, str]:
    """
    Maps logical filenames to their corresponding output directories.

    This mapping is based on the existing directory structure without
    modifying any existing configuration.

    Returns:
        Dict[str, str]: Logical name -> directory path mapping
    """
    return {
        'basic_calculation': 'results/basic_calculation',
        'stage_analysis': 'results/stage_analysis',
        'rs_analysis': 'results/rs',
        'per_analysis': 'results/per',
        'pvb_screener': 'results/screeners/pvb',
        'atr1_screener': 'results/screeners/atr1',
        'drwish_screener': 'results/screeners/drwish',
        'giusti_screener': 'results/screeners/giusti',
        'minervini_screener': 'results/screeners/minervini',
        'stockbee_screener': 'results/screeners/stockbee',
        'qullamaggie_screener': 'results/screeners/qullamaggie',
        'adl_screener': 'results/screeners/adl',
        'guppy_screener': 'results/screeners/guppy',
        'gold_launch_pad_screener': 'results/screeners/gold_launch_pad',
        'rti_screener': 'results/screeners/rti'
    }


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract YYYYMMDD date from filename using regex pattern.

    Expected patterns:
    - basic_calculation_2-5_daily_20250905.csv
    - stage_analysis_2-5_weekly_20250901.csv

    Args:
        filename: The filename to extract date from

    Returns:
        datetime object if date found, None otherwise
    """
    # Pattern to match YYYYMMDD at the end of filename before .csv
    date_pattern = r'.*_(\d{8})\.csv$'
    match = re.search(date_pattern, filename)

    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, '%Y%m%d')
        except ValueError as e:
            logger.warning(f"Invalid date format in filename {filename}: {e}")
            return None

    return None


def get_latest_file(logical_filename: str, base_path: str = ".") -> Optional[str]:
    """
    Find the latest file for a given logical filename based on date in filename.

    Args:
        logical_filename: Logical name (e.g., 'basic_calculation', 'stage_analysis')
        base_path: Base directory path (default: current directory)

    Returns:
        Full path to the latest file, or None if no files found
    """
    # Get directory mapping
    mapping = get_logical_file_mapping()

    if logical_filename not in mapping:
        logger.error(f"Unknown logical filename: {logical_filename}")
        logger.info(f"Available logical filenames: {list(mapping.keys())}")
        return None

    # Get the directory for this logical file
    relative_dir = mapping[logical_filename]
    full_dir = Path(base_path) / relative_dir

    if not full_dir.exists():
        logger.warning(f"Directory does not exist: {full_dir}")
        return None

    # Find all CSV files in the directory
    csv_pattern = str(full_dir / "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        logger.warning(f"No CSV files found in: {full_dir}")
        return None

    # Extract dates and find the latest file
    latest_file = None
    latest_date = None

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        file_date = extract_date_from_filename(filename)

        if file_date:
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_file = file_path
        else:
            logger.debug(f"No date found in filename: {filename}")

    if latest_file:
        logger.info(f"Latest file for '{logical_filename}': {os.path.basename(latest_file)} ({latest_date.strftime('%Y-%m-%d')})")
        return latest_file
    else:
        logger.warning(f"No files with valid dates found for: {logical_filename}")
        return None


def list_available_files(logical_filename: str, base_path: str = ".") -> list:
    """
    List all available files for a logical filename with their dates.

    Args:
        logical_filename: Logical name to search for
        base_path: Base directory path

    Returns:
        List of tuples: (file_path, date) sorted by date (newest first)
    """
    mapping = get_logical_file_mapping()

    if logical_filename not in mapping:
        return []

    relative_dir = mapping[logical_filename]
    full_dir = Path(base_path) / relative_dir

    if not full_dir.exists():
        return []

    csv_files = glob.glob(str(full_dir / "*.csv"))
    file_dates = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        file_date = extract_date_from_filename(filename)
        if file_date:
            file_dates.append((file_path, file_date))

    # Sort by date (newest first)
    file_dates.sort(key=lambda x: x[1], reverse=True)
    return file_dates


if __name__ == "__main__":
    # Test the file resolution functionality
    import sys

    print("Testing file resolution...")

    # Test logical filename mapping
    mapping = get_logical_file_mapping()
    print(f"Available logical filenames: {list(mapping.keys())}")

    # Test file resolution for basic_calculation
    latest_basic = get_latest_file('basic_calculation')
    if latest_basic:
        print(f"Latest basic_calculation file: {latest_basic}")

    # Test file resolution for stage_analysis
    latest_stage = get_latest_file('stage_analysis')
    if latest_stage:
        print(f"Latest stage_analysis file: {latest_stage}")

    # List available files
    if len(sys.argv) > 1:
        logical_name = sys.argv[1]
        files = list_available_files(logical_name)
        print(f"\nAvailable {logical_name} files:")
        for file_path, date in files:
            print(f"  {os.path.basename(file_path)} - {date.strftime('%Y-%m-%d')}")