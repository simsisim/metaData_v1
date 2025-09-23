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
import fnmatch
from datetime import datetime
from typing import Optional, Dict, List, Union
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


def determine_routing_strategy(logical_filename: str, template_routing_options: Optional[str]) -> str:
    """
    Determines how to route files based on logical name and routing options.

    Args:
        logical_filename: Logical name (e.g., 'basic_calculation', 'rs/per')
        template_routing_options: Routing pattern (e.g., '*.daily.*', '*.daily.ibd.*')

    Returns:
        str: Routing strategy ('single_file', 'single_file_filtered', 'multi_directory', 'enhanced_matrix')
    """
    has_routing = template_routing_options and template_routing_options.strip()
    is_multi_dir = '/' in logical_filename

    if not is_multi_dir and not has_routing:
        return 'single_file'  # Backward compatibility
    elif not is_multi_dir and has_routing:
        return 'single_file_filtered'  # Single directory with filtering
    elif is_multi_dir and not has_routing:
        return 'multi_directory'  # Multi-directory without filtering
    else:
        return 'enhanced_matrix'  # Multi-directory with filtering


def apply_routing_filter(file_list: List[str], routing_pattern: str) -> List[str]:
    """
    Apply routing pattern filter to a list of file paths.

    Args:
        file_list: List of file paths
        routing_pattern: Pattern like '*.daily.*', '*.daily.ibd.*', '*QQQ*'

    Returns:
        List of file paths matching the pattern
    """
    if not routing_pattern or not routing_pattern.strip():
        return file_list

    filtered_files = []
    for file_path in file_list:
        filename = os.path.basename(file_path)
        if fnmatch.fnmatch(filename, routing_pattern):
            filtered_files.append(file_path)

    logger.info(f"Routing filter '{routing_pattern}' matched {len(filtered_files)}/{len(file_list)} files")
    return filtered_files


def get_latest_file(logical_filename: str, base_path: str = ".", template_routing_options: Optional[str] = None):
    """
    Find the latest file(s) for a given logical filename with optional routing.

    Supports multi-directory syntax and routing patterns for enhanced file selection.

    Args:
        logical_filename: Logical name (e.g., 'basic_calculation', 'rs/per')
        base_path: Base directory path (default: current directory)
        template_routing_options: Optional routing pattern (e.g., '*.daily.ibd.*')

    Returns:
        - For single directory: Full path to latest file, or None if no files found
        - For multi-directory: Dict[str, str] or Dict[str, Dict[str, str]] depending on routing
    """
    strategy = determine_routing_strategy(logical_filename, template_routing_options)

    logger.info(f"Using routing strategy '{strategy}' for '{logical_filename}' with pattern '{template_routing_options}'")

    if strategy == 'single_file':
        # Existing behavior - backward compatibility
        return _get_latest_file_single_directory(logical_filename, base_path)

    elif strategy == 'single_file_filtered':
        # Single directory with routing filter
        return _get_latest_file_with_routing(logical_filename, base_path, template_routing_options)

    elif strategy == 'multi_directory':
        # Multi-directory without routing (existing behavior)
        return _get_latest_files_multi_directory(logical_filename, base_path)

    elif strategy == 'enhanced_matrix':
        # Multi-directory with routing filters
        return _get_enhanced_file_matrix(logical_filename, base_path, template_routing_options)

    else:
        logger.error(f"Unknown routing strategy: {strategy}")
        return None


def _get_latest_file_single_directory(logical_filename: str, base_path: str = ".") -> Optional[str]:
    """
    Original single-directory logic for backward compatibility.
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

    # Extract dates and find the latest file with priority logic
    latest_file = None
    latest_date = None

    # Group files by date to apply priority logic
    files_by_date = {}
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        file_date = extract_date_from_filename(filename)

        if file_date:
            if file_date not in files_by_date:
                files_by_date[file_date] = []
            files_by_date[file_date].append(file_path)
        else:
            logger.debug(f"No date found in filename: {filename}")

    # Find the latest date
    if files_by_date:
        latest_date = max(files_by_date.keys())
        candidate_files = files_by_date[latest_date]

        # Apply priority logic for files with the same date
        # Priority order: stocks > industries > sectors
        priority_keywords = ['_stocks_', '_industries_', '_sectors_']

        selected_file = None
        for keyword in priority_keywords:
            for file_path in candidate_files:
                if keyword in os.path.basename(file_path):
                    selected_file = file_path
                    break
            if selected_file:
                break

        # If no priority keyword found, use the first file
        if not selected_file:
            selected_file = candidate_files[0]

        latest_file = selected_file

    if latest_file:
        logger.info(f"Latest file for '{logical_filename}': {os.path.basename(latest_file)} ({latest_date.strftime('%Y-%m-%d')})")
        return latest_file
    else:
        logger.warning(f"No files with valid dates found for: {logical_filename}")
        return None


def _get_latest_files_multi_directory(logical_filename: str, base_path: str = ".") -> Dict[str, str]:
    """
    Handle multi-directory logical names like 'rs/per' or 'rs/per/basic_calculation'.

    Args:
        logical_filename: Multi-directory logical name (e.g., 'rs/per')
        base_path: Base directory path

    Returns:
        Dict mapping directory name to latest file path
    """
    # Split the logical filename by '/'
    directory_names = logical_filename.split('/')
    mapping = get_logical_file_mapping()
    result = {}

    logger.info(f"Processing multi-directory request: {logical_filename}")
    logger.info(f"Directory components: {directory_names}")

    for dir_name in directory_names:
        # Find the logical name that corresponds to this directory
        logical_name = None
        for key, value in mapping.items():
            if value.endswith(f'/{dir_name}') or value == f'results/{dir_name}':
                logical_name = key
                break

        if logical_name:
            logger.info(f"Found logical name '{logical_name}' for directory '{dir_name}'")
            latest_file = _get_latest_file_single_directory(logical_name, base_path)
            if latest_file:
                result[dir_name] = latest_file
                logger.info(f"Added {dir_name}: {os.path.basename(latest_file)}")
            else:
                logger.warning(f"No files found for directory: {dir_name}")
        else:
            logger.error(f"Unknown directory name in multi-directory request: {dir_name}")
            logger.info(f"Available directories: {[v.split('/')[-1] for v in mapping.values()]}")

    return result


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


def _get_latest_file_with_routing(logical_filename: str, base_path: str, routing_pattern: str) -> Optional[str]:
    """
    Get latest file from single directory with routing filter applied.

    Args:
        logical_filename: Single logical name (e.g., 'basic_calculation')
        base_path: Base directory path
        routing_pattern: Pattern to filter files (e.g., '*.daily.*')

    Returns:
        Path to latest file matching pattern, or None if no matches
    """
    mapping = get_logical_file_mapping()

    if logical_filename not in mapping:
        logger.error(f"Unknown logical filename: {logical_filename}")
        return None

    # Get all files in directory
    relative_dir = mapping[logical_filename]
    full_dir = Path(base_path) / relative_dir

    if not full_dir.exists():
        logger.warning(f"Directory does not exist: {full_dir}")
        return None

    # Get all CSV files
    csv_pattern = str(full_dir / "*.csv")
    all_files = glob.glob(csv_pattern)

    # Apply routing filter
    filtered_files = apply_routing_filter(all_files, routing_pattern)

    if not filtered_files:
        logger.warning(f"No files match routing pattern '{routing_pattern}' in {full_dir}")
        return None

    # Find latest file from filtered results
    latest_file = None
    latest_date = None

    files_by_date = {}
    for file_path in filtered_files:
        filename = os.path.basename(file_path)
        file_date = extract_date_from_filename(filename)

        if file_date:
            if file_date not in files_by_date:
                files_by_date[file_date] = []
            files_by_date[file_date].append(file_path)

    if files_by_date:
        latest_date = max(files_by_date.keys())
        candidate_files = files_by_date[latest_date]

        # Apply priority logic
        priority_keywords = ['_stocks_', '_industries_', '_sectors_']
        selected_file = None
        for keyword in priority_keywords:
            for file_path in candidate_files:
                if keyword in os.path.basename(file_path):
                    selected_file = file_path
                    break
            if selected_file:
                break

        if not selected_file:
            selected_file = candidate_files[0]

        latest_file = selected_file

    if latest_file:
        logger.info(f"Latest file for '{logical_filename}' with routing '{routing_pattern}': {os.path.basename(latest_file)}")
        return latest_file
    else:
        logger.warning(f"No valid files found for '{logical_filename}' with routing '{routing_pattern}'")
        return None


def _get_enhanced_file_matrix(logical_filename: str, base_path: str, routing_pattern: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Get enhanced file matrix for multi-directory with routing filters.

    Args:
        logical_filename: Multi-directory logical name (e.g., 'rs/per')
        base_path: Base directory path
        routing_pattern: Pattern to filter files

    Returns:
        Dictionary with structured file results based on routing complexity
    """
    # Get basic multi-directory result
    basic_result = _get_latest_files_multi_directory(logical_filename, base_path)

    if not basic_result:
        return {}

    enhanced_result = {}

    for dir_name, file_path in basic_result.items():
        # Get directory path
        mapping = get_logical_file_mapping()
        logical_name = None
        for key, value in mapping.items():
            if value.endswith(f'/{dir_name}') or value == f'results/{dir_name}':
                logical_name = key
                break

        if logical_name:
            # Get all files in this directory
            relative_dir = mapping[logical_name]
            full_dir = Path(base_path) / relative_dir

            if full_dir.exists():
                csv_pattern = str(full_dir / "*.csv")
                all_files = glob.glob(csv_pattern)

                # Apply routing filter
                filtered_files = apply_routing_filter(all_files, routing_pattern)

                if filtered_files:
                    # Check if we need to return multiple files or just one
                    if _should_return_multiple_files(routing_pattern, filtered_files):
                        # Group by benchmark, method, or other criteria
                        grouped_files = _group_filtered_files(filtered_files)
                        enhanced_result[dir_name] = grouped_files
                    else:
                        # Return single latest file with priority logic
                        latest_file = _select_latest_with_priority(filtered_files)
                        if latest_file:
                            enhanced_result[dir_name] = latest_file
                else:
                    logger.warning(f"No files match routing pattern '{routing_pattern}' in {dir_name}")

    return enhanced_result


def _should_return_multiple_files(routing_pattern: str, filtered_files: List[str]) -> bool:
    """
    Determine if routing pattern requires multiple files or single file.

    Args:
        routing_pattern: The routing pattern
        filtered_files: List of filtered files

    Returns:
        True if multiple files should be returned, False for single file
    """
    # If pattern doesn't specify benchmark/method specifically, return multiple files
    if not any(keyword in routing_pattern.lower() for keyword in ['qqq', 'spy', 'ibd', 'ma', 'stocks', 'industries', 'sectors']):
        return True

    # If we have multiple different types of files, return structured result
    benchmarks = set()
    methods = set()
    levels = set()

    for file_path in filtered_files:
        filename = os.path.basename(file_path).lower()
        if 'qqq' in filename:
            benchmarks.add('qqq')
        if 'spy' in filename:
            benchmarks.add('spy')
        if 'ibd' in filename:
            methods.add('ibd')
        if 'ma' in filename:
            methods.add('ma')
        if 'stocks' in filename:
            levels.add('stocks')
        if 'industries' in filename:
            levels.add('industries')
        if 'sectors' in filename:
            levels.add('sectors')

    # Return multiple if we have variety in benchmarks/methods/levels
    return len(benchmarks) > 1 or len(methods) > 1 or len(levels) > 1


def _group_filtered_files(filtered_files: List[str]) -> Dict[str, str]:
    """
    Group filtered files by benchmark, method, level for structured return.

    Args:
        filtered_files: List of filtered file paths

    Returns:
        Dictionary mapping classification to file path
    """
    grouped = {}
    files_by_date = {}

    # Group by date first
    for file_path in filtered_files:
        filename = os.path.basename(file_path)
        file_date = extract_date_from_filename(filename)
        if file_date:
            if file_date not in files_by_date:
                files_by_date[file_date] = []
            files_by_date[file_date].append(file_path)

    if not files_by_date:
        return grouped

    # Get latest date files
    latest_date = max(files_by_date.keys())
    latest_files = files_by_date[latest_date]

    # Group latest files by classification
    for file_path in latest_files:
        filename = os.path.basename(file_path).lower()

        # Build classification key
        key_parts = []

        # Benchmark
        if 'qqq' in filename:
            key_parts.append('QQQ')
        elif 'spy' in filename:
            key_parts.append('SPY')

        # Method
        if 'ibd' in filename:
            key_parts.append('ibd')
        elif 'ma' in filename:
            key_parts.append('ma')

        # Level
        if 'stocks' in filename:
            key_parts.append('stocks')
        elif 'industries' in filename:
            key_parts.append('industries')
        elif 'sectors' in filename:
            key_parts.append('sectors')

        if key_parts:
            key = '_'.join(key_parts)
            grouped[key] = file_path
        else:
            # Fallback to filename-based key
            base_name = os.path.basename(file_path).replace('.csv', '')
            grouped[base_name] = file_path

    return grouped


def _select_latest_with_priority(filtered_files: List[str]) -> Optional[str]:
    """
    Select latest file from filtered list using priority logic.

    Args:
        filtered_files: List of filtered file paths

    Returns:
        Path to selected file, or None
    """
    if not filtered_files:
        return None

    # Group by date
    files_by_date = {}
    for file_path in filtered_files:
        filename = os.path.basename(file_path)
        file_date = extract_date_from_filename(filename)
        if file_date:
            if file_date not in files_by_date:
                files_by_date[file_date] = []
            files_by_date[file_date].append(file_path)

    if not files_by_date:
        return filtered_files[0]  # Fallback

    # Get latest date files
    latest_date = max(files_by_date.keys())
    candidate_files = files_by_date[latest_date]

    # Apply priority logic
    priority_keywords = ['_stocks_', '_industries_', '_sectors_']
    for keyword in priority_keywords:
        for file_path in candidate_files:
            if keyword in os.path.basename(file_path):
                return file_path

    # Return first if no priority match
    return candidate_files[0]


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