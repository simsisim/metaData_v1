"""
SR Configuration Reader Module
==============================

Parses CSV panel configuration for multi-panel chart generation.
Reads user_data_panel.csv and extracts panel and indicator information.

Functions:
- parse_panel_config: Parse user_data_panel.csv
- load_sr_configuration: Load main SR settings
- validate_panel_config: Validate configuration
"""

import pandas as pd
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from ..indicators.indicator_parser import parse_panel_indicators, validate_indicator_string
from .enhanced_panel_parser import parse_enhanced_panel_entry, validate_enhanced_entry

logger = logging.getLogger(__name__)


def detect_csv_format(header_line: str) -> str:
    """
    Detect CSV format based on header structure.

    Args:
        header_line: The header line from CSV file

    Returns:
        'old_format': timeframe,file_name_id,Panel_1,...
        'new_format': file_name_id,Panel_1,Panel_2,...
    """
    if header_line.startswith('timeframe') or 'timeframe,' in header_line:
        return 'old_format'
    elif header_line.startswith('file_name_id') or header_line.startswith('#file_name_id'):
        return 'new_format'
    else:
        # Fallback detection
        if 'timeframe' in header_line and 'file_name_id' in header_line:
            return 'old_format'
        else:
            return 'new_format'


def get_global_timeframe(user_config=None) -> str:
    """
    Get timeframe from global SR_timeframe_* settings in user_data.csv.

    Args:
        user_config: User configuration object (optional)

    Returns:
        Active timeframe string ('daily', 'weekly', 'monthly')
    """
    if user_config:
        if hasattr(user_config, 'sr_timeframe_daily') and user_config.sr_timeframe_daily:
            return 'daily'
        elif hasattr(user_config, 'sr_timeframe_weekly') and user_config.sr_timeframe_weekly:
            return 'weekly'
        elif hasattr(user_config, 'sr_timeframe_monthly') and user_config.sr_timeframe_monthly:
            return 'monthly'

    # Default fallback
    return 'daily'


def _read_csv_with_comment_headers(csv_file_path: str) -> pd.DataFrame:
    """
    Read CSV file with special handling for headers that start with #.

    Args:
        csv_file_path: Path to CSV file

    Returns:
        DataFrame with proper headers
    """
    try:
        # Read all lines and find the header and data
        with open(csv_file_path, 'r') as f:
            lines = f.readlines()

        header_line = None
        data_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip quoted comment lines
            if line.startswith('"#'):
                continue

            # Header line detection - support both old and new formats
            if (line.startswith('#timeframe') or line.startswith('timeframe') or
                line.startswith('#file_name_id') or line.startswith('file_name_id')):
                if header_line is None:  # Take first header found
                    header_line = line.lstrip('#')  # Remove leading #
                continue

            # Data line (starts with daily, weekly, etc. - not a comment)
            if not line.startswith('#') and ',' in line:
                data_lines.append(line)

        if header_line is None:
            raise ValueError("No header line found in CSV")

        if not data_lines:
            logger.warning("No data lines found in CSV")
            return pd.DataFrame()

        # Create CSV content with header and data
        csv_content = header_line + '\n' + '\n'.join(data_lines)

        # Read into DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content))

        logger.debug(f"Successfully parsed CSV: header='{header_line}', data_lines={len(data_lines)}")
        return df

    except Exception as e:
        logger.error(f"Error reading CSV with comment headers: {e}")
        # Fallback to normal CSV reading
        return pd.read_csv(csv_file_path)


def parse_panel_config(csv_file_path: str) -> List[Dict[str, Dict[str, Any]]]:
    """
    Parse user_data_panel.csv to extract panel configuration.

    Supports two formats:

    Format 1 (Simple): Headers as column names
    timeframe,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
    daily,QQQ,SPY,,,,,,,,,,

    Format 2 (Complex): Headers as data values in row 3
    ,,,,,,,,,,,,,
    ,,,,,,,,,,,,,
    Panel_1,timeframe,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
    PPO,daily,QQQ,CPCE,XLY:XLP,CORR,SPY,,"PPO(12,6, 9)",,,,,"RSI(12, 6, 9)",

    Args:
        csv_file_path: Path to user_data_panel.csv

    Returns:
        List of panel configurations (one per CSV row)
        [
            {  # Row 1 configuration
                'Panel_1_QQQ': {
                    'panel_name': 'Panel_1',
                    'data_source': 'QQQ',
                    'indicator': '',
                    'timeframe': 'daily',
                    'panel_type': 'Panel_1'
                },
                ...
            },
            {  # Row 2 configuration
                ...
            }
        ]
    """
    try:
        # Read CSV file with special handling for comment headers
        df = _read_csv_with_comment_headers(csv_file_path)

        # Remove empty rows
        df = df.dropna(how='all')

        if df.empty:
            logger.warning(f"Empty CSV file: {csv_file_path}")
            return {}

        # Detect format: Check if Panel_1 is in column names (Format 1) or data values (Format 2)
        has_panel_columns = any('Panel_1' in str(col) for col in df.columns)

        if has_panel_columns:
            logger.info("Detected Format 1: Panel headers as column names")
            return _parse_simple_format_by_rows(df)
        else:
            logger.info("Detected Format 2: Panel headers as data values")
            return [_parse_complex_format(df)]  # Complex format returns single config

    except Exception as e:
        logger.error(f"Error parsing panel configuration from {csv_file_path}: {e}")
        return []


def load_sr_configuration(config, user_config) -> Dict[str, Any]:
    """
    Load SR configuration from user_data.csv and other sources.

    Args:
        config: Main system configuration
        user_config: User configuration object

    Returns:
        Dict with SR configuration settings
    """
    try:
        sr_config = {
            'sr_enable': getattr(user_config, 'sr_enable', False),
            'sr_output_dir': getattr(config.directories, 'SR_output_dir', 'results/sustainability_ratios'),
            'timeframes': ['daily'],  # Can be extended
            'default_lookback_days': 252,  # 1 year
            'chart_generation': True,
            'save_detailed_results': True
        }

        # Check if SR is enabled
        if not sr_config['sr_enable']:
            logger.info("SR module is disabled in configuration")

        return sr_config

    except Exception as e:
        logger.error(f"Error loading SR configuration: {e}")
        return {
            'sr_enable': False,
            'sr_output_dir': 'results/sustainability_ratios',
            'timeframes': ['daily'],
            'default_lookback_days': 252,
            'chart_generation': True,
            'save_detailed_results': True
        }


def validate_panel_config(panel_config: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Validate panel configuration and return any issues.
    Enhanced to support new format types and multi-ticker indicators.

    Args:
        panel_config: Panel configuration dict

    Returns:
        Dict with validation results
        {
            'errors': [list of error messages],
            'warnings': [list of warning messages],
            'valid_panels': [list of valid panel names],
            'format_summary': Dict with format type counts
        }
    """
    validation_results = {
        'errors': [],
        'warnings': [],
        'valid_panels': [],
        'format_summary': {}
    }

    if not panel_config:
        validation_results['errors'].append("No panel configuration found")
        return validation_results

    format_counts = {}

    # Check each panel
    for panel_key, panel_info in panel_config.items():
        panel_name = panel_info.get('panel_name', panel_key)
        format_type = panel_info.get('format_type', 'unknown')

        # Track format types
        format_counts[format_type] = format_counts.get(format_type, 0) + 1

        # Check required fields
        if not panel_info.get('data_source'):
            validation_results['errors'].append(f"{panel_name}: Missing data source")
            continue

        # Enhanced validation for data sources
        data_source = panel_info['data_source']
        _validate_data_source(data_source, panel_name, validation_results)

        # Enhanced indicator validation
        indicator = panel_info.get('indicator', '')
        if indicator:
            _validate_indicator_enhanced(indicator, panel_info, panel_name, validation_results)

        # Validate multi-ticker configurations
        if panel_info.get('is_multi_ticker', False):
            _validate_multi_ticker_config(panel_info, panel_name, validation_results)

        # Check timeframe
        timeframe = panel_info.get('timeframe', 'daily')
        if timeframe not in ['daily', 'weekly', 'monthly', '1d', '1wk', '1mo']:
            validation_results['warnings'].append(f"{panel_name}: Unusual timeframe: {timeframe}")

        # Validate panel positioning
        _validate_panel_positioning(panel_info, panel_name, validation_results)

        # If we get here, panel is valid
        validation_results['valid_panels'].append(panel_key)

    validation_results['format_summary'] = format_counts

    logger.info(f"Validation complete: {len(validation_results['valid_panels'])} valid panels, "
                f"{len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings")
    logger.info(f"Format distribution: {format_counts}")

    return validation_results


def _validate_data_source(data_source: str, panel_name: str, validation_results: Dict[str, List[str]]) -> None:
    """Validate data source format with enhanced support for ratios and multi-ticker."""
    try:
        if ':' in data_source:
            # Ratio format (e.g., XLY:XLP)
            parts = data_source.split(':')
            if len(parts) != 2:
                validation_results['warnings'].append(f"{panel_name}: Invalid ratio format: {data_source}")
            else:
                for part in parts:
                    if not part.strip():
                        validation_results['errors'].append(f"{panel_name}: Empty ticker in ratio: {data_source}")

        elif ',' in data_source:
            # Multi-ticker format (e.g., SPY,QQQ,IWM)
            tickers = [t.strip() for t in data_source.split(',')]
            if len(tickers) < 2:
                validation_results['warnings'].append(f"{panel_name}: Multi-ticker format with single ticker: {data_source}")
            for ticker in tickers:
                if not ticker:
                    validation_results['errors'].append(f"{panel_name}: Empty ticker in multi-ticker format: {data_source}")

        else:
            # Single ticker - validate format
            if not data_source.strip():
                validation_results['errors'].append(f"{panel_name}: Empty data source")
            elif not re.match(r'^[A-Z0-9\.\-\^_]+$', data_source.upper()):
                validation_results['warnings'].append(f"{panel_name}: Unusual ticker format: {data_source}")

    except Exception as e:
        validation_results['errors'].append(f"{panel_name}: Error validating data source: {str(e)}")


def _validate_indicator_enhanced(indicator: str, panel_info: Dict[str, Any], panel_name: str, validation_results: Dict[str, List[str]]) -> None:
    """Enhanced indicator validation supporting new indicator types."""
    try:
        from ..indicators.indicator_parser import INDICATOR_REGISTRY

        # Basic indicator validation
        if not validate_indicator_string(indicator) and indicator not in ['PRICE', 'RATIO']:
            validation_results['errors'].append(f"{panel_name}: Invalid indicator: {indicator}")
            return

        # Check if indicator is supported
        if indicator not in INDICATOR_REGISTRY:
            validation_results['warnings'].append(f"{panel_name}: Indicator {indicator} not in registry")

        # Validate indicator parameters if present
        parameters = panel_info.get('indicator_parameters', {})
        if parameters:
            _validate_indicator_parameters(indicator, parameters, panel_name, validation_results)

        # Special validation for multi-ticker indicators
        if indicator in ['RATIO', 'CORR', 'BETA'] and not panel_info.get('is_multi_ticker', False):
            validation_results['warnings'].append(f"{panel_name}: {indicator} typically requires multiple tickers")

    except Exception as e:
        validation_results['errors'].append(f"{panel_name}: Error validating indicator: {str(e)}")


def _validate_indicator_parameters(indicator: str, parameters: Dict[str, Any], panel_name: str, validation_results: Dict[str, List[str]]) -> None:
    """Validate indicator-specific parameters."""
    try:
        if indicator == 'PPO':
            required_params = ['fast_period', 'slow_period', 'signal_period']
            for param in required_params:
                if param not in parameters:
                    validation_results['warnings'].append(f"{panel_name}: Missing PPO parameter: {param}")
                elif not isinstance(parameters[param], (int, float)) or parameters[param] <= 0:
                    validation_results['errors'].append(f"{panel_name}: Invalid PPO {param}: {parameters[param]}")

        elif indicator == 'RSI':
            if 'period' in parameters:
                period = parameters['period']
                if not isinstance(period, (int, float)) or period <= 0 or period > 100:
                    validation_results['errors'].append(f"{panel_name}: Invalid RSI period: {period}")

        elif indicator in ['EMA', 'SMA']:
            if 'period' in parameters:
                period = parameters['period']
                if not isinstance(period, (int, float)) or period <= 0:
                    validation_results['errors'].append(f"{panel_name}: Invalid {indicator} period: {period}")

    except Exception as e:
        validation_results['warnings'].append(f"{panel_name}: Error validating {indicator} parameters: {str(e)}")


def _validate_multi_ticker_config(panel_info: Dict[str, Any], panel_name: str, validation_results: Dict[str, List[str]]) -> None:
    """Validate multi-ticker panel configuration."""
    try:
        tickers = panel_info.get('tickers', [])
        indicator = panel_info.get('indicator', '')

        if len(tickers) < 2:
            validation_results['errors'].append(f"{panel_name}: Multi-ticker config with insufficient tickers: {len(tickers)}")

        # Check if indicator supports multi-ticker
        multi_ticker_indicators = ['RATIO', 'CORR', 'BETA', 'SPREAD']
        if indicator and indicator not in multi_ticker_indicators:
            validation_results['warnings'].append(f"{panel_name}: Indicator {indicator} may not support multi-ticker mode")

        # Validate ticker format
        for ticker in tickers:
            if not ticker or not isinstance(ticker, str):
                validation_results['errors'].append(f"{panel_name}: Invalid ticker in multi-ticker config: {ticker}")

    except Exception as e:
        validation_results['errors'].append(f"{panel_name}: Error validating multi-ticker config: {str(e)}")


def _validate_panel_positioning(panel_info: Dict[str, Any], panel_name: str, validation_results: Dict[str, List[str]]) -> None:
    """Validate panel positioning and stacking configuration."""
    try:
        position = panel_info.get('position', 'main')
        base_panel = panel_info.get('base_panel')

        if position in ['above', 'below']:
            if not base_panel:
                logger.debug(f"{panel_name}: Positioned panel missing base_panel reference (orphaned positioned panel - allowed)")

            # Check if prefix is consistent with position
            prefix = panel_info.get('prefix')
            if prefix:
                expected_prefix = 'A_' if position == 'above' else 'B_'
                if prefix != expected_prefix:
                    validation_results['warnings'].append(f"{panel_name}: Position {position} inconsistent with prefix {prefix}")

        elif position == 'main':
            # Main panels shouldn't have positioning attributes
            if base_panel:
                validation_results['warnings'].append(f"{panel_name}: Main panel has unexpected base_panel reference")

    except Exception as e:
        validation_results['warnings'].append(f"{panel_name}: Error validating panel positioning: {str(e)}")


def get_data_sources_from_config(panel_config: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Extract all unique data sources from panel configuration.

    Args:
        panel_config: Panel configuration dict

    Returns:
        List of unique data sources (tickers, ratios, etc.)
    """
    data_sources = set()

    for panel_info in panel_config.values():
        data_source = panel_info.get('data_source')
        if data_source:
            data_sources.add(data_source)

    return sorted(list(data_sources))


def get_required_tickers_from_config(panel_config: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Extract all individual tickers needed for the configuration.
    This includes expanding ratios like XLY:XLP into [XLY, XLP] and
    extracting tickers from enhanced format indicators.

    Args:
        panel_config: Panel configuration dict

    Returns:
        List of unique ticker symbols needed
    """
    tickers = set()

    for panel_info in panel_config.values():
        data_source = panel_info.get('data_source')
        if not data_source:
            continue

        # Check if data source is enhanced format or bundled format
        if '_for_(' in data_source or panel_info.get('is_bundled', False):
            # Enhanced/bundled format - extract tickers from parsed data
            panel_tickers = panel_info.get('tickers', [])
            for ticker in panel_tickers:
                if ticker and ticker != 'AUTO_DETECT' and not any(char in ticker for char in ['(', ')', ',']):
                    tickers.add(ticker)
        elif ':' in data_source:
            # Legacy ratio format - split and add both tickers
            parts = data_source.split(':')
            for part in parts:
                if part.strip():
                    tickers.add(part.strip())
        elif ',' in data_source:
            # Multi-ticker format (e.g., SPY,QQQ,IWM)
            parts = data_source.split(',')
            for part in parts:
                if part.strip():
                    tickers.add(part.strip())
        else:
            # Single ticker
            tickers.add(data_source)

    return sorted(list(tickers))


def _parse_simple_format_by_rows(df: pd.DataFrame) -> List[Dict[str, Dict[str, Any]]]:
    """
    Parse simple format CSV and return separate configurations for each row.

    Args:
        df: DataFrame with panel configuration

    Returns:
        List of panel configurations (one per CSV row)
    """
    try:
        row_configurations = []

        # Process each data row separately
        for row_idx, row in df.iterrows():
            # Skip commented rows
            first_val = str(row.iloc[0]).strip()
            if first_val.startswith('#'):
                continue

            # Parse this single row as a separate configuration
            single_row_df = pd.DataFrame([row], columns=df.columns)
            # Convert row_idx to int if it's not already
            row_num = int(row_idx) + 2 if isinstance(row_idx, (int, float)) else len(row_configurations) + 2
            row_config = _parse_single_row(single_row_df.iloc[0], df.columns, row_num)

            if row_config:  # Only add non-empty configurations
                row_configurations.append(row_config)

        logger.info(f"Parsed {len(row_configurations)} row configurations (simple format)")
        return row_configurations

    except Exception as e:
        logger.error(f"Error parsing simple format by rows: {e}")
        return []


def _parse_single_row(row: pd.Series, columns: pd.Index, row_number: int) -> Dict[str, Dict[str, Any]]:
    """
    Parse a single CSV row into panel configuration.

    Args:
        row: Single row from DataFrame
        columns: Column names
        row_number: Row number for reference

    Returns:
        Panel configuration for this row only
    """
    try:
        # Determine format and extract data accordingly
        csv_format = detect_csv_format(','.join(columns))

        if csv_format == 'old_format':
            # Old format: timeframe, file_name_id, Panel_1, Panel_2, ...
            timeframe = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else 'daily'
            file_name_id = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ''
            panel_start_idx = 2
        else:
            # New format: file_name_id, Panel_1, Panel_2, Panel_3, ...
            file_name_id = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''
            timeframe = get_global_timeframe()  # From user_data.csv settings
            panel_start_idx = 1

        # Extract chart_type from CSV if available
        chart_type = 'candle'  # Default as specified by user
        for col_idx, col_name in enumerate(columns):
            col_name_str = str(col_name).strip().lower()
            if col_name_str == 'chart_type':
                chart_type_value = str(row.iloc[col_idx]).strip() if pd.notna(row.iloc[col_idx]) else ''
                if chart_type_value == '':
                    chart_type = 'candle'  # Default when empty
                elif chart_type_value.lower() in ['candle', 'candlestick']:
                    chart_type = 'candle'
                elif chart_type_value.lower() == 'line':
                    chart_type = 'line'
                elif chart_type_value.lower() == 'no_drawing':
                    chart_type = 'no_drawing'
                else:
                    chart_type = 'candle'  # Fallback to default
                break

        # Extract panel data sources and indicators for this row
        panel_data_sources = {}
        panel_indicators = {}

        for col_idx, col_name in enumerate(columns):
            col_name_str = str(col_name).strip()

            # Skip timeframe and file_name_id columns for panel processing
            if col_idx < panel_start_idx:
                continue

            # Panel data sources (Panel_1, Panel_2, etc.)
            if col_name_str.startswith('Panel_') and not col_name_str.endswith('_index'):
                if pd.notna(row.iloc[col_idx]) and str(row.iloc[col_idx]).strip():
                    data_source = str(row.iloc[col_idx]).strip()
                    if data_source not in ['', 'nan']:
                        # Any enhanced format is valid in data source position
                        panel_data_sources[col_name_str] = data_source

            # Panel indicators (Panel_1_index, Panel_2_index, etc.)
            elif col_name_str.endswith('_index'):
                if pd.notna(row.iloc[col_idx]) and str(row.iloc[col_idx]).strip():
                    indicator_string = str(row.iloc[col_idx]).strip()
                    if indicator_string not in ['', 'nan']:
                        # Use enhanced parser to handle all formats including bundling
                        parsed_entry = parse_enhanced_panel_entry(indicator_string)
                        if parsed_entry:
                            panel_indicators[col_name_str] = {
                                'original_string': indicator_string,
                                'parsed_data': parsed_entry
                            }
                        else:
                            logger.warning(f"Failed to parse indicator string: {indicator_string}")

                        # Legacy compatibility: also store original string for direct access
                        if (indicator_string.startswith('A_') or indicator_string.startswith('B_')) and len(indicator_string) > 2:
                            panel_indicators[col_name_str + '_legacy'] = indicator_string
                        elif validate_indicator_string(indicator_string):
                            panel_indicators[col_name_str + '_legacy'] = indicator_string

        # Create panel configuration entries for this row using enhanced parsing
        panel_config = {}

        # Process main panel data sources
        for panel_name, data_source in panel_data_sources.items():
            indicator_key = f"{panel_name}_index"
            indicator_info = panel_indicators.get(indicator_key, {})

            # Check if data source itself is enhanced format OR bundled format
            data_source_parsed = None
            if '_for_(' in data_source or '+' in data_source:
                data_source_parsed = parse_enhanced_panel_entry(data_source)

            # 🔧 FIX: Handle both bundled data source AND positioned indicators
            # First process the main panel data source (bundled or enhanced)
            main_panel_processed = False
            if data_source_parsed:
                if data_source_parsed.get('is_bundled', False):
                    _process_bundled_panel_entry(panel_config, panel_name, data_source_parsed, timeframe, row_number, file_name_id, chart_type)
                    main_panel_processed = True
                else:
                    _process_enhanced_data_source_panel(panel_config, panel_name, data_source_parsed, timeframe, row_number, file_name_id, chart_type)
                    main_panel_processed = True

            # Then handle indicator information (positioned panels or main panel indicators)
            if isinstance(indicator_info, dict) and 'parsed_data' in indicator_info:
                parsed_data = indicator_info['parsed_data']

                # Handle positioned panels (A_/B_) - these are always separate from main panel
                if parsed_data.get('position') in ['above', 'below']:
                    # This will be handled in the positioned panels section below
                    pass

                # Handle main panel indicators (only if main panel wasn't already processed)
                elif not main_panel_processed:
                    # Handle bundled format indicators
                    if parsed_data.get('format_type') == 'bundled':
                        _process_bundled_panel_entry(panel_config, panel_name, parsed_data, timeframe, row_number, file_name_id, chart_type)

                    # Handle enhanced format indicators
                    elif parsed_data.get('format_type') == 'enhanced':
                        _process_enhanced_panel_entry(panel_config, panel_name, data_source, parsed_data, timeframe, row_number, file_name_id, chart_type)

                    # Handle legacy formats that don't create positioned panels
                    elif parsed_data.get('format_type') in ['legacy_indicator', 'simple_ticker']:
                        _process_legacy_main_panel(panel_config, panel_name, data_source, parsed_data, timeframe, row_number, file_name_id, chart_type)
                    main_panel_processed = True

            # Fallback if no main panel was processed
            if not main_panel_processed:
                # Fallback to simple panel without indicators
                _process_simple_main_panel(panel_config, panel_name, data_source, timeframe, row_number, file_name_id, chart_type)

        # Process positioned panels (A_/B_) from enhanced parsing
        for indicator_key, indicator_info in panel_indicators.items():
            if isinstance(indicator_info, dict) and 'parsed_data' in indicator_info:
                parsed_data = indicator_info['parsed_data']
                logger.info(f"🔍 PROCESSING PANEL INDEX: {indicator_key}")
                logger.info(f"   parsed_data keys: {list(parsed_data.keys())}")
                logger.info(f"   position: {parsed_data.get('position')}")
                logger.info(f"   indicator: {parsed_data.get('indicator')}")
                logger.info(f"   tickers: {parsed_data.get('tickers')}")

                # Handle positioned panels (A_/B_) and bundled positioned panels
                if parsed_data.get('position') in ['above', 'below']:
                    base_panel_name = indicator_key.replace('_index', '')
                    logger.info(f"✅ POSITIONED PANEL DETECTED: {indicator_key} → {base_panel_name}")
                    if parsed_data.get('format_type') == 'bundled':
                        logger.info(f"   Processing as bundled positioned panel")
                        _process_bundled_panel_entry(panel_config, base_panel_name, parsed_data, timeframe, row_number, file_name_id, chart_type)
                    else:
                        logger.info(f"   Processing as regular positioned panel")
                        _process_positioned_panel(panel_config, base_panel_name, parsed_data, timeframe, row_number, file_name_id, chart_type)
                else:
                    logger.info(f"❌ NOT A POSITIONED PANEL: position={parsed_data.get('position')}")

        # Legacy fallback: Process additional panels from Panel_*_index A_/B_ definitions
        for indicator_key, indicator_value in panel_indicators.items():
            if isinstance(indicator_value, str) and (indicator_value.startswith('A_') or indicator_value.startswith('B_')):
                # Extract position and ticker from A_TICKER or B_TICKER
                prefix = indicator_value[:2]  # 'A_' or 'B_'
                ticker = indicator_value[2:]  # Everything after 'A_' or 'B_'

                # Determine base panel name from indicator_key (e.g., Panel_1_index -> Panel_1)
                base_panel_name = indicator_key.replace('_index', '')

                # Create unique config key for positioned panel
                position_name = 'above' if prefix == 'A_' else 'below'
                config_key = f"{base_panel_name}_{position_name}_{ticker}_row{row_number}"

                panel_config[config_key] = {
                    'panel_name': f"{base_panel_name}_{position_name}",
                    'data_source': ticker,
                    'indicator': '',  # A_/B_ panels are price charts, no indicators
                    'timeframe': timeframe,
                    'panel_type': base_panel_name,
                    'has_indicator': False,
                    'config_row': row_number,
                    'position': position_name,  # 'above' or 'below'
                    'base_panel': base_panel_name,  # Reference to main panel
                    'prefix': prefix,  # 'A_' or 'B_'
                    'file_name_id': file_name_id,
                    'chart_type': chart_type
                }

        # Apply priority system and vertical stacking for this row
        if panel_config:
            panel_config = _apply_panel_priority_and_stacking(panel_config)

        logger.info(f"🚀 ROW {row_number} FINAL PANEL CONFIG:")
        for panel_key, panel_info in panel_config.items():
            logger.info(f"   ✅ {panel_key}")
            logger.info(f"      data_source: {panel_info.get('data_source')}")
            logger.info(f"      indicator: {panel_info.get('indicator')}")
            logger.info(f"      position: {panel_info.get('position')}")

        return panel_config

    except Exception as e:
        logger.error(f"Error parsing single row {row_number}: {e}")
        return {}


def _parse_simple_format(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Parse simple format where Panel_1, Panel_2 etc. are column names.

    Format:
    timeframe,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
    daily,QQQ,SPY,,,,,,,,,,
    """
    try:
        panel_config = {}

        # Process each data row
        for _, row in df.iterrows():
            # Skip commented rows
            first_val = str(row.iloc[0]).strip()
            if first_val.startswith('#'):
                continue

            # Get timeframe (usually first column)
            timeframe = first_val if first_val not in ['', 'nan'] else 'daily'

            # Extract panel data sources and indicators
            panel_data_sources = {}
            panel_indicators = {}

            for col_name in df.columns:
                col_name_str = str(col_name).strip()

                # Panel data sources (Panel_1, Panel_2, etc.)
                if col_name_str.startswith('Panel_') and not col_name_str.endswith('_index'):
                    if pd.notna(row[col_name]) and str(row[col_name]).strip():
                        data_source = str(row[col_name]).strip()
                        if data_source not in ['', 'nan']:
                            panel_data_sources[col_name_str] = data_source

                # Panel indicators (Panel_1_index, Panel_2_index, etc.)
                elif col_name_str.endswith('_index'):
                    if pd.notna(row[col_name]) and str(row[col_name]).strip():
                        indicator_string = str(row[col_name]).strip()
                        if indicator_string not in ['', 'nan']:
                            # Check if it's A_TICKER or B_TICKER format (panel definition)
                            if (indicator_string.startswith('A_') or indicator_string.startswith('B_')) and len(indicator_string) > 2:
                                panel_indicators[col_name_str] = indicator_string
                            # Check if it's a technical indicator
                            elif validate_indicator_string(indicator_string):
                                panel_indicators[col_name_str] = indicator_string

            # Create panel configuration entries
            for panel_name, data_source in panel_data_sources.items():
                indicator_key = f"{panel_name}_index"
                indicator_value = panel_indicators.get(indicator_key, '')

                # For main panels, only use technical indicators, not A_/B_ definitions
                if indicator_value.startswith('A_') or indicator_value.startswith('B_'):
                    indicator = ''  # A_/B_ are position definitions, not indicators for main panel
                else:
                    indicator = indicator_value

                config_key = f"{panel_name}_{data_source}"
                if indicator:
                    config_key += f"_{indicator.split('(')[0]}"

                panel_config[config_key] = {
                    'panel_name': panel_name,
                    'data_source': data_source,
                    'indicator': indicator,
                    'timeframe': timeframe,
                    'panel_type': panel_name,
                    'has_indicator': bool(indicator),
                    'config_row': len(panel_config) + 1,
                    'position': 'main'  # Main panel position
                }

            # Create additional panels from Panel_*_index A_/B_ definitions
            for indicator_key, indicator_value in panel_indicators.items():
                if indicator_value.startswith('A_') or indicator_value.startswith('B_'):
                    # Extract position and ticker from A_TICKER or B_TICKER
                    prefix = indicator_value[:2]  # 'A_' or 'B_'
                    ticker = indicator_value[2:]  # Everything after 'A_' or 'B_'

                    # Determine base panel name from indicator_key (e.g., Panel_1_index -> Panel_1)
                    base_panel_name = indicator_key.replace('_index', '')

                    # Create unique config key for positioned panel
                    position_name = 'above' if prefix == 'A_' else 'below'
                    config_key = f"{base_panel_name}_{position_name}_{ticker}"

                    panel_config[config_key] = {
                        'panel_name': f"{base_panel_name}_{position_name}",
                        'data_source': ticker,
                        'indicator': '',  # A_/B_ panels are price charts, no indicators
                        'timeframe': timeframe,
                        'panel_type': base_panel_name,
                        'has_indicator': False,
                        'config_row': len(panel_config) + 1,
                        'position': position_name,  # 'above' or 'below'
                        'base_panel': base_panel_name,  # Reference to main panel
                        'prefix': prefix  # 'A_' or 'B_'
                    }

        # Apply priority system and vertical stacking
        panel_config = _apply_panel_priority_and_stacking(panel_config)

        logger.info(f"Parsed {len(panel_config)} panel configurations (simple format)")
        return panel_config

    except Exception as e:
        logger.error(f"Error parsing simple format: {e}")
        return {}


def _parse_complex_format(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Parse complex format where Panel_1 appears as a data value in row 3.

    Format:
    ,,,,,,,,,,,,,
    ,,,,,,,,,,,,,
    Panel_1,timeframe,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
    PPO,daily,QQQ,CPCE,XLY:XLP,CORR,SPY,,"PPO(12,6, 9)",,,,,,"RSI(12, 6, 9)",
    """
    try:
        # Find header row (contains Panel_1, Panel_2, etc.)
        header_row = None
        for idx, row in df.iterrows():
            row_values = [str(val) for val in row.values if pd.notna(val)]
            if any('Panel_1' in str(val) for val in row_values):
                header_row = idx
                break

        if header_row is None:
            logger.error("Could not find header row with Panel_1, Panel_2, etc. in complex format")
            return {}

        # Extract column names from header row and handle duplicates
        headers = df.iloc[header_row].values
        unique_headers = []
        for i, header in enumerate(headers):
            if pd.isna(header):
                unique_headers.append(f'Unnamed_{i}')
            else:
                header_str = str(header)
                if header_str in unique_headers:
                    unique_headers.append(f'{header_str}_data')
                else:
                    unique_headers.append(header_str)
        df.columns = unique_headers

        # Process data rows (after header)
        data_rows = df.iloc[header_row + 1:].dropna(how='all')

        panel_config = {}

        # Process each data row
        for _, row in data_rows.iterrows():
            try:
                # Get timeframe (usually in second column)
                timeframe = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else 'daily'

                # Extract panel data sources (Panel_1 to Panel_6)
                panel_data_sources = {}
                panel_indicators = {}

                for col_name in df.columns:
                    if pd.isna(col_name):
                        continue

                    col_name_str = str(col_name).strip()

                    # Panel data sources (Panel_1_data, Panel_2, etc.)
                    if (col_name_str.startswith('Panel_') and
                        not col_name_str.endswith('_index') and
                        col_name_str != 'Panel_1'):  # Skip the first Panel_1 which is the row type indicator
                        if pd.notna(row[col_name]) and str(row[col_name]).strip():
                            # Clean up the column name for mapping
                            clean_panel_name = col_name_str.replace('_data', '')
                            panel_data_sources[clean_panel_name] = str(row[col_name]).strip()

                    # Panel indicators (Panel_1_index, Panel_2_index, etc.)
                    elif col_name_str.endswith('_index'):
                        if pd.notna(row[col_name]) and str(row[col_name]).strip():
                            indicator_string = str(row[col_name]).strip()
                            if validate_indicator_string(indicator_string):
                                panel_indicators[col_name_str] = indicator_string
                            else:
                                logger.warning(f"Invalid indicator string: {indicator_string}")

                # Create panel configuration entries
                for panel_name, data_source in panel_data_sources.items():
                    indicator_key = f"{panel_name}_index"
                    indicator = panel_indicators.get(indicator_key, '')

                    config_key = f"{panel_name}_{data_source}"
                    if indicator:
                        config_key += f"_{indicator.split('(')[0]}"  # Add indicator type

                    panel_config[config_key] = {
                        'panel_name': panel_name,
                        'data_source': data_source,
                        'indicator': indicator,
                        'timeframe': timeframe,
                        'panel_type': panel_name,
                        'has_indicator': bool(indicator),
                        'config_row': len(panel_config) + 1
                    }

            except Exception as e:
                logger.warning(f"Error processing row in panel config: {e}")
                continue

        logger.info(f"Parsed {len(panel_config)} panel configurations (complex format)")
        return panel_config

    except Exception as e:
        logger.error(f"Error parsing complex format: {e}")
        return {}


def _apply_panel_priority_and_stacking(panel_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Apply priority system and vertical stacking to panel configuration.

    Rules:
    1. Panel_*_index panels override and position relative to main panels
    2. Vertical stacking order: A_ (above), main, B_ (below)
    3. Panel numbers determine overall order (Panel_1, Panel_2, Panel_3...)

    Args:
        panel_config: Original panel configuration

    Returns:
        Reordered panel configuration with stacking_order field
    """
    try:
        # Separate panels by type and organize by panel number
        main_panels = {}      # Panel_1 -> main panel info
        above_panels = {}     # Panel_1 -> [above panels for Panel_1]
        below_panels = {}     # Panel_1 -> [below panels for Panel_1]

        # Group panels by their base panel and position
        for key, panel_info in panel_config.items():
            position = panel_info.get('position', 'main')
            panel_type = panel_info.get('panel_type', panel_info.get('panel_name', ''))

            if position == 'main':
                main_panels[panel_type] = (key, panel_info)
            elif position == 'above':
                base_panel = panel_info.get('base_panel', panel_type)
                if base_panel not in above_panels:
                    above_panels[base_panel] = []
                above_panels[base_panel].append((key, panel_info))
            elif position == 'below':
                base_panel = panel_info.get('base_panel', panel_type)
                if base_panel not in below_panels:
                    below_panels[base_panel] = []
                below_panels[base_panel].append((key, panel_info))

        # Sort panel types by number (Panel_1, Panel_2, Panel_3...)
        def get_panel_number(panel_type):
            try:
                return int(panel_type.split('_')[1]) if '_' in panel_type else 999
            except (IndexError, ValueError):
                return 999

        # Collect all panel types from main, above, and below panels
        # This allows positioned panels to exist without requiring base main panels
        all_panel_types = set(main_panels.keys())
        all_panel_types.update(above_panels.keys())
        all_panel_types.update(below_panels.keys())
        sorted_panel_types = sorted(all_panel_types, key=get_panel_number)

        # Build final ordered configuration with stacking_order
        final_config = {}
        stacking_order = 1

        for panel_type in sorted_panel_types:
            # Add above panels first
            if panel_type in above_panels:
                for key, panel_info in above_panels[panel_type]:
                    panel_info = panel_info.copy()
                    panel_info['stacking_order'] = stacking_order
                    panel_info['stacking_group'] = panel_type
                    final_config[key] = panel_info
                    stacking_order += 1

            # Add main panel
            if panel_type in main_panels:
                key, panel_info = main_panels[panel_type]
                panel_info = panel_info.copy()
                panel_info['stacking_order'] = stacking_order
                panel_info['stacking_group'] = panel_type
                final_config[key] = panel_info
                stacking_order += 1

            # Add below panels last
            if panel_type in below_panels:
                for key, panel_info in below_panels[panel_type]:
                    panel_info = panel_info.copy()
                    panel_info['stacking_order'] = stacking_order
                    panel_info['stacking_group'] = panel_type
                    final_config[key] = panel_info
                    stacking_order += 1

        logger.debug(f"Applied stacking order to {len(final_config)} panels")
        return final_config

    except Exception as e:
        logger.error(f"Error applying panel priority and stacking: {e}")
        return panel_config


def _process_enhanced_panel_entry(panel_config: Dict[str, Dict[str, Any]],
                                 panel_name: str,
                                 data_source: str,
                                 parsed_data: Dict[str, Any],
                                 timeframe: str,
                                 row_number: int,
                                 file_name_id: str = '',
                                 chart_type: str = 'candle') -> None:
    """
    Process enhanced format panel entry and add to panel configuration.

    Args:
        panel_config: Panel configuration dict to update
        panel_name: Base panel name (e.g., Panel_1)
        data_source: Data source from main panel
        parsed_data: Parsed data from enhanced parser
        timeframe: Chart timeframe
        row_number: Row number for reference
        file_name_id: File identifier from CSV
    """
    try:
        # Enhanced format with explicit tickers
        tickers = parsed_data.get('tickers', [])
        indicator = parsed_data.get('indicator')
        parameters = parsed_data.get('parameters', {})
        position = parsed_data.get('position', 'main')
        prefix = parsed_data.get('prefix')

        # Create panel entry based on position
        if position == 'main':
            # Main panel with enhanced indicator
            config_key = f"{panel_name}_{indicator or 'main'}_row{row_number}"
            if len(tickers) > 1:
                # Multi-ticker indicator (like RATIO)
                config_key = f"{panel_name}_{indicator}_{'_'.join(tickers)}_row{row_number}"
                # Override data_source for multi-ticker indicators
                data_source = f"{tickers[0]}:{tickers[1]}" if len(tickers) == 2 else ','.join(tickers)

            panel_config[config_key] = {
                'panel_name': panel_name,
                'data_source': data_source if len(tickers) <= 1 else data_source,
                'indicator': indicator or '',
                'indicator_parameters': parameters,
                'timeframe': timeframe,
                'panel_type': panel_name,
                'has_indicator': bool(indicator),
                'config_row': row_number,
                'position': position,
                'format_type': 'enhanced',
                'tickers': tickers,
                'is_multi_ticker': len(tickers) > 1,
                'file_name_id': file_name_id,
                'chart_type': chart_type
            }

        elif position in ['above', 'below']:
            # Positioned panel from enhanced format
            ticker = tickers[0] if tickers else data_source
            config_key = f"{panel_name}_{position}_{ticker}_{indicator or 'price'}_row{row_number}"

            panel_config[config_key] = {
                'panel_name': f"{panel_name}_{position}",
                'data_source': ticker,
                'indicator': indicator or 'PRICE',
                'indicator_parameters': parameters,
                'timeframe': timeframe,
                'panel_type': panel_name,
                'has_indicator': bool(indicator) or True,  # PRICE is considered an indicator
                'config_row': row_number,
                'position': position,
                'base_panel': panel_name,
                'prefix': prefix,
                'format_type': 'enhanced',
                'tickers': tickers,
                'file_name_id': file_name_id,
                'chart_type': chart_type
            }

        logger.debug(f"Processed enhanced panel entry: {config_key}")

    except Exception as e:
        logger.error(f"Error processing enhanced panel entry: {e}")


def _process_legacy_main_panel(panel_config: Dict[str, Dict[str, Any]],
                              panel_name: str,
                              data_source: str,
                              parsed_data: Dict[str, Any],
                              timeframe: str,
                              row_number: int,
                              file_name_id: str = '',
                              chart_type: str = 'candle') -> None:
    """
    Process legacy format main panel entry.

    Args:
        panel_config: Panel configuration dict to update
        panel_name: Base panel name
        data_source: Data source for panel
        parsed_data: Parsed data from enhanced parser
        timeframe: Chart timeframe
        row_number: Row number for reference
        file_name_id: File identifier from CSV
    """
    try:
        indicator = parsed_data.get('indicator')
        parameters = parsed_data.get('parameters', {})

        config_key = f"{panel_name}_{data_source}"
        if indicator:
            config_key += f"_{indicator}"
        config_key += f"_row{row_number}"

        panel_config[config_key] = {
            'panel_name': panel_name,
            'data_source': data_source,
            'indicator': indicator or '',
            'indicator_parameters': parameters,
            'timeframe': timeframe,
            'panel_type': panel_name,
            'has_indicator': bool(indicator),
            'config_row': row_number,
            'position': 'main',
            'format_type': parsed_data.get('format_type', 'legacy'),
            'file_name_id': file_name_id,
            'chart_type': chart_type
        }

        logger.debug(f"Processed legacy main panel: {config_key}")

    except Exception as e:
        logger.error(f"Error processing legacy main panel: {e}")


def _process_simple_main_panel(panel_config: Dict[str, Dict[str, Any]],
                              panel_name: str,
                              data_source: str,
                              timeframe: str,
                              row_number: int,
                              file_name_id: str = '',
                              chart_type: str = 'candle') -> None:
    """
    Process simple main panel without indicators.

    Args:
        panel_config: Panel configuration dict to update
        panel_name: Base panel name
        data_source: Data source for panel
        timeframe: Chart timeframe
        row_number: Row number for reference
    """
    try:
        config_key = f"{panel_name}_{data_source}_row{row_number}"

        panel_config[config_key] = {
            'panel_name': panel_name,
            'data_source': data_source,
            'indicator': '',
            'indicator_parameters': {},
            'timeframe': timeframe,
            'panel_type': panel_name,
            'has_indicator': False,
            'config_row': row_number,
            'position': 'main',
            'format_type': 'simple',
            'file_name_id': file_name_id,
            'chart_type': chart_type
        }

        logger.debug(f"Processed simple main panel: {config_key}")

    except Exception as e:
        logger.error(f"Error processing simple main panel: {e}")


def _process_positioned_panel(panel_config: Dict[str, Dict[str, Any]],
                             base_panel_name: str,
                             parsed_data: Dict[str, Any],
                             timeframe: str,
                             row_number: int,
                             file_name_id: str = '',
                             chart_type: str = 'candle') -> None:
    """
    Process positioned panel (above/below) from enhanced parsing.

    Args:
        panel_config: Panel configuration dict to update
        base_panel_name: Base panel name
        parsed_data: Parsed data with position information
        timeframe: Chart timeframe
        row_number: Row number for reference
        file_name_id: File identifier from CSV
    """
    try:
        position = parsed_data.get('position', 'main')
        tickers = parsed_data.get('tickers', [])
        indicator = parsed_data.get('indicator')
        parameters = parsed_data.get('parameters', {})
        prefix = parsed_data.get('prefix')

        logger.info(f"🚀 _process_positioned_panel CALLED:")
        logger.info(f"   base_panel_name: {base_panel_name}")
        logger.info(f"   position: {position}")
        logger.info(f"   tickers: {tickers}")
        logger.info(f"   indicator: {indicator}")
        logger.info(f"   parameters: {parameters}")

        if position not in ['above', 'below']:
            logger.info(f"❌ INVALID POSITION: {position} - Returning early")
            return

        ticker = tickers[0] if tickers else 'UNKNOWN'
        config_key = f"{base_panel_name}_{position}_{ticker}_{indicator or 'price'}_row{row_number}"

        logger.info(f"✅ CREATING CONFIG KEY: {config_key}")
        logger.info(f"   ticker resolved: {ticker}")
        logger.info(f"   indicator for config: {indicator or 'PRICE'}")

        panel_config[config_key] = {
            'panel_name': f"{base_panel_name}_{position}",
            'data_source': ticker,
            'indicator': indicator or 'PRICE',
            'indicator_parameters': parameters,
            'timeframe': timeframe,
            'panel_type': base_panel_name,
            'has_indicator': bool(indicator) or True,  # PRICE is considered an indicator
            'config_row': row_number,
            'position': position,
            'base_panel': base_panel_name,
            'prefix': prefix,
            'format_type': parsed_data.get('format_type', 'enhanced'),
            'tickers': tickers,
            'file_name_id': file_name_id,
            'chart_type': chart_type
        }

        logger.info(f"✅ POSITIONED PANEL CREATED: {config_key}")
        logger.info(f"   panel_name: {panel_config[config_key]['panel_name']}")
        logger.info(f"   data_source: {panel_config[config_key]['data_source']}")
        logger.info(f"   indicator: {panel_config[config_key]['indicator']}")
        logger.info(f"   has_indicator: {panel_config[config_key]['has_indicator']}")

    except Exception as e:
        logger.error(f"Error processing positioned panel: {e}")


def _process_enhanced_data_source_panel(panel_config: Dict[str, Dict[str, Any]],
                                        panel_name: str,
                                        parsed_data: Dict[str, Any],
                                        timeframe: str,
                                        row_number: int,
                                        file_name_id: str = '',
                                        chart_type: str = 'candle') -> None:
    """
    Process enhanced format as data source (e.g., PPO(10,20,40)_for_(QQQ) in Panel_1).

    Args:
        panel_config: Panel configuration dict to update
        panel_name: Base panel name
        parsed_data: Parsed enhanced format data
        timeframe: Chart timeframe
        row_number: Row number for reference
        file_name_id: File identifier from CSV
    """
    try:
        indicator = parsed_data.get('indicator')
        parameters = parsed_data.get('parameters', {})
        tickers = parsed_data.get('tickers', [])
        position = parsed_data.get('position', 'main')
        prefix = parsed_data.get('prefix')
        original_entry = parsed_data.get('original_entry', 'unknown')

        # Create data source from the enhanced format
        if len(tickers) > 1:
            # Multi-ticker: create composite data source identifier
            data_source = f"{tickers[0]}:{tickers[1]}" if len(tickers) == 2 else ','.join(tickers)
        else:
            # Single ticker
            data_source = tickers[0] if tickers else 'AUTO_DETECT'

        # Create panel configuration
        if position in ['above', 'below']:
            # Positioned panel
            config_key = f"{panel_name}_{position}_{indicator}_{data_source}_row{row_number}"
            panel_config[config_key] = {
                'panel_name': f"{panel_name}_{position}",
                'data_source': original_entry,  # Use original enhanced format as data source
                'indicator': indicator,
                'indicator_parameters': parameters,
                'timeframe': timeframe,
                'panel_type': panel_name,
                'has_indicator': True,
                'config_row': row_number,
                'position': position,
                'base_panel': panel_name,
                'prefix': prefix,
                'format_type': 'enhanced',
                'tickers': tickers,
                'is_multi_ticker': len(tickers) > 1,
                'is_data_source_indicator': True,
                'file_name_id': file_name_id,
                'chart_type': chart_type
            }
        else:
            # Main panel with enhanced data source
            config_key = f"{panel_name}_{indicator}_{'_'.join(tickers)}_row{row_number}"
            panel_config[config_key] = {
                'panel_name': panel_name,
                'data_source': original_entry,  # Use original enhanced format as data source
                'indicator': indicator,
                'indicator_parameters': parameters,
                'timeframe': timeframe,
                'panel_type': panel_name,
                'has_indicator': True,
                'config_row': row_number,
                'position': 'main',
                'format_type': 'enhanced',
                'tickers': tickers,
                'is_multi_ticker': len(tickers) > 1,
                'is_data_source_indicator': True,
                'file_name_id': file_name_id,
                'chart_type': chart_type
            }

        logger.debug(f"Processed enhanced data source panel: {config_key}")

    except Exception as e:
        logger.error(f"Error processing enhanced data source panel: {e}")


def _process_bundled_panel_entry(panel_config: Dict[str, Dict[str, Any]],
                                 panel_name: str,
                                 parsed_data: Dict[str, Any],
                                 timeframe: str,
                                 row_number: int,
                                 file_name_id: str = '',
                                 chart_type: str = 'candle') -> None:
    """
    Process bundled panel entry (e.g., QQQ+EMA(20)+EMA(50) or A_QQQ+EMA(20)).

    Args:
        panel_config: Panel configuration dict to update
        panel_name: Base panel name
        parsed_data: Parsed bundled format data
        timeframe: Chart timeframe
        row_number: Row number for reference
        file_name_id: File identifier from CSV
    """
    try:
        base_data = parsed_data.get('base_data', 'unknown')
        indicators = parsed_data.get('indicators', [])
        tickers = parsed_data.get('tickers', [])
        position = parsed_data.get('position', 'main')
        prefix = parsed_data.get('prefix')
        original_entry = parsed_data.get('original_entry', 'unknown')

        # Create config key
        if position in ['above', 'below']:
            # Positioned bundled panel
            base_identifier = base_data.replace('_for_(', '_').replace(')', '_').replace(',', '_')
            config_key = f"{panel_name}_{position}_bundled_{base_identifier}_row{row_number}"
            panel_name_final = f"{panel_name}_{position}"
        else:
            # Main bundled panel
            base_identifier = base_data.replace('_for_(', '_').replace(')', '_').replace(',', '_')
            config_key = f"{panel_name}_bundled_{base_identifier}_row{row_number}"
            panel_name_final = panel_name

        # Create panel configuration
        panel_config[config_key] = {
            'panel_name': panel_name_final,
            'data_source': original_entry,  # Use original bundled string as data source
            'base_data': base_data,
            'indicators': indicators,
            'bundled_components': parsed_data.get('bundled_components', []),
            'timeframe': timeframe,
            'panel_type': panel_name,
            'has_indicator': len(indicators) > 0,
            'config_row': row_number,
            'position': position,
            'base_panel': panel_name if position in ['above', 'below'] else None,
            'prefix': prefix,
            'format_type': 'bundled',
            'tickers': tickers,
            'is_multi_ticker': len(tickers) > 1,
            'file_name_id': file_name_id,
            'is_bundled': True,
            'chart_type': chart_type,
            'overlay_indicators': [ind for ind in indicators if ind.get('display_type') == 'overlay'],
            'subplot_indicators': [ind for ind in indicators if ind.get('display_type') == 'subplot']
        }

        logger.debug(f"Processed bundled panel entry: {config_key}")
        logger.debug(f"  Base data: {base_data}")
        logger.debug(f"  Indicators: {[ind['indicator'] for ind in indicators]}")
        logger.debug(f"  Overlay indicators: {len(panel_config[config_key]['overlay_indicators'])}")
        logger.debug(f"  Subplot indicators: {len(panel_config[config_key]['subplot_indicators'])}")

    except Exception as e:
        logger.error(f"Error processing bundled panel entry: {e}")


def convert_legacy_csv_to_enhanced(csv_file_path: str, output_file_path: str = None) -> str:
    """
    Convert legacy CSV format to enhanced format with backward compatibility.

    Args:
        csv_file_path: Path to existing CSV file
        output_file_path: Optional output path (default: adds _enhanced suffix)

    Returns:
        Path to converted file

    Example conversion:
        Legacy: A_QQQ in Panel_1_index
        Enhanced: A_PRICE_for_(QQQ) in Panel_1_index

        Legacy: PPO(12,26,9) in Panel_1_index
        Enhanced: PPO(12,26,9)_for_(AUTO_DETECT) in Panel_1_index
    """
    try:
        if not output_file_path:
            base_name = csv_file_path.replace('.csv', '')
            output_file_path = f"{base_name}_enhanced.csv"

        # Read original CSV
        df = pd.read_csv(csv_file_path)

        # Create enhanced version
        enhanced_df = df.copy()

        # Process each cell that might contain legacy format
        for col in enhanced_df.columns:
            if '_index' in str(col):
                # This is an indicator column - convert legacy formats
                enhanced_df[col] = enhanced_df[col].apply(_convert_cell_to_enhanced)

        # Save enhanced CSV
        enhanced_df.to_csv(output_file_path, index=False)

        logger.info(f"Converted legacy CSV to enhanced format: {output_file_path}")
        return output_file_path

    except Exception as e:
        logger.error(f"Error converting legacy CSV: {e}")
        return csv_file_path


def _convert_cell_to_enhanced(cell_value) -> str:
    """
    Convert individual cell value from legacy to enhanced format.

    Args:
        cell_value: Original cell value

    Returns:
        Enhanced format string
    """
    try:
        if pd.isna(cell_value) or str(cell_value).strip() in ['', 'nan']:
            return cell_value

        cell_str = str(cell_value).strip()

        # Use enhanced parser's conversion function
        from .enhanced_panel_parser import convert_legacy_to_enhanced
        converted = convert_legacy_to_enhanced(cell_str)

        if converted != cell_str:
            logger.debug(f"Converted '{cell_str}' -> '{converted}'")

        return converted

    except Exception as e:
        logger.warning(f"Error converting cell value '{cell_value}': {e}")
        return cell_value


def detect_csv_format_type(csv_file_path: str) -> Dict[str, Any]:
    """
    Detect whether CSV uses legacy or enhanced format.

    Args:
        csv_file_path: Path to CSV file

    Returns:
        Dict with format detection results:
        {
            'format_type': 'legacy' | 'enhanced' | 'mixed',
            'legacy_patterns': List[str],
            'enhanced_patterns': List[str],
            'recommendations': List[str]
        }
    """
    detection_result = {
        'format_type': 'unknown',
        'legacy_patterns': [],
        'enhanced_patterns': [],
        'recommendations': []
    }

    try:
        # Read CSV
        df = pd.read_csv(csv_file_path)

        legacy_count = 0
        enhanced_count = 0

        # Check each indicator column
        for col in df.columns:
            if '_index' in str(col):
                for value in df[col].dropna():
                    value_str = str(value).strip()
                    if not value_str or value_str in ['', 'nan']:
                        continue

                    # Check for legacy patterns
                    if _is_legacy_pattern(value_str):
                        legacy_count += 1
                        detection_result['legacy_patterns'].append(value_str)

                    # Check for enhanced patterns
                    elif '_for_(' in value_str:
                        enhanced_count += 1
                        detection_result['enhanced_patterns'].append(value_str)

        # Determine format type
        total_patterns = legacy_count + enhanced_count

        if total_patterns == 0:
            detection_result['format_type'] = 'simple'
            detection_result['recommendations'].append("No advanced indicators detected")

        elif legacy_count > 0 and enhanced_count == 0:
            detection_result['format_type'] = 'legacy'
            detection_result['recommendations'].append("Consider converting to enhanced format for better functionality")

        elif enhanced_count > 0 and legacy_count == 0:
            detection_result['format_type'] = 'enhanced'
            detection_result['recommendations'].append("Already using enhanced format")

        else:
            detection_result['format_type'] = 'mixed'
            detection_result['recommendations'].append("Mixed format detected - consider standardizing")

        logger.info(f"Format detection: {detection_result['format_type']} "
                   f"(legacy: {legacy_count}, enhanced: {enhanced_count})")

        return detection_result

    except Exception as e:
        logger.error(f"Error detecting CSV format: {e}")
        detection_result['format_type'] = 'error'
        return detection_result


def _is_legacy_pattern(value_str: str) -> bool:
    """Check if string matches legacy pattern."""
    # A_/B_ prefix patterns
    if value_str.startswith(('A_', 'B_')):
        return True

    # Ratio patterns (XLY:XLP)
    if ':' in value_str and not '(' in value_str:
        return True

    # Legacy indicator patterns (PPO(12,26,9) without _for_)
    if '(' in value_str and ')' in value_str and '_for_(' not in value_str:
        from ..indicators.indicator_parser import validate_indicator_string
        return validate_indicator_string(value_str)

    return False


def create_migration_guide() -> str:
    """
    Create comprehensive migration guide for enhanced format.

    Returns:
        String with migration guide content
    """
    guide = """
# SR Module Enhanced Format Migration Guide

## Overview
The SR module now supports an enhanced panel configuration format that provides better flexibility and multi-ticker support while maintaining full backward compatibility.

## Format Comparison

### Legacy Format Examples:
```
Panel_1_index: A_QQQ              -> Positions QQQ above main panel
Panel_1_index: B_SPY              -> Positions SPY below main panel
Panel_1_index: PPO(12,26,9)       -> PPO indicator on main panel ticker
Panel_1_index: XLY:XLP            -> Ratio between XLY and XLP
```

### Enhanced Format Examples:
```
Panel_1_index: A_PRICE_for_(QQQ)           -> Same as A_QQQ
Panel_1_index: B_PRICE_for_(SPY)           -> Same as B_SPY
Panel_1_index: PPO(12,26,9)_for_(QQQ)      -> PPO indicator for QQQ
Panel_1_index: RATIO_for_(XLY,XLP)         -> Same as XLY:XLP
Panel_1_index: A_PPO(12,26,9)_for_(SPY)    -> PPO above main panel for SPY
```

## New Capabilities

### Multi-Ticker Indicators:
```
Panel_1_index: RATIO_for_(SPY,QQQ,IWM)     -> 3-way ratio comparison
Panel_1_index: CORR_for_(SPY,QQQ)          -> Correlation indicator (future)
```

### Explicit Ticker Assignment:
```
Panel_1: QQQ                               -> Main panel shows QQQ
Panel_1_index: PPO(12,26,9)_for_(SPY)      -> PPO calculated for SPY, not QQQ
```

## Migration Process

### Automatic Conversion:
The system automatically detects and parses legacy formats, so existing CSV files continue to work without changes.

### Manual Conversion (Recommended):
1. Use `convert_legacy_csv_to_enhanced()` function
2. Review converted file for accuracy
3. Take advantage of new enhanced features

### Format Detection:
Use `detect_csv_format_type()` to analyze your current CSV files.

## Best Practices

### Enhanced Format Advantages:
- Explicit ticker specification eliminates ambiguity
- Multi-ticker support for advanced indicators
- Better validation and error reporting
- Future-proof for new indicator types

### When to Migrate:
- Using complex multi-panel configurations
- Need multi-ticker indicators
- Want explicit control over indicator assignments
- Preparing for advanced features

## Supported Indicators

### Current Indicators:
- PPO: Percentage Price Oscillator
- RSI: Relative Strength Index
- EMA: Exponential Moving Average
- SMA: Simple Moving Average
- RATIO: Price ratio between tickers
- PRICE: Simple price chart

### Enhanced Format Benefits by Indicator:
- RATIO: Multi-ticker support (RATIO_for_(A,B,C))
- PPO/RSI: Explicit ticker assignment
- PRICE: Consistent with other indicators

## Example Configurations

### Simple Configuration:
```csv
timeframe,Panel_1,Panel_2,Panel_1_index,Panel_2_index
daily,QQQ,SPY,PPO(12,26,9)_for_(QQQ),RSI(14)_for_(SPY)
```

### Advanced Configuration:
```csv
timeframe,Panel_1,Panel_2,Panel_3,Panel_1_index,Panel_2_index,Panel_3_index
daily,QQQ,SPY,XLY,A_PRICE_for_(SPY),RATIO_for_(XLY,XLP),B_PPO(12,26,9)_for_(QQQ)
```

## Migration Functions

### Available Functions:
- `convert_legacy_csv_to_enhanced(csv_path)`: Convert entire CSV file
- `detect_csv_format_type(csv_path)`: Analyze format usage
- `validate_panel_config(config)`: Enhanced validation

### Usage Example:
```python
from src.sustainability_ratios.sr_config_reader import convert_legacy_csv_to_enhanced

# Convert existing file
enhanced_file = convert_legacy_csv_to_enhanced('user_data_panel.csv')
print(f"Enhanced file created: {enhanced_file}")
```
""".strip()

    return guide


def create_sample_panel_config() -> str:
    """
    Create a sample user_data_panel.csv content with enhanced format examples.

    Returns:
        String with sample CSV content
    """
    sample_content = """#timeframe,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
daily,QQQ,SPY,,,,,,,,,,
daily,QQQ,SPY,SPY,,,,,,,,,
daily,QQQ,SPY,,,,,A_PRICE_for_(QQQ),,,,,
daily,SPY,,,,,,B_PRICE_for_(QQQ),,,,,
daily,QQQ,,,,,,"PPO(12,26,9)_for_(QQQ)",,,,,
daily,XLY,XLP,,,,,,"RATIO_for_(XLY,XLP)",,,,
daily,QQQ,SPY,,,,,A_PPO(12,26,9)_for_(SPY),"RSI(14)_for_(SPY)",,,,

# Enhanced Panel Configuration Guide:
#
# Panel_1-6: Data sources (tickers like QQQ, SPY)
# Panel_1_index-6_index: Enhanced indicators in format:
#   - [Prefix]_IndicatorName(params)_for_(Ticker1[,Ticker2,...])
#   - Examples:
#     * PPO(12,26,9)_for_(QQQ)      -> PPO indicator for QQQ
#     * A_PRICE_for_(SPY)           -> SPY price chart above main panel
#     * B_RSI(14)_for_(QQQ)         -> RSI indicator below main panel
#     * RATIO_for_(XLY,XLP)         -> Ratio between XLY and XLP
#
# Supported Indicators: PPO, RSI, EMA, SMA, RATIO, PRICE
# Supported Positions: A_ (above), B_ (below), or main panel
# Legacy format still supported for backward compatibility
# Empty cells are ignored

# Legacy Format Examples (still supported):
# daily,QQQ,SPY,,,,,A_QQQ,PPO(12,26,9),XLY:XLP,,,
""".strip()

    return sample_content