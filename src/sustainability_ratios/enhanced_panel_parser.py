"""
Enhanced Panel Parser Module
============================

Parses enhanced panel configuration format: [Prefix]_IndicatorName(Params)_for_(Ticker1[,Ticker2,...])

Supports:
- Simple tickers: QQQ
- Legacy A_/B_: A_QQQ, B_QQQ
- Legacy indicators: PPO(12,26,9)
- Enhanced format: A_PPO(12,26,9)_for_(QQQ)
- Multi-ticker: B_RATIO_for_(SPY,QQQ)

Functions:
- parse_enhanced_panel_entry: Main entry point for parsing
- convert_legacy_to_enhanced: Backward compatibility conversion
- validate_enhanced_entry: Validation for new format
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union

from ..indicators.indicator_parser import validate_indicator_string, extract_indicator_name, parse_indicator_string

logger = logging.getLogger(__name__)


# Legacy format mappings for backward compatibility
LEGACY_MAPPINGS = {
    'XLY:XLP': 'RATIO_for_(XLY,XLP)',
    'SPY:QQQ': 'RATIO_for_(SPY,QQQ)',
    'QQQ:SPY': 'RATIO_for_(QQQ,SPY)',
}


def parse_enhanced_panel_entry(entry: str, main_panel_ticker: str = None) -> Optional[Dict[str, Any]]:
    """
    Parse enhanced panel entry supporting multiple formats including bundling.

    Supports:
    - Enhanced: A_PPO(12,26,9)_for_(QQQ)
    - Bundled: QQQ+EMA(20)+EMA(50)
    - Bundled with positioning: A_QQQ+EMA(20)+EMA(50)
    - Legacy A_/B_: A_QQQ, B_QQQ
    - Legacy indicators: PPO(12,26,9)
    - Legacy ratios: XLY:XLP
    - Simple tickers: QQQ

    Args:
        entry: Panel entry string
        main_panel_ticker: Default ticker for legacy indicators

    Returns:
        Dict with parsed information or None if invalid
    """
    try:
        entry = str(entry).strip()
        if not entry or entry in ['', 'nan', 'NaN']:
            return None

        logger.debug(f"Parsing panel entry: '{entry}'")

        # Pattern 1: Bundled indicators (contains +)
        if '+' in entry:
            return _parse_bundled_format(entry, main_panel_ticker)

        # Pattern 2: Enhanced format with _for_()
        elif '_for_(' in entry:
            return _parse_enhanced_format(entry)

        # Pattern 3: Legacy A_/B_ positioning
        elif entry.startswith(('A_', 'B_')):
            return _parse_legacy_ab_format(entry)

        # Pattern 4: Legacy ratios (XLY:XLP)
        elif ':' in entry and not '(' in entry:
            return _parse_legacy_ratio_format(entry)

        # Pattern 5: Ticker-first indicators (PPO(QQQ,10,20,30))
        elif '(' in entry and _is_ticker_first_indicator(entry):
            return _parse_ticker_first_format(entry)

        # Pattern 6: Legacy technical indicators
        elif '(' in entry and validate_indicator_string(entry):
            return _parse_legacy_indicator_format(entry, main_panel_ticker)

        # Pattern 7: Simple ticker
        else:
            return _parse_simple_ticker_format(entry)

    except Exception as e:
        logger.error(f"Error parsing panel entry '{entry}': {e}")
        return None


def _parse_bundled_format(entry: str, main_panel_ticker: str = None) -> Dict[str, Any]:
    """
    Parse bundled format with + operator: QQQ+EMA(20)+EMA(50) or A_QQQ+EMA(20)

    Examples:
    - QQQ+EMA(20)+EMA(50)
    - A_QQQ+EMA(20)+EMA(50)
    - SPY+RSI(14)+EMA(20)
    - PRICE_for_(QQQ)+EMA(20)_for_(QQQ)+EMA(50)_for_(QQQ)
    """
    try:
        # Check for A_/B_ prefix
        prefix = None
        position = 'main'
        working_entry = entry

        if entry.startswith(('A_', 'B_')):
            prefix = entry[:2]
            position = 'above' if prefix == 'A_' else 'below'
            working_entry = entry[2:]  # Remove A_ or B_ prefix

        # Split by + operator (handle spaces around +)
        components = [comp.strip() for comp in working_entry.split('+') if comp.strip()]

        if not components:
            raise ValueError("No valid components found after splitting by +")

        # Parse each component
        base_data = None
        indicators = []
        all_tickers = set()

        for i, component in enumerate(components):
            # Parse individual component
            if i == 0:
                # First component is usually the base data (ticker or PRICE_for_)
                base_component = _parse_bundled_component(component, main_panel_ticker)
                if base_component['is_ticker']:
                    base_data = component
                    all_tickers.add(component)
                else:
                    # Enhanced format as base
                    base_data = component
                    all_tickers.update(base_component.get('tickers', []))
                    if base_component.get('indicator'):
                        indicators.append({
                            'indicator': base_component['indicator'],
                            'parameters': base_component.get('parameters', {}),
                            'tickers': base_component.get('tickers', []),
                            'display_type': _get_indicator_display_type(base_component['indicator'])
                        })
            else:
                # Subsequent components are indicators
                comp_parsed = _parse_bundled_component(component, main_panel_ticker)
                if comp_parsed.get('indicator'):
                    indicators.append({
                        'indicator': comp_parsed['indicator'],
                        'parameters': comp_parsed.get('parameters', {}),
                        'tickers': comp_parsed.get('tickers', []),
                        'display_type': _get_indicator_display_type(comp_parsed['indicator'])
                    })
                    all_tickers.update(comp_parsed.get('tickers', []))

        # Build result
        result = {
            'format_type': 'bundled',
            'prefix': prefix,
            'position': position,
            'base_data': base_data,
            'indicators': indicators,
            'bundled_components': components,
            'tickers': list(all_tickers),
            'has_indicator': len(indicators) > 0,
            'original_entry': entry,
            'is_multi_ticker': len(all_tickers) > 1,
            'is_bundled': True
        }

        logger.debug(f"Parsed bundled format: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing bundled format '{entry}': {e}")
        raise ValueError(f"Invalid bundled format: {entry}")


def _parse_bundled_component(component: str, main_panel_ticker: str = None) -> Dict[str, Any]:
    """
    Parse individual component within bundled format.

    Args:
        component: Single component like "QQQ", "EMA(20)", "PRICE_for_(QQQ)"
        main_panel_ticker: Default ticker for indicators without explicit ticker

    Returns:
        Dict with component information
    """
    try:
        component = component.strip()

        # Check if it's enhanced format
        if '_for_(' in component:
            parsed = parse_enhanced_panel_entry(component, main_panel_ticker)
            if parsed:
                return {
                    'is_ticker': False,
                    'indicator': parsed.get('indicator'),
                    'parameters': parsed.get('parameters', {}),
                    'tickers': parsed.get('tickers', []),
                    'original': component
                }

        # Check if it's a technical indicator
        elif '(' in component and validate_indicator_string(component):
            indicator_name = extract_indicator_name(component)

            # Extract ticker from indicator parameters if present (e.g., EMA(QQQ,10))
            extracted_tickers = _extract_tickers_from_indicator_params(component, {})

            # Create version without ticker for parameter parsing
            component_for_params = _remove_ticker_from_indicator_params(component)

            # Parse parameters using the ticker-free version
            parameters = parse_indicator_string(component_for_params)

            if not extracted_tickers:
                extracted_tickers = [main_panel_ticker] if main_panel_ticker else ['AUTO_DETECT']

            return {
                'is_ticker': False,
                'indicator': indicator_name,
                'parameters': parameters,
                'tickers': extracted_tickers,
                'original': component
            }

        # Simple ticker
        else:
            return {
                'is_ticker': True,
                'ticker': component,
                'tickers': [component],
                'original': component
            }

    except Exception as e:
        logger.error(f"Error parsing bundled component '{component}': {e}")
        return {'is_ticker': False, 'original': component, 'tickers': []}


def _get_indicator_display_type(indicator_name: str) -> str:
    """
    Determine if indicator should be overlaid or shown in subplot.

    Args:
        indicator_name: Name of the indicator

    Returns:
        'overlay' or 'subplot'
    """
    # Overlay indicators (shown on price chart)
    overlay_indicators = ['EMA', 'SMA', 'MA', 'VWAP', 'BOLLINGER', 'PRICE']

    # Subplot indicators (shown below price chart)
    subplot_indicators = ['RSI', 'PPO', 'MACD', 'STOCH', 'WILLIAMS', 'CCI']

    if indicator_name.upper() in overlay_indicators:
        return 'overlay'
    elif indicator_name.upper() in subplot_indicators:
        return 'subplot'
    else:
        return 'overlay'  # Default to overlay for unknown indicators


def _extract_tickers_from_indicator_params(component: str, parameters: Dict[str, Any]) -> List[str]:
    """
    Extract ticker symbols from indicator parameters.

    Supports formats like:
    - EMA(QQQ,10) → ['QQQ']
    - RSI(SPY,14) → ['SPY']
    - PPO(QQQ,12,26,9) → ['QQQ']
    - EMA(10) → [] (no ticker specified)

    Args:
        component: Original component string
        parameters: Parsed parameters dict

    Returns:
        List of ticker symbols found in parameters
    """
    try:
        tickers = []

        # Extract from the raw component string to get parameter order
        if '(' in component and ')' in component:
            params_str = component.split('(')[1].split(')')[0].strip()
            if params_str:
                param_parts = [p.strip() for p in params_str.split(',')]

                # Check if first parameter looks like a ticker (alphabetic, 1-5 chars)
                if param_parts and len(param_parts) > 0:
                    first_param = param_parts[0].strip()
                    if _is_ticker_like(first_param):
                        tickers.append(first_param)

        return tickers

    except Exception as e:
        logger.error(f"Error extracting tickers from indicator params '{component}': {e}")
        return []


def _is_ticker_like(param: str) -> bool:
    """
    Check if a parameter looks like a ticker symbol.

    Args:
        param: Parameter string to check

    Returns:
        True if parameter looks like ticker symbol
    """
    try:
        # Basic ticker criteria:
        # - 1-5 characters
        # - Alphabetic (with possible . or -)
        # - Not a number
        if not param or len(param) == 0:
            return False

        # Remove common ticker suffixes/prefixes
        clean_param = param.upper().strip()

        # Check if it's purely numeric (not a ticker)
        try:
            float(clean_param)
            return False  # It's a number, not a ticker
        except ValueError:
            pass  # Good, it's not a number

        # Check ticker-like criteria
        if (1 <= len(clean_param) <= 6 and
            clean_param.replace('.', '').replace('-', '').replace('^', '').isalpha()):
            return True

        return False

    except Exception:
        return False


def _remove_ticker_from_indicator_params(component: str) -> str:
    """
    Remove ticker from indicator parameters for legacy parser compatibility.

    Transforms:
    - EMA(QQQ,10) → EMA(10)
    - RSI(SPY,14) → RSI(14)
    - PPO(QQQ,12,26,9) → PPO(12,26,9)
    - EMA(10) → EMA(10) (unchanged if no ticker)

    Args:
        component: Original indicator string

    Returns:
        Indicator string without ticker parameter
    """
    try:
        if '(' not in component or ')' not in component:
            return component

        indicator_name = component.split('(')[0]
        params_str = component.split('(')[1].split(')')[0].strip()

        if not params_str:
            return component

        param_parts = [p.strip() for p in params_str.split(',')]

        # If first parameter looks like a ticker, remove it
        if param_parts and _is_ticker_like(param_parts[0]):
            # Remove the ticker (first parameter)
            numeric_params = param_parts[1:]
            if numeric_params:
                new_params_str = ','.join(numeric_params)
                return f"{indicator_name}({new_params_str})"
            else:
                # No numeric parameters left after removing ticker
                return f"{indicator_name}()"

        # No ticker found, return as-is
        return component

    except Exception as e:
        logger.error(f"Error removing ticker from indicator params '{component}': {e}")
        return component


def _parse_enhanced_format(entry: str) -> Dict[str, Any]:
    """
    Parse enhanced format: [Prefix]_IndicatorName(Params)_for_(Ticker1[,Ticker2,...])

    Examples:
    - A_PPO(12,26,9)_for_(QQQ)
    - B_RATIO_for_(SPY,QQQ)
    - PPO(12,26,9)_for_(QQQ)
    """
    try:
        # Split at _for_( delimiter
        if '_for_(' not in entry:
            raise ValueError("Missing _for_( delimiter in enhanced format")

        main_part, ticker_part = entry.split('_for_(', 1)

        # Extract tickers from parentheses
        if not ticker_part.endswith(')'):
            raise ValueError("Missing closing parenthesis in ticker section")

        ticker_string = ticker_part.rstrip(')')
        tickers = [t.strip() for t in ticker_string.split(',') if t.strip()]

        if not tickers:
            raise ValueError("No tickers specified in _for_() section")

        # Parse prefix
        prefix = None
        position = 'main'

        if main_part.startswith(('A_', 'B_')):
            prefix = main_part[:2]
            position = 'above' if prefix == 'A_' else 'below'
            indicator_part = main_part[2:]
        else:
            indicator_part = main_part

        # Parse indicator and parameters
        indicator_name = None
        parameters = {}

        if indicator_part:
            if '(' in indicator_part:
                # Extract indicator name before parentheses
                indicator_name = indicator_part.split('(')[0]

                # Extract parameters
                params_match = re.search(r'\(([^)]*)\)', indicator_part)
                if params_match:
                    params_str = params_match.group(1)
                    if params_str.strip():
                        parameters = _parse_indicator_parameters(indicator_name, params_str)
            else:
                indicator_name = indicator_part

        result = {
            'format_type': 'enhanced',
            'prefix': prefix,
            'position': position,
            'indicator': indicator_name,
            'parameters': parameters,
            'tickers': tickers,
            'has_indicator': bool(indicator_name),
            'original_entry': entry,
            'is_multi_ticker': len(tickers) > 1
        }

        logger.debug(f"Parsed enhanced format: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing enhanced format '{entry}': {e}")
        raise ValueError(f"Invalid enhanced format: {entry}")


def _parse_legacy_ab_format(entry: str) -> Dict[str, Any]:
    """
    Parse legacy A_/B_ format: A_QQQ, B_QQQ, A_PPO(12,26,9)

    Convert to enhanced equivalent:
    - A_QQQ -> A_PRICE_for_(QQQ)
    - A_PPO(12,26,9) -> A_PPO(12,26,9)_for_(AUTO_DETECT)
    """
    try:
        prefix = entry[:2]  # 'A_' or 'B_'
        remainder = entry[2:]  # Everything after A_ or B_
        position = 'above' if prefix == 'A_' else 'below'

        # Check if it's an indicator (contains parentheses)
        if '(' in remainder and validate_indicator_string(remainder):
            # Legacy indicator with A_/B_ prefix
            indicator_name = extract_indicator_name(remainder)
            parameters = parse_indicator_string(remainder)

            result = {
                'format_type': 'legacy_ab_indicator',
                'prefix': prefix,
                'position': position,
                'indicator': indicator_name,
                'parameters': parameters,
                'tickers': ['AUTO_DETECT'],  # Will be resolved later
                'has_indicator': True,
                'original_entry': entry,
                'is_multi_ticker': False,
                'requires_main_ticker': True
            }
        else:
            # Legacy A_/B_ ticker positioning
            ticker = remainder

            result = {
                'format_type': 'legacy_ab_ticker',
                'prefix': prefix,
                'position': position,
                'indicator': 'PRICE',  # Implicit price chart
                'parameters': {},
                'tickers': [ticker],
                'has_indicator': True,  # PRICE is an indicator
                'original_entry': entry,
                'is_multi_ticker': False
            }

        logger.debug(f"Parsed legacy A_/B_ format: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing legacy A_/B_ format '{entry}': {e}")
        raise ValueError(f"Invalid legacy A_/B_ format: {entry}")


def _parse_legacy_ratio_format(entry: str) -> Dict[str, Any]:
    """
    Parse legacy ratio format: XLY:XLP

    Convert to: RATIO_for_(XLY,XLP)
    """
    try:
        if ':' not in entry:
            raise ValueError("No colon separator found in ratio format")

        parts = entry.split(':')
        if len(parts) != 2:
            raise ValueError("Ratio format must have exactly two parts separated by colon")

        ticker1, ticker2 = [t.strip() for t in parts]
        if not ticker1 or not ticker2:
            raise ValueError("Both tickers must be specified in ratio format")

        result = {
            'format_type': 'legacy_ratio',
            'prefix': None,
            'position': 'main',
            'indicator': 'RATIO',
            'parameters': {},
            'tickers': [ticker1, ticker2],
            'has_indicator': True,
            'original_entry': entry,
            'is_multi_ticker': True
        }

        logger.debug(f"Parsed legacy ratio format: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing legacy ratio format '{entry}': {e}")
        raise ValueError(f"Invalid legacy ratio format: {entry}")


def _parse_legacy_indicator_format(entry: str, main_panel_ticker: str = None) -> Dict[str, Any]:
    """
    Parse legacy indicator format: PPO(12,26,9)

    Convert to: PPO(12,26,9)_for_(main_panel_ticker)
    """
    try:
        if not validate_indicator_string(entry):
            raise ValueError("Invalid indicator string")

        indicator_name = extract_indicator_name(entry)
        parameters = parse_indicator_string(entry)

        # Use main panel ticker or mark for auto-detection
        tickers = [main_panel_ticker] if main_panel_ticker else ['AUTO_DETECT']

        result = {
            'format_type': 'legacy_indicator',
            'prefix': None,
            'position': 'main',
            'indicator': indicator_name,
            'parameters': parameters,
            'tickers': tickers,
            'has_indicator': True,
            'original_entry': entry,
            'is_multi_ticker': False,
            'requires_main_ticker': not main_panel_ticker
        }

        logger.debug(f"Parsed legacy indicator format: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing legacy indicator format '{entry}': {e}")
        raise ValueError(f"Invalid legacy indicator format: {entry}")


def _parse_simple_ticker_format(entry: str) -> Dict[str, Any]:
    """
    Parse simple ticker format: QQQ

    This is already in the correct format for simple tickers.
    """
    try:
        ticker = entry.strip()
        if not ticker:
            raise ValueError("Empty ticker")

        # Basic ticker validation (alphanumeric plus common symbols)
        if not re.match(r'^[A-Z0-9\.\-\^]+$', ticker.upper()):
            logger.warning(f"Unusual ticker format: {ticker}")

        result = {
            'format_type': 'simple_ticker',
            'prefix': None,
            'position': 'main',
            'indicator': None,
            'parameters': {},
            'tickers': [ticker],
            'has_indicator': False,
            'original_entry': entry,
            'is_multi_ticker': False
        }

        logger.debug(f"Parsed simple ticker format: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing simple ticker format '{entry}': {e}")
        raise ValueError(f"Invalid simple ticker format: {entry}")


def _parse_indicator_parameters(indicator_name: str, params_str: str) -> Dict[str, Any]:
    """
    Parse indicator parameters from parameter string.

    Args:
        indicator_name: Name of the indicator
        params_str: Parameter string (contents of parentheses)

    Returns:
        Dict with parsed parameters
    """
    try:
        if not params_str.strip():
            return {}

        # Use existing indicator parser
        full_indicator_string = f"{indicator_name}({params_str})"
        return parse_indicator_string(full_indicator_string)

    except Exception as e:
        logger.warning(f"Error parsing parameters for {indicator_name}({params_str}): {e}")
        return {}


def convert_legacy_to_enhanced(entry: str, main_panel_ticker: str = None) -> str:
    """
    Convert legacy format to enhanced format.

    Args:
        entry: Original entry string
        main_panel_ticker: Default ticker for legacy indicators

    Returns:
        Enhanced format string
    """
    try:
        # Direct mappings for known legacy formats
        if entry in LEGACY_MAPPINGS:
            return LEGACY_MAPPINGS[entry]

        # Parse entry to get structured information
        parsed = parse_enhanced_panel_entry(entry, main_panel_ticker)
        if not parsed:
            return entry  # Return original if parsing fails

        # Convert based on format type
        if parsed['format_type'] == 'enhanced':
            return entry  # Already enhanced

        elif parsed['format_type'] == 'simple_ticker':
            return entry  # Simple tickers stay as-is

        elif parsed['format_type'] == 'legacy_ab_ticker':
            # A_QQQ -> A_PRICE_for_(QQQ)
            prefix = parsed['prefix']
            ticker = parsed['tickers'][0]
            return f"{prefix}PRICE_for_({ticker})"

        elif parsed['format_type'] == 'legacy_ab_indicator':
            # A_PPO(12,26,9) -> A_PPO(12,26,9)_for_(main_panel_ticker)
            prefix = parsed['prefix']
            indicator = parsed['indicator']
            params = _format_parameters(parsed['parameters'])
            ticker = main_panel_ticker or 'AUTO_DETECT'
            return f"{prefix}{indicator}{params}_for_({ticker})"

        elif parsed['format_type'] == 'legacy_ratio':
            # XLY:XLP -> RATIO_for_(XLY,XLP)
            tickers = ','.join(parsed['tickers'])
            return f"RATIO_for_({tickers})"

        elif parsed['format_type'] == 'legacy_indicator':
            # PPO(12,26,9) -> PPO(12,26,9)_for_(main_panel_ticker)
            indicator = parsed['indicator']
            params = _format_parameters(parsed['parameters'])
            ticker = main_panel_ticker or 'AUTO_DETECT'
            return f"{indicator}{params}_for_({ticker})"

        else:
            return entry  # Unknown format, return original

    except Exception as e:
        logger.error(f"Error converting legacy format '{entry}': {e}")
        return entry  # Return original on error


def _format_parameters(parameters: Dict[str, Any]) -> str:
    """
    Format parameters dict back to string format.

    Args:
        parameters: Dict with parameter values

    Returns:
        Formatted parameter string like "(12,26,9)"
    """
    if not parameters:
        return ""

    # Convert parameter values to comma-separated string
    param_values = []
    for key in sorted(parameters.keys()):
        value = parameters[key]
        param_values.append(str(value))

    if param_values:
        return f"({','.join(param_values)})"
    else:
        return ""


def validate_enhanced_entry(entry: str) -> Dict[str, Any]:
    """
    Validate enhanced panel entry and return validation results.

    Args:
        entry: Panel entry string to validate

    Returns:
        Dict with validation results:
        {
            'is_valid': bool,
            'format_type': str,
            'errors': List[str],
            'warnings': List[str],
            'parsed_data': Dict or None
        }
    """
    validation_result = {
        'is_valid': False,
        'format_type': None,
        'errors': [],
        'warnings': [],
        'parsed_data': None
    }

    try:
        # Attempt to parse entry
        parsed = parse_enhanced_panel_entry(entry)

        if parsed is None:
            validation_result['errors'].append("Failed to parse entry")
            return validation_result

        validation_result['parsed_data'] = parsed
        validation_result['format_type'] = parsed['format_type']

        # Format-specific validation
        if parsed['format_type'] == 'enhanced':
            _validate_enhanced_format(parsed, validation_result)
        elif parsed['format_type'].startswith('legacy_'):
            _validate_legacy_format(parsed, validation_result)
        elif parsed['format_type'] == 'simple_ticker':
            _validate_simple_ticker(parsed, validation_result)

        # Check if validation passed
        validation_result['is_valid'] = len(validation_result['errors']) == 0

    except Exception as e:
        validation_result['errors'].append(f"Validation error: {str(e)}")

    return validation_result


def _validate_enhanced_format(parsed: Dict[str, Any], validation_result: Dict[str, Any]) -> None:
    """Validate enhanced format specific requirements."""

    # Check indicator validity
    if parsed['has_indicator'] and parsed['indicator']:
        from ..indicators.indicator_parser import INDICATOR_REGISTRY
        if parsed['indicator'] not in INDICATOR_REGISTRY:
            validation_result['errors'].append(f"Unsupported indicator: {parsed['indicator']}")

    # Check ticker requirements
    if not parsed['tickers'] or 'AUTO_DETECT' in parsed['tickers']:
        validation_result['errors'].append("Enhanced format requires explicit ticker specification")

    # Multi-ticker validation
    if parsed['is_multi_ticker'] and parsed['indicator'] not in ['RATIO', 'CORR', 'BETA']:
        validation_result['warnings'].append(f"Indicator {parsed['indicator']} may not support multiple tickers")


def _validate_legacy_format(parsed: Dict[str, Any], validation_result: Dict[str, Any]) -> None:
    """Validate legacy format and add conversion recommendations."""

    validation_result['warnings'].append(f"Legacy format detected: {parsed['format_type']}")

    # Recommend conversion to enhanced format
    if parsed.get('requires_main_ticker'):
        validation_result['warnings'].append("Legacy indicator format requires main panel ticker for conversion")


def _validate_simple_ticker(parsed: Dict[str, Any], validation_result: Dict[str, Any]) -> None:
    """Validate simple ticker format."""

    ticker = parsed['tickers'][0] if parsed['tickers'] else ''

    # Basic ticker format validation
    if not re.match(r'^[A-Z0-9\.\-\^]+$', ticker.upper()):
        validation_result['warnings'].append(f"Unusual ticker format: {ticker}")

    if len(ticker) > 10:
        validation_result['warnings'].append(f"Unusually long ticker symbol: {ticker}")


def get_supported_formats() -> List[Dict[str, str]]:
    """
    Get list of supported panel entry formats with examples.

    Returns:
        List of format descriptions with examples
    """
    return [
        {
            'format': 'Enhanced Format',
            'pattern': '[Prefix]_IndicatorName(Params)_for_(Ticker1[,Ticker2,...])',
            'examples': [
                'A_PPO(12,26,9)_for_(QQQ)',
                'B_RATIO_for_(SPY,QQQ)',
                'PPO(12,26,9)_for_(AAPL)',
                'RATIO_for_(XLY,XLP)'
            ]
        },
        {
            'format': 'Legacy A_/B_ Positioning',
            'pattern': 'A_TICKER or B_TICKER or A_INDICATOR(params)',
            'examples': [
                'A_QQQ',
                'B_SPY',
                'A_PPO(12,26,9)'
            ]
        },
        {
            'format': 'Legacy Ratio',
            'pattern': 'TICKER1:TICKER2',
            'examples': [
                'XLY:XLP',
                'SPY:QQQ'
            ]
        },
        {
            'format': 'Legacy Indicator',
            'pattern': 'INDICATOR(params)',
            'examples': [
                'PPO(12,26,9)',
                'RSI(14)',
                'EMA(20)'
            ]
        },
        {
            'format': 'Simple Ticker',
            'pattern': 'TICKER',
            'examples': [
                'QQQ',
                'SPY',
                'AAPL'
            ]
        },
        {
            'format': 'Ticker-First Indicator',
            'pattern': 'INDICATOR(TICKER,params)',
            'examples': [
                'PPO(QQQ,12,26,9)',
                'RSI(SPY,14)',
                'EMA(QQQ,20)'
            ]
        }
    ]


def _is_ticker_first_indicator(entry: str) -> bool:
    """
    Check if entry is a ticker-first indicator format: INDICATOR(TICKER,params)

    Examples:
    - PPO(QQQ,12,26,9) → True
    - RSI(SPY,14) → True
    - EMA(QQQ,20) → True
    - PPO(12,26,9) → False (legacy format)
    - QQQ → False (simple ticker)

    Args:
        entry: Panel entry string

    Returns:
        True if entry matches ticker-first indicator pattern
    """
    try:
        if not entry or '(' not in entry or ')' not in entry:
            return False

        # Extract indicator name and parameters
        if not entry.count('(') == 1 or not entry.count(')') == 1:
            return False

        parts = entry.split('(')
        if len(parts) != 2:
            return False

        indicator_name = parts[0].strip()
        params_part = parts[1].rstrip(')').strip()

        if not params_part:
            return False

        # Check if indicator name is valid
        valid_indicators = ['PPO', 'RSI', 'EMA', 'SMA', 'MACD', 'RATIO', 'PRICE']
        if indicator_name.upper() not in valid_indicators:
            return False

        # Split parameters
        param_parts = [p.strip() for p in params_part.split(',')]

        # Must have at least 2 parameters (ticker + at least one numeric)
        if len(param_parts) < 2:
            return False

        # First parameter should look like a ticker
        first_param = param_parts[0]
        if not _is_ticker_like(first_param):
            return False

        # Remaining parameters should be numeric (for most indicators)
        remaining_params = param_parts[1:]
        for param in remaining_params:
            try:
                float(param)
            except ValueError:
                return False

        return True

    except Exception as e:
        logger.debug(f"Error checking ticker-first format for '{entry}': {e}")
        return False


def _parse_ticker_first_format(entry: str) -> Dict[str, Any]:
    """
    Parse ticker-first indicator format and convert to enhanced format.

    Converts:
    - PPO(QQQ,12,26,9) → PPO(12,26,9)_for_(QQQ)
    - RSI(SPY,14) → RSI(14)_for_(SPY)
    - EMA(QQQ,20) → EMA(20)_for_(QQQ)

    Args:
        entry: Ticker-first indicator string

    Returns:
        Parsed result equivalent to enhanced format
    """
    try:
        # Extract indicator name and parameters
        parts = entry.split('(')
        indicator_name = parts[0].strip()
        params_part = parts[1].rstrip(')').strip()

        param_parts = [p.strip() for p in params_part.split(',')]

        # Extract ticker (first parameter)
        ticker = param_parts[0]

        # Extract numeric parameters (remaining parameters)
        numeric_params = param_parts[1:]

        # Construct enhanced format equivalent
        if numeric_params:
            params_str = ','.join(numeric_params)
            enhanced_format = f"{indicator_name}({params_str})_for_({ticker})"
        else:
            enhanced_format = f"{indicator_name}_for_({ticker})"

        # Parse using enhanced format parser
        result = _parse_enhanced_format(enhanced_format)

        if result:
            # Add metadata about original format
            result['original_format'] = 'ticker_first'
            result['original_entry'] = entry

        return result

    except Exception as e:
        logger.error(f"Error parsing ticker-first format '{entry}': {e}")
        return None