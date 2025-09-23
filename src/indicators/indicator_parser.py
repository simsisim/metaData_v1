"""
Indicator Parameter Parser Module
=================================

Universal parser for CSV indicator parameter strings.
Supports parsing various indicator formats and routing to appropriate functions.

Usage:
    from src.indicators.indicator_parser import parse_indicator_string, get_indicator_function

    params = parse_indicator_string("PPO(12,26,9)")
    func = get_indicator_function("PPO")
    result = func(data, **params)
"""

import re
from typing import Dict, Union, Callable, Optional, Any
from .PPO import parse_ppo_params, calculate_ppo_for_chart
from .RSI import parse_rsi_params, calculate_rsi_for_chart
from .MAs import parse_ma_params, calculate_ema_for_chart, calculate_sma_for_chart
from .RATIO import parse_ratio_params, calculate_ratio_for_chart
from .PRICE import parse_price_params, calculate_price_for_chart
from .MACD import parse_macd_params, calculate_macd_for_chart
from .ADLINE import parse_adline_params, calculate_adline_for_chart


# Registry of supported indicators
INDICATOR_REGISTRY = {
    'PPO': {
        'parser': parse_ppo_params,
        'calculator': calculate_ppo_for_chart,
        'description': 'Percentage Price Oscillator'
    },
    'RSI': {
        'parser': parse_rsi_params,
        'calculator': calculate_rsi_for_chart,
        'description': 'Relative Strength Index'
    },
    'EMA': {
        'parser': parse_ma_params,
        'calculator': calculate_ema_for_chart,
        'description': 'Exponential Moving Average'
    },
    'SMA': {
        'parser': parse_ma_params,
        'calculator': calculate_sma_for_chart,
        'description': 'Simple Moving Average'
    },
    'RATIO': {
        'parser': parse_ratio_params,
        'calculator': calculate_ratio_for_chart,
        'description': 'Price Ratio between two tickers'
    },
    'PRICE': {
        'parser': parse_price_params,
        'calculator': calculate_price_for_chart,
        'description': 'Simple price chart'
    },
    'MACD': {
        'parser': parse_macd_params,
        'calculator': calculate_macd_for_chart,
        'description': 'Moving Average Convergence Divergence'
    },
    'ADLINE': {
        'parser': parse_adline_params,
        'calculator': calculate_adline_for_chart,
        'description': 'Accumulation Distribution Line'
    }
}


def extract_indicator_name(param_string: str) -> str:
    """
    Extract indicator name from parameter string.

    Args:
        param_string: String like "PPO(12,26,9)" or "RSI(14)"

    Returns:
        Indicator name (e.g., "PPO", "RSI")

    Example:
        >>> extract_indicator_name("PPO(12,26,9)")
        "PPO"
    """
    if not param_string or not isinstance(param_string, str):
        raise ValueError("Invalid parameter string")

    # Extract indicator name before opening parenthesis
    match = re.match(r'^([A-Z]+)', param_string.strip().upper())
    if not match:
        raise ValueError(f"Cannot extract indicator name from: {param_string}")

    return match.group(1)


def parse_indicator_string(param_string: str) -> Dict[str, Any]:
    """
    Parse any supported indicator parameter string.

    Args:
        param_string: String like "PPO(12,26,9)", "RSI(14)", etc.

    Returns:
        Dict with parsed parameters

    Example:
        >>> parse_indicator_string("PPO(12,26,9)")
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
    """
    if not param_string or param_string.strip() == '':
        return {}

    # Extract indicator name
    indicator_name = extract_indicator_name(param_string)

    # Check if indicator is supported
    if indicator_name not in INDICATOR_REGISTRY:
        raise ValueError(f"Unsupported indicator: {indicator_name}")

    # Use appropriate parser
    parser_func = INDICATOR_REGISTRY[indicator_name]['parser']
    return parser_func(param_string)


def get_indicator_function(indicator_name: str) -> Callable:
    """
    Get the calculation function for a specific indicator.

    Args:
        indicator_name: Name of the indicator (e.g., "PPO", "RSI")

    Returns:
        Calculation function

    Example:
        >>> func = get_indicator_function("PPO")
        >>> result = func(data, fast_period=12, slow_period=26, signal_period=9)
    """
    if indicator_name not in INDICATOR_REGISTRY:
        raise ValueError(f"Unsupported indicator: {indicator_name}")

    return INDICATOR_REGISTRY[indicator_name]['calculator']


def calculate_indicator(data, indicator_string: str, **kwargs) -> Dict[str, Any]:
    """
    Universal indicator calculation function.

    Args:
        data: Price data (Series or DataFrame)
        indicator_string: Parameter string like "PPO(12,26,9)"
        **kwargs: Additional parameters

    Returns:
        Dict with calculated indicator data

    Example:
        >>> result = calculate_indicator(data, "PPO(12,26,9)")
        >>> ppo_values = result['ppo']
    """
    if not indicator_string or indicator_string.strip() == '':
        return {}

    # Parse parameters
    params = parse_indicator_string(indicator_string)

    # Get indicator name and function
    indicator_name = extract_indicator_name(indicator_string)
    calc_func = get_indicator_function(indicator_name)

    # Merge with additional kwargs
    params.update(kwargs)

    # Calculate indicator
    try:
        return calc_func(data, **params)
    except Exception as e:
        raise ValueError(f"Error calculating {indicator_name}: {str(e)}")


def validate_indicator_string(param_string: str) -> bool:
    """
    Validate if an indicator parameter string is properly formatted.

    Args:
        param_string: String to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_indicator_string("PPO(12,26,9)")
        True
        >>> validate_indicator_string("INVALID()")
        False
    """
    try:
        # First check if indicator name exists
        indicator_name = extract_indicator_name(param_string)
        if indicator_name not in INDICATOR_REGISTRY:
            return False

        # Then try to parse the parameters to ensure format is valid
        parse_indicator_string(param_string)
        return True
    except ValueError:
        return False


def get_supported_indicators() -> Dict[str, str]:
    """
    Get list of all supported indicators.

    Returns:
        Dict mapping indicator names to descriptions

    Example:
        >>> indicators = get_supported_indicators()
        >>> print(indicators['PPO'])
        "Percentage Price Oscillator"
    """
    return {name: info['description'] for name, info in INDICATOR_REGISTRY.items()}


def register_indicator(name: str, parser_func: Callable, calc_func: Callable, description: str = ''):
    """
    Register a new indicator in the system.

    Args:
        name: Indicator name (e.g., "MACD")
        parser_func: Function to parse parameter strings
        calc_func: Function to calculate the indicator
        description: Human-readable description

    Example:
        >>> register_indicator("MACD", parse_macd_params, calculate_macd_for_chart, "MACD Indicator")
    """
    INDICATOR_REGISTRY[name.upper()] = {
        'parser': parser_func,
        'calculator': calc_func,
        'description': description or f'{name} Indicator'
    }


def parse_panel_indicators(panel_config: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Parse all indicator strings from a panel configuration.

    Args:
        panel_config: Dict with panel names as keys and indicator strings as values

    Returns:
        Dict with parsed parameters for each panel

    Example:
        >>> config = {"Panel_1_index": "PPO(12,26,9)", "Panel_6_index": "RSI(14)"}
        >>> parsed = parse_panel_indicators(config)
        >>> print(parsed["Panel_1_index"])
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
    """
    parsed_indicators = {}

    for panel_name, indicator_string in panel_config.items():
        if indicator_string and indicator_string.strip():
            try:
                parsed_indicators[panel_name] = {
                    'indicator_name': extract_indicator_name(indicator_string),
                    'parameters': parse_indicator_string(indicator_string),
                    'original_string': indicator_string
                }
            except ValueError as e:
                # Log warning but continue processing other indicators
                print(f"Warning: Could not parse indicator '{indicator_string}' for {panel_name}: {e}")
                parsed_indicators[panel_name] = {
                    'indicator_name': None,
                    'parameters': {},
                    'original_string': indicator_string,
                    'error': str(e)
                }

    return parsed_indicators


def get_indicator_chart_type(indicator_name: str) -> str:
    """
    Get the recommended chart type for an indicator.

    Args:
        indicator_name: Name of the indicator

    Returns:
        Chart type: 'overlay' or 'subplot'

    Example:
        >>> get_indicator_chart_type("PPO")
        'subplot'
        >>> get_indicator_chart_type("EMA")
        'overlay'
    """
    overlay_indicators = ['EMA', 'SMA', 'MA', 'ADLINE']
    subplot_indicators = ['PPO', 'RSI', 'MACD', 'MFI']

    if indicator_name.upper() in overlay_indicators:
        return 'overlay'
    elif indicator_name.upper() in subplot_indicators:
        return 'subplot'
    else:
        return 'subplot'  # Default to subplot for unknown indicators