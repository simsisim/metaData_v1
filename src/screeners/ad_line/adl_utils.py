"""
ADL Utilities Module
===================

Shared utility functions for ADL analysis modules.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def validate_ohlcv_data(df: pd.DataFrame, min_length: int = 50) -> bool:
    """
    Validate DataFrame has required OHLCV columns and sufficient data.

    Args:
        df: DataFrame to validate
        min_length: Minimum number of rows required

    Returns:
        True if valid, False otherwise
    """
    if df is None or len(df) == 0:
        return False

    if not isinstance(df, pd.DataFrame):
        return False

    # Check required columns
    required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return False

    # Check minimum length
    if len(df) < min_length:
        return False

    # Check for NaN values in critical columns
    if df[required_cols].isna().any().any():
        logger.warning("DataFrame contains NaN values in OHLCV columns")
        return False

    return True


def parse_semicolon_list(value: str, dtype: type = int) -> List:
    """
    Parse semicolon-separated string into list of specified type.

    Args:
        value: String like "5;10;20"
        dtype: Target data type (int or float)

    Returns:
        List of parsed values

    Examples:
        >>> parse_semicolon_list("5;10;20", int)
        [5, 10, 20]

        >>> parse_semicolon_list("0.4;0.3;0.3", float)
        [0.4, 0.3, 0.3]
    """
    try:
        return [dtype(x.strip()) for x in value.split(';') if x.strip()]
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing semicolon list '{value}': {e}")
        return []


def calculate_score_category(score: float) -> str:
    """
    Categorize score (0-100) into strength categories.

    Args:
        score: Numerical score from 0-100

    Returns:
        Category string
    """
    if score >= 90:
        return 'elite'
    elif score >= 80:
        return 'excellent'
    elif score >= 70:
        return 'good'
    elif score >= 60:
        return 'fair'
    else:
        return 'weak'


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return on division by zero

    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except:
        return default


def normalize_score(value: float, min_val: float, max_val: float,
                   reverse: bool = False) -> float:
    """
    Normalize value to 0-100 scale.

    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        reverse: If True, higher input values = lower scores

    Returns:
        Normalized score (0-100)
    """
    if max_val == min_val:
        return 50.0  # Middle score if no range

    # Clamp value to range
    value = max(min_val, min(max_val, value))

    # Normalize to 0-1
    normalized = (value - min_val) / (max_val - min_val)

    # Reverse if needed
    if reverse:
        normalized = 1.0 - normalized

    # Scale to 0-100
    return normalized * 100.0


def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculate percentage change between two values.

    Args:
        current: Current value
        previous: Previous value

    Returns:
        Percentage change
    """
    if previous == 0 or np.isnan(previous) or np.isnan(current):
        return 0.0

    return ((current - previous) / abs(previous)) * 100.0


def detect_trend(values: pd.Series, method: str = 'slope') -> str:
    """
    Detect trend direction in a series of values.

    Args:
        values: Series of values to analyze
        method: 'slope' for linear regression, 'simple' for first vs last

    Returns:
        'up', 'down', or 'neutral'
    """
    if len(values) < 2:
        return 'neutral'

    # Remove NaN values
    clean_values = values.dropna()
    if len(clean_values) < 2:
        return 'neutral'

    if method == 'slope':
        # Linear regression slope
        try:
            x = np.arange(len(clean_values))
            slope = np.polyfit(x, clean_values.values, 1)[0]

            if slope > 0.01:  # Threshold for positive trend
                return 'up'
            elif slope < -0.01:  # Threshold for negative trend
                return 'down'
            else:
                return 'neutral'
        except:
            return 'neutral'

    else:  # Simple comparison
        first = clean_values.iloc[0]
        last = clean_values.iloc[-1]

        if last > first * 1.02:  # 2% threshold
            return 'up'
        elif last < first * 0.98:
            return 'down'
        else:
            return 'neutral'


def format_result_dict(ticker: str, date, price: float, volume: int,
                      **additional_fields) -> Dict[str, Any]:
    """
    Create standardized result dictionary for screening output.

    Args:
        ticker: Stock ticker symbol
        date: Date of signal
        price: Current price
        volume: Current volume
        **additional_fields: Additional fields to include

    Returns:
        Standardized result dictionary
    """
    result = {
        'ticker': ticker,
        'date': date,
        'price': round(price, 2),
        'volume': int(volume),
    }

    # Add additional fields
    result.update(additional_fields)

    return result


def filter_by_threshold(df: pd.DataFrame, column: str,
                       threshold: float, operator: str = '>=') -> pd.DataFrame:
    """
    Filter DataFrame by threshold value.

    Args:
        df: DataFrame to filter
        column: Column name to filter on
        threshold: Threshold value
        operator: Comparison operator ('>=', '>', '<=', '<', '==')

    Returns:
        Filtered DataFrame
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return df

    if operator == '>=':
        return df[df[column] >= threshold]
    elif operator == '>':
        return df[df[column] > threshold]
    elif operator == '<=':
        return df[df[column] <= threshold]
    elif operator == '<':
        return df[df[column] < threshold]
    elif operator == '==':
        return df[df[column] == threshold]
    else:
        logger.warning(f"Unknown operator '{operator}'")
        return df


def calculate_weighted_score(scores: Dict[str, float],
                            weights: Dict[str, float]) -> float:
    """
    Calculate weighted composite score from component scores.

    Args:
        scores: Dictionary of component scores (e.g., {'longterm': 80, 'shortterm': 75})
        weights: Dictionary of weights (e.g., {'longterm': 0.4, 'shortterm': 0.3})

    Returns:
        Weighted composite score
    """
    total_weight = sum(weights.values())

    if total_weight == 0:
        logger.warning("Total weight is zero, returning 0")
        return 0.0

    weighted_sum = sum(scores.get(k, 0) * weights.get(k, 0) for k in weights.keys())

    return weighted_sum / total_weight * (1.0 / total_weight) * 100 if total_weight != 1.0 else weighted_sum


def extract_date_from_dataframe(df: pd.DataFrame) -> str:
    """
    Extract date string from DataFrame index (last date).

    Args:
        df: DataFrame with datetime index

    Returns:
        Date string in YYYY-MM-DD format
    """
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index[-1].strftime('%Y-%m-%d')
        elif 'Date' in df.columns:
            return pd.to_datetime(df['Date'].iloc[-1]).strftime('%Y-%m-%d')
        else:
            return pd.Timestamp.now().strftime('%Y-%m-%d')
    except:
        return pd.Timestamp.now().strftime('%Y-%m-%d')