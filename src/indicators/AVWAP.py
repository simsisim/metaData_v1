"""
Anchored Volume Weighted Average Price (AVWAP) Module
====================================================

Simple, direct implementation of Anchored VWAP calculation.
Calculates volume-weighted average price starting from a specific anchor date.

AVWAP Formula:
AVWAP = Cumulative(Price * Volume) / Cumulative(Volume) from anchor date onward

Usage:
    from src.indicators.AVWAP import calculate_anchored_vwap

    # Calculate AVWAP anchored at 2024-02-02 using Open prices
    avwap = calculate_anchored_vwap(data, '2024-02-02', 'Open')
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_anchored_vwap(
    data: pd.DataFrame,
    anchor_date: str,
    price_type: str = 'Close'
) -> pd.Series:
    """
    Calculate Anchored Volume Weighted Average Price (AVWAP).

    Calculates cumulative volume-weighted average price starting from the anchor date.
    Values before the anchor date are set to NaN.

    Args:
        data: DataFrame with OHLCV data
              Required columns: 'Open', 'High', 'Low', 'Close', 'Volume'
              Index must be datetime
        anchor_date: Anchor date as string (e.g., '2024-02-02')
        price_type: Price column to use ('Open', 'High', 'Low', 'Close', 'TP')
                   'TP' = Typical Price (High + Low + Close) / 3

    Returns:
        pd.Series: AVWAP values with same index as input data
                  NaN before anchor_date, calculated AVWAP from anchor_date onward

    Raises:
        ValueError: If anchor_date not found in data or invalid price_type
        KeyError: If required columns missing from data

    Example:
        >>> data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)
        >>> avwap = calculate_anchored_vwap(data, '2024-02-02', 'Open')
        >>> print(avwap.dropna().head())
    """
    try:
        # Input validation
        _validate_input(data, anchor_date, price_type)

        # Convert anchor_date to datetime
        anchor_dt = pd.to_datetime(anchor_date)

        # Find anchor index
        anchor_idx = _find_anchor_index(data, anchor_dt)

        # Get price series
        price_series = _get_price_series(data, price_type)

        # Calculate AVWAP from anchor onward
        anchor_data = data.loc[anchor_idx:]
        price_data = price_series.loc[anchor_idx:]
        volume_data = anchor_data['Volume']

        # Calculate cumulative price * volume and cumulative volume
        cumulative_price_volume = (price_data * volume_data).cumsum()
        cumulative_volume = volume_data.cumsum()

        # Avoid division by zero
        avwap_values = cumulative_price_volume / cumulative_volume.replace(0, np.nan)

        # Create full series with NaN before anchor
        avwap_series = pd.Series(
            np.nan,
            index=data.index,
            name=f'AVWAP_{anchor_date}_{price_type}'
        )
        avwap_series.loc[anchor_idx:] = avwap_values

        logger.info(f"Calculated AVWAP for anchor {anchor_date} using {price_type} price")
        return avwap_series

    except Exception as e:
        logger.error(f"Error calculating AVWAP: {e}")
        # Return series of NaN on error
        return pd.Series(np.nan, index=data.index, name=f'AVWAP_{anchor_date}_{price_type}')


def _validate_input(data: pd.DataFrame, anchor_date: str, price_type: str) -> None:
    """Validate input parameters."""
    # Check if data is DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    # Check if data is empty
    if data.empty:
        raise ValueError("Input data is empty")

    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Check if index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    # Validate price_type
    valid_price_types = ['Open', 'High', 'Low', 'Close', 'TP']
    if price_type not in valid_price_types:
        raise ValueError(f"Invalid price_type '{price_type}'. Must be one of: {valid_price_types}")

    # Check anchor_date format
    try:
        pd.to_datetime(anchor_date)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid anchor_date format: '{anchor_date}'. Use format like '2024-02-02'")


def _find_anchor_index(data: pd.DataFrame, anchor_dt: pd.Timestamp) -> pd.Timestamp:
    """Find the first index on or after the anchor date."""
    # Find indices on or after anchor date
    valid_indices = data.index[data.index >= anchor_dt]

    if len(valid_indices) == 0:
        raise ValueError(f"Anchor date {anchor_dt.date()} is after all available data")

    # Return the first valid index
    anchor_idx = valid_indices[0]

    # Log if anchor date was adjusted
    if anchor_idx.date() != anchor_dt.date():
        logger.warning(f"Anchor date {anchor_dt.date()} not found in data. Using next available date: {anchor_idx.date()}")

    return anchor_idx


def _get_price_series(data: pd.DataFrame, price_type: str) -> pd.Series:
    """Get the appropriate price series based on price_type."""
    if price_type == 'TP':
        # Typical Price: (High + Low + Close) / 3
        return (data['High'] + data['Low'] + data['Close']) / 3
    else:
        return data[price_type]


def calculate_multiple_avwap(
    data: pd.DataFrame,
    anchor_dates: list,
    price_type: str = 'Close'
) -> pd.DataFrame:
    """
    Calculate multiple AVWAP lines with different anchor dates.

    Args:
        data: DataFrame with OHLCV data
        anchor_dates: List of anchor date strings
        price_type: Price column to use

    Returns:
        DataFrame with multiple AVWAP columns

    Example:
        >>> anchors = ['2024-01-15', '2024-02-02', '2024-03-01']
        >>> avwaps = calculate_multiple_avwap(data, anchors, 'Close')
    """
    result_df = pd.DataFrame(index=data.index)

    for anchor_date in anchor_dates:
        try:
            avwap_series = calculate_anchored_vwap(data, anchor_date, price_type)
            column_name = f'AVWAP_{anchor_date}_{price_type}'
            result_df[column_name] = avwap_series
        except Exception as e:
            logger.error(f"Failed to calculate AVWAP for {anchor_date}: {e}")
            continue

    return result_df


def get_avwap_statistics(avwap_series: pd.Series, current_price: float) -> dict:
    """
    Get statistics about AVWAP performance.

    Args:
        avwap_series: Calculated AVWAP series
        current_price: Current price for comparison

    Returns:
        Dict with AVWAP statistics
    """
    # Get latest AVWAP value
    latest_avwap = avwap_series.dropna().iloc[-1] if len(avwap_series.dropna()) > 0 else np.nan

    if pd.isna(latest_avwap):
        return {
            'latest_avwap': np.nan,
            'price_distance_pct': np.nan,
            'price_above_avwap': False,
            'days_calculated': 0
        }

    # Calculate statistics
    price_distance_pct = ((current_price - latest_avwap) / latest_avwap) * 100
    price_above_avwap = current_price > latest_avwap
    days_calculated = len(avwap_series.dropna())

    return {
        'latest_avwap': latest_avwap,
        'price_distance_pct': price_distance_pct,
        'price_above_avwap': price_above_avwap,
        'days_calculated': days_calculated
    }


def validate_avwap_input(data: pd.DataFrame) -> bool:
    """
    Validate input data for AVWAP calculation.

    Args:
        data: Input DataFrame

    Returns:
        True if valid, False otherwise
    """
    try:
        _validate_input(data, '2024-01-01', 'Close')  # Use dummy values for validation
        return True
    except (ValueError, KeyError):
        return False


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 2,
        'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 2,
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'Volume': np.random.randint(10000, 100000, len(dates))
    }, index=dates)

    # Test the function
    print("Testing Anchored VWAP calculation...")
    avwap = calculate_anchored_vwap(sample_data, '2024-02-02', 'Open')

    print(f"AVWAP calculated for {len(avwap.dropna())} days")
    print(f"Latest AVWAP value: {avwap.dropna().iloc[-1]:.2f}")

    # Test statistics
    current_price = sample_data['Close'].iloc[-1]
    stats = get_avwap_statistics(avwap, current_price)
    print(f"Statistics: {stats}")