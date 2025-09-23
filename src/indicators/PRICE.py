"""
PRICE Indicator Module
======================

Simple price chart indicator for displaying ticker price data.
Used for basic price charts without additional technical indicators.

Functions:
- parse_price_params: Parse PRICE() parameters
- calculate_price_for_chart: Prepare price data for chart display
- format_price_data: Format OHLCV data for charting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Union, Optional, Any

logger = logging.getLogger(__name__)


def parse_price_params(param_string: str) -> Dict[str, Any]:
    """
    Parse PRICE indicator parameter string.

    PRICE indicators typically don't have parameters, but this function
    provides consistency with other indicator parsers.

    Args:
        param_string: String like "PRICE()" or "PRICE"

    Returns:
        Dict with price display parameters

    Example:
        >>> parse_price_params("PRICE()")
        {'chart_type': 'candlestick', 'show_volume': True}
    """
    try:
        # Default parameters for price charts
        params = {
            'chart_type': 'candlestick',  # 'candlestick', 'ohlc', 'line'
            'show_volume': True,
            'price_column': 'Close',
            'volume_subplot': True,
            'show_gaps': True
        }

        # Extract parameters if present
        if '(' in param_string and ')' in param_string:
            params_content = param_string.split('(')[1].split(')')[0].strip()

            # Parse any future parameters here
            if params_content:
                # Reserved for future parameter parsing like PRICE(line) or PRICE(volume=false)
                logger.debug(f"PRICE parameters not yet implemented: {params_content}")

        return params

    except Exception as e:
        logger.error(f"Error parsing PRICE parameters '{param_string}': {e}")
        return {
            'chart_type': 'candlestick',
            'show_volume': True,
            'price_column': 'Close',
            'volume_subplot': True,
            'show_gaps': True
        }


def calculate_price_for_chart(data: Union[pd.Series, pd.DataFrame],
                             chart_type: str = 'candlestick',
                             show_volume: bool = True,
                             price_column: str = 'Close',
                             volume_subplot: bool = True,
                             show_gaps: bool = True) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    """
    Prepare price data for chart generation.

    Args:
        data: Price data (Series for single column, DataFrame for OHLCV)
        chart_type: Type of price chart ('candlestick', 'ohlc', 'line')
        show_volume: Whether to include volume data
        price_column: Column to use for line charts (default: 'Close')
        volume_subplot: Whether volume should be in subplot
        show_gaps: Whether to show gaps in data

    Returns:
        Dict with formatted price data:
        {
            'price_data': pd.DataFrame with OHLCV data,
            'chart_config': Dict with chart configuration,
            'metadata': Dict with data metadata
        }

    Example:
        >>> result = calculate_price_for_chart(ticker_data)
        >>> price_df = result['price_data']
    """
    try:
        # Validate input data
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ValueError("Input data cannot be empty")

        # Convert Series to DataFrame if needed
        if isinstance(data, pd.Series):
            price_data = pd.DataFrame({'Close': data})
            chart_type = 'line'  # Force line chart for series data
            show_volume = False  # No volume data available
        else:
            price_data = data.copy()

        # Validate required columns based on chart type
        required_columns = _get_required_columns(chart_type, show_volume)
        missing_columns = [col for col in required_columns if col not in price_data.columns]

        if missing_columns:
            logger.warning(f"Missing columns for {chart_type} chart: {missing_columns}")
            # Fallback to line chart if OHLC data not available
            if 'Close' in price_data.columns:
                chart_type = 'line'
                show_volume = False
                logger.info("Falling back to line chart using Close prices")
            else:
                raise ValueError(f"Required columns missing: {missing_columns}")

        # Prepare chart data based on type
        formatted_data = _format_chart_data(price_data, chart_type, price_column)

        # Add volume data if requested and available
        volume_data = None
        if show_volume and 'Volume' in price_data.columns:
            volume_data = price_data['Volume'].copy()
            volume_data = volume_data.dropna()

        # Calculate price statistics
        price_stats = _calculate_price_statistics(price_data, price_column)

        # Prepare chart configuration
        chart_config = {
            'chart_type': chart_type,
            'show_volume': show_volume and volume_data is not None,
            'volume_subplot': volume_subplot,
            'show_gaps': show_gaps,
            'price_column': price_column,
            'y_axis_label': 'Price ($)',
            'title_suffix': f"Price Chart ({chart_type.title()})"
        }

        # Prepare metadata
        metadata = {
            'calculation_type': 'price',
            'chart_type': chart_type,
            'ticker': getattr(data, 'attrs', {}).get('ticker', 'Unknown'),
            'total_periods': len(price_data.dropna()),
            'date_range': {
                'start': str(price_data.index.min().date()) if not price_data.empty else 'N/A',
                'end': str(price_data.index.max().date()) if not price_data.empty else 'N/A'
            },
            'price_range': {
                'min': float(price_data[price_column].min()) if price_column in price_data.columns else None,
                'max': float(price_data[price_column].max()) if price_column in price_data.columns else None,
                'current': float(price_data[price_column].iloc[-1]) if price_column in price_data.columns and len(price_data) > 0 else None
            },
            'has_volume': volume_data is not None,
            'statistics': price_stats
        }

        result = {
            'price_data': formatted_data,
            'chart_config': chart_config,
            'metadata': metadata
        }

        if volume_data is not None:
            result['volume_data'] = volume_data

        logger.debug(f"Prepared price data: {len(formatted_data)} periods, "
                    f"chart type: {chart_type}, volume: {show_volume}")

        return result

    except Exception as e:
        logger.error(f"Error preparing price data for chart: {e}")
        # Return empty result with error metadata
        return {
            'price_data': pd.DataFrame(),
            'chart_config': {'chart_type': 'line', 'show_volume': False},
            'metadata': {
                'calculation_type': 'price',
                'error': str(e),
                'total_periods': 0
            }
        }


def _get_required_columns(chart_type: str, show_volume: bool) -> list:
    """
    Get required columns for specific chart type.

    Args:
        chart_type: Type of chart
        show_volume: Whether volume is required

    Returns:
        List of required column names
    """
    required = []

    if chart_type in ['candlestick', 'ohlc']:
        required.extend(['Open', 'High', 'Low', 'Close'])
    elif chart_type == 'line':
        required.append('Close')

    if show_volume:
        required.append('Volume')

    return required


def _format_chart_data(data: pd.DataFrame, chart_type: str, price_column: str) -> pd.DataFrame:
    """
    Format price data based on chart type requirements.

    Args:
        data: Original price data
        chart_type: Type of chart to format for
        price_column: Primary price column

    Returns:
        Formatted DataFrame for charting
    """
    try:
        if chart_type in ['candlestick', 'ohlc']:
            # Ensure OHLC columns exist and are properly formatted
            required_cols = ['Open', 'High', 'Low', 'Close']
            formatted_data = data[required_cols].copy()

            # Add volume if available
            if 'Volume' in data.columns:
                formatted_data['Volume'] = data['Volume']

            # Remove rows where any OHLC value is NaN
            formatted_data = formatted_data.dropna(subset=required_cols)

        elif chart_type == 'line':
            # Simple line chart using specified price column
            formatted_data = pd.DataFrame()
            formatted_data['Close'] = data[price_column]

            # Add volume if available
            if 'Volume' in data.columns:
                formatted_data['Volume'] = data['Volume']

            # Remove rows where price is NaN
            formatted_data = formatted_data.dropna(subset=['Close'])

        else:
            # Default fallback
            formatted_data = data.copy()

        return formatted_data

    except Exception as e:
        logger.error(f"Error formatting chart data: {e}")
        return pd.DataFrame()


def _calculate_price_statistics(data: pd.DataFrame, price_column: str) -> Dict[str, float]:
    """
    Calculate statistical measures for price data.

    Args:
        data: Price data DataFrame
        price_column: Column to analyze

    Returns:
        Dict with statistical measures
    """
    try:
        if data.empty or price_column not in data.columns:
            return {}

        prices = data[price_column].dropna()
        if prices.empty:
            return {}

        stats = {
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'current': float(prices.iloc[-1]) if len(prices) > 0 else np.nan
        }

        # Calculate returns and volatility if we have enough data
        if len(prices) > 1:
            returns = prices.pct_change().dropna()
            stats.update({
                'daily_volatility': float(returns.std()),
                'annualized_volatility': float(returns.std() * np.sqrt(252)),
                'total_return': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
                'max_drawdown': float(_calculate_max_drawdown(prices))
            })

        # Calculate price levels
        stats.update({
            'percentile_25': float(prices.quantile(0.25)),
            'percentile_75': float(prices.quantile(0.75)),
            'recent_high': float(prices.tail(20).max()) if len(prices) >= 20 else stats['max'],
            'recent_low': float(prices.tail(20).min()) if len(prices) >= 20 else stats['min']
        })

        return stats

    except Exception as e:
        logger.error(f"Error calculating price statistics: {e}")
        return {}


def _calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown for price series.

    Args:
        prices: Price data series

    Returns:
        Maximum drawdown as percentage
    """
    try:
        if len(prices) < 2:
            return 0.0

        # Calculate running maximum
        running_max = prices.expanding().max()

        # Calculate drawdown as percentage
        drawdown = (prices - running_max) / running_max * 100

        return abs(drawdown.min())

    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return 0.0


def validate_price_data(data: Union[pd.Series, pd.DataFrame],
                       chart_type: str = 'candlestick') -> Dict[str, Any]:
    """
    Validate price data for chart generation.

    Args:
        data: Price data to validate
        chart_type: Intended chart type

    Returns:
        Dict with validation results
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'data_summary': {}
    }

    try:
        # Check if data exists
        if data is None or (hasattr(data, 'empty') and data.empty):
            validation['errors'].append("Price data is empty")
            validation['is_valid'] = False
            return validation

        # Check required columns
        if isinstance(data, pd.Series):
            # Series is always valid for line charts
            validation['data_summary'] = {
                'data_type': 'Series',
                'periods': len(data),
                'null_values': data.isna().sum()
            }
        else:
            # DataFrame validation
            required_cols = _get_required_columns(chart_type, False)  # Check without volume
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                if 'Close' in data.columns:
                    validation['warnings'].append(
                        f"Missing columns for {chart_type}: {missing_cols}. "
                        "Will fallback to line chart."
                    )
                else:
                    validation['errors'].append(f"Missing required columns: {missing_cols}")
                    validation['is_valid'] = False

            validation['data_summary'] = {
                'data_type': 'DataFrame',
                'columns': list(data.columns),
                'periods': len(data),
                'missing_columns': missing_cols
            }

        # Data quality checks
        if validation['is_valid']:
            if isinstance(data, pd.Series):
                null_count = data.isna().sum()
                total_count = len(data)
            else:
                price_col = 'Close' if 'Close' in data.columns else data.columns[0]
                null_count = data[price_col].isna().sum()
                total_count = len(data)

            null_percentage = (null_count / total_count * 100) if total_count > 0 else 0

            if null_percentage > 50:
                validation['errors'].append(f"Too many null values: {null_percentage:.1f}%")
                validation['is_valid'] = False
            elif null_percentage > 10:
                validation['warnings'].append(f"High null values: {null_percentage:.1f}%")

            validation['data_summary'].update({
                'null_percentage': round(null_percentage, 2),
                'date_range': {
                    'start': str(data.index.min().date()) if hasattr(data, 'index') and not data.empty else 'N/A',
                    'end': str(data.index.max().date()) if hasattr(data, 'index') and not data.empty else 'N/A'
                }
            })

    except Exception as e:
        validation['errors'].append(f"Validation error: {str(e)}")
        validation['is_valid'] = False

    return validation


def get_price_chart_config() -> Dict[str, Any]:
    """
    Get recommended chart configuration for price indicators.

    Returns:
        Dict with chart configuration settings
    """
    return {
        'chart_type': 'candlestick',
        'show_volume': True,
        'volume_subplot': True,
        'volume_height_ratio': 0.2,
        'price_height_ratio': 0.8,
        'y_axis_label': 'Price ($)',
        'volume_y_axis_label': 'Volume',
        'grid': True,
        'line_style': 'solid',
        'line_width': 1.0,
        'candlestick_colors': {
            'up': 'green',
            'down': 'red',
            'up_wick': 'green',
            'down_wick': 'red'
        },
        'volume_colors': {
            'up': 'green',
            'down': 'red',
            'alpha': 0.7
        },
        'annotations': {
            'show_current_price': True,
            'show_price_change': True,
            'show_volume_average': True
        }
    }