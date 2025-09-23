"""
RATIO Indicator Module
======================

Calculates price ratios between two tickers for relative strength analysis.
Supports both simple ratios and normalized ratios for chart generation.

Functions:
- parse_ratio_params: Parse RATIO() parameters
- calculate_ratio_for_chart: Calculate ratio data for chart display
- calculate_normalized_ratio: Calculate ratio with normalization
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Union, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def parse_ratio_params(param_string: str) -> Dict[str, Any]:
    """
    Parse RATIO indicator parameter string.

    RATIO indicators typically don't have parameters, but this function
    provides consistency with other indicator parsers.

    Args:
        param_string: String like "RATIO()" or "RATIO"

    Returns:
        Dict with ratio parameters (empty for basic ratio)

    Example:
        >>> parse_ratio_params("RATIO()")
        {'normalize': False, 'baseline_date': None}
    """
    try:
        # Extract parameters if present
        if '(' in param_string and ')' in param_string:
            # For future extensions like RATIO(normalize=True)
            params_content = param_string.split('(')[1].split(')')[0].strip()

            # Default parameters
            params = {
                'normalize': False,
                'baseline_date': None,
                'smoothing_period': None
            }

            # Parse any future parameters here
            if params_content:
                # Reserved for future parameter parsing
                logger.debug(f"RATIO parameters not yet implemented: {params_content}")

            return params
        else:
            # Simple RATIO without parameters
            return {
                'normalize': False,
                'baseline_date': None,
                'smoothing_period': None
            }

    except Exception as e:
        logger.error(f"Error parsing RATIO parameters '{param_string}': {e}")
        return {
            'normalize': False,
            'baseline_date': None,
            'smoothing_period': None
        }


def calculate_ratio_for_chart(numerator_data: pd.DataFrame,
                             denominator_data: pd.DataFrame,
                             normalize: bool = False,
                             baseline_date: Optional[str] = None,
                             smoothing_period: Optional[int] = None,
                             price_column: str = 'Close') -> Dict[str, pd.Series]:
    """
    Calculate ratio between two ticker datasets for chart generation.

    Args:
        numerator_data: Price data for numerator ticker (DataFrame with OHLCV)
        denominator_data: Price data for denominator ticker (DataFrame with OHLCV)
        normalize: Whether to normalize ratio to baseline value (default: False)
        baseline_date: Date to use as baseline for normalization (default: first date)
        smoothing_period: Optional smoothing period for ratio (default: None)
        price_column: Column to use for ratio calculation (default: 'Close')

    Returns:
        Dict with calculated ratio data:
        {
            'ratio': pd.Series with ratio values,
            'ratio_smooth': pd.Series with smoothed ratio (if smoothing_period specified),
            'metadata': Dict with calculation metadata
        }

    Example:
        >>> result = calculate_ratio_for_chart(spy_data, qqq_data)
        >>> ratio_values = result['ratio']
    """
    try:
        # Validate input data
        if numerator_data.empty or denominator_data.empty:
            raise ValueError("Input data cannot be empty")

        if price_column not in numerator_data.columns:
            raise ValueError(f"Price column '{price_column}' not found in numerator data")

        if price_column not in denominator_data.columns:
            raise ValueError(f"Price column '{price_column}' not found in denominator data")

        # Align data on common dates
        common_dates = numerator_data.index.intersection(denominator_data.index)
        if common_dates.empty:
            raise ValueError("No common dates found between datasets")

        # Extract aligned price series
        numerator_prices = numerator_data.loc[common_dates, price_column]
        denominator_prices = denominator_data.loc[common_dates, price_column]

        # Calculate basic ratio
        # Avoid division by zero
        denominator_safe = denominator_prices.replace(0, np.nan)
        ratio = numerator_prices / denominator_safe

        # Remove infinite values
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        # Apply normalization if requested
        if normalize:
            ratio = _normalize_ratio(ratio, baseline_date)

        # Apply smoothing if requested
        ratio_smooth = None
        if smoothing_period and smoothing_period > 1:
            ratio_smooth = ratio.rolling(window=smoothing_period, min_periods=1).mean()

        # Calculate additional ratio statistics
        ratio_stats = _calculate_ratio_statistics(ratio)

        # Prepare metadata
        metadata = {
            'calculation_type': 'ratio',
            'numerator_ticker': getattr(numerator_data, 'attrs', {}).get('ticker', 'Unknown'),
            'denominator_ticker': getattr(denominator_data, 'attrs', {}).get('ticker', 'Unknown'),
            'price_column': price_column,
            'total_periods': len(ratio.dropna()),
            'date_range': {
                'start': str(ratio.index.min().date()) if not ratio.empty else 'N/A',
                'end': str(ratio.index.max().date()) if not ratio.empty else 'N/A'
            },
            'normalized': normalize,
            'smoothing_period': smoothing_period,
            'statistics': ratio_stats
        }

        result = {
            'ratio': ratio,
            'metadata': metadata
        }

        if ratio_smooth is not None:
            result['ratio_smooth'] = ratio_smooth

        logger.debug(f"Calculated ratio: {len(ratio)} data points, "
                    f"range {ratio.min():.4f} to {ratio.max():.4f}")

        return result

    except Exception as e:
        logger.error(f"Error calculating ratio: {e}")
        # Return empty result with error metadata
        return {
            'ratio': pd.Series(dtype=float),
            'metadata': {
                'calculation_type': 'ratio',
                'error': str(e),
                'total_periods': 0
            }
        }


def calculate_ratio_from_single_data(data: Union[pd.Series, pd.DataFrame],
                                   ratio_tickers: Tuple[str, str],
                                   **kwargs) -> Dict[str, pd.Series]:
    """
    Calculate ratio when data is provided as single source with multiple tickers.

    This is a compatibility function for the main indicator interface.

    Args:
        data: Combined data source (not used for RATIO)
        ratio_tickers: Tuple of (numerator_ticker, denominator_ticker)
        **kwargs: Additional parameters

    Returns:
        Dict with ratio calculation results

    Note:
        This function is a placeholder. Actual ratio calculation requires
        separate data for each ticker, which should be handled by the
        calling code through calculate_ratio_for_chart().
    """
    logger.warning("RATIO calculation requires separate ticker data. "
                  "Use calculate_ratio_for_chart() with individual ticker datasets.")

    return {
        'ratio': pd.Series(dtype=float),
        'metadata': {
            'calculation_type': 'ratio',
            'error': 'RATIO requires separate ticker datasets',
            'ratio_tickers': ratio_tickers
        }
    }


def _normalize_ratio(ratio: pd.Series, baseline_date: Optional[str] = None) -> pd.Series:
    """
    Normalize ratio to baseline value.

    Args:
        ratio: Original ratio series
        baseline_date: Date to use as baseline (default: first non-null value)

    Returns:
        Normalized ratio series
    """
    try:
        if ratio.empty:
            return ratio

        # Determine baseline value
        if baseline_date:
            try:
                baseline_date_parsed = pd.to_datetime(baseline_date)
                if baseline_date_parsed in ratio.index:
                    baseline_value = ratio.loc[baseline_date_parsed]
                else:
                    # Use closest date
                    closest_idx = ratio.index.get_indexer([baseline_date_parsed], method='nearest')[0]
                    baseline_value = ratio.iloc[closest_idx]
            except (ValueError, KeyError):
                logger.warning(f"Invalid baseline date '{baseline_date}', using first value")
                baseline_value = ratio.dropna().iloc[0] if not ratio.dropna().empty else 1.0
        else:
            # Use first non-null value
            baseline_value = ratio.dropna().iloc[0] if not ratio.dropna().empty else 1.0

        if baseline_value == 0 or pd.isna(baseline_value):
            logger.warning("Invalid baseline value for normalization, skipping normalization")
            return ratio

        # Normalize to baseline (baseline = 100)
        normalized_ratio = (ratio / baseline_value) * 100

        return normalized_ratio

    except Exception as e:
        logger.error(f"Error normalizing ratio: {e}")
        return ratio


def _calculate_ratio_statistics(ratio: pd.Series) -> Dict[str, float]:
    """
    Calculate statistical measures for ratio series.

    Args:
        ratio: Ratio data series

    Returns:
        Dict with statistical measures
    """
    try:
        if ratio.empty:
            return {}

        clean_ratio = ratio.dropna()
        if clean_ratio.empty:
            return {}

        stats = {
            'mean': float(clean_ratio.mean()),
            'median': float(clean_ratio.median()),
            'std': float(clean_ratio.std()),
            'min': float(clean_ratio.min()),
            'max': float(clean_ratio.max()),
            'current': float(clean_ratio.iloc[-1]) if len(clean_ratio) > 0 else np.nan,
            'volatility': float(clean_ratio.pct_change().std() * np.sqrt(252)) if len(clean_ratio) > 1 else np.nan
        }

        # Calculate percentiles
        stats.update({
            'percentile_25': float(clean_ratio.quantile(0.25)),
            'percentile_75': float(clean_ratio.quantile(0.75))
        })

        return stats

    except Exception as e:
        logger.error(f"Error calculating ratio statistics: {e}")
        return {}


def validate_ratio_data(numerator_data: pd.DataFrame,
                       denominator_data: pd.DataFrame,
                       price_column: str = 'Close') -> Dict[str, Any]:
    """
    Validate data quality for ratio calculation.

    Args:
        numerator_data: Numerator ticker data
        denominator_data: Denominator ticker data
        price_column: Price column to validate

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
        if numerator_data.empty:
            validation['errors'].append("Numerator data is empty")
            validation['is_valid'] = False

        if denominator_data.empty:
            validation['errors'].append("Denominator data is empty")
            validation['is_valid'] = False

        if not validation['is_valid']:
            return validation

        # Check required columns
        if price_column not in numerator_data.columns:
            validation['errors'].append(f"Price column '{price_column}' missing from numerator data")
            validation['is_valid'] = False

        if price_column not in denominator_data.columns:
            validation['errors'].append(f"Price column '{price_column}' missing from denominator data")
            validation['is_valid'] = False

        if not validation['is_valid']:
            return validation

        # Check for common dates
        common_dates = numerator_data.index.intersection(denominator_data.index)
        if common_dates.empty:
            validation['errors'].append("No common dates between datasets")
            validation['is_valid'] = False
            return validation

        # Data quality checks
        num_prices = numerator_data.loc[common_dates, price_column]
        den_prices = denominator_data.loc[common_dates, price_column]

        # Check for null values
        num_nulls = num_prices.isna().sum()
        den_nulls = den_prices.isna().sum()

        if num_nulls > 0:
            validation['warnings'].append(f"Numerator has {num_nulls} null values")

        if den_nulls > 0:
            validation['warnings'].append(f"Denominator has {den_nulls} null values")

        # Check for zero values in denominator
        zero_denominators = (den_prices == 0).sum()
        if zero_denominators > 0:
            validation['warnings'].append(f"Denominator has {zero_denominators} zero values")

        # Data summary
        validation['data_summary'] = {
            'common_periods': len(common_dates),
            'date_range': {
                'start': str(common_dates.min().date()) if len(common_dates) > 0 else 'N/A',
                'end': str(common_dates.max().date()) if len(common_dates) > 0 else 'N/A'
            },
            'numerator_nulls': int(num_nulls),
            'denominator_nulls': int(den_nulls),
            'zero_denominators': int(zero_denominators)
        }

    except Exception as e:
        validation['errors'].append(f"Validation error: {str(e)}")
        validation['is_valid'] = False

    return validation


def get_ratio_chart_config() -> Dict[str, Any]:
    """
    Get recommended chart configuration for ratio indicators.

    Returns:
        Dict with chart configuration settings
    """
    return {
        'chart_type': 'subplot',
        'y_axis_label': 'Ratio',
        'line_style': 'solid',
        'line_width': 1.5,
        'color_scheme': 'blue',
        'grid': True,
        'subplot_height_ratio': 0.3,
        'technical_levels': {
            'show_mean': True,
            'show_std_bands': True,
            'show_percentiles': [25, 75]
        },
        'annotations': {
            'show_current_value': True,
            'show_statistics': True
        }
    }