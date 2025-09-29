"""
Centralized timeframe filtering utilities.

This module provides consistent timeframe determination logic across all analysis modules.
It implements a hierarchical flag system where:
- Master switches (YF_*_data) act as global gates for data availability
- Feature switches (RS_*_enable) act as feature-specific toggles

Only timeframes where BOTH levels are enabled will be processed.
"""

import logging

logger = logging.getLogger(__name__)


def get_enabled_timeframes(user_config, feature_prefix='rs'):
    """
    Get timeframes enabled at both master and feature levels.

    Args:
        user_config: User configuration object with flag attributes
        feature_prefix: Prefix for feature-specific flags ('rs', 'per', etc.)

    Returns:
        List of enabled timeframe strings: ['daily', 'weekly', 'monthly']

    Examples:
        # For RS/PER analysis
        timeframes = get_enabled_timeframes(user_config, 'rs')

        # For future modules that might have different prefixes
        timeframes = get_enabled_timeframes(user_config, 'indicators')
    """
    timeframes = []

    # Daily: Master YF_daily_data AND feature RS_daily_enable must both be TRUE
    master_daily = getattr(user_config, 'yf_daily_data', True)
    feature_daily = getattr(user_config, f'{feature_prefix}_daily_enable', True)
    if master_daily and feature_daily:
        timeframes.append('daily')

    # Weekly: Master YF_weekly_data AND feature RS_weekly_enable must both be TRUE
    master_weekly = getattr(user_config, 'yf_weekly_data', True)
    feature_weekly = getattr(user_config, f'{feature_prefix}_weekly_enable', False)
    if master_weekly and feature_weekly:
        timeframes.append('weekly')

    # Monthly: Master YF_monthly_data AND feature RS_monthly_enable must both be TRUE
    master_monthly = getattr(user_config, 'yf_monthly_data', True)
    feature_monthly = getattr(user_config, f'{feature_prefix}_monthly_enable', False)
    if master_monthly and feature_monthly:
        timeframes.append('monthly')

    # Intraday: Master TW_intraday_data AND feature support (if applicable)
    master_intraday = getattr(user_config, 'tw_intraday_data', False)
    feature_intraday = getattr(user_config, f'{feature_prefix}_intraday_enable', False)
    if master_intraday and feature_intraday:
        timeframes.append('intraday')

    logger.debug(f"Enabled timeframes for {feature_prefix}: {timeframes}")
    logger.debug(f"Master flags - daily:{master_daily}, weekly:{master_weekly}, monthly:{master_monthly}, intraday:{master_intraday}")
    logger.debug(f"Feature flags - daily:{feature_daily}, weekly:{feature_weekly}, monthly:{feature_monthly}, intraday:{feature_intraday}")

    return timeframes


def get_enabled_rs_timeframes(user_config):
    """
    Get timeframes enabled for RS analysis specifically.

    This is a convenience wrapper for get_enabled_timeframes with 'rs' prefix.

    Args:
        user_config: User configuration object

    Returns:
        List of enabled timeframe strings for RS analysis
    """
    return get_enabled_timeframes(user_config, feature_prefix='rs')


def validate_timeframe_consistency(user_config):
    """
    Validate timeframe configuration for potential issues.

    Args:
        user_config: User configuration object

    Returns:
        Dict with validation results:
        {
            'valid': bool,
            'warnings': list of warning messages,
            'errors': list of error messages
        }
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': []
    }

    # Check for feature flags enabled but master flags disabled
    timeframe_configs = [
        ('daily', 'yf_daily_data', 'rs_daily_enable'),
        ('weekly', 'yf_weekly_data', 'rs_weekly_enable'),
        ('monthly', 'yf_monthly_data', 'rs_monthly_enable')
    ]

    for timeframe, master_flag, feature_flag in timeframe_configs:
        master_enabled = getattr(user_config, master_flag, True)
        feature_enabled = getattr(user_config, feature_flag, False)

        if feature_enabled and not master_enabled:
            warning = f"{feature_flag}=TRUE but {master_flag}=FALSE - {timeframe} analysis will be skipped"
            validation['warnings'].append(warning)

    # Check if no timeframes are enabled at all
    enabled_timeframes = get_enabled_rs_timeframes(user_config)
    if not enabled_timeframes:
        validation['valid'] = False
        validation['errors'].append("No timeframes enabled for analysis - enable at least one YF_*_data and corresponding RS_*_enable flag")

    return validation


def log_timeframe_status(user_config, module_name="Analysis"):
    """
    Log current timeframe configuration status for debugging.

    Args:
        user_config: User configuration object
        module_name: Name of the module requesting timeframes
    """
    enabled_timeframes = get_enabled_rs_timeframes(user_config)
    validation = validate_timeframe_consistency(user_config)

    logger.info(f"{module_name} timeframes enabled: {enabled_timeframes}")

    if validation['warnings']:
        for warning in validation['warnings']:
            logger.warning(f"{module_name}: {warning}")

    if validation['errors']:
        for error in validation['errors']:
            logger.error(f"{module_name}: {error}")