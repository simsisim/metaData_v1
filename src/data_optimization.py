"""
Data Optimization Utilities
===========================

Performance optimization utilities for limiting historical data processing
based on user configuration settings.
"""

import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def cap_historical_data(batch_data: Dict, timeframe: str, cap_setting: int) -> Dict:
    """
    Limit historical data based on cap_history_data setting.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with market data
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        cap_setting: Capping mode (0=all, 1=yearly, 2=timeframe-specific)
        
    Returns:
        Dictionary with capped data
    """
    if cap_setting == 0:
        # No capping - return all data
        return batch_data
    
    # Determine data limits based on timeframe and setting
    if cap_setting == 1:
        # Yearly limit for all timeframes
        if timeframe == 'daily':
            limit = 252  # ~1 trading year
        elif timeframe == 'weekly':
            limit = 52   # ~1 year of weeks
        elif timeframe == 'monthly':
            limit = 12   # ~1 year of months
        else:
            limit = 252  # Default
    elif cap_setting == 2:
        # Timeframe-specific optimized limits
        if timeframe == 'daily':
            limit = 252  # 1 trading year for daily
        elif timeframe == 'weekly':
            limit = 104  # 2 years for weekly (more context needed)
        elif timeframe == 'monthly':
            limit = 36   # 3 years for monthly (more context needed)
        else:
            limit = 252  # Default
    else:
        # Unknown setting - no capping
        return batch_data
    
    # Apply capping to each ticker
    capped_data = {}
    original_total = sum(len(df) for df in batch_data.values())
    
    for ticker, df in batch_data.items():
        if df is not None and not df.empty:
            if len(df) > limit:
                # Take the most recent data
                capped_data[ticker] = df.tail(limit).copy()
            else:
                capped_data[ticker] = df.copy()
        else:
            capped_data[ticker] = df
    
    capped_total = sum(len(df) for df in capped_data.values() if df is not None)
    
    if cap_setting > 0:
        reduction_pct = ((original_total - capped_total) / original_total) * 100 if original_total > 0 else 0
        logger.info(f"Data capping applied: {original_total} -> {capped_total} rows ({reduction_pct:.1f}% reduction)")
        print(f"⚡ Data optimization: {reduction_pct:.1f}% reduction ({timeframe}, limit={limit})")
    
    return capped_data