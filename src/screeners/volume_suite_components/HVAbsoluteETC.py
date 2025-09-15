#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced HVAbsoluteETC.py - High Volume Absolute Analysis with Enhanced Filtering
=================================================================================

Enhanced maximum volume finder that uses comprehensive filtering parameters
for detecting significant volume events with statistical validation.

Based on: /home/imagda/_invest2024/python/volume_suite/src/HVAbsoluteETC.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def find_max_value_enhanced(data, params):
    """
    Enhanced maximum value finder that uses filtering parameters
    
    Args:
        data (pd.DataFrame): Stock data with Volume, Close columns
        params (dict): Parameters including filters and thresholds
    
    Returns:
        dict: {'Date': <date_of_max>, 'MaxValue': <max_value>, 'Filtered': <bool>}
    """
    if data.empty or 'Volume' not in data.columns:
        return {'Date': None, 'MaxValue': None, 'Filtered': False}
    
    # Extract parameters
    month_cutoff = params.get('month_cuttoff', 15)  # Only analyze last N months
    day_cutoff = params.get('day_cuttoff', 3)       # Only consider data older than N days  
    std_cutoff = params.get('std_cuttoff', 10)      # Standard deviation filter
    min_stock_volume = params.get('min_stock_volume', 100000)  # Minimum volume threshold
    min_price = params.get('min_price', 20)         # Minimum price threshold
    
    # Step 1: Apply date filtering
    current_date = datetime.now().date()
    
    # Filter by month cutoff (only last N months)
    date_threshold_months = current_date - timedelta(days=month_cutoff * 30)
    
    # Filter by day cutoff (exclude last N days)  
    date_threshold_days = current_date - timedelta(days=day_cutoff)
    
    # Apply date filters
    data_dates = pd.to_datetime(data.index).date
    date_mask = (data_dates >= date_threshold_months) & (data_dates <= date_threshold_days)
    filtered_data = data[date_mask].copy()
    
    if filtered_data.empty:
        return {'Date': None, 'MaxValue': None, 'Filtered': True, 'Reason': 'No data in date range'}
    
    # Step 2: Apply price filtering
    if 'Close' in filtered_data.columns:
        price_mask = filtered_data['Close'] >= min_price
        filtered_data = filtered_data[price_mask]
        
        if filtered_data.empty:
            return {'Date': None, 'MaxValue': None, 'Filtered': True, 'Reason': f'No stocks above ${min_price}'}
    
    # Step 3: Apply minimum volume filtering
    volume_mask = filtered_data['Volume'] >= min_stock_volume
    filtered_data = filtered_data[volume_mask]
    
    if filtered_data.empty:
        return {'Date': None, 'MaxValue': None, 'Filtered': True, 'Reason': f'No volume above {min_stock_volume:,}'}
    
    # Step 4: Apply statistical filtering (std_cutoff)
    if std_cutoff > 0:
        volume_mean = filtered_data['Volume'].mean()
        volume_std = filtered_data['Volume'].std()
        
        if volume_std > 0:  # Avoid division by zero
            # Only consider volumes that are std_cutoff standard deviations above mean
            statistical_threshold = volume_mean + (std_cutoff * volume_std)
            stats_mask = filtered_data['Volume'] >= statistical_threshold
            filtered_data = filtered_data[stats_mask]
            
            if filtered_data.empty:
                return {
                    'Date': None, 
                    'MaxValue': None, 
                    'Filtered': True, 
                    'Reason': f'No volume above {std_cutoff}σ threshold ({statistical_threshold:,.0f})',
                    'Statistical_Info': {
                        'volume_mean': volume_mean,
                        'volume_std': volume_std,
                        'threshold': statistical_threshold
                    }
                }
    
    # Step 5: Find maximum volume in filtered data
    idx_max = filtered_data['Volume'].idxmax()
    max_value = filtered_data.loc[idx_max, 'Volume']
    
    # Get the date
    date_of_max = (
        filtered_data.loc[idx_max, 'Date'] 
        if 'Date' in filtered_data.columns else idx_max
    )
    
    # Calculate additional statistics
    total_days_analyzed = len(filtered_data)
    original_days = len(data)
    filter_efficiency = (total_days_analyzed / original_days) * 100 if original_days > 0 else 0
    
    return {
        'Date': date_of_max, 
        'MaxValue': max_value,
        'Filtered': True,
        'Filter_Stats': {
            'original_days': original_days,
            'analyzed_days': total_days_analyzed,
            'filter_efficiency_pct': filter_efficiency,
            'filters_applied': {
                'date_range': f'{date_threshold_months} to {date_threshold_days}',
                'min_price': min_price,
                'min_volume': min_stock_volume,
                'std_threshold': f'{std_cutoff}σ' if std_cutoff > 0 else 'None'
            }
        }
    }


def run_HVAbsoluteStrategy_Enhanced(batch_data, params):
    """
    Enhanced HVAbsolute strategy that actually uses all the configured parameters
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary of parameters for filtering and analysis
        
    Returns:
        List of results with enhanced filtering statistics
    """
    results = []
    
    logger.info(f"HVAbsolute Enhanced - Using parameters:")
    logger.info(f"  Month cutoff: {params.get('month_cuttoff', 15)} months")
    logger.info(f"  Day cutoff: {params.get('day_cuttoff', 3)} days")
    logger.info(f"  Std cutoff: {params.get('std_cuttoff', 10)}σ")
    logger.info(f"  Min volume: {params.get('min_stock_volume', 100000):,}")
    logger.info(f"  Min price: ${params.get('min_price', 20)}")

    for symbol, data in batch_data.items():
        try:
            # Use enhanced max value finder
            analysis_result = find_max_value_enhanced(data, params)
            
            if analysis_result['Date'] is not None:
                result_row = {
                    'Ticker': symbol,
                    'Date': analysis_result['Date'],
                    'MaxVolume': analysis_result['MaxValue']
                }
                
                # Add filter statistics if available
                if 'Filter_Stats' in analysis_result:
                    stats = analysis_result['Filter_Stats']
                    result_row.update({
                        'Original_Days': stats['original_days'],
                        'Analyzed_Days': stats['analyzed_days'],
                        'Filter_Efficiency_Pct': round(stats['filter_efficiency_pct'], 1),
                        'Date_Range': stats['filters_applied']['date_range'],
                        'Min_Price_Filter': stats['filters_applied']['min_price'],
                        'Min_Volume_Filter': stats['filters_applied']['min_volume'],
                        'Std_Filter': stats['filters_applied']['std_threshold']
                    })
                
                # Add statistical info if available
                if 'Statistical_Info' in analysis_result:
                    stat_info = analysis_result['Statistical_Info']
                    result_row.update({
                        'Volume_Mean': round(stat_info['volume_mean'], 0),
                        'Volume_Std': round(stat_info['volume_std'], 0),
                        'Statistical_Threshold': round(stat_info['threshold'], 0),
                        'Std_Deviations_Above_Mean': round((analysis_result['MaxValue'] - stat_info['volume_mean']) / stat_info['volume_std'], 2) if stat_info['volume_std'] > 0 else 0
                    })
                
                # Add current price if available
                try:
                    if analysis_result['Date'] in data.index and 'Close' in data.columns:
                        result_row['Close'] = data.loc[analysis_result['Date'], 'Close']
                except:
                    pass
                
                results.append(result_row)
            else:
                # Track filtered out stocks
                results.append({
                    'Ticker': symbol,
                    'Date': None,
                    'MaxVolume': None,
                    'Filter_Reason': analysis_result.get('Reason', 'Unknown'),
                    'Filtered_Out': True
                })
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            results.append({
                'Ticker': symbol,
                'Date': None,
                'MaxVolume': None,
                'Error': str(e)
            })

    return results


# Backward compatibility - keep original function name
def run_HVAbsoluteStrategy(batch_data, params):
    """
    Wrapper to maintain compatibility - choose enhanced or simple version
    """
    use_enhanced = params.get('use_enhanced_filtering', True)
    
    if use_enhanced:
        return run_HVAbsoluteStrategy_Enhanced(batch_data, params)
    else:
        # Original simple implementation
        results = []
        for symbol, data in batch_data.items():
            if 'Volume' not in data.columns or data.empty:
                continue
                
            idx_max = data['Volume'].idxmax()
            max_value = data.loc[idx_max, 'Volume']
            date_of_max = data.loc[idx_max, 'Date'] if 'Date' in data.columns else idx_max
            
            results.append({
                'Ticker': symbol,
                'Date': date_of_max,
                'MaxVolume': max_value
            })
        return results


# Export main functions
__all__ = ['find_max_value_enhanced', 'run_HVAbsoluteStrategy_Enhanced', 'run_HVAbsoluteStrategy']