#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVStdv.py - High Volume Statistical Deviation Analysis
======================================================

Detects dates with unusually high volume using statistical deviation methods.
Identifies volume anomalies by analyzing how many standard deviations above
the mean a volume spike is.

Based on: /home/imagda/_invest2024/python/volume_suite/src/HVStdv.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def find_anomalies(data, std_cutoff=12, min_stock_volume=10000):
    """
    Detects dates with unusually high volume using statistical analysis.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Volume' column.
        std_cutoff (float): Number of std devs above mean to be considered anomaly.
        min_stock_volume (int): Minimum volume threshold.
    
    Returns:
        dict: {'Dates': [...], 'Volume': [...], 'Statistics': {...}}
    """
    if 'Volume' not in data.columns or data.empty:
        return {'Dates': [], 'Volume': [], 'Statistics': {}}

    # Calculate volume statistics
    data_std = np.std(data['Volume'])
    data_mean = np.mean(data['Volume'])
    anomaly_cutoff = data_std * std_cutoff
    upper_limit = data_mean + anomaly_cutoff

    # Reset index to ensure 'Date' is accessible
    data = data.reset_index()
    result_dates = []
    result_volumes = []
    result_statistics = []

    for i in range(len(data)):
        volume = data['Volume'].iloc[i]
        if volume > upper_limit and volume > min_stock_volume:
            # Try to get date from 'Date', falling back to index if needed
            date_val = data['Date'].iloc[i] if 'Date' in data.columns else data.iloc[i, 0]
            result_dates.append(date_val)
            result_volumes.append(volume)
            
            # Calculate statistical metrics for this anomaly
            z_score = (volume - data_mean) / data_std if data_std > 0 else 0
            percentile = (np.sum(data['Volume'] <= volume) / len(data)) * 100
            
            result_statistics.append({
                'z_score': z_score,
                'percentile': percentile,
                'volume_vs_mean_ratio': volume / data_mean if data_mean > 0 else 0,
                'volume_vs_threshold_ratio': volume / upper_limit if upper_limit > 0 else 0
            })

    # Calculate overall statistics for the dataset
    overall_stats = {
        'volume_mean': data_mean,
        'volume_std': data_std,
        'upper_threshold': upper_limit,
        'std_multiplier': std_cutoff,
        'total_observations': len(data),
        'anomalies_found': len(result_dates),
        'anomaly_rate_pct': (len(result_dates) / len(data)) * 100 if len(data) > 0 else 0
    }

    return {
        'Dates': result_dates, 
        'Volume': result_volumes,
        'Individual_Statistics': result_statistics,
        'Overall_Statistics': overall_stats
    }


def run_HVStdvStrategy(batch_data, params):
    """
    For each stock in batch_data, finds dates with unusually high volume,
    using enhanced statistical anomaly detection.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary of parameters for analysis
        
    Returns:
        List of results with statistical metrics
    """
    results = []
    std_cutoff = params.get('std_cuttoff', 12)
    min_stock_volume = params.get('min_stock_volume', 10000)
    
    logger.info(f"HVStdv Analysis - Using parameters:")
    logger.info(f"  Standard deviation cutoff: {std_cutoff}σ")
    logger.info(f"  Minimum volume threshold: {min_stock_volume:,}")

    for symbol, data in batch_data.items():
        try:
            # Run anomaly detection
            anomalies = find_anomalies(data, std_cutoff, min_stock_volume)
            
            # Process each anomaly found
            for i, (dt, vol) in enumerate(zip(anomalies['Dates'], anomalies['Volume'])):
                result_row = {
                    'Ticker': symbol,
                    'Date': dt,
                    'UnusualVolume': vol
                }
                
                # Add individual statistics if available
                if i < len(anomalies.get('Individual_Statistics', [])):
                    ind_stats = anomalies['Individual_Statistics'][i]
                    result_row.update({
                        'Z_Score': round(ind_stats['z_score'], 2),
                        'Volume_Percentile': round(ind_stats['percentile'], 1),
                        'Volume_vs_Mean_Ratio': round(ind_stats['volume_vs_mean_ratio'], 2),
                        'Volume_vs_Threshold_Ratio': round(ind_stats['volume_vs_threshold_ratio'], 2)
                    })
                
                # Add overall statistics
                if 'Overall_Statistics' in anomalies:
                    overall = anomalies['Overall_Statistics']
                    result_row.update({
                        'Dataset_Volume_Mean': round(overall['volume_mean'], 0),
                        'Dataset_Volume_Std': round(overall['volume_std'], 0),
                        'Statistical_Threshold': round(overall['upper_threshold'], 0),
                        'Total_Anomalies_Found': overall['anomalies_found'],
                        'Anomaly_Rate_Pct': round(overall['anomaly_rate_pct'], 2)
                    })
                
                # Add current price if available
                try:
                    if dt in data.index and 'Close' in data.columns:
                        result_row['Close'] = data.loc[dt, 'Close']
                except:
                    pass
                
                # Classify anomaly strength
                z_score = result_row.get('Z_Score', 0)
                if z_score >= 20:
                    strength = 'Extreme'
                elif z_score >= 15:
                    strength = 'Very Strong'
                elif z_score >= 12:
                    strength = 'Strong'
                elif z_score >= 8:
                    strength = 'Moderate'
                else:
                    strength = 'Weak'
                
                result_row['Anomaly_Strength'] = strength
                results.append(result_row)
            
            # If no anomalies found, add a summary entry
            if not anomalies['Dates'] and 'Overall_Statistics' in anomalies:
                overall = anomalies['Overall_Statistics']
                results.append({
                    'Ticker': symbol,
                    'Date': None,
                    'UnusualVolume': None,
                    'No_Anomalies_Found': True,
                    'Dataset_Volume_Mean': round(overall['volume_mean'], 0),
                    'Dataset_Volume_Std': round(overall['volume_std'], 0),
                    'Statistical_Threshold': round(overall['upper_threshold'], 0),
                    'Reason': f'No volume above {std_cutoff}σ threshold'
                })
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            results.append({
                'Ticker': symbol,
                'Date': None,
                'UnusualVolume': None,
                'Error': str(e)
            })

    # Summary statistics
    valid_results = [r for r in results if r.get('Date') is not None]
    logger.info(f"HVStdv completed: {len(valid_results)} anomalies found across {len(batch_data)} stocks")
    
    return results


# Export main functions
__all__ = ['find_anomalies', 'run_HVStdvStrategy']