#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume Indicators - VROC, RVOL, ADTV, MFI, VPT Calculations
============================================================

Comprehensive volume indicator calculations including:
- VROC: Volume Rate of Change
- RVOL: Relative Volume  
- ADTV: Average Daily Trading Volume analysis
- MFI: Money Flow Index
- VPT: Volume Price Trend

Based on: /home/imagda/_invest2024/python/volume_suite/src/volume_indicators_*.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ====================================================================
# VROC, RVOL, ADTV CALCULATIONS (Part 1)
# ====================================================================

def add_vroc(df, period=25):
    """Add Volume Rate of Change"""
    df['VROC'] = ((df['Volume'] - df['Volume'].shift(period)) / df['Volume'].shift(period)) * 100
    return df


def add_relative_volume(df, period=20):
    """Add Relative Volume (RVOL) - Current volume vs average volume"""
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(window=period).mean()
    return df


def add_adtv_trend_analysis(df, ma_period=50, lookback_3m=63, lookback_6m=126, lookback_1y=252):
    """
    Add ADTV Trend Analysis indicators
    - Calculate rolling ADTV (Average Daily Trading Volume)
    - Compare current ADTV to historical periods
    """
    # Calculate ADTV using rolling mean
    df['ADTV'] = df['Volume'].rolling(window=ma_period).mean()
    
    # Calculate historical ADTV ratios
    df['ADTV_3M_Ratio'] = df['ADTV'] / df['ADTV'].shift(lookback_3m)
    df['ADTV_6M_Ratio'] = df['ADTV'] / df['ADTV'].shift(lookback_6m)  
    df['ADTV_1Y_Ratio'] = df['ADTV'] / df['ADTV'].shift(lookback_1y)
    
    return df


def detect_vroc_anomalies(df, ticker, vroc_threshold=50, lookback_period=20):
    """Detect VROC (Volume Rate of Change) anomalies"""
    anomalies = []
    
    if 'VROC' not in df.columns:
        return anomalies
    
    # Calculate VROC statistics
    vroc_mean = df['VROC'].rolling(window=lookback_period).mean()
    vroc_std = df['VROC'].rolling(window=lookback_period).std()
    
    # Detect EXTREME VROC values
    extreme_vroc = (df['VROC'].abs() > vroc_threshold)
    
    vroc_dates = df.index[extreme_vroc & df['VROC'].notna()]
    
    for date in vroc_dates:
        try:
            vroc_value = df.loc[date, 'VROC']
            if pd.notna(vroc_value):
                anomalies.append({
                    'Ticker': ticker,
                    'Date': date,
                    'Close': df.loc[date, 'Close'],
                    'Volume': df.loc[date, 'Volume'],
                    'VROC': vroc_value,
                    'VROC_Mean_20d': vroc_mean.loc[date] if pd.notna(vroc_mean.loc[date]) else 0,
                    'VROC_Std_20d': vroc_std.loc[date] if pd.notna(vroc_std.loc[date]) else 0,
                    'VROC_Zscore': (vroc_value - vroc_mean.loc[date]) / vroc_std.loc[date] if pd.notna(vroc_mean.loc[date]) and pd.notna(vroc_std.loc[date]) and vroc_std.loc[date] != 0 else 0,
                    'VROC_Extreme_Type': 'Positive' if vroc_value > 0 else 'Negative',
                    'Signal_Strength': 'Strong' if abs(vroc_value) > vroc_threshold else 'Moderate'
                })
        except Exception as e:
            continue
    
    return anomalies


def detect_rvol_anomalies(df, ticker, rvol_threshold=2.0, extreme_threshold=5.0):
    """Detect Relative Volume (RVOL) anomalies"""
    anomalies = []
    
    if 'RVOL' not in df.columns:
        return anomalies
    
    # Detect EXTREME relative volume
    high_rvol = df['RVOL'] > rvol_threshold
    extreme_rvol = df['RVOL'] > extreme_threshold
    # Add minimum volume filter for true anomalies (1M+ shares)
    min_volume_filter = df['Volume'] > 1000000
    
    rvol_dates = df.index[high_rvol & df['RVOL'].notna() & min_volume_filter]
    
    for date in rvol_dates:
        try:
            rvol_value = df.loc[date, 'RVOL']
            if pd.notna(rvol_value):
                anomalies.append({
                    'Ticker': ticker,
                    'Date': date,
                    'Close': df.loc[date, 'Close'],
                    'Volume': df.loc[date, 'Volume'],
                    'RVOL': rvol_value,
                    'Volume_vs_Average': f"{rvol_value:.2f}x",
                    'Signal_Strength': 'Extreme' if rvol_value > extreme_threshold else 'Strong',
                    'Volume_Category': 'Exceptional' if rvol_value > 10 else ('Very High' if rvol_value > 5 else ('High' if rvol_value > 2 else 'Moderate')),
                    'Average_Volume_20d': df.loc[date, 'Volume'] / rvol_value if rvol_value != 0 else 0
                })
        except Exception as e:
            continue
    
    return anomalies


def detect_adtv_trend_anomalies(df, ticker, threshold_3m=2.0, threshold_6m=2.0, threshold_1y=2.0, min_volume=1000000, signal_start_date='2025-01-01'):
    """
    Detect ADTV trend anomalies - significant increases in average volume
    Only signals from signal_start_date onwards
    """
    anomalies = []
    
    required_columns = ['ADTV', 'ADTV_3M_Ratio', 'ADTV_6M_Ratio', 'ADTV_1Y_Ratio']
    if not all(col in df.columns for col in required_columns):
        return anomalies
    
    # Convert signal_start_date to pandas timestamp
    try:
        start_date = pd.to_datetime(signal_start_date)
    except:
        logger.warning(f"Invalid signal_start_date '{signal_start_date}', using 2025-01-01")
        start_date = pd.to_datetime('2025-01-01')
    
    # Filter for significant volume, signal start date, and recent dates
    volume_filter = df['Volume'] > min_volume
    date_filter = df.index >= start_date
    
    # Detect anomalies for each timeframe
    anomaly_3m = (df['ADTV_3M_Ratio'] >= threshold_3m) & volume_filter & date_filter
    anomaly_6m = (df['ADTV_6M_Ratio'] >= threshold_6m) & volume_filter & date_filter
    anomaly_1y = (df['ADTV_1Y_Ratio'] >= threshold_1y) & volume_filter & date_filter
    
    # Combine all anomalies
    any_anomaly = anomaly_3m | anomaly_6m | anomaly_1y
    signal_dates = df.index[any_anomaly & df['ADTV'].notna()]
    
    for date in signal_dates:
        try:
            adtv_3m_ratio = df.loc[date, 'ADTV_3M_Ratio']
            adtv_6m_ratio = df.loc[date, 'ADTV_6M_Ratio'] 
            adtv_1y_ratio = df.loc[date, 'ADTV_1Y_Ratio']
            
            # Handle NaN values safely
            if pd.isna(adtv_3m_ratio):
                adtv_3m_ratio = 0
            if pd.isna(adtv_6m_ratio):
                adtv_6m_ratio = 0
            if pd.isna(adtv_1y_ratio):
                adtv_1y_ratio = 0
            
            # Determine strongest signal
            max_ratio = max(adtv_3m_ratio, adtv_6m_ratio, adtv_1y_ratio)
            
            if max_ratio < min(threshold_3m, threshold_6m, threshold_1y):
                continue  # Skip if no threshold is met
            
            if adtv_3m_ratio == max_ratio and adtv_3m_ratio >= threshold_3m:
                timeframe = '3M'
                ratio_value = adtv_3m_ratio
            elif adtv_6m_ratio == max_ratio and adtv_6m_ratio >= threshold_6m:
                timeframe = '6M' 
                ratio_value = adtv_6m_ratio
            elif adtv_1y_ratio >= threshold_1y:
                timeframe = '1Y'
                ratio_value = adtv_1y_ratio
            else:
                continue  # No valid signal
                
            anomalies.append({
                'Ticker': ticker,
                'Date': date,
                'Close': df.loc[date, 'Close'],
                'Volume': df.loc[date, 'Volume'],
                'ADTV_Current': df.loc[date, 'ADTV'],
                'Timeframe': timeframe,
                'ADTV_Ratio': ratio_value,
                'ADTV_3M_Ratio': adtv_3m_ratio if not pd.isna(adtv_3m_ratio) else 0,
                'ADTV_6M_Ratio': adtv_6m_ratio if not pd.isna(adtv_6m_ratio) else 0,
                'ADTV_1Y_Ratio': adtv_1y_ratio if not pd.isna(adtv_1y_ratio) else 0,
                'Signal_Strength': 'Extreme' if max_ratio >= 5.0 else ('Strong' if max_ratio >= 3.0 else 'Moderate'),
                'Trend_Category': 'Massive Increase' if max_ratio >= 5.0 else ('Major Increase' if max_ratio >= 3.0 else 'Significant Increase')
            })
        except Exception as e:
            continue
    
    return anomalies


# ====================================================================
# MFI AND VPT CALCULATIONS (Part 2)
# ====================================================================

def add_money_flow_index(df, period=14):
    """Add Money Flow Index (MFI)"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    # Calculate positive and negative money flow
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    
    # Calculate money ratio
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    money_ratio = positive_sum / negative_sum
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    return df


def add_volume_price_trend(df):
    """Add Volume Price Trend (VPT)"""
    # VPT = Previous VPT + Volume * (Close - Previous Close) / Previous Close
    price_change_ratio = df['Close'].pct_change()
    df['VPT'] = (df['Volume'] * price_change_ratio).cumsum()
    return df


def detect_mfi_signals(df, ticker, overbought_threshold=80, oversold_threshold=20):
    """Detect Money Flow Index (MFI) overbought/oversold signals"""
    signals = []
    
    if 'MFI' not in df.columns:
        return signals
    
    # Detect MFI extreme levels
    overbought = df['MFI'] > overbought_threshold
    oversold = df['MFI'] < oversold_threshold
    
    # Find MFI signal dates
    mfi_signal_dates = df.index[(overbought | oversold) & df['MFI'].notna()]
    
    for date in mfi_signal_dates:
        try:
            mfi_value = df.loc[date, 'MFI']
            if pd.notna(mfi_value):
                signal_type = 'Overbought' if mfi_value > overbought_threshold else 'Oversold'
                
                signals.append({
                    'Ticker': ticker,
                    'Date': date,
                    'Close': df.loc[date, 'Close'],
                    'Volume': df.loc[date, 'Volume'],
                    'MFI': mfi_value,
                    'Signal_Type': signal_type,
                    'Signal_Strength': 'Extreme' if (mfi_value > 90 or mfi_value < 10) else 'Strong',
                    'Typical_Price': (df.loc[date, 'High'] + df.loc[date, 'Low'] + df.loc[date, 'Close']) / 3,
                    'Money_Flow': ((df.loc[date, 'High'] + df.loc[date, 'Low'] + df.loc[date, 'Close']) / 3) * df.loc[date, 'Volume']
                })
        except Exception as e:
            continue
    
    return signals


def detect_vpt_signals(df, ticker, signal_threshold=0.05, vpt_ma_period=20):
    """Detect Volume Price Trend (VPT) signals"""
    signals = []
    
    if 'VPT' not in df.columns:
        return signals
    
    # Calculate VPT momentum (timeframe-aware)
    vpt_change = df['VPT'].pct_change()
    vpt_ma = df['VPT'].rolling(window=vpt_ma_period).mean()
    
    # Detect significant VPT changes
    significant_change = vpt_change.abs() > signal_threshold
    
    signal_dates = df.index[significant_change & df['VPT'].notna()]
    
    for date in signal_dates:
        try:
            vpt_value = df.loc[date, 'VPT']
            vpt_change_value = vpt_change.loc[date]
            if pd.notna(vpt_value) and pd.notna(vpt_change_value):
                signals.append({
                    'Ticker': ticker,
                    'Date': date,
                    'Close': df.loc[date, 'Close'],
                    'Volume': df.loc[date, 'Volume'],
                    'VPT': vpt_value,
                    'VPT_Change': vpt_change_value,
                    'VPT_MA_20d': vpt_ma.loc[date] if pd.notna(vpt_ma.loc[date]) else 0,
                    'Signal_Direction': 'Positive' if vpt_change_value > 0 else 'Negative',
                    'Signal_Strength': 'Strong' if abs(vpt_change_value) > (signal_threshold * 2) else 'Moderate',
                    'Momentum_Category': 'Extreme' if abs(vpt_change_value) > 0.5 else ('High' if abs(vpt_change_value) > 0.2 else 'Moderate')
                })
        except Exception as e:
            continue
    
    return signals


# ====================================================================
# UNIFIED CALCULATION FUNCTIONS
# ====================================================================

def calculate_all_volume_indicators(df, vroc_period=25, rvol_period=20, mfi_period=14, 
                                  adtv_ma_period=50, adtv_3m=63, adtv_6m=126, adtv_1y=252):
    """
    Calculate all volume indicators for a single ticker
    
    Args:
        df: DataFrame with OHLCV data
        Various period parameters for each indicator
        
    Returns:
        DataFrame with all volume indicators added
    """
    try:
        # Part 1: VROC, RVOL, ADTV
        df = add_vroc(df, vroc_period)
        df = add_relative_volume(df, rvol_period)
        df = add_adtv_trend_analysis(df, adtv_ma_period, adtv_3m, adtv_6m, adtv_1y)
        
        # Part 2: MFI, VPT
        df = add_money_flow_index(df, mfi_period)
        df = add_volume_price_trend(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating volume indicators: {e}")
        return df


def run_volume_indicators_analysis(batch_data, params):
    """
    Run comprehensive volume indicators analysis on batch data
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary of parameters for analysis
        
    Returns:
        Dictionary containing processed data and detected signals
    """
    # Extract parameters (timeframe-scaled)
    vroc_period = params.get('vroc_period', 25)
    rvol_period = params.get('rvol_period', 20)
    mfi_period = params.get('mfi_period', 14)
    vpt_ma_period = params.get('vpt_ma_period', 20)
    adtv_ma_period = params.get('adtv_ma_period', 50)
    
    # Detection thresholds
    vroc_threshold = params.get('vroc_threshold', 50)
    rvol_threshold = params.get('rvol_threshold', 2.0)
    rvol_extreme_threshold = params.get('rvol_extreme_threshold', 5.0)
    mfi_overbought = params.get('mfi_overbought_threshold', 80)
    mfi_oversold = params.get('mfi_oversold_threshold', 20)
    vpt_signal_threshold = params.get('vpt_signal_threshold', 0.05)
    
    # ADTV parameters
    adtv_3m_threshold = params.get('adtv_3m_threshold', 2.0)
    adtv_6m_threshold = params.get('adtv_6m_threshold', 2.0)
    adtv_1y_threshold = params.get('adtv_1y_threshold', 2.0)
    adtv_min_volume = params.get('adtv_min_volume', 1000000)
    
    logger.info(f"Running volume indicators analysis on {len(batch_data)} tickers")
    
    processed_data = {}
    all_signals = {
        'vroc_anomalies': [],
        'rvol_anomalies': [],
        'adtv_trend_anomalies': [],
        'mfi_signals': [],
        'vpt_signals': []
    }
    
    for symbol, df in batch_data.items():
        try:
            # Calculate all volume indicators
            enhanced_df = calculate_all_volume_indicators(
                df.copy(),
                vroc_period, rvol_period, mfi_period, adtv_ma_period
            )
            processed_data[symbol] = enhanced_df
            
            # Detect signals for each indicator
            
            # 1. VROC Anomalies
            vroc_signals = detect_vroc_anomalies(enhanced_df, symbol, vroc_threshold)
            all_signals['vroc_anomalies'].extend(vroc_signals)
            
            # 2. RVOL Anomalies
            rvol_signals = detect_rvol_anomalies(enhanced_df, symbol, rvol_threshold, rvol_extreme_threshold)
            all_signals['rvol_anomalies'].extend(rvol_signals)
            
            # 3. ADTV Trend Anomalies
            adtv_signals = detect_adtv_trend_anomalies(
                enhanced_df, symbol, adtv_3m_threshold, adtv_6m_threshold, 
                adtv_1y_threshold, adtv_min_volume
            )
            all_signals['adtv_trend_anomalies'].extend(adtv_signals)
            
            # 4. MFI Signals
            mfi_signals = detect_mfi_signals(enhanced_df, symbol, mfi_overbought, mfi_oversold)
            all_signals['mfi_signals'].extend(mfi_signals)
            
            # 5. VPT Signals (timeframe-aware)
            vpt_signals = detect_vpt_signals(enhanced_df, symbol, vpt_signal_threshold, vpt_ma_period)
            all_signals['vpt_signals'].extend(vpt_signals)
            
        except Exception as e:
            logger.error(f"Error processing volume indicators for {symbol}: {e}")
            processed_data[symbol] = df  # Save original data if processing fails
            continue
    
    # Summary statistics
    total_signals = sum(len(signals) for signals in all_signals.values())
    logger.info(f"Volume indicators analysis completed: {total_signals} total signals across {len(processed_data)} stocks")
    
    return {
        'processed_data': processed_data,
        'signals': all_signals,
        'summary': {
            'total_signals': total_signals,
            'vroc_signals': len(all_signals['vroc_anomalies']),
            'rvol_signals': len(all_signals['rvol_anomalies']),
            'adtv_signals': len(all_signals['adtv_trend_anomalies']),
            'mfi_signals': len(all_signals['mfi_signals']),
            'vpt_signals': len(all_signals['vpt_signals'])
        }
    }


# Export main functions
__all__ = [
    'add_vroc', 'add_relative_volume', 'add_adtv_trend_analysis',
    'add_money_flow_index', 'add_volume_price_trend',
    'detect_vroc_anomalies', 'detect_rvol_anomalies', 'detect_adtv_trend_anomalies',
    'detect_mfi_signals', 'detect_vpt_signals',
    'calculate_all_volume_indicators', 'run_volume_indicators_analysis'
]