#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Volume Anomaly Detection
=================================

Multi-method volume anomaly detection system with statistical validation,
benchmark comparisons, and comprehensive signal analysis.

Based on: /home/imagda/_invest2024/python/volume_suite/src/enhanced_volume_anomaly.py
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ParametersConfig:
    """
    Centralized parameter management system for volume anomaly detection.
    Parameters can be loaded from CSV file or defined programmatically.
    """
    
    def __init__(self, config_file='volume_parameters.csv'):
        self.config_file = config_file
        self.parameters = self.load_parameters()
    
    def load_parameters(self):
        """Load parameters from CSV file or create defaults"""
        if os.path.exists(self.config_file):
            try:
                logger.info(f"Loading parameters from {self.config_file}")
                df = pd.read_csv(self.config_file, quotechar='"', escapechar='\\')
                
                # Check if CSV has the right structure
                required_columns = ['category', 'parameter', 'value', 'type', 'description']
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"{self.config_file} has wrong format. Recreating...")
                    return self.create_default_parameters()
                
                # Convert DataFrame to dict structure
                params = {}
                for _, row in df.iterrows():
                    try:
                        category = row['category']
                        if category not in params:
                            params[category] = {}
                        params[category][row['parameter']] = self._convert_value(row['value'], row['type'])
                    except Exception as e:
                        logger.warning(f"Error parsing parameter {row.get('parameter', 'unknown')}: {e}")
                        continue
                
                return params
            
            except Exception as e:
                logger.error(f"Error reading {self.config_file}: {e}")
                return self.create_default_parameters()
        else:
            logger.info(f"Creating default parameters file: {self.config_file}")
            return self.create_default_parameters()
    
    def _convert_value(self, value, value_type):
        """Convert string values to appropriate types"""
        try:
            if value_type == 'int':
                return int(float(value))
            elif value_type == 'float':
                return float(value)
            elif value_type == 'bool':
                return str(value).lower() in ['true', '1', 'yes']
            elif value_type == 'str':
                return str(value)
            elif value_type == 'list':
                if isinstance(value, str):
                    value = value.strip()
                    if value.startswith('[') and value.endswith(']'):
                        try:
                            return eval(value)
                        except:
                            content = value[1:-1]
                            items = [item.strip().strip('"\'') for item in content.split(',')]
                            return [item for item in items if item]
                    else:
                        return [value]
                else:
                    return value if isinstance(value, list) else [value]
            return value
        except Exception as e:
            logger.warning(f"Could not convert value '{value}' of type '{value_type}': {e}")
            return value
    
    def create_default_parameters(self):
        """Create default parameter configuration"""
        default_params = {
            'volume_anomaly': {
                'lookback_period': 50,
                'std_deviation_threshold': 3.0,
                'min_volume_threshold': 100000,
                'min_relative_volume': 1.5,
                'price_change_threshold': 0.05
            },
            'price_volume_breakout': {
                'price_breakout_period': 30,
                'volume_breakout_period': 30,
                'trendline_length': 50,
                'volume_multiplier': 1.5,
                'price_change_min': 0.02
            },
            'absolute_volume': {
                'ranking_period': 252,
                'top_percentile': 0.95,
                'min_dollar_volume': 1000000
            },
            'statistical_anomaly': {
                'ema_period': 20,
                'bollinger_period': 20,
                'bollinger_std': 2.0,
                'rsi_period': 14,
                'mfi_period': 14
            }
        }
        return default_params
    
    def get_params(self, category):
        """Get parameters for specific category"""
        return self.parameters.get(category, {})


class VolumeAnomalyDetector:
    """Enhanced volume anomaly detection with intuitive standard deviation thresholds"""
    
    def __init__(self, params):
        self.params = params
        self.detection_methods = {
            'statistical': self._detect_statistical_anomalies,
            'breakout': self._detect_breakout_anomalies,
            'absolute': self._detect_absolute_volume_anomalies,
            'pattern': self._detect_pattern_anomalies
        }
    
    def detect_anomalies(self, data, ticker, method='all'):
        """Detect volume anomalies using specified method(s)"""
        if data.empty or 'Volume' not in data.columns:
            return pd.DataFrame()
        
        anomalies = []
        
        if method == 'all':
            methods_to_use = self.detection_methods.keys()
        else:
            methods_to_use = [method] if method in self.detection_methods else []
        
        for method_name in methods_to_use:
            try:
                method_anomalies = self.detection_methods[method_name](data, ticker)
                for anomaly in method_anomalies:
                    anomaly['detection_method'] = method_name
                    anomalies.append(anomaly)
            except Exception as e:
                logger.warning(f"Error in {method_name} detection for {ticker}: {e}")
                continue
        
        return pd.DataFrame(anomalies)
    
    def _detect_statistical_anomalies(self, data, ticker):
        """Detect anomalies using standard deviation multipliers"""
        anomalies = []
        lookback = self.params.get('lookback_period', 50)
        std_threshold = self.params.get('std_deviation_threshold', 3.0)
        min_volume = self.params.get('min_volume_threshold', 100000)
        min_relative = self.params.get('min_relative_volume', 1.5)
        
        # Calculate rolling statistics
        volume_mean = data['Volume'].rolling(window=lookback).mean()
        volume_std = data['Volume'].rolling(window=lookback).std()
        
        # Calculate threshold using standard deviation multiplier
        volume_threshold = volume_mean + (std_threshold * volume_std)
        
        # Calculate additional metrics
        volume_zscore = (data['Volume'] - volume_mean) / volume_std
        volume_ratio = data['Volume'] / volume_mean
        
        # Apply detection criteria
        detection_conditions = (
            (data['Volume'] > volume_threshold) &
            (data['Volume'] > min_volume) &
            (volume_ratio > min_relative)
        )
        
        anomaly_dates = data.index[detection_conditions]
        
        for date in anomaly_dates:
            try:
                if (pd.notna(volume_threshold.loc[date]) and 
                    pd.notna(volume_zscore.loc[date]) and
                    pd.notna(volume_ratio.loc[date])):
                    
                    # Calculate volume excess
                    volume_excess = data.loc[date, 'Volume'] - volume_threshold.loc[date]
                    volume_excess_pct = (volume_excess / volume_threshold.loc[date]) * 100
                    
                    anomalies.append({
                        'ticker': ticker,
                        'signal_date': date,
                        'signal_type': f'Volume {std_threshold}σ Spike',
                        'volume': data.loc[date, 'Volume'],
                        'price': data.loc[date, 'Close'],
                        'volume_mean_50d': volume_mean.loc[date],
                        'volume_std_50d': volume_std.loc[date],
                        'volume_threshold': volume_threshold.loc[date],
                        'std_deviations_above_mean': volume_zscore.loc[date],
                        'volume_vs_avg_ratio': volume_ratio.loc[date],
                        'volume_excess_shares': volume_excess,
                        'volume_excess_pct': volume_excess_pct,
                        'threshold_confidence': self._get_confidence_level(std_threshold),
                        'anomaly_strength': self._classify_anomaly_strength(volume_zscore.loc[date])
                    })
            except Exception as e:
                logger.warning(f"Error processing statistical anomaly for {ticker} on {date}: {e}")
                continue
        
        return anomalies
    
    def _detect_breakout_anomalies(self, data, ticker):
        """Detect volume anomalies coinciding with price breakouts"""
        anomalies = []
        price_period = self.params.get('price_breakout_period', 30)
        volume_period = self.params.get('volume_breakout_period', 30)
        volume_multiplier = self.params.get('volume_multiplier', 1.5)
        
        # Calculate breakout levels
        price_high = data['High'].rolling(window=price_period).max()
        volume_avg = data['Volume'].rolling(window=volume_period).mean()
        
        # Detect breakouts
        price_breakout = data['Close'] > price_high.shift(1)
        volume_breakout = data['Volume'] > (volume_avg.shift(1) * volume_multiplier)
        combined_breakout = price_breakout & volume_breakout
        
        breakout_dates = data.index[combined_breakout]
        
        for date in breakout_dates:
            if pd.notna(volume_avg.loc[date]):
                anomalies.append({
                    'ticker': ticker,
                    'signal_date': date,
                    'signal_type': 'Price-Volume Breakout',
                    'volume': data.loc[date, 'Volume'],
                    'price': data.loc[date, 'Close'],
                    'volume_vs_avg': data.loc[date, 'Volume'] / volume_avg.loc[date],
                    'price_vs_high': (data.loc[date, 'Close'] / price_high.shift(1).loc[date]) - 1
                })
        
        return anomalies
    
    def _detect_absolute_volume_anomalies(self, data, ticker):
        """Detect absolute volume anomalies (highest volume days)"""
        anomalies = []
        ranking_period = self.params.get('ranking_period', 252)
        top_percentile = self.params.get('top_percentile', 0.95)
        
        # Calculate volume percentiles
        volume_percentile = data['Volume'].rolling(window=ranking_period).quantile(top_percentile)
        
        # Detect high volume days
        high_volume_days = data.index[data['Volume'] > volume_percentile]
        
        for date in high_volume_days:
            if pd.notna(volume_percentile.loc[date]):
                anomalies.append({
                    'ticker': ticker,
                    'signal_date': date,
                    'signal_type': 'Absolute Volume Spike',
                    'volume': data.loc[date, 'Volume'],
                    'price': data.loc[date, 'Close'],
                    'volume_percentile': (data.loc[date, 'Volume'] / volume_percentile.loc[date]) * 100
                })
        
        return anomalies
    
    def _detect_pattern_anomalies(self, data, ticker):
        """Detect pattern-based anomalies (unusual volume patterns)"""
        anomalies = []
        
        # Volume surge after quiet period
        volume_ma_short = data['Volume'].rolling(window=5).mean()
        volume_ma_long = data['Volume'].rolling(window=20).mean()
        
        # Detect volume surge patterns
        volume_surge = (volume_ma_short > volume_ma_long * 2) & (data['Volume'] > volume_ma_long * 3)
        surge_dates = data.index[volume_surge]
        
        for date in surge_dates:
            anomalies.append({
                'ticker': ticker,
                'signal_date': date,
                'signal_type': 'Volume Surge Pattern',
                'volume': data.loc[date, 'Volume'],
                'price': data.loc[date, 'Close'],
                'volume_surge_ratio': volume_ma_short.loc[date] / volume_ma_long.loc[date]
            })
        
        return anomalies
    
    def _get_confidence_level(self, std_threshold):
        """Convert standard deviation threshold to confidence level"""
        confidence_map = {
            1.0: "68.3%",   # 1 sigma
            1.5: "86.6%",   # 1.5 sigma  
            2.0: "95.4%",   # 2 sigma
            2.5: "98.8%",   # 2.5 sigma
            3.0: "99.7%",   # 3 sigma
            3.5: "99.95%",  # 3.5 sigma
            4.0: "99.997%"  # 4 sigma
        }
        
        # Find closest threshold
        closest_threshold = min(confidence_map.keys(), key=lambda x: abs(x - std_threshold))
        return confidence_map.get(closest_threshold, f"~{99.7:.1f}%")
    
    def _classify_anomaly_strength(self, zscore):
        """Classify anomaly strength based on Z-score"""
        if zscore >= 4.0:
            return "Extreme"
        elif zscore >= 3.5:
            return "Very Strong"
        elif zscore >= 3.0:
            return "Strong"
        elif zscore >= 2.5:
            return "Moderate"
        elif zscore >= 2.0:
            return "Weak"
        else:
            return "Minimal"


def run_enhanced_volume_anomaly_detection(batch_data: dict, params: dict = None) -> list:
    """
    Run enhanced volume anomaly detection across all tickers
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Optional parameters dictionary (uses defaults if None)
        
    Returns:
        List of anomaly detection results
    """
    if params is None:
        # Use default parameters if none provided
        params_config = ParametersConfig()
        params = params_config.get_params('volume_anomaly')
    
    # Initialize detector
    detector = VolumeAnomalyDetector(params)
    
    all_results = []
    
    logger.info(f"Running enhanced volume anomaly detection on {len(batch_data)} tickers")
    logger.info(f"Parameters: {params}")
    
    for ticker, data in batch_data.items():
        try:
            # Detect anomalies for this ticker
            anomalies_df = detector.detect_anomalies(data, ticker, method='all')
            
            if not anomalies_df.empty:
                # Convert to list of dictionaries
                anomalies = anomalies_df.to_dict('records')
                all_results.extend(anomalies)
                
                logger.debug(f"Found {len(anomalies)} anomalies for {ticker}")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            continue
    
    logger.info(f"Enhanced volume anomaly detection completed: {len(all_results)} total anomalies")
    return all_results


# Export main classes and functions
__all__ = [
    'ParametersConfig', 
    'VolumeAnomalyDetector', 
    'run_enhanced_volume_anomaly_detection'
]