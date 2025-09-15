"""
Volume Suite Screener
=====================

Comprehensive volume analysis screener that combines multiple volume-based
screening methodologies for detecting volume anomalies and breakouts.

Components:
- HVAbsoluteETC: Enhanced absolute volume analysis 
- HVStdv: Statistical volume anomaly detection
- Enhanced Volume Anomaly: Multi-method anomaly detection
- VROC, RVOL, ADTV: Volume rate of change and relative volume
- MFI, VPT: Money flow and volume price trend indicators
- PVB Integration: Price-volume breakout detection

Based on the original volume_suite implementation at:
/home/imagda/_invest2024/python/volume_suite
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

# Import volume suite components
from .volume_suite_components.HVAbsoluteETC import run_HVAbsoluteStrategy_Enhanced
from .volume_suite_components.HVStdv import run_HVStdvStrategy 
from .volume_suite_components.enhanced_volume_anomaly import (
    VolumeAnomalyDetector, 
    ParametersConfig,
    run_enhanced_volume_anomaly_detection
)
from .volume_suite_components.volume_indicators import run_volume_indicators_analysis
from .pvb_screener import pvb_screener

logger = logging.getLogger(__name__)


class VolumeSuiteScreener:
    """
    Main volume suite screener coordinating all volume analysis methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_volume_suite = config.get('enable_volume_suite', True)
        self.timeframe = config.get('timeframe', 'daily')
        
        # Volume suite specific configuration
        self.volume_config = config.get('volume_suite', {})
        self.enable_hv_absolute = self.volume_config.get('enable_hv_absolute', True)
        self.enable_hv_stdv = self.volume_config.get('enable_hv_stdv', True) 
        self.enable_enhanced_anomaly = self.volume_config.get('enable_enhanced_anomaly', True)
        self.enable_volume_indicators = self.volume_config.get('enable_volume_indicators', True)
        self.enable_pvb_integration = self.volume_config.get('enable_pvb_integration', True)
        
        # Output configuration
        self.output_dir = config.get('volume_output_dir', 'results/screeners/volume_suite')
        self.save_individual_files = self.volume_config.get('save_individual_files', True)
        
        # Initialize parameter configuration
        self.params_config = ParametersConfig()
        
        logger.info(f"Volume Suite Screener initialized (enabled: {self.enable_volume_suite})")

    def run_volume_suite_screening(self, batch_data: Dict[str, pd.DataFrame], 
                                 batch_info: Dict[str, Any] = None) -> List[Dict]:
        """
        Run comprehensive volume suite screening
        
        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            batch_info: Optional batch processing information
            
        Returns:
            List of screening results
        """
        if not self.enable_volume_suite:
            logger.info("Volume suite screening disabled")
            return []
            
        if not batch_data:
            logger.warning("No data provided for volume suite screening")
            return []
        
        logger.info(f"Running volume suite screening on {len(batch_data)} tickers")
        
        all_results = []
        component_results = {}
        
        try:
            # 1. HV Absolute Analysis
            if self.enable_hv_absolute:
                logger.info("Running HV Absolute analysis...")
                hv_absolute_params = self._get_hv_absolute_params()
                hv_absolute_results = run_HVAbsoluteStrategy_Enhanced(batch_data, hv_absolute_params)
                component_results['hv_absolute'] = hv_absolute_results
                all_results.extend(self._format_hv_absolute_results(hv_absolute_results))
            
            # 2. HV Standard Deviation Analysis
            if self.enable_hv_stdv:
                logger.info("Running HV Standard Deviation analysis...")
                hv_stdv_params = self._get_hv_stdv_params()
                hv_stdv_results = run_HVStdvStrategy(batch_data, hv_stdv_params)
                component_results['hv_stdv'] = hv_stdv_results
                all_results.extend(self._format_hv_stdv_results(hv_stdv_results))
            
            # 3. Enhanced Volume Anomaly Detection
            if self.enable_enhanced_anomaly:
                logger.info("Running Enhanced Volume Anomaly detection...")
                enhanced_results = self._run_enhanced_anomaly_detection(batch_data)
                component_results['enhanced_anomaly'] = enhanced_results
                all_results.extend(enhanced_results)
            
            # 4. Volume Indicators (VROC, RVOL, ADTV, MFI, VPT)
            if self.enable_volume_indicators:
                logger.info("Running Volume Indicators analysis...")
                volume_indicator_results = self._run_volume_indicators(batch_data)
                component_results['volume_indicators'] = volume_indicator_results
                all_results.extend(volume_indicator_results)
            
            # 5. PVB Integration
            if self.enable_pvb_integration:
                logger.info("Running PVB integration...")
                pvb_results = self._run_pvb_integration(batch_data)
                component_results['pvb'] = pvb_results
                all_results.extend(pvb_results)
            
            # Save results if enabled
            if self.save_individual_files:
                self._save_component_results(component_results)
            
            logger.info(f"Volume suite screening completed: {len(all_results)} total signals")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in volume suite screening: {e}")
            return []

    def _get_hv_absolute_params(self) -> Dict[str, Any]:
        """Get HV Absolute analysis parameters"""
        return {
            'month_cuttoff': self.volume_config.get('hv_month_cutoff', 15),
            'day_cuttoff': self.volume_config.get('hv_day_cutoff', 3),
            'std_cuttoff': self.volume_config.get('hv_std_cutoff', 10),
            'min_stock_volume': self.volume_config.get('hv_min_volume', 100000),
            'min_price': self.volume_config.get('hv_min_price', 20),
            'use_enhanced_filtering': True
        }

    def _get_hv_stdv_params(self) -> Dict[str, Any]:
        """Get HV Standard Deviation parameters"""
        return {
            'std_cuttoff': self.volume_config.get('stdv_cutoff', 12),
            'min_stock_volume': self.volume_config.get('stdv_min_volume', 10000)
        }

    def _run_enhanced_anomaly_detection(self, batch_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Run enhanced volume anomaly detection"""
        try:
            # Get parameters for volume anomaly detection
            anomaly_params = self.params_config.get_params('volume_anomaly')
            
            # Initialize detector
            detector = VolumeAnomalyDetector(anomaly_params)
            
            # Process each ticker
            all_anomalies = []
            for ticker, data in batch_data.items():
                try:
                    anomalies_df = detector.detect_anomalies(data, ticker, method='all')
                    if not anomalies_df.empty:
                        anomalies = anomalies_df.to_dict('records')
                        all_anomalies.extend(anomalies)
                except Exception as e:
                    logger.warning(f"Error detecting anomalies for {ticker}: {e}")
                    continue
            
            # Format results with screening metadata
            formatted_results = []
            for anomaly in all_anomalies:
                formatted_results.append({
                    'ticker': anomaly['ticker'],
                    'signal_date': anomaly['signal_date'],
                    'signal_type': anomaly['signal_type'],
                    'screen_type': 'volume_anomaly',
                    'price': anomaly['price'],
                    'volume': anomaly['volume'],
                    'method': anomaly['detection_method'],
                    'confidence': anomaly.get('threshold_confidence', 'N/A'),
                    'strength': anomaly.get('anomaly_strength', 'N/A'),
                    'raw_data': anomaly
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in enhanced anomaly detection: {e}")
            return []

    def _run_volume_indicators(self, batch_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Run volume indicators analysis (VROC, RVOL, ADTV, MFI, VPT)"""
        try:
            # Get volume indicator parameters
            volume_params = {
                'vroc_threshold': self.volume_config.get('vroc_threshold', 50),
                'rvol_threshold': self.volume_config.get('rvol_threshold', 2.0),
                'rvol_extreme_threshold': self.volume_config.get('rvol_extreme_threshold', 5.0),
                'mfi_overbought_threshold': self.volume_config.get('mfi_overbought', 80),
                'mfi_oversold_threshold': self.volume_config.get('mfi_oversold', 20),
                'vpt_signal_threshold': self.volume_config.get('vpt_threshold', 0.05),
                'adtv_3m_threshold': self.volume_config.get('adtv_3m_threshold', 2.0),
                'adtv_6m_threshold': self.volume_config.get('adtv_6m_threshold', 2.0),
                'adtv_1y_threshold': self.volume_config.get('adtv_1y_threshold', 2.0),
                'adtv_min_volume': self.volume_config.get('adtv_min_volume', 1000000)
            }
            
            # Run volume indicators analysis
            indicators_output = run_volume_indicators_analysis(batch_data, volume_params)
            
            # Process results and format for screening output
            results = []
            if indicators_output and isinstance(indicators_output, dict):
                signals = indicators_output.get('signals', {})
                
                # Process each signal type
                for signal_type, signal_list in signals.items():
                    for signal in signal_list:
                        try:
                            # Convert to screening result format
                            results.append({
                                'ticker': signal['Ticker'],
                                'signal_date': signal['Date'],
                                'signal_type': signal_type,
                                'screen_type': 'volume_indicators',
                                'price': signal.get('Close', signal.get('price', 0)),
                                'volume': signal.get('Volume', signal.get('volume', 0)),
                                'strength': signal.get('Signal_Strength', 'moderate'),
                                'raw_data': signal
                            })
                        except Exception as e:
                            logger.warning(f"Error formatting {signal_type} signal: {e}")
                            continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in volume indicators analysis: {e}")
            return []

    def _run_pvb_integration(self, batch_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Run PVB integration for price-volume breakout detection"""
        try:
            # Get PVB parameters
            pvb_config = {
                'enable_pvb': True,
                'pvb_price_breakout_period': self.volume_config.get('pvb_price_period', 30),
                'pvb_volume_breakout_period': self.volume_config.get('pvb_volume_period', 30),
                'pvb_trendline_length': self.volume_config.get('pvb_trend_length', 50),
                'pvb_volume_multiplier': self.volume_config.get('pvb_volume_multiplier', 1.5),
                'pvb_order_direction': self.volume_config.get('pvb_direction', 'Long'),
                'pvb_output_dir': self.output_dir
            }
            
            # Run PVB screener
            pvb_results = pvb_screener(batch_data, pvb_config)
            
            # Format PVB results for volume suite
            formatted_results = []
            for result in pvb_results:
                formatted_results.append({
                    'ticker': result['ticker'],
                    'signal_date': result['signal_date'],
                    'signal_type': 'price_volume_breakout',
                    'screen_type': 'volume_suite_pvb',
                    'price': result['price'],
                    'volume': result.get('volume', 0),
                    'pvb_signal': result.get('signal_type', 'breakout'),
                    'strength': result.get('signal_strength', 'moderate'),
                    'raw_data': result
                })
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Error in PVB integration: {e}")
            return []

    def _format_hv_absolute_results(self, hv_results: List[Dict]) -> List[Dict]:
        """Format HV Absolute results for screening output"""
        formatted = []
        for result in hv_results:
            if result.get('Date') is not None and result.get('MaxVolume') is not None:
                formatted.append({
                    'ticker': result['Ticker'],
                    'signal_date': result['Date'],
                    'signal_type': 'high_volume_absolute',
                    'screen_type': 'volume_suite_hv_absolute',
                    'price': result.get('Close', 0),
                    'volume': result['MaxVolume'],
                    'max_volume': result['MaxVolume'],
                    'strength': 'strong',
                    'raw_data': result
                })
        return formatted

    def _format_hv_stdv_results(self, hv_stdv_results: List[Dict]) -> List[Dict]:
        """Format HV Standard Deviation results for screening output"""
        formatted = []
        for result in hv_stdv_results:
            if result.get('Date') is not None and result.get('UnusualVolume') is not None:
                formatted.append({
                    'ticker': result['Ticker'],
                    'signal_date': result['Date'],
                    'signal_type': 'high_volume_statistical',
                    'screen_type': 'volume_suite_hv_stdv',
                    'price': result.get('Close', 0),
                    'volume': result['UnusualVolume'],
                    'unusual_volume': result['UnusualVolume'],
                    'strength': 'strong',
                    'raw_data': result
                })
        return formatted

    def _save_component_results(self, component_results: Dict[str, Any]):
        """Save individual component results to files"""
        try:
            # Create output directory
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d')
            
            # Save each component's results
            for component_name, results in component_results.items():
                if results:
                    filename = f"volume_suite_{component_name}_{self.timeframe}_{timestamp}.csv"
                    filepath = output_dir / filename
                    
                    # Convert to DataFrame and save
                    if isinstance(results, list) and results:
                        df = pd.DataFrame(results)
                        df.to_csv(filepath, index=False)
                        logger.info(f"Saved {component_name} results: {len(df)} signals to {filepath}")
                    
        except Exception as e:
            logger.error(f"Error saving component results: {e}")


def run_volume_suite_screener(batch_data: Dict[str, pd.DataFrame], 
                             config: Dict[str, Any]) -> List[Dict]:
    """
    Main entry point for volume suite screening
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        config: Configuration dictionary
        
    Returns:
        List of screening results
    """
    screener = VolumeSuiteScreener(config)
    return screener.run_volume_suite_screening(batch_data)


# Export main functions
__all__ = ['VolumeSuiteScreener', 'run_volume_suite_screener']