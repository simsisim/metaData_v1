"""
Range Tightening Indicator (RTI) Screener
=========================================

Implementation of the Range Tightening Indicator strategy from TradingView.
Identifies stocks with low-volatility consolidations that often precede breakouts
by measuring price range volatility and detecting range expansion signals.

Strategy Components:
1. RTI Calculation: Range volatility measurement over configurable periods
2. Volatility Zones: Extremely tight (0-5), Low (5-10), Moderate low (10-15)
3. Orange Dot Signals: Extended low-volatility consolidation periods
4. Range Expansion: Detection of volatility doubling after compression

Based on: https://www.tradingview.com/script/yaIeno72-Range-Tightening-Indicator-RTI/
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class RTIScreener:
    """
    Range Tightening Indicator screener implementation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_rti = config.get('enable_rti', True)
        self.timeframe = config.get('timeframe', 'daily')
        
        # RTI specific configuration
        self.rti_config = config.get('rti_screener', {})
        
        # RTI Calculation Parameters
        self.rti_period = self.rti_config.get('rti_period', 50)
        self.rti_short_period = self.rti_config.get('rti_short_period', 5)
        self.rti_swing_period = self.rti_config.get('rti_swing_period', 15)
        
        # Volatility Zone Thresholds
        self.zone1_threshold = self.rti_config.get('zone1_threshold', 5.0)  # Extremely tight
        self.zone2_threshold = self.rti_config.get('zone2_threshold', 10.0)  # Low volatility
        self.zone3_threshold = self.rti_config.get('zone3_threshold', 15.0)  # Moderate low
        self.low_volatility_threshold = self.rti_config.get('low_volatility_threshold', 20.0)
        
        # Range Expansion Parameters
        self.expansion_multiplier = self.rti_config.get('expansion_multiplier', 2.0)
        self.consecutive_low_vol_bars = self.rti_config.get('consecutive_low_vol_bars', 2)
        
        # Signal Detection Parameters
        self.min_consolidation_period = self.rti_config.get('min_consolidation_period', 3)
        self.breakout_confirmation_period = self.rti_config.get('breakout_confirmation_period', 2)
        
        # Base Filters
        self.min_price = self.rti_config.get('min_price', 5.0)
        self.min_volume = self.rti_config.get('min_volume', 100000)
        self.min_data_length = max(self.rti_period, 60)
        
        # Output configuration
        self.save_individual_files = self.rti_config.get('save_individual_files', True)
        
        logger.info(f"RTI Screener initialized (enabled: {self.enable_rti})")
        logger.info(f"RTI period: {self.rti_period}, Low vol threshold: {self.low_volatility_threshold}")

    def _calculate_rti(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Range Tightening Indicator values"""
        rti_data = data.copy()
        
        # Calculate bar ranges (High - Low)
        bar_ranges = data['High'] - data['Low']
        
        # Calculate RTI for different periods
        rti_data['RTI'] = bar_ranges.rolling(window=self.rti_period).mean()
        rti_data['RTI_Short'] = bar_ranges.rolling(window=self.rti_short_period).mean()
        rti_data['RTI_Swing'] = bar_ranges.rolling(window=self.rti_swing_period).mean()
        
        # Normalize RTI by average price for percentage-based analysis
        avg_price = (data['High'] + data['Low'] + data['Close']) / 3
        rti_data['RTI_Pct'] = (rti_data['RTI'] / avg_price) * 100
        rti_data['RTI_Short_Pct'] = (rti_data['RTI_Short'] / avg_price) * 100
        rti_data['RTI_Swing_Pct'] = (rti_data['RTI_Swing'] / avg_price) * 100
        
        return rti_data

    def _detect_volatility_zones(self, rti_data: pd.DataFrame) -> pd.DataFrame:
        """Classify volatility into zones"""
        zones_data = rti_data.copy()
        
        # Classify RTI into volatility zones
        conditions = [
            (rti_data['RTI_Pct'] <= self.zone1_threshold),
            (rti_data['RTI_Pct'] <= self.zone2_threshold),
            (rti_data['RTI_Pct'] <= self.zone3_threshold),
        ]
        
        choices = ['Zone1_Extremely_Tight', 'Zone2_Low_Volatility', 'Zone3_Moderate_Low']
        zones_data['Volatility_Zone'] = np.select(conditions, choices, default='Normal_Volatility')
        
        # Mark low volatility periods
        zones_data['Low_Volatility'] = rti_data['RTI_Pct'] < self.low_volatility_threshold
        
        return zones_data

    def _detect_orange_dot_signals(self, zones_data: pd.DataFrame) -> pd.DataFrame:
        """Detect orange dot signals (consecutive low volatility periods)"""
        signals_data = zones_data.copy()
        
        # Count consecutive low volatility bars
        low_vol_groups = (zones_data['Low_Volatility'] != zones_data['Low_Volatility'].shift()).cumsum()
        consecutive_count = zones_data.groupby(low_vol_groups)['Low_Volatility'].cumsum()
        
        # Orange dot when we have consecutive low volatility bars
        signals_data['Orange_Dot'] = (
            zones_data['Low_Volatility'] & 
            (consecutive_count >= self.consecutive_low_vol_bars)
        )
        
        return signals_data

    def _detect_range_expansion(self, signals_data: pd.DataFrame) -> pd.DataFrame:
        """Detect range expansion after compression"""
        expansion_data = signals_data.copy()
        
        # Calculate recent volatility vs historical average
        recent_rti = signals_data['RTI_Pct'].rolling(window=3).mean()
        historical_avg = signals_data['RTI_Pct'].rolling(window=self.rti_period).mean()
        
        # Range expansion when recent RTI exceeds historical average by multiplier
        expansion_data['Range_Expansion'] = (
            recent_rti > (historical_avg * self.expansion_multiplier)
        )
        
        # Range expansion after compression (green signal condition)
        prev_low_vol = signals_data['Low_Volatility'].shift(1)
        expansion_data['Expansion_After_Compression'] = (
            expansion_data['Range_Expansion'] & prev_low_vol
        )
        
        return expansion_data

    def _apply_base_filters(self, batch_data: Dict[str, pd.DataFrame], 
                          ticker_info: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """Apply base filtering criteria"""
        filtered_data = {}
        
        for ticker, data in batch_data.items():
            if len(data) < self.min_data_length:
                continue
                
            latest_data = data.iloc[-1]
            
            # Price filter
            if latest_data['Close'] < self.min_price:
                continue
                
            # Volume filter
            avg_volume = data['Volume'].tail(20).mean()
            if avg_volume < self.min_volume:
                continue
                
            # ETF filter (basic)
            if ticker.upper() in ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'ARKK']:
                continue
                
            filtered_data[ticker] = data
            
        logger.info(f"RTI base filters: {len(filtered_data)}/{len(batch_data)} tickers passed")
        return filtered_data

    def _run_rti_screening(self, batch_data: Dict[str, pd.DataFrame],
                         ticker_info: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Run the RTI screening logic"""
        results = []
        
        for ticker, data in batch_data.items():
            try:
                # Calculate RTI values
                rti_data = self._calculate_rti(data)
                
                # Detect volatility zones
                zones_data = self._detect_volatility_zones(rti_data)
                
                # Detect orange dot signals
                signals_data = self._detect_orange_dot_signals(zones_data)
                
                # Detect range expansion
                expansion_data = self._detect_range_expansion(signals_data)
                
                latest_data = expansion_data.iloc[-1]
                
                # Check for different RTI signals
                signals_found = []
                
                # 1. Orange Dot Signal (Low volatility consolidation)
                if latest_data['Orange_Dot']:
                    signals_found.append({
                        'signal_type': 'Orange Dot Consolidation',
                        'strength': self._calculate_consolidation_strength(expansion_data, ticker),
                        'rti_value': latest_data['RTI_Pct'],
                        'volatility_zone': latest_data['Volatility_Zone']
                    })
                
                # 2. Range Expansion Signal (Breakout after compression)
                if latest_data['Expansion_After_Compression']:
                    signals_found.append({
                        'signal_type': 'Range Expansion Breakout',
                        'strength': self._calculate_expansion_strength(expansion_data, ticker),
                        'rti_value': latest_data['RTI_Pct'],
                        'expansion_ratio': latest_data['RTI_Pct'] / expansion_data['RTI_Pct'].rolling(window=self.rti_period).mean().iloc[-1] if expansion_data['RTI_Pct'].rolling(window=self.rti_period).mean().iloc[-1] > 0 else 1.0
                    })
                
                # 3. Extremely Tight Range Signal (Zone 1)
                if latest_data['Volatility_Zone'] == 'Zone1_Extremely_Tight':
                    signals_found.append({
                        'signal_type': 'Extremely Tight Range',
                        'strength': 'Extreme',
                        'rti_value': latest_data['RTI_Pct'],
                        'volatility_zone': latest_data['Volatility_Zone']
                    })
                
                # 4. Low Volatility Signal (Zone 2)
                elif latest_data['Volatility_Zone'] == 'Zone2_Low_Volatility':
                    signals_found.append({
                        'signal_type': 'Low Volatility Range',
                        'strength': 'Strong',
                        'rti_value': latest_data['RTI_Pct'],
                        'volatility_zone': latest_data['Volatility_Zone']
                    })
                
                # Create result entries for each signal
                for signal in signals_found:
                    results.append({
                        'ticker': ticker,
                        'screen_type': 'rti_screener',
                        'signal_type': signal['signal_type'],
                        'signal_date': latest_data.name.strftime('%Y-%m-%d'),
                        'price': latest_data['Close'],
                        'volume': latest_data['Volume'],
                        'rti_value': signal['rti_value'],
                        'rti_period': self.rti_period,
                        'volatility_zone': signal.get('volatility_zone', 'N/A'),
                        'expansion_ratio': signal.get('expansion_ratio', 1.0),
                        'strength': signal['strength'],
                        'timeframe': self.timeframe
                    })
                    
            except Exception as e:
                logger.warning(f"RTI screening failed for {ticker}: {e}")
                continue
                
        return results

    def _calculate_consolidation_strength(self, data: pd.DataFrame, ticker: str) -> str:
        """Calculate strength of consolidation signal"""
        try:
            recent_rti = data['RTI_Pct'].tail(10).mean()
            if recent_rti < self.zone1_threshold:
                return 'Extreme'
            elif recent_rti < self.zone2_threshold:
                return 'Strong'
            elif recent_rti < self.zone3_threshold:
                return 'Moderate'
            else:
                return 'Weak'
        except:
            return 'Moderate'

    def _calculate_expansion_strength(self, data: pd.DataFrame, ticker: str) -> str:
        """Calculate strength of expansion signal"""
        try:
            latest_rti = data['RTI_Pct'].iloc[-1]
            avg_rti = data['RTI_Pct'].rolling(window=self.rti_period).mean().iloc[-1]
            
            if latest_rti > avg_rti * 3:
                return 'Extreme'
            elif latest_rti > avg_rti * 2:
                return 'Strong'
            elif latest_rti > avg_rti * 1.5:
                return 'Moderate'
            else:
                return 'Weak'
        except:
            return 'Moderate'

    def run_rti_screening(self, batch_data: Dict[str, pd.DataFrame],
                        ticker_info: Optional[pd.DataFrame] = None,
                        batch_info: Optional[Dict] = None) -> List[Dict]:
        """
        Run RTI screening on batch data
        
        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            ticker_info: Optional DataFrame with ticker information
            batch_info: Optional batch processing information
            
        Returns:
            List of RTI screening results
        """
        if not self.enable_rti:
            logger.info("RTI screening disabled")
            return []
            
        if not batch_data:
            logger.warning("No data provided for RTI screening")
            return []
        
        logger.info(f"Running RTI screening on {len(batch_data)} tickers")
        
        # Apply base filters
        filtered_data = self._apply_base_filters(batch_data, ticker_info)
        if not filtered_data:
            logger.warning("No tickers passed RTI base filters")
            return []
        
        # Run RTI screening
        results = self._run_rti_screening(filtered_data, ticker_info)
        
        logger.info(f"RTI screening completed: {len(results)} signals found")
        return results


def run_rti_screener(batch_data: Dict[str, pd.DataFrame], 
                   config: Dict[str, Any],
                   ticker_info: Optional[pd.DataFrame] = None,
                   batch_info: Optional[Dict] = None) -> List[Dict]:
    """
    Standalone function to run RTI screening
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        config: Configuration dictionary with RTI parameters
        ticker_info: Optional DataFrame with ticker information
        batch_info: Optional batch processing information
        
    Returns:
        List of RTI screening results
    """
    screener = RTIScreener(config)
    return screener.run_rti_screening(batch_data, ticker_info, batch_info)