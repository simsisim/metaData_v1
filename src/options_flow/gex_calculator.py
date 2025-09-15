#!/usr/bin/env python3
"""
Gamma Exposure (GEX) Calculator
===============================

Calculates gamma exposure levels from options chains to identify:
- Market maker hedging levels
- Support/resistance zones  
- Institutional positioning
- Delta neutral price points

GEX Formula: Σ(OI_calls × Gamma_calls) - Σ(OI_puts × Gamma_puts)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class GEXCalculator:
    """Calculate Gamma Exposure levels and analyze market maker positioning"""
    
    def __init__(self, min_oi_threshold=100, min_volume_threshold=10):
        """
        Initialize GEX calculator
        
        Args:
            min_oi_threshold: Minimum open interest to consider
            min_volume_threshold: Minimum volume to consider
        """
        self.min_oi_threshold = min_oi_threshold
        self.min_volume_threshold = min_volume_threshold
        
        # GEX interpretation thresholds
        self.gex_thresholds = {
            'very_high': 1000000,      # Very strong support/resistance
            'high': 500000,            # Strong level
            'moderate': 100000,        # Moderate significance
            'low': 50000              # Weak significance
        }
    
    def calculate_gex_from_chain(self, options_chain: pd.DataFrame, spot_price: float) -> Dict:
        """
        Calculate GEX levels from options chain data
        
        Args:
            options_chain: DataFrame with options data
            spot_price: Current underlying price
            
        Returns:
            Dictionary with GEX analysis results
        """
        try:
            if options_chain.empty:
                return self._empty_gex_result()
            
            # Separate calls and puts
            calls = options_chain[options_chain['type'] == 'call'].copy()
            puts = options_chain[options_chain['type'] == 'put'].copy()
            
            # Filter by minimum thresholds
            calls = calls[
                (calls['openInterest'] >= self.min_oi_threshold) & 
                (calls['volume'] >= self.min_volume_threshold)
            ]
            puts = puts[
                (puts['openInterest'] >= self.min_oi_threshold) & 
                (puts['volume'] >= self.min_volume_threshold)
            ]
            
            # Calculate per-strike GEX
            gex_levels = self._calculate_per_strike_gex(calls, puts, spot_price)
            
            # Identify significant levels
            significant_levels = self._identify_significant_gex_levels(gex_levels)
            
            # Calculate net GEX and market impact
            net_gex = gex_levels['gex_value'].sum()
            market_impact = self._analyze_market_impact(net_gex, significant_levels, spot_price)
            
            # Generate GEX profile summary
            gex_summary = self._generate_gex_summary(gex_levels, significant_levels, net_gex, spot_price)
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'spot_price': spot_price,
                'net_gex': net_gex,
                'gex_levels': gex_levels,
                'significant_levels': significant_levels,
                'market_impact': market_impact,
                'gex_summary': gex_summary,
                'calls_analyzed': len(calls),
                'puts_analyzed': len(puts)
            }
            
        except Exception as e:
            logger.error(f"Error calculating GEX: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_per_strike_gex(self, calls: pd.DataFrame, puts: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """Calculate GEX value for each strike price"""
        
        all_strikes = []
        
        # Process calls
        for _, call in calls.iterrows():
            strike = call['strike']
            oi = call['openInterest']
            gamma = call.get('gamma', self._estimate_gamma(spot_price, strike, 'call'))
            
            gex_value = oi * gamma * 100  # 100 shares per contract
            
            all_strikes.append({
                'strike': strike,
                'gex_value': gex_value,
                'type': 'call',
                'open_interest': oi,
                'volume': call['volume'],
                'gamma': gamma,
                'distance_from_spot_pct': ((strike - spot_price) / spot_price) * 100
            })
        
        # Process puts (negative GEX contribution)
        for _, put in puts.iterrows():
            strike = put['strike']
            oi = put['openInterest']
            gamma = put.get('gamma', self._estimate_gamma(spot_price, strike, 'put'))
            
            gex_value = -oi * gamma * 100  # Negative for puts
            
            all_strikes.append({
                'strike': strike,
                'gex_value': gex_value,
                'type': 'put',
                'open_interest': oi,
                'volume': put['volume'],
                'gamma': gamma,
                'distance_from_spot_pct': ((strike - spot_price) / spot_price) * 100
            })
        
        gex_df = pd.DataFrame(all_strikes)
        
        # Aggregate by strike (calls and puts at same strike)
        if not gex_df.empty:
            strike_gex = gex_df.groupby('strike').agg({
                'gex_value': 'sum',
                'open_interest': 'sum',
                'volume': 'sum',
                'distance_from_spot_pct': 'first'
            }).reset_index()
            
            strike_gex = strike_gex.sort_values('strike')
            return strike_gex
        
        return pd.DataFrame()
    
    def _estimate_gamma(self, spot_price: float, strike_price: float, option_type: str) -> float:
        """Estimate gamma when not provided (simplified Black-Scholes approximation)"""
        
        # Simplified gamma estimation
        # In production, use more sophisticated option pricing models
        
        moneyness = strike_price / spot_price
        
        if option_type == 'call':
            if 0.95 <= moneyness <= 1.05:  # ATM
                return 0.05
            elif 0.90 <= moneyness < 0.95 or 1.05 < moneyness <= 1.10:  # Near money
                return 0.03
            else:  # Far OTM/ITM
                return 0.01
        else:  # put
            if 0.95 <= moneyness <= 1.05:  # ATM
                return 0.05
            elif 0.90 <= moneyness < 0.95 or 1.05 < moneyness <= 1.10:  # Near money
                return 0.03
            else:  # Far OTM/ITM
                return 0.01
    
    def _identify_significant_gex_levels(self, gex_levels: pd.DataFrame) -> pd.DataFrame:
        """Identify the most significant GEX levels for support/resistance"""
        
        if gex_levels.empty:
            return pd.DataFrame()
        
        # Calculate absolute GEX for significance ranking
        gex_levels = gex_levels.copy()
        gex_levels['abs_gex'] = gex_levels['gex_value'].abs()
        
        # Identify significant levels
        significant = gex_levels[gex_levels['abs_gex'] >= self.gex_thresholds['low']].copy()
        
        if significant.empty:
            return pd.DataFrame()
        
        # Add significance classification
        conditions = [
            significant['abs_gex'] >= self.gex_thresholds['very_high'],
            significant['abs_gex'] >= self.gex_thresholds['high'],
            significant['abs_gex'] >= self.gex_thresholds['moderate'],
            significant['abs_gex'] >= self.gex_thresholds['low']
        ]
        
        choices = ['VERY_HIGH', 'HIGH', 'MODERATE', 'LOW']
        significant['significance'] = np.select(conditions, choices, default='MINIMAL')
        
        # Add market maker behavior prediction
        significant['mm_behavior'] = significant['gex_value'].apply(
            lambda x: 'PRICE_MAGNET' if x > 0 else 'PRICE_REPULSION' if x < 0 else 'NEUTRAL'
        )
        
        # Sort by absolute GEX value (most significant first)
        significant = significant.sort_values('abs_gex', ascending=False)
        
        return significant
    
    def _analyze_market_impact(self, net_gex: float, significant_levels: pd.DataFrame, spot_price: float) -> Dict:
        """Analyze the market impact of current GEX positioning"""
        
        impact_analysis = {
            'net_gex': net_gex,
            'gex_regime': 'POSITIVE' if net_gex > 0 else 'NEGATIVE' if net_gex < 0 else 'NEUTRAL',
            'volatility_forecast': 'LOW' if net_gex > 1000000 else 'HIGH' if net_gex < -500000 else 'MODERATE',
            'price_behavior': 'MEAN_REVERTING' if net_gex > 0 else 'TRENDING' if net_gex < 0 else 'MIXED'
        }
        
        # Find nearest significant levels
        if not significant_levels.empty:
            above_spot = significant_levels[significant_levels['strike'] > spot_price]
            below_spot = significant_levels[significant_levels['strike'] < spot_price]
            
            impact_analysis['resistance_levels'] = above_spot['strike'].tolist()[:3]
            impact_analysis['support_levels'] = below_spot['strike'].tolist()[:3]
            
            # Find strongest nearby level
            nearby_levels = significant_levels[
                abs(significant_levels['distance_from_spot_pct']) <= 5.0
            ]
            
            if not nearby_levels.empty:
                strongest_nearby = nearby_levels.iloc[0]
                impact_analysis['key_level'] = {
                    'strike': strongest_nearby['strike'],
                    'gex_value': strongest_nearby['gex_value'],
                    'significance': strongest_nearby['significance'],
                    'distance_pct': strongest_nearby['distance_from_spot_pct']
                }
        
        return impact_analysis
    
    def _generate_gex_summary(self, gex_levels: pd.DataFrame, significant_levels: pd.DataFrame, 
                             net_gex: float, spot_price: float) -> Dict:
        """Generate executive summary of GEX analysis"""
        
        summary = {
            'total_strikes_analyzed': len(gex_levels),
            'significant_levels_count': len(significant_levels),
            'net_gex': net_gex,
            'gex_regime': 'POSITIVE' if net_gex > 0 else 'NEGATIVE' if net_gex < 0 else 'NEUTRAL',
            'market_structure': 'SUPPORTIVE' if net_gex > 1000000 else 'DESTABILIZING' if net_gex < -500000 else 'NEUTRAL'
        }
        
        if not significant_levels.empty:
            # Key statistics
            max_positive_gex = significant_levels[significant_levels['gex_value'] > 0]['gex_value'].max()
            max_negative_gex = significant_levels[significant_levels['gex_value'] < 0]['gex_value'].min()
            
            summary['strongest_support'] = max_positive_gex if pd.notna(max_positive_gex) else 0
            summary['strongest_resistance'] = abs(max_negative_gex) if pd.notna(max_negative_gex) else 0
            
            # Nearest levels analysis
            above_spot = significant_levels[significant_levels['strike'] > spot_price]
            below_spot = significant_levels[significant_levels['strike'] < spot_price]
            
            if not above_spot.empty:
                nearest_resistance = above_spot.iloc[-1]  # Closest above
                summary['nearest_resistance'] = {
                    'strike': nearest_resistance['strike'],
                    'distance_pct': nearest_resistance['distance_from_spot_pct']
                }
            
            if not below_spot.empty:
                nearest_support = below_spot.iloc[0]  # Closest below
                summary['nearest_support'] = {
                    'strike': nearest_support['strike'],
                    'distance_pct': abs(nearest_support['distance_from_spot_pct'])
                }
        
        return summary
    
    def _empty_gex_result(self) -> Dict:
        """Return empty result structure when no data available"""
        return {
            'success': False,
            'error': 'No options data available',
            'net_gex': 0,
            'gex_levels': pd.DataFrame(),
            'significant_levels': pd.DataFrame(),
            'market_impact': {},
            'gex_summary': {}
        }
    
    def calculate_gex_change(self, current_gex: Dict, previous_gex: Dict) -> Dict:
        """Calculate GEX changes between periods for flow analysis"""
        
        if not current_gex.get('success') or not previous_gex.get('success'):
            return {'success': False, 'error': 'Invalid GEX data for comparison'}
        
        net_gex_change = current_gex['net_gex'] - previous_gex['net_gex']
        
        gex_change_analysis = {
            'success': True,
            'net_gex_change': net_gex_change,
            'change_percentage': (net_gex_change / abs(previous_gex['net_gex'])) * 100 if previous_gex['net_gex'] != 0 else 0,
            'regime_shift': current_gex['gex_summary']['gex_regime'] != previous_gex['gex_summary']['gex_regime'],
            'flow_direction': 'BULLISH_FLOW' if net_gex_change > 0 else 'BEARISH_FLOW' if net_gex_change < 0 else 'NEUTRAL_FLOW'
        }
        
        # Detect significant level changes
        current_levels = set(current_gex['significant_levels']['strike'].tolist())
        previous_levels = set(previous_gex['significant_levels']['strike'].tolist())
        
        gex_change_analysis['new_levels'] = list(current_levels - previous_levels)
        gex_change_analysis['removed_levels'] = list(previous_levels - current_levels)
        gex_change_analysis['level_stability'] = len(current_levels & previous_levels) / max(len(current_levels | previous_levels), 1)
        
        return gex_change_analysis
    
    def get_gex_trading_signals(self, gex_result: Dict, price_data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals based on GEX analysis"""
        
        signals = []
        
        if not gex_result.get('success'):
            return signals
        
        spot_price = gex_result['spot_price']
        market_impact = gex_result['market_impact']
        significant_levels = gex_result['significant_levels']
        
        # GEX regime signals
        if market_impact.get('gex_regime') == 'POSITIVE':
            if market_impact.get('volatility_forecast') == 'LOW':
                signals.append({
                    'type': 'GEX_REGIME',
                    'signal': 'MEAN_REVERSION_SETUP',
                    'confidence': 0.75,
                    'description': 'Positive GEX suggests mean-reverting environment - fade extremes',
                    'action': 'Look for reversal setups at key levels'
                })
        
        elif market_impact.get('gex_regime') == 'NEGATIVE':
            if market_impact.get('volatility_forecast') == 'HIGH':
                signals.append({
                    'type': 'GEX_REGIME',
                    'signal': 'MOMENTUM_BREAKOUT',
                    'confidence': 0.80,
                    'description': 'Negative GEX suggests trending environment - follow momentum',
                    'action': 'Look for breakout continuations'
                })
        
        # Key level proximity signals
        if 'key_level' in market_impact:
            key_level = market_impact['key_level']
            distance_pct = abs(key_level['distance_pct'])
            
            if distance_pct <= 2.0:  # Within 2% of key level
                level_type = 'SUPPORT' if key_level['gex_value'] > 0 else 'RESISTANCE'
                
                signals.append({
                    'type': 'GEX_LEVEL',
                    'signal': f'{level_type}_TEST',
                    'confidence': min(0.95, 0.5 + (key_level['significance'] == 'VERY_HIGH') * 0.3),
                    'description': f'Price approaching key GEX {level_type.lower()} at ${key_level["strike"]:.2f}',
                    'action': f'Monitor for {level_type.lower()} hold/break',
                    'key_level': key_level['strike'],
                    'current_distance_pct': key_level['distance_pct']
                })
        
        return signals
    
    def analyze_dealer_positioning(self, gex_result: Dict) -> Dict:
        """Analyze market maker positioning and hedging requirements"""
        
        if not gex_result.get('success'):
            return {'success': False}
        
        net_gex = gex_result['net_gex']
        significant_levels = gex_result['significant_levels']
        spot_price = gex_result['spot_price']
        
        # Determine dealer positioning
        if net_gex > 1000000:
            dealer_position = 'LONG_GAMMA'
            hedging_behavior = 'STABILIZING'
            volatility_impact = 'SUPPRESSING'
        elif net_gex < -500000:
            dealer_position = 'SHORT_GAMMA'
            hedging_behavior = 'DESTABILIZING'
            volatility_impact = 'AMPLIFYING'
        else:
            dealer_position = 'NEUTRAL_GAMMA'
            hedging_behavior = 'MINIMAL'
            volatility_impact = 'NEUTRAL'
        
        # Calculate hedging pressure zones
        hedging_zones = []
        if not significant_levels.empty:
            for _, level in significant_levels.head(5).iterrows():
                distance = abs(level['distance_from_spot_pct'])
                zone_strength = 'HIGH' if level['significance'] in ['VERY_HIGH', 'HIGH'] else 'MODERATE'
                
                hedging_zones.append({
                    'strike': level['strike'],
                    'gex_value': level['gex_value'],
                    'zone_type': 'SUPPORT' if level['gex_value'] > 0 else 'RESISTANCE',
                    'strength': zone_strength,
                    'distance_pct': level['distance_from_spot_pct']
                })
        
        return {
            'success': True,
            'dealer_position': dealer_position,
            'hedging_behavior': hedging_behavior,
            'volatility_impact': volatility_impact,
            'net_gamma_exposure': net_gex,
            'hedging_zones': hedging_zones,
            'market_maker_bias': 'BULLISH' if net_gex > 0 else 'BEARISH' if net_gex < 0 else 'NEUTRAL'
        }


def run_gex_analysis(ticker: str, options_chain: pd.DataFrame, current_price: float) -> Dict:
    """
    Main entry point for GEX analysis
    
    Args:
        ticker: Stock ticker symbol
        options_chain: Options chain data from yfinance
        current_price: Current stock price
        
    Returns:
        Complete GEX analysis results
    """
    calculator = GEXCalculator()
    
    # Calculate GEX levels
    gex_result = calculator.calculate_gex_from_chain(options_chain, current_price)
    
    if gex_result.get('success'):
        # Add ticker information
        gex_result['ticker'] = ticker
        
        # Generate trading signals
        gex_result['trading_signals'] = calculator.get_gex_trading_signals(gex_result, None)
        
        # Analyze dealer positioning
        gex_result['dealer_analysis'] = calculator.analyze_dealer_positioning(gex_result)
        
        logger.info(f"GEX analysis completed for {ticker}: Net GEX = {gex_result['net_gex']:,.0f}")
    
    return gex_result