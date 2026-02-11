#!/usr/bin/env python3
"""
GMI (General Market Index) Calculator - Phase 1

Implements Dr. Eric Wish's 6-component market timing methodology.
Currently implements 4 of 6 components (excluding P4 breadth and P6 distribution).

GMI-P Components:
- P1: Short-term Trend (Close > 50-day SMA)
- P2: Trend Momentum (50-day SMA rising)
- P3: Long-term Trend (Close > 150-day SMA)
- P5: Price Momentum (Closed higher ≥2 of last 3 days)

Future: P4 (Market Breadth), P6 (Distribution Days)

Signal Logic: 
- GREEN: GMI > threshold for confirmation_days
- RED: GMI < threshold for confirmation_days
- NEUTRAL: Otherwise
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import os
from datetime import datetime

class GMICalculator:
    """
    General Market Index (GMI) Calculator implementing Dr. Eric Wish's market timing methodology.
    
    This implements a 4-component proxy model (GMI-P) that produces a daily score from 0-4.
    Signal Logic: GMI > threshold for confirmation_days = Green (bullish), 
                  GMI < threshold for confirmation_days = Red (bearish)
    """
    
    def __init__(self, market_index: str = 'QQQ', paths: Dict[str, str] = None, 
                 threshold: int = 3, confirmation_days: int = 2):
        """
        Initialize GMI Calculator
        
        Args:
            market_index: Market index to analyze (QQQ, SPY, etc.)
            paths: Dictionary containing data paths
            threshold: GMI score threshold for signals (default: 3)
            confirmation_days: Days required for signal confirmation (default: 2)
        """
        self.market_index = market_index
        self.paths = paths or {}
        self.threshold = threshold
        self.confirmation_days = confirmation_days
        self.gmi_history = []
        self.current_signal = None
        self.signal_days = 0
        
        # Component configuration for future extension
        self.components_config = {
            'P1': {'name': 'Short-term Trend', 'enabled': True},
            'P2': {'name': 'Trend Momentum', 'enabled': True},
            'P3': {'name': 'Long-term Trend', 'enabled': True},
            'P4': {'name': 'Market Breadth', 'enabled': False},  # Future implementation
            'P5': {'name': 'Price Momentum', 'enabled': True},
            'P6': {'name': 'Distribution Days', 'enabled': False}  # Future implementation
        }
        
    def load_market_data(self) -> pd.DataFrame:
        """Load market index data (QQQ, SPY, etc.)"""
        try:
            file_path = os.path.join(
                self.paths.get('source_market_data', ''),
                f"{self.market_index}.csv"
            )
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Market data file not found: {file_path}")
                
            df = pd.read_csv(file_path, index_col='Date', parse_dates=False)
            
            # Clean and standardize date index
            df.index = df.index.str.split(' ').str[0]
            df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return df.sort_index()
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Market data file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading market data: {str(e)}")
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required moving averages"""
        df = df.copy()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=50).mean()
        df['SMA_150'] = df['Close'].rolling(window=150, min_periods=150).mean()
        return df
    
    def gmi_p1_short_term_trend(self, df: pd.DataFrame) -> pd.Series:
        """GMI-P1: Close > 50-day SMA"""
        if not self.components_config['P1']['enabled']:
            return pd.Series(0, index=df.index)
        return (df['Close'] > df['SMA_50']).astype(int)
    
    def gmi_p2_short_term_momentum(self, df: pd.DataFrame) -> pd.Series:
        """GMI-P2: 50-day SMA rising (today > yesterday)"""
        if not self.components_config['P2']['enabled']:
            return pd.Series(0, index=df.index)
        return (df['SMA_50'] > df['SMA_50'].shift(1)).astype(int)
    
    def gmi_p3_long_term_trend(self, df: pd.DataFrame) -> pd.Series:
        """GMI-P3: Close > 150-day SMA"""
        if not self.components_config['P3']['enabled']:
            return pd.Series(0, index=df.index)
        return (df['Close'] > df['SMA_150']).astype(int)
    
    def gmi_p4_market_breadth(self, df: pd.DataFrame) -> pd.Series:
        """
        GMI-P4: Market breadth proxy (FUTURE IMPLEMENTATION)
        
        This will require T2108 data or similar breadth indicator.
        For now, using a simple momentum-based proxy.
        """
        if not self.components_config['P4']['enabled']:
            return pd.Series(0, index=df.index)
            
        # Placeholder: Simple momentum-based proxy
        # TODO: Replace with actual T2108 or breadth data
        momentum_5 = df['Close'].pct_change(5)
        momentum_10 = df['Close'].pct_change(10)
        return (momentum_5 > momentum_10).astype(int)
    
    def gmi_p5_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        """GMI-P5: Closed higher in at least 2 of last 3 days"""
        if not self.components_config['P5']['enabled']:
            return pd.Series(0, index=df.index)
        daily_change = (df['Close'] > df['Close'].shift(1)).astype(int)
        return (daily_change.rolling(window=3, min_periods=3).sum() >= 2).astype(int)
    
    def gmi_p6_distribution_days(self, df: pd.DataFrame) -> pd.Series:
        """
        GMI-P6: No distribution days in last 10 sessions (FUTURE IMPLEMENTATION)
        
        Distribution day: Close down >1% on higher volume than previous day
        """
        if not self.components_config['P6']['enabled']:
            return pd.Series(0, index=df.index)
            
        # Placeholder implementation
        pct_change = df['Close'].pct_change()
        volume_higher = df['Volume'] > df['Volume'].shift(1)
        distribution_day = (pct_change < -0.01) & volume_higher
        
        # Check if any distribution days in last 10 sessions
        distribution_in_window = distribution_day.rolling(window=10, min_periods=10).sum()
        return (distribution_in_window == 0).astype(int)
    
    def calculate_daily_gmi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily GMI score (0-6, currently 0-4)"""
        df = df.copy()
        
        # Calculate all components
        df['GMI_P1'] = self.gmi_p1_short_term_trend(df)
        df['GMI_P2'] = self.gmi_p2_short_term_momentum(df)
        df['GMI_P3'] = self.gmi_p3_long_term_trend(df)
        df['GMI_P4'] = self.gmi_p4_market_breadth(df)
        df['GMI_P5'] = self.gmi_p5_price_momentum(df)
        df['GMI_P6'] = self.gmi_p6_distribution_days(df)
        
        # Sum enabled components for daily GMI score
        enabled_components = []
        for component in ['GMI_P1', 'GMI_P2', 'GMI_P3', 'GMI_P4', 'GMI_P5', 'GMI_P6']:
            component_id = component.split('_')[1]
            if self.components_config[component_id]['enabled']:
                enabled_components.append(component)
        
        df['GMI_Score'] = df[enabled_components].sum(axis=1)
        
        return df
    
    def determine_gmi_signal(self, gmi_scores: pd.Series) -> Tuple[str, pd.Series]:
        """
        Determine GMI signal based on confirmation rule
        
        Args:
            gmi_scores: Series of daily GMI scores
            
        Returns:
            Tuple of (current_signal, signal_series)
        """
        signals = pd.Series(index=gmi_scores.index, dtype=str)
        signals.iloc[:] = 'NEUTRAL'
        
        # Apply confirmation rule
        for i in range(self.confirmation_days - 1, len(gmi_scores)):
            # Check if last N days meet the threshold criteria
            recent_scores = gmi_scores.iloc[i - self.confirmation_days + 1:i + 1]
            
            # Green signal: GMI > threshold for confirmation_days consecutive days
            if all(score > self.threshold for score in recent_scores):
                signals.iloc[i] = 'GREEN'
            # Red signal: GMI < threshold for confirmation_days consecutive days  
            elif all(score < self.threshold for score in recent_scores):
                signals.iloc[i] = 'RED'
            else:
                signals.iloc[i] = 'NEUTRAL'
        
        current_signal = signals.iloc[-1] if len(signals) > 0 else 'NEUTRAL'
        return current_signal, signals
    
    def run_gmi_analysis(self) -> Dict:
        """
        Run complete GMI analysis
        
        Returns:
            Dictionary containing GMI results
        """
        try:
            # Load market data
            market_df = self.load_market_data()
            
            if len(market_df) < 150:  # Need at least 150 days for long-term SMA
                raise ValueError(f"Insufficient data: {len(market_df)} days. Need at least 150 days.")
            
            # Calculate moving averages
            market_df = self.calculate_moving_averages(market_df)
            
            # Calculate daily GMI scores
            market_df = self.calculate_daily_gmi(market_df)
            
            # Determine signals
            current_signal, signal_series = self.determine_gmi_signal(market_df['GMI_Score'])
            market_df['GMI_Signal'] = signal_series
            
            # Get latest values
            latest_data = market_df.iloc[-1]
            
            # Count enabled components for max score
            max_score = sum(1 for comp in self.components_config.values() if comp['enabled'])
            
            results = {
                'success': True,
                'current_signal': current_signal,
                'current_score': int(latest_data['GMI_Score']),
                'max_score': max_score,
                'current_date': latest_data.name.strftime('%Y-%m-%d'),
                'market_index': self.market_index,
                'threshold': self.threshold,
                'confirmation_days': self.confirmation_days,
                'components': {
                    'GMI_P1_trend': bool(latest_data['GMI_P1']),
                    'GMI_P2_momentum': bool(latest_data['GMI_P2']), 
                    'GMI_P3_long_trend': bool(latest_data['GMI_P3']),
                    'GMI_P4_breadth': bool(latest_data['GMI_P4']),
                    'GMI_P5_price_momentum': bool(latest_data['GMI_P5']),
                    'GMI_P6_distribution': bool(latest_data['GMI_P6'])
                },
                'components_enabled': {
                    comp_id: comp_config['enabled'] 
                    for comp_id, comp_config in self.components_config.items()
                },
                'raw_data': market_df,
                'signal_history': signal_series.tail(10).to_dict()
            }
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': f"GMI analysis failed: {str(e)}",
                'current_signal': 'RED',  # Default to defensive
                'current_score': 0,
                'max_score': 0
            }
    
    def is_market_favorable(self) -> bool:
        """
        Quick method to check if market conditions are favorable for taking positions
        
        Returns:
            True if GMI signal is GREEN, False otherwise
        """
        try:
            results = self.run_gmi_analysis()
            return results.get('current_signal') == 'GREEN'
        except:
            return False
    
    def get_market_status_summary(self) -> str:
        """Get human-readable market status summary"""
        try:
            results = self.run_gmi_analysis()
            
            if not results.get('success', False):
                return f"GMI Analysis Error: {results.get('error', 'Unknown error')}"
            
            signal = results['current_signal']
            score = results['current_score']
            max_score = results['max_score']
            date = results['current_date']
            
            # Signal indicator
            signal_indicator = {
                'GREEN': '🟢',
                'RED': '🔴',
                'NEUTRAL': '🟡'
            }.get(signal, '❓')
            
            summary = f"""
GMI Market Analysis Summary ({date}):
═══════════════════════════════════════
Market Signal: {signal_indicator} {signal}
GMI Score: {score}/{max_score} (Threshold: {self.threshold})
Market Index: {self.market_index}
Confirmation Period: {self.confirmation_days} days

Component Breakdown:
"""
            
            components = results['components']
            components_enabled = results['components_enabled']
            component_names = {
                'GMI_P1_trend': 'Short-term Trend (P1)',
                'GMI_P2_momentum': 'Trend Momentum (P2)',
                'GMI_P3_long_trend': 'Long-term Trend (P3)',
                'GMI_P4_breadth': 'Market Breadth (P4)',
                'GMI_P5_price_momentum': 'Price Momentum (P5)',
                'GMI_P6_distribution': 'No Distribution (P6)'
            }
            
            for comp_key, comp_value in components.items():
                comp_id = comp_key.split('_')[1]
                enabled = components_enabled.get(comp_id, False)
                name = component_names.get(comp_key, comp_key)
                
                if enabled:
                    status = '✓' if comp_value else '✗'
                    summary += f"- {name}: {status}\n"
                else:
                    summary += f"- {name}: ⏸️ (Not implemented)\n"
            
            # Recommendation
            if signal == 'GREEN':
                recommendation = '🟢 FAVORABLE - Consider taking positions'
            elif signal == 'RED':
                recommendation = '🔴 DEFENSIVE - Avoid new long positions'
            else:
                recommendation = '🟡 NEUTRAL - Wait for confirmation'
                
            summary += f"\nRecommendation: {recommendation}"
            
            return summary.strip()
            
        except Exception as e:
            return f"Error generating market summary: {str(e)}"
    
    def enable_component(self, component_id: str, enabled: bool = True):
        """Enable or disable a GMI component"""
        if component_id in self.components_config:
            self.components_config[component_id]['enabled'] = enabled
        else:
            raise ValueError(f"Invalid component ID: {component_id}")
    
    def get_component_status(self) -> Dict:
        """Get current component configuration"""
        return {
            comp_id: {
                'name': comp_config['name'],
                'enabled': comp_config['enabled']
            }
            for comp_id, comp_config in self.components_config.items()
        }

def calculate_gmi_for_index(market_index: str = 'QQQ', paths: Dict[str, str] = None, 
                           threshold: int = 3, confirmation_days: int = 2) -> Dict:
    """
    Convenience function to calculate GMI for a specific market index
    
    Args:
        market_index: Index symbol (QQQ, SPY, etc.)
        paths: Data paths dictionary
        threshold: GMI score threshold for signals
        confirmation_days: Days required for signal confirmation
        
    Returns:
        GMI analysis results
    """
    gmi_calc = GMICalculator(
        market_index=market_index, 
        paths=paths,
        threshold=threshold,
        confirmation_days=confirmation_days
    )
    return gmi_calc.run_gmi_analysis()

if __name__ == "__main__":
    # Test the GMI calculator
    print("Testing GMI Calculator...")
    
    # Basic test with QQQ
    test_paths = {
        'source_market_data': '../downloadData_v1/data/market_data/daily/'
    }
    
    gmi_calc = GMICalculator(market_index='QQQ', paths=test_paths)
    results = gmi_calc.run_gmi_analysis()
    
    if results.get('success', False):
        print(gmi_calc.get_market_status_summary())
    else:
        print(f"Test failed: {results.get('error')}")