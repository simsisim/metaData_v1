"""
ADL (Accumulation/Distribution Line) Screener
==============================================

Detects accumulation vs distribution patterns using volume-weighted price movements.
The ADL is a cumulative indicator that uses volume and price to assess whether 
a stock is being accumulated or distributed.

ADL Formula: AD = AD + ((Close - Low) - (High - Close)) / (High - Low) * Volume

Key signals:
- ADL trending up while price declining: Potential bullish divergence (accumulation)
- ADL trending down while price rising: Potential bearish divergence (distribution)
- ADL breakouts above previous highs: Strong accumulation signal
- ADL breakdowns below previous lows: Strong distribution signal

Based on the ADL methodology from TradingView and technical analysis resources.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class ADLScreener:
    """
    ADL (Accumulation/Distribution Line) screener for detecting accumulation/distribution patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_adl_screener = config.get('enable_adl_screener', True)
        self.timeframe = config.get('timeframe', 'daily')
        
        # ADL screener specific configuration
        self.adl_config = config.get('adl_screener', {})
        
        # ADL calculation parameters
        self.adl_lookback_period = self.adl_config.get('adl_lookback_period', 50)
        self.divergence_period = self.adl_config.get('divergence_period', 20)
        self.breakout_period = self.adl_config.get('breakout_period', 30)
        
        # Signal thresholds
        self.min_divergence_strength = self.adl_config.get('min_divergence_strength', 0.7)
        self.min_breakout_strength = self.adl_config.get('min_breakout_strength', 1.2)
        self.min_volume_avg = self.adl_config.get('min_volume_avg', 100_000)
        self.min_price = self.adl_config.get('min_price', 5.0)
        
        # Output configuration
        self.output_dir = config.get('adl_output_dir', 'results/screeners/adl')
        self.save_individual_files = self.adl_config.get('save_individual_files', True)
        
        logger.info(f"ADL Screener initialized (enabled: {self.enable_adl_screener})")

    def run_adl_screening(self, batch_data: Dict[str, pd.DataFrame], 
                         ticker_info: Optional[pd.DataFrame] = None,
                         rs_data: Optional[Dict] = None,
                         batch_info: Dict[str, Any] = None) -> List[Dict]:
        """
        Run ADL screening to detect accumulation/distribution patterns
        
        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            ticker_info: DataFrame with ticker information  
            rs_data: Dictionary with RS data for additional scoring
            batch_info: Optional batch processing information
            
        Returns:
            List of screening results
        """
        if not self.enable_adl_screener:
            logger.info("ADL screener disabled")
            return []
            
        if not batch_data:
            logger.warning("No data provided for ADL screening")
            return []
        
        logger.info(f"Running ADL screening on {len(batch_data)} tickers")
        
        all_results = []
        component_results = {
            'bullish_divergence': [],
            'bearish_divergence': [],
            'adl_breakout': [],
            'adl_breakdown': []
        }
        
        try:
            # Pre-filter tickers based on basic criteria
            filtered_tickers = self._apply_base_filters(batch_data)
            logger.info(f"Filtered to {len(filtered_tickers)} tickers meeting base criteria")
            
            # Process each ticker for ADL signals
            for ticker, data in filtered_tickers.items():
                try:
                    # Calculate ADL line
                    adl_line = self._calculate_adl(data)
                    if adl_line is None or len(adl_line) < self.divergence_period:
                        continue
                    
                    # Add ADL to data for analysis
                    data_with_adl = data.copy()
                    data_with_adl['ADL'] = adl_line
                    
                    # Check for divergence signals
                    bullish_div = self._check_bullish_divergence(ticker, data_with_adl, rs_data)
                    if bullish_div:
                        component_results['bullish_divergence'].append(bullish_div)
                        all_results.append(bullish_div)
                    
                    bearish_div = self._check_bearish_divergence(ticker, data_with_adl, rs_data)
                    if bearish_div:
                        component_results['bearish_divergence'].append(bearish_div)
                        all_results.append(bearish_div)
                    
                    # Check for breakout/breakdown signals
                    breakout = self._check_adl_breakout(ticker, data_with_adl, rs_data)
                    if breakout:
                        component_results['adl_breakout'].append(breakout)
                        all_results.append(breakout)
                    
                    breakdown = self._check_adl_breakdown(ticker, data_with_adl, rs_data)
                    if breakdown:
                        component_results['adl_breakdown'].append(breakdown)
                        all_results.append(breakdown)
                        
                except Exception as e:
                    logger.warning(f"Error processing {ticker} in ADL screening: {e}")
                    continue
            
            # Save results if enabled
            if self.save_individual_files:
                self._save_component_results(component_results)
            
            logger.info(f"ADL screening completed: {len(all_results)} total signals")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in ADL screening: {e}")
            return []

    def _apply_base_filters(self, batch_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply base filters for ADL analysis
        
        Args:
            batch_data: Dictionary of ticker data
            
        Returns:
            Filtered dictionary of ticker data
        """
        filtered_data = {}
        
        for ticker, data in batch_data.items():
            try:
                # Basic data validation
                if data.empty or len(data) < self.adl_lookback_period:
                    continue
                
                # Price filter
                current_price = data['Close'].iloc[-1]
                if current_price < self.min_price:
                    continue
                
                # Volume filter
                avg_volume = data['Volume'].tail(20).mean()
                if avg_volume < self.min_volume_avg:
                    continue
                
                # Exclude obvious funds/ETFs
                if any(fund_suffix in ticker.upper() for fund_suffix in ['ETF', 'QQQ', 'SPY', 'IWM', 'XL']):
                    continue
                
                filtered_data[ticker] = data
                
            except Exception as e:
                logger.warning(f"Error filtering ticker {ticker}: {e}")
                continue
        
        return filtered_data

    def _calculate_adl(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculate Accumulation/Distribution Line
        
        Formula: AD = AD + ((Close - Low) - (High - Close)) / (High - Low) * Volume
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            ADL Series or None if calculation fails
        """
        try:
            # Ensure we have required columns
            required_cols = ['High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                return None
            
            # Calculate Money Flow Multiplier (MFM)
            # MFM = ((Close - Low) - (High - Close)) / (High - Low)
            high_low_diff = data['High'] - data['Low']
            
            # Avoid division by zero for doji/narrow range bars
            high_low_diff = high_low_diff.replace(0, np.nan)
            
            close_position = ((data['Close'] - data['Low']) - (data['High'] - data['Close']))
            mfm = close_position / high_low_diff
            
            # Handle NaN values (doji bars) - assume neutral (0)
            mfm = mfm.fillna(0)
            
            # Calculate Money Flow Volume (MFV)
            # MFV = MFM * Volume
            mfv = mfm * data['Volume']
            
            # Calculate ADL (cumulative sum of MFV)
            adl = mfv.cumsum()
            
            return adl
            
        except Exception as e:
            logger.warning(f"Error calculating ADL: {e}")
            return None

    def _check_bullish_divergence(self, ticker: str, data: pd.DataFrame, 
                                 rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check for bullish divergence: Price making lower lows while ADL making higher lows
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with ADL column
            rs_data: RS data for additional scoring
            
        Returns:
            Signal result or None
        """
        try:
            if len(data) < self.divergence_period:
                return None
                
            # Analyze recent period for divergence
            recent_data = data.tail(self.divergence_period)
            
            # Find price lows and ADL lows
            price_lows = self._find_lows(recent_data['Close'])
            adl_lows = self._find_lows(recent_data['ADL'])
            
            if len(price_lows) < 2 or len(adl_lows) < 2:
                return None
            
            # Check for divergence pattern
            latest_price_low = price_lows[-1]
            prev_price_low = price_lows[-2]
            latest_adl_low = adl_lows[-1]
            prev_adl_low = adl_lows[-2]
            
            # Bullish divergence: price lower low, ADL higher low
            price_declining = latest_price_low['value'] < prev_price_low['value']
            adl_improving = latest_adl_low['value'] > prev_adl_low['value']
            
            if price_declining and adl_improving:
                # Calculate divergence strength
                price_decline_pct = ((prev_price_low['value'] - latest_price_low['value']) / prev_price_low['value']) * 100
                adl_improvement_pct = ((latest_adl_low['value'] - prev_adl_low['value']) / abs(prev_adl_low['value'])) * 100
                
                divergence_strength = (price_decline_pct + adl_improvement_pct) / 2
                
                if divergence_strength >= self.min_divergence_strength:
                    latest = data.iloc[-1]
                    rs_score = self._get_rs_score(ticker, rs_data) if rs_data else None
                    
                    return {
                        'ticker': ticker,
                        'signal_date': latest.name,
                        'signal_type': 'adl_bullish_divergence',
                        'screen_type': 'adl_divergence',
                        'price': latest['Close'],
                        'volume': latest['Volume'],
                        'adl_value': latest['ADL'],
                        'divergence_strength': divergence_strength,
                        'price_decline_pct': price_decline_pct,
                        'adl_improvement_pct': adl_improvement_pct,
                        'rs_score': rs_score,
                        'strength': self._calculate_signal_strength(divergence_strength),
                        'raw_data': latest.to_dict()
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking bullish divergence for {ticker}: {e}")
            return None

    def _check_bearish_divergence(self, ticker: str, data: pd.DataFrame, 
                                 rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check for bearish divergence: Price making higher highs while ADL making lower highs
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with ADL column
            rs_data: RS data for additional scoring
            
        Returns:
            Signal result or None
        """
        try:
            if len(data) < self.divergence_period:
                return None
                
            # Analyze recent period for divergence
            recent_data = data.tail(self.divergence_period)
            
            # Find price highs and ADL highs
            price_highs = self._find_highs(recent_data['Close'])
            adl_highs = self._find_highs(recent_data['ADL'])
            
            if len(price_highs) < 2 or len(adl_highs) < 2:
                return None
            
            # Check for divergence pattern
            latest_price_high = price_highs[-1]
            prev_price_high = price_highs[-2]
            latest_adl_high = adl_highs[-1]
            prev_adl_high = adl_highs[-2]
            
            # Bearish divergence: price higher high, ADL lower high
            price_advancing = latest_price_high['value'] > prev_price_high['value']
            adl_weakening = latest_adl_high['value'] < prev_adl_high['value']
            
            if price_advancing and adl_weakening:
                # Calculate divergence strength
                price_advance_pct = ((latest_price_high['value'] - prev_price_high['value']) / prev_price_high['value']) * 100
                adl_weakness_pct = ((prev_adl_high['value'] - latest_adl_high['value']) / abs(prev_adl_high['value'])) * 100
                
                divergence_strength = (price_advance_pct + adl_weakness_pct) / 2
                
                if divergence_strength >= self.min_divergence_strength:
                    latest = data.iloc[-1]
                    rs_score = self._get_rs_score(ticker, rs_data) if rs_data else None
                    
                    return {
                        'ticker': ticker,
                        'signal_date': latest.name,
                        'signal_type': 'adl_bearish_divergence',
                        'screen_type': 'adl_divergence',
                        'price': latest['Close'],
                        'volume': latest['Volume'],
                        'adl_value': latest['ADL'],
                        'divergence_strength': divergence_strength,
                        'price_advance_pct': price_advance_pct,
                        'adl_weakness_pct': adl_weakness_pct,
                        'rs_score': rs_score,
                        'strength': self._calculate_signal_strength(divergence_strength),
                        'raw_data': latest.to_dict()
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking bearish divergence for {ticker}: {e}")
            return None

    def _check_adl_breakout(self, ticker: str, data: pd.DataFrame, 
                           rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check for ADL breakout above previous highs (strong accumulation signal)
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with ADL column
            rs_data: RS data for additional scoring
            
        Returns:
            Signal result or None
        """
        try:
            if len(data) < self.breakout_period:
                return None
            
            latest = data.iloc[-1]
            recent_data = data.tail(self.breakout_period)
            
            # Find highest ADL in the lookback period (excluding last bar)
            previous_period = recent_data.iloc[:-1]
            max_adl_previous = previous_period['ADL'].max()
            current_adl = latest['ADL']
            
            # Check if current ADL breaks above previous highs
            if current_adl > max_adl_previous:
                breakout_strength = (current_adl - max_adl_previous) / abs(max_adl_previous)
                
                if breakout_strength >= self.min_breakout_strength:
                    # Additional validation: volume confirmation
                    avg_volume = data['Volume'].tail(10).mean()
                    volume_confirmation = latest['Volume'] >= avg_volume
                    
                    if volume_confirmation:
                        rs_score = self._get_rs_score(ticker, rs_data) if rs_data else None
                        
                        return {
                            'ticker': ticker,
                            'signal_date': latest.name,
                            'signal_type': 'adl_breakout',
                            'screen_type': 'adl_breakout',
                            'price': latest['Close'],
                            'volume': latest['Volume'],
                            'adl_value': current_adl,
                            'previous_adl_high': max_adl_previous,
                            'breakout_strength': breakout_strength,
                            'volume_confirmation': volume_confirmation,
                            'rs_score': rs_score,
                            'strength': self._calculate_signal_strength(breakout_strength * 100),
                            'raw_data': latest.to_dict()
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking ADL breakout for {ticker}: {e}")
            return None

    def _check_adl_breakdown(self, ticker: str, data: pd.DataFrame, 
                            rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check for ADL breakdown below previous lows (strong distribution signal)
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with ADL column
            rs_data: RS data for additional scoring
            
        Returns:
            Signal result or None
        """
        try:
            if len(data) < self.breakout_period:
                return None
            
            latest = data.iloc[-1]
            recent_data = data.tail(self.breakout_period)
            
            # Find lowest ADL in the lookback period (excluding last bar)
            previous_period = recent_data.iloc[:-1]
            min_adl_previous = previous_period['ADL'].min()
            current_adl = latest['ADL']
            
            # Check if current ADL breaks below previous lows
            if current_adl < min_adl_previous:
                breakdown_strength = abs(current_adl - min_adl_previous) / abs(min_adl_previous)
                
                if breakdown_strength >= self.min_breakout_strength:
                    # Additional validation: volume confirmation
                    avg_volume = data['Volume'].tail(10).mean()
                    volume_confirmation = latest['Volume'] >= avg_volume
                    
                    if volume_confirmation:
                        rs_score = self._get_rs_score(ticker, rs_data) if rs_data else None
                        
                        return {
                            'ticker': ticker,
                            'signal_date': latest.name,
                            'signal_type': 'adl_breakdown',
                            'screen_type': 'adl_breakdown',
                            'price': latest['Close'],
                            'volume': latest['Volume'],
                            'adl_value': current_adl,
                            'previous_adl_low': min_adl_previous,
                            'breakdown_strength': breakdown_strength,
                            'volume_confirmation': volume_confirmation,
                            'rs_score': rs_score,
                            'strength': self._calculate_signal_strength(breakdown_strength * 100),
                            'raw_data': latest.to_dict()
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking ADL breakdown for {ticker}: {e}")
            return None

    def _find_lows(self, series: pd.Series, window: int = 5) -> List[Dict]:
        """
        Find local lows in a price/ADL series
        
        Args:
            series: Price or ADL series
            window: Window size for local extrema detection
            
        Returns:
            List of low points with index and value
        """
        lows = []
        
        for i in range(window, len(series) - window):
            current_value = series.iloc[i]
            left_window = series.iloc[i-window:i]
            right_window = series.iloc[i+1:i+window+1]
            
            # Check if current value is lower than surrounding values
            if current_value <= left_window.min() and current_value <= right_window.min():
                lows.append({
                    'index': i,
                    'date': series.index[i],
                    'value': current_value
                })
        
        return lows

    def _find_highs(self, series: pd.Series, window: int = 5) -> List[Dict]:
        """
        Find local highs in a price/ADL series
        
        Args:
            series: Price or ADL series
            window: Window size for local extrema detection
            
        Returns:
            List of high points with index and value
        """
        highs = []
        
        for i in range(window, len(series) - window):
            current_value = series.iloc[i]
            left_window = series.iloc[i-window:i]
            right_window = series.iloc[i+1:i+window+1]
            
            # Check if current value is higher than surrounding values
            if current_value >= left_window.max() and current_value >= right_window.max():
                highs.append({
                    'index': i,
                    'date': series.index[i],
                    'value': current_value
                })
        
        return highs

    def _get_rs_score(self, ticker: str, rs_data: Optional[Dict]) -> Optional[float]:
        """
        Extract RS score for a ticker from RS data
        
        Args:
            ticker: Ticker symbol
            rs_data: RS data dictionary
            
        Returns:
            RS score or None if not available
        """
        try:
            if not rs_data or 'daily' not in rs_data:
                return None
            
            daily_rs = rs_data['daily']
            if 'period_1' in daily_rs:
                period_data = daily_rs['period_1']
                if ticker in period_data.index:
                    return period_data.loc[ticker, 'rs_percentile_1']
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting RS score for {ticker}: {e}")
            return None

    def _calculate_signal_strength(self, strength_value: float) -> str:
        """
        Calculate signal strength category based on divergence/breakout strength
        
        Args:
            strength_value: Numerical strength value
            
        Returns:
            Signal strength category
        """
        if strength_value >= 5.0:
            return 'very_strong'
        elif strength_value >= 3.0:
            return 'strong'
        elif strength_value >= 1.0:
            return 'moderate'
        else:
            return 'weak'

    def _save_component_results(self, component_results: Dict[str, List]):
        """Save individual component results to files"""
        try:
            # Create output directory
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save each component's results
            for component_name, results in component_results.items():
                if results:
                    filename = f"adl_{component_name}_{self.timeframe}_{timestamp}.csv"
                    filepath = output_dir / filename
                    
                    # Convert to DataFrame and save
                    if isinstance(results, list) and results:
                        df = pd.DataFrame(results)
                        df.to_csv(filepath, index=False)
                        logger.info(f"Saved {component_name} results: {len(df)} signals to {filepath}")
                    
        except Exception as e:
            logger.error(f"Error saving component results: {e}")


def run_adl_screener(batch_data: Dict[str, pd.DataFrame], 
                    config: Dict[str, Any],
                    ticker_info: Optional[pd.DataFrame] = None,
                    rs_data: Optional[Dict] = None) -> List[Dict]:
    """
    Main entry point for ADL screening
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        config: Configuration dictionary
        ticker_info: Optional ticker information DataFrame
        rs_data: Optional RS data for enhanced screening
        
    Returns:
        List of screening results
    """
    screener = ADLScreener(config)
    return screener.run_adl_screening(batch_data, ticker_info, rs_data)


# Export main functions
__all__ = ['ADLScreener', 'run_adl_screener']