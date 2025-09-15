"""
Gold Launch Pad Screener
========================

Implementation of the Golden Launch Pad strategy from TradingView.
Identifies stocks with tightly grouped, bullishly stacked moving averages
with positive slope where price is near the MA cluster.

Strategy Components:
1. MAs Are Tightly Grouped (Z-score spread analysis)
2. MAs Are Bullishly Stacked (ascending order)
3. All MAs Have Positive Slope
4. Price Is Above Fastest MA
5. Price Is Near MA Cluster

Based on: https://www.tradingview.com/script/DvE0wDfI-Golden-Launch-Pad/
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


class GoldLaunchPadScreener:
    """
    Gold Launch Pad screener implementation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_gold_launch_pad = config.get('enable_gold_launch_pad', True)
        self.timeframe = config.get('timeframe', 'daily')
        
        # Gold Launch Pad specific configuration
        self.glp_config = config.get('gold_launch_pad', {})
        
        # Moving Average Configuration
        self.ma_periods = self.glp_config.get('ma_periods', [10, 20, 50])
        self.ma_type = self.glp_config.get('ma_type', 'EMA')  # EMA, SMA, WMA
        
        # Z-score Analysis Configuration
        self.zscore_window = self.glp_config.get('zscore_window', 50)
        self.max_spread_threshold = self.glp_config.get('max_spread_threshold', 1.0)
        
        # Slope Analysis Configuration
        self.slope_lookback_pct = self.glp_config.get('slope_lookback_pct', 0.3)  # 30% of MA period
        self.min_slope_threshold = self.glp_config.get('min_slope_threshold', 0.0001)
        
        # Price Proximity Configuration
        self.price_proximity_stdv = self.glp_config.get('price_proximity_stdv', 2.0)
        self.proximity_window = self.glp_config.get('proximity_window', 20)
        
        # Base Filters
        self.min_price = self.glp_config.get('min_price', 5.0)
        self.min_volume = self.glp_config.get('min_volume', 100000)
        self.min_data_length = max(self.ma_periods) + self.zscore_window
        
        # Output configuration
        self.save_individual_files = self.glp_config.get('save_individual_files', True)
        
        logger.info(f"Gold Launch Pad Screener initialized (enabled: {self.enable_gold_launch_pad})")
        logger.info(f"MA periods: {self.ma_periods}, MA type: {self.ma_type}")

    def _calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages based on configuration"""
        ma_data = data.copy()
        
        for period in self.ma_periods:
            if self.ma_type == 'EMA':
                ma_data[f'MA_{period}'] = data['Close'].ewm(span=period).mean()
            elif self.ma_type == 'SMA':
                ma_data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
            elif self.ma_type == 'WMA':
                # Weighted moving average
                weights = np.arange(1, period + 1)
                ma_data[f'MA_{period}'] = data['Close'].rolling(window=period).apply(
                    lambda x: np.average(x, weights=weights), raw=True
                )
            
        return ma_data

    def _calculate_zscore_spread(self, ma_data: pd.DataFrame) -> pd.Series:
        """Calculate Z-score spread of moving averages"""
        ma_columns = [f'MA_{period}' for period in self.ma_periods]
        ma_values = ma_data[ma_columns].values
        
        spreads = []
        for i in range(len(ma_data)):
            if i < self.zscore_window:
                spreads.append(np.nan)
                continue
                
            # Get window of MA values for Z-score calculation
            window_data = ma_values[i-self.zscore_window:i+1]
            
            # Calculate Z-scores for current MAs
            current_mas = ma_values[i]
            window_mean = np.nanmean(window_data, axis=0)
            window_std = np.nanstd(window_data, axis=0)
            
            # Avoid division by zero
            window_std[window_std == 0] = 1e-8
            
            z_scores = (current_mas - window_mean) / window_std
            
            # Calculate spread as difference between max and min Z-scores
            spread = np.nanmax(z_scores) - np.nanmin(z_scores)
            spreads.append(spread)
            
        return pd.Series(spreads, index=ma_data.index)

    def _check_bullish_stacking(self, ma_data: pd.DataFrame) -> pd.Series:
        """Check if MAs are bullishly stacked (fastest > slowest)"""
        ma_columns = [f'MA_{period}' for period in sorted(self.ma_periods)]
        
        bullish_stack = pd.Series(True, index=ma_data.index)
        
        for i in range(len(ma_columns) - 1):
            faster_ma = ma_columns[i]
            slower_ma = ma_columns[i + 1]
            bullish_stack &= (ma_data[faster_ma] > ma_data[slower_ma])
            
        return bullish_stack

    def _check_positive_slopes(self, ma_data: pd.DataFrame) -> pd.Series:
        """Check if all MAs have positive slope"""
        all_positive_slopes = pd.Series(True, index=ma_data.index)
        
        for period in self.ma_periods:
            ma_col = f'MA_{period}'
            lookback = max(1, int(period * self.slope_lookback_pct))
            
            # Calculate slope using linear regression
            slopes = []
            for i in range(len(ma_data)):
                if i < lookback:
                    slopes.append(False)
                    continue
                    
                y_values = ma_data[ma_col].iloc[i-lookback:i+1].values
                x_values = np.arange(len(y_values))
                
                if len(y_values) < 2 or np.any(pd.isna(y_values)):
                    slopes.append(False)
                    continue
                
                try:
                    slope, _, _, _, _ = stats.linregress(x_values, y_values)
                    slopes.append(slope > self.min_slope_threshold)
                except:
                    slopes.append(False)
            
            ma_positive_slope = pd.Series(slopes, index=ma_data.index)
            all_positive_slopes &= ma_positive_slope
            
        return all_positive_slopes

    def _check_price_above_fastest_ma(self, ma_data: pd.DataFrame) -> pd.Series:
        """Check if price is above the fastest (shortest period) MA"""
        fastest_ma = f'MA_{min(self.ma_periods)}'
        return ma_data['Close'] > ma_data[fastest_ma]

    def _check_price_near_cluster(self, ma_data: pd.DataFrame) -> pd.Series:
        """Check if price is near the MA cluster"""
        ma_columns = [f'MA_{period}' for period in self.ma_periods]
        
        # Calculate MA cluster average
        ma_cluster_avg = ma_data[ma_columns].mean(axis=1)
        
        # Calculate price volatility for proximity threshold
        price_volatility = ma_data['Close'].rolling(window=self.proximity_window).std()
        proximity_threshold = price_volatility * self.price_proximity_stdv
        
        # Check if price is within threshold distance of cluster
        price_distance = abs(ma_data['Close'] - ma_cluster_avg)
        near_cluster = price_distance <= proximity_threshold
        
        return near_cluster

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
            
        logger.info(f"Gold Launch Pad base filters: {len(filtered_data)}/{len(batch_data)} tickers passed")
        return filtered_data

    def _run_gold_launch_pad_screening(self, batch_data: Dict[str, pd.DataFrame],
                                     ticker_info: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Run the Gold Launch Pad screening logic"""
        results = []
        
        for ticker, data in batch_data.items():
            try:
                # Calculate moving averages
                ma_data = self._calculate_moving_averages(data)
                
                # Calculate Z-score spread
                zscore_spread = self._calculate_zscore_spread(ma_data)
                
                # Check all conditions
                tightly_grouped = zscore_spread <= self.max_spread_threshold
                bullish_stacked = self._check_bullish_stacking(ma_data)
                positive_slopes = self._check_positive_slopes(ma_data)
                price_above_fastest = self._check_price_above_fastest_ma(ma_data)
                price_near_cluster = self._check_price_near_cluster(ma_data)
                
                # Combine all conditions
                launch_pad_signal = (
                    tightly_grouped & 
                    bullish_stacked & 
                    positive_slopes & 
                    price_above_fastest & 
                    price_near_cluster
                )
                
                # Get latest signal
                if launch_pad_signal.iloc[-1] and not pd.isna(zscore_spread.iloc[-1]):
                    latest_data = ma_data.iloc[-1]
                    
                    # Calculate signal strength based on how well conditions are met
                    spread_score = max(0, 1 - (zscore_spread.iloc[-1] / self.max_spread_threshold))
                    
                    # Calculate MA cluster average for reference
                    ma_columns = [f'MA_{period}' for period in self.ma_periods]
                    cluster_avg = latest_data[ma_columns].mean()
                    
                    # Calculate price distance from cluster
                    price_distance_pct = abs(latest_data['Close'] - cluster_avg) / cluster_avg * 100
                    
                    results.append({
                        'ticker': ticker,
                        'screen_type': 'gold_launch_pad',
                        'signal_type': 'Gold Launch Pad',
                        'signal_date': latest_data.name.strftime('%Y-%m-%d'),
                        'price': latest_data['Close'],
                        'volume': latest_data['Volume'],
                        'zscore_spread': zscore_spread.iloc[-1],
                        'spread_score': spread_score,
                        'cluster_avg': cluster_avg,
                        'price_distance_pct': price_distance_pct,
                        'ma_periods': str(self.ma_periods),
                        'ma_type': self.ma_type,
                        'strength': 'Strong' if spread_score > 0.7 else 'Moderate',
                        'timeframe': self.timeframe
                    })
                    
            except Exception as e:
                logger.warning(f"Gold Launch Pad screening failed for {ticker}: {e}")
                continue
                
        return results

    def run_gold_launch_pad_screening(self, batch_data: Dict[str, pd.DataFrame],
                                    ticker_info: Optional[pd.DataFrame] = None,
                                    batch_info: Optional[Dict] = None) -> List[Dict]:
        """
        Run Gold Launch Pad screening on batch data
        
        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            ticker_info: Optional DataFrame with ticker information
            batch_info: Optional batch processing information
            
        Returns:
            List of Gold Launch Pad screening results
        """
        if not self.enable_gold_launch_pad:
            logger.info("Gold Launch Pad screening disabled")
            return []
            
        if not batch_data:
            logger.warning("No data provided for Gold Launch Pad screening")
            return []
        
        logger.info(f"Running Gold Launch Pad screening on {len(batch_data)} tickers")
        
        # Apply base filters
        filtered_data = self._apply_base_filters(batch_data, ticker_info)
        if not filtered_data:
            logger.warning("No tickers passed Gold Launch Pad base filters")
            return []
        
        # Run Gold Launch Pad screening
        results = self._run_gold_launch_pad_screening(filtered_data, ticker_info)
        
        logger.info(f"Gold Launch Pad screening completed: {len(results)} signals found")
        return results


def run_gold_launch_pad_screener(batch_data: Dict[str, pd.DataFrame], 
                               config: Dict[str, Any],
                               ticker_info: Optional[pd.DataFrame] = None,
                               batch_info: Optional[Dict] = None) -> List[Dict]:
    """
    Standalone function to run Gold Launch Pad screening
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        config: Configuration dictionary with Gold Launch Pad parameters
        ticker_info: Optional DataFrame with ticker information
        batch_info: Optional batch processing information
        
    Returns:
        List of Gold Launch Pad screening results
    """
    screener = GoldLaunchPadScreener(config)
    return screener.run_gold_launch_pad_screening(batch_data, ticker_info, batch_info)