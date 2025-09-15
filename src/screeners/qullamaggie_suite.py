"""
Qullamaggie Suite Screener
==========================

Implementation of Qullamaggie's momentum screening methodology focusing on:
1. High relative strength (RS ≥ 97 on 1w/1m/3m/6m timeframes)
2. Perfect moving averages alignment (Price ≥ EMA10 ≥ SMA20 ≥ SMA50 ≥ SMA100 ≥ SMA200)
3. ATR relative strength ≥ 50 vs $1B+ universe
4. Price in upper half of 20-day range (≥ 50%)
5. Market cap ≥ $1B for liquidity

Results sorted by ATR extension to SMA50 with red/bold formatting for 7x+ and 11x+ extensions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class QullamaggieScreener:
    """
    Qullamaggie momentum screener with strict filtering criteria
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_qullamaggie_suite = config.get('enable_qullamaggie_suite', True)
        self.timeframe = config.get('timeframe', 'daily')
        
        # Qullamaggie suite specific configuration
        self.qullamaggie_config = config.get('qullamaggie_suite', {})
        
        # Core filtering parameters
        self.min_market_cap = self.qullamaggie_config.get('min_market_cap', 1_000_000_000)  # $1B
        self.min_price = self.qullamaggie_config.get('min_price', 5.0)
        self.rs_threshold = self.qullamaggie_config.get('rs_threshold', 97.0)  # Top 3%
        self.atr_rs_threshold = self.qullamaggie_config.get('atr_rs_threshold', 50.0)  # Top 50%
        self.range_position_threshold = self.qullamaggie_config.get('range_position_threshold', 0.5)  # 50%
        
        # Extension thresholds for formatting
        self.extension_warning_threshold = self.qullamaggie_config.get('extension_warning', 7.0)  # 7x
        self.extension_danger_threshold = self.qullamaggie_config.get('extension_danger', 11.0)   # 11x
        
        # Output configuration
        self.output_dir = config.get('qullamaggie_output_dir', 'results/screeners/qullamaggie_suite')
        self.save_individual_files = self.qullamaggie_config.get('save_individual_files', True)
        
        logger.info(f"Qullamaggie Suite Screener initialized (enabled: {self.enable_qullamaggie_suite})")

    def run_qullamaggie_screening(self, batch_data: Dict[str, pd.DataFrame], 
                                 ticker_info: Optional[pd.DataFrame] = None,
                                 rs_data: Optional[Dict] = None,
                                 atr_universe_data: Optional[Dict] = None,
                                 batch_info: Dict[str, Any] = None) -> List[Dict]:
        """
        Run Qullamaggie momentum screening with strict criteria
        
        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            ticker_info: DataFrame with ticker information (market cap, exchange, industry)
            rs_data: Dictionary with RS data for different timeframes/periods
            atr_universe_data: ATR data for $1B+ universe for ATR RS calculation
            batch_info: Optional batch processing information
            
        Returns:
            List of screening results sorted by ATR extension
        """
        if not self.enable_qullamaggie_suite:
            logger.info("Qullamaggie suite screening disabled")
            return []
            
        if not batch_data:
            logger.warning("No data provided for Qullamaggie suite screening")
            return []
        
        logger.info(f"Running Qullamaggie suite screening on {len(batch_data)} tickers")
        
        results = []
        
        try:
            # Calculate ATR universe rankings if data provided
            atr_universe_rankings = self._calculate_atr_universe_rankings(atr_universe_data) if atr_universe_data else {}
            
            for ticker, data in batch_data.items():
                try:
                    # Apply all Qullamaggie filters
                    qulla_result = self._apply_qullamaggie_filters(
                        ticker, data, ticker_info, rs_data, atr_universe_rankings
                    )
                    
                    if qulla_result:
                        results.append(qulla_result)
                        
                except Exception as e:
                    logger.warning(f"Error processing {ticker} in Qullamaggie screening: {e}")
                    continue
            
            # Sort by ATR extension to SMA50 (most extended first)
            results.sort(key=lambda x: x['atr_extension_sma50'], reverse=True)
            
            # Apply extension formatting
            for result in results:
                result.update(self._apply_extension_formatting(result['atr_extension_sma50']))
            
            # Save results if enabled
            if self.save_individual_files and results:
                self._save_screening_results(results)
            
            logger.info(f"Qullamaggie screening completed: {len(results)} stocks passed all criteria")
            return results
            
        except Exception as e:
            logger.error(f"Error in Qullamaggie suite screening: {e}")
            return []

    def _apply_qullamaggie_filters(self, ticker: str, data: pd.DataFrame,
                                  ticker_info: Optional[pd.DataFrame],
                                  rs_data: Optional[Dict],
                                  atr_universe_rankings: Dict) -> Optional[Dict]:
        """
        Apply all Qullamaggie filtering criteria to a single ticker
        
        Returns:
            Dictionary with screening result or None if filters not passed
        """
        # Check minimum data requirements
        min_data_length = 250  # Need ~1 year for SMA200
        if len(data) < min_data_length:
            # IPO handling - adjust requirements based on available data
            return self._handle_ipo_stock(ticker, data, ticker_info, rs_data, atr_universe_rankings)
        
        latest = data.iloc[-1]
        
        # 1. MARKET CAP FILTER
        if not self._check_market_cap_filter(ticker, ticker_info):
            return None
        
        # 2. RELATIVE STRENGTH FILTER (≥ 97 on at least one timeframe)
        rs_check_result = self._check_rs_filter(ticker, rs_data)
        if not rs_check_result['passed']:
            return None
        
        # 3. MOVING AVERAGES ALIGNMENT CHECK
        ma_alignment_result = self._check_moving_averages_alignment(data)
        if not ma_alignment_result['aligned']:
            return None
        
        # 4. ATR RS FILTER (≥ 50 vs $1B+ universe)
        atr_rs_result = self._check_atr_rs_filter(ticker, data, atr_universe_rankings)
        if not atr_rs_result['passed']:
            return None
        
        # 5. RANGE POSITION FILTER (≥ 50% of 20-day range)
        range_position_result = self._check_range_position_filter(data)
        if not range_position_result['passed']:
            return None
        
        # Calculate ATR extension to SMA50 for sorting
        atr_extension = self._calculate_atr_extension_sma50(data, ma_alignment_result['sma50'])
        
        # Compile final result
        result = {
            'ticker': ticker,
            'signal_date': latest.name,
            'signal_type': 'qullamaggie_momentum',
            'screen_type': 'qullamaggie_suite',
            'price': latest['Close'],
            'volume': latest['Volume'],
            
            # RS metrics
            'rs_timeframes_qualified': rs_check_result['qualified_timeframes'],
            'best_rs_score': rs_check_result['best_rs_score'],
            'best_rs_timeframe': rs_check_result['best_rs_timeframe'],
            
            # Moving averages
            'ema10': ma_alignment_result['ema10'],
            'sma20': ma_alignment_result['sma20'],
            'sma50': ma_alignment_result['sma50'],
            'sma100': ma_alignment_result['sma100'],
            'sma200': ma_alignment_result['sma200'],
            'ma_alignment_score': ma_alignment_result['alignment_score'],
            
            # ATR metrics
            'atr_14': atr_rs_result['atr_value'],
            'atr_rs_score': atr_rs_result['atr_rs_score'],
            'atr_extension_sma50': atr_extension,
            
            # Range position
            'range_position_pct': range_position_result['range_position_pct'],
            'day_20_high': range_position_result['day_20_high'],
            'day_20_low': range_position_result['day_20_low'],
            
            # Metadata
            'strength': 'strong',  # All Qullamaggie stocks are strong by definition
            'data_length': len(data),
            'ipo_adjusted': len(data) < min_data_length,
            'raw_data': latest.to_dict()
        }
        
        return result

    def _check_market_cap_filter(self, ticker: str, ticker_info: Optional[pd.DataFrame]) -> bool:
        """Check market cap ≥ $1B filter"""
        if ticker_info is None:
            return True  # Skip filter if no ticker info available
        
        ticker_row = ticker_info[ticker_info['ticker'] == ticker]
        if ticker_row.empty:
            return True  # Skip filter if ticker not found
        
        # Check market cap columns
        market_cap_cols = ['market_cap', 'Market_Capitalization', 'marketCap']
        for col in market_cap_cols:
            if col in ticker_row.columns:
                market_cap = ticker_row[col].iloc[0]
                if pd.notna(market_cap):
                    return market_cap >= self.min_market_cap
        
        return True  # Skip filter if market cap not available

    def _check_rs_filter(self, ticker: str, rs_data: Optional[Dict]) -> Dict:
        """
        Check RS ≥ 97 on at least one timeframe (1w/1m/3m/6m)
        
        Returns:
            Dict with 'passed', 'qualified_timeframes', 'best_rs_score', 'best_rs_timeframe'
        """
        result = {
            'passed': False,
            'qualified_timeframes': [],
            'best_rs_score': 0,
            'best_rs_timeframe': None
        }
        
        if not rs_data:
            logger.warning("No RS data available - skipping RS filter")
            return {'passed': True, 'qualified_timeframes': ['no_rs_data'], 'best_rs_score': None, 'best_rs_timeframe': None}
        
        # Define timeframe mappings (adjust based on your RS data structure)
        timeframe_mappings = {
            '1w': ('daily', 5),    # 1 week ≈ 5 daily periods
            '1m': ('daily', 20),   # 1 month ≈ 20 daily periods  
            '3m': ('daily', 60),   # 3 months ≈ 60 daily periods
            '6m': ('daily', 120)   # 6 months ≈ 120 daily periods
        }
        
        best_rs = 0
        best_timeframe = None
        
        for qulla_timeframe, (rs_timeframe, rs_period) in timeframe_mappings.items():
            rs_score = self._get_rs_score(ticker, rs_data, rs_timeframe, rs_period)
            
            if rs_score is not None and rs_score >= self.rs_threshold:
                result['qualified_timeframes'].append(qulla_timeframe)
                if rs_score > best_rs:
                    best_rs = rs_score
                    best_timeframe = qulla_timeframe
        
        result['passed'] = len(result['qualified_timeframes']) > 0
        result['best_rs_score'] = best_rs if best_rs > 0 else None
        result['best_rs_timeframe'] = best_timeframe
        
        return result

    def _check_moving_averages_alignment(self, data: pd.DataFrame) -> Dict:
        """
        Check Price ≥ EMA10 ≥ SMA20 ≥ SMA50 ≥ SMA100 ≥ SMA200 alignment
        
        Returns:
            Dict with 'aligned', moving average values, and 'alignment_score'
        """
        try:
            # Calculate all required moving averages
            ema10 = data['Close'].ewm(span=10).mean().iloc[-1]
            sma20 = data['Close'].tail(20).mean()
            sma50 = data['Close'].tail(50).mean() if len(data) >= 50 else sma20
            sma100 = data['Close'].tail(100).mean() if len(data) >= 100 else sma50
            sma200 = data['Close'].tail(200).mean() if len(data) >= 200 else sma100
            
            current_price = data['Close'].iloc[-1]
            
            # Check alignment: Price ≥ EMA10 ≥ SMA20 ≥ SMA50 ≥ SMA100 ≥ SMA200
            alignment_checks = [
                current_price >= ema10,
                ema10 >= sma20,
                sma20 >= sma50,
                sma50 >= sma100,
                sma100 >= sma200
            ]
            
            all_aligned = all(alignment_checks)
            alignment_score = sum(alignment_checks) / len(alignment_checks) * 100
            
            return {
                'aligned': all_aligned,
                'ema10': ema10,
                'sma20': sma20,
                'sma50': sma50,
                'sma100': sma100,
                'sma200': sma200,
                'price': current_price,
                'alignment_score': alignment_score,
                'alignment_details': alignment_checks
            }
            
        except Exception as e:
            logger.warning(f"Error calculating moving averages alignment: {e}")
            return {'aligned': False, 'alignment_score': 0}

    def _check_atr_rs_filter(self, ticker: str, data: pd.DataFrame, 
                            atr_universe_rankings: Dict) -> Dict:
        """
        Check ATR RS ≥ 50 vs $1B+ universe
        
        Returns:
            Dict with 'passed', 'atr_value', 'atr_rs_score'
        """
        try:
            # Calculate 14-period ATR
            atr_value = self._calculate_atr(data, period=14)
            
            if atr_value is None:
                return {'passed': False, 'atr_value': None, 'atr_rs_score': None}
            
            # Get ATR RS ranking vs universe
            if ticker in atr_universe_rankings:
                atr_rs_score = atr_universe_rankings[ticker]
            else:
                # If no universe data, calculate approximate ATR RS
                atr_rs_score = self._estimate_atr_rs(atr_value, data)
            
            passed = atr_rs_score >= self.atr_rs_threshold
            
            return {
                'passed': passed,
                'atr_value': atr_value,
                'atr_rs_score': atr_rs_score
            }
            
        except Exception as e:
            logger.warning(f"Error calculating ATR RS for {ticker}: {e}")
            return {'passed': False, 'atr_value': None, 'atr_rs_score': None}

    def _check_range_position_filter(self, data: pd.DataFrame) -> Dict:
        """
        Check price in upper half of 20-day range (≥ 50%)
        
        Returns:
            Dict with 'passed', 'range_position_pct', '20d_high', '20d_low'
        """
        try:
            if len(data) < 20:
                return {'passed': False, 'range_position_pct': 0}
            
            # Get 20-day high and low
            recent_20d = data.tail(20)
            day_20_high = recent_20d['High'].max()
            day_20_low = recent_20d['Low'].min()
            current_price = data['Close'].iloc[-1]
            
            # Calculate range position: (Price - 20D Low) / (20D High - 20D Low)
            if day_20_high == day_20_low:
                range_position_pct = 1.0  # If no range, assume top
            else:
                range_position_pct = (current_price - day_20_low) / (day_20_high - day_20_low)
            
            passed = range_position_pct >= self.range_position_threshold
            
            return {
                'passed': passed,
                'range_position_pct': range_position_pct * 100,  # Convert to percentage
                'day_20_high': day_20_high,
                'day_20_low': day_20_low
            }
            
        except Exception as e:
            logger.warning(f"Error calculating range position: {e}")
            return {'passed': False, 'range_position_pct': 0}

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range (ATR)"""
        try:
            if len(data) < period + 1:
                return None
            
            # Calculate True Range components
            data_copy = data.copy()
            data_copy['prev_close'] = data_copy['Close'].shift(1)
            
            # True Range = max(high-low, high-prev_close, prev_close-low)
            data_copy['tr1'] = data_copy['High'] - data_copy['Low']
            data_copy['tr2'] = abs(data_copy['High'] - data_copy['prev_close'])
            data_copy['tr3'] = abs(data_copy['prev_close'] - data_copy['Low'])
            
            data_copy['true_range'] = data_copy[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calculate ATR as simple moving average of True Range
            atr = data_copy['true_range'].tail(period).mean()
            
            return atr
            
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            return None

    def _calculate_atr_extension_sma50(self, data: pd.DataFrame, sma50: float) -> float:
        """
        Calculate ATR extension relative to SMA50
        Extension = (Price - SMA50) / ATR
        """
        try:
            current_price = data['Close'].iloc[-1]
            atr = self._calculate_atr(data)
            
            if atr is None or atr == 0:
                return 0
            
            extension = (current_price - sma50) / atr
            return max(0, extension)  # Only positive extensions
            
        except Exception as e:
            logger.warning(f"Error calculating ATR extension: {e}")
            return 0

    def _calculate_atr_universe_rankings(self, atr_universe_data: Dict) -> Dict:
        """
        Calculate ATR rankings for the $1B+ universe
        
        Returns:
            Dict mapping ticker -> ATR RS percentile
        """
        try:
            atr_values = {}
            
            # Calculate ATR for all tickers in universe
            for ticker, data in atr_universe_data.items():
                atr = self._calculate_atr(data)
                if atr is not None:
                    atr_values[ticker] = atr
            
            if not atr_values:
                return {}
            
            # Calculate percentile rankings
            atr_series = pd.Series(atr_values)
            atr_rankings = atr_series.rank(pct=True) * 100  # Convert to percentile
            
            return atr_rankings.to_dict()
            
        except Exception as e:
            logger.error(f"Error calculating ATR universe rankings: {e}")
            return {}

    def _estimate_atr_rs(self, atr_value: float, data: pd.DataFrame) -> float:
        """
        Estimate ATR RS when universe data is not available
        Uses stock's own ATR vs its price volatility
        """
        try:
            # Simple estimation based on ATR relative to price
            price = data['Close'].iloc[-1]
            atr_pct = (atr_value / price) * 100
            
            # Rough mapping: higher ATR% -> higher volatility RS
            if atr_pct >= 4.0:
                return 90
            elif atr_pct >= 3.0:
                return 75
            elif atr_pct >= 2.0:
                return 60
            elif atr_pct >= 1.5:
                return 50
            else:
                return 30
                
        except Exception:
            return 50  # Default to middle

    def _handle_ipo_stock(self, ticker: str, data: pd.DataFrame,
                         ticker_info: Optional[pd.DataFrame],
                         rs_data: Optional[Dict],
                         atr_universe_rankings: Dict) -> Optional[Dict]:
        """
        Handle stocks with insufficient data (recent IPOs)
        Adjust requirements based on available data length
        """
        try:
            data_length = len(data)
            
            # Minimum viable data length
            if data_length < 50:
                return None
            
            latest = data.iloc[-1]
            
            # 1. Market cap filter (same)
            if not self._check_market_cap_filter(ticker, ticker_info):
                return None
            
            # 2. Adjusted RS filter (if available)
            rs_check_result = self._check_rs_filter(ticker, rs_data)
            
            # 3. Adjusted moving averages (use what's available)
            ma_result = self._check_ipo_moving_averages(data)
            if not ma_result['aligned']:
                return None
            
            # 4. ATR filter (adjusted period)
            atr_period = min(14, data_length // 3)
            atr = self._calculate_atr(data, atr_period)
            atr_rs_score = atr_universe_rankings.get(ticker, 50)  # Default to middle
            
            # 5. Range position (adjusted to available data)
            range_period = min(20, data_length)
            range_result = self._check_range_position_filter_ipo(data, period=range_period)
            if not range_result['passed']:
                return None
            
            # Calculate extension
            atr_extension = self._calculate_atr_extension_sma50(data, ma_result.get('sma50', latest['Close']))
            
            return {
                'ticker': ticker,
                'signal_date': latest.name,
                'signal_type': 'qullamaggie_ipo',
                'screen_type': 'qullamaggie_suite',
                'price': latest['Close'],
                'volume': latest['Volume'],
                'rs_timeframes_qualified': rs_check_result.get('qualified_timeframes', []),
                'best_rs_score': rs_check_result.get('best_rs_score'),
                'atr_extension_sma50': atr_extension,
                'range_position_pct': range_result['range_position_pct'],
                'ipo_adjusted': True,
                'data_length': data_length,
                'strength': 'moderate',
                'raw_data': latest.to_dict()
            }
            
        except Exception as e:
            logger.warning(f"Error handling IPO stock {ticker}: {e}")
            return None

    def _check_ipo_moving_averages(self, data: pd.DataFrame) -> Dict:
        """Check moving averages alignment for IPO stocks with limited data"""
        try:
            data_length = len(data)
            current_price = data['Close'].iloc[-1]
            
            # Use available periods or shorter
            ema10 = data['Close'].ewm(span=min(10, data_length//2)).mean().iloc[-1]
            sma20 = data['Close'].tail(min(20, data_length)).mean()
            sma50 = data['Close'].tail(min(50, data_length)).mean() if data_length >= 25 else sma20
            
            # Check basic uptrend alignment
            aligned = current_price >= ema10 >= sma20 >= sma50
            
            return {
                'aligned': aligned,
                'ema10': ema10,
                'sma20': sma20,
                'sma50': sma50,
                'alignment_score': 100 if aligned else 0
            }
            
        except Exception as e:
            logger.warning(f"Error checking IPO moving averages: {e}")
            return {'aligned': False, 'alignment_score': 0}

    def _get_rs_score(self, ticker: str, rs_data: Optional[Dict], 
                     timeframe: str, period: int) -> Optional[float]:
        """Extract RS score for a ticker from RS data"""
        try:
            if not rs_data or timeframe not in rs_data:
                return None
            
            timeframe_data = rs_data[timeframe]
            if f'period_{period}' in timeframe_data:
                period_data = timeframe_data[f'period_{period}']
                if ticker in period_data.index:
                    return period_data.loc[ticker, f'rs_percentile_{period}']
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting RS score for {ticker}: {e}")
            return None

    def _apply_extension_formatting(self, atr_extension: float) -> Dict:
        """
        Apply red/bold formatting based on ATR extension levels
        
        Returns:
            Dict with formatting information
        """
        formatting = {
            'extension_level': 'normal',
            'color': 'black',
            'bold': False,
            'warning_level': 'none'
        }
        
        if atr_extension >= self.extension_danger_threshold:  # 11x+
            formatting.update({
                'extension_level': 'extreme_danger',
                'color': 'red',
                'bold': True,
                'warning_level': 'extreme'
            })
        elif atr_extension >= self.extension_warning_threshold:  # 7x+
            formatting.update({
                'extension_level': 'warning',
                'color': 'red',
                'bold': True,
                'warning_level': 'high'
            })
        
        return formatting

    def _save_screening_results(self, results: List[Dict]):
        """Save Qullamaggie screening results to file"""
        try:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"qullamaggie_suite_{self.timeframe}_{timestamp}.csv"
            filepath = output_dir / filename
            
            # Create comprehensive results DataFrame
            df = pd.DataFrame(results)
            
            # Add extension warning columns for Excel formatting
            df['extension_warning'] = df['atr_extension_sma50'].apply(
                lambda x: 'EXTREME' if x >= self.extension_danger_threshold 
                         else 'HIGH' if x >= self.extension_warning_threshold 
                         else 'NORMAL'
            )
            
            # Sort by ATR extension (most extended first)
            df = df.sort_values('atr_extension_sma50', ascending=False)
            
            df.to_csv(filepath, index=False)
            logger.info(f"Saved Qullamaggie results: {len(df)} signals to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving Qullamaggie results: {e}")

    def _check_range_position_filter_ipo(self, data: pd.DataFrame, period: int = 20) -> Dict:
        """Check range position filter for IPO stocks with adjusted period"""
        try:
            if len(data) < 5:
                return {'passed': False, 'range_position_pct': 0}
            
            # Use available data up to specified period
            recent_data = data.tail(period)
            day_high = recent_data['High'].max()
            day_low = recent_data['Low'].min()
            current_price = data['Close'].iloc[-1]
            
            # Calculate range position
            if day_high == day_low:
                range_position_pct = 1.0
            else:
                range_position_pct = (current_price - day_low) / (day_high - day_low)
            
            passed = range_position_pct >= self.range_position_threshold
            
            return {
                'passed': passed,
                'range_position_pct': range_position_pct * 100,
                'day_20_high': day_high,
                'day_20_low': day_low
            }
            
        except Exception as e:
            logger.warning(f"Error calculating IPO range position: {e}")
            return {'passed': False, 'range_position_pct': 0}


def run_qullamaggie_suite_screener(batch_data: Dict[str, pd.DataFrame], 
                                  config: Dict[str, Any],
                                  ticker_info: Optional[pd.DataFrame] = None,
                                  rs_data: Optional[Dict] = None,
                                  atr_universe_data: Optional[Dict] = None) -> List[Dict]:
    """
    Main entry point for Qullamaggie suite screening
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        config: Configuration dictionary
        ticker_info: Optional ticker information DataFrame
        rs_data: Optional RS data for enhanced screening
        atr_universe_data: Optional ATR universe data for ATR RS calculation
        
    Returns:
        List of screening results sorted by ATR extension
    """
    screener = QullamaggieScreener(config)
    return screener.run_qullamaggie_screening(batch_data, ticker_info, rs_data, atr_universe_data)


# Export main functions
__all__ = ['QullamaggieScreener', 'run_qullamaggie_suite_screener']