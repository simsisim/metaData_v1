"""
Guppy Multiple Moving Average (GMMA) Screener
=============================================

Implementation of Daryl Guppy's Multiple Moving Average methodology for 
detecting trend alignment, breakouts, and momentum shifts.

GMMA uses two groups of EMAs:
- Short-term group: 3, 5, 8, 10, 12, 15 (trader behavior/short-term sentiment)
- Long-term group: 30, 35, 40, 45, 50, 60 (investor behavior/long-term trend)

Key signals:
1. Bullish Alignment: All short-term EMAs > all long-term EMAs
2. Bearish Alignment: All short-term EMAs < all long-term EMAs  
3. Compression: EMA groups converging (low volatility, potential breakout)
4. Expansion: EMA groups diverging (high momentum, strong trend)
5. Crossover: Short-term group crossing above/below long-term group

Based on Daryl Guppy's GMMA methodology for trend analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class GuppyScreener:
    """
    Guppy Multiple Moving Average (GMMA) screener for trend analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_guppy_screener = config.get('enable_guppy_screener', True)
        self.timeframe = config.get('timeframe', 'daily')
        
        # Guppy screener specific configuration
        self.guppy_config = config.get('guppy_screener', {})

        # Moving average type (NEW - user configurable)
        self.ma_type = self.guppy_config.get('ma_type', 'EMA').upper()

        # MA groups (NOW from user configuration, not hardcoded)
        self.short_term_emas = self.guppy_config.get('short_term_emas', [3, 5, 8, 10, 12, 15])
        self.long_term_emas = self.guppy_config.get('long_term_emas', [30, 35, 40, 45, 50, 60])
        
        # Signal detection parameters
        self.min_compression_ratio = self.guppy_config.get('min_compression_ratio', 0.02)  # 2%
        self.min_expansion_ratio = self.guppy_config.get('min_expansion_ratio', 0.05)  # 5%
        self.crossover_confirmation_days = self.guppy_config.get('crossover_confirmation_days', 3)
        self.volume_confirmation_threshold = self.guppy_config.get('volume_confirmation_threshold', 1.2)
        
        # Base filtering parameters
        self.min_price = self.guppy_config.get('min_price', 5.0)
        self.min_volume_avg = self.guppy_config.get('min_volume_avg', 100_000)
        self.min_data_length = self.guppy_config.get('min_data_length', 65)  # Need 60+ for long EMAs
        
        # Output configuration
        self.output_dir = config.get('guppy_output_dir', 'results/screeners/guppy')
        self.save_individual_files = self.guppy_config.get('save_individual_files', True)
        
        logger.info(f"Guppy GMMA Screener initialized (enabled: {self.enable_guppy_screener})")

    def run_guppy_screening(self, batch_data: Dict[str, pd.DataFrame], 
                           ticker_info: Optional[pd.DataFrame] = None,
                           rs_data: Optional[Dict] = None,
                           batch_info: Dict[str, Any] = None) -> List[Dict]:
        """
        Run Guppy GMMA screening for trend analysis
        
        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            ticker_info: DataFrame with ticker information
            rs_data: Dictionary with RS data for additional scoring
            batch_info: Optional batch processing information
            
        Returns:
            List of screening results
        """
        if not self.enable_guppy_screener:
            logger.info("Guppy GMMA screener disabled")
            return []
            
        if not batch_data:
            logger.warning("No data provided for Guppy GMMA screening")
            return []
        
        logger.info(f"Running Guppy GMMA screening on {len(batch_data)} tickers")
        
        all_results = []
        component_results = {
            'bullish_alignment': [],
            'bearish_alignment': [],
            'compression_breakout': [],
            'expansion_signal': [],
            'bullish_crossover': [],
            'bearish_crossover': []
        }
        
        try:
            # Pre-filter tickers based on basic criteria
            filtered_tickers = self._apply_base_filters(batch_data)
            logger.info(f"Filtered to {len(filtered_tickers)} tickers meeting base criteria")
            
            # Process each ticker for GMMA signals
            for ticker, data in filtered_tickers.items():
                try:
                    # Calculate all EMAs for GMMA analysis
                    data_with_emas = self._calculate_gmma_emas(data)
                    if data_with_emas is None:
                        continue
                    
                    # Check for different GMMA signal types
                    signals = self._detect_gmma_signals(ticker, data_with_emas, rs_data)
                    
                    for signal in signals:
                        signal_type = signal['signal_type']
                        if signal_type in component_results:
                            component_results[signal_type].append(signal)
                        all_results.append(signal)
                        
                except Exception as e:
                    logger.warning(f"Error processing {ticker} in Guppy GMMA screening: {e}")
                    continue
            
            # Save results if enabled
            if self.save_individual_files:
                self._save_component_results(component_results)
            
            logger.info(f"Guppy GMMA screening completed: {len(all_results)} total signals")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in Guppy GMMA screening: {e}")
            return []

    def _apply_base_filters(self, batch_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply base filters for GMMA analysis
        
        Args:
            batch_data: Dictionary of ticker data
            
        Returns:
            Filtered dictionary of ticker data
        """
        filtered_data = {}
        
        for ticker, data in batch_data.items():
            try:
                # Basic data validation
                if data.empty or len(data) < self.min_data_length:
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

    def _calculate_gmma_emas(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate all GMMA moving averages for both short-term and long-term groups.
        Now supports both EMA and SMA based on user configuration.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with all MAs added or None if calculation fails
        """
        try:
            data_with_emas = data.copy()

            # Calculate short-term moving averages with user's periods
            for period in self.short_term_emas:
                if self.ma_type == 'EMA':
                    data_with_emas[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
                elif self.ma_type == 'SMA':
                    data_with_emas[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
                else:
                    # Default fallback to EMA for unknown types
                    logger.warning(f"Unknown MA type: {self.ma_type}, using EMA")
                    data_with_emas[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()

            # Calculate long-term moving averages with user's periods
            for period in self.long_term_emas:
                if self.ma_type == 'EMA':
                    data_with_emas[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
                elif self.ma_type == 'SMA':
                    data_with_emas[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
                else:
                    # Default fallback to EMA for unknown types
                    data_with_emas[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()

            return data_with_emas

        except Exception as e:
            logger.warning(f"Error calculating GMMA moving averages: {e}")
            return None

    def _get_ma_column_names(self) -> Tuple[List[str], List[str]]:
        """
        Get moving average column names based on user's MA type and periods.

        Returns:
            Tuple of (short_term_columns, long_term_columns)
        """
        ma_prefix = self.ma_type  # "EMA" or "SMA"
        short_cols = [f'{ma_prefix}_{period}' for period in self.short_term_emas]
        long_cols = [f'{ma_prefix}_{period}' for period in self.long_term_emas]
        return short_cols, long_cols

    def _detect_gmma_signals(self, ticker: str, data: pd.DataFrame,
                            rs_data: Optional[Dict] = None) -> List[Dict]:
        """
        Detect all GMMA signal types for a ticker
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with EMAs calculated
            rs_data: RS data for additional scoring
            
        Returns:
            List of detected signals
        """
        signals = []
        latest = data.iloc[-1]
        
        try:
            # 1. Check for bullish/bearish alignment
            alignment_signal = self._check_alignment(ticker, data, rs_data)
            if alignment_signal:
                signals.append(alignment_signal)
            
            # 2. Check for compression/expansion patterns
            compression_signal = self._check_compression_breakout(ticker, data, rs_data)
            if compression_signal:
                signals.append(compression_signal)
            
            expansion_signal = self._check_expansion(ticker, data, rs_data)
            if expansion_signal:
                signals.append(expansion_signal)
            
            # 3. Check for crossover signals
            crossover_signal = self._check_crossover(ticker, data, rs_data)
            if crossover_signal:
                signals.append(crossover_signal)
            
            return signals
            
        except Exception as e:
            logger.warning(f"Error detecting GMMA signals for {ticker}: {e}")
            return []

    def _check_alignment(self, ticker: str, data: pd.DataFrame, 
                        rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check for bullish/bearish GMMA alignment
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with EMAs
            rs_data: RS data for additional scoring
            
        Returns:
            Signal result or None
        """
        try:
            latest = data.iloc[-1]

            # Get dynamic column names based on user's MA type
            short_cols, long_cols = self._get_ma_column_names()

            # Get all MA values for latest bar using dynamic column names
            st_emas = [latest[col] for col in short_cols]
            lt_emas = [latest[col] for col in long_cols]
            
            # Check for perfect bullish alignment (all ST > all LT)
            bullish_alignment = all(st > lt for st in st_emas for lt in lt_emas)
            
            # Check for perfect bearish alignment (all ST < all LT)
            bearish_alignment = all(st < lt for st in st_emas for lt in lt_emas)
            
            if bullish_alignment or bearish_alignment:
                signal_type = 'bullish_alignment' if bullish_alignment else 'bearish_alignment'
                
                # Calculate alignment strength
                price = latest['Close']
                st_avg = np.mean(st_emas)
                lt_avg = np.mean(lt_emas)
                separation_pct = abs((st_avg - lt_avg) / lt_avg) * 100
                
                # Volume confirmation
                volume_confirmation = self._check_volume_confirmation(data)
                
                rs_score = self._get_rs_score(ticker, rs_data) if rs_data else None
                
                return {
                    'ticker': ticker,
                    'signal_date': latest.name,
                    'signal_type': signal_type,
                    'screen_type': 'guppy_alignment',
                    'price': price,
                    'volume': latest['Volume'],
                    'separation_pct': separation_pct,
                    'st_ema_avg': st_avg,
                    'lt_ema_avg': lt_avg,
                    'volume_confirmation': volume_confirmation,
                    'rs_score': rs_score,
                    'strength': self._calculate_signal_strength(separation_pct, volume_confirmation),
                    'raw_data': latest.to_dict()
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking GMMA alignment for {ticker}: {e}")
            return None

    def _check_compression_breakout(self, ticker: str, data: pd.DataFrame, 
                                   rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check for compression breakout (EMAs compressed then expanding)
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with EMAs
            rs_data: RS data for additional scoring
            
        Returns:
            Signal result or None
        """
        try:
            if len(data) < 20:  # Need history for compression analysis
                return None
            
            latest = data.iloc[-1]
            
            # Get dynamic column names based on user's MA type
            short_cols, long_cols = self._get_ma_column_names()

            # Calculate MA group spreads over time
            spreads = []
            for i in range(-10, 0):  # Last 10 days
                row = data.iloc[i]
                st_emas = [row[col] for col in short_cols]
                lt_emas = [row[col] for col in long_cols]
                
                st_spread = (max(st_emas) - min(st_emas)) / min(st_emas)
                lt_spread = (max(lt_emas) - min(lt_emas)) / min(lt_emas)
                total_spread = st_spread + lt_spread
                spreads.append(total_spread)
            
            # Check if moving from compression to expansion
            recent_compression = min(spreads[-5:])  # Most compressed in last 5 days
            current_spread = spreads[-1]
            
            # Compression breakout: was compressed, now expanding
            was_compressed = recent_compression <= self.min_compression_ratio
            now_expanding = current_spread > recent_compression * 1.5
            
            if was_compressed and now_expanding:
                # Determine breakout direction
                price = latest['Close']
                st_avg = np.mean([latest[col] for col in short_cols])
                lt_avg = np.mean([latest[col] for col in long_cols])
                
                direction = 'bullish' if st_avg > lt_avg else 'bearish'
                volume_confirmation = self._check_volume_confirmation(data)
                
                rs_score = self._get_rs_score(ticker, rs_data) if rs_data else None
                
                return {
                    'ticker': ticker,
                    'signal_date': latest.name,
                    'signal_type': 'compression_breakout',
                    'screen_type': 'guppy_compression',
                    'price': price,
                    'volume': latest['Volume'],
                    'direction': direction,
                    'compression_ratio': recent_compression,
                    'expansion_ratio': current_spread,
                    'breakout_strength': current_spread / recent_compression,
                    'volume_confirmation': volume_confirmation,
                    'rs_score': rs_score,
                    'strength': self._calculate_signal_strength(current_spread * 100, volume_confirmation),
                    'raw_data': latest.to_dict()
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking compression breakout for {ticker}: {e}")
            return None

    def _check_expansion(self, ticker: str, data: pd.DataFrame, 
                        rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check for strong expansion signal (wide EMA separation indicating strong trend)
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with EMAs
            rs_data: RS data for additional scoring
            
        Returns:
            Signal result or None
        """
        try:
            latest = data.iloc[-1]

            # Get dynamic column names based on user's MA type
            short_cols, long_cols = self._get_ma_column_names()

            # Calculate current MA group spreads using dynamic column names
            st_emas = [latest[col] for col in short_cols]
            lt_emas = [latest[col] for col in long_cols]
            
            st_spread = (max(st_emas) - min(st_emas)) / min(st_emas)
            lt_spread = (max(lt_emas) - min(lt_emas)) / min(lt_emas)
            total_spread = st_spread + lt_spread
            
            # Check for strong expansion
            if total_spread >= self.min_expansion_ratio:
                # Determine trend direction
                price = latest['Close']
                st_avg = np.mean(st_emas)
                lt_avg = np.mean(lt_emas)
                
                # Strong expansion with clear direction
                if st_avg > lt_avg * 1.02:  # Bullish expansion
                    direction = 'bullish'
                elif st_avg < lt_avg * 0.98:  # Bearish expansion
                    direction = 'bearish'
                else:
                    return None  # Neutral expansion
                
                volume_confirmation = self._check_volume_confirmation(data)
                rs_score = self._get_rs_score(ticker, rs_data) if rs_data else None
                
                return {
                    'ticker': ticker,
                    'signal_date': latest.name,
                    'signal_type': 'expansion_signal',
                    'screen_type': 'guppy_expansion',
                    'price': price,
                    'volume': latest['Volume'],
                    'direction': direction,
                    'expansion_ratio': total_spread,
                    'st_spread': st_spread,
                    'lt_spread': lt_spread,
                    'separation_pct': abs((st_avg - lt_avg) / lt_avg) * 100,
                    'volume_confirmation': volume_confirmation,
                    'rs_score': rs_score,
                    'strength': self._calculate_signal_strength(total_spread * 100, volume_confirmation),
                    'raw_data': latest.to_dict()
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking expansion for {ticker}: {e}")
            return None

    def _check_crossover(self, ticker: str, data: pd.DataFrame, 
                        rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check for GMMA crossover signals (short-term group crossing long-term group)
        
        Args:
            ticker: Ticker symbol
            data: OHLCV data with EMAs
            rs_data: RS data for additional scoring
            
        Returns:
            Signal result or None
        """
        try:
            if len(data) < self.crossover_confirmation_days + 1:
                return None

            latest = data.iloc[-1]

            # Get dynamic column names based on user's MA type
            short_cols, long_cols = self._get_ma_column_names()

            # Check crossover pattern over confirmation period
            crossover_direction = None
            crossover_strength = 0

            for i in range(1, self.crossover_confirmation_days + 1):
                prev_row = data.iloc[-i-1]
                curr_row = data.iloc[-i]

                # Calculate group averages using dynamic column names
                prev_st_avg = np.mean([prev_row[col] for col in short_cols])
                prev_lt_avg = np.mean([prev_row[col] for col in long_cols])
                curr_st_avg = np.mean([curr_row[col] for col in short_cols])
                curr_lt_avg = np.mean([curr_row[col] for col in long_cols])
                
                # Check for crossover
                if prev_st_avg <= prev_lt_avg and curr_st_avg > curr_lt_avg:
                    if crossover_direction is None:
                        crossover_direction = 'bullish'
                    elif crossover_direction != 'bullish':
                        crossover_direction = None
                        break
                elif prev_st_avg >= prev_lt_avg and curr_st_avg < curr_lt_avg:
                    if crossover_direction is None:
                        crossover_direction = 'bearish'
                    elif crossover_direction != 'bearish':
                        crossover_direction = None
                        break
                
                # Calculate crossover strength
                separation = abs(curr_st_avg - curr_lt_avg) / curr_lt_avg
                crossover_strength += separation
            
            if crossover_direction:
                volume_confirmation = self._check_volume_confirmation(data)
                rs_score = self._get_rs_score(ticker, rs_data) if rs_data else None
                
                signal_type = f'{crossover_direction}_crossover'
                
                return {
                    'ticker': ticker,
                    'signal_date': latest.name,
                    'signal_type': signal_type,
                    'screen_type': 'guppy_crossover',
                    'price': latest['Close'],
                    'volume': latest['Volume'],
                    'crossover_direction': crossover_direction,
                    'crossover_strength': crossover_strength,
                    'confirmation_days': self.crossover_confirmation_days,
                    'volume_confirmation': volume_confirmation,
                    'rs_score': rs_score,
                    'strength': self._calculate_signal_strength(crossover_strength * 100, volume_confirmation),
                    'raw_data': latest.to_dict()
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking crossover for {ticker}: {e}")
            return None

    def _check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """
        Check for volume confirmation of GMMA signals
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            True if volume confirms the signal
        """
        try:
            if len(data) < 10:
                return False
            
            latest_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].tail(10).mean()
            
            return latest_volume >= avg_volume * self.volume_confirmation_threshold
            
        except Exception:
            return False

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

    def _calculate_signal_strength(self, strength_value: float, volume_confirmation: bool) -> str:
        """
        Calculate signal strength category
        
        Args:
            strength_value: Numerical strength value
            volume_confirmation: Whether volume confirms the signal
            
        Returns:
            Signal strength category
        """
        # Base strength from numerical value
        if strength_value >= 5.0:
            base_strength = 3
        elif strength_value >= 3.0:
            base_strength = 2
        elif strength_value >= 1.0:
            base_strength = 1
        else:
            base_strength = 0
        
        # Volume confirmation adds strength
        if volume_confirmation:
            base_strength += 1
        
        if base_strength >= 4:
            return 'very_strong'
        elif base_strength >= 3:
            return 'strong'
        elif base_strength >= 2:
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
                    filename = f"guppy_{component_name}_{self.timeframe}_{timestamp}.csv"
                    filepath = output_dir / filename
                    
                    # Convert to DataFrame and save
                    if isinstance(results, list) and results:
                        df = pd.DataFrame(results)
                        df.to_csv(filepath, index=False)
                        logger.info(f"Saved {component_name} results: {len(df)} signals to {filepath}")
                    
        except Exception as e:
            logger.error(f"Error saving component results: {e}")


def run_guppy_screener(batch_data: Dict[str, pd.DataFrame], 
                      config: Dict[str, Any],
                      ticker_info: Optional[pd.DataFrame] = None,
                      rs_data: Optional[Dict] = None) -> List[Dict]:
    """
    Main entry point for Guppy GMMA screening
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        config: Configuration dictionary
        ticker_info: Optional ticker information DataFrame
        rs_data: Optional RS data for enhanced screening
        
    Returns:
        List of screening results
    """
    screener = GuppyScreener(config)
    return screener.run_guppy_screening(batch_data, ticker_info, rs_data)


# Export main functions
__all__ = ['GuppyScreener', 'run_guppy_screener']