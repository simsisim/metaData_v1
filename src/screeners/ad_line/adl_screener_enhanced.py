"""
ADL Enhanced Screener Orchestrator
==================================

Main orchestrator that integrates all ADL analysis modules:
- Step 1: Basic ADL calculation (existing functionality preserved)
- Step 2: Month-over-month accumulation analysis
- Step 3: Short-term momentum detection
- Step 4: Moving average overlay and alignment
- Step 5: Composite scoring and ranking

This enhanced screener maintains backward compatibility with existing
divergence and breakout detection while adding new analysis capabilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .adl_calculator import ADLCalculator
from .adl_mom_analysis import ADLMoMAnalyzer
from .adl_short_term import ADLShortTermAnalyzer
from .adl_ma_analysis import ADLMAAnalyzer
from .adl_composite_scoring import ADLCompositeScorer
from .adl_utils import validate_ohlcv_data, extract_date_from_dataframe

logger = logging.getLogger(__name__)


class ADLScreenerEnhanced:
    """
    Enhanced ADL screener with multi-dimensional analysis.

    Preserves existing divergence/breakout detection functionality
    while adding month-over-month, short-term momentum, MA analysis,
    and composite scoring.
    """

    def __init__(self, config: Dict[str, Any], user_config=None):
        """
        Initialize enhanced ADL screener.

        Args:
            config: Configuration dictionary
            user_config: UserConfiguration object with all parameters
        """
        self.config = config
        self.user_config = user_config
        self.timeframe = config.get('timeframe', 'daily')

        # Extract configuration from user_config if available
        if user_config:
            self._init_from_user_config(user_config)
        else:
            self._init_from_dict_config(config)

        # Initialize analysis modules
        self._init_analyzers()

        # Output configuration
        self.output_dir = config.get('adl_output_dir', 'results/screeners/adl')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"ADL Enhanced Screener initialized for {self.timeframe} timeframe")
        logger.debug(f"Modules enabled: MoM={self.mom_enabled}, "
                    f"ShortTerm={self.short_term_enabled}, MA={self.ma_enabled}, "
                    f"Composite={self.composite_enabled}")

    def _init_from_user_config(self, user_config):
        """Initialize from UserConfiguration object."""
        # Base parameters (existing functionality)
        self.lookback_period = user_config.adl_screener_lookback_period
        self.divergence_period = user_config.adl_screener_divergence_period
        self.breakout_period = user_config.adl_screener_breakout_period
        self.min_divergence_strength = user_config.adl_screener_min_divergence_strength
        self.min_breakout_strength = user_config.adl_screener_min_breakout_strength
        self.min_volume_avg = user_config.adl_screener_min_volume_avg
        self.min_price = user_config.adl_screener_min_price
        self.save_individual_files = user_config.adl_screener_save_individual_files

        # Enhanced module enables
        self.mom_enabled = user_config.adl_screener_mom_analysis_enable
        self.short_term_enabled = user_config.adl_screener_short_term_enable
        self.ma_enabled = user_config.adl_screener_ma_enable
        self.composite_enabled = user_config.adl_screener_composite_scoring_enable

        # MoM parameters
        self.mom_params = {
            'mom_period': user_config.adl_screener_mom_period,
            'mom_min_threshold_pct': user_config.adl_screener_mom_min_threshold_pct,
            'mom_max_threshold_pct': user_config.adl_screener_mom_max_threshold_pct,
            'mom_consecutive_months': user_config.adl_screener_mom_consecutive_months,
            'mom_lookback_months': user_config.adl_screener_mom_lookback_months,
            'mom_min_consistency_score': user_config.adl_screener_mom_min_consistency_score
        }

        # Short-term parameters
        self.short_term_params = {
            'short_term_periods': user_config.adl_screener_short_term_periods,
            'short_term_momentum_threshold': user_config.adl_screener_short_term_momentum_threshold,
            'short_term_acceleration_detect': user_config.adl_screener_short_term_acceleration_detect,
            'short_term_min_score': user_config.adl_screener_short_term_min_score
        }

        # MA parameters
        self.ma_params = {
            'ma_periods': user_config.adl_screener_ma_periods,
            'ma_type': user_config.adl_screener_ma_type,
            'ma_bullish_alignment_required': user_config.adl_screener_ma_bullish_alignment_required,
            'ma_crossover_detection': user_config.adl_screener_ma_crossover_detection,
            'ma_crossover_lookback': user_config.adl_screener_ma_crossover_lookback,
            'ma_min_slope_threshold': user_config.adl_screener_ma_min_slope_threshold,
            'ma_min_alignment_score': user_config.adl_screener_ma_min_alignment_score
        }

        # Composite parameters
        self.composite_params = {
            'composite_weight_longterm': user_config.adl_screener_composite_weight_longterm,
            'composite_weight_shortterm': user_config.adl_screener_composite_weight_shortterm,
            'composite_weight_ma_align': user_config.adl_screener_composite_weight_ma_align,
            'composite_min_score': user_config.adl_screener_composite_min_score,
            'ranking_method': user_config.adl_screener_ranking_method,
            'top_candidates_count': user_config.adl_screener_top_candidates_count
        }

        # Output parameters
        self.output_separate_signals = user_config.adl_screener_output_separate_signals
        self.output_summary_stats = user_config.adl_screener_output_summary_stats

    def _init_from_dict_config(self, config):
        """Initialize from dictionary configuration (fallback)."""
        adl_config = config.get('adl_screener', {})

        # Base parameters
        self.lookback_period = adl_config.get('lookback_period', 50)
        self.divergence_period = adl_config.get('divergence_period', 20)
        self.breakout_period = adl_config.get('breakout_period', 30)
        self.min_divergence_strength = adl_config.get('min_divergence_strength', 0.7)
        self.min_breakout_strength = adl_config.get('min_breakout_strength', 1.2)
        self.min_volume_avg = adl_config.get('min_volume_avg', 100000)
        self.min_price = adl_config.get('min_price', 5.0)
        self.save_individual_files = adl_config.get('save_individual_files', True)

        # Module enables (default all True for testing)
        self.mom_enabled = adl_config.get('mom_analysis_enable', True)
        self.short_term_enabled = adl_config.get('short_term_enable', True)
        self.ma_enabled = adl_config.get('ma_enable', True)
        self.composite_enabled = adl_config.get('composite_scoring_enable', True)

        # Default parameters for each module
        self.mom_params = adl_config.get('mom_params', {
            'mom_period': 22,
            'mom_min_threshold_pct': 15.0,
            'mom_max_threshold_pct': 30.0,
            'mom_consecutive_months': 3,
            'mom_lookback_months': 6,
            'mom_min_consistency_score': 60.0
        })

        self.short_term_params = adl_config.get('short_term_params', {
            'short_term_periods': '5;10;20',
            'short_term_momentum_threshold': 5.0,
            'short_term_acceleration_detect': True,
            'short_term_min_score': 50.0
        })

        self.ma_params = adl_config.get('ma_params', {
            'ma_periods': '20;50;100',
            'ma_type': 'SMA',
            'ma_bullish_alignment_required': True,
            'ma_crossover_detection': True,
            'ma_crossover_lookback': 10,
            'ma_min_slope_threshold': 0.01,
            'ma_min_alignment_score': 70.0
        })

        self.composite_params = adl_config.get('composite_params', {
            'composite_weight_longterm': 0.4,
            'composite_weight_shortterm': 0.3,
            'composite_weight_ma_align': 0.3,
            'composite_min_score': 70.0,
            'ranking_method': 'composite',
            'top_candidates_count': 50
        })

        self.output_separate_signals = adl_config.get('output_separate_signals', True)
        self.output_summary_stats = adl_config.get('output_summary_stats', True)

    def _init_analyzers(self):
        """Initialize analysis modules."""
        self.calculator = ADLCalculator()

        if self.mom_enabled:
            self.mom_analyzer = ADLMoMAnalyzer(self.mom_params)
        else:
            self.mom_analyzer = None

        if self.short_term_enabled:
            self.short_term_analyzer = ADLShortTermAnalyzer(self.short_term_params)
        else:
            self.short_term_analyzer = None

        if self.ma_enabled:
            self.ma_analyzer = ADLMAAnalyzer(self.ma_params)
        else:
            self.ma_analyzer = None

        if self.composite_enabled:
            self.composite_scorer = ADLCompositeScorer(self.composite_params)
        else:
            self.composite_scorer = None

    def run_enhanced_screening(self,
                              batch_data: Dict[str, pd.DataFrame],
                              ticker_info: Optional[pd.DataFrame] = None,
                              rs_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run enhanced ADL screening with all modules.

        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            ticker_info: Optional ticker information
            rs_data: Optional RS data for additional analysis

        Returns:
            Dictionary with all results:
            - 'composite_results': List of composite scores (if enabled)
            - 'mom_results': List of MoM analysis results
            - 'short_term_results': List of short-term momentum results
            - 'ma_results': List of MA analysis results
            - 'divergence_results': List of divergence signals (existing)
            - 'breakout_results': List of breakout signals (existing)
            - 'summary': Summary statistics
        """
        if not batch_data:
            logger.warning("No data provided for enhanced ADL screening")
            return {}

        logger.info(f"Running enhanced ADL screening on {len(batch_data)} tickers")

        # Filter tickers by base criteria
        filtered_tickers = self._apply_base_filters(batch_data)
        logger.info(f"Filtered to {len(filtered_tickers)} tickers meeting base criteria")

        # Initialize result containers
        all_results = {
            'composite_results': [],
            'mom_results': [],
            'short_term_results': [],
            'ma_results': [],
            'divergence_results': [],  # Existing functionality
            'breakout_results': []      # Existing functionality
        }

        # Process each ticker
        for ticker, df in filtered_tickers.items():
            try:
                ticker_results = self._process_single_ticker(ticker, df, rs_data)

                if ticker_results:
                    # Collect results by type
                    if ticker_results.get('composite'):
                        all_results['composite_results'].append(ticker_results['composite'])

                    if ticker_results.get('mom'):
                        all_results['mom_results'].append(ticker_results['mom'])

                    if ticker_results.get('short_term'):
                        all_results['short_term_results'].append(ticker_results['short_term'])

                    if ticker_results.get('ma'):
                        all_results['ma_results'].append(ticker_results['ma'])

                    # Existing signal types (preserved)
                    if ticker_results.get('divergence'):
                        all_results['divergence_results'].extend(ticker_results['divergence'])

                    if ticker_results.get('breakout'):
                        all_results['breakout_results'].extend(ticker_results['breakout'])

            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")
                continue

        # Generate composite rankings if enabled
        if self.composite_enabled and all_results['composite_results']:
            all_results['composite_results'] = self.composite_scorer.rank_stocks(
                all_results['composite_results']
            )
            all_results['top_candidates'] = self.composite_scorer.generate_top_candidates(
                all_results['composite_results']
            )

        # Generate summary statistics
        if self.output_summary_stats:
            all_results['summary'] = self._generate_summary(all_results)

        logger.info(f"Enhanced screening completed: {len(all_results['composite_results'])} "
                   f"composite scores generated")

        return all_results

    def _process_single_ticker(self,
                              ticker: str,
                              df: pd.DataFrame,
                              rs_data: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """
        Process single ticker through all analysis modules.

        Args:
            ticker: Ticker symbol
            df: OHLCV DataFrame
            rs_data: Optional RS data

        Returns:
            Dictionary with all analysis results for this ticker
        """
        results = {}

        # Extract current price, volume, date
        current_price = df['Close'].iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        current_date = extract_date_from_dataframe(df)

        # Step 2: Month-over-Month Analysis
        if self.mom_enabled and self.mom_analyzer:
            mom_result = self.mom_analyzer.analyze_monthly_accumulation(df, ticker)
            if mom_result:
                results['mom'] = mom_result

        # Step 3: Short-term Momentum Analysis
        if self.short_term_enabled and self.short_term_analyzer:
            short_term_result = self.short_term_analyzer.calculate_short_term_changes(df, ticker)
            if short_term_result:
                results['short_term'] = short_term_result

        # Step 4: Moving Average Analysis
        if self.ma_enabled and self.ma_analyzer:
            ma_result = self.ma_analyzer.calculate_adl_mas(df, ticker)
            if ma_result:
                results['ma'] = ma_result

        # Step 5: Composite Scoring (if all components available)
        if self.composite_enabled and self.composite_scorer:
            composite_result = self.composite_scorer.score_ticker(
                ticker=ticker,
                mom_result=results.get('mom'),
                shortterm_result=results.get('short_term'),
                ma_result=results.get('ma'),
                price=current_price,
                volume=current_volume,
                date=current_date
            )
            if composite_result:
                results['composite'] = composite_result

        # Existing functionality: Divergence and Breakout detection
        # (Only run if composite scoring is disabled or as additional signals)
        divergence_signals = self._check_divergence_signals(ticker, df, rs_data)
        if divergence_signals:
            results['divergence'] = divergence_signals

        breakout_signals = self._check_breakout_signals(ticker, df, rs_data)
        if breakout_signals:
            results['breakout'] = breakout_signals

        return results if results else None

    def _apply_base_filters(self, batch_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply base filters for ADL analysis.

        Args:
            batch_data: Dictionary of ticker data

        Returns:
            Filtered dictionary of ticker data
        """
        filtered_data = {}

        for ticker, df in batch_data.items():
            try:
                # Validate OHLCV data
                if not validate_ohlcv_data(df, min_length=self.lookback_period):
                    continue

                # Price filter
                current_price = df['Close'].iloc[-1]
                if current_price < self.min_price:
                    continue

                # Volume filter
                avg_volume = df['Volume'].tail(20).mean()
                if avg_volume < self.min_volume_avg:
                    continue

                # Exclude obvious funds/ETFs
                if any(suffix in ticker.upper() for suffix in ['ETF', 'QQQ', 'SPY', 'IWM', 'XL']):
                    continue

                filtered_data[ticker] = df

            except Exception as e:
                logger.debug(f"Error filtering {ticker}: {e}")
                continue

        return filtered_data

    def _check_divergence_signals(self,
                                  ticker: str,
                                  df: pd.DataFrame,
                                  rs_data: Optional[Dict]) -> List[Dict]:
        """
        Check for ADL divergence signals (existing functionality preserved).

        Args:
            ticker: Ticker symbol
            df: OHLCV DataFrame
            rs_data: Optional RS data

        Returns:
            List of divergence signal dictionaries
        """
        signals = []

        try:
            # Calculate ADL
            adl = self.calculator.calculate_adl(df)
            if adl is None or len(adl) < self.divergence_period:
                return signals

            # Add ADL to dataframe
            df_with_adl = df.copy()
            df_with_adl['ADL'] = adl

            # Check bullish divergence
            bullish_div = self._check_bullish_divergence(ticker, df_with_adl, rs_data)
            if bullish_div:
                signals.append(bullish_div)

            # Check bearish divergence
            bearish_div = self._check_bearish_divergence(ticker, df_with_adl, rs_data)
            if bearish_div:
                signals.append(bearish_div)

        except Exception as e:
            logger.debug(f"Error checking divergence for {ticker}: {e}")

        return signals

    def _check_breakout_signals(self,
                               ticker: str,
                               df: pd.DataFrame,
                               rs_data: Optional[Dict]) -> List[Dict]:
        """
        Check for ADL breakout/breakdown signals (existing functionality preserved).

        Args:
            ticker: Ticker symbol
            df: OHLCV DataFrame
            rs_data: Optional RS data

        Returns:
            List of breakout signal dictionaries
        """
        signals = []

        try:
            # Calculate ADL
            adl = self.calculator.calculate_adl(df)
            if adl is None or len(adl) < self.breakout_period:
                return signals

            # Add ADL to dataframe
            df_with_adl = df.copy()
            df_with_adl['ADL'] = adl

            # Check ADL breakout
            breakout = self._check_adl_breakout(ticker, df_with_adl, rs_data)
            if breakout:
                signals.append(breakout)

            # Check ADL breakdown
            breakdown = self._check_adl_breakdown(ticker, df_with_adl, rs_data)
            if breakdown:
                signals.append(breakdown)

        except Exception as e:
            logger.debug(f"Error checking breakout for {ticker}: {e}")

        return signals

    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for all results.

        Args:
            all_results: Dictionary with all result types

        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total_tickers_analyzed': (
                len(all_results.get('composite_results', [])) or
                len(all_results.get('mom_results', [])) or
                len(all_results.get('short_term_results', []))
            ),
            'composite_count': len(all_results.get('composite_results', [])),
            'mom_count': len(all_results.get('mom_results', [])),
            'short_term_count': len(all_results.get('short_term_results', [])),
            'ma_count': len(all_results.get('ma_results', [])),
            'divergence_count': len(all_results.get('divergence_results', [])),
            'breakout_count': len(all_results.get('breakout_results', [])),
            'top_candidates_count': len(all_results.get('top_candidates', []))
        }

        # Add composite score statistics if available
        if self.composite_enabled and self.composite_scorer and all_results.get('composite_results'):
            summary['composite_stats'] = self.composite_scorer.create_summary_statistics(
                all_results['composite_results']
            )

        return summary

    def _check_bullish_divergence(self, ticker: str, data: pd.DataFrame,
                                   rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """Check for bullish divergence: Price making lower lows while ADL making higher lows."""
        try:
            if len(data) < self.divergence_period:
                return None

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

            # Bullish divergence: price declining but ADL improving
            if latest_price_low < prev_price_low and latest_adl_low > prev_adl_low:
                price_decline_pct = abs((latest_price_low - prev_price_low) / prev_price_low) * 100
                adl_improvement_pct = abs((latest_adl_low - prev_adl_low) / prev_adl_low) * 100
                divergence_strength = (price_decline_pct + adl_improvement_pct) / 2

                if divergence_strength >= self.min_divergence_strength:
                    latest = data.iloc[-1]
                    return {
                        'ticker': ticker,
                        'signal_date': str(latest.name),
                        'signal_type': 'adl_bullish_divergence',
                        'price': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'adl_value': float(latest['ADL']),
                        'divergence_strength': round(divergence_strength, 2),
                        'price_decline_pct': round(price_decline_pct, 2),
                        'adl_improvement_pct': round(adl_improvement_pct, 2)
                    }

            return None
        except Exception as e:
            logger.debug(f"Error checking bullish divergence for {ticker}: {e}")
            return None

    def _check_bearish_divergence(self, ticker: str, data: pd.DataFrame,
                                   rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """Check for bearish divergence: Price making higher highs while ADL making lower highs."""
        try:
            if len(data) < self.divergence_period:
                return None

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

            # Bearish divergence: price advancing but ADL weakening
            if latest_price_high > prev_price_high and latest_adl_high < prev_adl_high:
                price_advance_pct = abs((latest_price_high - prev_price_high) / prev_price_high) * 100
                adl_weakness_pct = abs((latest_adl_high - prev_adl_high) / prev_adl_high) * 100
                divergence_strength = (price_advance_pct + adl_weakness_pct) / 2

                if divergence_strength >= self.min_divergence_strength:
                    latest = data.iloc[-1]
                    return {
                        'ticker': ticker,
                        'signal_date': str(latest.name),
                        'signal_type': 'adl_bearish_divergence',
                        'price': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'adl_value': float(latest['ADL']),
                        'divergence_strength': round(divergence_strength, 2),
                        'price_advance_pct': round(price_advance_pct, 2),
                        'adl_weakness_pct': round(adl_weakness_pct, 2)
                    }

            return None
        except Exception as e:
            logger.debug(f"Error checking bearish divergence for {ticker}: {e}")
            return None

    def _check_adl_breakout(self, ticker: str, data: pd.DataFrame,
                            rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """Check for ADL breakout above previous highs (strong accumulation signal)."""
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
                    return {
                        'ticker': ticker,
                        'signal_date': str(latest.name),
                        'signal_type': 'adl_breakout',
                        'price': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'adl_value': float(current_adl),
                        'breakout_strength': round(breakout_strength, 2),
                        'previous_high': float(max_adl_previous)
                    }

            return None
        except Exception as e:
            logger.debug(f"Error checking ADL breakout for {ticker}: {e}")
            return None

    def _check_adl_breakdown(self, ticker: str, data: pd.DataFrame,
                             rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """Check for ADL breakdown below previous lows (distribution signal)."""
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
                breakdown_strength = abs((current_adl - min_adl_previous) / abs(min_adl_previous))

                if breakdown_strength >= self.min_breakout_strength:
                    return {
                        'ticker': ticker,
                        'signal_date': str(latest.name),
                        'signal_type': 'adl_breakdown',
                        'price': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'adl_value': float(current_adl),
                        'breakdown_strength': round(breakdown_strength, 2),
                        'previous_low': float(min_adl_previous)
                    }

            return None
        except Exception as e:
            logger.debug(f"Error checking ADL breakdown for {ticker}: {e}")
            return None

    def _find_highs(self, series: pd.Series, window: int = 5) -> List[float]:
        """Find local highs in a series."""
        highs = []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                highs.append(series.iloc[i])
        return highs

    def _find_lows(self, series: pd.Series, window: int = 5) -> List[float]:
        """Find local lows in a series."""
        lows = []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                lows.append(series.iloc[i])
        return lows


def create_enhanced_screener(config: Dict[str, Any], user_config=None) -> ADLScreenerEnhanced:
    """
    Factory function to create ADL Enhanced Screener.

    Args:
        config: Configuration dictionary
        user_config: UserConfiguration object

    Returns:
        ADLScreenerEnhanced instance
    """
    return ADLScreenerEnhanced(config, user_config)