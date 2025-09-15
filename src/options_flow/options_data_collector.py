#!/usr/bin/env python3
"""
Options Data Collector
======================

Collects real-time options chain data using yfinance API.
Handles data validation, cleaning, and preparation for GEX analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

class OptionsDataCollector:
    """Collect and process options chain data for flow analysis"""
    
    def __init__(self, rate_limit_delay=0.1):
        """
        Initialize options data collector
        
        Args:
            rate_limit_delay: Delay between API calls to respect rate limits
        """
        self.rate_limit_delay = rate_limit_delay
        self.session_calls = 0
        self.last_call_time = None
        
        # Data quality filters
        self.quality_filters = {
            'min_bid': 0.01,           # Minimum bid price
            'max_bid_ask_spread': 0.50, # Maximum bid-ask spread
            'min_volume': 1,           # Minimum daily volume
            'min_open_interest': 10,   # Minimum open interest
            'max_dte': 60             # Maximum days to expiration
        }
    
    def get_options_chain(self, ticker: str, include_greeks: bool = True) -> Dict:
        """
        Get complete options chain for ticker
        
        Args:
            ticker: Stock ticker symbol
            include_greeks: Whether to include Greek calculations
            
        Returns:
            Dictionary with options chain data and metadata
        """
        try:
            self._rate_limit()
            
            # Get ticker object
            stock = yf.Ticker(ticker)
            
            # Get current stock info
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            
            if current_price == 0:
                # Try to get price from history
                hist = stock.history(period='1d')
                current_price = hist['Close'].iloc[-1] if not hist.empty else 0
            
            if current_price == 0:
                return {'success': False, 'error': f'Could not get current price for {ticker}'}
            
            # Get options expirations
            expirations = stock.options
            
            if not expirations:
                return {'success': False, 'error': f'No options available for {ticker}'}
            
            # Collect options data for all expirations
            all_options = []
            
            for exp_date in expirations[:4]:  # Limit to first 4 expirations
                try:
                    option_chain = stock.option_chain(exp_date)
                    
                    # Process calls
                    calls = option_chain.calls.copy()
                    calls['type'] = 'call'
                    calls['expiration'] = exp_date
                    calls = self._clean_options_data(calls, current_price)
                    
                    # Process puts
                    puts = option_chain.puts.copy()
                    puts['type'] = 'put'
                    puts['expiration'] = exp_date
                    puts = self._clean_options_data(puts, current_price)
                    
                    # Combine and append
                    combined = pd.concat([calls, puts], ignore_index=True)
                    all_options.append(combined)
                    
                except Exception as e:
                    logger.warning(f"Error getting options for {ticker} exp {exp_date}: {e}")
                    continue
            
            if not all_options:
                return {'success': False, 'error': f'No valid options data for {ticker}'}
            
            # Combine all expirations
            complete_chain = pd.concat(all_options, ignore_index=True)
            
            # Calculate additional metrics
            complete_chain = self._add_derived_metrics(complete_chain, current_price)
            
            # Apply quality filters
            filtered_chain = self._apply_quality_filters(complete_chain)
            
            return {
                'success': True,
                'ticker': ticker,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'options_chain': filtered_chain,
                'expirations_analyzed': len(all_options),
                'total_contracts': len(filtered_chain),
                'calls_count': len(filtered_chain[filtered_chain['type'] == 'call']),
                'puts_count': len(filtered_chain[filtered_chain['type'] == 'put']),
                'data_quality_score': self._calculate_data_quality_score(filtered_chain)
            }
            
        except Exception as e:
            logger.error(f"Error collecting options data for {ticker}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        
        if self.last_call_time and (current_time - self.last_call_time) < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - (current_time - self.last_call_time)
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
        self.session_calls += 1
    
    def _clean_options_data(self, options_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
        """Clean and standardize options data"""
        
        if options_df.empty:
            return options_df
        
        # Rename columns to standard format
        column_mapping = {
            'openInterest': 'openInterest',
            'volume': 'volume', 
            'strike': 'strike',
            'bid': 'bid',
            'ask': 'ask',
            'lastPrice': 'lastPrice',
            'impliedVolatility': 'impliedVolatility'
        }
        
        # Ensure required columns exist
        for old_col, new_col in column_mapping.items():
            if old_col in options_df.columns:
                options_df = options_df.rename(columns={old_col: new_col})
        
        # Fill missing values
        options_df['openInterest'] = options_df['openInterest'].fillna(0)
        options_df['volume'] = options_df['volume'].fillna(0)
        options_df['bid'] = options_df['bid'].fillna(0)
        options_df['ask'] = options_df['ask'].fillna(0)
        
        # Calculate derived metrics
        options_df['midPrice'] = (options_df['bid'] + options_df['ask']) / 2
        options_df['bidAskSpread'] = options_df['ask'] - options_df['bid']
        options_df['moneyness'] = options_df['strike'] / current_price
        
        # Calculate days to expiration
        if 'expiration' in options_df.columns:
            exp_dates = pd.to_datetime(options_df['expiration'])
            options_df['daysToExpiration'] = (exp_dates - datetime.now()).dt.days
        
        # Estimate Greeks if not available
        if 'gamma' not in options_df.columns:
            options_df['gamma'] = options_df.apply(
                lambda row: self._estimate_gamma(current_price, row['strike'], row['type'], 
                                               row.get('daysToExpiration', 30)), axis=1
            )
        
        return options_df
    
    def _estimate_gamma(self, spot: float, strike: float, option_type: str, dte: int) -> float:
        """Estimate gamma using simplified Black-Scholes approximation"""
        
        # Moneyness calculation
        moneyness = strike / spot
        
        # Time decay factor
        time_factor = max(0.1, dte / 365.0)
        
        # Simplified gamma estimation
        if 0.95 <= moneyness <= 1.05:  # ATM
            base_gamma = 0.05 * time_factor
        elif 0.90 <= moneyness <= 1.10:  # Near money
            base_gamma = 0.03 * time_factor
        elif 0.85 <= moneyness <= 1.15:  # Slightly OTM/ITM
            base_gamma = 0.02 * time_factor
        else:  # Far OTM/ITM
            base_gamma = 0.01 * time_factor
        
        return base_gamma
    
    def _add_derived_metrics(self, options_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
        """Add calculated metrics for flow analysis"""
        
        if options_df.empty:
            return options_df
        
        # Dollar volume
        options_df['dollarVolume'] = options_df['volume'] * options_df['midPrice'] * 100
        
        # Notional open interest
        options_df['notionalOI'] = options_df['openInterest'] * options_df['midPrice'] * 100
        
        # Volume to OI ratio
        options_df['volumeOIRatio'] = options_df['volume'] / np.maximum(options_df['openInterest'], 1)
        
        # Distance from spot (percentage)
        options_df['distanceFromSpotPct'] = ((options_df['strike'] - current_price) / current_price) * 100
        
        # Liquidity score (simple metric)
        options_df['liquidityScore'] = (
            options_df['volume'] * 0.4 + 
            options_df['openInterest'] * 0.3 + 
            (1 / np.maximum(options_df['bidAskSpread'], 0.01)) * 0.3
        )
        
        return options_df
    
    def _apply_quality_filters(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to options data"""
        
        if options_df.empty:
            return options_df
        
        # Apply filters
        filtered = options_df[
            (options_df['bid'] >= self.quality_filters['min_bid']) &
            (options_df['bidAskSpread'] <= self.quality_filters['max_bid_ask_spread']) &
            (options_df['volume'] >= self.quality_filters['min_volume']) &
            (options_df['openInterest'] >= self.quality_filters['min_open_interest']) &
            (options_df['daysToExpiration'] <= self.quality_filters['max_dte']) &
            (options_df['daysToExpiration'] > 0)
        ].copy()
        
        logger.info(f"Options quality filter: {len(options_df)} → {len(filtered)} contracts")
        
        return filtered
    
    def _calculate_data_quality_score(self, options_df: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)"""
        
        if options_df.empty:
            return 0.0
        
        # Quality metrics
        has_volume = (options_df['volume'] > 0).mean()
        has_oi = (options_df['openInterest'] > 0).mean()
        reasonable_spreads = (options_df['bidAskSpread'] <= 0.50).mean()
        has_greeks = 'gamma' in options_df.columns
        
        # Composite score
        quality_score = (
            has_volume * 0.3 +
            has_oi * 0.3 +
            reasonable_spreads * 0.2 +
            has_greeks * 0.2
        )
        
        return min(1.0, quality_score)
    
    def get_batch_options_data(self, tickers: List[str], max_workers: int = 3) -> Dict[str, Dict]:
        """
        Collect options data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            max_workers: Maximum concurrent API calls
            
        Returns:
            Dictionary mapping tickers to their options data
        """
        results = {}
        
        logger.info(f"Collecting options data for {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers):
            try:
                logger.debug(f"Processing {ticker} ({i+1}/{len(tickers)})")
                
                options_result = self.get_options_chain(ticker)
                results[ticker] = options_result
                
                if options_result.get('success'):
                    logger.info(f"✅ {ticker}: {options_result['total_contracts']} contracts collected")
                else:
                    logger.warning(f"❌ {ticker}: {options_result.get('error', 'Unknown error')}")
                
                # Rate limiting between tickers
                if i < len(tickers) - 1:
                    time.sleep(self.rate_limit_delay * 2)
                    
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                results[ticker] = {'success': False, 'error': str(e)}
        
        success_count = sum(1 for r in results.values() if r.get('success'))
        logger.info(f"Options data collection completed: {success_count}/{len(tickers)} successful")
        
        return results
    
    def get_liquid_options_universe(self, min_dollar_volume: float = 100000) -> List[str]:
        """Get list of tickers with liquid options markets"""
        
        # High-volume tickers with active options
        liquid_universe = [
            # Major indices
            'SPY', 'QQQ', 'IWM', 'DIA',
            
            # Mega caps
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            
            # Popular options names
            'AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'COIN', 'PLTR',
            'GME', 'AMC', 'BB', 'NOK', 'SNDL',
            
            # Bank and finance
            'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
            
            # Technology
            'ORCL', 'IBM', 'INTC', 'CSCO', 'CRWD', 'SNOW',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB',
            
            # Sector ETFs
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE'
        ]
        
        return liquid_universe
    
    def validate_options_data(self, options_result: Dict) -> Dict:
        """Validate options data quality and completeness"""
        
        if not options_result.get('success'):
            return {'valid': False, 'reason': 'Data collection failed'}
        
        options_df = options_result['options_chain']
        
        if options_df.empty:
            return {'valid': False, 'reason': 'Empty options chain'}
        
        # Check required columns
        required_columns = ['strike', 'volume', 'openInterest', 'type', 'gamma']
        missing_columns = [col for col in required_columns if col not in options_df.columns]
        
        if missing_columns:
            return {'valid': False, 'reason': f'Missing columns: {missing_columns}'}
        
        # Check data quality
        quality_score = options_result.get('data_quality_score', 0)
        
        if quality_score < 0.3:
            return {'valid': False, 'reason': f'Low data quality: {quality_score:.2f}'}
        
        # Check contract counts
        total_contracts = len(options_df)
        if total_contracts < 10:
            return {'valid': False, 'reason': f'Insufficient contracts: {total_contracts}'}
        
        return {
            'valid': True,
            'quality_score': quality_score,
            'total_contracts': total_contracts,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def get_options_summary_stats(self, options_result: Dict) -> Dict:
        """Generate summary statistics for options data"""
        
        if not options_result.get('success'):
            return {}
        
        options_df = options_result['options_chain']
        
        if options_df.empty:
            return {}
        
        calls = options_df[options_df['type'] == 'call']
        puts = options_df[options_df['type'] == 'put']
        
        summary = {
            'total_volume': options_df['volume'].sum(),
            'total_open_interest': options_df['openInterest'].sum(),
            'total_dollar_volume': options_df['dollarVolume'].sum(),
            'call_volume': calls['volume'].sum(),
            'put_volume': puts['volume'].sum(),
            'call_put_volume_ratio': calls['volume'].sum() / max(puts['volume'].sum(), 1),
            'call_oi': calls['openInterest'].sum(),
            'put_oi': puts['openInterest'].sum(),
            'call_put_oi_ratio': calls['openInterest'].sum() / max(puts['openInterest'].sum(), 1),
            'avg_implied_volatility': options_df['impliedVolatility'].mean(),
            'most_active_strike': options_df.loc[options_df['volume'].idxmax(), 'strike'] if not options_df.empty else 0,
            'highest_oi_strike': options_df.loc[options_df['openInterest'].idxmax(), 'strike'] if not options_df.empty else 0
        }
        
        return summary
    
    def detect_unusual_volume(self, options_result: Dict, volume_threshold: float = 300.0) -> List[Dict]:
        """Detect options contracts with unusual volume activity"""
        
        unusual_contracts = []
        
        if not options_result.get('success'):
            return unusual_contracts
        
        options_df = options_result['options_chain']
        
        # Calculate volume percentiles for unusual detection
        volume_95th = options_df['volume'].quantile(0.95)
        dollar_volume_95th = options_df['dollarVolume'].quantile(0.95)
        
        # Find unusual contracts
        unusual = options_df[
            (options_df['volume'] >= volume_95th) |
            (options_df['dollarVolume'] >= dollar_volume_95th) |
            (options_df['volumeOIRatio'] >= 2.0)  # Volume > 2x open interest
        ].copy()
        
        for _, contract in unusual.iterrows():
            unusual_score = self._calculate_unusual_score(contract, options_df)
            
            if unusual_score >= 0.7:  # 70% threshold for "unusual"
                unusual_contracts.append({
                    'ticker': options_result['ticker'],
                    'strike': contract['strike'],
                    'type': contract['type'],
                    'expiration': contract['expiration'],
                    'volume': contract['volume'],
                    'open_interest': contract['openInterest'],
                    'dollar_volume': contract['dollarVolume'],
                    'unusual_score': unusual_score,
                    'volume_oi_ratio': contract['volumeOIRatio'],
                    'distance_from_spot_pct': contract['distanceFromSpotPct'],
                    'detection_timestamp': datetime.now().isoformat()
                })
        
        # Sort by unusual score
        unusual_contracts.sort(key=lambda x: x['unusual_score'], reverse=True)
        
        return unusual_contracts
    
    def _calculate_unusual_score(self, contract: pd.Series, full_chain: pd.DataFrame) -> float:
        """Calculate how unusual this contract's activity is (0-1 scale)"""
        
        # Volume percentile
        volume_percentile = (full_chain['volume'] <= contract['volume']).mean()
        
        # Dollar volume percentile  
        dollar_vol_percentile = (full_chain['dollarVolume'] <= contract['dollarVolume']).mean()
        
        # Volume to OI ratio (high suggests new positions)
        vol_oi_score = min(1.0, contract['volumeOIRatio'] / 5.0)
        
        # Distance from spot (ATM options more significant)
        distance_score = max(0.1, 1.0 - abs(contract['distanceFromSpotPct']) / 20.0)
        
        # Composite unusual score
        unusual_score = (
            volume_percentile * 0.35 +
            dollar_vol_percentile * 0.35 +
            vol_oi_score * 0.20 +
            distance_score * 0.10
        )
        
        return min(1.0, unusual_score)


def collect_options_data_for_tickers(tickers: List[str]) -> Dict[str, Dict]:
    """
    Main entry point for batch options data collection
    
    Args:
        tickers: List of ticker symbols to collect options data for
        
    Returns:
        Dictionary mapping tickers to their options data results
    """
    collector = OptionsDataCollector()
    
    # Filter to liquid universe if too many tickers provided
    if len(tickers) > 50:
        liquid_universe = collector.get_liquid_options_universe()
        tickers = [t for t in tickers if t in liquid_universe]
        logger.info(f"Filtered to liquid options universe: {len(tickers)} tickers")
    
    # Collect data
    results = collector.get_batch_options_data(tickers)
    
    return results