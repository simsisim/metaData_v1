"""
Stockbee Suite Screener
=======================

Comprehensive screener implementing Stockbee's four core screening strategies:
1. 9M MOVERS - High volume momentum stocks with 9M+ share volume
2. 20% Weekly Movers - Strong weekly momentum with 20%+ weekly gains  
3. 4% Daily Gainers - Daily momentum with 4%+ daily gains
4. Top 20% Industries & Top 4 Performers - Industry leadership analysis

Based on Stockbee methodology for detecting institutional activity and momentum.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class StockbeeScreener:
    """
    Main Stockbee suite screener implementing all four screening strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_stockbee_suite = config.get('enable_stockbee_suite', True)
        self.timeframe = config.get('timeframe', 'daily')
        
        # Stockbee suite specific configuration
        self.stockbee_config = config.get('stockbee_suite', {})
        self.enable_9m_movers = self.stockbee_config.get('enable_9m_movers', True)
        self.enable_weekly_movers = self.stockbee_config.get('enable_weekly_movers', True)
        self.enable_daily_gainers = self.stockbee_config.get('enable_daily_gainers', True)
        self.enable_industry_leaders = self.stockbee_config.get('enable_industry_leaders', True)
        
        # Output configuration
        self.output_dir = config.get('stockbee_output_dir', 'results/screeners/stockbee_suite')
        self.save_individual_files = self.stockbee_config.get('save_individual_files', True)
        
        # Core filtering parameters
        self.min_market_cap = self.stockbee_config.get('min_market_cap', 1_000_000_000)  # $1B
        self.min_price = self.stockbee_config.get('min_price', 5.0)
        self.exclude_funds = self.stockbee_config.get('exclude_funds', True)
        
        logger.info(f"Stockbee Suite Screener initialized (enabled: {self.enable_stockbee_suite})")

    def run_stockbee_screening(self, batch_data: Dict[str, pd.DataFrame], 
                              ticker_info: Optional[pd.DataFrame] = None,
                              rs_data: Optional[Dict] = None,
                              batch_info: Dict[str, Any] = None) -> List[Dict]:
        """
        Run comprehensive Stockbee suite screening
        
        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            ticker_info: DataFrame with ticker information (market cap, exchange, industry)
            rs_data: Dictionary with RS data for different timeframes
            batch_info: Optional batch processing information
            
        Returns:
            List of screening results
        """
        if not self.enable_stockbee_suite:
            logger.info("Stockbee suite screening disabled")
            return []
            
        if not batch_data:
            logger.warning("No data provided for Stockbee suite screening")
            return []
        
        logger.info(f"Running Stockbee suite screening on {len(batch_data)} tickers")
        
        all_results = []
        component_results = {}
        
        try:
            # Pre-filter tickers based on basic criteria (market cap, exchange, etc.)
            filtered_tickers = self._apply_base_filters(batch_data, ticker_info)
            logger.info(f"Filtered to {len(filtered_tickers)} tickers meeting base criteria")
            
            # 1. 9M MOVERS Screener
            if self.enable_9m_movers:
                logger.info("Running 9M MOVERS screener...")
                movers_9m_results = self._run_9m_movers(filtered_tickers, rs_data)
                component_results['9m_movers'] = movers_9m_results
                all_results.extend(movers_9m_results)
            
            # 2. 20% Weekly Movers Screener
            if self.enable_weekly_movers:  # Can run on any timeframe - uses weekly calculations internally
                logger.info("Running 20% Weekly Movers screener...")
                weekly_movers_results = self._run_weekly_movers(filtered_tickers, rs_data)
                component_results['weekly_movers'] = weekly_movers_results
                all_results.extend(weekly_movers_results)
            
            # 3. 4% Daily Gainers Screener
            if self.enable_daily_gainers:  # Can run on any timeframe - uses daily calculations internally
                logger.info("Running 4% Daily Gainers screener...")
                daily_gainers_results = self._run_daily_gainers(filtered_tickers, rs_data)
                component_results['daily_gainers'] = daily_gainers_results
                all_results.extend(daily_gainers_results)
            
            # 4. Top 20% Industries & Top 4 Performers
            if self.enable_industry_leaders:
                if ticker_info is not None:
                    logger.info("Running Industry Leaders screener...")
                    industry_leaders_results = self._run_industry_leaders(filtered_tickers, ticker_info, rs_data)
                    component_results['industry_leaders'] = industry_leaders_results
                    all_results.extend(industry_leaders_results)
                else:
                    logger.warning("Industry Leaders screener skipped - no ticker info available")
                    component_results['industry_leaders'] = []
            
            # Save results if enabled
            if self.save_individual_files:
                self._save_component_results(component_results)
            
            logger.info(f"Stockbee suite screening completed: {len(all_results)} total signals")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in Stockbee suite screening: {e}")
            return []

    def _apply_base_filters(self, batch_data: Dict[str, pd.DataFrame], 
                           ticker_info: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply base filters to all tickers (market cap, exchange, funds exclusion)
        
        Args:
            batch_data: Dictionary of ticker data
            ticker_info: Ticker information DataFrame
            
        Returns:
            Filtered dictionary of ticker data
        """
        filtered_data = {}
        
        for ticker, data in batch_data.items():
            try:
                # Basic data validation
                if data.empty or len(data) < 50:  # Need sufficient history
                    continue
                
                # Price filter
                current_price = data['Close'].iloc[-1]
                if current_price < self.min_price:
                    continue
                
                # Exchange and fund filtering using ticker info (if available)
                if ticker_info is not None:
                    ticker_row = ticker_info[ticker_info['ticker'] == ticker]
                    if not ticker_row.empty:
                        # Market cap filter
                        market_cap_cols = ['market_cap', 'Market_Capitalization', 'marketCap']
                        market_cap = None
                        for col in market_cap_cols:
                            if col in ticker_row.columns:
                                market_cap = ticker_row[col].iloc[0]
                                break
                        
                        if market_cap is not None and pd.notna(market_cap):
                            if market_cap < self.min_market_cap:
                                continue
                        
                        # Exchange filter (US exchanges only)
                        if 'exchange' in ticker_row.columns:
                            exchange = ticker_row['exchange'].iloc[0]
                            if pd.notna(exchange) and exchange not in ['NASDAQ', 'NYSE', 'AMEX']:
                                continue
                        
                        # Fund exclusion
                        if self.exclude_funds:
                            name = ticker_row.get('name', [''])[0] if 'name' in ticker_row.columns else ''
                            if any(fund_word in str(name).upper() for fund_word in ['ETF', 'FUND', 'TRUST']):
                                continue
                else:
                    # When no ticker info available, apply basic fund exclusion by ticker symbol
                    if self.exclude_funds:
                        if any(fund_suffix in ticker.upper() for fund_suffix in ['ETF', 'QQQ', 'SPY', 'IWM', 'XL']):
                            continue
                
                filtered_data[ticker] = data
                
            except Exception as e:
                logger.warning(f"Error filtering ticker {ticker}: {e}")
                continue
        
        return filtered_data

    def _run_9m_movers(self, batch_data: Dict[str, pd.DataFrame], 
                       rs_data: Optional[Dict] = None) -> List[Dict]:
        """
        9M MOVERS Screener
        
        Criteria:
        - Market cap $1B+ (applied in base filters)
        - Today's volume > 9M shares
        - Today's relative volume >= 1.25x (volume today / average volume)
        - Daily candle green (close > open)
        - Optionally filter out red on higher timeframes
        
        Args:
            batch_data: Filtered ticker data
            rs_data: RS data for additional scoring
            
        Returns:
            List of screening results
        """
        results = []
        volume_threshold = 9_000_000  # 9M shares
        rel_vol_threshold = 1.25
        
        for ticker, data in batch_data.items():
            try:
                if len(data) < 20:  # Need volume average
                    continue
                
                latest = data.iloc[-1]
                
                # Volume criteria
                today_volume = latest['Volume']
                if today_volume < volume_threshold:
                    continue
                
                # Relative volume calculation (20-day average)
                avg_volume = data['Volume'].tail(20).mean()
                relative_volume = today_volume / avg_volume if avg_volume > 0 else 0
                
                if relative_volume < rel_vol_threshold:
                    continue
                
                # Green daily candle
                if latest['Close'] <= latest['Open']:
                    continue
                
                # Calculate additional metrics
                price_change_pct = ((latest['Close'] - latest['Open']) / latest['Open']) * 100
                
                # Get RS score if available
                rs_score = self._get_rs_score(ticker, rs_data, 'daily', 1) if rs_data else None
                
                result = {
                    'ticker': ticker,
                    'signal_date': latest.name,
                    'signal_type': '9m_movers',
                    'screen_type': 'stockbee_9m_movers',
                    'price': latest['Close'],
                    'volume': today_volume,
                    'relative_volume': relative_volume,
                    'price_change_pct': price_change_pct,
                    'avg_volume_20d': avg_volume,
                    'rs_score': rs_score,
                    'strength': self._calculate_signal_strength(relative_volume, price_change_pct),
                    'raw_data': latest.to_dict()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing {ticker} in 9M movers: {e}")
                continue
        
        # Sort by relative volume descending
        results.sort(key=lambda x: x['relative_volume'], reverse=True)
        logger.info(f"9M MOVERS: {len(results)} signals found")
        
        return results

    def _run_weekly_movers(self, batch_data: Dict[str, pd.DataFrame], 
                          rs_data: Optional[Dict] = None) -> List[Dict]:
        """
        20% Weekly Movers Screener
        
        Criteria:
        - Market cap $1B+ (applied in base filters)
        - Price change (%) this week >= +20%
        - Weekly relative volume >= 1.25x (current week / average week)
        - Weekly average volume >100k shares
        - Weekly candle green, preferably above key moving averages
        
        Args:
            batch_data: Filtered ticker data
            rs_data: RS data for additional scoring
            
        Returns:
            List of screening results
        """
        results = []
        weekly_gain_threshold = 20.0  # 20%
        weekly_rel_vol_threshold = 1.25
        min_weekly_avg_volume = 100_000
        
        for ticker, data in batch_data.items():
            try:
                if len(data) < 20:  # Need sufficient history
                    continue
                
                # Get week-to-date data (last 5 trading days)
                weekly_data = data.tail(5)
                if len(weekly_data) < 3:  # Need at least 3 days
                    continue
                
                week_start_price = weekly_data['Open'].iloc[0]
                week_end_price = weekly_data['Close'].iloc[-1]
                week_high = weekly_data['High'].max()
                week_low = weekly_data['Low'].min()
                
                # Weekly gain calculation
                weekly_gain_pct = ((week_end_price - week_start_price) / week_start_price) * 100
                if weekly_gain_pct < weekly_gain_threshold:
                    continue
                
                # Weekly volume metrics
                weekly_avg_volume = weekly_data['Volume'].mean()
                if weekly_avg_volume < min_weekly_avg_volume:
                    continue
                
                # Weekly relative volume (compare to 4-week average)
                historical_weekly_avg = data['Volume'].tail(20).mean()
                weekly_rel_volume = weekly_avg_volume / historical_weekly_avg if historical_weekly_avg > 0 else 0
                
                if weekly_rel_volume < weekly_rel_vol_threshold:
                    continue
                
                # Green weekly candle
                if week_end_price <= week_start_price:
                    continue
                
                # Moving average analysis (SMA50, SMA200)
                sma_50 = data['Close'].tail(50).mean() if len(data) >= 50 else None
                sma_200 = data['Close'].tail(200).mean() if len(data) >= 200 else None
                above_sma50 = week_end_price > sma_50 if sma_50 else None
                above_sma200 = week_end_price > sma_200 if sma_200 else None
                
                # Get RS score if available
                rs_score = self._get_rs_score(ticker, rs_data, 'weekly', 1) if rs_data else None
                
                result = {
                    'ticker': ticker,
                    'signal_date': weekly_data.index[-1],
                    'signal_type': '20pct_weekly_movers',
                    'screen_type': 'stockbee_weekly_movers',
                    'price': week_end_price,
                    'volume': weekly_avg_volume,
                    'weekly_gain_pct': weekly_gain_pct,
                    'weekly_rel_volume': weekly_rel_volume,
                    'week_high': week_high,
                    'week_low': week_low,
                    'above_sma50': above_sma50,
                    'above_sma200': above_sma200,
                    'rs_score': rs_score,
                    'strength': self._calculate_signal_strength(weekly_rel_volume, weekly_gain_pct),
                    'raw_data': weekly_data.iloc[-1].to_dict()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing {ticker} in weekly movers: {e}")
                continue
        
        # Sort by weekly gain percentage descending
        results.sort(key=lambda x: x['weekly_gain_pct'], reverse=True)
        logger.info(f"20% WEEKLY MOVERS: {len(results)} signals found")
        
        return results

    def _run_daily_gainers(self, batch_data: Dict[str, pd.DataFrame], 
                          rs_data: Optional[Dict] = None) -> List[Dict]:
        """
        4% Daily Gainers Screener
        
        Criteria:
        - Market cap $1B+ (applied in base filters)
        - Price change today >= +4%
        - Relative volume >= 1.5-2x
        - Daily volume >100k
        - Price > SMA50/200, high RS, green candle
        
        Args:
            batch_data: Filtered ticker data
            rs_data: RS data for additional scoring
            
        Returns:
            List of screening results
        """
        results = []
        daily_gain_threshold = 4.0  # 4%
        rel_vol_threshold = 1.5
        min_daily_volume = 100_000
        
        for ticker, data in batch_data.items():
            try:
                if len(data) < 50:  # Need history for moving averages
                    continue
                
                latest = data.iloc[-1]
                previous = data.iloc[-2]
                
                # Daily gain calculation
                daily_gain_pct = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
                if daily_gain_pct < daily_gain_threshold:
                    continue
                
                # Volume criteria
                today_volume = latest['Volume']
                if today_volume < min_daily_volume:
                    continue
                
                # Relative volume (20-day average)
                avg_volume = data['Volume'].tail(20).mean()
                relative_volume = today_volume / avg_volume if avg_volume > 0 else 0
                
                if relative_volume < rel_vol_threshold:
                    continue
                
                # Green candle
                if latest['Close'] <= latest['Open']:
                    continue
                
                # Moving average analysis
                sma_50 = data['Close'].tail(50).mean()
                sma_200 = data['Close'].tail(200).mean() if len(data) >= 200 else None
                
                price_above_sma50 = latest['Close'] > sma_50
                price_above_sma200 = latest['Close'] > sma_200 if sma_200 else None
                
                # Require price above key moving averages
                if not price_above_sma50:
                    continue
                
                # Get RS score (high RS preferred)
                rs_score = self._get_rs_score(ticker, rs_data, 'daily', 1) if rs_data else None
                high_rs = rs_score is not None and rs_score > 70
                
                result = {
                    'ticker': ticker,
                    'signal_date': latest.name,
                    'signal_type': '4pct_daily_gainers',
                    'screen_type': 'stockbee_daily_gainers',
                    'price': latest['Close'],
                    'volume': today_volume,
                    'daily_gain_pct': daily_gain_pct,
                    'relative_volume': relative_volume,
                    'price_above_sma50': price_above_sma50,
                    'price_above_sma200': price_above_sma200,
                    'high_rs': high_rs,
                    'rs_score': rs_score,
                    'strength': self._calculate_signal_strength(relative_volume, daily_gain_pct),
                    'raw_data': latest.to_dict()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing {ticker} in daily gainers: {e}")
                continue
        
        # Sort by daily gain percentage descending
        results.sort(key=lambda x: x['daily_gain_pct'], reverse=True)
        logger.info(f"4% DAILY GAINERS: {len(results)} signals found")
        
        return results

    def _run_industry_leaders(self, batch_data: Dict[str, pd.DataFrame],
                             ticker_info: pd.DataFrame,
                             rs_data: Optional[Dict] = None) -> List[Dict]:
        """
        Top 20% Industries & Top 4 Performers Screener
        
        Criteria:
        - Calculate RS on day/week/month/quarter for each industry
        - Rank industries, select top ~30 (out of 145)
        - For each selected industry, find top 4 stocks by RS and/or momentum metrics
        - Color-coding for leadership status
        
        Args:
            batch_data: Filtered ticker data
            ticker_info: Ticker information with industry classifications
            rs_data: RS data for ranking
            
        Returns:
            List of screening results
        """
        results = []
        
        try:
            # Calculate industry RS rankings
            industry_rankings = self._calculate_industry_rankings(batch_data, ticker_info, rs_data)
            
            if not industry_rankings:
                logger.warning("No industry rankings calculated")
                return results
            
            # Select top 20% of industries
            num_top_industries = max(1, int(len(industry_rankings) * 0.2))
            top_industries = industry_rankings[:num_top_industries]
            
            logger.info(f"Selected top {len(top_industries)} industries from {len(industry_rankings)} total")
            
            # For each top industry, find top 4 performers
            for industry_data in top_industries:
                industry_name = industry_data['industry']
                industry_tickers = industry_data['tickers']
                
                # Calculate individual stock metrics for this industry
                industry_stock_metrics = []
                
                for ticker in industry_tickers:
                    if ticker in batch_data:
                        try:
                            stock_metrics = self._calculate_stock_metrics(ticker, batch_data[ticker], rs_data)
                            if stock_metrics:
                                stock_metrics['industry'] = industry_name
                                stock_metrics['industry_rs_rank'] = industry_data['rs_rank']
                                industry_stock_metrics.append(stock_metrics)
                        except Exception as e:
                            logger.warning(f"Error calculating metrics for {ticker}: {e}")
                            continue
                
                # Sort by composite score and select top 4
                industry_stock_metrics.sort(key=lambda x: x['composite_score'], reverse=True)
                top_4_stocks = industry_stock_metrics[:4]
                
                # Format results
                for i, stock in enumerate(top_4_stocks):
                    leadership_level = ['leader', 'strong', 'moderate', 'emerging'][i]
                    
                    result = {
                        'ticker': stock['ticker'],
                        'signal_date': stock['signal_date'],
                        'signal_type': 'industry_leader',
                        'screen_type': 'stockbee_industry_leaders',
                        'price': stock['price'],
                        'volume': stock['volume'],
                        'industry': industry_name,
                        'industry_rank': industry_data['rs_rank'],
                        'stock_rank_in_industry': i + 1,
                        'leadership_level': leadership_level,
                        'rs_score': stock['rs_score'],
                        'momentum_score': stock['momentum_score'],
                        'composite_score': stock['composite_score'],
                        'strength': leadership_level,
                        'raw_data': stock
                    }
                    
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Error in industry leaders screening: {e}")
        
        logger.info(f"INDUSTRY LEADERS: {len(results)} signals found")
        return results

    def _calculate_industry_rankings(self, batch_data: Dict[str, pd.DataFrame],
                                   ticker_info: pd.DataFrame,
                                   rs_data: Optional[Dict] = None) -> List[Dict]:
        """
        Calculate industry RS rankings for top industry selection
        
        Returns:
            List of industries sorted by RS performance
        """
        try:
            # Group tickers by industry
            industry_groups = {}
            for ticker in batch_data.keys():
                ticker_row = ticker_info[ticker_info['ticker'] == ticker]
                if not ticker_row.empty and 'industry' in ticker_row.columns:
                    industry = ticker_row['industry'].iloc[0]
                    if pd.notna(industry):
                        if industry not in industry_groups:
                            industry_groups[industry] = []
                        industry_groups[industry].append(ticker)
            
            # Calculate composite RS for each industry
            industry_rankings = []
            
            for industry, tickers in industry_groups.items():
                if len(tickers) < 3:  # Need minimum industry size
                    continue
                
                # Calculate average RS scores across multiple timeframes
                rs_scores = []
                for ticker in tickers:
                    if rs_data:
                        # Get RS scores for different periods
                        daily_rs = self._get_rs_score(ticker, rs_data, 'daily', 1)
                        weekly_rs = self._get_rs_score(ticker, rs_data, 'weekly', 4)
                        monthly_rs = self._get_rs_score(ticker, rs_data, 'monthly', 3)
                        
                        # Create composite RS score
                        valid_scores = [score for score in [daily_rs, weekly_rs, monthly_rs] if score is not None]
                        if valid_scores:
                            composite_rs = sum(valid_scores) / len(valid_scores)
                            rs_scores.append(composite_rs)
                
                if rs_scores:
                    industry_avg_rs = sum(rs_scores) / len(rs_scores)
                    industry_rankings.append({
                        'industry': industry,
                        'avg_rs': industry_avg_rs,
                        'ticker_count': len(tickers),
                        'tickers': tickers
                    })
            
            # Sort by average RS descending
            industry_rankings.sort(key=lambda x: x['avg_rs'], reverse=True)
            
            # Add rankings
            for i, industry_data in enumerate(industry_rankings):
                industry_data['rs_rank'] = i + 1
            
            return industry_rankings
            
        except Exception as e:
            logger.error(f"Error calculating industry rankings: {e}")
            return []

    def _calculate_stock_metrics(self, ticker: str, data: pd.DataFrame, 
                               rs_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Calculate comprehensive stock metrics for industry leader selection
        
        Returns:
            Dictionary with stock metrics or None if insufficient data
        """
        try:
            if len(data) < 50:
                return None
            
            latest = data.iloc[-1]
            
            # Price metrics
            price_change_1d = ((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close']) * 100
            price_change_5d = ((latest['Close'] - data.iloc[-6]['Close']) / data.iloc[-6]['Close']) * 100 if len(data) >= 6 else 0
            price_change_20d = ((latest['Close'] - data.iloc[-21]['Close']) / data.iloc[-21]['Close']) * 100 if len(data) >= 21 else 0
            
            # Volume metrics
            avg_volume = data['Volume'].tail(20).mean()
            relative_volume = latest['Volume'] / avg_volume if avg_volume > 0 else 0
            
            # Moving averages
            sma_50 = data['Close'].tail(50).mean()
            sma_200 = data['Close'].tail(200).mean() if len(data) >= 200 else sma_50
            
            # RS scores
            rs_1d = self._get_rs_score(ticker, rs_data, 'daily', 1) if rs_data else 50
            rs_1w = self._get_rs_score(ticker, rs_data, 'weekly', 1) if rs_data else 50
            rs_1m = self._get_rs_score(ticker, rs_data, 'monthly', 1) if rs_data else 50
            
            # Calculate momentum score (0-100)
            momentum_score = (
                (price_change_1d * 0.3) +
                (price_change_5d * 0.4) +
                (price_change_20d * 0.3)
            )
            
            # Calculate composite score
            composite_score = (
                (rs_1d * 0.4) +
                (rs_1w * 0.3) +
                (rs_1m * 0.2) +
                (min(momentum_score * 2, 100) * 0.1)  # Cap momentum contribution
            )
            
            return {
                'ticker': ticker,
                'signal_date': latest.name,
                'price': latest['Close'],
                'volume': latest['Volume'],
                'relative_volume': relative_volume,
                'price_change_1d': price_change_1d,
                'price_change_5d': price_change_5d,
                'price_change_20d': price_change_20d,
                'above_sma50': latest['Close'] > sma_50,
                'above_sma200': latest['Close'] > sma_200,
                'rs_score': rs_1d,
                'rs_1w': rs_1w,
                'rs_1m': rs_1m,
                'momentum_score': momentum_score,
                'composite_score': composite_score
            }
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for {ticker}: {e}")
            return None

    def _get_rs_score(self, ticker: str, rs_data: Optional[Dict], 
                     timeframe: str, period: int) -> Optional[float]:
        """
        Extract RS score for a ticker from RS data
        
        Args:
            ticker: Ticker symbol
            rs_data: RS data dictionary
            timeframe: Timeframe for RS data
            period: Period number
            
        Returns:
            RS score or None if not available
        """
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

    def _calculate_signal_strength(self, relative_volume: float, price_change_pct: float) -> str:
        """
        Calculate signal strength based on volume and price metrics
        
        Args:
            relative_volume: Relative volume multiplier
            price_change_pct: Price change percentage
            
        Returns:
            Signal strength category
        """
        # Volume strength
        vol_strength = 0
        if relative_volume >= 5.0:
            vol_strength = 3
        elif relative_volume >= 3.0:
            vol_strength = 2
        elif relative_volume >= 2.0:
            vol_strength = 1
        
        # Price strength
        price_strength = 0
        if price_change_pct >= 10.0:
            price_strength = 3
        elif price_change_pct >= 7.0:
            price_strength = 2
        elif price_change_pct >= 4.0:
            price_strength = 1
        
        # Combined strength
        total_strength = vol_strength + price_strength
        
        if total_strength >= 5:
            return 'very_strong'
        elif total_strength >= 3:
            return 'strong'
        elif total_strength >= 2:
            return 'moderate'
        else:
            return 'weak'

    def _save_component_results(self, component_results: Dict[str, Any]):
        """Save individual component results to files"""
        try:
            # Create output directory
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save each component's results
            for component_name, results in component_results.items():
                if results:
                    filename = f"stockbee_{component_name}_{self.timeframe}_{timestamp}.csv"
                    filepath = output_dir / filename
                    
                    # Convert to DataFrame and save
                    if isinstance(results, list) and results:
                        df = pd.DataFrame(results)
                        df.to_csv(filepath, index=False)
                        logger.info(f"Saved {component_name} results: {len(df)} signals to {filepath}")
                    
        except Exception as e:
            logger.error(f"Error saving component results: {e}")


def run_stockbee_suite_screener(batch_data: Dict[str, pd.DataFrame], 
                               config: Dict[str, Any],
                               ticker_info: Optional[pd.DataFrame] = None,
                               rs_data: Optional[Dict] = None) -> List[Dict]:
    """
    Main entry point for Stockbee suite screening
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        config: Configuration dictionary
        ticker_info: Optional ticker information DataFrame
        rs_data: Optional RS data for enhanced screening
        
    Returns:
        List of screening results
    """
    screener = StockbeeScreener(config)
    return screener.run_stockbee_screening(batch_data, ticker_info, rs_data)


# Export main functions
__all__ = ['StockbeeScreener', 'run_stockbee_suite_screener']