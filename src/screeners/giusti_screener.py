"""
Giusti Momentum Screener Module
===============================

Implements the Giusti momentum-based screening methodology using monthly returns
and rolling performance analysis to identify top momentum performers.

This screener uses a multi-stage filtering approach:
1. Calculate monthly returns for all stocks
2. Apply rolling 12/6/3 month performance rankings
3. Select top performers using progressive filtering

Output format matches the pattern: results_giusti_{choice}_{date}.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def giusti_screener(batch_data, params=None):
    """
    Run Giusti momentum screener on batch data.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with screener parameters
        
    Returns:
        list: List of screening results with ticker, score, and analysis data
    """
    if not batch_data:
        logger.warning("Giusti screener: No batch data provided")
        return []
    
    # Default parameters
    default_params = {
        'min_price': 5.0,
        'min_volume': 100000,
        'rolling_12m': 12,
        'rolling_6m': 6,
        'rolling_3m': 3,
        'top_12m_count': 50,
        'top_6m_count': 30,
        'top_3m_count': 10,
        'min_history_months': 12,
        'show_all_stocks': False
    }
    
    if params:
        default_params.update(params)
    
    logger.info(f"Running Giusti screener on {len(batch_data)} tickers")
    
    try:
        # Process all tickers to get monthly data
        monthly_data = {}
        valid_tickers = []
        
        for ticker, df in batch_data.items():
            if df is None or df.empty:
                continue
                
            try:
                # Basic filters
                latest_price = df['Close'].iloc[-1] if not df.empty else 0
                latest_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns and not df.empty else 0
                
                if latest_price < default_params['min_price'] or latest_volume < default_params['min_volume']:
                    continue
                
                # Convert to monthly data - resample to month-end
                if len(df) < 20:  # Need minimum data
                    continue
                    
                df_copy = df.copy()
                df_copy.index = pd.to_datetime(df_copy.index)
                
                # Monthly resampling using last trading day of month
                monthly_prices = df_copy['Close'].resample('ME').last()
                
                if len(monthly_prices) < default_params['min_history_months']:
                    continue
                    
                monthly_data[ticker] = monthly_prices
                valid_tickers.append(ticker)
                
            except Exception as e:
                logger.warning(f"Giusti screener: Error processing {ticker}: {e}")
                continue
        
        if not monthly_data:
            logger.warning("Giusti screener: No valid tickers after filtering")
            return []
        
        # Calculate monthly returns
        monthly_returns = {}
        for ticker, prices in monthly_data.items():
            returns = prices.pct_change().dropna()
            if len(returns) >= default_params['min_history_months']:
                monthly_returns[ticker] = returns
        
        if not monthly_returns:
            logger.warning("Giusti screener: No valid monthly returns calculated")
            return []
        
        # Convert to DataFrame for easier processing
        returns_df = pd.DataFrame(monthly_returns).fillna(0)
        
        # Calculate rolling performance
        rolling_12m = (1 + returns_df).rolling(default_params['rolling_12m']).apply(np.prod) - 1
        rolling_6m = (1 + returns_df).rolling(default_params['rolling_6m']).apply(np.prod) - 1
        rolling_3m = (1 + returns_df).rolling(default_params['rolling_3m']).apply(np.prod) - 1
        
        results = []
        analysis_date = datetime.now()
        
        # Get most recent complete data point
        if len(rolling_12m) > 0:
            latest_date = rolling_12m.index[-1]
            
            # Get rolling performance for latest date
            perf_12m = rolling_12m.loc[latest_date].dropna()
            perf_6m = rolling_6m.loc[latest_date].dropna()
            perf_3m = rolling_3m.loc[latest_date].dropna()
            
            # Apply Giusti methodology: progressive filtering
            # Step 1: Top performers over 12 months
            top_12m = perf_12m.nlargest(default_params['top_12m_count'])
            
            # Step 2: From top 12m performers, get top 6m performers
            top_6m_candidates = perf_6m.reindex(top_12m.index).dropna()
            top_6m = top_6m_candidates.nlargest(default_params['top_6m_count'])
            
            # Step 3: From top 6m performers, get top 3m performers
            top_3m_candidates = perf_3m.reindex(top_6m.index).dropna()
            top_3m = top_3m_candidates.nlargest(default_params['top_3m_count'])
            
            # Create results for all analyzed tickers or just top performers
            tickers_to_report = valid_tickers if default_params['show_all_stocks'] else top_3m.index.tolist()
            
            for ticker in tickers_to_report:
                try:
                    # Get performance metrics
                    perf_12m_val = perf_12m.get(ticker, 0)
                    perf_6m_val = perf_6m.get(ticker, 0)
                    perf_3m_val = perf_3m.get(ticker, 0)
                    
                    # Check if ticker passed progressive filters
                    in_top_12m = ticker in top_12m.index
                    in_top_6m = ticker in top_6m.index
                    in_top_3m = ticker in top_3m.index
                    
                    # Calculate composite score (higher is better)
                    score = 0
                    if in_top_12m:
                        score += 30
                    if in_top_6m:
                        score += 40
                    if in_top_3m:
                        score += 60  # Highest weight for final selection
                    
                    # Add performance-based scoring
                    score += min(perf_12m_val * 100, 50)  # Cap at 50 points
                    
                    # Get current price and volume for context
                    current_data = batch_data[ticker]
                    current_price = current_data['Close'].iloc[-1] if not current_data.empty else 0
                    current_volume = current_data['Volume'].iloc[-1] if 'Volume' in current_data.columns and not current_data.empty else 0
                    
                    result = {
                        'ticker': ticker,
                        'screen_type': 'giusti',
                        'score': round(score, 2),
                        'price': round(current_price, 2),
                        'volume': int(current_volume),
                        'perf_12m': round(perf_12m_val * 100, 2),  # Convert to percentage
                        'perf_6m': round(perf_6m_val * 100, 2),
                        'perf_3m': round(perf_3m_val * 100, 2),
                        'rank_12m': list(top_12m.index).index(ticker) + 1 if in_top_12m else 0,
                        'rank_6m': list(top_6m.index).index(ticker) + 1 if in_top_6m else 0,
                        'rank_3m': list(top_3m.index).index(ticker) + 1 if in_top_3m else 0,
                        'top_12m': in_top_12m,
                        'top_6m': in_top_6m,
                        'top_3m': in_top_3m,
                        'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                        'signal_day': latest_date.strftime('%Y-%m-%d'),
                        'screener': 'Giusti Momentum'
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Giusti screener: Error processing result for {ticker}: {e}")
                    continue
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Giusti screener completed: {len(results)} results generated")
        
        # Save results in Giusti-specific format if results exist
        if results:
            _save_giusti_results(results, params)
            _save_giusti_portfolio_performance(monthly_returns, params)
        
        return results
        
    except Exception as e:
        logger.error(f"Giusti screener failed: {e}")
        return []


def _save_giusti_results(results: List[Dict], params: Optional[Dict] = None) -> None:
    """
    Save Giusti screener results in the expected format.
    
    Args:
        results: List of screening results
        params: Dictionary with screener parameters including paths and config
    """
    if not params:
        logger.warning("No parameters provided for Giusti results saving")
        return
        
    try:
        # Extract parameters
        user_choice = params.get('ticker_choice', 0)
        timeframe = params.get('timeframe', 'daily')
        
        # Create giusti subdirectory in screeners folder
        screeners_dir = Path("results/screeners")
        giusti_dir = screeners_dir / 'giusti'
        giusti_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename matching expected format
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"results_giusti_{user_choice}_{timestamp}.csv"
        output_file = giusti_dir / filename
        
        if results:
            # Convert to DataFrame and save
            df = pd.DataFrame(results)
            
            # Reorder columns to match expected format
            column_order = [
                'ticker', 'price', 'volume', 'score',
                'perf_12m', 'perf_6m', 'perf_3m',
                'rank_12m', 'rank_6m', 'rank_3m',
                'top_12m', 'top_6m', 'top_3m',
                'analysis_date', 'signal_day', 'screener'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in df.columns]
            df_ordered = df[available_columns]
            
            df_ordered.to_csv(output_file, index=False)
            logger.info(f"Giusti results saved: {output_file}")
        else:
            # Create empty file with headers
            headers = [
                'ticker', 'price', 'volume', 'score',
                'perf_12m', 'perf_6m', 'perf_3m',
                'rank_12m', 'rank_6m', 'rank_3m',
                'top_12m', 'top_6m', 'top_3m',
                'analysis_date', 'signal_day', 'screener'
            ]
            
            empty_df = pd.DataFrame(columns=headers)
            empty_df.to_csv(output_file, index=False)
            logger.info(f"Giusti results (empty) saved: {output_file}")
            
    except Exception as e:
        logger.error(f"Failed to save Giusti results: {e}")


def _save_giusti_portfolio_performance(monthly_returns: Dict, params: Optional[Dict] = None) -> None:
    """
    Save Giusti portfolio performance results in historical format.
    
    Args:
        monthly_returns: Dictionary of monthly returns by ticker
        params: Dictionary with screener parameters
    """
    if not params or not monthly_returns:
        logger.warning("No parameters or monthly returns provided for portfolio performance saving")
        return
        
    try:
        # Extract parameters
        user_choice = params.get('ticker_choice', 0)
        
        # Create giusti subdirectory in screeners folder
        screeners_dir = Path("results/screeners")
        giusti_dir = screeners_dir / 'giusti'
        giusti_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename for portfolio performance
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"results_giusti_{user_choice + 1}_{timestamp}.csv"
        output_file = giusti_dir / filename
        
        # Convert monthly returns to DataFrame for processing
        returns_df = pd.DataFrame(monthly_returns).fillna(0)
        
        # Calculate rolling performance like the original Giusti class
        rolling_12m = (1 + returns_df).rolling(12).apply(np.prod) - 1
        rolling_6m = (1 + returns_df).rolling(6).apply(np.prod) - 1
        rolling_3m = (1 + returns_df).rolling(3).apply(np.prod) - 1
        
        # Generate portfolio performance data
        portfolio_results = []
        
        for date in returns_df.index:
            if pd.isna(rolling_12m.loc[date]).all():
                continue
                
            # Get performance for this date
            perf_12m = rolling_12m.loc[date].dropna()
            perf_6m = rolling_6m.loc[date].dropna()
            perf_3m = rolling_3m.loc[date].dropna()
            
            if len(perf_12m) < 10:  # Need minimum stocks
                continue
                
            # Apply progressive filtering (like original)
            top_12m = perf_12m.nlargest(50)
            top_6m_candidates = perf_6m.reindex(top_12m.index).dropna()
            top_6m = top_6m_candidates.nlargest(30)
            top_3m_candidates = perf_3m.reindex(top_6m.index).dropna()
            top_3m = top_3m_candidates.nlargest(10)
            
            # Calculate portfolio performance (next month return)
            next_month_idx = returns_df.index.get_loc(date) + 1
            if next_month_idx < len(returns_df):
                next_date = returns_df.index[next_month_idx]
                portfolio_return = returns_df.loc[next_date, top_3m.index].mean()
                portfolio_performance = 1 + portfolio_return
            else:
                portfolio_performance = 1.0
            
            portfolio_results.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Top10': list(top_3m.index),
                'Performance': portfolio_performance
            })
        
        # Save to CSV
        if portfolio_results:
            df = pd.DataFrame(portfolio_results)
            df.to_csv(output_file, index=False)
            logger.info(f"Giusti portfolio performance saved: {output_file}")
        else:
            # Create empty file with headers
            empty_df = pd.DataFrame(columns=['Date', 'Top10', 'Performance'])
            empty_df.to_csv(output_file, index=False)
            logger.info(f"Giusti portfolio performance (empty) saved: {output_file}")
            
    except Exception as e:
        logger.error(f"Failed to save Giusti portfolio performance: {e}")


# Maintain module interface consistency
__all__ = ['giusti_screener', '_save_giusti_results', '_save_giusti_portfolio_performance']