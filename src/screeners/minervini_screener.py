"""
Minervini Template Screener
===========================

Implements Mark Minervini's 8-criteria screener based on "Trade Like a Stock Market Wizard".
Identifies stocks with strong technical setups using price action and RS data.

8 Minervini Criteria:
1. Current price > 150 SMA and > 200 SMA  
2. 150 SMA > 200 SMA
3. 200 SMA trending up for at least 1 month
4. 50 EMA > 150 SMA and 50 EMA > 200 SMA
5. Current price > 50 EMA
6. Current price at least 30% above 52-week low
7. Current price within 25% of 52-week high  
8. RS Rating > 70 (uses RS data from RS module)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def _calculate_minervini_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Minervini technical indicators.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    try:
        df = data.copy()
        
        # Moving averages
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # 52-week high/low
        df['High_52W'] = df['Close'].rolling(window=252).max()
        df['Low_52W'] = df['Close'].rolling(window=252).min()
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating Minervini indicators: {e}")
        raise


def _evaluate_minervini_criteria(data: pd.DataFrame, rs_rating: float) -> Dict:
    """
    Evaluate all 8 Minervini criteria for a stock.
    
    Args:
        data: DataFrame with calculated indicators
        rs_rating: RS Rating from RS module
        
    Returns:
        Dictionary with criterion evaluations and pass/fail status
    """
    try:
        current_row = data.iloc[-1]
        current_price = current_row['Close']
        
        # Criterion 1: Current price > 150 SMA and > 200 SMA
        criterion1_150 = current_price > current_row['SMA_150']
        criterion1_200 = current_price > current_row['SMA_200']
        criterion1 = criterion1_150 and criterion1_200
        criterion1_score = min(current_price / current_row['SMA_150'], current_price / current_row['SMA_200'])
        
        # Criterion 2: 150 SMA > 200 SMA
        criterion2 = current_row['SMA_150'] > current_row['SMA_200']
        criterion2_score = current_row['SMA_150'] / current_row['SMA_200']
        
        # Criterion 3: 200 SMA trending up for at least 1 month (22 trading days)
        if len(data) >= 22:
            sma200_22_ago = data['SMA_200'].iloc[-22]
            criterion3 = current_row['SMA_200'] > sma200_22_ago
            criterion3_score = current_row['SMA_200'] / sma200_22_ago
        else:
            criterion3 = False
            criterion3_score = 0.0
        
        # Criterion 4: 50 EMA > 150 SMA and 50 EMA > 200 SMA
        criterion4_150 = current_row['EMA_50'] > current_row['SMA_150']
        criterion4_200 = current_row['EMA_50'] > current_row['SMA_200']
        criterion4 = criterion4_150 and criterion4_200
        criterion4_score = min(current_row['EMA_50'] / current_row['SMA_150'], current_row['EMA_50'] / current_row['SMA_200'])
        
        # Criterion 5: Current price > 50 EMA
        criterion5 = current_price > current_row['EMA_50']
        criterion5_score = current_price / current_row['EMA_50']
        
        # Criterion 6: Current price at least 30% above 52-week low
        criterion6 = current_price >= (current_row['Low_52W'] * 1.3)
        criterion6_score = current_price / (current_row['Low_52W'] * 1.3)
        
        # Criterion 7: Current price within 25% of 52-week high
        criterion7 = current_price >= (current_row['High_52W'] * 0.75)
        criterion7_score = current_price / (current_row['High_52W'] * 0.75)
        
        # Criterion 8: RS Rating > 70
        criterion8 = rs_rating > 70
        criterion8_score = rs_rating
        
        # Overall pass
        all_pass = all([
            criterion1, criterion2, criterion3, criterion4,
            criterion5, criterion6, criterion7, criterion8
        ])
        
        return {
            'criterion1': criterion1,
            'criterion1_score': criterion1_score,
            'criterion2': criterion2,
            'criterion2_score': criterion2_score,
            'criterion3': criterion3,
            'criterion3_score': criterion3_score,
            'criterion4': criterion4,
            'criterion4_score': criterion4_score,
            'criterion5': criterion5,
            'criterion5_score': criterion5_score,
            'criterion6': criterion6,
            'criterion6_score': criterion6_score,
            'criterion7': criterion7,
            'criterion7_score': criterion7_score,
            'criterion8': criterion8,
            'criterion8_score': criterion8_score,
            'all_pass': all_pass,
            'pass_count': sum([criterion1, criterion2, criterion3, criterion4, criterion5, criterion6, criterion7, criterion8])
        }
        
    except Exception as e:
        logger.error(f"Error evaluating Minervini criteria: {e}")
        return {}


def minervini_screener(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    Minervini Template screener for identifying strong technical setups.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with Minervini parameters
        
    Returns:
        List of dictionaries with screening results sorted by pass count
    """
    results = _minervini_screener_logic(batch_data, params)
    
    # Always save results (even if empty) for debugging and verification
    _save_minervini_results(results, params)
    
    return results


def _minervini_screener_logic(batch_data: Dict, params: Optional[Dict] = None) -> List[Dict]:
    """
    Core Minervini screener logic separated for testing and reuse.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        params: Dictionary with Minervini parameters
        
    Returns:
        List of dictionaries with screening results
    """
    if params is None:
        params = {
            'rs_min_rating': 70,
            'min_volume': 100000,
            'min_price': 5.0,
            'show_all_stocks': True  # Show all stocks for debugging
        }
    
    results = []
    
    # Load RS data
    rs_data = _load_rs_data(params)
    if rs_data is None or rs_data.empty:
        logger.error("No RS data available for Minervini screener")
        return []
    
    logger.info(f"Running Minervini screener on {len(batch_data)} tickers with RS data")
    
    for ticker, df in batch_data.items():
        try:
            # Data validation
            if df is None or df.empty:
                continue
                
            required_columns = ['Close', 'High', 'Low', 'Volume', 'Open']
            if not all(col in df.columns for col in required_columns):
                logger.debug(f"Skipping {ticker}: Missing required columns")
                continue
                
            # Minimum data length check (need 252 days for 52-week calculations)
            if len(df) < 252:
                logger.debug(f"Skipping {ticker}: Insufficient data ({len(df)} < 252)")
                continue
                
            # Current price and volume filters
            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            if (current_price < params['min_price'] or 
                current_volume < params['min_volume']):
                continue
                
            # Get RS rating for this ticker
            rs_row = rs_data[rs_data['Symbol'] == ticker]
            if rs_row.empty:
                logger.debug(f"Skipping {ticker}: No RS data available")
                continue
                
            # Use the highest RS percentile as the RS rating (15-day is typically most comprehensive)
            rs_rating = rs_row.iloc[0].get('rs_percentile_15d', 0)
            if pd.isna(rs_rating):
                # Fallback to other percentiles if 15d is not available
                for col in ['rs_percentile_10d', 'rs_percentile_5d', 'rs_percentile_3d', 'rs_percentile_1d']:
                    rs_rating = rs_row.iloc[0].get(col, 0)
                    if not pd.isna(rs_rating):
                        break
                else:
                    rs_rating = 0
            
            # Calculate Minervini indicators
            df_with_indicators = _calculate_minervini_indicators(df)
            
            # Skip if not enough data for indicators
            if pd.isna(df_with_indicators['SMA_200'].iloc[-1]):
                continue
                
            # Evaluate all criteria
            criteria_results = _evaluate_minervini_criteria(df_with_indicators, rs_rating)
            
            if not criteria_results:
                continue
                
            # Include stock if it passes all criteria OR if show_all_stocks is True
            # For debugging: always include stocks when show_all_stocks is True
            include_stock = criteria_results['all_pass'] or params.get('show_all_stocks', False)
            
            if include_stock:
                
                current_row = df_with_indicators.iloc[-1]
                
                # Calculate signal day metrics (using latest data)
                signal_open = current_row['Open']
                signal_close = current_row['Close']
                signal_day_change_abs = abs(signal_close - signal_open)
                signal_day_change_pct = ((signal_close - signal_open) / signal_open) * 100 if signal_open != 0 else 0
                
                # Performance tracking (same day = 0%)
                performance_since_signal = 0.0
                
                result = {
                    'ticker': ticker,
                    'screen_type': 'minervini',
                    'current_price': current_price,
                    'sma_150': current_row['SMA_150'],
                    'sma_200': current_row['SMA_200'],
                    'ema_50': current_row['EMA_50'],
                    'high_52w': current_row['High_52W'],
                    'low_52w': current_row['Low_52W'],
                    'rs_rating': rs_rating,
                    'volume': current_volume,
                    'criterion1': criteria_results['criterion1'],
                    'criterion1_score': criteria_results['criterion1_score'],
                    'criterion2': criteria_results['criterion2'],
                    'criterion2_score': criteria_results['criterion2_score'],
                    'criterion3': criteria_results['criterion3'],
                    'criterion3_score': criteria_results['criterion3_score'],
                    'criterion4': criteria_results['criterion4'],
                    'criterion4_score': criteria_results['criterion4_score'],
                    'criterion5': criteria_results['criterion5'],
                    'criterion5_score': criteria_results['criterion5_score'],
                    'criterion6': criteria_results['criterion6'],
                    'criterion6_score': criteria_results['criterion6_score'],
                    'criterion7': criteria_results['criterion7'],
                    'criterion7_score': criteria_results['criterion7_score'],
                    'criterion8': criteria_results['criterion8'],
                    'criterion8_score': criteria_results['criterion8_score'],
                    'all_pass': criteria_results['all_pass'],
                    'pass_count': criteria_results['pass_count'],
                    'signal_day_change_abs': signal_day_change_abs,
                    'signal_day_change_pct': signal_day_change_pct,
                    'performance_since_signal': performance_since_signal,
                    'score': criteria_results['pass_count'] * 10 + rs_rating  # Compatible with other screeners
                }
                
                # Add RS data columns
                for col in rs_row.columns:
                    if col != 'Symbol':
                        result[f'rs_{col.lower()}'] = rs_row.iloc[0][col]
                
                results.append(result)
                    
        except Exception as e:
            logger.debug(f"Error processing {ticker} in Minervini screener: {e}")
            continue
    
    # Sort by pass count (descending), then by RS rating (descending)
    results.sort(key=lambda x: (x['pass_count'], x['rs_rating']), reverse=True)
    
    logger.info(f"Minervini screener found {len(results)} stocks")
    return results


def _load_rs_data(params: Dict) -> Optional[pd.DataFrame]:
    """
    Load RS data from results directory.
    
    Args:
        params: Dictionary with parameters including ticker_choice and timeframe
        
    Returns:
        DataFrame with RS data or None if not found
    """
    try:
        ticker_choice = params.get('ticker_choice', 0)
        timeframe = params.get('timeframe', 'daily')
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Try to find RS summary file
        rs_dir = Path('results/RS')
        
        # Pattern for RS files: rs_ibd_stocks_{timeframe}_{choice}_{date}.csv (removed _summary for consistency)
        # Try current date first, then fall back to 20250820 pattern
        rs_file_pattern = f"rs_ibd_stocks_{timeframe}_{ticker_choice}_{date_str}.csv"
        rs_file = rs_dir / rs_file_pattern
        
        if rs_file.exists():
            logger.info(f"Loading RS data from: {rs_file}")
            rs_data = pd.read_csv(rs_file)
            # Rename first column to 'Symbol' if it's 'Unnamed: 0'
            if 'Unnamed: 0' in rs_data.columns:
                rs_data = rs_data.rename(columns={'Unnamed: 0': 'Symbol'})
            return rs_data
        else:
            # Try specific fallback to 20250820 file
            rs_file_fallback = rs_dir / f"rs_ibd_stocks_{timeframe}_{ticker_choice}_20250820.csv"
            if rs_file_fallback.exists():
                logger.info(f"Loading RS data from fallback file: {rs_file_fallback}")
                rs_data = pd.read_csv(rs_file_fallback)
                # Rename first column to 'Symbol' if it's unnamed
                if rs_data.columns[0] in ['Unnamed: 0', '']:
                    rs_data = rs_data.rename(columns={rs_data.columns[0]: 'Symbol'})
                return rs_data
            else:
                # Try glob pattern as last resort
                rs_file_pattern_nodate = f"rs_ibd_stocks_{timeframe}_{ticker_choice}_*.csv"
                import glob
                matching_files = glob.glob(str(rs_dir / rs_file_pattern_nodate))
                if matching_files:
                    latest_file = max(matching_files)
                    logger.info(f"Loading RS data from latest file: {latest_file}")
                    rs_data = pd.read_csv(latest_file)
                    # Rename first column to 'Symbol' if it's unnamed
                    if rs_data.columns[0] in ['Unnamed: 0', '']:
                        rs_data = rs_data.rename(columns={rs_data.columns[0]: 'Symbol'})
                    return rs_data
                else:
                    logger.warning(f"No RS data file found matching pattern: {rs_file_pattern}")
                    return None
                
    except Exception as e:
        logger.error(f"Error loading RS data: {e}")
        return None


def _save_minervini_results(results: List[Dict], params: Optional[Dict] = None) -> None:
    """
    Save Minervini results in CSV format.
    
    Args:
        results: List of Minervini screening results
        params: Minervini parameters (contains ticker_choice for filename)
    """
    try:
        # Always create output file for tracking, even if no passing stocks
        if not results:
            logger.info("No Minervini results to save - creating empty file")
            results = []  # Ensure results is a list for processing
            
        # Create DataFrame with structured columns
        output_data = []
        
        # Create output directory and filename first
        output_dir = Path('results/screeners/minervini')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ticker_choice and timeframe from params
        ticker_choice = params.get('ticker_choice', 0) if params else 0
        timeframe = params.get('timeframe', '') if params else ''
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Create filename with timeframe if available
        if timeframe:
            filename = f'MINERVINI_{ticker_choice}_{date_str}_{timeframe}.csv'
        else:
            filename = f'MINERVINI_{ticker_choice}_{date_str}.csv'
        
        output_file = output_dir / filename
        
        # If no results, create empty file with headers
        if not results:
            empty_df = pd.DataFrame(columns=[
                'Symbol', 'Price', 'SMA150', 'SMA200', 'EMA50', 'High_52W', 'Low_52W', 
                'RS_Rating', 'Volume', 'C1_Price>SMA', 'C1_Score', 'C2_SMA150>200', 'C2_Score',
                'C3_SMA200_Up', 'C3_Score', 'C4_EMA50>SMA', 'C4_Score', 'C5_Price>EMA50', 'C5_Score',
                'C6_Price>52WLow', 'C6_Score', 'C7_Price<52WHigh', 'C7_Score', 'C8_RS>70', 'C8_Score',
                'All_Pass', 'Pass_Count', 'Signal Day Open-Close $', 'Signal Day Open-Close %', 
                'Performance Since Signal %'
            ])
            empty_df.to_csv(output_file, index=False)
            print(f"📊 Minervini results saved to: {output_file}")
            print(f"📈 No stocks met Minervini criteria (empty file created)")
            logger.info(f"Empty Minervini file created at {output_file}")
            return
        
        for result in results:
            output_data.append({
                'Symbol': result['ticker'],
                'Price': round(result['current_price'], 2),
                'SMA150': round(result['sma_150'], 2),
                'SMA200': round(result['sma_200'], 2),
                'EMA50': round(result['ema_50'], 2),
                'High_52W': round(result['high_52w'], 2),
                'Low_52W': round(result['low_52w'], 2),
                'RS_Rating': round(result['rs_rating'], 1),
                'Volume': int(result['volume']),
                'C1_Price>SMA': result['criterion1'],
                'C1_Score': round(result['criterion1_score'], 2),
                'C2_SMA150>200': result['criterion2'],
                'C2_Score': round(result['criterion2_score'], 2),
                'C3_SMA200_Up': result['criterion3'],
                'C3_Score': round(result['criterion3_score'], 2),
                'C4_EMA50>SMA': result['criterion4'],
                'C4_Score': round(result['criterion4_score'], 2),
                'C5_Price>EMA50': result['criterion5'],
                'C5_Score': round(result['criterion5_score'], 2),
                'C6_Price>52WLow': result['criterion6'],
                'C6_Score': round(result['criterion6_score'], 2),
                'C7_Price<52WHigh': result['criterion7'],
                'C7_Score': round(result['criterion7_score'], 2),
                'C8_RS>70': result['criterion8'],
                'C8_Score': round(result['criterion8_score'], 1),
                'All_Pass': result['all_pass'],
                'Pass_Count': result['pass_count'],
                'Signal Day Open-Close $': round(result['signal_day_change_abs'], 2),
                'Signal Day Open-Close %': round(result['signal_day_change_pct'], 2),
                'Performance Since Signal %': round(result['performance_since_signal'], 2)
            })
        
        # Create DataFrame
        df = pd.DataFrame(output_data)
        
        # Sort by pass count (descending), then by RS rating (descending)
        df = df.sort_values(['Pass_Count', 'RS_Rating'], ascending=[False, False])
        
        # Use the output file already defined above
        # Save with proper formatting
        df.to_csv(output_file, index=False, float_format='%.2f')
        
        print(f"📊 Minervini results saved to: {output_file}")
        print(f"📈 Total stocks analyzed: {len(results)}")
        
        # Show results breakdown
        passing_stocks = len(df[df['All_Pass'] == True])
        print(f"   • Stocks passing all 8 criteria: {passing_stocks}")
        print(f"   • Stocks meeting 6-7 criteria: {len(df[(df['Pass_Count'] >= 6) & (df['Pass_Count'] < 8)])}")
        print(f"   • Average RS Rating: {df['RS_Rating'].mean():.1f}")
        
        logger.info(f"Minervini results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving Minervini results: {e}")
        print(f"❌ Error saving Minervini results: {e}")