"""
Analytical Models Module
=======================

Implements various analytical models for price prediction, risk assessment,
and portfolio optimization using market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def simple_trend_model(df, ticker):
    """
    Simple trend prediction model based on moving averages and momentum.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with model predictions and confidence metrics
    """
    if df is None or df.empty or 'Close' not in df.columns or len(df) < 50:
        return {}
        
    close = df['Close']
    result = {
        'ticker': ticker,
        'model_type': 'simple_trend',
        'current_price': close.iloc[-1],
        'data_points': len(df)
    }
    
    try:
        # Calculate multiple moving averages
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean() 
        sma_50 = close.rolling(50).mean()
        
        # Current trend signals
        result['above_sma_10'] = close.iloc[-1] > sma_10.iloc[-1]
        result['above_sma_20'] = close.iloc[-1] > sma_20.iloc[-1]
        result['above_sma_50'] = close.iloc[-1] > sma_50.iloc[-1]
        
        # Moving average slopes (trend strength)
        result['sma_10_slope'] = (sma_10.iloc[-1] - sma_10.iloc[-6]) / 5 if len(df) >= 15 else 0
        result['sma_20_slope'] = (sma_20.iloc[-1] - sma_20.iloc[-11]) / 10 if len(df) >= 30 else 0
        result['sma_50_slope'] = (sma_50.iloc[-1] - sma_50.iloc[-21]) / 20 if len(df) >= 70 else 0
        
        # Simple trend score (-100 to +100)
        trend_signals = 0
        if result['above_sma_10']:
            trend_signals += 1
        if result['above_sma_20']:
            trend_signals += 1
        if result['above_sma_50']:
            trend_signals += 1
            
        slope_score = (result['sma_10_slope'] + result['sma_20_slope'] + result['sma_50_slope']) * 1000
        result['trend_score'] = (trend_signals / 3) * 50 + np.clip(slope_score, -50, 50)
        
        # Simple price prediction (next 5-10 days)
        recent_momentum = (close.iloc[-1] / close.iloc[-6]) - 1 if len(df) >= 6 else 0
        trend_momentum = result['sma_20_slope'] / close.iloc[-1] if close.iloc[-1] != 0 else 0
        
        predicted_return = (recent_momentum * 0.3 + trend_momentum * 0.7) * 5  # 5-day projection
        result['predicted_price_5d'] = close.iloc[-1] * (1 + predicted_return)
        result['predicted_return_5d'] = predicted_return
        
        # Confidence based on trend consistency
        if len(df) >= 20:
            trend_consistency = sum(1 for i in range(5, 20) if 
                                  (close.iloc[-i] > sma_20.iloc[-i]) == result['above_sma_20']) / 15
            result['prediction_confidence'] = trend_consistency
        else:
            result['prediction_confidence'] = 0.5
            
    except Exception as e:
        logger.debug(f"Error in simple_trend_model for {ticker}: {e}")
        
    return result


def mean_reversion_model(df, ticker):
    """
    Mean reversion model identifying oversold/overbought conditions.
    """
    if df is None or df.empty or 'Close' not in df.columns or len(df) < 30:
        return {}
        
    close = df['Close']
    result = {
        'ticker': ticker,
        'model_type': 'mean_reversion',
        'current_price': close.iloc[-1]
    }
    
    try:
        # Calculate Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        current_price = close.iloc[-1]
        result['sma_20'] = sma_20.iloc[-1]
        result['upper_band'] = upper_band.iloc[-1]
        result['lower_band'] = lower_band.iloc[-1]
        
        # Position within bands
        band_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        result['band_position'] = band_position  # 0 = at lower band, 1 = at upper band
        
        # RSI calculation
        if len(df) >= 15:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) and rs.iloc[-1] > 0 else 50
            result['rsi'] = rsi
            
            # Mean reversion signals
            result['oversold_signal'] = rsi < 30 and band_position < 0.2  # Strong oversold
            result['overbought_signal'] = rsi > 70 and band_position > 0.8  # Strong overbought
            
            # Mean reversion score
            if result['oversold_signal']:
                result['reversion_score'] = 100 - rsi  # Higher score for more oversold
                result['predicted_direction'] = 'up'
            elif result['overbought_signal']:
                result['reversion_score'] = rsi - 100  # More negative for more overbought
                result['predicted_direction'] = 'down'
            else:
                result['reversion_score'] = 0
                result['predicted_direction'] = 'neutral'
                
        # Estimate potential reversion target
        result['reversion_target'] = sma_20.iloc[-1]  # Mean reversion to SMA20
        result['potential_return'] = (result['reversion_target'] / current_price) - 1
        
    except Exception as e:
        logger.debug(f"Error in mean_reversion_model for {ticker}: {e}")
        
    return result


def volatility_model(df, ticker):
    """
    Volatility prediction model for risk assessment.
    """
    if df is None or df.empty or 'Close' not in df.columns or len(df) < 30:
        return {}
        
    close = df['Close']
    result = {
        'ticker': ticker,
        'model_type': 'volatility',
        'current_price': close.iloc[-1]
    }
    
    try:
        # Calculate returns
        returns = close.pct_change().dropna()
        
        if len(returns) < 20:
            return result
            
        # Historical volatility metrics
        result['volatility_daily'] = returns.std()
        result['volatility_annualized'] = returns.std() * np.sqrt(252)
        
        # Rolling volatility analysis
        vol_20 = returns.rolling(20).std()
        vol_50 = returns.rolling(50).std() if len(returns) >= 50 else vol_20
        
        result['current_vol_20d'] = vol_20.iloc[-1] * np.sqrt(252)
        result['avg_vol_50d'] = vol_50.mean() * np.sqrt(252) if len(vol_50.dropna()) > 0 else result['volatility_annualized']
        
        # Volatility regime detection
        current_vol = vol_20.iloc[-1]
        historical_vol_median = vol_20.median()
        
        vol_percentile = (vol_20 <= current_vol).mean() * 100
        result['volatility_percentile'] = vol_percentile
        
        if vol_percentile > 80:
            result['volatility_regime'] = 'high'
        elif vol_percentile < 20:
            result['volatility_regime'] = 'low'
        else:
            result['volatility_regime'] = 'normal'
            
        # Predict volatility clustering (GARCH-like concept)
        recent_vol = returns.tail(5).std()
        result['vol_clustering_signal'] = recent_vol / historical_vol_median if historical_vol_median > 0 else 1.0
        
        # Risk metrics
        if len(returns) >= 252:
            # VaR estimation (95% confidence)
            result['var_95_1d'] = np.percentile(returns, 5)  # Daily VaR
            result['var_95_1w'] = result['var_95_1d'] * np.sqrt(5)  # Weekly VaR
            
        # Downside volatility
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 5:
            result['downside_volatility'] = downside_returns.std() * np.sqrt(252)
        else:
            result['downside_volatility'] = result['volatility_annualized']
            
    except Exception as e:
        logger.debug(f"Error in volatility_model for {ticker}: {e}")
        
    return result


def portfolio_correlation_model(batch_data):
    """
    Analyze correlations between assets for portfolio construction.
    """
    if not batch_data or len(batch_data) < 2:
        return {}
        
    # Extract returns for correlation analysis
    returns_data = {}
    
    for ticker, df in batch_data.items():
        if df is not None and not df.empty and 'Close' in df.columns and len(df) > 20:
            returns = df['Close'].pct_change().dropna()
            if len(returns) >= 20:  # Minimum data requirement
                returns_data[ticker] = returns
                
    if len(returns_data) < 2:
        return {}
        
    try:
        # Align returns data by date
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()  # Remove rows with any NaN
        
        if len(returns_df) < 10:  # Not enough overlapping data
            return {}
            
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        result = {
            'model_type': 'portfolio_correlation',
            'analysis_date': returns_df.index[-1].strftime('%Y-%m-%d') if hasattr(returns_df.index[-1], 'strftime') else str(returns_df.index[-1]),
            'assets_analyzed': len(returns_data),
            'overlapping_periods': len(returns_df)
        }
        
        # Find highest and lowest correlations
        corr_values = []
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                ticker1 = correlation_matrix.index[i]
                ticker2 = correlation_matrix.index[j]
                corr = correlation_matrix.iloc[i, j]
                if not np.isnan(corr):
                    corr_values.append((ticker1, ticker2, corr))
                    
        if corr_values:
            corr_values.sort(key=lambda x: x[2], reverse=True)
            
            result['highest_correlation'] = corr_values[0][2]
            result['highest_corr_pair'] = f"{corr_values[0][0]}-{corr_values[0][1]}"
            result['lowest_correlation'] = corr_values[-1][2] 
            result['lowest_corr_pair'] = f"{corr_values[-1][0]}-{corr_values[-1][1]}"
            result['avg_correlation'] = np.mean([c[2] for c in corr_values])
            
        # Portfolio diversification potential
        avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        result['diversification_ratio'] = 1 - abs(avg_corr)  # Higher = better diversification
        
        return result
        
    except Exception as e:
        logger.error(f"Error in portfolio correlation model: {e}")
        return {}


def run_models(batch_data, output_path, timeframe):
    """
    Run analytical models on batch data.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        output_path: Path to save model results
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
    """
    logger.info(f"Running analytical models on {len(batch_data)} tickers ({timeframe})")
    
    all_results = []
    
    # Run individual ticker models
    for ticker, df in batch_data.items():
        # Trend prediction model
        trend_result = simple_trend_model(df, ticker)
        if trend_result:
            all_results.append(trend_result)
            
        # Mean reversion model
        reversion_result = mean_reversion_model(df, ticker)
        if reversion_result:
            all_results.append(reversion_result)
            
        # Volatility model
        volatility_result = volatility_model(df, ticker)
        if volatility_result:
            all_results.append(volatility_result)
    
    # Portfolio-level analysis
    portfolio_result = portfolio_correlation_model(batch_data)
    if portfolio_result:
        all_results.append(portfolio_result)
    
    # Extract data date from batch_data instead of using file generation timestamp
    data_date = None
    for ticker, df in batch_data.items():
        if df is not None and not df.empty and hasattr(df, 'index') and len(df.index) > 0:
            latest_date = df.index[-1]
            if hasattr(latest_date, 'strftime'):
                data_date = latest_date.strftime('%Y%m%d')
            else:
                # Handle string dates
                data_date = str(latest_date).replace('-', '')[:8]
            break
    
    # Fallback to file generation timestamp if no data date found
    if not data_date:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.warning(f"Using file generation timestamp as fallback for {timeframe} models: {timestamp}")
    else:
        timestamp = f"{data_date}_{datetime.now().strftime('%H%M%S')}"
        logger.info(f"Using data date for {timeframe} models filename: {data_date}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = output_path / f'model_results_{timeframe}_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        # Create summary by model type
        model_summary = {}
        for model_type in results_df['model_type'].unique():
            model_results = results_df[results_df['model_type'] == model_type]
            model_summary[f'{model_type}_count'] = len(model_results)
            
        # Add portfolio analysis summary
        if portfolio_result:
            model_summary.update({
                'portfolio_assets': portfolio_result.get('assets_analyzed', 0),
                'avg_correlation': portfolio_result.get('avg_correlation', 0),
                'diversification_ratio': portfolio_result.get('diversification_ratio', 0)
            })
            
        # Save summary
        summary_data = {
            'analysis_date': data_date[:4] + '-' + data_date[4:6] + '-' + data_date[6:8] if data_date and len(data_date) >= 8 else datetime.now().strftime('%Y-%m-%d'),
            'timeframe': timeframe,
            'total_models_run': len(all_results),
            **model_summary
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_file = output_path / f'model_summary_{timeframe}_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        
        # Print results summary
        print(f"🤖 Model Results ({timeframe}):")
        print(f"  • Total model runs: {len(all_results)}")
        
        for model_type in ['simple_trend', 'mean_reversion', 'volatility']:
            count = model_summary.get(f'{model_type}_count', 0)
            print(f"  • {model_type.replace('_', ' ').title()}: {count} results")
            
        if portfolio_result:
            print(f"  • Portfolio analysis: {portfolio_result.get('assets_analyzed', 0)} assets")
            print(f"  • Avg correlation: {portfolio_result.get('avg_correlation', 0):.3f}")
            
        print(f"  • Results saved: {results_file.name}")
        
        # Highlight interesting findings
        trend_models = results_df[results_df['model_type'] == 'simple_trend']
        if not trend_models.empty and 'trend_score' in trend_models.columns:
            strong_trends = trend_models[abs(trend_models['trend_score']) > 60]
            if not strong_trends.empty:
                print(f"  • Strong trends detected: {len(strong_trends)} tickers")
                
        reversion_models = results_df[results_df['model_type'] == 'mean_reversion']
        if not reversion_models.empty:
            signals = reversion_models[
                (reversion_models.get('oversold_signal', False)) | 
                (reversion_models.get('overbought_signal', False))
            ]
            if not signals.empty:
                print(f"  • Mean reversion signals: {len(signals)} tickers")
    
    else:
        print(f"⚠️  No model results generated for {timeframe} batch")
    
    logger.info(f"Model analysis completed: {len(all_results)} total model results")
    
    return len(all_results)


# Market Breadth Configuration
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class MarketBreadthConfig:
    """
    Configuration class for market breadth calculations.
    """
    # Moving average periods for breadth analysis
    ma_periods: List[int] = None
    
    # New highs/lows lookback period (trading days)
    highs_lows_lookback_days: int = 252  # ~1 year
    
    # Threshold values for specialized indicators
    daily_new_highs_threshold: int = 100
    ten_day_success_threshold: int = 5  # Out of 10 days
    
    # Advance/decline ratio thresholds
    strong_ad_ratio_threshold: float = 2.0
    weak_ad_ratio_threshold: float = 0.5
    
    # Breadth percentage thresholds
    strong_advance_threshold: float = 70.0  # % of stocks advancing
    weak_advance_threshold: float = 30.0
    strong_ma_breadth_threshold: float = 80.0  # % above MA
    weak_ma_breadth_threshold: float = 20.0
    
    # Universe configuration
    default_universe: str = "all"
    include_sectors: bool = True
    include_industries: bool = True
    include_market_caps: bool = True
    
    # Output configuration
    save_detailed_results: bool = True
    save_summary_results: bool = True
    output_format: str = "csv"  # csv, parquet, json
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.ma_periods is None:
            self.ma_periods = [20, 50, 200]
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'ma_periods': self.ma_periods,
            'highs_lows_lookback_days': self.highs_lows_lookback_days,
            'daily_new_highs_threshold': self.daily_new_highs_threshold,
            'ten_day_success_threshold': self.ten_day_success_threshold,
            'strong_ad_ratio_threshold': self.strong_ad_ratio_threshold,
            'weak_ad_ratio_threshold': self.weak_ad_ratio_threshold,
            'strong_advance_threshold': self.strong_advance_threshold,
            'weak_advance_threshold': self.weak_advance_threshold,
            'strong_ma_breadth_threshold': self.strong_ma_breadth_threshold,
            'weak_ma_breadth_threshold': self.weak_ma_breadth_threshold,
            'default_universe': self.default_universe,
            'include_sectors': self.include_sectors,
            'include_industries': self.include_industries,
            'include_market_caps': self.include_market_caps,
            'save_detailed_results': self.save_detailed_results,
            'save_summary_results': self.save_summary_results,
            'output_format': self.output_format
        }