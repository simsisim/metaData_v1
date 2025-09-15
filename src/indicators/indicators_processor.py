"""
Technical Indicators Processor Module
====================================

Main processing module that integrates technical indicators calculation and charting
into the main data processing pipeline.

Handles:
- Batch processing of indicators for multiple tickers
- Configuration-based indicator selection
- Chart generation based on CSV configuration
- Integration with existing workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .indicators_calculation import calculate_all_indicators, _get_default_config
from .indicators_charts import create_charts_from_config_file, analyze_indicator_signals

logger = logging.getLogger(__name__)


def process_indicators_batch(batch_data: Dict[str, pd.DataFrame], 
                           output_dir: Path,
                           timeframe: str,
                           user_config) -> int:
    """
    Process technical indicators for a batch of tickers.
    
    Args:
        batch_data: Dict of {ticker: DataFrame} with OHLCV data
        output_dir: Output directory for results
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: User configuration object
        
    Returns:
        int: Number of tickers processed
    """
    if not user_config.indicators_enable:
        logger.info("Indicators processing disabled")
        return 0
    
    logger.info(f"Processing indicators for {len(batch_data)} tickers ({timeframe})")
    
    # Create output directories
    indicators_dir = Path(user_config.indicators_output_dir) / timeframe
    charts_dir = Path(user_config.indicators_charts_dir) / timeframe
    indicators_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Build configuration from user settings
    config = _build_indicators_config(user_config)
    
    processed_count = 0
    all_indicators_data = {}
    
    # Calculate indicators for each ticker
    for ticker, data in batch_data.items():
        try:
            # Calculate all indicators
            indicators_data = calculate_all_indicators(data, config)
            all_indicators_data[ticker] = indicators_data
            
            # Save individual ticker indicators
            output_file = indicators_dir / f"{ticker}_indicators_{timeframe}.csv"
            indicators_data.to_csv(output_file)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing indicators for {ticker}: {e}")
            continue
    
    # Save batch summary
    _save_indicators_summary(all_indicators_data, indicators_dir, timeframe)
    
    # Generate charts if configuration file exists
    _generate_charts_if_configured(all_indicators_data, charts_dir, timeframe, user_config)
    
    logger.info(f"Indicators processing completed: {processed_count} tickers")
    return processed_count


def _build_indicators_config(user_config) -> Dict:
    """
    Build indicators configuration from user settings.
    
    Args:
        user_config: User configuration object
        
    Returns:
        Dict with indicator configuration
    """
    return {
        'kurutoga': {
            'enabled': user_config.indicators_kurutoga_enable,
            'length': user_config.indicators_kurutoga_length,
            'source': user_config.indicators_kurutoga_source
        },
        'tsi': {
            'enabled': user_config.indicators_tsi_enable,
            'fast': user_config.indicators_tsi_fast,
            'slow': user_config.indicators_tsi_slow,
            'signal': user_config.indicators_tsi_signal
        },
        'macd': {
            'enabled': user_config.indicators_macd_enable,
            'fast': user_config.indicators_macd_fast,
            'slow': user_config.indicators_macd_slow,
            'signal': user_config.indicators_macd_signal
        },
        'mfi': {
            'enabled': user_config.indicators_mfi_enable,
            'length': user_config.indicators_mfi_length,
            'include_signal': user_config.indicators_mfi_signal_enable,
            'signal_period': user_config.indicators_mfi_signal_period
        },
        'cog': {
            'enabled': user_config.indicators_cog_enable,
            'length': user_config.indicators_cog_length,
            'source': user_config.indicators_cog_source
        },
        'momentum': {
            'enabled': user_config.indicators_momentum_enable,
            'length': user_config.indicators_momentum_length
        },
        'rsi': {
            'enabled': user_config.indicators_rsi_enable,
            'length': user_config.indicators_rsi_length
        },
        'ma_crosses': {
            'enabled': user_config.indicators_ma_crosses_enable,
            'fast_period': user_config.indicators_ma_fast_period,
            'slow_period': user_config.indicators_ma_slow_period
        },
        'easy_trade': {
            'enabled': user_config.indicators_easy_trade_enable,
            'fast_length': user_config.indicators_easy_trade_fast,
            'slow_length': user_config.indicators_easy_trade_slow,
            'signal_length': user_config.indicators_easy_trade_signal
        }
    }


def _save_indicators_summary(indicators_data: Dict[str, pd.DataFrame], 
                           output_dir: Path, timeframe: str):
    """Save summary of indicators across all tickers."""
    try:
        # Analyze signals across all tickers
        signals_summary = analyze_indicator_signals(indicators_data)
        
        if not signals_summary.empty:
            summary_file = output_dir / f"indicators_summary_{timeframe}.csv"
            signals_summary.to_csv(summary_file, index=False)
            logger.info(f"Indicators summary saved: {summary_file}")
            
            # Create signal counts summary
            signal_counts = {}
            for col in signals_summary.columns:
                if col.startswith(('all_', 'trend_', 'golden_', 'death_')):
                    signal_counts[col] = signals_summary[col].sum()
            
            if signal_counts:
                counts_file = output_dir / f"signal_counts_{timeframe}.csv"
                pd.DataFrame([signal_counts]).to_csv(counts_file, index=False)
                logger.info(f"Signal counts saved: {counts_file}")
                
    except Exception as e:
        logger.error(f"Error saving indicators summary: {e}")


def _generate_charts_if_configured(indicators_data: Dict[str, pd.DataFrame],
                                 charts_dir: Path, timeframe: str, user_config):
    """Generate charts if configuration file exists."""
    try:
        config_file = Path(user_config.indicators_config_file)
        
        if config_file.exists():
            logger.info(f"Generating charts from config: {config_file}")
            
            created_charts = create_charts_from_config_file(
                indicators_data, config_file, charts_dir, timeframe
            )
            
            logger.info(f"Charts generated: {len(created_charts)} files")
            
            # Save chart file list
            if created_charts:
                chart_list_file = charts_dir / f"chart_files_{timeframe}.txt"
                with open(chart_list_file, 'w') as f:
                    for chart_path in created_charts:
                        f.write(f"{chart_path}\n")
                        
        else:
            logger.info(f"No chart configuration file found: {config_file}")
            
    except Exception as e:
        logger.error(f"Error generating charts: {e}")


def get_indicators_config_for_timeframe(user_config, timeframe: str) -> Dict:
    """
    Get indicators configuration for a specific timeframe.
    
    Args:
        user_config: User configuration object
        timeframe: Target timeframe
        
    Returns:
        Dict with indicators configuration parameters
    """
    if not user_config.indicators_enable:
        return {'enable_indicators': False}
    
    return {
        'enable_indicators': True,
        'config': _build_indicators_config(user_config),
        'output_dir': user_config.indicators_output_dir,
        'charts_dir': user_config.indicators_charts_dir,
        'config_file': user_config.indicators_config_file
    }