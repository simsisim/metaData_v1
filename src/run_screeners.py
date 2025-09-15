"""
Stock Screeners Orchestrator
============================

Main orchestrator for running multiple stock screening strategies.
Coordinates individual screener modules and aggregates results.

This module serves as the central hub for all screening operations,
maintaining the same interface as the original monolithic screener.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Import individual screener modules
from .screeners import pvb_screener, atr1_screener, atr2_screener, giusti_screener, minervini_screener, drwish_screener, run_volume_suite_screener, run_stockbee_suite_screener, run_qullamaggie_suite_screener, run_adl_screener, run_guppy_screener, run_gold_launch_pad_screener, run_rti_screener
from .screeners.basic_screeners_claude import run_basic_screeners
from .user_defined_data import get_pvb_params_for_timeframe, get_atr1_params_for_timeframe, get_atr2_params_for_timeframe, get_giusti_params_for_timeframe, get_minervini_params_for_timeframe, get_drwish_params_for_timeframe, get_volume_suite_params_for_timeframe, get_stockbee_suite_params_for_timeframe, get_qullamaggie_suite_params_for_timeframe, get_adl_screener_params_for_timeframe, get_guppy_screener_params_for_timeframe, get_gold_launch_pad_params_for_timeframe, get_rti_screener_params_for_timeframe

logger = logging.getLogger(__name__)


def has_any_screeners_enabled(user_config):
    """
    Check if any screeners are enabled in the user configuration.
    
    Args:
        user_config: User configuration object
        
    Returns:
        bool: True if any screeners are enabled
    """
    if not user_config:
        return False
        
    screener_flags = [
        'basic_momentum_enable',
        'basic_breakout_enable', 
        'basic_value_momentum_enable',
        'pvb_enable',
        'atr1_enable',
        'atr2_enable',
        'giusti_enable',
        'minervini_enable',
        'drwish_enable',
        'volume_suite_enable',
        'stockbee_suite_enable',
        'qullamaggie_suite_enable',
        'adl_screener_enable',
        'guppy_screener_enable',
        'gold_launch_pad_enable',
        'rti_enable',
    ]
    
    return any(getattr(user_config, flag, False) for flag in screener_flags)


def run_screeners(batch_data, output_path, timeframe, user_config=None, data_reader=None):
    """
    Run multiple screening strategies on batch data.
    
    Args:
        batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
        output_path: Path to save screening results
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        user_config: Optional user configuration for screener parameters
        data_reader: Optional DataReader instance with ticker info loaded
        
    Returns:
        int: Total number of screening hits across all screeners
    """
    logger.info(f"Running screeners on {len(batch_data)} tickers ({timeframe})")
    
    # Load ticker info if data_reader is provided
    ticker_info = None
    if data_reader is not None and hasattr(data_reader, 'ticker_info'):
        ticker_info = data_reader.ticker_info
    
    all_results = []
    screener_summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'timeframe': timeframe,
        'total_tickers': len(batch_data)
    }
    
    # Run basic screeners (now configurable)
    basic_screener_results = run_basic_screeners(batch_data, user_config)
    momentum_results = basic_screener_results['momentum']
    breakout_results = basic_screener_results['breakout']
    value_momentum_results = basic_screener_results['value_momentum']
    
    all_results.extend(momentum_results)
    all_results.extend(breakout_results)
    all_results.extend(value_momentum_results)
    
    screener_summary['momentum_hits'] = len(momentum_results)
    screener_summary['breakout_hits'] = len(breakout_results)
    screener_summary['value_momentum_hits'] = len(value_momentum_results)
 
    # Run Giusti screener (momentum analysis)
    giusti_results = []
    if user_config and hasattr(user_config, 'giusti_enable') and user_config.giusti_enable:
        print("🔍 Running Giusti momentum screener...")
        giusti_params = get_giusti_params_for_timeframe(user_config, timeframe)
        giusti_results = giusti_screener(batch_data, giusti_params)
        all_results.extend(giusti_results)
    screener_summary['giusti_hits'] = len(giusti_results)
    
     # Run Minervini screener (requires RS data - runs after RS calculations)
    minervini_results = []
    if user_config and hasattr(user_config, 'minervini_enable') and user_config.minervini_enable:
        print("🔍 Running Minervini Template screener...")
        minervini_params = get_minervini_params_for_timeframe(user_config, timeframe)
        minervini_results = minervini_screener(batch_data, minervini_params)
        all_results.extend(minervini_results)
    screener_summary['minervini_hits'] = len(minervini_results)
    
    # Run Dr. Wish suite screener (GLB, Blue Dot, Black Dot)
    drwish_results = []
    if user_config and hasattr(user_config, 'drwish_enable') and user_config.drwish_enable:
        print("🔍 Running Dr. Wish suite screener (GLB, Blue Dot, Black Dot)...")
        drwish_params = get_drwish_params_for_timeframe(user_config, timeframe)
        drwish_results = drwish_screener(batch_data, drwish_params)
        all_results.extend(drwish_results)
    screener_summary['drwish_hits'] = len(drwish_results)
    
    # Run PVB screener (if enabled)
    pvb_results = []
    if user_config and hasattr(user_config, 'pvb_enable') and user_config.pvb_enable:
        print("🔍 Running PVB (Price Volume Breakout) screener...")
        pvb_params = get_pvb_params_for_timeframe(user_config, timeframe)
        pvb_results = pvb_screener(batch_data, pvb_params)
        all_results.extend(pvb_results)
    screener_summary['pvb_hits'] = len(pvb_results)
    
    # Run ATR1 screener (TradingView-validated)
    atr1_results = []
    if user_config and hasattr(user_config, 'atr1_enable') and user_config.atr1_enable:
        print("🔍 Running ATR1 (TradingView-validated) screener...")
        atr1_params = get_atr1_params_for_timeframe(user_config, timeframe)
        atr1_results = atr1_screener(batch_data, atr1_params)
        all_results.extend(atr1_results)
    screener_summary['atr1_hits'] = len(atr1_results)
    
    # Run ATR2 screener (Volatility analysis)
    atr2_results = []
    if user_config and hasattr(user_config, 'atr2_enable') and user_config.atr2_enable:
        print("🔍 Running ATR2 (Volatility analysis) screener...")
        atr2_params = get_atr2_params_for_timeframe(user_config, timeframe)
        atr2_results = atr2_screener(batch_data, atr2_params)
        all_results.extend(atr2_results)
    screener_summary['atr2_hits'] = len(atr2_results)
    
    # Run Volume Suite screener (comprehensive volume analysis)
    volume_suite_results = []
    if user_config and hasattr(user_config, 'volume_suite_enable') and user_config.volume_suite_enable:
        print("🔍 Running Volume Suite screener (HV Absolute, HV StdDev, VROC, RVOL, ADTV, MFI, VPT)...")
        volume_suite_config = get_volume_suite_params_for_timeframe(user_config, timeframe)
        volume_suite_results = run_volume_suite_screener(batch_data, volume_suite_config)
        all_results.extend(volume_suite_results)
    screener_summary['volume_suite_hits'] = len(volume_suite_results)
    
    # Run Stockbee Suite screener (9M Movers, Weekly Movers, Daily Gainers, Industry Leaders)
    stockbee_suite_results = []
    if user_config and hasattr(user_config, 'stockbee_suite_enable') and user_config.stockbee_suite_enable:
        print("🔍 Running Stockbee Suite screener (9M Movers, 20% Weekly, 4% Daily, Industry Leaders)...")
        stockbee_suite_config = get_stockbee_suite_params_for_timeframe(user_config, timeframe)
        
        # Use ticker info loaded at function start
        rs_data = None  # TODO: Load RS data from previous pipeline step
        
        stockbee_suite_results = run_stockbee_suite_screener(batch_data, stockbee_suite_config, ticker_info, rs_data)
        all_results.extend(stockbee_suite_results)
    screener_summary['stockbee_suite_hits'] = len(stockbee_suite_results)
    
    # Run Qullamaggie Suite screener (RS ≥ 97, MA alignment, ATR RS ≥ 50, range position)
    qullamaggie_suite_results = []
    if user_config and hasattr(user_config, 'qullamaggie_suite_enable') and user_config.qullamaggie_suite_enable:
        print("🔍 Running Qullamaggie Suite screener (RS≥97, MA alignment, ATR RS≥50, range position)...")
        qullamaggie_suite_config = get_qullamaggie_suite_params_for_timeframe(user_config, timeframe)
        
        # Use ticker info loaded at function start
        rs_data = None  # TODO: Load RS data from previous pipeline step
        atr_universe_data = batch_data  # Use batch data for ATR universe calculations
        
        qullamaggie_suite_results = run_qullamaggie_suite_screener(
            batch_data, qullamaggie_suite_config, ticker_info, rs_data, atr_universe_data
        )
        all_results.extend(qullamaggie_suite_results)
    screener_summary['qullamaggie_suite_hits'] = len(qullamaggie_suite_results)
    
    # Run ADL screener (Accumulation/Distribution Line divergence and breakout analysis)
    adl_screener_results = []
    if user_config and hasattr(user_config, 'adl_screener_enable') and user_config.adl_screener_enable:
        print("🔍 Running ADL screener (Accumulation/Distribution divergence and breakout analysis)...")
        adl_screener_config = get_adl_screener_params_for_timeframe(user_config, timeframe)
        
        # Use ticker info and RS data loaded at function start
        rs_data = None  # TODO: Load RS data from previous pipeline step
        
        adl_screener_results = run_adl_screener(batch_data, adl_screener_config, ticker_info, rs_data)
        all_results.extend(adl_screener_results)
    screener_summary['adl_screener_hits'] = len(adl_screener_results)
    
    # Run Guppy GMMA screener (trend alignment, compression/expansion, crossovers)
    guppy_screener_results = []
    if user_config and hasattr(user_config, 'guppy_screener_enable') and user_config.guppy_screener_enable:
        print("🔍 Running Guppy GMMA screener (trend alignment, compression/expansion, crossovers)...")
        guppy_screener_config = get_guppy_screener_params_for_timeframe(user_config, timeframe)
        
        # Use ticker info and RS data loaded at function start
        rs_data = None  # TODO: Load RS data from previous pipeline step
        
        guppy_screener_results = run_guppy_screener(batch_data, guppy_screener_config, ticker_info, rs_data)
        all_results.extend(guppy_screener_results)
    screener_summary['guppy_screener_hits'] = len(guppy_screener_results)
    
    # 11. Gold Launch Pad Screener (MA alignment and momentum detection)
    gold_launch_pad_results = []
    if user_config and hasattr(user_config, 'gold_launch_pad_enable') and user_config.gold_launch_pad_enable:
        print("🔍 Running Gold Launch Pad screener (MA alignment, momentum detection)...")
        
        gold_launch_pad_config = get_gold_launch_pad_params_for_timeframe(user_config, timeframe)
        gold_launch_pad_results = run_gold_launch_pad_screener(batch_data, gold_launch_pad_config, ticker_info)
        all_results.extend(gold_launch_pad_results)
    screener_summary['gold_launch_pad_hits'] = len(gold_launch_pad_results)
    
    # 12. RTI Screener (Range Tightening Indicator - volatility compression/expansion)
    rti_results = []
    if user_config and hasattr(user_config, 'rti_enable') and user_config.rti_enable:
        print("🔍 Running RTI screener (volatility compression/expansion detection)...")
        
        rti_config = get_rti_screener_params_for_timeframe(user_config, timeframe)
        rti_results = run_rti_screener(batch_data, rti_config, ticker_info)
        all_results.extend(rti_results)
    screener_summary['rti_hits'] = len(rti_results)
    
    # Market Pulse Analysis moved to main.py pipeline level
    
    
    # Extract data date from batch_data instead of using file generation timestamp
    data_date = None
    if batch_data:
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
        logger.warning(f"Using file generation timestamp as fallback for {timeframe} screener results: {timestamp}")
    else:
        timestamp = f"{data_date}_{datetime.now().strftime('%H%M%S')}"
        logger.info(f"Using data date for {timeframe} screener results filename: {data_date}")
    
    if all_results:
        # Individual screening results
        results_df = pd.DataFrame(all_results)
        results_file = output_path / f'screener_results_{timeframe}_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        # Create top picks summary (best from each screener)
        top_picks = []
        for screen_type in ['momentum', 'breakout', 'value_momentum', 'pvb', 'atr1', 'atr2', 'giusti', 'minervini', 'drwish_glb', 'drwish_blue_dot', 'drwish_black_dot', 'volume_suite_hv_absolute', 'volume_suite_hv_stdv', 'volume_anomaly', 'volume_indicators', 'stockbee_9m_movers', 'stockbee_weekly_movers', 'stockbee_daily_gainers', 'stockbee_industry_leaders', 'qullamaggie_suite', 'adl_divergence', 'adl_breakout', 'adl_breakdown', 'guppy_alignment', 'guppy_compression', 'guppy_expansion', 'guppy_crossover', 'gold_launch_pad', 'rti_screener']:
            screen_results = [r for r in all_results if r['screen_type'] == screen_type]
            if screen_results:
                top_picks.extend(screen_results[:5])  # Top 5 from each screener
                
        if top_picks:
            top_picks_df = pd.DataFrame(top_picks)
            top_picks_file = output_path / f'top_picks_{timeframe}_{timestamp}.csv'
            top_picks_df.to_csv(top_picks_file, index=False)
    
    # Save screener summary only if there are enabled screeners
    if has_any_screeners_enabled(user_config):
        summary_df = pd.DataFrame([screener_summary])
        summary_file = output_path / f'screener_summary_{timeframe}_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
    
    # Print results - only show enabled screeners with results
    enabled_screeners = []
    
    # Add enabled basic screeners to output
    if user_config and hasattr(user_config, 'basic_momentum_enable') and user_config.basic_momentum_enable:
        enabled_screeners.append(f"Momentum hits: {screener_summary['momentum_hits']}")
    if user_config and hasattr(user_config, 'basic_breakout_enable') and user_config.basic_breakout_enable:
        enabled_screeners.append(f"Breakout hits: {screener_summary['breakout_hits']}")
    if user_config and hasattr(user_config, 'basic_value_momentum_enable') and user_config.basic_value_momentum_enable:
        enabled_screeners.append(f"Value-Momentum hits: {screener_summary['value_momentum_hits']}")
    
    # Only show advanced screeners if enabled
    if user_config and hasattr(user_config, 'pvb_enable') and user_config.pvb_enable:
        enabled_screeners.append(f"PVB hits: {screener_summary['pvb_hits']}")
    if user_config and hasattr(user_config, 'atr1_enable') and user_config.atr1_enable:
        enabled_screeners.append(f"ATR1 hits: {screener_summary['atr1_hits']}")
    if user_config and hasattr(user_config, 'atr2_enable') and user_config.atr2_enable:
        enabled_screeners.append(f"ATR2 hits: {screener_summary['atr2_hits']}")
    if user_config and hasattr(user_config, 'giusti_enable') and user_config.giusti_enable:
        enabled_screeners.append(f"Giusti hits: {screener_summary['giusti_hits']}")
    if user_config and hasattr(user_config, 'minervini_enable') and user_config.minervini_enable:
        enabled_screeners.append(f"Minervini hits: {screener_summary['minervini_hits']}")
    if user_config and hasattr(user_config, 'drwish_enable') and user_config.drwish_enable:
        enabled_screeners.append(f"Dr. Wish hits: {screener_summary['drwish_hits']}")
    if user_config and hasattr(user_config, 'volume_suite_enable') and user_config.volume_suite_enable:
        enabled_screeners.append(f"Volume Suite hits: {screener_summary['volume_suite_hits']}")
    if user_config and hasattr(user_config, 'stockbee_suite_enable') and user_config.stockbee_suite_enable:
        enabled_screeners.append(f"Stockbee Suite hits: {screener_summary['stockbee_suite_hits']}")
    if user_config and hasattr(user_config, 'qullamaggie_suite_enable') and user_config.qullamaggie_suite_enable:
        enabled_screeners.append(f"Qullamaggie Suite hits: {screener_summary['qullamaggie_suite_hits']}")
    if user_config and hasattr(user_config, 'adl_screener_enable') and user_config.adl_screener_enable:
        enabled_screeners.append(f"ADL Screener hits: {screener_summary['adl_screener_hits']}")
    if user_config and hasattr(user_config, 'guppy_screener_enable') and user_config.guppy_screener_enable:
        enabled_screeners.append(f"Guppy GMMA hits: {screener_summary['guppy_screener_hits']}")
    if user_config and hasattr(user_config, 'gold_launch_pad_enable') and user_config.gold_launch_pad_enable:
        enabled_screeners.append(f"Gold Launch Pad hits: {screener_summary['gold_launch_pad_hits']}")
    if user_config and hasattr(user_config, 'rti_enable') and user_config.rti_enable:
        enabled_screeners.append(f"RTI hits: {screener_summary['rti_hits']}")
    # Market Pulse reporting moved to main.py pipeline level
    
    if enabled_screeners or len(all_results) > 0:
        print(f"📊 Screening Results ({timeframe}):")
        for result in enabled_screeners:
            print(f"  • {result}")
        print(f"  • Total screening hits: {len(all_results)}")
    
    if all_results:
        print(f"  • Results saved: {results_file.name}")
        if top_picks:
            print(f"  • Top picks saved: {top_picks_file.name}")
            # Show top 3 overall picks (by volume or signal strength)
            best_picks = sorted(all_results, key=lambda x: x.get('score', x.get('volume', x.get('price', 0))), reverse=True)[:3]
            print("  • Top 3 picks:", [f"{pick['ticker']} ({pick['screen_type']})" for pick in best_picks])
    
    logger.info(f"Screening completed: {len(all_results)} total hits across all screeners")
    
    return len(all_results)


# Maintain backward compatibility by exposing individual screeners
__all__ = [
    'run_screeners',
    'momentum_screener',
    'breakout_screener', 
    'value_momentum_screener',
    'pvb_screener',
    'atr1_screener',
    'atr2_screener',
    'giusti_screener',
    'minervini_screener',
    'drwish_screener'
]
