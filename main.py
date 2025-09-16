"""
Post-Processing Financial Data Analysis System - Main Pipeline
==============================================================

This is the main entry point for post-processing downloaded market data.
Serves as the high-level orchestrator coordinating all processing stages.
"""

import pandas as pd
import os
import logging
import math
from pathlib import Path
from typing import List, Optional

from src.data_reader import DataReader
from src.config import Config
from src.user_defined_data import read_user_data, UserConfiguration
from src.unified_ticker_generator import generate_all_ticker_files

# Import post-processing modules
from src.basic_calculations import basic_calculations, save_basic_calculations_matrix
from src.market_breadth_calculation import save_market_breadth_matrix
from src.stage_analysis_processor import run_stage_analysis_processing, StageAnalysisProcessor
from src.pvb_screener_processor import PVBScreenerProcessor
from src.index_overview_file import create_index_overview
from src.run_screeners import run_screeners
from src.models import run_models
from src.percentage_movers import run_movers_analysis

logger = logging.getLogger(__name__)


def setup_timeframe_directories(config, timeframe):
    """
    Create output directories for a specific timeframe.
    
    Args:
        config: Config object with directory paths
        timeframe: Timeframe name ('daily', 'weekly', etc.)
        
    Returns:
        Path: Output base directory for the timeframe
    """
    try:
        # Create output directories for this timeframe
        output_base = config.directories['RESULTS_DIR'] / timeframe
        os.makedirs(output_base, exist_ok=True)
        os.makedirs(output_base / 'calculations', exist_ok=True)
        os.makedirs(output_base / 'overview', exist_ok=True)
        os.makedirs(output_base / 'screeners', exist_ok=True)
        os.makedirs(output_base / 'models', exist_ok=True)
        
        print(f"📁 Output directory: {output_base}")
        return output_base
        
    except Exception as e:
        logger.error(f"Error setting up directories for {timeframe}: {e}")
        return None


def print_data_summary(data_reader, timeframe):
    """
    Print data summary for a timeframe.
    
    Args:
        data_reader: DataReader instance
        timeframe: Timeframe name
    """
    try:
        # Get data summary
        summary = data_reader.get_data_summary()
        print(f"📊 Data summary: {summary['available_files']} files, "
              f"{summary['valid_files']} valid, "
              f"avg {summary['avg_data_points']} points per ticker")
        
        if summary['date_range']['start']:
            print(f"📅 Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        logger.info(f"{timeframe} data summary: {summary}")
        
    except Exception as e:
        print(f"⚠️  Could not generate data summary: {e}")
        logger.warning(f"Could not generate data summary for {timeframe}: {e}")


def setup_logging() -> None:
    """Configure logging for the main pipeline."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().setLevel(logging.WARNING)  # Reduce noise


def process_timeframe_batches(
    data_reader: DataReader, 
    ticker_list: List[str], 
    timeframe: str, 
    output_base: Path,
    batch_size: int,
    config: Config,
    user_config: UserConfiguration
) -> int:
    """
    Process all batches for a specific timeframe.
    
    Args:
        data_reader: Configured DataReader instance
        ticker_list: List of ticker symbols to process
        timeframe: Data timeframe ('daily', 'weekly', 'monthly')
        output_base: Output directory path
        batch_size: Number of tickers per batch
        config: System configuration
        user_config: User configuration
        
    Returns:
        Total number of tickers processed
    """
    total_tickers = len(ticker_list)
    total_batches = math.ceil(total_tickers / batch_size)
    total_processed = 0
    
    logger.info(f"Processing {total_tickers} tickers in {total_batches} batches for {timeframe}")
    print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")
    
    try:
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1
            
            print(f"\n🔄 Processing batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")
            
            try:
                # Read CSV data for the batch
                batch_data = data_reader.read_batch_data(batch_tickers, validate=True)
                
                if not batch_data:
                    logger.warning(f"No valid data in batch {batch_count}")
                    print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                    continue
                    
                print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")
                total_processed += len(batch_data)
                
                # Run all post-processing operations
                _run_batch_operations(batch_data, output_base, timeframe, batch_count, total_batches, user_config, config, data_reader)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_count}: {e}")
                print(f"❌ Error processing batch {batch_count}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Critical error in batch processing for {timeframe}: {e}")
        raise
    
    return total_processed


def run_all_basic_calculations_streaming(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Run basic calculations for all timeframes using streaming approach.
    Memory-efficient version that writes results immediately instead of accumulating.

    Returns:
        dict: Results summary for all timeframes
    """
    print(f"\n" + "="*60)
    print("🔧 BASIC CALCULATIONS - ALL TIMEFRAMES, ALL BATCHES (STREAMING)")
    print("="*60)

    from src.basic_calculations_streaming import BasicCalculationsStreamingProcessor

    # Initialize streaming processor
    processor = BasicCalculationsStreamingProcessor(config, user_config)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        basic_calc_enabled = getattr(user_config, f'basic_calc_{timeframe}_enable', True)
        if not basic_calc_enabled:
            print(f"⏭️  Basic calculations disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        batches = []

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        # Collect all batches first
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Loading batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")
            batches.append(batch_data)

        # Process all batches with streaming
        if batches:
            result = processor.process_timeframe_streaming(batches, timeframe, user_config.ticker_choice)

            if result and 'tickers_processed' in result:
                timeframe_processed = result['tickers_processed']
                print(f"✅ Basic calculations completed for {timeframe}: {timeframe_processed} tickers")
                print(f"📁 Results saved to: {result['output_file']}")

                if 'memory_saved_mb' in result:
                    print(f"💾 Memory saved: {result['memory_saved_mb']:.1f} MB")

                results_summary[timeframe] = timeframe_processed
                total_processed += timeframe_processed
            else:
                print(f"⚠️  No results from streaming processing for {timeframe}")
        else:
            print(f"⚠️  No valid batches for {timeframe}")

    print(f"\n✅ BASIC CALCULATIONS COMPLETED!")
    print(f"📊 Total tickers processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def run_all_stage_analysis_streaming(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Run stage analysis for all timeframes using streaming approach.
    Memory-efficient version that writes results immediately instead of accumulating.

    Returns:
        dict: Results summary for all timeframes
    """
    if not getattr(user_config, 'enable_stage_analysis', True):
        print(f"\n⏭️  Stage analysis disabled - skipping stage analysis processing")
        return {}

    print(f"\n" + "="*60)
    print("🎯 STAGE ANALYSIS - ALL TIMEFRAMES, ALL BATCHES (STREAMING)")
    print("="*60)

    from src.stage_analysis_streaming import StageAnalysisStreamingProcessor

    # Initialize streaming processor
    processor = StageAnalysisStreamingProcessor(config, user_config)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        stage_enabled = getattr(user_config, f'stage_analysis_{timeframe}_enabled', True)
        if not stage_enabled:
            print(f"⏭️  Stage analysis disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        batches = []

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        # Collect all batches first
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Loading batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")
            batches.append(batch_data)

        # Process all batches with streaming
        if batches:
            result = processor.process_timeframe_streaming(batches, timeframe, user_config.ticker_choice)

            if result and 'tickers_processed' in result:
                timeframe_processed = result['tickers_processed']
                print(f"✅ Stage analysis completed for {timeframe}: {timeframe_processed} tickers")
                print(f"📁 Results saved to: {result['output_file']}")

                if 'memory_saved_mb' in result:
                    print(f"💾 Memory saved: {result['memory_saved_mb']:.1f} MB")

                results_summary[timeframe] = timeframe_processed
                total_processed += timeframe_processed
            else:
                print(f"⚠️  No results from streaming processing for {timeframe}")
        else:
            print(f"⚠️  No valid batches for {timeframe}")

    print(f"\n✅ STAGE ANALYSIS COMPLETED!")
    print(f"📊 Total tickers processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def run_all_pvb_screener_streaming(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Run PVB screener for all timeframes using streaming approach.
    Memory-efficient version that writes results immediately instead of accumulating.

    Returns:
        dict: Results summary for all timeframes
    """
    if not getattr(user_config, 'pvb_enable', False):
        print(f"\n⏭️  PVB screener disabled - skipping PVB processing")
        return {}

    print(f"\n" + "="*60)
    print("🔍 PVB SCREENER - ALL TIMEFRAMES, ALL BATCHES (STREAMING)")
    print("="*60)

    from src.screeners_streaming import PVBScreenerStreamingProcessor

    # Initialize streaming processor
    processor = PVBScreenerStreamingProcessor(config, user_config)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        pvb_enabled = getattr(user_config, f'pvb_{timeframe}_enable', True)
        if not pvb_enabled:
            print(f"⏭️  PVB screener disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        batches = []

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        # Collect all batches first
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Loading batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")
            batches.append(batch_data)

        # Process all batches with streaming
        if batches:
            result = processor.process_timeframe_streaming(batches, timeframe, user_config.ticker_choice)

            if result and 'tickers_processed' in result:
                timeframe_processed = result['tickers_processed']
                print(f"✅ PVB screener completed for {timeframe}: {timeframe_processed} results")
                print(f"📁 Results saved to: {result['output_file']}")

                if 'memory_saved_mb' in result:
                    print(f"💾 Memory saved: {result['memory_saved_mb']:.1f} MB")

                results_summary[timeframe] = timeframe_processed
                total_processed += timeframe_processed
            else:
                print(f"⚠️  No results from streaming processing for {timeframe}")
        else:
            print(f"⚠️  No valid batches for {timeframe}")

    print(f"\n✅ PVB SCREENER COMPLETED!")
    print(f"📊 Total results processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def run_all_atr1_screener_streaming(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Run ATR1 screener for all timeframes using streaming approach.
    Memory-efficient version that writes results immediately instead of accumulating.

    Returns:
        dict: Results summary for all timeframes
    """
    if not getattr(user_config, 'atr1_enable', False):
        print(f"\n⏭️  ATR1 screener disabled - skipping ATR1 processing")
        return {}

    print(f"\n" + "="*60)
    print("🔍 ATR1 SCREENER - ALL TIMEFRAMES, ALL BATCHES (STREAMING)")
    print("="*60)

    from src.screeners_streaming import ATR1ScreenerStreamingProcessor

    # Initialize streaming processor
    processor = ATR1ScreenerStreamingProcessor(config, user_config)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        atr1_enabled = getattr(user_config, f'atr1_{timeframe}_enable', True)
        if not atr1_enabled:
            print(f"⏭️  ATR1 screener disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        batches = []

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        # Collect all batches first
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Loading batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")
            batches.append(batch_data)

        # Process all batches with streaming
        if batches:
            result = processor.process_timeframe_streaming(batches, timeframe, user_config.ticker_choice)

            if result and 'tickers_processed' in result:
                timeframe_processed = result['tickers_processed']
                print(f"✅ ATR1 screener completed for {timeframe}: {timeframe_processed} results")
                print(f"📁 Results saved to: {result['output_file']}")

                if 'memory_saved_mb' in result:
                    print(f"💾 Memory saved: {result['memory_saved_mb']:.1f} MB")

                results_summary[timeframe] = timeframe_processed
                total_processed += timeframe_processed
            else:
                print(f"⚠️  No results from streaming processing for {timeframe}")
        else:
            print(f"⚠️  No valid batches for {timeframe}")

    print(f"\n✅ ATR1 SCREENER COMPLETED!")
    print(f"📊 Total results processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


# Legacy function for backward compatibility
def run_all_basic_calculations(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Legacy function - redirects to streaming version.
    """
    return run_all_basic_calculations_streaming(config, user_config, timeframes, clean_file_path)


def run_all_stage_analysis(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Legacy function - redirects to streaming version.
    """
    return run_all_stage_analysis_streaming(config, user_config, timeframes, clean_file_path)


def run_all_pvb_screener(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Legacy function - redirects to streaming version.
    """
    return run_all_pvb_screener_streaming(config, user_config, timeframes, clean_file_path)


def run_all_atr1_screener(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Legacy function - redirects to streaming version.
    """
    return run_all_atr1_screener_streaming(config, user_config, timeframes, clean_file_path)


def run_all_drwish_screener_streaming(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Run DRWISH screener for all timeframes using streaming approach.
    Memory-efficient version that writes results immediately instead of accumulating.

    Returns:
        dict: Results summary for all timeframes
    """
    if not getattr(user_config, 'drwish_enable', False):
        print(f"\n⏭️  DRWISH screener disabled - skipping DRWISH processing")
        return {}

    print(f"\n" + "="*60)
    print("🔍 DRWISH SCREENER - ALL TIMEFRAMES, ALL BATCHES (STREAMING)")
    print("="*60)

    from src.screeners_streaming import DRWISHScreenerStreamingProcessor

    # Initialize streaming processor
    processor = DRWISHScreenerStreamingProcessor(config, user_config)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        drwish_enabled = getattr(user_config, f'drwish_{timeframe}_enable', True)
        if not drwish_enabled:
            print(f"⏭️  DRWISH screener disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        batches = []

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        # Collect all batches first
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Loading batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")
            batches.append(batch_data)

        # Process all batches with streaming
        if batches:
            result = processor.process_timeframe_streaming(batches, timeframe, user_config.ticker_choice)

            if result and 'tickers_processed' in result:
                timeframe_processed = result['tickers_processed']
                print(f"✅ DRWISH screener completed for {timeframe}: {timeframe_processed} results")
                print(f"📁 Results saved to: {result['output_file']}")

                if 'memory_saved_mb' in result:
                    print(f"💾 Memory saved: {result['memory_saved_mb']:.1f} MB")

                results_summary[timeframe] = timeframe_processed
                total_processed += timeframe_processed
            else:
                print(f"⚠️  No results from streaming processing for {timeframe}")
        else:
            print(f"⚠️  No valid batches for {timeframe}")

    print(f"\n✅ DRWISH SCREENER COMPLETED!")
    print(f"📊 Total results processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def run_all_drwish_screener(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Legacy function - redirects to streaming version.
    """
    return run_all_drwish_screener_streaming(config, user_config, timeframes, clean_file_path)


# End of streaming functions - main() function below


def legacy_batch_operations(
    batch_num: int,
    batch_count: int,
    batch_tickers: List[str],
    total_batches: int,
    user_config: UserConfiguration,
    config: Config,
    data_reader=None
) -> None:
    """
    DEPRECATED: Legacy batch operations function.
    This function is being replaced by the new grouped calculation approach.
    Kept for compatibility during transition.
    """
    pass


def old_legacy_batch_operations(
    batch_num: int,
    batch_count: int,
    batch_tickers: List[str],
    total_batches: int,
    user_config: UserConfiguration,
    config: Config,
    data_reader=None
) -> None:
    """
    DEPRECATED: Legacy batch operations function.
    This function is being replaced by the new grouped calculation approach.
    Kept for compatibility during transition.
    """
    pass

    for timeframe in timeframes:
        # Check if PVB is enabled for this timeframe (if timeframe-specific settings exist)
        pvb_timeframe_enabled = getattr(user_config, f'pvb_{timeframe}_enable', True)
        if not pvb_timeframe_enabled:
            print(f"⏭️  PVB screener disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches for PVB screener
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        timeframe_processed = 0

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Processing batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")

            try:
                # Process this batch (accumulates in processor.all_results)
                success = processor.process_pvb_batch(batch_data, timeframe, user_config.ticker_choice)
                if success:
                    timeframe_processed += len(batch_data)

            except Exception as e:
                logger.error(f"Error in PVB screener batch {batch_count} for {timeframe}: {e}")
                print(f"❌ Error processing batch {batch_count}: {e}")
                continue

        if timeframe_processed > 0:
            print(f"✅ PVB screener completed for {timeframe}: {timeframe_processed} tickers")
            results_summary[timeframe] = timeframe_processed
            total_processed += timeframe_processed

            # Save matrix immediately after processing all batches for this timeframe
            print(f"💾 Saving PVB screener matrix for {timeframe}...")
            matrix_results = processor.save_pvb_matrix(user_config.ticker_choice)
            if matrix_results and timeframe in matrix_results:
                print(f"✅ Matrix saved: {matrix_results[timeframe]}")

    print(f"\n✅ PVB SCREENER COMPLETED!")
    print(f"📊 Total tickers processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def run_all_atr1_screener(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Run ATR1 screener for all timeframes and all batches.
    Now uses accumulation pattern like basic_calculations.
    """
    if not getattr(user_config, 'atr1_enable', False):
        print(f"\n⏭️  ATR1 screener disabled - skipping ATR1 processing")
        return {}

    print(f"\n" + "="*60)
    print("🔍 ATR1 SCREENER - ALL TIMEFRAMES, ALL BATCHES")
    print("="*60)

    # Initialize global ATR1 screener processor
    from src.atr1_screener_processor import ATR1ScreenerProcessor
    processor = ATR1ScreenerProcessor(config, user_config)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        # Check if ATR1 is enabled for this timeframe (if timeframe-specific settings exist)
        atr1_timeframe_enabled = getattr(user_config, f'atr1_{timeframe}_enable', True)
        if not atr1_timeframe_enabled:
            print(f"⏭️  ATR1 screener disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches for ATR1 screener
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        timeframe_processed = 0

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Processing batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")

            try:
                # Process this batch (accumulates in processor.all_results)
                success = processor.process_atr1_batch(batch_data, timeframe, user_config.ticker_choice)
                if success:
                    timeframe_processed += len(batch_data)

            except Exception as e:
                logger.error(f"Error in ATR1 screener batch {batch_count} for {timeframe}: {e}")
                print(f"❌ Error processing batch {batch_count}: {e}")
                continue

        if timeframe_processed > 0:
            print(f"✅ ATR1 screener completed for {timeframe}: {timeframe_processed} tickers")
            results_summary[timeframe] = timeframe_processed
            total_processed += timeframe_processed

            # Save matrix immediately after processing all batches for this timeframe
            print(f"💾 Saving ATR1 screener matrix for {timeframe}...")
            matrix_results = processor.save_atr1_matrix(user_config.ticker_choice)
            if matrix_results and timeframe in matrix_results:
                print(f"✅ Matrix saved: {matrix_results[timeframe]}")

    print(f"\n✅ ATR1 SCREENER COMPLETED!")
    print(f"📊 Total tickers processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def run_all_screeners(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Run screeners for all timeframes and all batches.
    """
    from src.run_screeners import has_any_screeners_enabled

    if not has_any_screeners_enabled(user_config):
        print(f"\n⏭️  All screeners disabled - skipping screener processing")
        return {}

    print(f"\n" + "="*60)
    print("🔍 SCREENERS - ALL TIMEFRAMES, ALL BATCHES")
    print("="*60)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Create output directory
        output_base = config.directories['RESULTS_DIR'] / timeframe
        (output_base / 'screeners').mkdir(parents=True, exist_ok=True)

        # Process all batches for screeners
        batch_size = getattr(user_config, 'batch_size', 100)
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        timeframe_processed = 0

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Processing batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")

            try:
                run_screeners(batch_data, output_base / 'screeners', timeframe, user_config, data_reader)
                timeframe_processed += len(batch_data)

            except Exception as e:
                logger.error(f"Error in screeners batch {batch_count} for {timeframe}: {e}")
                print(f"❌ Error processing batch {batch_count}: {e}")
                continue

        if timeframe_processed > 0:
            print(f"✅ Screeners completed for {timeframe}: {timeframe_processed} tickers")
            results_summary[timeframe] = timeframe_processed
            total_processed += timeframe_processed

    print(f"\n✅ SCREENERS COMPLETED!")
    print(f"📊 Total tickers processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def run_all_market_breadth(config: Config, user_config: UserConfiguration, timeframes: List[str]) -> dict:
    """
    Run market breadth analysis for all timeframes.
    Positioned between basic calculations and stage analysis for optimal data flow.
    """
    if not user_config.market_breadth_enable:
        print(f"\n⏭️  Market breadth analysis disabled - skipping")
        return {}

    print(f"\n" + "="*60)
    print("📊 MARKET BREADTH ANALYSIS - ALL TIMEFRAMES")
    print("="*60)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        try:
            from src.market_breadth_calculation import MarketBreadthCalculator

            breadth_calculator = MarketBreadthCalculator(config)
            breadth_matrix_results = breadth_calculator.calculate_universe_breadth_matrix(timeframe, user_config)

            if breadth_matrix_results:
                print(f"✅ Market breadth matrix calculated for {timeframe}")
                print(f"  • Universes processed: {len(breadth_matrix_results)}")
                for universe_name, result_df in breadth_matrix_results.items():
                    if not result_df.empty:
                        print(f"  • {universe_name}: {len(result_df)} trading days")

                # Save market breadth matrices immediately
                print(f"💾 Saving market breadth matrices for {timeframe}...")
                from src.market_breadth_calculation import save_market_breadth_matrix
                saved_results = save_market_breadth_matrix(config, user_config, timeframe)

                if saved_results:
                    print(f"✅ Market breadth matrices saved for {timeframe}:")
                    for result in saved_results:
                        universe = result['universe']
                        print(f"  • {universe}: {result['output_file']}")
                        print(f"    📅 Data date: {result['formatted_date']}")

                    results_summary[timeframe] = len(saved_results)
                    total_processed += len(saved_results)
                else:
                    print(f"⚠️  No market breadth matrices saved for {timeframe}")

            else:
                print(f"⚠️  No market breadth matrices generated for {timeframe}")

        except Exception as e:
            print(f"❌ Market breadth calculation error for {timeframe}: {e}")
            logger.error(f"Market breadth calculation failed for {timeframe}: {e}")
            continue

    print(f"\n✅ MARKET BREADTH ANALYSIS COMPLETED!")
    print(f"📊 Total matrices processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def run_all_additional_calculations(config: Config, user_config: UserConfiguration, timeframes: List[str], clean_file_path: str) -> dict:
    """
    Run additional calculations (technical indicators, models) for all timeframes.
    Market breadth is now handled by run_all_market_breadth() function.
    """
    print(f"\n" + "="*60)
    print("📈 ADDITIONAL CALCULATIONS - ALL TIMEFRAMES, ALL BATCHES")
    print("="*60)

    results_summary = {}
    total_processed = 0

    for timeframe in timeframes:
        print(f"\n📊 Processing {timeframe.upper()} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Create output directories
        output_base = config.directories['RESULTS_DIR'] / timeframe
        (output_base / 'indicators').mkdir(parents=True, exist_ok=True)
        (output_base / 'models').mkdir(parents=True, exist_ok=True)

        # Process all batches for additional calculations
        batch_size = getattr(user_config, 'batch_size', 100)
        total_tickers = len(ticker_list)
        total_batches = math.ceil(total_tickers / batch_size)
        timeframe_processed = 0

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]
            batch_count = batch_num + 1

            print(f"🔄 Processing batch {batch_count}/{total_batches} ({len(batch_tickers)} tickers) - {(batch_count/total_batches)*100:.1f}%")

            # Read CSV data for the batch
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if not batch_data:
                print(f"⚠️  No valid data in batch {batch_count}, skipping...")
                continue

            print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_count}")

            try:
                # Technical Indicators
                print("📈 Running technical indicators...")
                from src.indicators import process_indicators_batch
                process_indicators_batch(batch_data, output_base / 'indicators', timeframe, user_config)

                # Models
                print("🤖 Running models...")
                run_models(batch_data, output_base / 'models', timeframe)

                timeframe_processed += len(batch_data)

            except Exception as e:
                logger.error(f"Error in additional calculations batch {batch_count} for {timeframe}: {e}")
                print(f"❌ Error processing batch {batch_count}: {e}")
                continue

        if timeframe_processed > 0:
            print(f"✅ Additional calculations completed for {timeframe}: {timeframe_processed} tickers")
            results_summary[timeframe] = timeframe_processed
            total_processed += timeframe_processed

    print(f"\n✅ ADDITIONAL CALCULATIONS COMPLETED!")
    print(f"📊 Total tickers processed: {total_processed}")
    print(f"🕒 Timeframes processed: {', '.join(results_summary.keys())}")

    return results_summary


def _run_batch_operations(
    batch_data: dict,
    output_base: Path,
    timeframe: str,
    batch_count: int,
    total_batches: int,
    user_config: UserConfiguration,
    config: Config,
    data_reader=None
) -> None:
    """
    DEPRECATED: Legacy batch operations function.
    This function is being replaced by the new grouped calculation approach.
    Kept for compatibility during transition.
    """
    pass


def main() -> None:
    """
    Main post-processing pipeline orchestrator.
    
    Coordinates all processing stages:
    1. Configuration and preprocessing
    2. Timeframe processing with batch operations  
    3. Post-processing aggregation and analysis
    """
    # Configuration parameters
    batch_size = 100  # TODO: Move to user_data.csv
    
    # Setup logging
    setup_logging()
    logger.info("Starting financial data post-processing system")
    
    try:
        print("🚀 FINANCIAL DATA POST-PROCESSING SYSTEM")
        print("="*60)
        print("Phase 2: Batch Analysis of Downloaded Market Data")
        print("="*60)
        
        # Read configuration
        user_config = read_user_data()
        config = Config()
        logger.info(f"Configuration loaded: ticker choice {user_config.ticker_choice}")

        # Ensure all directories exist
        print("📁 Creating required directories...")
        config.create_directories()
        logger.info("All required directories created/verified")

        # Generate all ticker files using unified system
        print(f"📋 User choice: {user_config.ticker_choice}")
        success = generate_all_ticker_files(config, user_config.ticker_choice)
        if not success:
            logger.error("Ticker file generation failed")
            print("❌ Ticker file generation failed. Cannot continue with data processing.")
            return
        
        # Use the generated clean info file for processing
        clean_file = str(config.directories['TICKERS_DIR'] / f'combined_info_tickers_clean_{user_config.ticker_choice}.csv')
        print(f"📊 Using clean ticker file for processing: {clean_file}")
        
        # Validate clean file exists and load tickers
        if not Path(clean_file).exists():
            logger.error(f"Clean ticker file not found: {clean_file}")
            print(f"❌ Clean ticker file not found: {clean_file}")
            return
            
        tickers_df = pd.read_csv(clean_file)
        print(f"📊 Total tickers for processing: {len(tickers_df)}")
        print(f"🔍 Sample tickers: {tickers_df['ticker'].head(10).tolist()}")
        
        # Define timeframes to process
        timeframes_to_process = []
        if user_config.load_daily_data:
            timeframes_to_process.append('daily')
        if user_config.load_weekly_data:
            timeframes_to_process.append('weekly') 
        if user_config.load_monthly_data:
            timeframes_to_process.append('monthly')
        if user_config.load_intraday_data:
            timeframes_to_process.append('intraday')
        
        if not timeframes_to_process:
            print("❌ No timeframes enabled for processing!")
            print("   Please enable at least one: load_daily_data, load_weekly_data, etc.")
            return
        
        print(f"📈 Processing timeframes: {', '.join(timeframes_to_process)}")
            
    except Exception as e:
        logger.error(f"Critical error during initialization: {e}")
        print(f"❌ Critical error during initialization: {e}")
        return
    
    # UNIVERSE GENERATION (before RS analysis)
    print(f"\n" + "="*60)
    print("UNIVERSE GENERATION")
    print("="*60)
    
    # Import and run simple universe generation
    from src.simple_universe_generator import SimpleUniverseGenerator
    universe_generator = SimpleUniverseGenerator(config)
    
    # Generate all possible universe files from boolean data
    universe_results = universe_generator.generate_all_universes()
    
    print(f"✅ Generated {len(universe_results)} universe files:")
    
    # Show master universe first (if exists)
    if 'ticker_universe_all' in universe_results:
        master_count = universe_results['ticker_universe_all']
        print(f"  🌐 Master universe: {master_count} tickers (source of truth)")
    
    # Group results by type for better display (exclude master universe from index list)
    index_universes = {k: v for k, v in universe_results.items() if k.startswith('ticker_universe_') and not any(x in k for x in ['sectors_', 'industry_', 'market_cap_', 'all'])}
    sector_universes = {k: v for k, v in universe_results.items() if 'sectors_' in k}
    industry_universes = {k: v for k, v in universe_results.items() if 'industry_' in k}
    market_cap_universes = {k: v for k, v in universe_results.items() if 'market_cap_' in k}
    
    if index_universes:
        print(f"  📈 Index universes: {len(index_universes)}")
        for name, count in list(index_universes.items())[:5]:  # Show first 5
            short_name = name.replace('ticker_universe_', '')
            print(f"    • {short_name}: {count} tickers")
        if len(index_universes) > 5:
            print(f"    ... and {len(index_universes) - 5} more")
    
    if sector_universes:
        print(f"  🏭 Sector universes: {len(sector_universes)}")
        for name, count in list(sector_universes.items())[:3]:  # Show first 3
            short_name = name.replace('ticker_universe_sectors_', '')
            print(f"    • {short_name}: {count} tickers")
        if len(sector_universes) > 3:
            print(f"    ... and {len(sector_universes) - 3} more")
    
    if industry_universes:
        print(f"  🔧 Industry universes: {len(industry_universes)}")
        for name, count in list(industry_universes.items())[:3]:  # Show first 3
            short_name = name.replace('ticker_universe_industry_', '')
            print(f"    • {short_name}: {count} tickers")
        if len(industry_universes) > 3:
            print(f"    ... and {len(industry_universes) - 3} more")
    
    if market_cap_universes:
        print(f"  💰 Market cap universes: {len(market_cap_universes)}")
        for name, count in market_cap_universes.items():
            short_name = name.replace('ticker_universe_market_cap_', '')
            print(f"    • {short_name}: {count} tickers")
    
    print(f"📊 Universe files directory: {config.directories['RESULTS_DIR'] / 'ticker_universes'}")
    
    # RELATIVE STRENGTH ANALYSIS (moved before timeframe processing)
    print(f"\n" + "="*60)
    print("RELATIVE STRENGTH (RS) ANALYSIS")
    print("="*60)
    
    # Import RS processing module
    from src.rs_processor import run_rs_analysis
    
    # Run RS analysis if enabled
    rs_results = run_rs_analysis(
        ticker_list=tickers_df['ticker'].tolist(),
        config=config,
        user_config=user_config,
        ticker_choice=user_config.ticker_choice
    )
    
    if rs_results.get('status') == 'skipped':
        print(f"⏭️  RS analysis skipped: {rs_results['reason']}")
    else:
        # RS analysis completion output is now handled by the processor itself
        # Just show final summary details
        print(f"📊 Benchmark: {rs_results['benchmark_ticker']}")
        print(f"📁 Files created: {len(rs_results['files_created'])}")

        if rs_results['errors']:
            print(f"⚠️  Errors encountered: {len(rs_results['errors'])}")
            for error in rs_results['errors'][:3]:  # Show first 3 errors
                print(f"   • {error}")

        # Show group statistics if available
        if 'summary' in rs_results and rs_results['summary']['group_statistics']:
            group_stats = rs_results['summary']['group_statistics']
            print(f"📈 Sector composites: {group_stats['sector_count']} sectors")
            print(f"🏭 Industry composites: {group_stats['industry_count']} industries")

        # RS files now contain only RS values - percentiles handled separately
        print(f"✅ RS files generated containing RS values and returns")
        print(f"📁 RS files contain all periods and benchmarks combined")
    
    # PERCENTILE (PER) ANALYSIS - Separate from RS analysis
    print(f"\n" + "="*60)
    print("PERCENTILE (PER) ANALYSIS")
    print("="*60)
    
    # Import PER processing module
    from src.per_processor import run_per_analysis
    
    # Run PER analysis if RS analysis was successful
    if rs_results.get('status') != 'skipped':
        per_results = run_per_analysis(
            config=config,
            user_config=user_config,
            ticker_choice=user_config.ticker_choice
        )
        
        if per_results.get('status') == 'skipped':
            print(f"⏭️  PER analysis skipped: {per_results['reason']}")
        else:
            print(f"✅ PER analysis completed")
            print(f"📊 Benchmarks: {per_results['benchmark_tickers']}")
            print(f"🕒 Timeframes processed: {', '.join(per_results['timeframes_processed'])}")
            print(f"📁 Files created: {len(per_results['files_created'])}")
            
            if per_results['errors']:
                print(f"⚠️  Errors encountered: {len(per_results['errors'])}")
                for error in per_results['errors'][:3]:  # Show first 3 errors
                    print(f"   • {error}")
            
            print(f"\n✅ PER files generated with multi-universe percentile rankings")
            print(f"📁 PER files contain percentiles for different universe configurations")
    else:
        print(f"⏭️  PER analysis skipped: RS analysis was not performed")
    
    # NEW GROUPED CALCULATION STRUCTURE
    # =====================================
    # Process calculations grouped by type across all timeframes for better efficiency

    # 1. Basic Calculations - All timeframes, all batches
    basic_calc_results = run_all_basic_calculations(config, user_config, timeframes_to_process, clean_file)

    # 2. Market Breadth Analysis - All timeframes (positioned after basic calculations)
    market_breadth_results = run_all_market_breadth(config, user_config, timeframes_to_process)

    # 3. Stage Analysis - All timeframes, all batches
    stage_analysis_results = run_all_stage_analysis(config, user_config, timeframes_to_process, clean_file)

    # 4. PVB Screener - All timeframes, all batches (NEW - individual screener)
    pvb_screener_results = run_all_pvb_screener(config, user_config, timeframes_to_process, clean_file)

    # 5. ATR1 Screener - All timeframes, all batches (NEW - individual screener)
    atr1_screener_results = run_all_atr1_screener(config, user_config, timeframes_to_process, clean_file)

    # 6. DRWISH Screener - All timeframes, all batches (NEW - individual screener)
    drwish_screener_results = run_all_drwish_screener(config, user_config, timeframes_to_process, clean_file)

    # 7. Screeners - All timeframes, all batches (OLD - remaining mixed screeners)
    # COMMENTED OUT - Using individual screener implementations instead
    # screener_results = run_all_screeners(config, user_config, timeframes_to_process, clean_file)
    screener_results = {}  # Placeholder to avoid breaking summary section

    # 5. Additional Calculations - All timeframes, all batches
    # COMMENTED OUT - Additional calculations disabled
    # additional_calc_results = run_all_additional_calculations(config, user_config, timeframes_to_process, clean_file)
    additional_calc_results = {}  # Placeholder to avoid breaking summary section

    # Summary of all calculation phases
    print(f"\n" + "="*60)
    print("📋 CALCULATION PHASES SUMMARY")
    print("="*60)
    print(f"✅ Basic Calculations: {sum(basic_calc_results.values())} total tickers processed")
    print(f"✅ Market Breadth Analysis: {sum(market_breadth_results.values())} total matrices processed")
    print(f"✅ Stage Analysis: {sum(stage_analysis_results.values())} total tickers processed")
    print(f"✅ PVB Screener: {sum(pvb_screener_results.values())} total tickers processed")
    print(f"✅ ATR1 Screener: {sum(atr1_screener_results.values())} total tickers processed")
    print(f"✅ DRWISH Screener: {sum(drwish_screener_results.values())} total tickers processed")
    # print(f"✅ Screeners: {sum(screener_results.values())} total tickers processed")  # OLD - now using individual screeners
    print(f"✅ Additional Calculations: {sum(additional_calc_results.values())} total tickers processed")
    print(f"🕒 Timeframes completed: {', '.join(timeframes_to_process)}")

    # Store data dates for downstream processes (maintaining compatibility)
    if not hasattr(main, 'data_dates'):
        main.data_dates = {}

    # Only initialize with fallback values for timeframes that don't have actual dates
    for timeframe in timeframes_to_process:
        if timeframe not in main.data_dates:
            # Try to extract date from basic calculation files as fallback
            basic_calc_file = config.directories['BASIC_CALCULATION_DIR'] / f'basic_calculation_{user_config.ticker_choice}_{timeframe}_*.csv'
            import glob
            matching_files = glob.glob(str(basic_calc_file))

            if matching_files:
                # Extract date from filename (e.g., basic_calculation_0-5_daily_20250905.csv)
                filename = Path(matching_files[-1]).stem  # Get most recent file
                parts = filename.split('_')
                if len(parts) >= 4:
                    extracted_date = parts[-1]  # Last part should be date
                    formatted_date = f"{extracted_date[:4]}-{extracted_date[4:6]}-{extracted_date[6:8]}" if len(extracted_date) >= 8 else extracted_date
                    main.data_dates[timeframe] = {
                        'data_date': extracted_date,
                        'formatted_date': formatted_date,
                        'output_file': matching_files[-1]
                    }
                    continue

            # Final fallback to placeholder values
            main.data_dates[timeframe] = {
                'data_date': 'calculated',  # Will be extracted from matrices as needed
                'formatted_date': 'calculated',
                'output_file': 'multiple_files_generated'
            }
    
    # 🟢 MARKET PULSE ANALYSIS - NEW PIPELINE POSITION
    # Now runs after all calculation phases are completed for better efficiency
    if user_config.market_pulse_enable:
        print(f"\n" + "="*60)
        print("MARKET PULSE ANALYSIS")
        print("="*60)
        
        from src.market_pulse import MarketPulseManager
        
        for timeframe in timeframes_to_process:
            print(f"📊 Running market pulse analysis for {timeframe}...")
            
            # Get centralized data date for this timeframe
            timeframe_data_date = None
            formatted_date = "unknown"
            if hasattr(main, 'data_dates') and timeframe in main.data_dates:
                timeframe_data_date = main.data_dates[timeframe]['data_date']
                formatted_date = main.data_dates[timeframe]['formatted_date']
                print(f"📅 Using centralized data date: {formatted_date}")
            else:
                print(f"⚠️  No centralized data date available for {timeframe}")
                # Could extract from basic calculation files as fallback
            
            # Initialize market pulse manager
            pulse_manager = MarketPulseManager(config, user_config)
            
            # Run analysis with proper date
            pulse_results = pulse_manager.run_complete_analysis(timeframe, timeframe_data_date)
            
            if pulse_results.get('success'):
                # Save with proper naming convention
                pulse_output_dir = config.directories['RESULTS_DIR'] / 'market_pulse'
                pulse_output_dir.mkdir(exist_ok=True)
                saved_files = pulse_manager.save_results(pulse_output_dir, timeframe, timeframe_data_date or "unknown")
                
                print(f"✅ Market pulse analysis completed for {timeframe}")
                print(f"📈 Indexes analyzed: {len(pulse_results.get('indexes', {}))}")
                print(f"📊 Market state: {pulse_results.get('market_summary', {}).get('overall_market_state', 'Unknown')}")
                print(f"📁 Files saved: {len(saved_files)}")
                
                # Display key insights
                market_summary = pulse_results.get('market_summary', {})
                if market_summary:
                    print(f"💡 Market Health: {market_summary.get('breadth_health', 'Unknown')}")
                    print(f"🎯 Recommendation: {market_summary.get('recommendation', 'Neutral')}")
                
                # Show top alerts
                alerts = pulse_results.get('alerts', [])
                if alerts:
                    print(f"🚨 Active alerts: {len(alerts)}")
                    for alert in alerts[:2]:  # Show top 2 alerts
                        alert_type = alert.get('type', 'Unknown')
                        alert_msg = alert.get('alert', 'Unknown')
                        print(f"  • [{alert_type}] {alert_msg}")
                
            else:
                error_msg = pulse_results.get('error', 'Unknown error')
                print(f"❌ Market pulse analysis failed for {timeframe}: {error_msg}")
    else:
        print(f"\n⏭️  Market pulse analysis disabled in user configuration")
    
    # INDEX OVERVIEW GENERATION (after RS analysis, before dashboard)
    if user_config.index_overview_enable:
        print(f"\n" + "="*60)
        print("INDEX OVERVIEW GENERATION")
        print("="*60)
        print("📋 Creating comprehensive index overview...")
        
        try:
            # Process overview for all timeframes (daily, weekly, monthly)
            overview_timeframes = ['daily', 'weekly', 'monthly']
            overview_output_path = config.directories['RESULTS_DIR'] / 'overview'
            overview_output_path.mkdir(exist_ok=True)
            
            for overview_timeframe in overview_timeframes:
                print(f"📊 Processing {overview_timeframe} timeframe overview...")
                
                try:
                    # Load data for current timeframe
                    data_reader_for_overview = DataReader(config, overview_timeframe, 500)  # Larger batch for overview
                    data_reader_for_overview.load_tickers_from_file(clean_file)
                    data_reader_for_overview.load_ticker_info()
                    
                    # Load batch data for overview
                    ticker_list = tickers_df['ticker'].tolist()
                    batch_data_for_overview = {}
                    
                    for ticker in ticker_list:
                        try:
                            ticker_data = data_reader_for_overview.load_ticker_data(ticker)
                            if ticker_data is not None and not ticker_data.empty:
                                batch_data_for_overview[ticker] = ticker_data
                        except Exception:
                            continue
                    
                    # Generate overview for current timeframe with centralized data dates
                    if batch_data_for_overview or (user_config.index_overview_indexes.strip()):
                        # Get data date for this timeframe from centralized storage
                        timeframe_data_date = None
                        if hasattr(main, 'data_dates') and overview_timeframe in main.data_dates:
                            timeframe_data_date = main.data_dates[overview_timeframe]['data_date']
                            print(f"📅 Using centralized data date for {overview_timeframe} overview: {main.data_dates[overview_timeframe]['formatted_date']}")
                        else:
                            print(f"⚠️  No centralized data date available for {overview_timeframe}, module will use discovery")
                        
                        create_index_overview(batch_data_for_overview, overview_output_path, overview_timeframe, user_config, config, timeframe_data_date)
                        if batch_data_for_overview:
                            print(f"✅ {overview_timeframe.title()} overview created for {len(batch_data_for_overview)} tickers")
                        else:
                            print(f"✅ {overview_timeframe.title()} overview created using basic calculation data (index mode)")
                    else:
                        print(f"⚠️  No data available for {overview_timeframe} overview generation")
                        
                except Exception as e:
                    logger.error(f"{overview_timeframe.title()} overview generation failed: {e}")
                    print(f"❌ {overview_timeframe.title()} overview generation failed: {e}")
            
            print(f"📁 All overview files saved to: {overview_output_path}")
                
        except Exception as e:
            logger.error(f"Index overview generation failed: {e}")
            print(f"❌ Index overview generation failed: {e}")
    
    # DASHBOARD GENERATION (final step - uses all processed data)
    if user_config.dashboard_enable:
        print(f"\n" + "="*60)
        print("GENERATING MARKET OVERVIEW DASHBOARD")
        print("="*60)
        
        try:
            from src.dashboard.real_data_connector import create_production_dashboard
            
            # Create dashboard using latest results
            dashboard_path = create_production_dashboard(
                config=config,
                user_config=user_config, 
                results_dir=config.directories['RESULTS_DIR'],
                timeframe='daily',
                data_reader=data_reader if 'data_reader' in locals() else None
            )
            
            print(f"✅ Market overview dashboard created: {dashboard_path}")
            print(f"📊 Dashboard includes: Market Pulse, Screener Results, Sector Analysis, Alerts")
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            print(f"❌ Dashboard generation failed: {e}")
    
    # Final summary
    print(f"\n" + "="*60)
    print("ALL POST-PROCESSING COMPLETED")
    print("="*60)
    print(f"✅ Processed {len(timeframes_to_process)} timeframes")
    print(f"📊 Ticker selection: Choice {user_config.ticker_choice} ({len(tickers_df)} tickers)")
    print(f"📁 Results directory: {config.directories['RESULTS_DIR']}")
    if user_config.dashboard_enable:
        print(f"📈 Dashboard output: {user_config.dashboard_output_dir}")
    
    # Generate automated report if enabled
    if getattr(user_config, 'report_enable', False):
        print("\n📄 Generating Automated Report...")
        try:
            from src.report_generator import ReportGenerator
            
            report_generator = ReportGenerator(config, user_config)
            report_file = report_generator.generate_report(main.data_dates)
            
            if report_file:
                print(f"  ✅ Report generated: {report_file}")
                print(f"  📄 Report type: {getattr(user_config, 'report_template_type', 'indexes_overview')}")
                print(f"  📐 Format: {getattr(user_config, 'report_page_size', 'A4_landscape')}")
            else:
                print(f"  ❌ Report generation failed")
                
        except Exception as e:
            print(f"  ❌ Report generation error: {e}")
            logger.warning(f"Report generation failed: {e}")
    
    print("\n🎯 Next steps:")
    print("  • Review results in the results/ directory")
    if getattr(user_config, 'report_enable', False):
        print("  • Check generated PDF report for comprehensive overview")
    if user_config.dashboard_enable:
        print("  • Open Excel dashboard for market overview")
    if user_config.index_overview_enable:
        print("  • Check comprehensive index overview analysis")
    print("  • Check percentage movers reports for significant movements")
    print("  • Implement specific calculation modules in src/")
    print("  • Customize screeners and models as needed")
    
    print("\nFinished.")

if __name__ == "__main__":
    main()
