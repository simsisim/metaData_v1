# Screener Implementation Guide

Complete step-by-step workflow for implementing new screeners in the metaData_v1 system.

Based on analysis of GUPPY and PVB TW streaming patterns.

## Overview

This guide provides a standardized 8-phase approach for adding new screeners with:
- Memory-efficient streaming processing
- Hierarchical configuration flags
- DataReader integration
- Comprehensive memory cleanup
- User-visible feedback

## Phase 1: Configuration Setup

### 1.1 Add User Configuration Flags (`user_data.csv`)
```csv
SCREENER_NAME_enable,TRUE/FALSE
SCREENER_NAME_daily_enable,TRUE
SCREENER_NAME_weekly_enable,TRUE
SCREENER_NAME_monthly_enable,TRUE
SCREENER_NAME_param1,default_value
SCREENER_NAME_param2,default_value
```

### 1.2 Add Configuration Mapping (`user_defined_data.py`)
```python
CONFIG_MAPPING = {
    # Master and timeframe flags
    'SCREENER_NAME_enable': 'screener_name_enable',
    'SCREENER_NAME_daily_enable': 'screener_name_daily_enable',
    'SCREENER_NAME_weekly_enable': 'screener_name_weekly_enable',
    'SCREENER_NAME_monthly_enable': 'screener_name_monthly_enable',

    # Screener parameters
    'SCREENER_NAME_param1': 'screener_name_param1',
    'SCREENER_NAME_param2': 'screener_name_param2',
}
```

### 1.3 Add UserConfiguration Fields (`user_defined_data.py`)
```python
class UserConfiguration:
    def __init__(self):
        # Master and timeframe flags
        self.screener_name_enable = False
        self.screener_name_daily_enable = True
        self.screener_name_weekly_enable = True
        self.screener_name_monthly_enable = True

        # Screener parameters
        self.screener_name_param1 = default_value
        self.screener_name_param2 = default_value
```

### 1.4 Add Parameter Helper Function (if needed)
```python
def get_screener_name_params_for_timeframe(user_config: UserConfiguration, timeframe: str) -> Optional[Dict[str, Any]]:
    """Get screener parameters with hierarchical flag checking."""

    # Check master flag
    if not getattr(user_config, "screener_name_enable", False):
        return None

    # Check timeframe flag
    timeframe_flag = f"screener_name_{timeframe}_enable"
    if not getattr(user_config, timeframe_flag, True):
        return None

    return {
        'enable_screener_name': True,
        'screener_name': {
            'param1': getattr(user_config, 'screener_name_param1', default_value),
            'param2': getattr(user_config, 'screener_name_param2', default_value),
        }
    }
```

## Phase 2: Core Screener Class

### 2.1 Create Screener File (`src/screeners/screener_name.py`)
```python
"""
Screener Name Implementation
===========================

Description of the screener strategy and methodology.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ScreenerNameScreener:
    """
    Screener Name implementation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_screener_name = config.get('enable_screener_name', True)
        self.timeframe = config.get('timeframe', 'daily')

        # Screener specific configuration
        self.screener_config = config.get('screener_name', {})

        # Parameters from user configuration
        self.param1 = self.screener_config.get('param1', default_value)
        self.param2 = self.screener_config.get('param2', default_value)

    def run_screener_name_screening(self, batch_data: Dict[str, pd.DataFrame],
                                   ticker_info: Optional[pd.DataFrame] = None,
                                   rs_data: Optional[Dict] = None,
                                   batch_info: Dict[str, Any] = None) -> List[Dict]:
        """
        Run Screener Name screening

        Args:
            batch_data: Dictionary of {ticker: DataFrame} with OHLCV data
            ticker_info: DataFrame with ticker information (optional)
            rs_data: Relative strength data (optional)
            batch_info: Additional batch information (optional)

        Returns:
            List of screening results
        """
        if not self.enable_screener_name:
            return []

        results = []

        for ticker, df in batch_data.items():
            try:
                # Implement screening logic here
                ticker_results = self._screen_single_ticker(ticker, df)
                if ticker_results:
                    results.extend(ticker_results)

            except Exception as e:
                logger.warning(f"Error screening {ticker}: {e}")
                continue

        return results

    def _screen_single_ticker(self, ticker: str, df: pd.DataFrame) -> List[Dict]:
        """Screen single ticker and return results."""
        # Implement specific screening logic
        # Return list of result dictionaries
        pass


def run_screener_name_screener(batch_data: Dict[str, pd.DataFrame],
                              screener_params: Dict[str, Any],
                              ticker_info: Optional[pd.DataFrame] = None) -> List[Dict]:
    """
    Standalone function to run screener name screening.
    """
    config = {
        'enable_screener_name': True,
        'screener_name': screener_params
    }

    screener = ScreenerNameScreener(config)
    return screener.run_screener_name_screening(batch_data, ticker_info)
```

## Phase 3: Streaming Processor

### 3.1 Add to `src/screeners_streaming.py`
```python
class ScreenerNameStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for Screener Name following PVB TW pattern.
    Memory-efficient batch processing with immediate writes.
    """

    def __init__(self, config, user_config):
        """
        Initialize screener streaming processor.
        """
        super().__init__(config, user_config)

        # Create output directory
        self.screener_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'screener_name'
        self.screener_dir.mkdir(parents=True, exist_ok=True)

        # Initialize screener instance with configuration
        screener_config = {
            'timeframe': 'daily',  # Will be overridden per timeframe
            'screener_name': {
                'param1': getattr(user_config, 'screener_name_param1', default_value),
                'param2': getattr(user_config, 'screener_name_param2', default_value),
            }
        }
        self.screener = ScreenerNameScreener(screener_config)

        logger.info(f"Screener Name streaming processor initialized, output dir: {self.screener_dir}")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming."""
        return "screener_name"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation."""
        return self.screener_dir

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str) -> Dict[str, Any]:
        """
        Process screener batch using memory-efficient streaming pattern.
        Following PVB TW streaming pattern with immediate writes and memory cleanup.
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} screener streaming")
            return {}

        logger.debug(f"Processing screener batch for {timeframe}: {len(batch_data)} tickers")

        # Get screener parameters for this timeframe
        try:
            screener_params = get_screener_name_params_for_timeframe(self.user_config, timeframe)
            if not screener_params or not screener_params.get('enable_screener_name'):
                logger.debug(f"Screener disabled for {timeframe}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get screener parameters for {timeframe}: {e}")
            return {}

        # Initialize result containers
        all_results = []
        component_results = {'signal_type1': [], 'signal_type2': []}  # Adjust as needed
        current_date = self.extract_date_from_batch_data(batch_data)
        processed_tickers = 0

        try:
            # Run screener for entire batch
            batch_results = self.screener.run_screener_name_screening(batch_data)

            if batch_results:
                all_results.extend(batch_results)

                # Sort results by signal type for individual files (if needed)
                for result in batch_results:
                    signal_type = result.get('signal_type', 'unknown')
                    if signal_type in component_results:
                        component_results[signal_type].append(result)

                processed_tickers = len(batch_results)

            # Memory cleanup after batch processing
            gc.collect()

            # Write consolidated results immediately
            output_files = []
            if all_results:
                consolidated_filename = f"screener_name_{timeframe}_{current_date}.csv"
                consolidated_file = self.screener_dir / consolidated_filename
                self._write_results_to_csv(consolidated_file, all_results)
                output_files.append(str(consolidated_file))
                logger.info(f"Screener consolidated: {len(all_results)} results saved to {consolidated_file}")

            # Write individual component files if needed
            for component_name, component_data in component_results.items():
                if component_data:
                    component_filename = f"screener_{component_name}_{timeframe}_{current_date}.csv"
                    component_file = self.screener_dir / component_filename
                    self._write_results_to_csv(component_file, component_data)
                    output_files.append(str(component_file))

            # Memory cleanup
            self.cleanup_memory(all_results, component_results, batch_data)

        except Exception as e:
            logger.error(f"Error in screener batch processing: {e}")

        logger.info(f"Screener batch summary ({timeframe}): {processed_tickers} tickers processed, "
                   f"{len(all_results)} total results")

        return {
            "tickers_processed": processed_tickers,
            "total_results": len(all_results),
            "output_files": output_files
        }
```

## Phase 4: Main Runner Function

### 4.1 Add Runner Function to `src/screeners_streaming.py`
```python
def run_all_screener_name_streaming(config, user_config, timeframes: List[str], clean_file_path: str) -> Dict[str, int]:
    """
    Run Screener Name using streaming processing with hierarchical flag validation.
    Following PVB TW pattern with DataReader for data loading.
    """
    # Check master flag first
    if not getattr(user_config, "screener_name_enable", False):
        print(f"\n⏭️  Screener Name disabled - skipping processing")
        logger.info("Screener Name disabled (master flag)")
        return {}

    # Check if any timeframe is enabled
    enabled_timeframes = []
    for timeframe in timeframes:
        if getattr(user_config, f"screener_name_{timeframe}_enable", True):
            enabled_timeframes.append(timeframe)

    if not enabled_timeframes:
        print(f"\n⚠️  Screener Name master enabled but all timeframes disabled - skipping processing")
        logger.warning("Screener Name master enabled but all timeframes disabled")
        return {}

    print(f"\n🔍 SCREENER NAME - Processing timeframes: {', '.join(enabled_timeframes)}")
    logger.info(f"Screener Name enabled for: {', '.join(enabled_timeframes)}")

    # Initialize processor
    processor = ScreenerNameStreamingProcessor(config, user_config)
    results = {}

    # Process each enabled timeframe using PVB TW pattern
    for timeframe in enabled_timeframes:
        screener_enabled = getattr(user_config, f'screener_name_{timeframe}_enable', True)
        if not screener_enabled:
            print(f"⏭️  Screener Name disabled for {timeframe} timeframe")
            continue

        print(f"\n📊 Processing Screener Name {timeframe.upper()} timeframe...")
        logger.info(f"Starting Screener Name for {timeframe} timeframe...")

        # Initialize DataReader for this timeframe (following PVB TW pattern)
        batch_size = getattr(user_config, 'batch_size', 100)
        from src.data_reader import DataReader
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers from file
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        import math
        total_batches = math.ceil(total_tickers / batch_size)

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        total_results = 0
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]

            logger.info(f"Processing Screener batch {batch_num + 1}/{total_batches}: tickers {start_idx + 1}-{end_idx}")

            # Get batch data using DataReader
            batch_data = data_reader.read_batch_data(batch_tickers)

            if batch_data:
                # Process batch using screener
                batch_result = processor.process_batch_streaming(batch_data, timeframe)
                if batch_result and "tickers_processed" in batch_result:
                    total_results += batch_result["tickers_processed"]
                    logger.info(f"Screener batch {batch_num + 1} completed: {batch_result['tickers_processed']} results")

        results[timeframe] = total_results
        logger.info(f"Screener Name completed for {timeframe}: {total_results} total results processed")

    if results:
        print(f"✅ Screener Name processing completed")
    else:
        print(f"⚠️  Screener Name completed with no results")

    return results
```

## Phase 5: Main.py Integration

### 5.1 Add Import
```python
from src.screeners_streaming import run_all_screener_name_streaming
```

### 5.2 Initialize Variables
```python
screener_name_results = {}
```

### 5.3 Add to SCREENERS Phase
```python
# ===============================
# SCREENER NAME
# ===============================
try:
    screener_name_results = run_all_screener_name_streaming(config, user_config, timeframes_to_process, clean_file)
    logger.info(f"Screener Name completed")
except Exception as e:
    logger.error(f"Error running Screener Name: {e}")
    screener_name_results = {}
```

### 5.4 Add to Results Summary
```python
screener_name_count = sum(screener_name_results.values()) if screener_name_results else 0

print(f"\n" + "="*60)
print("📊 CALCULATION PHASES SUMMARY")
print("="*60)
# ... other results ...
print(f"🔍 Screener Name Results: {screener_name_count}")
```

## Phase 6: Memory Management Best Practices

### 6.1 Implement Comprehensive Memory Cleanup
- Use `self.cleanup_memory(results, batch_data, other_objects)` after each batch
- Add `gc.collect()` after heavy operations
- Write results immediately, don't accumulate in memory
- Delete large objects explicitly before garbage collection

### 6.2 Memory Cleanup Pattern
```python
try:
    # Process batch
    batch_results = self.screener.run_screening(batch_data)

    # Write results immediately
    if batch_results:
        self._write_results_to_csv(output_file, batch_results)

    # Memory cleanup after batch processing
    gc.collect()

    # Explicit memory cleanup
    self.cleanup_memory(batch_results, batch_data, other_objects)

except Exception as e:
    logger.error(f"Error in batch processing: {e}")
```

## Phase 7: Testing and Validation

### 7.1 Test Configuration
- Verify flags work correctly in `user_data.csv`
- Test hierarchical flag logic (master → timeframe)
- Validate parameter parsing

### 7.2 Test Data Processing
- Run with small batch sizes to verify memory management
- Check output file creation and formatting
- Verify progress feedback messages

### 7.3 Test Integration
- Ensure screener runs in main pipeline
- Verify results appear in summary
- Check no conflicts with other screeners

## Phase 8: Documentation

### 8.1 Update CLAUDE.md
Add screener description and any specific commands or parameters.

### 8.2 Update Code Comments
Ensure all methods have proper docstrings and implementation notes.

## Key Pattern Differences

### PVB TW Pattern (Simpler)
- Direct screener method calls
- Single consolidated output file
- Simple batch processing loop
- Minimal result categorization

### GUPPY Pattern (Complex)
- Component result separation
- Multiple output files per signal type
- Advanced result categorization
- More sophisticated parameter handling

### Memory Management (Both Patterns)
- Use DataReader for data loading
- Process in configurable batch sizes
- Immediate file writes after each batch
- Comprehensive memory cleanup with base class methods
- Explicit garbage collection

## Common Issues and Solutions

1. **Method Name Errors**: Verify screener class method names match streaming processor calls
2. **DataReader Methods**: Use `read_batch_data()`, not `get_batch_data()`
3. **Memory Issues**: Always implement cleanup methods and garbage collection
4. **Configuration Errors**: Ensure all user config fields are properly mapped
5. **Import Errors**: Add proper imports to main.py and streaming modules

## Template Files

This guide provides complete templates that can be copied and adapted for any new screener implementation. Follow the phases in order to ensure proper integration with the existing system architecture.