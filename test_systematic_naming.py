#!/usr/bin/env python3
"""
Systematic Naming Implementation Test
====================================

Comprehensive test script to validate the systematic naming implementation
across all components: multi-benchmark RS calculator, percentile calculator,
savers, and processor integration.

Tests:
1. Systematic column name generation
2. Multi-benchmark RS calculation with systematic naming
3. Percentile calculation with systematic naming
4. File saving with organized columns
5. End-to-end processor integration

Usage:
    python test_systematic_naming.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging
import traceback

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import the systematic naming components
from src.multi_benchmark_rs import MultiBenchmarkRSCalculator
from src.systematic_percentile_calculator import SystematicPercentileCalculator
from src.rs_values_saver import RSValuesSaver
from src.percentile_saver import PercentileSaver
from src.rs_per_processor import RSPerProcessor
from src.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockConfig:
    """Mock config for testing."""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.directories = {
            'RESULTS_DIR': self.base_dir / 'test_results',
            'MARKET_DATA_DIR': self.base_dir / 'data' / 'market_data'
        }
        # Create test directories
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)


class MockUserConfig:
    """Mock user config for testing."""
    
    def __init__(self):
        # Multi-benchmark configuration
        self.rs_benchmark_tickers = "SPY;QQQ"
        self.rs_benchmark_spy_enable = True
        self.rs_benchmark_qqq_enable = True
        self.rs_benchmark_ticker = "SPY"  # Legacy single benchmark
        
        # Analysis configuration
        self.rs_enable_stocks = True
        self.rs_enable_sectors = True
        self.rs_enable_industries = True
        self.rs_composite_method = "market_cap_weighted"
        
        # Timeframe configuration
        self.load_daily_data = True
        self.load_weekly_data = False
        self.load_monthly_data = False
        self.rs_daily_enable = True
        
        # Period configuration
        self.daily_daily_periods = "1;3;5"
        self.daily_weekly_periods = "7;14"
        self.daily_monthly_periods = "22;44"
        self.daily_quarterly_periods = "66;132"
        self.daily_yearly_periods = "252"


def create_test_data():
    """Create test price data and benchmarks."""
    
    logger.info("Creating test data...")
    
    # Create date range (ensure timezone-aware for benchmark compatibility)
    dates = pd.date_range(
        start='2024-01-01', 
        end='2024-12-31', 
        freq='B',  # Business days only
        tz='America/New_York'
    )
    
    # Create test stock tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Generate mock price data (random walk with trend)
    np.random.seed(42)  # Reproducible results
    
    price_data = {}
    for ticker in tickers:
        # Start at $100, add daily returns
        returns = np.random.normal(0.001, 0.02, len(dates))  # Slight upward bias
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[ticker] = pd.Series(prices, index=dates)
    
    price_df = pd.DataFrame(price_data)
    
    # Create benchmark data (SPY and QQQ)
    spy_returns = np.random.normal(0.0008, 0.015, len(dates))
    spy_prices = 400 * np.exp(np.cumsum(spy_returns))
    spy_data = pd.Series(spy_prices, index=dates)
    
    qqq_returns = np.random.normal(0.0009, 0.018, len(dates))
    qqq_prices = 350 * np.exp(np.cumsum(qqq_returns))
    qqq_data = pd.Series(qqq_prices, index=dates)
    
    logger.info(f"Created price data: {len(price_df)} dates, {len(tickers)} tickers")
    logger.info(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    return price_df, {'SPY': spy_data, 'QQQ': qqq_data}


def test_systematic_column_naming():
    """Test systematic column name generation."""
    
    logger.info("=" * 60)
    logger.info("Testing Systematic Column Name Generation")
    logger.info("=" * 60)
    
    config = MockConfig()
    user_config = MockUserConfig()
    calculator = MultiBenchmarkRSCalculator(config, user_config)
    
    test_cases = [
        (5, 'SPY', 'daily', 'daily_daily_daily_5d_rs_vs_SPY'),
        (22, 'QQQ', 'daily', 'daily_daily_monthly_22d_rs_vs_QQQ'),
        (252, 'SPY', 'daily', 'daily_daily_yearly_252d_rs_vs_SPY'),
        (14, 'QQQ', 'weekly', 'weekly_weekly_monthly_14d_rs_vs_QQQ'),
    ]
    
    success_count = 0
    for period, benchmark, timeframe, expected in test_cases:
        actual = calculator._generate_rs_column_name(period, benchmark, timeframe)
        if actual == expected:
            logger.info(f"✓ {period}d {benchmark} {timeframe}: {actual}")
            success_count += 1
        else:
            logger.error(f"✗ {period}d {benchmark} {timeframe}: Expected {expected}, got {actual}")
    
    logger.info(f"Column naming tests: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)


def test_multi_benchmark_rs_calculation():
    """Test multi-benchmark RS calculation with systematic naming."""
    
    logger.info("=" * 60)
    logger.info("Testing Multi-Benchmark RS Calculation")
    logger.info("=" * 60)
    
    config = MockConfig()
    user_config = MockUserConfig()
    calculator = MultiBenchmarkRSCalculator(config, user_config)
    
    # Create test data
    price_df, benchmark_data = create_test_data()
    
    # Mock benchmark data loading
    def mock_load_benchmark_data(ticker, timeframe):
        return benchmark_data.get(ticker)
    
    calculator.load_benchmark_data = mock_load_benchmark_data
    
    # Test RS calculation
    periods = [1, 3, 5, 22]
    
    try:
        benchmark_results = calculator.calculate_multi_benchmark_rs(
            price_df, periods, timeframe='daily'
        )
        
        if not benchmark_results:
            logger.error("No benchmark results generated")
            return False
        
        logger.info(f"Generated results for {len(benchmark_results)} benchmarks")
        
        # Validate results structure
        for benchmark, benchmark_data in benchmark_results.items():
            logger.info(f"Benchmark {benchmark}:")
            for period, period_df in benchmark_data.items():
                if not period_df.empty:
                    # Check for systematic column names
                    rs_cols = [col for col in period_df.columns if '_rs_vs_' in col]
                    if rs_cols:
                        logger.info(f"  {period}d: {len(period_df)} tickers, RS columns: {rs_cols}")
                        
                        # Validate column name format
                        for col in rs_cols:
                            parts = col.split('_')
                            if len(parts) >= 6 and parts[4] == 'RS' and parts[5] == 'vs':
                                logger.info(f"    ✓ Systematic naming: {col}")
                            else:
                                logger.error(f"    ✗ Invalid naming: {col}")
                                return False
                    else:
                        logger.warning(f"  {period}d: No RS columns found")
                else:
                    logger.warning(f"  {period}d: Empty DataFrame")
        
        return True
        
    except Exception as e:
        logger.error(f"Multi-benchmark RS calculation failed: {e}")
        traceback.print_exc()
        return False


def test_systematic_percentile_calculation():
    """Test systematic percentile calculation."""
    
    logger.info("=" * 60)
    logger.info("Testing Systematic Percentile Calculation")
    logger.info("=" * 60)
    
    config = MockConfig()
    user_config = MockUserConfig()
    
    # First calculate RS values
    rs_calculator = MultiBenchmarkRSCalculator(config, user_config)
    percentile_calculator = SystematicPercentileCalculator(config, user_config)
    
    # Create test data
    price_df, benchmark_data = create_test_data()
    
    # Mock benchmark data loading
    rs_calculator.load_benchmark_data = lambda ticker, timeframe: benchmark_data.get(ticker)
    
    # Calculate RS
    periods = [1, 3, 5]
    benchmark_results = rs_calculator.calculate_multi_benchmark_rs(
        price_df, periods, timeframe='daily'
    )
    
    if not benchmark_results:
        logger.error("No RS results to test percentiles")
        return False
    
    try:
        # Calculate percentiles
        percentile_results = percentile_calculator.calculate_percentiles(
            benchmark_results, method='ibd'
        )
        
        if not percentile_results:
            logger.error("No percentile results generated")
            return False
        
        logger.info(f"Generated percentiles for {len(percentile_results)} benchmarks")
        
        # Validate percentile results
        for benchmark, benchmark_data in percentile_results.items():
            logger.info(f"Benchmark {benchmark}:")
            for period, period_df in benchmark_data.items():
                if not period_df.empty:
                    # Check for both RS and percentile columns
                    rs_cols = [col for col in period_df.columns if '_rs_vs_' in col]
                    per_cols = [col for col in period_df.columns if '_rs_per' in col]
                    
                    logger.info(f"  {period}d: {len(period_df)} tickers, RS: {len(rs_cols)}, Percentiles: {len(per_cols)}")
                    
                    # Validate percentile values (should be 1-99)
                    for per_col in per_cols:
                        per_values = period_df[per_col].dropna()
                        if len(per_values) > 0:
                            min_val, max_val = per_values.min(), per_values.max()
                            if 1 <= min_val <= max_val <= 99:
                                logger.info(f"    ✓ {per_col}: range {min_val}-{max_val}")
                            else:
                                logger.error(f"    ✗ {per_col}: invalid range {min_val}-{max_val}")
                                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Percentile calculation failed: {e}")
        traceback.print_exc()
        return False


def test_file_saving():
    """Test systematic file saving."""
    
    logger.info("=" * 60)
    logger.info("Testing Systematic File Saving")
    logger.info("=" * 60)
    
    config = MockConfig()
    user_config = MockUserConfig()
    
    # Calculate RS and percentiles first
    rs_calculator = MultiBenchmarkRSCalculator(config, user_config)
    percentile_calculator = SystematicPercentileCalculator(config, user_config)
    rs_saver = RSValuesSaver(config, user_config)
    percentile_saver = PercentileSaver(config, user_config)
    
    # Create test data
    price_df, benchmark_data = create_test_data()
    rs_calculator.load_benchmark_data = lambda ticker, timeframe: benchmark_data.get(ticker)
    
    # Calculate RS and percentiles
    periods = [1, 5, 22]
    benchmark_results = rs_calculator.calculate_multi_benchmark_rs(
        price_df, periods, timeframe='daily'
    )
    
    percentile_results = percentile_calculator.calculate_percentiles(
        benchmark_results, method='ibd'
    )
    
    try:
        # Save RS values
        rs_files = rs_saver.save_multi_benchmark_rs_results(
            benchmark_results, 'stocks', '0', 'daily'
        )
        
        # Save percentiles
        percentile_files = percentile_saver.save_multi_benchmark_percentiles(
            percentile_results, 'stocks', '0', 'daily'
        )
        
        logger.info(f"Saved {len(rs_files)} RS files and {len(percentile_files)} percentile files")
        
        # Validate saved files
        validation_passed = True
        
        for file_path in rs_files + percentile_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"✓ {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Check column organization
                    cols = df.columns.tolist()
                    if 'ticker' in cols and cols[0] == 'ticker':
                        logger.info(f"  ✓ Ticker column first")
                    else:
                        logger.error(f"  ✗ Ticker column not first: {cols[:3]}")
                        validation_passed = False
                    
                    # Check for systematic columns
                    systematic_cols = [col for col in cols if '_rs_vs_' in col or '_rs_per' in col]
                    if systematic_cols:
                        logger.info(f"  ✓ Systematic columns: {len(systematic_cols)}")
                    else:
                        logger.warning(f"  ? No systematic columns found")
                    
                except Exception as e:
                    logger.error(f"✗ Error reading {file_path.name}: {e}")
                    validation_passed = False
            else:
                logger.error(f"✗ File not found: {file_path}")
                validation_passed = False
        
        return validation_passed
        
    except Exception as e:
        logger.error(f"File saving failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_processor():
    """Test end-to-end processor integration."""
    
    logger.info("=" * 60)
    logger.info("Testing End-to-End Processor Integration")
    logger.info("=" * 60)
    
    config = MockConfig()
    user_config = MockUserConfig()
    processor = RSPerProcessor(config, user_config)
    
    # Mock data loading for processor
    def mock_load_price_data(ticker_list, timeframe):
        price_df, _ = create_test_data()
        return price_df[ticker_list[:3]]  # Use subset for testing
    
    def mock_load_benchmark_data(ticker, timeframe):
        _, benchmark_data = create_test_data()
        return benchmark_data.get(ticker)
    
    # Apply mocks
    processor.multi_benchmark_calculator._load_price_data = mock_load_price_data
    processor.multi_benchmark_calculator.load_benchmark_data = mock_load_benchmark_data
    
    try:
        # Run full analysis
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        results = processor.process_rs_analysis(test_tickers, ticker_choice=0)
        
        logger.info(f"Processor results: {results}")
        
        # Validate results
        if results['files_created']:
            logger.info(f"✓ Created {len(results['files_created'])} files")
            
            # Get file summaries
            summaries = processor.get_file_summaries(ticker_choice=0)
            logger.info(f"File summaries: RS files: {summaries['rs_values']}")
            logger.info(f"File summaries: Percentile files: {summaries['percentiles']}")
            
            # Validate systematic naming
            validation = processor.validate_systematic_naming(ticker_choice=0)
            if validation['overall_status'] == 'valid':
                logger.info("✓ Systematic naming validation passed")
                return True
            else:
                logger.error(f"✗ Systematic naming validation failed: {validation['issues_found']}")
                return False
        else:
            logger.error("✗ No files created by processor")
            return False
        
    except Exception as e:
        logger.error(f"End-to-end processor test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all systematic naming tests."""
    
    logger.info("Starting Systematic Naming Implementation Tests...")
    
    test_results = {
        'column_naming': test_systematic_column_naming(),
        'multi_benchmark_rs': test_multi_benchmark_rs_calculation(),
        'percentile_calculation': test_systematic_percentile_calculation(),
        'file_saving': test_file_saving(),
        'end_to_end_processor': test_end_to_end_processor()
    }
    
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:25s}: {status}")
        if result:
            passed_tests += 1
    
    logger.info("=" * 60)
    logger.info(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All systematic naming tests PASSED!")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)