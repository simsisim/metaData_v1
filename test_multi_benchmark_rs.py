"""
Multi-Benchmark RS Testing Script
=================================

Test script to validate the new multi-benchmark RS calculation system.
Tests configuration parsing, RS calculation, and file output.
"""

import sys
sys.path.append('.')

from src.config import Config
from src.user_defined_data import read_user_data
from src.multi_benchmark_rs import MultiBenchmarkRSCalculator
from src.rs_values_saver import RSValuesSaver
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration parsing for multi-benchmark setup."""
    logger.info("Testing configuration parsing...")
    
    try:
        config = Config()
        user_config = read_user_data()
        
        logger.info("Configuration loaded successfully")
        logger.info(f"RS benchmark tickers: {getattr(user_config, 'rs_benchmark_tickers', 'Not set')}")
        logger.info(f"SPY enabled: {getattr(user_config, 'rs_benchmark_SPY_enable', 'Not set')}")
        logger.info(f"QQQ enabled: {getattr(user_config, 'rs_benchmark_QQQ_enable', 'Not set')}")
        logger.info(f"IWM enabled: {getattr(user_config, 'rs_benchmark_IWM_enable', 'Not set')}")
        
        # Test directory creation
        rs_values_dir = config.directories.get('RS_VALUES_DIR')
        logger.info(f"RS Values directory: {rs_values_dir}")
        
        return config, user_config
    
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return None, None

def test_calculator_initialization(config, user_config):
    """Test multi-benchmark calculator initialization."""
    logger.info("Testing calculator initialization...")
    
    try:
        calculator = MultiBenchmarkRSCalculator(config, user_config)
        
        logger.info(f"Calculator initialized with {len(calculator.enabled_benchmarks)} benchmarks")
        logger.info(f"Enabled benchmarks: {calculator.enabled_benchmarks}")
        
        return calculator
    
    except Exception as e:
        logger.error(f"Calculator initialization failed: {e}")
        return None

def test_benchmark_data_loading(calculator):
    """Test benchmark data loading."""
    logger.info("Testing benchmark data loading...")
    
    try:
        for benchmark in calculator.enabled_benchmarks:
            logger.info(f"Testing benchmark: {benchmark}")
            
            # Test daily data loading
            daily_data = calculator._load_benchmark_data_cached(benchmark, 'daily')
            if daily_data is not None:
                logger.info(f"{benchmark} daily data: {len(daily_data)} points, latest: {daily_data.index[-1] if len(daily_data) > 0 else 'No data'}")
            else:
                logger.warning(f"No daily data for {benchmark}")
        
        return True
    
    except Exception as e:
        logger.error(f"Benchmark data loading test failed: {e}")
        return False

def create_test_data():
    """Create minimal test price data."""
    logger.info("Creating test price data...")
    
    # Create simple test data with recent dates to overlap with real benchmark data
    # Use timezone-aware dates to match the benchmark data format
    dates = pd.date_range('2025-08-15', '2025-09-05', freq='D', tz='UTC-04:00')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Create price data with some variation
    test_data = {}
    for i, ticker in enumerate(tickers):
        # Create price series with different trends
        base_price = 100 + i * 10
        price_series = base_price + (pd.Series(range(len(dates))) * (0.5 + i * 0.2)) + pd.Series([(-1)**j * 2 for j in range(len(dates))])
        test_data[ticker] = price_series.values
    
    price_df = pd.DataFrame(test_data, index=dates)
    
    logger.info(f"Created test data: {len(price_df)} dates, {len(price_df.columns)} tickers")
    return price_df

def test_rs_calculation(calculator, test_data):
    """Test RS calculation with test data."""
    logger.info("Testing RS calculation...")
    
    try:
        periods = [5, 10, 20]
        timeframe = 'daily'
        
        # Test multi-benchmark calculation
        benchmark_results = calculator.calculate_multi_benchmark_rs(test_data, periods, timeframe)
        
        logger.info(f"RS calculation results:")
        for benchmark, results in benchmark_results.items():
            logger.info(f"  {benchmark}: {len(results)} periods calculated")
            for period, df in results.items():
                logger.info(f"    {period}d: {len(df)} tickers, columns: {list(df.columns)}")
        
        return benchmark_results
    
    except Exception as e:
        logger.error(f"RS calculation test failed: {e}")
        return None

def test_file_saving(config, user_config, benchmark_results):
    """Test RS file saving."""
    logger.info("Testing file saving...")
    
    try:
        saver = RSValuesSaver(config, user_config)
        
        saved_files = saver.save_multi_benchmark_rs_results(
            benchmark_results, 
            level='stocks', 
            choice='test', 
            timeframe='daily'
        )
        
        logger.info(f"File saving completed: {len(saved_files)} files created")
        for file_path in saved_files:
            logger.info(f"  Created: {file_path.name}")
            
            # Read back and validate
            df = pd.read_csv(file_path)
            logger.info(f"    {len(df)} rows, {len(df.columns)} columns")
            rs_cols = [col for col in df.columns if col.startswith('rs_')]
            logger.info(f"    RS columns: {rs_cols}")
        
        return saved_files
    
    except Exception as e:
        logger.error(f"File saving test failed: {e}")
        return []

def run_full_test():
    """Run complete test suite."""
    logger.info("=" * 60)
    logger.info("MULTI-BENCHMARK RS SYSTEM TEST")
    logger.info("=" * 60)
    
    # Test 1: Configuration
    config, user_config = test_configuration()
    if not config or not user_config:
        logger.error("Configuration test failed - aborting")
        return False
    
    # Test 2: Calculator initialization
    calculator = test_calculator_initialization(config, user_config)
    if not calculator:
        logger.error("Calculator initialization failed - aborting")
        return False
    
    # Test 3: Benchmark data loading
    if not test_benchmark_data_loading(calculator):
        logger.warning("Benchmark data loading had issues - continuing with test data")
    
    # Test 4: Create test data
    test_data = create_test_data()
    
    # Test 5: RS calculation
    benchmark_results = test_rs_calculation(calculator, test_data)
    if not benchmark_results:
        logger.error("RS calculation failed - aborting")
        return False
    
    # Test 6: File saving
    saved_files = test_file_saving(config, user_config, benchmark_results)
    if not saved_files:
        logger.error("File saving failed")
        return False
    
    logger.info("=" * 60)
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
    logger.info(f"Files created in: {config.directories['RS_VALUES_DIR']}")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)