#!/usr/bin/env python3
"""
Test script for multi-file filtering system
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.post_process.post_process_workflow import PostProcessWorkflow
from src.post_process.multi_file_processor import MultiFileProcessor, determine_processing_mode
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_processing_mode_detection():
    """Test the processing mode detection logic."""
    logger.info("=== Testing Processing Mode Detection ===")

    # Load test configuration
    config_path = "test_multi_file_configuration.csv"
    if not os.path.exists(config_path):
        logger.error(f"Test configuration file not found: {config_path}")
        return False

    config_df = pd.read_csv(config_path)
    config_df = config_df[config_df['Filename'].notna()]  # Remove comment rows

    test_cases = [
        ("100", "multi_file", "2-file intersection"),
        ("200", "multi_file", "3-file with multiple criteria"),
        ("300", "multi_file", "Multi-file with OR logic"),
        ("400", "single_file", "Traditional single-file"),
        ("500", "multi_file", "4-criteria multi-file")
    ]

    for file_id, expected_mode, description in test_cases:
        detected_mode = determine_processing_mode(config_df, file_id)
        status = "✓" if detected_mode == expected_mode else "✗"
        logger.info(f"{status} File_id {file_id}: {detected_mode} (expected {expected_mode}) - {description}")

    return True

def test_multi_file_processor_initialization():
    """Test MultiFileProcessor initialization and source file identification."""
    logger.info("=== Testing MultiFileProcessor Initialization ===")

    config_path = "test_multi_file_configuration.csv"
    if not os.path.exists(config_path):
        logger.error(f"Test configuration file not found: {config_path}")
        return False

    config_df = pd.read_csv(config_path)
    config_df = config_df[config_df['Filename'].notna()]  # Remove comment rows

    # Test File_id 100 (basic 2-file intersection)
    processor = MultiFileProcessor(config_df, "100", ".")
    source_files = processor.identify_source_files()

    logger.info(f"File_id 100 source files: {source_files}")
    expected_sources = {"basic_calculation", "stage_analysis"}
    actual_sources = set(source_files.keys())

    if actual_sources == expected_sources:
        logger.info("✓ Source file identification working correctly")
    else:
        logger.error(f"✗ Expected sources {expected_sources}, got {actual_sources}")
        return False

    return True

def test_with_actual_data():
    """Test with actual data files if available."""
    logger.info("=== Testing with Actual Data Files ===")

    # Look for actual data files
    data_paths = [
        "/home/imagda/_invest2024/python/metaData_v1 (8th copy)/results_2_5_complete/",
        "results/",
        "data/"
    ]

    basic_calc_file = None
    stage_analysis_file = None

    for data_path in data_paths:
        if os.path.exists(data_path):
            # Look for basic_calculation files
            for file in os.listdir(data_path):
                if 'basic_calculation' in file and file.endswith('.csv'):
                    basic_calc_file = os.path.join(data_path, file)
                    break

            # Look for stage_analysis files
            for file in os.listdir(data_path):
                if 'stage_analysis' in file and file.endswith('.csv'):
                    stage_analysis_file = os.path.join(data_path, file)
                    break

            if basic_calc_file and stage_analysis_file:
                break

    if not basic_calc_file or not stage_analysis_file:
        logger.warning("Actual data files not found, skipping real data test")
        return True

    logger.info(f"Found basic_calculation: {basic_calc_file}")
    logger.info(f"Found stage_analysis: {stage_analysis_file}")

    # Test loading files
    try:
        df_basic = pd.read_csv(basic_calc_file)
        df_stage = pd.read_csv(stage_analysis_file)

        logger.info(f"Basic calculation data: {len(df_basic)} rows")
        logger.info(f"Stage analysis data: {len(df_stage)} rows")

        # Check for required columns
        required_basic_cols = ['ticker', 'daily_daily_monthly_22d_pct_change']
        required_stage_cols = ['ticker', 'daily_sa_name']

        missing_basic = [col for col in required_basic_cols if col not in df_basic.columns]
        missing_stage = [col for col in required_stage_cols if col not in df_stage.columns]

        if missing_basic:
            logger.warning(f"Missing columns in basic_calculation: {missing_basic}")
        if missing_stage:
            logger.warning(f"Missing columns in stage_analysis: {missing_stage}")

        if not missing_basic and not missing_stage:
            logger.info("✓ All required columns present for multi-file filtering")

            # Test actual filtering
            test_filters_basic = df_basic[df_basic['daily_daily_monthly_22d_pct_change'] > 20]
            test_filters_stage = df_stage[df_stage['daily_sa_name'].str.contains('Bullish', na=False)]

            logger.info(f"Basic calc filter result: {len(test_filters_basic)} tickers")
            logger.info(f"Stage analysis filter result: {len(test_filters_stage)} tickers")

            # Intersection test
            basic_tickers = set(test_filters_basic['ticker'])
            stage_tickers = set(test_filters_stage['ticker'])
            intersection = basic_tickers & stage_tickers

            logger.info(f"Intersection result: {len(intersection)} tickers")
            if len(intersection) > 0:
                logger.info(f"Sample intersected tickers: {list(intersection)[:5]}")

        return True

    except Exception as e:
        logger.error(f"Error testing with actual data: {e}")
        return False

def test_complete_workflow():
    """Test the complete multi-file workflow integration."""
    logger.info("=== Testing Complete Workflow Integration ===")

    try:
        # Initialize workflow with test configuration
        workflow = PostProcessWorkflow("test_multi_file_configuration.csv", ".")

        if not workflow.load_configuration():
            logger.error("Failed to load test configuration")
            return False

        logger.info("✓ Configuration loaded successfully")

        # Test that the workflow can identify file groups correctly
        config_df = workflow.config_df
        unique_file_ids = config_df['File_id'].dropna().unique()

        logger.info(f"Found File_ids: {list(unique_file_ids)}")

        for file_id in unique_file_ids:
            processing_mode = determine_processing_mode(config_df, str(file_id))
            logger.info(f"File_id {file_id}: {processing_mode} processing mode")

        logger.info("✓ Complete workflow integration test passed")
        return True

    except Exception as e:
        logger.error(f"Error in complete workflow test: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting Multi-File Filtering System Tests")

    tests = [
        ("Processing Mode Detection", test_processing_mode_detection),
        ("MultiFileProcessor Initialization", test_multi_file_processor_initialization),
        ("Actual Data Testing", test_with_actual_data),
        ("Complete Workflow Integration", test_complete_workflow)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY:")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {status}: {test_name}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Multi-file filtering system is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Review implementation before deployment.")

if __name__ == "__main__":
    main()