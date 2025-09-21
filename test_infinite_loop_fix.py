#!/usr/bin/env python3
"""
Test script to verify infinite loop fixes
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.post_process.post_process_workflow import PostProcessWorkflow

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_minimal_config():
    """Test with minimal configuration to ensure no infinite loops."""
    logger.info("=== Testing Minimal Configuration ===")

    try:
        # Test with timeout
        start_time = time.time()
        timeout = 30  # 30 seconds max

        logger.info("Initializing workflow with minimal config...")
        workflow = PostProcessWorkflow("minimal_test_config.csv", ".")

        if time.time() - start_time > timeout:
            logger.error("Initialization took too long - potential infinite loop")
            return False

        logger.info("Loading configuration...")
        if not workflow.load_configuration():
            logger.error("Failed to load configuration")
            return False

        if time.time() - start_time > timeout:
            logger.error("Configuration loading took too long - potential infinite loop")
            return False

        logger.info("Configuration loaded successfully")
        logger.info(f"Config validation result: {workflow.config_df is not None}")

        # Check if we have any file groups to process
        if 'File_id' in workflow.config_df.columns:
            file_groups = workflow.config_df.groupby(['Filename', 'File_id']).size().index.tolist()
            logger.info(f"Found file groups: {file_groups}")
        else:
            logger.info("No File_id column found")

        elapsed = time.time() - start_time
        logger.info(f"Test completed in {elapsed:.2f} seconds")

        if elapsed > timeout:
            logger.error("Test exceeded timeout - likely infinite loop")
            return False

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error in minimal config test after {elapsed:.2f}s: {e}")
        return False

def test_empty_dataframe_filters():
    """Test filter application on empty DataFrame."""
    logger.info("=== Testing Empty DataFrame Filters ===")

    try:
        import pandas as pd

        workflow = PostProcessWorkflow("minimal_test_config.csv", ".")

        # Create empty DataFrame
        empty_df = pd.DataFrame()
        filter_ops = pd.DataFrame({
            'Column': ['test_col'],
            'Condition': ['greater_than'],
            'Value': [10],
            'Logic': ['AND']
        })

        start_time = time.time()
        result = workflow.apply_filters(empty_df, filter_ops)
        elapsed = time.time() - start_time

        logger.info(f"Empty DataFrame filter test completed in {elapsed:.2f}s")
        logger.info(f"Result: {len(result)} rows")

        return elapsed < 5  # Should complete very quickly

    except Exception as e:
        logger.error(f"Error in empty DataFrame test: {e}")
        return False

def test_large_or_conditions():
    """Test with many OR conditions to stress-test the logic."""
    logger.info("=== Testing Large OR Conditions ===")

    try:
        import pandas as pd

        workflow = PostProcessWorkflow("minimal_test_config.csv", ".")

        # Create test DataFrame
        test_df = pd.DataFrame({
            'test_col': range(100),
            'ticker': [f'TEST{i}' for i in range(100)]
        })

        # Create many OR conditions
        or_conditions = []
        for i in range(10):  # 10 OR conditions
            or_conditions.append({
                'Column': 'test_col',
                'Condition': 'equals',
                'Value': i * 10,
                'Logic': 'OR'
            })

        # Add final AND to close the OR group
        or_conditions.append({
            'Column': 'test_col',
            'Condition': 'greater_than',
            'Value': -1,
            'Logic': 'AND'
        })

        filter_ops = pd.DataFrame(or_conditions)

        start_time = time.time()
        result = workflow.apply_filters(test_df, filter_ops)
        elapsed = time.time() - start_time

        logger.info(f"Large OR conditions test completed in {elapsed:.2f}s")
        logger.info(f"Result: {len(result)} rows from {len(test_df)} input rows")

        return elapsed < 10  # Should complete within 10 seconds

    except Exception as e:
        logger.error(f"Error in large OR conditions test: {e}")
        return False

def main():
    """Run all infinite loop prevention tests."""
    logger.info("Starting Infinite Loop Prevention Tests")

    tests = [
        ("Minimal Configuration", test_minimal_config),
        ("Empty DataFrame Filters", test_empty_dataframe_filters),
        ("Large OR Conditions", test_large_or_conditions)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")

        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time

            results.append((test_name, result, elapsed))
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status} ({elapsed:.2f}s)")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{test_name}: ERROR after {elapsed:.2f}s - {e}")
            results.append((test_name, False, elapsed))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("INFINITE LOOP PREVENTION TEST SUMMARY:")
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, elapsed in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {status}: {test_name} ({elapsed:.2f}s)")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All infinite loop prevention tests passed!")
        logger.info("The fixes should prevent the infinite loop issue.")
    else:
        logger.warning("⚠️  Some tests failed. Additional investigation needed.")

if __name__ == "__main__":
    main()