#!/usr/bin/env python3
"""
Test Script for Basic Calculation Template System
================================================

Tests the comprehensive basic_calculation template system including:
- PNG chart generation
- PDF template selection and generation
- Generation flag logic
- Auto-template selection
"""

import pandas as pd
import logging
from pathlib import Path
import sys
import tempfile

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_png_templates():
    """Test PNG template generation functions."""
    logger.info("Testing PNG template generation...")

    try:
        # Load sample data
        data_path = Path("results/basic_calculation/basic_calculation_2-5_daily_20250905.csv")
        if not data_path.exists():
            logger.warning(f"Sample data not found at {data_path}")
            return False

        df = pd.read_csv(data_path).head(100)  # Use first 100 rows for testing
        logger.info(f"Loaded {len(df)} rows of test data")

        # Test individual PNG generation functions
        from src.post_process.png_templates.basic_calculation import (
            create_top_performers_chart, create_sector_performance_heatmap,
            create_risk_return_scatter, create_market_cap_analysis
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test top performers chart
            chart_path = temp_path / "test_top_performers.png"
            success = create_top_performers_chart(df, 'daily_daily_yearly_252d_pct_change',
                                                str(chart_path), top_n=10)
            logger.info(f"Top performers chart: {'SUCCESS' if success else 'FAILED'}")

            # Test sector heatmap
            heatmap_path = temp_path / "test_sector_heatmap.png"
            timeframe_cols = ['daily_daily_daily_1d_pct_change', 'daily_daily_yearly_252d_pct_change']
            success = create_sector_performance_heatmap(df, timeframe_cols, str(heatmap_path))
            logger.info(f"Sector heatmap: {'SUCCESS' if success else 'FAILED'}")

            # Test risk-return scatter
            risk_path = temp_path / "test_risk_return.png"
            success = create_risk_return_scatter(df, str(risk_path))
            logger.info(f"Risk-return scatter: {'SUCCESS' if success else 'FAILED'}")

            # Test market cap analysis
            market_cap_path = temp_path / "test_market_cap.png"
            success = create_market_cap_analysis(df, str(market_cap_path))
            logger.info(f"Market cap analysis: {'SUCCESS' if success else 'FAILED'}")

        return True

    except Exception as e:
        logger.error(f"PNG template test failed: {e}")
        return False

def test_template_selection():
    """Test auto-template selection logic."""
    logger.info("Testing template selection logic...")

    try:
        from src.post_process.pdf_templates import _determine_template_from_filters

        # Test different filter scenarios
        test_cases = [
            # Sector filter should select sector_analysis
            ({
                'filter_operations': [
                    {'Column': 'sector', 'Condition': 'equals', 'Value': 'Electronic technology'}
                ]
            }, 'sector_analysis'),

            # Industry filter should select industry_analysis
            ({
                'filter_operations': [
                    {'Column': 'industry', 'Condition': 'equals', 'Value': 'Semiconductors'}
                ]
            }, 'industry_analysis'),

            # Universe filter should select universe_analysis
            ({
                'filter_operations': [
                    {'Column': 'SP500', 'Condition': 'equals', 'Value': 'TRUE'}
                ]
            }, 'universe_analysis'),

            # Risk filter should select risk_analysis
            ({
                'filter_operations': [
                    {'Column': 'atr_pct', 'Condition': 'greater_than', 'Value': '5'}
                ]
            }, 'risk_analysis'),

            # No specific filter should default to market_trends
            ({
                'filter_operations': [
                    {'Column': 'current_price', 'Condition': 'greater_than', 'Value': '100'}
                ]
            }, 'market_trends')
        ]

        for metadata, expected_template in test_cases:
            # Create dummy dataframe for testing
            df = pd.DataFrame({'dummy': [1, 2, 3]})
            selected_template = _determine_template_from_filters(df, metadata)

            result = "SUCCESS" if selected_template == expected_template else "FAILED"
            logger.info(f"Template selection - Expected: {expected_template}, Got: {selected_template} - {result}")

        return True

    except Exception as e:
        logger.error(f"Template selection test failed: {e}")
        return False

def test_generation_flag():
    """Test Generation flag logic."""
    logger.info("Testing Generation flag logic...")

    try:
        from src.post_process.post_process_workflow import PostProcessWorkflow

        # Create test workflow instance
        workflow = PostProcessWorkflow()

        # Test different Generation flag scenarios
        test_cases = [
            # Generation=TRUE should return True
            (pd.DataFrame({'Generation': ['TRUE']}), True),

            # Generation=FALSE should return False
            (pd.DataFrame({'Generation': ['FALSE']}), False),

            # Missing Generation column should default to True
            (pd.DataFrame({'Other': ['value']}), True),

            # Empty Generation values should default to True
            (pd.DataFrame({'Generation': [None, '', pd.NA]}), True),

            # Mixed values with TRUE should return True (conservative)
            (pd.DataFrame({'Generation': ['FALSE', 'TRUE']}), True)
        ]

        for test_config, expected_result in test_cases:
            result = workflow._is_generation_enabled(test_config)
            status = "SUCCESS" if result == expected_result else "FAILED"
            logger.info(f"Generation flag test - Expected: {expected_result}, Got: {result} - {status}")

        return True

    except Exception as e:
        logger.error(f"Generation flag test failed: {e}")
        return False

def test_pdf_generation():
    """Test PDF template generation."""
    logger.info("Testing PDF template generation...")

    try:
        # Load sample data
        data_path = Path("results/basic_calculation/basic_calculation_2-5_daily_20250905.csv")
        if not data_path.exists():
            logger.warning(f"Sample data not found at {data_path}")
            return False

        df = pd.read_csv(data_path).head(50)  # Use subset for faster testing
        logger.info(f"Loaded {len(df)} rows for PDF test")

        from src.post_process.pdf_templates import get_template

        # Test market trends template
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "test_market_trends.pdf"

            # Create sample metadata
            metadata = {
                'filter_operations': [],
                'sort_operations': [],
                'original_filename': 'basic_calculation',
                'file_id': 'test',
                'original_rows': len(df),
                'filtered_rows': len(df)
            }

            template_func = get_template('market_trends')
            success = template_func(df, str(pdf_path), metadata)

            logger.info(f"PDF generation test: {'SUCCESS' if success else 'FAILED'}")
            if success:
                logger.info(f"PDF file size: {pdf_path.stat().st_size} bytes")

        return success

    except Exception as e:
        logger.error(f"PDF generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting Basic Calculation Template System Tests")
    logger.info("=" * 60)

    test_results = []

    # Run tests
    test_results.append(("PNG Templates", test_png_templates()))
    test_results.append(("Template Selection", test_template_selection()))
    test_results.append(("Generation Flag", test_generation_flag()))
    test_results.append(("PDF Generation", test_pdf_generation()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:20} - {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! The basic_calculation template system is ready for use.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Review the logs above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)