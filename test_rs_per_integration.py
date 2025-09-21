#!/usr/bin/env python3
"""
Test script for RS/PER template integration
===========================================

Tests the complete RS/PER analysis and PDF generation system.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.post_process.rs_per_processor import validate_rs_per_requirements, run_rs_per_analysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_rs_per_system():
    """Test the complete RS/PER analysis system."""

    logger.info("="*60)
    logger.info("TESTING RS/PER ANALYSIS SYSTEM")
    logger.info("="*60)

    try:
        # Initialize configuration
        config = Config()
        logger.info(f"Configuration initialized")
        logger.info(f"Results directory: {config.directories['RESULTS_DIR']}")

        # Step 1: Validate requirements
        logger.info("\nStep 1: Validating RS/PER requirements...")
        validation = validate_rs_per_requirements(config)

        logger.info(f"Requirements met: {validation['requirements_met']}")
        logger.info(f"System ready: {validation['system_ready']}")

        if validation['missing_components']:
            logger.warning(f"Missing components: {validation['missing_components']}")

        for recommendation in validation['recommendations']:
            logger.info(f"Recommendation: {recommendation}")

        if not validation['system_ready']:
            logger.warning("System not ready for RS/PER analysis")
            logger.info("Please run RS and PER analysis first with BASIC=TRUE")
            return False

        # Step 2: Test RS/PER analysis
        logger.info("\nStep 2: Running RS/PER analysis...")
        results = run_rs_per_analysis(config=config)

        if results['success']:
            logger.info("✅ RS/PER analysis completed successfully!")

            # Display results summary
            processing_summary = results.get('processing_summary', {})
            analysis_summary = processing_summary.get('analysis_summary', {})

            logger.info(f"\nAnalysis Results:")
            logger.info(f"  Date: {results['date']}")
            logger.info(f"  Market Condition: {analysis_summary.get('market_condition', 'Unknown')}")
            logger.info(f"  Total Stocks: {analysis_summary.get('total_stocks', 0):,}")
            logger.info(f"  Market Breadth: {analysis_summary.get('market_breadth_pct', 0):.1f}%")
            logger.info(f"  Elite Performers: {analysis_summary.get('elite_performers', 0)}")

            # Display output files
            charts = results.get('charts', {})
            logger.info(f"\nGenerated Files:")
            logger.info(f"  PDF Report: {results.get('pdf_path')}")
            logger.info(f"  Charts Generated: {len([c for c in charts.values() if c])}/6")

            for chart_name, chart_path in charts.items():
                if chart_path:
                    logger.info(f"    ✅ {chart_name}: {Path(chart_path).name}")
                else:
                    logger.info(f"    ❌ {chart_name}: Failed")

            return True

        else:
            logger.error(f"❌ RS/PER analysis failed: {results.get('error')}")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False


def test_template_integration():
    """Test the PDF template integration."""

    logger.info("\n" + "="*60)
    logger.info("TESTING TEMPLATE INTEGRATION")
    logger.info("="*60)

    try:
        # Test template registration
        from src.post_process.pdf_templates import get_template

        logger.info("Testing template registration...")
        template_func = get_template('rs_per_template')

        if template_func:
            logger.info("✅ rs_per_template successfully registered")
            return True
        else:
            logger.error("❌ rs_per_template registration failed")
            return False

    except Exception as e:
        logger.error(f"❌ Template integration test failed: {e}")
        return False


def main():
    """Main test function."""

    logger.info("Starting RS/PER integration tests...")

    # Test 1: Template Integration
    template_test = test_template_integration()

    # Test 2: RS/PER System
    system_test = test_rs_per_system()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Template Integration: {'✅ PASSED' if template_test else '❌ FAILED'}")
    logger.info(f"RS/PER System: {'✅ PASSED' if system_test else '❌ FAILED'}")

    overall_success = template_test and system_test
    logger.info(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    if not overall_success:
        logger.info("\nTo fix issues:")
        logger.info("1. Ensure BASIC=TRUE in user_data.csv")
        logger.info("2. Run main.py to generate RS and PER data files")
        logger.info("3. Ensure POST_PROCESS=TRUE in user_data.csv")
        logger.info("4. Check that rs_per_analysis is enabled in user_data_pp.csv")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)