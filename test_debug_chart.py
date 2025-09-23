#!/usr/bin/env python3
"""
Test script for debug-enhanced chart generation.
This script tests the enhanced SR dashboard generator with comprehensive logging.
"""

import logging
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_chart_test.log')
    ]
)

logger = logging.getLogger(__name__)

# Import the dashboard generator
from src.sustainability_ratios.sr_dashboard_generator import create_multi_panel_chart

def create_test_data():
    """Create test data that mimics the bundled format structure."""
    logger.info("🚀 Creating test data for bundled format: 'QQQ + EMA(QQQ, 10)'")

    # Create date range
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')

    # Create QQQ price data (base ticker)
    np.random.seed(42)  # For reproducible results
    base_price = 100.0
    price_changes = np.random.normal(0, 0.5, len(dates))
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] + change)

    # Create EMA data
    ema_values = []
    alpha = 2 / (10 + 1)  # EMA smoothing factor for period 10
    ema = prices[0]  # Initialize with first price
    ema_values.append(ema)

    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
        ema_values.append(ema)

    # Create the result structure that mimics what sr_market_data produces
    result = {
        'Close': pd.Series(prices, index=dates, name='Close'),
        'EMA_ema': pd.Series(ema_values, index=dates, name='EMA_ema'),
        'EMA_price': pd.Series(prices, index=dates, name='EMA_price'),  # Base price for EMA
        'EMA_signals': pd.Series([0] * len(dates), index=dates, name='EMA_signals'),  # Signal data
        'metadata': {
            'chart_type': 'overlay',
            'base_ticker': 'QQQ',
            'indicator': 'EMA(QQQ, 10)',
            'calculation_date': datetime.now().isoformat()
        }
    }

    logger.info(f"✅ Test data created:")
    logger.info(f"   Date range: {dates[0]} to {dates[-1]}")
    logger.info(f"   Data keys: {list(result.keys())}")
    logger.info(f"   Close price range: {min(prices):.2f} to {max(prices):.2f}")
    logger.info(f"   EMA range: {min(ema_values):.2f} to {max(ema_values):.2f}")

    return result

def test_debug_chart_generation():
    """Test the debug-enhanced chart generation."""
    logger.info("🧪 STARTING DEBUG CHART GENERATION TEST")

    try:
        # Create test data
        test_result = create_test_data()

        # Create panel results structure
        panel_results = {
            'Panel_1': {
                'data_source': 'QQQ + EMA(QQQ, 10)',
                'indicator': 'EMA(QQQ, 10)',
                'result': test_result,
                'is_bundled': True,
                'metadata': {
                    'chart_type': 'overlay',
                    'calculation_time': datetime.now().isoformat()
                }
            }
        }

        logger.info(f"📊 Panel results structure:")
        logger.info(f"   Panels: {list(panel_results.keys())}")
        for panel_name, panel_data in panel_results.items():
            logger.info(f"   {panel_name}:")
            logger.info(f"     data_source: {panel_data['data_source']}")
            logger.info(f"     indicator: {panel_data['indicator']}")
            logger.info(f"     is_bundled: {panel_data['is_bundled']}")
            logger.info(f"     result keys: {list(panel_data['result'].keys())}")

        # Test chart generation
        output_path = "test_debug_enhanced_chart.png"
        logger.info(f"🎯 CALLING create_multi_panel_chart with debug data")
        logger.info(f"   output_path: {output_path}")

        chart_path = create_multi_panel_chart(
            panel_results=panel_results,
            output_path=output_path,
            chart_title="Debug Test: QQQ + EMA(10)"
        )

        if chart_path:
            logger.info(f"✅ DEBUG CHART GENERATION SUCCESSFUL!")
            logger.info(f"   Chart saved to: {chart_path}")
            logger.info(f"   File exists: {Path(chart_path).exists()}")
            logger.info(f"   File size: {Path(chart_path).stat().st_size if Path(chart_path).exists() else 'N/A'} bytes")
        else:
            logger.error(f"❌ DEBUG CHART GENERATION FAILED - No output path returned")

    except Exception as e:
        import traceback
        logger.error(f"❌ DEBUG TEST FAILED WITH EXCEPTION:")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception message: {str(e)}")
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("DEBUG CHART GENERATION TEST STARTING")
    logger.info("=" * 80)

    test_debug_chart_generation()

    logger.info("=" * 80)
    logger.info("DEBUG CHART GENERATION TEST COMPLETED")
    logger.info("=" * 80)

    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("Check 'debug_chart_test.log' for detailed debug output")
    print("Check 'test_debug_enhanced_chart.png' for generated chart")
    print("="*60)