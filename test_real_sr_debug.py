#!/usr/bin/env python3
"""
Test script for real SR configuration with enhanced debug logging.
This tests the complete data processing pipeline to identify where EMA calculation fails.
"""

import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('real_sr_debug.log')
    ]
)

logger = logging.getLogger(__name__)

def create_realistic_test_data():
    """Create realistic QQQ price data for testing."""
    logger.info("🚀 Creating realistic QQQ test data")

    # Create 30 days of realistic QQQ data
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')

    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 350.0  # Realistic QQQ price
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns

    prices = []
    current_price = base_price
    for return_rate in returns:
        current_price = current_price * (1 + return_rate)
        prices.append(current_price)

    # Create OHLCV structure
    ohlcv_data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close + np.random.normal(0, close * 0.005)
        volume = int(np.random.uniform(50000000, 100000000))  # Realistic QQQ volume

        ohlcv_data.append({
            'Date': date,
            'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close,
            'Volume': volume
        })

    df = pd.DataFrame(ohlcv_data)
    df.set_index('Date', inplace=True)

    logger.info(f"✅ Created realistic data:")
    logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"   Data points: {len(df)}")
    logger.info(f"   Columns: {list(df.columns)}")
    logger.info(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")

    return df

def test_enhanced_debug_logging():
    """Test the complete SR pipeline with enhanced debug logging."""
    logger.info("=" * 80)
    logger.info("TESTING REAL SR CONFIGURATION WITH ENHANCED DEBUG LOGGING")
    logger.info("=" * 80)

    try:
        # Import SR modules
        from src.sustainability_ratios.sr_market_data import calculate_bundled_indicator
        from src.sustainability_ratios.enhanced_panel_parser import parse_enhanced_panel_entry

        # Create realistic test data
        qqq_data = create_realistic_test_data()

        # Test the complete bundled indicator pipeline
        logger.info(f"\n🎯 TESTING BUNDLED INDICATOR CALCULATION:")
        logger.info(f"   Testing: 'EMA(QQQ, 10)' with real data structure")

        # Create ticker data dict (mimics real SR system)
        ticker_data = {
            'QQQ': qqq_data
        }

        # Test the bundled indicator calculation
        indicator_str = "EMA(QQQ, 10)"
        base_ticker = "QQQ"

        logger.info(f"   Calling calculate_bundled_indicator:")
        logger.info(f"     indicator_str: '{indicator_str}'")
        logger.info(f"     base_ticker: '{base_ticker}'")
        logger.info(f"     ticker_data keys: {list(ticker_data.keys())}")

        # This should trigger all our debug logging
        result = calculate_bundled_indicator(indicator_str, ticker_data, base_ticker)

        # Analyze the result
        logger.info(f"\n📊 BUNDLED INDICATOR RESULT ANALYSIS:")
        if result is not None:
            logger.info(f"✅ SUCCESS: Bundled indicator calculation completed")
            logger.info(f"   Result type: {type(result)}")
            logger.info(f"   Result keys: {list(result.keys())}")

            for key, value in result.items():
                logger.info(f"   {key}:")
                logger.info(f"     Type: {type(value)}")
                logger.info(f"     Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                if hasattr(value, 'min'):
                    logger.info(f"     Range: {value.min():.4f} to {value.max():.4f}")

            # Test chart generation with this data
            logger.info(f"\n🎨 TESTING CHART GENERATION WITH RESULT:")
            test_chart_generation_with_result(result, qqq_data)

        else:
            logger.error(f"❌ FAILURE: Bundled indicator calculation returned None")
            logger.error(f"   This explains why user sees only QQQ line!")

    except Exception as e:
        import traceback
        logger.error(f"❌ EXCEPTION IN TEST:")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception message: {str(e)}")
        logger.error(f"   Full traceback: {traceback.format_exc()}")

def test_chart_generation_with_result(overlay_result, base_data):
    """Test chart generation with the overlay result."""
    logger.info(f"📊 CHART GENERATION TEST:")

    try:
        # Create panel data structure like real SR system
        panel_data = {
            'data_source': 'QQQ + EMA(QQQ, 10)',
            'indicator': 'EMA(QQQ, 10)',
            'is_bundled': True,
            'result': {
                'Close': base_data['Close'],
                'metadata': {
                    'chart_type': 'overlay',
                    'calculation_date': datetime.now().isoformat()
                }
            }
        }

        # Add overlay data if it exists
        if overlay_result:
            for key, value in overlay_result.items():
                panel_data['result'][key] = value
                logger.info(f"   Added overlay: '{key}' → panel result")

        logger.info(f"   Panel data keys: {list(panel_data.keys())}")
        logger.info(f"   Panel result keys: {list(panel_data['result'].keys())}")

        # Test the chart generation
        from src.sustainability_ratios.sr_dashboard_generator import create_multi_panel_chart

        panel_results = {
            'Test_Panel': panel_data
        }

        output_path = "test_real_sr_debug.png"
        logger.info(f"   Calling create_multi_panel_chart...")

        chart_path = create_multi_panel_chart(
            panel_results=panel_results,
            output_path=output_path,
            chart_title="Real SR Debug Test"
        )

        if chart_path:
            logger.info(f"✅ CHART GENERATION SUCCESS: {chart_path}")
        else:
            logger.error(f"❌ CHART GENERATION FAILED")

    except Exception as e:
        import traceback
        logger.error(f"❌ CHART GENERATION EXCEPTION:")
        logger.error(f"   Exception: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("STARTING REAL SR DEBUG TEST")

    test_enhanced_debug_logging()

    logger.info("=" * 80)
    logger.info("REAL SR DEBUG TEST COMPLETED")
    logger.info("=" * 80)

    print("\n" + "="*60)
    print("REAL SR DEBUG TEST COMPLETED")
    print("Check 'real_sr_debug.log' for comprehensive debug output")
    print("Check 'test_real_sr_debug.png' for generated chart")
    print("="*60)