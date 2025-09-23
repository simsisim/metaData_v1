#!/usr/bin/env python3
"""
EMA Overlay Diagnostic Test
==========================

This test reproduces the exact failure scenario identified in the research:
"only a red line, a red label that says QQQ" with no EMA label.

The test simulates different data scenarios to isolate the exact cause
of missing EMA overlays in the SR system.
"""

import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ema_diagnostic_test.log')
    ]
)

logger = logging.getLogger(__name__)

# Import the system modules
from src.sustainability_ratios.sr_dashboard_generator import create_multi_panel_chart, plot_overlay_chart
from src.sustainability_ratios.sr_market_data import calculate_bundled_indicator, decompose_data_source
from src.sustainability_ratios.enhanced_panel_parser import parse_enhanced_panel_entry


class EMAADiagnosticTest:
    """Comprehensive diagnostic test for EMA overlay failures."""

    def __init__(self):
        self.output_dir = Path("diagnostic_output")
        self.output_dir.mkdir(exist_ok=True)
        self.test_results = {}

    def create_base_data(self, ticker="QQQ", days=30):
        """Create realistic base ticker data."""
        logger.info(f"📊 Creating base data for {ticker} ({days} days)")

        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate realistic price data
        np.random.seed(42)  # Reproducible results
        base_price = 350.0  # Realistic QQQ price

        # Create OHLCV data
        prices = []
        current_price = base_price

        for i in range(len(dates)):
            # Daily price movement (± 1-3%)
            change_pct = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + change_pct)

            # OHLCV simulation
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = current_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = current_price
            volume = np.random.randint(50000000, 100000000)  # Realistic QQQ volume

            prices.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })

        df = pd.DataFrame(prices, index=dates)

        logger.info(f"✅ Base data created:")
        logger.info(f"   Ticker: {ticker}")
        logger.info(f"   Date range: {dates[0]} to {dates[-1]}")
        logger.info(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
        logger.info(f"   Data shape: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")

        return df

    def test_scenario_1_working_ema_data(self):
        """Test Scenario 1: Working EMA data with proper column naming."""
        logger.info("\n🧪 TEST SCENARIO 1: Working EMA Data")
        logger.info("=" * 60)

        try:
            # Create base data
            base_data = self.create_base_data("QQQ", 30)

            # Calculate EMA manually (proper working data)
            ema_period = 10
            ema_values = base_data['Close'].ewm(span=ema_period).mean()

            # Create WORKING data structure (what should happen)
            working_result = {
                'Close': base_data['Close'].copy(),  # Base ticker data
                'EMA_ema': ema_values.copy(),        # EMA overlay - CORRECT NAMING
                'metadata': {
                    'chart_type': 'overlay',
                    'base_ticker': 'QQQ',
                    'indicator': 'EMA(QQQ, 10)',
                    'calculation_date': datetime.now().isoformat(),
                    'stacking_order': 1
                }
            }

            # Test panel structure
            panel_results = {
                'Working_EMA_Panel': {
                    'data_source': 'QQQ + EMA(QQQ, 10)',
                    'indicator': 'EMA(QQQ, 10)',
                    'result': working_result,
                    'is_bundled': True,
                    'bundled_components': ['QQQ', 'EMA(QQQ, 10)']
                }
            }

            # Generate chart
            output_path = self.output_dir / "scenario_1_working_ema.png"
            chart_path = create_multi_panel_chart(
                panel_results=panel_results,
                output_path=str(output_path),
                chart_title="SCENARIO 1: Working EMA Overlay (Should show QQQ + EMA)"
            )

            # Log results
            logger.info(f"📊 SCENARIO 1 RESULTS:")
            logger.info(f"   Chart generated: {bool(chart_path)}")
            logger.info(f"   Output path: {output_path}")
            logger.info(f"   Expected: QQQ (blue) + EMA (red) lines with both labels")
            logger.info(f"   Data keys: {list(working_result.keys())}")
            logger.info(f"   EMA_ema column present: {'EMA_ema' in working_result}")

            # Data analysis
            close_data = working_result['Close']
            ema_data = working_result['EMA_ema']
            logger.info(f"   Close data points: {len(close_data)}")
            logger.info(f"   EMA data points: {len(ema_data)}")
            logger.info(f"   Close range: ${close_data.min():.2f} - ${close_data.max():.2f}")
            logger.info(f"   EMA range: ${ema_data.min():.2f} - ${ema_data.max():.2f}")

            self.test_results['scenario_1'] = {
                'status': 'success',
                'chart_path': chart_path,
                'data_keys': list(working_result.keys()),
                'has_ema_column': 'EMA_ema' in working_result,
                'description': 'Working case with proper EMA_ema column naming'
            }

        except Exception as e:
            logger.error(f"❌ SCENARIO 1 FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results['scenario_1'] = {'status': 'failed', 'error': str(e)}

    def test_scenario_2_missing_ema_data(self):
        """Test Scenario 2: Missing EMA data - reproduces user's issue."""
        logger.info("\n🧪 TEST SCENARIO 2: Missing EMA Data (User's Issue)")
        logger.info("=" * 60)

        try:
            # Create base data
            base_data = self.create_base_data("QQQ", 30)

            # Create FAILING data structure (what user experiences)
            failing_result = {
                'Close': base_data['Close'].copy(),  # Only base ticker data
                # NO EMA_ema COLUMN - this is the problem!
                'metadata': {
                    'chart_type': 'overlay',
                    'base_ticker': 'QQQ',
                    'indicator': 'EMA(QQQ, 10)',
                    'calculation_date': datetime.now().isoformat(),
                    'stacking_order': 1,
                    'error': 'EMA calculation failed - no EMA_ema column generated'
                }
            }

            # Test panel structure
            panel_results = {
                'Failing_EMA_Panel': {
                    'data_source': 'QQQ + EMA(QQQ, 10)',
                    'indicator': 'EMA(QQQ, 10)',
                    'result': failing_result,
                    'is_bundled': True,
                    'bundled_components': ['QQQ', 'EMA(QQQ, 10)']
                }
            }

            # Generate chart
            output_path = self.output_dir / "scenario_2_missing_ema.png"
            chart_path = create_multi_panel_chart(
                panel_results=panel_results,
                output_path=str(output_path),
                chart_title="SCENARIO 2: Missing EMA Data (User's Issue - Only QQQ Line)"
            )

            # Log results
            logger.info(f"📊 SCENARIO 2 RESULTS:")
            logger.info(f"   Chart generated: {bool(chart_path)}")
            logger.info(f"   Output path: {output_path}")
            logger.info(f"   Expected: ONLY QQQ (red) line with QQQ label - NO EMA OVERLAY")
            logger.info(f"   Data keys: {list(failing_result.keys())}")
            logger.info(f"   EMA_ema column present: {'EMA_ema' in failing_result}")
            logger.info(f"   This reproduces user's issue: 'only a red line, a red label that says QQQ'")

            # Data analysis
            close_data = failing_result['Close']
            logger.info(f"   Close data points: {len(close_data)}")
            logger.info(f"   Close range: ${close_data.min():.2f} - ${close_data.max():.2f}")
            logger.info(f"   No EMA data available - this is the problem!")

            self.test_results['scenario_2'] = {
                'status': 'success',
                'chart_path': chart_path,
                'data_keys': list(failing_result.keys()),
                'has_ema_column': 'EMA_ema' in failing_result,
                'description': 'Failing case - missing EMA_ema column (reproduces user issue)'
            }

        except Exception as e:
            logger.error(f"❌ SCENARIO 2 FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results['scenario_2'] = {'status': 'failed', 'error': str(e)}

    def test_scenario_3_wrong_column_naming(self):
        """Test Scenario 3: Wrong EMA column naming - another failure mode."""
        logger.info("\n🧪 TEST SCENARIO 3: Wrong EMA Column Naming")
        logger.info("=" * 60)

        try:
            # Create base data
            base_data = self.create_base_data("QQQ", 30)

            # Calculate EMA
            ema_period = 10
            ema_values = base_data['Close'].ewm(span=ema_period).mean()

            # Create data with WRONG column naming
            wrong_naming_result = {
                'Close': base_data['Close'].copy(),
                'EMA': ema_values.copy(),          # WRONG: Should be 'EMA_ema'
                'ema_10': ema_values.copy(),       # WRONG: Different naming
                'moving_average': ema_values.copy(),  # WRONG: Generic naming
                'metadata': {
                    'chart_type': 'overlay',
                    'base_ticker': 'QQQ',
                    'indicator': 'EMA(QQQ, 10)',
                    'calculation_date': datetime.now().isoformat(),
                    'stacking_order': 1
                }
            }

            # Test panel structure
            panel_results = {
                'Wrong_Naming_Panel': {
                    'data_source': 'QQQ + EMA(QQQ, 10)',
                    'indicator': 'EMA(QQQ, 10)',
                    'result': wrong_naming_result,
                    'is_bundled': True,
                    'bundled_components': ['QQQ', 'EMA(QQQ, 10)']
                }
            }

            # Generate chart
            output_path = self.output_dir / "scenario_3_wrong_naming.png"
            chart_path = create_multi_panel_chart(
                panel_results=panel_results,
                output_path=str(output_path),
                chart_title="SCENARIO 3: Wrong EMA Column Naming (May show some overlays)"
            )

            # Log results
            logger.info(f"📊 SCENARIO 3 RESULTS:")
            logger.info(f"   Chart generated: {bool(chart_path)}")
            logger.info(f"   Output path: {output_path}")
            logger.info(f"   Expected: May show some overlays but with wrong labels")
            logger.info(f"   Data keys: {list(wrong_naming_result.keys())}")
            logger.info(f"   EMA_ema column present: {'EMA_ema' in wrong_naming_result}")
            logger.info(f"   Alternative EMA columns: {[k for k in wrong_naming_result.keys() if 'ema' in k.lower() or 'ma' in k.lower()]}")

            self.test_results['scenario_3'] = {
                'status': 'success',
                'chart_path': chart_path,
                'data_keys': list(wrong_naming_result.keys()),
                'has_ema_column': 'EMA_ema' in wrong_naming_result,
                'description': 'Wrong column naming case'
            }

        except Exception as e:
            logger.error(f"❌ SCENARIO 3 FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results['scenario_3'] = {'status': 'failed', 'error': str(e)}

    def test_scenario_4_real_sr_system(self):
        """Test Scenario 4: Test with real SR system data processing."""
        logger.info("\n🧪 TEST SCENARIO 4: Real SR System Data Processing")
        logger.info("=" * 60)

        try:
            # Create base data
            base_data = self.create_base_data("QQQ", 30)

            # Create ticker_data dict for real system
            ticker_data = {'QQQ': base_data}

            # Test the real calculate_bundled_indicator function
            logger.info("🔧 TESTING REAL SR SYSTEM:")
            logger.info("   Calling calculate_bundled_indicator('EMA(QQQ,10)', ticker_data, 'QQQ')")

            overlay_data = calculate_bundled_indicator('EMA(QQQ,10)', ticker_data, 'QQQ')

            logger.info(f"   calculate_bundled_indicator result: {overlay_data}")

            if overlay_data:
                logger.info(f"   ✅ Overlay data generated:")
                for key, series in overlay_data.items():
                    logger.info(f"      {key}: {type(series)} with {len(series)} points")
                    logger.info(f"      Range: {series.min():.2f} to {series.max():.2f}")
            else:
                logger.error(f"   ❌ No overlay data generated - this is the problem!")

            # Create result structure with real system data
            if overlay_data:
                real_system_result = {
                    'Close': base_data['Close'].copy()
                }
                real_system_result.update(overlay_data)
                real_system_result['metadata'] = {
                    'chart_type': 'overlay',
                    'base_ticker': 'QQQ',
                    'indicator': 'EMA(QQQ, 10)',
                    'calculation_date': datetime.now().isoformat(),
                    'stacking_order': 1
                }
            else:
                # No overlay data - simulate the failure
                real_system_result = {
                    'Close': base_data['Close'].copy(),
                    'metadata': {
                        'chart_type': 'overlay',
                        'base_ticker': 'QQQ',
                        'indicator': 'EMA(QQQ, 10)',
                        'calculation_date': datetime.now().isoformat(),
                        'stacking_order': 1,
                        'error': 'Real SR system failed to generate overlay data'
                    }
                }

            # Test panel structure
            panel_results = {
                'Real_System_Panel': {
                    'data_source': 'QQQ + EMA(QQQ, 10)',
                    'indicator': 'EMA(QQQ, 10)',
                    'result': real_system_result,
                    'is_bundled': True,
                    'bundled_components': ['QQQ', 'EMA(QQQ, 10)']
                }
            }

            # Generate chart
            output_path = self.output_dir / "scenario_4_real_system.png"
            chart_path = create_multi_panel_chart(
                panel_results=panel_results,
                output_path=str(output_path),
                chart_title="SCENARIO 4: Real SR System Processing"
            )

            # Log results
            logger.info(f"📊 SCENARIO 4 RESULTS:")
            logger.info(f"   Chart generated: {bool(chart_path)}")
            logger.info(f"   Output path: {output_path}")
            logger.info(f"   Data keys: {list(real_system_result.keys())}")
            logger.info(f"   Real system overlay success: {bool(overlay_data)}")

            self.test_results['scenario_4'] = {
                'status': 'success',
                'chart_path': chart_path,
                'data_keys': list(real_system_result.keys()),
                'overlay_data_generated': bool(overlay_data),
                'description': 'Real SR system processing test'
            }

        except Exception as e:
            logger.error(f"❌ SCENARIO 4 FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results['scenario_4'] = {'status': 'failed', 'error': str(e)}

    def test_scenario_5_enhanced_detection(self):
        """Test Scenario 5: Enhanced overlay detection with various column patterns."""
        logger.info("\n🧪 TEST SCENARIO 5: Enhanced Overlay Detection")
        logger.info("=" * 60)

        try:
            # Create base data
            base_data = self.create_base_data("QQQ", 30)

            # Calculate EMA
            ema_period = 10
            ema_values = base_data['Close'].ewm(span=ema_period).mean()

            # Create data with multiple potential overlay columns
            enhanced_result = {
                'Close': base_data['Close'].copy(),
                'EMA_ema': ema_values.copy(),        # Correct format
                'ema_value': ema_values.copy(),      # Alternative format
                'ma_10': ema_values.copy(),          # Alternative format
                'indicator_ema': ema_values.copy(),  # Alternative format
                'metadata': {
                    'chart_type': 'overlay',
                    'base_ticker': 'QQQ',
                    'indicator': 'EMA(QQQ, 10)',
                    'calculation_date': datetime.now().isoformat(),
                    'stacking_order': 1
                }
            }

            # Test enhanced overlay detection
            from src.sustainability_ratios.sr_dashboard_generator import is_indicator_column

            logger.info("🔍 TESTING ENHANCED OVERLAY DETECTION:")
            for column in enhanced_result.keys():
                if column != 'metadata':
                    is_indicator = is_indicator_column(column)
                    logger.info(f"   '{column}': is_indicator_column = {is_indicator}")

            # Test panel structure
            panel_results = {
                'Enhanced_Detection_Panel': {
                    'data_source': 'QQQ + EMA(QQQ, 10)',
                    'indicator': 'EMA(QQQ, 10)',
                    'result': enhanced_result,
                    'is_bundled': True,
                    'bundled_components': ['QQQ', 'EMA(QQQ, 10)']
                }
            }

            # Generate chart
            output_path = self.output_dir / "scenario_5_enhanced_detection.png"
            chart_path = create_multi_panel_chart(
                panel_results=panel_results,
                output_path=str(output_path),
                chart_title="SCENARIO 5: Enhanced Overlay Detection (Multiple potential overlays)"
            )

            # Log results
            logger.info(f"📊 SCENARIO 5 RESULTS:")
            logger.info(f"   Chart generated: {bool(chart_path)}")
            logger.info(f"   Output path: {output_path}")
            logger.info(f"   Data keys: {list(enhanced_result.keys())}")
            logger.info(f"   Multiple overlay columns available")

            self.test_results['scenario_5'] = {
                'status': 'success',
                'chart_path': chart_path,
                'data_keys': list(enhanced_result.keys()),
                'description': 'Enhanced overlay detection with multiple column patterns'
            }

        except Exception as e:
            logger.error(f"❌ SCENARIO 5 FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results['scenario_5'] = {'status': 'failed', 'error': str(e)}

    def generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        logger.info("\n📋 GENERATING COMPARISON REPORT")
        logger.info("=" * 60)

        report_path = self.output_dir / "diagnostic_report.md"

        with open(report_path, 'w') as f:
            f.write("# EMA Overlay Diagnostic Test Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Test Objective\n")
            f.write("Reproduce and diagnose the user's reported issue: 'only a red line, a red label that says QQQ' with no EMA overlay.\n\n")

            f.write("## Test Scenarios\n\n")

            for scenario_id, result in self.test_results.items():
                scenario_num = scenario_id.replace('scenario_', '')
                f.write(f"### Scenario {scenario_num}\n")
                f.write(f"**Status**: {result['status']}\n")
                f.write(f"**Description**: {result.get('description', 'N/A')}\n")

                if result['status'] == 'success':
                    f.write(f"**Chart Path**: {result.get('chart_path', 'N/A')}\n")
                    f.write(f"**Data Keys**: {result.get('data_keys', [])}\n")
                    f.write(f"**Has EMA Column**: {result.get('has_ema_column', False)}\n")

                    if scenario_id == 'scenario_2':
                        f.write("**⚠️ This reproduces the user's issue!**\n")
                else:
                    f.write(f"**Error**: {result.get('error', 'Unknown error')}\n")

                f.write("\n")

            f.write("## Key Findings\n\n")
            f.write("1. **Root Cause**: Missing `EMA_ema` column in panel data\n")
            f.write("2. **Expected Column**: The chart generation expects `EMA_ema` for overlays\n")
            f.write("3. **Failure Mode**: When `calculate_bundled_indicator()` fails, no overlay data is generated\n")
            f.write("4. **Result**: Only base ticker data is plotted (single red line with ticker label)\n\n")

            f.write("## Recommendations\n\n")
            f.write("1. Debug `calculate_bundled_indicator()` function in sr_market_data.py\n")
            f.write("2. Check EMA calculation in indicators/MAs.py\n")
            f.write("3. Verify indicator parsing in enhanced_panel_parser.py\n")
            f.write("4. Add fallback error handling for missing overlay data\n")
            f.write("5. Implement more robust column name detection\n")

        logger.info(f"📄 Diagnostic report saved: {report_path}")
        return str(report_path)

    def run_all_tests(self):
        """Run all diagnostic tests."""
        logger.info("\n🚀 STARTING COMPREHENSIVE EMA OVERLAY DIAGNOSTIC")
        logger.info("=" * 80)

        # Run all test scenarios
        self.test_scenario_1_working_ema_data()
        self.test_scenario_2_missing_ema_data()
        self.test_scenario_3_wrong_column_naming()
        self.test_scenario_4_real_sr_system()
        self.test_scenario_5_enhanced_detection()

        # Generate report
        report_path = self.generate_comparison_report()

        logger.info("\n✅ DIAGNOSTIC TEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Test Results Summary:")
        for scenario_id, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            logger.info(f"   {status_icon} {scenario_id}: {result['status']}")

        logger.info(f"\n📄 Reports Generated:")
        logger.info(f"   Diagnostic Report: {report_path}")
        logger.info(f"   Charts Directory: {self.output_dir}")
        logger.info(f"   Log File: ema_diagnostic_test.log")

        return self.test_results


def main():
    """Main function to run the diagnostic test."""
    print("EMA Overlay Diagnostic Test")
    print("=" * 40)
    print("This test reproduces the user's issue:")
    print("'only a red line, a red label that says QQQ'")
    print("=" * 40)

    # Run diagnostic tests
    diagnostic = EMAADiagnosticTest()
    results = diagnostic.run_all_tests()

    print(f"\n📊 Test completed! Check the following files:")
    print(f"   📈 Charts: {diagnostic.output_dir}/")
    print(f"   📄 Report: {diagnostic.output_dir}/diagnostic_report.md")
    print(f"   📝 Log: ema_diagnostic_test.log")

    return results


if __name__ == "__main__":
    main()