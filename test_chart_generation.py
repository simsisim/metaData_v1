#!/usr/bin/env python3

"""
Test actual chart generation with the candlestick fix.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.sustainability_ratios.sr_dashboard_generator import plot_candlestick_chart
from src.data_reader import DataReader
from src.config import Config
import matplotlib.pyplot as plt
import numpy as np

def test_chart_generation():
    """Test actual candlestick chart generation with SPY data."""

    print("🎨 Testing candlestick chart generation...")

    try:
        # Load SPY data
        config = Config()
        data_reader = DataReader(config)
        spy_data = data_reader.read_stock_data('SPY')

        if spy_data is None:
            print("❌ Could not load SPY data")
            return

        # Limit to last 30 days for testing
        spy_data = spy_data.tail(30)
        print(f"✅ SPY data loaded: {len(spy_data)} rows")
        print(f"📊 Columns: {list(spy_data.columns)}")

        # Create the result structure as the fixed code would
        result = {
            'Open': spy_data['Open'],
            'High': spy_data['High'],
            'Low': spy_data['Low'],
            'Close': spy_data['Close'],
            'Volume': spy_data['Volume'],
            'price': spy_data['Close'],
            'metadata': {
                'chart_type': 'price',
                'data_type': 'ohlcv'
            }
        }

        print(f"📋 Result structure created with keys: {list(result.keys())}")

        # Test the candlestick plotting function
        fig, ax = plt.subplots(figsize=(12, 8))
        x_positions = range(len(spy_data))
        main_label = "SPY"

        print(f"🎯 Testing plot_candlestick_chart function...")

        # Check if chart was plotted by examining axis contents
        ax_children_before = len(ax.get_children())
        print(f"📊 Axis children before plotting: {ax_children_before}")

        # Call the actual plotting function
        try:
            plot_result = plot_candlestick_chart(ax, result, x_positions, main_label)
            print(f"📊 plot_candlestick_chart returned: {type(plot_result)}")
        except Exception as plot_error:
            print(f"❌ plot_candlestick_chart failed: {plot_error}")
            plot_result = None

        if plot_result is None:
            # This is expected for candlestick plots - they return None but plot directly
            ax_children_after = len(ax.get_children())
            print(f"📊 Axis children after plotting: {ax_children_after}")

            if ax_children_after > ax_children_before:
                print(f"✅ Candlestick chart elements added to axis!")
                print(f"📈 Added {ax_children_after - ax_children_before} chart elements")

                # Save a test chart
                ax.set_title(f"SPY Candlestick Chart Test - Last {len(spy_data)} Days")
                ax.set_xlabel("Trading Days")
                ax.set_ylabel("Price ($)")
                ax.grid(True, alpha=0.3)

                test_chart_path = "test_spy_candlestick.png"
                plt.savefig(test_chart_path, dpi=150, bbox_inches='tight')
                print(f"💾 Test chart saved: {test_chart_path}")

                # Check data range on axis
                if ax.dataLim.width > 0 and ax.dataLim.height > 0:
                    print(f"📊 Chart data range: X={ax.dataLim.x0:.1f} to {ax.dataLim.x1:.1f}, Y=${ax.dataLim.y0:.2f} to ${ax.dataLim.y1:.2f}")
                else:
                    print(f"⚠️ Chart has no data range set")

            else:
                print(f"❌ No chart elements were added - candlestick plotting failed")
        else:
            print(f"🤔 Unexpected return value: {plot_result}")

        plt.close()

        print(f"\n✅ Chart generation test completed!")

    except Exception as e:
        print(f"❌ Error testing chart generation: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')

if __name__ == "__main__":
    test_chart_generation()