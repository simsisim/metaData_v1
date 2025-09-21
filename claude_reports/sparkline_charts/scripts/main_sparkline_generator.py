#!/usr/bin/env python3
"""
Main Sparkline Generator
Orchestrates the complete process of generating historical price sparkline charts
Processes first 100 tickers and creates interactive HTML report
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from sparkline_data_processor import SparklineDataProcessor
from sparkline_html_generator import SparklineHTMLGenerator

def main():
    """Main execution function"""
    print("🚀 Historical Price Sparkline Charts Generator")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Processing: First 100 tickers")
    print("📊 Timeframes: 1D, 1W, 1M, 3M, 6M, 1Y")
    print("=" * 60)

    try:
        # Step 1: Process ticker data
        print("\n🔄 STEP 1: Processing ticker data...")
        processor = SparklineDataProcessor()
        all_ticker_data = processor.process_all_tickers()

        if not all_ticker_data:
            print("❌ No ticker data processed successfully")
            return

        # Save processed data
        data_file = processor.save_processed_data(all_ticker_data)
        print(f"✅ Data processing complete: {len(all_ticker_data)} tickers")

        # Step 2: Generate HTML report
        print("\n🔄 STEP 2: Generating HTML report...")
        generator = SparklineHTMLGenerator()
        html_report = generator.create_complete_report()

        if html_report:
            print(f"✅ HTML report generated successfully!")

            # Display final summary
            print("\n" + "=" * 60)
            print("🎉 SPARKLINE CHARTS GENERATION COMPLETE!")
            print("=" * 60)
            print(f"📊 Tickers processed: {len(all_ticker_data)}")
            print(f"📈 Total sparklines: {len(all_ticker_data) * 6}")
            print(f"📄 HTML report: {os.path.basename(html_report)}")
            print(f"🌐 Open in browser: file://{html_report}")

            # Performance summary
            performers = [(td['ticker'], td['timeframes']['1y']['metrics']['total_return'])
                         for td in all_ticker_data
                         if td['timeframes']['1y']['prices']]

            if performers:
                best = max(performers, key=lambda x: x[1])
                worst = min(performers, key=lambda x: x[1])
                print(f"🚀 Best performer (1Y): {best[0]} (+{best[1]:.1f}%)")
                print(f"📉 Worst performer (1Y): {worst[0]} ({worst[1]:.1f}%)")

            print("=" * 60)
            print("✨ Features: Interactive search, sorting, tooltips, export")
            print("💡 Tip: Use search box to filter tickers, click headers to sort")

            return html_report

        else:
            print("❌ Failed to generate HTML report")
            return None

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()

    if result:
        print(f"\n🎯 Quick Start:")
        print(f"   Open: {result}")
        print(f"   Or run: python -m http.server 8000")
        print(f"   Then visit: http://localhost:8000/{os.path.basename(result)}")
    else:
        print("\n❌ Generation failed. Check error messages above.")
        sys.exit(1)