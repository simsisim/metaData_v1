#!/usr/bin/env python3
"""
Sparkline Data Processor
Processes historical price data for sparkline chart generation
Handles first 100 tickers for initial implementation
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class SparklineDataProcessor:
    """Process historical price data for sparkline visualization"""

    def __init__(self, data_dir="/home/imagda/_invest2024/python/downloadData_v1/data/market_data/daily/"):
        self.data_dir = data_dir
        self.ticker_data = {}
        self.first_100_tickers = self.get_first_100_tickers()

    def get_first_100_tickers(self):
        """Get the first 100 tickers from the data directory"""
        print("📊 Getting first 100 tickers from data directory...")

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Get all CSV files and sort them
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        csv_files.sort()  # Alphabetical order

        # Get first 100 and extract ticker symbols
        first_100_files = csv_files[:100]
        tickers = [f.replace('.csv', '') for f in first_100_files]

        print(f"✅ Found {len(tickers)} tickers to process")
        print(f"📈 Range: {tickers[0]} to {tickers[-1]}")

        return tickers

    def load_ticker_data(self, ticker):
        """Load and parse CSV for individual ticker"""
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")

        try:
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df = df.sort_values('Date')  # Ensure chronological order
            df = df.dropna(subset=['Close'])  # Remove any rows with missing Close prices

            if df.empty:
                print(f"⚠️  Warning: No data found for {ticker}")
                return None

            return df

        except Exception as e:
            print(f"❌ Error loading data for {ticker}: {e}")
            return None

    def extract_timeframe_data(self, df, timeframe):
        """Extract price data for specific timeframe"""
        if df is None or df.empty:
            return []

        # Define number of trading days for each timeframe
        timeframe_days = {
            '1d': 1,
            '1w': 5,
            '1m': 22,
            '3m': 66,   # Quarter
            '6m': 132,  # Half year
            '1y': 252   # Year
        }

        days = timeframe_days.get(timeframe, 22)

        # Get the last N trading days
        recent_data = df.tail(days)
        close_prices = recent_data['Close'].tolist()

        # For very short timeframes, ensure we have at least some data
        if len(close_prices) == 0:
            return []

        # Convert to list of floats and round to 2 decimal places
        close_prices = [round(float(price), 2) for price in close_prices if not pd.isna(price)]

        return close_prices

    def calculate_performance_metrics(self, prices):
        """Calculate basic performance metrics for sparkline data"""
        if len(prices) < 2:
            return {
                'total_return': 0,
                'volatility': 0,
                'trend': 'neutral'
            }

        # Calculate total return
        start_price = prices[0]
        end_price = prices[-1]
        total_return = ((end_price - start_price) / start_price) * 100

        # Calculate simple volatility (standard deviation of returns)
        returns = []
        for i in range(1, len(prices)):
            ret = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
            returns.append(ret)

        volatility = np.std(returns) if returns else 0

        # Determine trend
        if total_return > 1:
            trend = 'bullish'
        elif total_return < -1:
            trend = 'bearish'
        else:
            trend = 'neutral'

        return {
            'total_return': round(total_return, 2),
            'volatility': round(volatility, 2),
            'trend': trend
        }

    def generate_sparkline_data(self, ticker):
        """Generate all timeframe data for a ticker"""
        print(f"📈 Processing {ticker}...")

        df = self.load_ticker_data(ticker)
        if df is None:
            return None

        # Extract data for all timeframes
        timeframes = ['1d', '1w', '1m', '3m', '6m', '1y']
        sparkline_data = {
            'ticker': ticker,
            'current_price': round(float(df['Close'].iloc[-1]), 2) if not df.empty else 0,
            'last_update': df['Date'].iloc[-1].strftime('%Y-%m-%d') if not df.empty else '',
            'timeframes': {}
        }

        for timeframe in timeframes:
            prices = self.extract_timeframe_data(df, timeframe)
            metrics = self.calculate_performance_metrics(prices)

            sparkline_data['timeframes'][timeframe] = {
                'prices': prices,
                'data_points': len(prices),
                'metrics': metrics
            }

        return sparkline_data

    def process_all_tickers(self):
        """Process all first 100 tickers and generate sparkline data"""
        print("\n" + "="*60)
        print("🚀 Starting Sparkline Data Processing")
        print(f"📊 Processing {len(self.first_100_tickers)} tickers")
        print("="*60)

        all_ticker_data = []
        successful_count = 0
        failed_count = 0

        for i, ticker in enumerate(self.first_100_tickers, 1):
            print(f"[{i:3d}/100] Processing {ticker}...", end=" ")

            try:
                ticker_data = self.generate_sparkline_data(ticker)
                if ticker_data:
                    all_ticker_data.append(ticker_data)
                    successful_count += 1
                    print("✅")
                else:
                    failed_count += 1
                    print("❌")

            except Exception as e:
                print(f"❌ Error: {e}")
                failed_count += 1

        print("\n" + "="*60)
        print(f"✅ Processing Complete!")
        print(f"📈 Successful: {successful_count} tickers")
        print(f"❌ Failed: {failed_count} tickers")
        print(f"💾 Total data points: {sum(len(td['timeframes']['1y']['prices']) for td in all_ticker_data)}")
        print("="*60)

        return all_ticker_data

    def save_processed_data(self, all_ticker_data, output_file="processed_sparkline_data.json"):
        """Save processed data to JSON file"""
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'outputs',
            output_file
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(all_ticker_data, f, indent=2)

        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"💾 Processed data saved: {output_path}")
        print(f"📄 File size: {file_size:.1f} KB")

        return output_path

def main():
    """Main execution function"""
    try:
        # Initialize processor
        processor = SparklineDataProcessor()

        # Process all tickers
        all_ticker_data = processor.process_all_tickers()

        if all_ticker_data:
            # Save processed data
            output_file = processor.save_processed_data(all_ticker_data)

            # Display summary statistics
            print("\n📊 Data Summary:")
            print(f"   • Total tickers processed: {len(all_ticker_data)}")

            # Find ticker with most data points
            max_data_ticker = max(all_ticker_data,
                                key=lambda x: x['timeframes']['1y']['data_points'])
            print(f"   • Most data points: {max_data_ticker['ticker']} "
                  f"({max_data_ticker['timeframes']['1y']['data_points']} days)")

            # Find best/worst performers
            performers = [(td['ticker'], td['timeframes']['1y']['metrics']['total_return'])
                         for td in all_ticker_data
                         if td['timeframes']['1y']['prices']]

            if performers:
                best_performer = max(performers, key=lambda x: x[1])
                worst_performer = min(performers, key=lambda x: x[1])

                print(f"   • Best performer (1Y): {best_performer[0]} (+{best_performer[1]:.1f}%)")
                print(f"   • Worst performer (1Y): {worst_performer[0]} ({worst_performer[1]:.1f}%)")

            print(f"\n✅ Ready for HTML generation!")
            return output_file
        else:
            print("❌ No data processed successfully")
            return None

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()