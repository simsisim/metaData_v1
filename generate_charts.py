#!/usr/bin/env python3
"""
Chart Generation Script
=======================

Generate technical indicator charts for all available tickers using the
indicators module and configuration files.

Usage:
    python generate_charts.py [--tickers TICKER1,TICKER2] [--timeframe daily|weekly]
"""

import sys
from pathlib import Path
sys.path.insert(0, 'src')

from src.user_defined_data import read_user_data
from src.data_reader import DataReader
from src.config import Config
from src.indicators.indicators_charts import create_charts_from_config_file, create_indicator_chart
import pandas as pd
import argparse


def main():
    """Main chart generation function."""
    parser = argparse.ArgumentParser(description='Generate technical indicator charts')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers (default: use config file)')
    parser.add_argument('--timeframe', type=str, default='daily', choices=['daily', 'weekly'], 
                       help='Timeframe for charts (default: daily)')
    args = parser.parse_args()
    
    print("🚀 TECHNICAL INDICATORS CHART GENERATION")
    print("=" * 60)
    
    # Load configuration
    user_config = read_user_data()
    config = Config()
    
    if not user_config.indicators_enable:
        print("❌ Indicators are disabled in configuration")
        return
    
    # Initialize data reader
    data_reader = DataReader(config, args.timeframe, 100)
    data_reader.load_tickers_from_file(Path('data/tickers/combined_tickers_0.csv'))
    
    # Determine which tickers to process
    if args.tickers:
        # Use specified tickers
        tickers_to_process = [t.strip().upper() for t in args.tickers.split(',')]
        print(f"📋 Processing specified tickers: {tickers_to_process}")
    else:
        # Use configuration file
        config_file = Path(user_config.indicators_config_file)
        if config_file.exists():
            config_df = pd.read_csv(config_file)
            tickers_to_process = config_df['ticker'].tolist()
            print(f"📋 Processing tickers from config file: {len(tickers_to_process)} tickers")
        else:
            print(f"❌ Configuration file not found: {config_file}")
            return
    
    # Load market data
    print(f"📊 Loading {args.timeframe} market data...")
    batch_data = data_reader.read_batch_data(tickers_to_process, validate=True)
    
    if not batch_data:
        print("❌ No market data available for specified tickers")
        return
    
    print(f"✅ Loaded market data for {len(batch_data)} tickers")
    
    # Create charts directory
    charts_dir = Path(user_config.indicators_charts_dir) / args.timeframe
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    if args.tickers:
        # Generate charts with default config for specified tickers
        indicators_config = {
            'tsi': True,
            'macd': True,
            'mfi': True,
            'kurutoga': True,
            'momentum': True,
            'rsi': True,
            'ma_crosses': True
        }
        
        created_charts = []
        for ticker in batch_data.keys():
            try:
                print(f"📈 Generating chart for {ticker}...")
                chart_path = create_indicator_chart(
                    batch_data[ticker], ticker, charts_dir, indicators_config, args.timeframe
                )
                created_charts.append(chart_path)
                
            except Exception as e:
                print(f"❌ Error generating chart for {ticker}: {e}")
                continue
    else:
        # Generate charts from configuration file
        print(f"📈 Generating charts from configuration...")
        config_file = Path(user_config.indicators_config_file)
        created_charts = create_charts_from_config_file(
            batch_data, config_file, charts_dir, args.timeframe
        )
    
    # Summary
    print(f"\n✅ CHART GENERATION COMPLETED!")
    print(f"📊 Total charts created: {len(created_charts)}")
    print(f"📁 Charts saved to: {charts_dir}")
    
    if created_charts:
        print("\n📈 Generated charts:")
        for chart in created_charts:
            chart_name = Path(chart).name
            print(f"  • {chart_name}")
    
    print(f"\n🎯 To view charts, check: {charts_dir}")


if __name__ == "__main__":
    main()