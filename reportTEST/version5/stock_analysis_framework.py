
"""
Comprehensive Stock Market Analysis Framework
============================================

This script provides an expandable framework for analyzing stock market data
combining trading metrics with fundamental company information.

Features:
- Automatic data loading and merging
- Sector and industry analysis
- Index membership tracking
- Risk-return analysis with technical indicators
- Multi-period performance comparison
- Comprehensive visualizations
- Expandable design for new data

Usage:
1. Place trading data files as: basic_calculation_daily_YYYYMMDD.csv
2. Update tradingview_universe.csv with stock information
3. Run this script to generate analysis and charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self, config_file='analysis_config.json'):
        """Initialize the analyzer with configuration"""
        self.config = self.load_config(config_file)
        self.trading_data = None
        self.universe_data = None
        self.merged_data = None

    def load_config(self, config_file):
        """Load analysis configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_default_config()

    def create_default_config(self):
        """Create default configuration if none exists"""
        return {
            'data_files': {
                'trading_data_pattern': 'basic_calculation_daily_*.csv',
                'universe_data': 'tradingview_universe.csv'
            },
            'key_metrics': {
                'performance_columns': [
                    'daily_daily_daily_1d_pct_change',
                    'daily_daily_daily_5d_pct_change',
                    'daily_daily_weekly_7d_pct_change',
                    'daily_daily_monthly_22d_pct_change',
                    'daily_daily_quarterly_66d_pct_change',
                    'daily_daily_yearly_252d_pct_change'
                ],
                'technical_indicators': [
                    'daily_rsi_14', 'daily_momentum_20', 'daily_macd', 
                    'daily_macd_signal', 'daily_price_position_52w', 'atr_pct'
                ],
                'major_indices': [
                    'S&P 500', 'NASDAQ 100', 'NASDAQ Composite', 
                    'Dow Jones Industrial Average', 'Russell 1000', 'Russell 3000'
                ]
            }
        }

    def load_data(self, latest_only=True):
        """Load and prepare data"""
        print("Loading data...")

        # Load trading data (latest file or all files)
        trading_pattern = self.config['data_files']['trading_data_pattern']
        trading_files = glob.glob(trading_pattern)

        if not trading_files:
            raise FileNotFoundError(f"No trading data files found matching: {trading_pattern}")

        if latest_only:
            # Use most recent file
            latest_file = max(trading_files)
            self.trading_data = pd.read_csv(latest_file)
            print(f"Loaded trading data from: {latest_file}")
        else:
            # Combine all files
            all_trading_data = []
            for file in trading_files:
                df = pd.read_csv(file)
                all_trading_data.append(df)
            self.trading_data = pd.concat(all_trading_data, ignore_index=True)
            print(f"Loaded trading data from {len(trading_files)} files")

        # Load universe data
        universe_file = self.config['data_files']['universe_data']
        self.universe_data = pd.read_csv(universe_file)
        print(f"Loaded universe data: {len(self.universe_data)} stocks")

        # Merge datasets
        self.merged_data = self.trading_data.merge(
            self.universe_data, left_on='ticker', right_on='Symbol', how='left'
        )
        print(f"Merged data: {len(self.merged_data)} records")

        return self

    def prepare_data(self):
        """Prepare data for analysis"""
        print("Preparing data...")

        # Parse indices
        def parse_indices(index_str):
            if pd.isna(index_str):
                return []
            return [idx.strip() for idx in str(index_str).split(',')]

        self.merged_data['indices_list'] = self.merged_data['Index'].apply(parse_indices)

        # Create index membership flags
        major_indices = self.config['key_metrics']['major_indices']
        for index in major_indices:
            col_name = f'in_{index.replace(" ", "_").replace("&", "").lower()}'
            self.merged_data[col_name] = self.merged_data['indices_list'].apply(
                lambda x: index in x if x else False
            )

        # Add derived metrics
        self.merged_data['market_cap_billions'] = self.merged_data['Market capitalization'] / 1e9
        self.merged_data['rsi_category'] = pd.cut(
            self.merged_data['daily_rsi_14'], 
            bins=[0, 30, 70, 100], 
            labels=['Oversold (<30)', 'Neutral (30-70)', 'Overbought (>70)']
        )

        print("Data preparation complete!")
        return self

    def analyze_sectors(self):
        """Analyze sector performance"""
        print("Analyzing sector performance...")

        performance_cols = self.config['key_metrics']['performance_columns']
        sector_performance = self.merged_data.groupby('Sector')[performance_cols].mean().round(2)

        # Save results
        sector_performance.reset_index().to_csv('sector_performance.csv', index=False)

        return sector_performance

    def analyze_indices(self):
        """Analyze index membership performance"""
        print("Analyzing index membership...")

        major_indices = self.config['key_metrics']['major_indices']
        performance_cols = self.config['key_metrics']['performance_columns']

        index_analysis = []
        for index in major_indices:
            col_name = f'in_{index.replace(" ", "_").replace("&", "").lower()}'
            in_index = self.merged_data[self.merged_data[col_name] == True]
            not_in_index = self.merged_data[self.merged_data[col_name] == False]

            analysis_row = {'Index': index, 'In_Index_Count': len(in_index)}

            for perf_col in performance_cols:
                period = perf_col.split('_')[-1].replace('pct', '').replace('change', '')
                analysis_row[f'In_Index_{period}_Avg'] = in_index[perf_col].mean()
                analysis_row[f'Not_In_Index_{period}_Avg'] = not_in_index[perf_col].mean()

            index_analysis.append(analysis_row)

        index_df = pd.DataFrame(index_analysis)
        index_df.to_csv('index_analysis.csv', index=False)

        return index_df

    def analyze_risk_return(self):
        """Analyze risk-return characteristics"""
        print("Analyzing risk-return profiles...")

        risk_return_data = self.merged_data[[
            'ticker', 'Description', 'Sector', 'current_price',
            'daily_daily_yearly_252d_pct_change', 'atr_pct', 'daily_rsi_14',
            'daily_price_position_52w', 'market_cap_billions'
        ]].copy()

        risk_return_data.to_csv('risk_return_analysis.csv', index=False)

        return risk_return_data

    def analyze_technical_indicators(self):
        """Analyze technical indicators by sector"""
        print("Analyzing technical indicators...")

        technical_cols = self.config['key_metrics']['technical_indicators']
        technical_by_sector = self.merged_data.groupby('Sector')[technical_cols].mean().round(2)

        technical_by_sector.to_csv('technical_analysis_by_sector.csv')

        return technical_by_sector

    def analyze_time_performance(self):
        """Analyze performance across different time periods"""
        print("Analyzing time-based performance...")

        time_periods = {
            '1_Day': 'daily_daily_daily_1d_pct_change',
            '5_Day': 'daily_daily_daily_5d_pct_change',
            '7_Day': 'daily_daily_weekly_7d_pct_change',
            '1_Month': 'daily_daily_monthly_22d_pct_change',
            '3_Month': 'daily_daily_quarterly_66d_pct_change',
            '1_Year': 'daily_daily_yearly_252d_pct_change'
        }

        time_analysis = []
        for period, col in time_periods.items():
            time_analysis.append({
                'Period': period,
                'Mean_Return': self.merged_data[col].mean(),
                'Median_Return': self.merged_data[col].median(),
                'Std_Dev': self.merged_data[col].std(),
                'Min_Return': self.merged_data[col].min(),
                'Max_Return': self.merged_data[col].max(),
                'Positive_Count': (self.merged_data[col] > 0).sum(),
                'Negative_Count': (self.merged_data[col] < 0).sum()
            })

        time_df = pd.DataFrame(time_analysis)
        time_df.to_csv('time_performance_analysis.csv', index=False)

        return time_df

    def generate_summary(self):
        """Generate comprehensive analysis summary"""
        print("\n" + "="*60)
        print("COMPREHENSIVE STOCK ANALYSIS SUMMARY")
        print("="*60)

        # Basic stats
        if 'date' in self.merged_data.columns:
            print(f"Analysis Date: {self.merged_data['date'].iloc[0]}")
        print(f"Total Stocks Analyzed: {len(self.merged_data)}")
        print(f"Unique Sectors: {self.merged_data['Sector'].nunique()}")
        print(f"Unique Industries: {self.merged_data['Industry'].nunique()}")

        # Top performers
        if 'daily_daily_yearly_252d_pct_change' in self.merged_data.columns:
            top_performer = self.merged_data.loc[
                self.merged_data['daily_daily_yearly_252d_pct_change'].idxmax()
            ]
            print(f"\nTop 1Y Performer: {top_performer['ticker']} " +
                  f"(+{top_performer['daily_daily_yearly_252d_pct_change']:.1f}%)")

        if 'market_cap_billions' in self.merged_data.columns:
            largest = self.merged_data.loc[self.merged_data['market_cap_billions'].idxmax()]
            print(f"Largest Company: {largest['ticker']} (${largest['market_cap_billions']:.0f}B)")

        # Index membership
        major_indices = self.config['key_metrics']['major_indices']
        print(f"\nIndex Membership:")
        for index in major_indices[:3]:  # Show top 3
            col_name = f'in_{index.replace(" ", "_").replace("&", "").lower()}'
            if col_name in self.merged_data.columns:
                count = self.merged_data[col_name].sum()
                pct = count/len(self.merged_data)*100
                print(f"  • {index}: {count} stocks ({pct:.1f}%)")

    def run_full_analysis(self, latest_only=True):
        """Run complete analysis pipeline"""
        print("Starting comprehensive stock analysis...")

        # Load and prepare data
        self.load_data(latest_only=latest_only)
        self.prepare_data()

        # Run all analyses
        sector_perf = self.analyze_sectors()
        index_analysis = self.analyze_indices()
        risk_return = self.analyze_risk_return()
        technical = self.analyze_technical_indicators()
        time_perf = self.analyze_time_performance()

        # Generate summary
        self.generate_summary()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("Files generated:")
        output_files = [
            'sector_performance.csv',
            'index_analysis.csv', 
            'risk_return_analysis.csv',
            'technical_analysis_by_sector.csv',
            'time_performance_analysis.csv'
        ]
        for i, file in enumerate(output_files, 1):
            print(f"  {i}. {file}")

        return {
            'sector_performance': sector_perf,
            'index_analysis': index_analysis,
            'risk_return': risk_return,
            'technical': technical,
            'time_performance': time_perf,
            'merged_data': self.merged_data
        }

# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = StockAnalyzer()

    # Run full analysis
    results = analyzer.run_full_analysis(latest_only=True)

    print("\n🚀 Next steps:")
    print("1. Add new data files and re-run analysis")
    print("2. Create custom visualizations using the CSV outputs")
    print("3. Modify analysis_config.json to customize metrics")
    print("4. Use results['merged_data'] for additional analysis")
