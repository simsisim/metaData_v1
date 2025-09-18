import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
basic_calc = pd.read_csv('basic_calculation_daily_0_20250829.csv')
universe = pd.read_csv('tradingview_universe.csv')

# First, let's understand the structure of both files
print("Basic Calculation Dataset Info:")
print(f"Shape: {basic_calc.shape}")
print(f"Columns: {basic_calc.columns.tolist()}")
print(f"Date range: {basic_calc['date'].min()} to {basic_calc['date'].max()}")
print(f"Unique tickers: {basic_calc['ticker'].nunique()}")

print("\n" + "="*50 + "\n")

print("Universe Dataset Info:")
print(f"Shape: {universe.shape}")
print(f"Columns: {universe.columns.tolist()}")
print(f"Unique symbols: {universe['Symbol'].nunique()}")

# Check for overlapping tickers
basic_tickers = set(basic_calc['ticker'].unique())
universe_symbols = set(universe['Symbol'].unique())
overlap = basic_tickers.intersection(universe_symbols)
print(f"\nOverlapping tickers between datasets: {len(overlap)}")
print(f"Sample overlapping tickers: {list(overlap)[:10]}")