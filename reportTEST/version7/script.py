import pandas as pd
import numpy as np

# Load the basic calculation data
df_calc = pd.read_csv('basic_calculation_daily_0_20250829.csv')

print("Basic Calculation Data Overview:")
print(f"Shape: {df_calc.shape}")
print(f"\nColumns ({len(df_calc.columns)}):")
for i, col in enumerate(df_calc.columns):
    print(f"{i+1}. {col}")

print(f"\nFirst few rows:")
print(df_calc.head())

print(f"\nTickers included:")
print(df_calc['ticker'].unique()[:20])  # Show first 20 tickers
print(f"Total tickers: {len(df_calc['ticker'].unique())}")