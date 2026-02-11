import pandas as pd
import numpy as np

# Load and examine the data files to understand what we're working with
files = [
    'basic_calculation_daily_0_20250829.csv',
    'basic_calculation_weekly_0_20250825.csv', 
    'basic_calculation_monthly_0_20250829.csv',
    'rs_ibd_stocks_daily_0_20250829.csv',
    'stage_analysis_daily_0_20250829.csv'
]

# Load daily data first to see what we have
daily_df = pd.read_csv('basic_calculation_daily_0_20250829.csv')
print("Daily data shape:", daily_df.shape)
print("\nDaily data columns:")
print(daily_df.columns.tolist())
print("\nFirst few rows of daily data:")
print(daily_df.head())