import pandas as pd
import numpy as np

# Load and examine all the datasets
daily_df = pd.read_csv('basic_calculation_daily_0_20250829.csv')
weekly_df = pd.read_csv('basic_calculation_weekly_0_20250825.csv')
monthly_df = pd.read_csv('basic_calculation_monthly_0_20250829.csv')

print("Daily Data Shape:", daily_df.shape)
print("Weekly Data Shape:", weekly_df.shape)
print("Monthly Data Shape:", monthly_df.shape)

print("\n--- Daily Data Sample ---")
print(daily_df.head())
print("\nDaily Data Columns:")
print(list(daily_df.columns))