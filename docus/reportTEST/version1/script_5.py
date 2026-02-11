import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load all the data files
stocks_daily = pd.read_csv('rs_ibd_stocks_0_daily_20250829.csv')
stocks_weekly = pd.read_csv('rs_ibd_stocks_0_weekly_20250825.csv')
stocks_monthly = pd.read_csv('rs_ibd_stocks_0_monthly_20250829.csv')

sectors_daily = pd.read_csv('rs_ibd_sectors_0_daily_20250906.csv')
sectors_weekly = pd.read_csv('rs_ibd_sectors_0_weekly_20250906.csv')
sectors_monthly = pd.read_csv('rs_ibd_sectors_0_monthly_20250906.csv')

industries_daily = pd.read_csv('rs_ibd_industries_0_daily_20250906.csv')
industries_weekly = pd.read_csv('rs_ibd_industries_0_weekly_20250906.csv')
industries_monthly = pd.read_csv('rs_ibd_industries_0_monthly_20250906.csv')

# Display basic info about the datasets
print("=== DATASET OVERVIEW ===")
print(f"Stocks Daily: {len(stocks_daily)} records")
print(f"Stocks Weekly: {len(stocks_weekly)} records") 
print(f"Stocks Monthly: {len(stocks_monthly)} records")
print(f"Sectors Daily: {len(sectors_daily)} records")
print(f"Sectors Weekly: {len(sectors_weekly)} records")
print(f"Sectors Monthly: {len(sectors_monthly)} records")
print(f"Industries Daily: {len(industries_daily)} records")
print(f"Industries Weekly: {len(industries_weekly)} records")
print(f"Industries Monthly: {len(industries_monthly)} records")

# Display column names to understand structure
print("\n=== STOCKS DAILY COLUMNS ===")
print(stocks_daily.columns.tolist())