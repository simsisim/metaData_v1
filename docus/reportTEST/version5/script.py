import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load the datasets
print("Loading datasets...")
trading_data = pd.read_csv('basic_calculation_daily_0_20250829.csv')
universe_data = pd.read_csv('tradingview_universe.csv')

print(f"Trading data shape: {trading_data.shape}")
print(f"Universe data shape: {universe_data.shape}")
print("\nTrading data columns:", list(trading_data.columns)[:10], "...")
print("\nUniverse data columns:", list(universe_data.columns))