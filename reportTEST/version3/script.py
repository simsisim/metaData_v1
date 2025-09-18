import pandas as pd

# Load the stage analysis daily data to see which stocks are in bullish trends
stage_analysis_daily = pd.read_csv('stage_analysis_daily_0_20250829.csv')

# Display the structure and first few rows
print("Stage Analysis Daily Data Structure:")
print(stage_analysis_daily.columns.tolist())
print("\nFirst few rows:")
print(stage_analysis_daily.head())

print("\nUnique stage names:")
print(stage_analysis_daily['daily_stage_name'].unique())