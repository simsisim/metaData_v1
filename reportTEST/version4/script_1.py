# Load RS data to see relative strength analysis
rs_daily_df = pd.read_csv('rs_ibd_stocks_daily_0_20250829.csv')
print("RS Daily data shape:", rs_daily_df.shape)
print("\nRS Daily data columns:")
print(rs_daily_df.columns.tolist())
print("\nFirst few rows of RS daily data:")
print(rs_daily_df.head())

# Load stage analysis data
stage_daily_df = pd.read_csv('stage_analysis_daily_0_20250829.csv')
print("\n\nStage Analysis Daily data shape:", stage_daily_df.shape)
print("\nStage Analysis columns:")
print(stage_daily_df.columns.tolist())
print("\nFirst few rows of stage analysis:")
print(stage_daily_df[['ticker', 'current_price', 'daily_stage_name', 'daily_stage_color_code']].head(10))