# Get the top stocks by performance and prepare data for various charts
import matplotlib.pyplot as plt

# 1. Top performers by various time periods
top_performers = daily_df.nlargest(10, 'daily_daily_yearly_252d_pct_change')[['ticker', 'current_price', 'daily_daily_yearly_252d_pct_change', 'daily_daily_monthly_22d_pct_change', 'daily_daily_weekly_7d_pct_change']]
print("Top 10 yearly performers:")
print(top_performers)

# 2. Stage analysis distribution
stage_distribution = stage_daily_df['daily_stage_name'].value_counts()
print("\n\nStage Analysis Distribution:")
print(stage_distribution)

# 3. Sector performance from index data
tech_heavy_stocks = daily_df[daily_df['SP500InformationTechnology'] == True]
finance_stocks = daily_df[daily_df['SP500Financials'] == True]

print(f"\n\nTech stocks count: {len(tech_heavy_stocks)}")
print(f"Finance stocks count: {len(finance_stocks)}")

# 4. RSI distribution for momentum analysis
rsi_data = daily_df[['ticker', 'daily_rsi_14', 'current_price']].copy()
rsi_data['rsi_category'] = pd.cut(rsi_data['daily_rsi_14'], 
                                  bins=[0, 30, 70, 100], 
                                  labels=['Oversold', 'Neutral', 'Overbought'])
rsi_distribution = rsi_data['rsi_category'].value_counts()
print("\n\nRSI Distribution:")
print(rsi_distribution)