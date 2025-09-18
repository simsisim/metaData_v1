# Prepare data for relative strength analysis
rs_comparison = rs_daily_df[['ticker', 'daily_daily_yearly_252d_RS_per', 'daily_daily_monthly_22d_RS_per', 'daily_daily_weekly_7d_RS_per']].copy()

# Get top 15 stocks by yearly RS percentile
top_rs_stocks = rs_comparison.nlargest(15, 'daily_daily_yearly_252d_RS_per')
print("Top 15 stocks by yearly relative strength percentile:")
print(top_rs_stocks)

# Prepare scatter plot data for Price vs RSI
scatter_data = daily_df[['ticker', 'current_price', 'daily_rsi_14', 'daily_daily_yearly_252d_pct_change']].copy()
scatter_data = scatter_data.head(20)  # Take top 20 for readability

print("\n\nScatter plot data (Price vs RSI):")
print(scatter_data)