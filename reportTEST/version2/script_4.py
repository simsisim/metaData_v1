# Analyze relative strength rankings for key stocks
print("--- Relative Strength Analysis for Key Stocks ---")
rs_key_stocks = rs_daily[rs_daily['ticker'].isin(key_stocks)]

for stock in key_stocks:
    if stock in rs_key_stocks['ticker'].values:
        stock_rs = rs_key_stocks[rs_key_stocks['ticker'] == stock].iloc[0]
        print(f"\n{stock} RS Rankings:")
        print(f"  1-day RS percentile: {stock_rs['daily_daily_daily_1d_RS_per']}")
        print(f"  5-day RS percentile: {stock_rs['daily_daily_daily_5d_RS_per']}")
        print(f"  Monthly RS percentile: {stock_rs['daily_daily_monthly_22d_RS_per']}")
        print(f"  Quarterly RS percentile: {stock_rs['daily_daily_quarterly_66d_RS_per']}")
        print(f"  Yearly RS percentile: {stock_rs['daily_daily_yearly_252d_RS_per']}")

# Top and bottom performers by various metrics
print("\n--- Top Performers by YTD Returns ---")
top_ytd = daily_df.nlargest(10, 'daily_year_pct_change')[['ticker', 'current_price', 'daily_year_pct_change']]
print(top_ytd)

print("\n--- Bottom Performers by YTD Returns ---")
bottom_ytd = daily_df.nsmallest(10, 'daily_year_pct_change')[['ticker', 'current_price', 'daily_year_pct_change']]
print(bottom_ytd)