# Create summary analysis of top RS performers across different timeframes

# Extract key RS percentile columns for easier analysis
stocks_rs_summary = stocks_daily[['ticker', 'daily_daily_daily_1d_RS_per', 'daily_daily_weekly_7d_RS_per', 
                                 'daily_daily_monthly_22d_RS_per', 'daily_daily_quarterly_66d_RS_per', 
                                 'daily_daily_yearly_252d_RS_per']].copy()

stocks_rs_summary.columns = ['Ticker', '1D_RS%', '7D_RS%', '22D_RS%', '66D_RS%', '252D_RS%']

# Sort by different timeframes to see leaders
print("=== TOP RS LEADERS BY TIMEFRAME ===")
print("\n--- Top 5 by 1-Day RS Percentile ---")
print(stocks_rs_summary.sort_values('1D_RS%', ascending=False).head())

print("\n--- Top 5 by Weekly (7D) RS Percentile ---")
print(stocks_rs_summary.sort_values('7D_RS%', ascending=False).head())

print("\n--- Top 5 by Monthly (22D) RS Percentile ---")
print(stocks_rs_summary.sort_values('22D_RS%', ascending=False).head())

print("\n--- Top 5 by Yearly (252D) RS Percentile ---")
print(stocks_rs_summary.sort_values('252D_RS%', ascending=False).head())

# Create sector summary
sectors_rs_summary = sectors_daily[['ticker', 'daily_daily_daily_1d_RS_per', 'daily_daily_weekly_7d_RS_per', 
                                   'daily_daily_monthly_22d_RS_per', 'daily_daily_quarterly_66d_RS_per']].copy()
sectors_rs_summary.columns = ['Sector', '1D_RS%', '7D_RS%', '22D_RS%', '66D_RS%']

print("\n=== SECTOR RS RANKINGS ===")
print(sectors_rs_summary.sort_values('1D_RS%', ascending=False))

# Create industry summary  
industries_rs_summary = industries_daily[['ticker', 'daily_daily_daily_1d_RS_per', 'daily_daily_weekly_7d_RS_per', 
                                         'daily_daily_monthly_22d_RS_per', 'daily_daily_quarterly_66d_RS_per']].copy()
industries_rs_summary.columns = ['Industry', '1D_RS%', '7D_RS%', '22D_RS%', '66D_RS%']

print("\n=== INDUSTRY RS RANKINGS ===")
print(industries_rs_summary.sort_values('1D_RS%', ascending=False))