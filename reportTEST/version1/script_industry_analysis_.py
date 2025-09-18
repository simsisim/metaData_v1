# Create industry-specific analysis and comparison tables

# Combine all industry data across timeframes for comprehensive view
print("=== INDUSTRY RS ANALYSIS ACROSS ALL TIMEFRAMES ===")

# Daily RS data for industries
industries_daily_clean = industries_daily[['ticker', 'daily_daily_daily_1d_RS_per', 'daily_daily_weekly_7d_RS_per', 
                                         'daily_daily_monthly_22d_RS_per', 'daily_daily_quarterly_66d_RS_per']].copy()
industries_daily_clean.columns = ['Industry', 'Daily_1D%', 'Daily_7D%', 'Daily_22D%', 'Daily_66D%']

# Weekly RS data for industries
industries_weekly_clean = industries_weekly[['ticker', 'weekly_weekly_weekly_1w_RS_per', 'weekly_weekly_weekly_3w_RS_per', 
                                           'weekly_weekly_weekly_5w_RS_per', 'weekly_weekly_weekly_10w_RS_per']].copy()
industries_weekly_clean.columns = ['Industry', 'Weekly_1W%', 'Weekly_3W%', 'Weekly_5W%', 'Weekly_10W%']

# Monthly RS data for industries  
industries_monthly_clean = industries_monthly[['ticker', 'monthly_monthly_monthly_1m_RS_per', 'monthly_monthly_monthly_3m_RS_per', 
                                             'monthly_monthly_monthly_6m_RS_per']].copy()
industries_monthly_clean.columns = ['Industry', 'Monthly_1M%', 'Monthly_3M%', 'Monthly_6M%']

# Merge all timeframes
industries_combined = industries_daily_clean.merge(industries_weekly_clean, on='Industry').merge(industries_monthly_clean, on='Industry')
print(industries_combined.to_string(index=False))

# Rank industries by their average performance across all short-term timeframes
print("\n=== INDUSTRY RANKING BY SHORT-TERM AVERAGE RS ===")
short_term_cols = ['Daily_1D%', 'Daily_7D%', 'Daily_22D%', 'Weekly_1W%', 'Weekly_3W%']
industries_combined['Short_Term_Avg'] = industries_combined[short_term_cols].mean(axis=1)
industries_combined['Long_Term_Avg'] = industries_combined[['Daily_66D%', 'Weekly_10W%', 'Monthly_6M%']].mean(axis=1)

industry_ranking = industries_combined[['Industry', 'Short_Term_Avg', 'Long_Term_Avg']].sort_values('Short_Term_Avg', ascending=False)
industry_ranking = industry_ranking.round(1)
print(industry_ranking.to_string(index=False))

# Weekly vs Monthly comparison analysis
print("\n=== WEEKLY vs MONTHLY MOMENTUM COMPARISON ===")
print("Showing 1-week vs 1-month RS percentile differences for all groups:")

# Stocks comparison
stocks_weekly_clean = stocks_weekly[['ticker', 'weekly_weekly_weekly_1w_RS_per']].copy()
stocks_monthly_clean = stocks_monthly[['ticker', 'monthly_monthly_monthly_1m_RS_per']].copy()
stocks_wm_compare = stocks_weekly_clean.merge(stocks_monthly_clean, on='ticker')
stocks_wm_compare['W_vs_M_Diff'] = stocks_wm_compare['weekly_weekly_weekly_1w_RS_per'] - stocks_wm_compare['monthly_monthly_monthly_1m_RS_per']
stocks_wm_compare = stocks_wm_compare.sort_values('W_vs_M_Diff', ascending=False)
stocks_wm_compare.columns = ['Ticker', 'Weekly_1W%', 'Monthly_1M%', 'W_vs_M_Diff']

print("\nTop Stocks - Weekly vs Monthly RS Difference:")
print(stocks_wm_compare.head(10).to_string(index=False))