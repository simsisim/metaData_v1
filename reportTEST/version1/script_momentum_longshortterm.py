# Create detailed analysis tables for comprehensive understanding

# 1. Stock RS Analysis with RS Values and Percentiles
stocks_detailed = stocks_daily[['ticker', 'daily_daily_daily_1d_RS', 'daily_daily_daily_1d_RS_per',
                               'daily_daily_weekly_7d_RS', 'daily_daily_weekly_7d_RS_per',
                               'daily_daily_monthly_22d_RS', 'daily_daily_monthly_22d_RS_per',
                               'daily_daily_yearly_252d_RS', 'daily_daily_yearly_252d_RS_per']].copy()

stocks_detailed.columns = ['Ticker', '1D_RS', '1D_RS%', '7D_RS', '7D_RS%', '22D_RS', '22D_RS%', '252D_RS', '252D_RS%']
stocks_detailed = stocks_detailed.round(4)
stocks_detailed = stocks_detailed.sort_values('1D_RS%', ascending=False)

print("=== COMPREHENSIVE STOCK RS ANALYSIS ===")
print("(RS = Relative Strength ratio, RS% = Percentile ranking)")
print(stocks_detailed.to_string(index=False))

# 2. Momentum Analysis - identify stocks with consistent RS across timeframes
print("\n\n=== MOMENTUM CONSISTENCY ANALYSIS ===")
stocks_rs_summary['RS_Consistency'] = (stocks_rs_summary[['1D_RS%', '7D_RS%', '22D_RS%']].min(axis=1))
stocks_rs_summary['RS_Volatility'] = (stocks_rs_summary[['1D_RS%', '7D_RS%', '22D_RS%']].max(axis=1) - 
                                     stocks_rs_summary[['1D_RS%', '7D_RS%', '22D_RS%']].min(axis=1))

momentum_analysis = stocks_rs_summary[['Ticker', '1D_RS%', '7D_RS%', '22D_RS%', 'RS_Consistency', 'RS_Volatility']].sort_values('RS_Consistency', ascending=False)

print("Most Consistent High Performers (Min RS% across short-term timeframes):")
print(momentum_analysis.head(8).to_string(index=False))

# 3. Long-term vs Short-term RS comparison
print("\n\n=== LONG-TERM vs SHORT-TERM RS COMPARISON ===")
stocks_rs_summary['Short_Term_Avg'] = (stocks_rs_summary['1D_RS%'] + stocks_rs_summary['7D_RS%']) / 2
stocks_rs_summary['Long_Term'] = stocks_rs_summary['252D_RS%']
stocks_rs_summary['ST_vs_LT_Diff'] = stocks_rs_summary['Short_Term_Avg'] - stocks_rs_summary['Long_Term']

st_lt_comparison = stocks_rs_summary[['Ticker', 'Short_Term_Avg', 'Long_Term', 'ST_vs_LT_Diff']].sort_values('ST_vs_LT_Diff', ascending=False)
st_lt_comparison = st_lt_comparison.round(1)

print("Stocks with Strong Short-term vs Long-term RS (ST_vs_LT_Diff > 0 means better short-term):")
print(st_lt_comparison.to_string(index=False))