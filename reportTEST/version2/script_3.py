# Analyze sector performance
print("--- Sector Performance Analysis ---")
print(rs_sectors_daily[['ticker', 'rs_1', 'rs_percentile_1']].sort_values('rs_percentile_1', ascending=False))

print("\n--- Industry Performance Analysis ---")
print(rs_industries_daily[['ticker', 'rs_1', 'rs_percentile_1']].sort_values('rs_percentile_1', ascending=False))

# Analyze stage analysis for key stocks
print("\n--- Stage Analysis for Key Stocks ---")
stage_summary = stage_daily[stage_daily['ticker'].isin(key_stocks)][['ticker', 'current_price', 'daily_stage_name', 'daily_stage_color_code']]
print(stage_summary)