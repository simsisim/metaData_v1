# Create comprehensive analysis summary
print("--- COMPREHENSIVE STOCK MARKET ANALYSIS SUMMARY ---")
print("\n=== MARKET OVERVIEW ===")
print(f"Total stocks analyzed: {daily_df.shape[0]}")
print(f"Analysis date: {daily_df['date'].iloc[0]}")

# Market breadth analysis
bullish_stocks = len(stage_daily[stage_daily['daily_stage_name'] == 'Bullish Trend'])
bearish_stocks = len(stage_daily[stage_daily['daily_stage_name'] == 'Bearish Trend'])
total_staged = len(stage_daily[stage_daily['daily_stage_name'] != 'Undefined'])

print(f"\nMarket Breadth (Stage Analysis):")
print(f"  Bullish Trend stocks: {bullish_stocks}")
print(f"  Bearish Trend stocks: {bearish_stocks}")
print(f"  Total classified: {total_staged}")

# Performance distribution
positive_ytd = len(daily_df[daily_df['daily_year_pct_change'] > 0])
negative_ytd = len(daily_df[daily_df['daily_year_pct_change'] < 0])

print(f"\nYTD Performance Distribution:")
print(f"  Positive YTD returns: {positive_ytd} stocks ({positive_ytd/len(daily_df)*100:.1f}%)")
print(f"  Negative YTD returns: {negative_ytd} stocks ({negative_ytd/len(daily_df)*100:.1f}%)")

# RSI analysis
oversold_rsi = len(daily_df[daily_df['daily_rsi_14'] < 30])
overbought_rsi = len(daily_df[daily_df['daily_rsi_14'] > 70])

print(f"\nRSI Analysis:")
print(f"  Oversold stocks (RSI < 30): {oversold_rsi}")
print(f"  Overbought stocks (RSI > 70): {overbought_rsi}")

# Volume trend analysis
positive_volume = len(daily_df[daily_df['daily_volume_trend'] > 0])
negative_volume = len(daily_df[daily_df['daily_volume_trend'] < 0])

print(f"\nVolume Trend:")
print(f"  Positive volume trend: {positive_volume} stocks")
print(f"  Negative volume trend: {negative_volume} stocks")