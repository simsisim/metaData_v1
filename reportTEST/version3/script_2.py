# Now merge the stage analysis data with the universe data to get industry information
merged_data = stage_analysis_daily.merge(universe_data, left_on='ticker', right_on='Symbol', how='left')

# Check how many stocks we successfully matched
print(f"Total stocks in stage analysis: {len(stage_analysis_daily)}")
print(f"Successfully matched stocks: {len(merged_data.dropna(subset=['Industry']))}")

# Filter for bullish trend stocks only
bullish_stocks = merged_data[merged_data['daily_stage_name'] == 'Bullish Trend']

print(f"\nTotal bullish trend stocks: {len(bullish_stocks)}")
print("\nBullish trend stocks:")
for _, stock in bullish_stocks.iterrows():
    print(f"- {stock['ticker']}: {stock['Industry']}")