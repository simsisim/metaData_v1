# Let's also check what other stages exist to get a complete picture
all_stages = merged_data['daily_stage_name'].value_counts()
print("All stage distributions:")
print(all_stages)

# Let's create a comprehensive breakdown by industry for all bullish-related stages
bullish_related_stages = ['Bullish Trend', 'Breakout Confirmation', 'Bullish Extended']

# Filter for stocks that are NOT ETFs
individual_stocks = merged_data[~merged_data['ticker'].str.startswith('X')]
individual_stocks = individual_stocks[individual_stocks['ticker'] != 'SPY']

# Get all potentially bullish stocks (not just "Bullish Trend")
potentially_bullish = individual_stocks[individual_stocks['daily_stage_name'].isin(['Bullish Trend', 'Breakout Confirmation'])]

print(f"\nStocks in 'Bullish Trend' or 'Breakout Confirmation' stages: {len(potentially_bullish)}")

# Count by industry for these bullish stages
industry_bullish_counts = potentially_bullish['Industry'].value_counts()
print("\nIndustries with most bullish/breakout stocks:")
print(industry_bullish_counts)