# Filter out ETFs (tickers starting with X and SPY) and focus on individual stocks
individual_bullish_stocks = bullish_stocks[~bullish_stocks['ticker'].str.startswith('X')]
individual_bullish_stocks = individual_bullish_stocks[individual_bullish_stocks['ticker'] != 'SPY']

print(f"Individual bullish trend stocks (excluding ETFs): {len(individual_bullish_stocks)}")

# Count by industry
industry_counts = individual_bullish_stocks['Industry'].value_counts()
print("\nIndustries with most bullish trend stocks:")
print(industry_counts)

# Also show by sector
sector_counts = individual_bullish_stocks['Sector'].value_counts()
print("\nSectors with most bullish trend stocks:")
print(sector_counts)