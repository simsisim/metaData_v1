# Explore the data structure
print("=== TRADING DATA SAMPLE ===")
print(trading_data.head(3))
print("\n=== UNIVERSE DATA SAMPLE ===")
print(universe_data.head(3))

# Check for common tickers
common_tickers = set(trading_data['ticker']).intersection(set(universe_data['Symbol']))
print(f"\nNumber of common tickers: {len(common_tickers)}")
print(f"Trading data has {len(trading_data['ticker'].unique())} unique tickers")
print(f"Universe data has {len(universe_data['Symbol'].unique())} unique symbols")

print("\nSample of common tickers:", list(common_tickers)[:10])