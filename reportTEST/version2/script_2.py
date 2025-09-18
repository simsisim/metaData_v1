# Get overview of stock tickers in the data
print("--- Stock Tickers Overview ---")
print("Number of unique tickers in daily data:", daily_df['ticker'].nunique())
print("Tickers:", sorted(daily_df['ticker'].unique()))

print("\n--- Sample Stock Analysis ---")
# Focus on a few key stocks for detailed analysis
key_stocks = ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'META', 'GOOGL']

for stock in key_stocks:
    if stock in daily_df['ticker'].values:
        stock_data = daily_df[daily_df['ticker'] == stock].iloc[0]
        print(f"\n{stock}:")
        print(f"  Current Price: ${stock_data['current_price']:.2f}")
        print(f"  1-day change: {stock_data['daily_daily_daily_1d_pct_change']:.2f}%")
        print(f"  5-day change: {stock_data['daily_daily_daily_5d_pct_change']:.2f}%")
        print(f"  Monthly change: {stock_data['daily_daily_monthly_22d_pct_change']:.2f}%")
        print(f"  YTD change: {stock_data['daily_year_pct_change']:.2f}%")
        print(f"  RSI: {stock_data['daily_rsi_14']:.1f}")
        print(f"  52-week position: {stock_data['daily_price_position_52w']:.1f}%")