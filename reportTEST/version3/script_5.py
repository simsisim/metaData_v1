# Let's create a detailed breakdown showing which specific stocks are in each industry
print("Detailed breakdown of bullish trend stocks by industry:\n")

for industry in industry_counts.index:
    stocks_in_industry = individual_bullish_stocks[individual_bullish_stocks['Industry'] == industry]['ticker'].tolist()
    print(f"{industry} ({industry_counts[industry]} stocks):")
    print(f"  - {', '.join(stocks_in_industry)}")
    print()

# Let's also save this data to a CSV for further analysis
result_data = individual_bullish_stocks[['ticker', 'Industry', 'Sector', 'daily_stage_name', 'current_price']].copy()
result_data.to_csv('bullish_trend_stocks_by_industry.csv', index=False)
print("Data saved to bullish_trend_stocks_by_industry.csv")