# Let's create a dataset for sector performance analysis
sector_data = []

# Extract sector ETF data and major individual stocks
sector_etfs = {
    'Technology': ['XLK', 'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'GOOG', 'META', 'AVGO', 'AMD', 'ORCL', 'CSCO', 'ASML', 'ANET'],
    'Healthcare': ['XLV', 'LLY', 'JNJ', 'ABBV', 'AMGN', 'BSX'],
    'Financials': ['XLF', 'JPM', 'V', 'MA', 'BAC', 'C', 'NTRS'],
    'Consumer_Disc': ['XLY', 'AMZN', 'TSLA', 'HD', 'COST', 'LULU', 'NFLX'],
    'Energy': ['XLE', 'XOM', 'CVX'],
    'Industrials': ['XLI', 'GE', 'UNH'],
    'Consumer_Staples': ['XLP', 'WMT', 'PG', 'KO'],
    'Utilities': ['XLU', 'FE'],
    'Materials': ['XLB'],
    'Real_Estate': ['XLRE'],
    'Communication': ['XLC', 'TMUS']
}

for sector, tickers in sector_etfs.items():
    sector_stocks = df_calc[df_calc['ticker'].isin(tickers)]
    if not sector_stocks.empty:
        avg_return = sector_stocks['daily_year_pct_change'].mean()
        avg_volatility = sector_stocks['atr_pct'].mean()
        avg_rsi = sector_stocks['daily_rsi_14'].mean()
        stock_count = len(sector_stocks)
        
        sector_data.append({
            'Sector': sector,
            'Avg_Annual_Return': avg_return,
            'Avg_Volatility': avg_volatility,
            'Avg_RSI': avg_rsi,
            'Stock_Count': stock_count
        })

# Convert to DataFrame
df_sectors = pd.DataFrame(sector_data)
print("Sector Performance Analysis:")
print(df_sectors.round(2))

# Save for chart creation
sector_chart_data = df_sectors.to_dict('list')
print(f"\nSector chart data prepared with {len(df_sectors)} sectors")