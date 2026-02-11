# Prepare data for RSI momentum analysis chart
rsi_data_for_chart = []
for category in ['Oversold', 'Neutral', 'Overbought']:
    count = int(rsi_distribution[category]) if category in rsi_distribution else 0
    rsi_data_for_chart.append({"category": category, "count": count})

print("RSI Data for chart:")
print(rsi_data_for_chart)

# Prepare sector performance comparison
sectors_data = []
sector_columns = ['SP500InformationTechnology', 'SP500Financials', 'SP500HealthCare', 
                  'SP500ConsumerDiscretionary', 'SP500Energy', 'SP500Industrials', 'SP500Utilities']

for sector in sector_columns:
    sector_stocks = daily_df[daily_df[sector] == True]
    if len(sector_stocks) > 0:
        avg_return = sector_stocks['daily_daily_yearly_252d_pct_change'].mean()
        stock_count = len(sector_stocks)
        sector_name = sector.replace('SP500', '')
        sectors_data.append({
            "sector": sector_name, 
            "avg_yearly_return": round(avg_return, 2),
            "stock_count": stock_count
        })

print("\nSector performance data:")
print(sectors_data)