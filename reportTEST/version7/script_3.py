# Let's analyze volatility vs returns across different market cap ranges
# Create market cap categories based on price ranges (rough proxy)

df_calc['MarketCap_Category'] = pd.cut(df_calc['current_price'], 
                                       bins=[0, 100, 300, 600, float('inf')],
                                       labels=['Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap'])

volatility_analysis = []
for category in df_calc['MarketCap_Category'].unique():
    if pd.notna(category):
        subset = df_calc[df_calc['MarketCap_Category'] == category]
        volatility_analysis.append({
            'Category': category,
            'Avg_Return': subset['daily_year_pct_change'].mean(),
            'Avg_Volatility': subset['atr_pct'].mean(),
            'Return_StdDev': subset['daily_year_pct_change'].std(),
            'Stock_Count': len(subset),
            'Max_Return': subset['daily_year_pct_change'].max(),
            'Min_Return': subset['daily_year_pct_change'].min()
        })

df_volatility = pd.DataFrame(volatility_analysis)
print("Volatility Analysis by Market Cap Category:")
print(df_volatility.round(2))

# Prepare individual stock data for scatter plot
individual_stocks = df_calc[['ticker', 'current_price', 'daily_year_pct_change', 'atr_pct', 'MarketCap_Category']].copy()
individual_stocks = individual_stocks.dropna()

print(f"\nStock distribution by category:")
print(individual_stocks['MarketCap_Category'].value_counts())

# Convert for charting
individual_chart_data = individual_stocks.to_dict('list')
category_chart_data = df_volatility.to_dict('list')