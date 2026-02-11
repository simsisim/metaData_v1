# Let's create a momentum and trend analysis chart
# Focus on top performers and their technical indicators

# Get top and bottom performers
top_performers = df_calc.nlargest(15, 'daily_year_pct_change')[['ticker', 'daily_year_pct_change', 'daily_rsi_14', 'daily_momentum_20', 'daily_price_position_52w']]
bottom_performers = df_calc.nsmallest(10, 'daily_year_pct_change')[['ticker', 'daily_year_pct_change', 'daily_rsi_14', 'daily_momentum_20', 'daily_price_position_52w']]

print("Top 15 Performers by Annual Return:")
print(top_performers.round(2))

print("\nBottom 10 Performers by Annual Return:")
print(bottom_performers.round(2))

# Combine for charting
momentum_stocks = pd.concat([top_performers, bottom_performers])
momentum_stocks['Performance_Category'] = ['Top Performer'] * len(top_performers) + ['Bottom Performer'] * len(bottom_performers)

print(f"\nMomentum analysis prepared with {len(momentum_stocks)} stocks")

# Prepare data for chart
momentum_chart_data = momentum_stocks.to_dict('list')