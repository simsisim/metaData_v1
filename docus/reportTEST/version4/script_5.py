# Prepare data for time series performance comparison of top performers
weekly_df = pd.read_csv('basic_calculation_weekly_0_20250825.csv')
monthly_df = pd.read_csv('basic_calculation_monthly_0_20250829.csv')

# Get top 6 performers for a clean multi-line chart
top_6_tickers = ['APP', 'PLTR', 'GEV', 'AVGO', 'NFLX', 'ORCL']

performance_data = []
timeframes = ['1W', '1M', '3M', '6M', '1Y']

for ticker in top_6_tickers:
    daily_row = daily_df[daily_df['ticker'] == ticker].iloc[0] if len(daily_df[daily_df['ticker'] == ticker]) > 0 else None
    
    if daily_row is not None:
        performance_data.append({
            'ticker': ticker,
            '1W': daily_row['daily_daily_weekly_7d_pct_change'],
            '1M': daily_row['daily_daily_monthly_22d_pct_change'],
            '3M': daily_row['daily_daily_quarterly_66d_pct_change'],
            '6M': daily_row['daily_half_year_pct_change'] if pd.notna(daily_row['daily_half_year_pct_change']) else 0,
            '1Y': daily_row['daily_daily_yearly_252d_pct_change']
        })

print("Performance data for multi-timeframe analysis:")
for item in performance_data:
    print(item)