# Create summary statistics and insights
print("=== KEY INSIGHTS SUMMARY ===")
print(f"Date of analysis: {merged_data['date'].iloc[0]}")
print(f"Total stocks analyzed: {len(merged_data)}")

# Top performers
print("\n--- TOP PERFORMERS (1-Day) ---")
top_1d = merged_data.nlargest(5, 'daily_daily_daily_1d_pct_change')[['ticker', 'Description', 'daily_daily_daily_1d_pct_change', 'Sector']]
print(top_1d.to_string(index=False))

print("\n--- BOTTOM PERFORMERS (1-Day) ---")
bottom_1d = merged_data.nsmallest(5, 'daily_daily_daily_1d_pct_change')[['ticker', 'Description', 'daily_daily_daily_1d_pct_change', 'Sector']]
print(bottom_1d.to_string(index=False))

print("\n--- TOP PERFORMERS (1-Year) ---")
top_1y = merged_data.nlargest(5, 'daily_daily_yearly_252d_pct_change')[['ticker', 'Description', 'daily_daily_yearly_252d_pct_change', 'Sector']]
print(top_1y.to_string(index=False))

# Market cap analysis
print("\n=== MARKET CAP ANALYSIS ===")
merged_data['market_cap_billions'] = merged_data['Market capitalization'] / 1e9
print(f"Average market cap: ${merged_data['market_cap_billions'].mean():.1f}B")
print(f"Median market cap: ${merged_data['market_cap_billions'].median():.1f}B")
print(f"Largest company: {merged_data.loc[merged_data['market_cap_billions'].idxmax(), 'ticker']} (${merged_data['market_cap_billions'].max():.1f}B)")

# Create data for visualizations
print("\n=== PREPARING DATA FOR VISUALIZATIONS ===")

# Save sector performance data
sector_perf_df = sector_performance.reset_index()
sector_perf_df.to_csv('sector_performance.csv', index=False)

# Save index comparison data
sp500_comp_df = sp500_comparison.reset_index()
sp500_comp_df.to_csv('sp500_comparison.csv', index=False)

# Save top performers
top_performers_df = merged_data.nlargest(10, 'daily_daily_yearly_252d_pct_change')[
    ['ticker', 'Description', 'Sector', 'daily_daily_yearly_252d_pct_change', 'current_price', 'market_cap_billions']
]
top_performers_df.to_csv('top_performers_1year.csv', index=False)

print("Data files created for visualization!")
print(f"Sector performance file: {len(sector_perf_df)} sectors")
print(f"Top performers file: {len(top_performers_df)} stocks")