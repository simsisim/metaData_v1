# Analyze sectors and industries
print("=== SECTOR ANALYSIS ===")
sector_counts = merged_data['Sector'].value_counts()
print(sector_counts)

print("\n=== INDUSTRY ANALYSIS ===")
industry_counts = merged_data['Industry'].value_counts()
print(industry_counts.head(10))

print(f"\nTotal unique sectors: {merged_data['Sector'].nunique()}")
print(f"Total unique industries: {merged_data['Industry'].nunique()}")

# Performance metrics analysis
print("\n=== PERFORMANCE METRICS OVERVIEW ===")
performance_cols = [
    'daily_daily_daily_1d_pct_change',
    'daily_daily_daily_5d_pct_change', 
    'daily_daily_weekly_7d_pct_change',
    'daily_daily_monthly_22d_pct_change',
    'daily_daily_quarterly_66d_pct_change',
    'daily_daily_yearly_252d_pct_change'
]

print("Performance statistics:")
print(merged_data[performance_cols].describe().round(2))