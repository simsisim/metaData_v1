# Create performance analysis by sector and index membership
print("=== SECTOR PERFORMANCE ANALYSIS ===")
sector_performance = merged_data.groupby('Sector')[performance_cols].mean().round(2)
print(sector_performance)

print("\n=== SP500 vs NON-SP500 PERFORMANCE ===")
sp500_comparison = merged_data.groupby('in_sp_500')[performance_cols].mean().round(2)
print(sp500_comparison)

print("\n=== NASDAQ 100 vs NON-NASDAQ 100 PERFORMANCE ===")
nasdaq100_comparison = merged_data.groupby('in_nasdaq_100')[performance_cols].mean().round(2)
print(nasdaq100_comparison)

# Technical indicators analysis
technical_cols = [
    'daily_rsi_14', 'daily_momentum_20', 'daily_macd', 'daily_macd_signal', 
    'daily_price_position_52w', 'atr_pct'
]

print("\n=== TECHNICAL INDICATORS OVERVIEW ===")
print(merged_data[technical_cols].describe().round(2))