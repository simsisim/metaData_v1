# Create time-based performance analysis
time_periods = {
    '1_Day': 'daily_daily_daily_1d_pct_change',
    '5_Day': 'daily_daily_daily_5d_pct_change', 
    '7_Day': 'daily_daily_weekly_7d_pct_change',
    '1_Month': 'daily_daily_monthly_22d_pct_change',
    '3_Month': 'daily_daily_quarterly_66d_pct_change',
    '1_Year': 'daily_daily_yearly_252d_pct_change'
}

# Performance across different time periods
time_performance = []
for period, col in time_periods.items():
    time_performance.append({
        'Period': period,
        'Mean_Return': merged_data[col].mean(),
        'Median_Return': merged_data[col].median(),
        'Std_Dev': merged_data[col].std(),
        'Min_Return': merged_data[col].min(),
        'Max_Return': merged_data[col].max(),
        'Positive_Count': (merged_data[col] > 0).sum(),
        'Negative_Count': (merged_data[col] < 0).sum()
    })

time_performance_df = pd.DataFrame(time_performance)
time_performance_df.to_csv('time_performance_analysis.csv', index=False)

print("=== TIME-BASED PERFORMANCE ANALYSIS ===")
print(time_performance_df.round(2))

# Create expandable analysis framework
print("\n=== CREATING EXPANDABLE ANALYSIS FRAMEWORK ===")

analysis_config = {
    'data_files': {
        'trading_data': 'basic_calculation_daily_*.csv',
        'universe_data': 'tradingview_universe.csv'
    },
    'key_metrics': {
        'performance_columns': list(time_periods.values()),
        'technical_indicators': technical_cols,
        'risk_metrics': ['atr_pct', 'daily_rsi_14', 'daily_price_position_52w'],
        'index_flags': [f'in_{idx.replace(" ", "_").replace("&", "").lower()}' for idx in major_indices]
    },
    'analysis_outputs': [
        'sector_performance.csv',
        'index_analysis.csv', 
        'risk_return_analysis.csv',
        'technical_analysis_by_sector.csv',
        'time_performance_analysis.csv'
    ],
    'visualizations': [
        'sector_performance_chart',
        'market_cap_performance_scatter', 
        'index_performance_comparison',
        'risk_return_scatter'
    ]
}

# Save configuration for future use
import json
with open('analysis_config.json', 'w') as f:
    json.dump(analysis_config, f, indent=2)

print("Analysis framework configuration saved!")