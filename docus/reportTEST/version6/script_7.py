# Save detailed analysis to CSV files for each timeframe
print("SAVING DETAILED ANALYSIS TO CSV FILES...")

# Short-term analysis
short_term_export = short_term_results[[
    'ticker', 'sector', 'current_price', 'market_cap',
    '1d_change', '3d_change', '5d_change', '7d_change',
    'rsi_14', 'macd_histogram', 'ema10_slope', 'volume_trend',
    'short_term_score'
]].round(3)

# Medium-term analysis  
medium_term_export = medium_term_results[[
    'ticker', 'sector', 'current_price', 'market_cap', 'analyst_rating',
    '14d_change', '22d_change', '44d_change', '66d_change',
    'price_vs_sma20', 'price_vs_sma50', 'price_position_52w',
    'medium_term_score'
]].round(3)

# Long-term analysis
long_term_export = long_term_results[[
    'ticker', 'sector', 'industry', 'current_price', 'market_cap', 'analyst_rating',
    '132d_change', '252d_change', 'year_change',
    'price_vs_sma200', 'sma200_slope', 'sp500_member',
    'long_term_score'
]].round(3)

# Save to CSV
short_term_export.to_csv('short_term_analysis.csv', index=False)
medium_term_export.to_csv('medium_term_analysis.csv', index=False)
long_term_export.to_csv('long_term_analysis.csv', index=False)

print("✓ short_term_analysis.csv saved")
print("✓ medium_term_analysis.csv saved") 
print("✓ long_term_analysis.csv saved")

# Create executive summary
summary_data = []
for _, row in short_term_results.head(5).iterrows():
    summary_data.append({
        'timeframe': 'Short-term',
        'rank': len(summary_data) + 1,
        'ticker': row['ticker'],
        'sector': row['sector'],
        'score': row['short_term_score'],
        'key_metric': f"{row['5d_change']:.1f}% (5d)"
    })

for _, row in medium_term_results.head(5).iterrows():
    summary_data.append({
        'timeframe': 'Medium-term', 
        'rank': len([x for x in summary_data if x['timeframe'] == 'Medium-term']) + 1,
        'ticker': row['ticker'],
        'sector': row['sector'],
        'score': row['medium_term_score'],
        'key_metric': f"{row['44d_change']:.1f}% (44d)"
    })

for _, row in long_term_results.head(5).iterrows():
    summary_data.append({
        'timeframe': 'Long-term',
        'rank': len([x for x in summary_data if x['timeframe'] == 'Long-term']) + 1,
        'ticker': row['ticker'],
        'sector': row['sector'], 
        'score': row['long_term_score'],
        'key_metric': f"{row['252d_change']:.1f}% (252d)"
    })

executive_summary = pd.DataFrame(summary_data)
executive_summary.to_csv('executive_summary.csv', index=False)
print("✓ executive_summary.csv saved")

print(f"\nFiles saved successfully! Analysis covers {len(analyzer.merged_data)} stocks.")
print("Executive Summary:")
print(executive_summary.to_string(index=False))