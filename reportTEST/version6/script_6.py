# Let's look at the ATR values more closely to create proper risk categories
print("DETAILED ATR ANALYSIS:")
atr_stats = analyzer.merged_data['atr_percentile_100'].describe()
print(atr_stats)

# Create quintile-based risk categories
def quintile_risk_categorization(atr_pct):
    if atr_pct <= 0.05:  # Bottom quintile
        return 'Very Low Risk'
    elif atr_pct <= 0.15:  # Second quintile
        return 'Low Risk'
    elif atr_pct <= 0.30:  # Third quintile
        return 'Medium Risk' 
    elif atr_pct <= 0.50:  # Fourth quintile
        return 'High Risk'
    else:  # Top quintile
        return 'Very High Risk'

risk_data['risk_category'] = risk_data['atr_percentile_100'].apply(quintile_risk_categorization)

print("\nQUINTILE-BASED RISK DISTRIBUTION:")
risk_dist = risk_data['risk_category'].value_counts()
print(risk_dist)

# Create comprehensive summary report
print("\n" + "="*100)
print("COMPREHENSIVE MARKET ANALYSIS SUMMARY")
print("="*100)

# Portfolio recommendations by timeframe and risk
timeframes = ['Short-term', 'Medium-term', 'Long-term']
results_dict = {
    'Short-term': short_term_results,
    'Medium-term': medium_term_results, 
    'Long-term': long_term_results
}
score_cols = {
    'Short-term': 'short_term_score',
    'Medium-term': 'medium_term_score',
    'Long-term': 'long_term_score'
}

for timeframe in timeframes:
    print(f"\n{timeframe.upper()} RECOMMENDATIONS:")
    print("-" * 60)
    
    results = results_dict[timeframe]
    score_col = score_cols[timeframe]
    
    # Get top 3 overall
    top3 = results.head(3)
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"{i}. {row['ticker']} ({row['sector']}) - Score: {row[score_col]:.2f}")
    
    print(f"\nTop sectors for {timeframe.lower()}:")
    if timeframe == 'Short-term':
        sector_perf = short_term_results.groupby('sector')['short_term_score'].mean().sort_values(ascending=False).head(3)
    elif timeframe == 'Medium-term':
        sector_perf = medium_term_results.groupby('sector')['medium_term_score'].mean().sort_values(ascending=False).head(3)
    else:
        sector_perf = long_term_results.groupby('sector')['long_term_score'].mean().sort_values(ascending=False).head(3)
    
    for sector, score in sector_perf.items():
        print(f"- {sector}: {score:.2f}")

print(f"\nAnalysis completed for {len(analyzer.merged_data)} stocks as of {analyzer.merged_data['date'].iloc[0]}")
print("Note: All scores are normalized between -1 and 1, where 1 indicates highest opportunity")