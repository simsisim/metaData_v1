# Let me check the ATR percentile distribution to understand the risk categorization better
print("ATR PERCENTILE DISTRIBUTION:")
print(f"Min: {analyzer.merged_data['atr_percentile_100'].min()}")
print(f"Max: {analyzer.merged_data['atr_percentile_100'].max()}")
print(f"Mean: {analyzer.merged_data['atr_percentile_100'].mean():.1f}")
print(f"Median: {analyzer.merged_data['atr_percentile_100'].median():.1f}")

# Let's create more appropriate risk categories
def better_risk_categorization(atr_pct):
    if atr_pct < 25:
        return 'Low Risk'
    elif atr_pct < 75:
        return 'Medium Risk' 
    else:
        return 'High Risk'

risk_data['risk_category'] = risk_data['atr_percentile_100'].apply(better_risk_categorization)

print("\nIMPROVED RISK DISTRIBUTION:")
risk_dist = risk_data['risk_category'].value_counts()
print(risk_dist)

print("\nTOP OPPORTUNITIES BY IMPROVED RISK CATEGORY:")
print("-" * 60)

# Get top picks by risk category for each timeframe
for risk_cat in ['Low Risk', 'Medium Risk', 'High Risk']:
    print(f"\n{risk_cat.upper()} STOCKS:")
    
    # Filter by risk category
    risk_tickers = risk_data[risk_data['risk_category'] == risk_cat]['ticker'].tolist()
    
    if risk_tickers:
        # Short-term top pick
        st_filtered = short_term_results[short_term_results['ticker'].isin(risk_tickers)]
        if not st_filtered.empty:
            st_pick = st_filtered.iloc[0]
            print(f"Short-term: {st_pick['ticker']} (Score: {st_pick['short_term_score']:.2f}, 5d: {st_pick['5d_change']:.1f}%)")
        
        # Medium-term top pick  
        mt_filtered = medium_term_results[medium_term_results['ticker'].isin(risk_tickers)]
        if not mt_filtered.empty:
            mt_pick = mt_filtered.iloc[0]
            print(f"Medium-term: {mt_pick['ticker']} (Score: {mt_pick['medium_term_score']:.2f}, 44d: {mt_pick['44d_change']:.1f}%)")
        
        # Long-term top pick
        lt_filtered = long_term_results[long_term_results['ticker'].isin(risk_tickers)]
        if not lt_filtered.empty:
            lt_pick = lt_filtered.iloc[0]
            print(f"Long-term: {lt_pick['ticker']} (Score: {lt_pick['long_term_score']:.2f}, 252d: {lt_pick['252d_change']:.1f}%)")
    else:
        print("No stocks in this risk category")