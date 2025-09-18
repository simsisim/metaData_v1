# Risk Assessment and Volatility Analysis
print("RISK AND VOLATILITY ANALYSIS")
print("="*80)

# Create risk categories based on ATR percentile and volatility measures
def categorize_risk(row):
    atr_pct = row['atr_percentile_100']
    if atr_pct < 33:
        return 'Low Risk'
    elif atr_pct < 67:
        return 'Medium Risk' 
    else:
        return 'High Risk'

# Add risk categories to our datasets
risk_data = analyzer.merged_data.copy()
risk_data['risk_category'] = risk_data.apply(categorize_risk, axis=1)

# Risk distribution
print("RISK DISTRIBUTION:")
risk_dist = risk_data['risk_category'].value_counts()
print(risk_dist)

print("\nTOP OPPORTUNITIES BY RISK CATEGORY:")
print("-" * 50)

# Get top picks by risk category for each timeframe
for risk_cat in ['Low Risk', 'Medium Risk', 'High Risk']:
    print(f"\n{risk_cat.upper()} STOCKS:")
    
    # Filter by risk category
    risk_tickers = risk_data[risk_data['risk_category'] == risk_cat]['ticker'].tolist()
    
    if risk_tickers:
        # Short-term top pick
        st_pick = short_term_results[short_term_results['ticker'].isin(risk_tickers)].iloc[0]
        print(f"Short-term: {st_pick['ticker']} (Score: {st_pick['short_term_score']:.2f}, 5d: {st_pick['5d_change']:.1f}%)")
        
        # Medium-term top pick  
        mt_pick = medium_term_results[medium_term_results['ticker'].isin(risk_tickers)].iloc[0]
        print(f"Medium-term: {mt_pick['ticker']} (Score: {mt_pick['medium_term_score']:.2f}, 44d: {mt_pick['44d_change']:.1f}%)")
        
        # Long-term top pick
        lt_pick = long_term_results[long_term_results['ticker'].isin(risk_tickers)].iloc[0]
        print(f"Long-term: {lt_pick['ticker']} (Score: {lt_pick['long_term_score']:.2f}, 252d: {lt_pick['252d_change']:.1f}%)")
    else:
        print("No stocks in this risk category")