# Generate all three analyses
print("Generating Short-Term Analysis...")
short_term_results = analyzer.short_term_analysis()

print("Generating Medium-Term Analysis...")  
medium_term_results = analyzer.medium_term_analysis()

print("Generating Long-Term Analysis...")
long_term_results = analyzer.long_term_analysis()

# Display top performers in each timeframe
print("\n" + "="*80)
print("TOP 10 SHORT-TERM OPPORTUNITIES (1-7 days)")
print("="*80)
st_top10 = short_term_results.head(10)[['ticker', 'sector', 'current_price', '1d_change', '3d_change', '5d_change', 'rsi_14', 'short_term_score']]
print(st_top10.to_string(index=False))

print("\n" + "="*80)
print("TOP 10 MEDIUM-TERM OPPORTUNITIES (2 weeks - 3 months)")
print("="*80)
mt_top10 = medium_term_results.head(10)[['ticker', 'sector', 'analyst_rating', '22d_change', '44d_change', 'price_position_52w', 'medium_term_score']]
print(mt_top10.to_string(index=False))

print("\n" + "="*80)
print("TOP 10 LONG-TERM OPPORTUNITIES (6 months - 1 year+)")
print("="*80)
lt_top10 = long_term_results.head(10)[['ticker', 'sector', 'analyst_rating', '252d_change', 'year_change', 'price_vs_sma200', 'long_term_score']]
print(lt_top10.to_string(index=False))