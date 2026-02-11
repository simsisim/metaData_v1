# Create final summary tables and key insights

print("=== KEY MARKET INSIGHTS FROM RS ANALYSIS ===")
print()

# 1. Current Market Leaders by Category
print("📊 CURRENT MARKET LEADERS (1-Day RS Percentile)")
print("-" * 50)
print("STOCKS:")
top_stocks = stocks_rs_summary.nlargest(3, '1D_RS%')[['Ticker', '1D_RS%']]
for _, row in top_stocks.iterrows():
    print(f"  • {row['Ticker']}: {row['1D_RS%']}th percentile")

print("\nSECTORS:")
top_sectors = sectors_rs_summary.nlargest(2, '1D_RS%')[['Sector', '1D_RS%']]
for _, row in top_sectors.iterrows():
    print(f"  • {row['Sector']}: {row['1D_RS%']}th percentile")

print("\nINDUSTRIES:")
top_industries = industries_rs_summary.nlargest(2, '1D_RS%')[['Industry', '1D_RS%']]
for _, row in top_industries.iterrows():
    print(f"  • {row['Industry']}: {row['1D_RS%']}th percentile")

# 2. Momentum Patterns
print("\n\n🔄 MOMENTUM PATTERN ANALYSIS")
print("-" * 50)

# Consistent performers
consistent_stocks = momentum_analysis.head(3)
print("MOST CONSISTENT SHORT-TERM PERFORMERS:")
for _, row in consistent_stocks.iterrows():
    print(f"  • {row['Ticker']}: Min RS {row['RS_Consistency']}%, Volatility {row['RS_Volatility']}%")

# Momentum divergence stocks
print("\nMOMENTUM DIVERGENCE HIGHLIGHTS:")
print(f"  • Strongest Short-term Momentum: {st_lt_comparison.iloc[0]['Ticker']} (+{st_lt_comparison.iloc[0]['ST_vs_LT_Diff']:.1f}% vs long-term)")
print(f"  • Strongest Long-term Momentum: {st_lt_comparison.iloc[-1]['Ticker']} ({st_lt_comparison.iloc[-1]['ST_vs_LT_Diff']:.1f}% vs short-term)")

# 3. Sector Rotation Analysis
print("\n\n🔄 SECTOR ROTATION SIGNALS")
print("-" * 50)
finance_row = sectors_rs_summary[sectors_rs_summary['Sector'] == 'Finance'].iloc[0]
tech_row = sectors_rs_summary[sectors_rs_summary['Sector'] == 'Technology services'].iloc[0]

print(f"Finance Sector: Strong short-term ({finance_row['1D_RS%']}%, {finance_row['7D_RS%']}%, {finance_row['22D_RS%']}%) → Possible rotation INTO finance")
print(f"Technology Services: Improving longer-term ({tech_row['22D_RS%']}%, {tech_row['66D_RS%']}%) → Building momentum")

# 4. Industry Leadership
print(f"\nInternet Software/Services: Consistent leader across timeframes (avg {industry_ranking.iloc[0]['Short_Term_Avg']:.1f}%)")
print(f"Semiconductors: Long-term strength ({industry_ranking[industry_ranking['Industry']=='Semiconductors']['Long_Term_Avg'].iloc[0]:.1f}%) vs short-term weakness")

print("\n\n📈 TOP ACTIONABLE INSIGHTS")
print("-" * 50)
print("1. WATCH LIST - Stocks with strong consistent RS:")
watch_list = momentum_analysis.head(5)['Ticker'].tolist()
print(f"   {', '.join(watch_list)}")

print("\n2. MOMENTUM PLAYS - Strong short-term vs long-term:")
momentum_plays = st_lt_comparison.head(3)['Ticker'].tolist()  
print(f"   {', '.join(momentum_plays)}")

print("\n3. VALUE/TURNAROUND - Strong long-term, weak short-term:")
turnaround_plays = st_lt_comparison.tail(3)['Ticker'].tolist()
print(f"   {', '.join(reversed(turnaround_plays))}")

print("\n4. SECTOR FOCUS - Current leadership:")
print("   • Finance: Short-term leader")
print("   • Internet Software: Consistent across timeframes") 
print("   • Semiconductors: Long-term value, potential turnaround")

# Save comprehensive analysis to CSV
final_analysis = pd.merge(stocks_rs_summary, st_lt_comparison[['Ticker', 'ST_vs_LT_Diff']], on='Ticker')
final_analysis = final_analysis.merge(momentum_analysis[['Ticker', 'RS_Consistency', 'RS_Volatility']], on='Ticker')
final_analysis = final_analysis.round(1)

final_analysis.to_csv('comprehensive_rs_analysis.csv', index=False)
print(f"\n📁 Comprehensive analysis saved to 'comprehensive_rs_analysis.csv'")
print(f"   Contains {len(final_analysis)} stocks with complete RS metrics across all timeframes")