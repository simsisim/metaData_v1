# Create volatility and risk analysis
merged_data['volatility_score'] = merged_data['atr_pct']  # Average True Range as volatility proxy
merged_data['momentum_score'] = merged_data['daily_momentum_20']
merged_data['rsi_category'] = pd.cut(merged_data['daily_rsi_14'], 
                                   bins=[0, 30, 70, 100], 
                                   labels=['Oversold (<30)', 'Neutral (30-70)', 'Overbought (>70)'])

# Risk-Return analysis data
risk_return_data = merged_data[[
    'ticker', 'Description', 'Sector', 'current_price',
    'daily_daily_yearly_252d_pct_change', 'atr_pct', 'daily_rsi_14',
    'daily_price_position_52w', 'market_cap_billions', 'in_sp_500'
]].copy()

risk_return_data.to_csv('risk_return_analysis.csv', index=False)

print("=== RISK-RETURN ANALYSIS ===")
print(f"Average volatility (ATR%): {merged_data['atr_pct'].mean():.2f}%")
print(f"Most volatile stock: {merged_data.loc[merged_data['atr_pct'].idxmax(), 'ticker']} ({merged_data['atr_pct'].max():.2f}%)")
print(f"Least volatile stock: {merged_data.loc[merged_data['atr_pct'].idxmin(), 'ticker']} ({merged_data['atr_pct'].min():.2f}%)")

print("\n=== RSI DISTRIBUTION ===")
print(merged_data['rsi_category'].value_counts())

print("\n=== 52-WEEK POSITION ANALYSIS ===")
print(f"Average 52-week position: {merged_data['daily_price_position_52w'].mean():.2f}")
print(f"Stocks at 52-week highs (>0.95): {(merged_data['daily_price_position_52w'] > 0.95).sum()}")
print(f"Stocks near 52-week lows (<0.20): {(merged_data['daily_price_position_52w'] < 0.20).sum()}")

# Technical analysis summary
technical_summary = merged_data.groupby('Sector')[['daily_rsi_14', 'atr_pct', 'daily_price_position_52w']].mean().round(2)
technical_summary.to_csv('technical_analysis_by_sector.csv')
print("\n=== TECHNICAL INDICATORS BY SECTOR ===")
print(technical_summary)