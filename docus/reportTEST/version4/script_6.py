# Create a risk-return analysis using ATR and returns
risk_return_data = daily_df[['ticker', 'current_price', 'daily_daily_yearly_252d_pct_change', 'atr_pct', 'daily_rsi_14']].copy()
risk_return_data = risk_return_data.dropna()
risk_return_data = risk_return_data.head(25)  # Top 25 for readability

print("Risk-Return Analysis Data:")
print(risk_return_data[['ticker', 'daily_daily_yearly_252d_pct_change', 'atr_pct']].head(15))

# Export final summary data
summary_stats = {
    'total_stocks_analyzed': len(daily_df),
    'bullish_stage_stocks': len(stage_daily_df[stage_daily_df['daily_stage_name'] == 'Bullish Trend']),
    'avg_yearly_return': daily_df['daily_daily_yearly_252d_pct_change'].mean(),
    'median_rsi': daily_df['daily_rsi_14'].median(),
    'overbought_stocks': len(daily_df[daily_df['daily_rsi_14'] > 70]),
    'oversold_stocks': len(daily_df[daily_df['daily_rsi_14'] < 30])
}

print("\n\nMarket Summary Statistics:")
for key, value in summary_stats.items():
    print(f"{key}: {value}")
    
# Save processed data for reference
processed_data = daily_df[['ticker', 'current_price', 'daily_daily_yearly_252d_pct_change', 
                          'daily_daily_monthly_22d_pct_change', 'daily_rsi_14', 'atr_pct']].copy()
processed_data.to_csv('processed_market_analysis.csv', index=False)
print("\nProcessed data saved to 'processed_market_analysis.csv'")