# Let's create a technical analysis dashboard showing key indicators
# Focus on momentum, trend, and volume indicators

# Select key technical indicators for dashboard
tech_indicators = df_calc[['ticker', 'current_price', 'daily_year_pct_change', 
                           'daily_rsi_14', 'daily_macd', 'daily_momentum_20',
                           'daily_price_position_52w', 'atr_pct', 'directional_strength']].copy()

# Create technical signal categories
tech_indicators['RSI_Signal'] = pd.cut(tech_indicators['daily_rsi_14'], 
                                       bins=[0, 30, 70, 100], 
                                       labels=['Oversold', 'Neutral', 'Overbought'])

tech_indicators['MACD_Signal'] = tech_indicators['daily_macd'].apply(lambda x: 'Bullish' if x > 0 else 'Bearish')

tech_indicators['Momentum_Signal'] = tech_indicators['daily_momentum_20'].apply(
    lambda x: 'Strong Positive' if x > 5 else 'Positive' if x > 0 else 'Negative' if x > -5 else 'Strong Negative'
)

# Summary statistics
print("Technical Analysis Summary:")
print("\nRSI Distribution:")
print(tech_indicators['RSI_Signal'].value_counts())
print("\nMACD Signal Distribution:")
print(tech_indicators['MACD_Signal'].value_counts())
print("\nMomentum Signal Distribution:")
print(tech_indicators['Momentum_Signal'].value_counts())

# Get stocks with interesting technical patterns
interesting_stocks = tech_indicators[
    (tech_indicators['daily_rsi_14'] < 30) |  # Oversold
    (tech_indicators['daily_rsi_14'] > 70) |  # Overbought
    (tech_indicators['daily_momentum_20'].abs() > 10) |  # High momentum
    (tech_indicators['daily_price_position_52w'] < 0.3) |  # Near 52-week low
    (tech_indicators['daily_price_position_52w'] > 0.9)    # Near 52-week high
]

print(f"\nStocks with notable technical patterns: {len(interesting_stocks)}")
print(interesting_stocks[['ticker', 'daily_rsi_14', 'daily_momentum_20', 'daily_price_position_52w']].head(10).round(2))

# Prepare data for technical dashboard
tech_dashboard_data = interesting_stocks.to_dict('list')