# Create final summary statistics table
summary_stats = {
    'Metric': [
        'Total Stocks Analyzed',
        'Avg YTD Return (%)',
        'Median YTD Return (%)',
        'Best Performer',
        'Worst Performer',
        'Stocks in Bullish Trend',
        'Stocks in Bearish Trend',
        'Oversold Stocks (RSI < 30)',
        'Overbought Stocks (RSI > 70)',
        'Above 52-week High Range',
        'Near 52-week Low Range'
    ],
    'Value': [
        f"{daily_df.shape[0]}",
        f"{daily_df['daily_year_pct_change'].mean():.1f}%",
        f"{daily_df['daily_year_pct_change'].median():.1f}%",
        f"APP (+443.9%)",
        f"UNH (-46.4%)",
        f"{bullish_stocks} ({bullish_stocks/len(stage_daily)*100:.1f}%)",
        f"{bearish_stocks} ({bearish_stocks/len(stage_daily)*100:.1f}%)",
        f"{oversold_rsi}",
        f"{overbought_rsi}",
        f"{len(daily_df[daily_df['daily_price_position_52w'] > 0.8])}",
        f"{len(daily_df[daily_df['daily_price_position_52w'] < 0.2])}"
    ]
}

summary_df = pd.DataFrame(summary_stats)
print("=== MARKET SUMMARY STATISTICS ===")
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv('market_analysis_summary.csv', index=False)
print("\nSummary saved to market_analysis_summary.csv")