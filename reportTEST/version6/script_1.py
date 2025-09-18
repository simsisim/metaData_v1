# Create a comprehensive analysis framework
class MarketAnalysisFramework:
    def __init__(self, basic_data, universe_data):
        self.basic_data = basic_data
        self.universe_data = universe_data
        self.merged_data = self._merge_datasets()
        
    def _merge_datasets(self):
        """Merge the two datasets on ticker/symbol"""
        return pd.merge(
            self.basic_data, 
            self.universe_data, 
            left_on='ticker', 
            right_on='Symbol', 
            how='inner'
        )
    
    def short_term_analysis(self):
        """Short-term analysis (1-7 days)"""
        df = self.merged_data.copy()
        
        # Short-term momentum indicators
        short_term_signals = {
            'ticker': df['ticker'],
            'current_price': df['current_price'],
            'sector': df['Sector'],
            'market_cap': df['Market capitalization'],
            
            # Short-term price changes
            '1d_change': df['daily_daily_daily_1d_pct_change'],
            '3d_change': df['daily_daily_daily_3d_pct_change'],
            '5d_change': df['daily_daily_daily_5d_pct_change'],
            '7d_change': df['daily_daily_weekly_7d_pct_change'],
            
            # Technical indicators for short-term
            'rsi_14': df['daily_rsi_14'],
            'macd_histogram': df['daily_macd_histogram'],
            'price_vs_ema10': ((df['current_price'] - df['daily_ema10']) / df['daily_ema10'] * 100),
            'ema10_slope': df['daily_ema10slope'],
            'volume_trend': df['daily_volume_trend'],
            'atr_percentile': df['atr_percentile_100'],
            
            # Momentum and volatility
            'momentum_20': df['daily_momentum_20'],
            'candle_strength': df['daily_candle_strength'],
            'is_bullish': df['daily_is_bullish'],
            'at_20day_high': df['daily_at_20day_high'],
            'at_20day_low': df['daily_at_20day_low'],
        }
        
        short_df = pd.DataFrame(short_term_signals)
        
        # Short-term scoring
        short_df['short_term_score'] = self._calculate_short_term_score(short_df)
        
        return short_df.sort_values('short_term_score', ascending=False)
    
    def medium_term_analysis(self):
        """Medium-term analysis (2 weeks - 3 months)"""
        df = self.merged_data.copy()
        
        medium_term_signals = {
            'ticker': df['ticker'],
            'current_price': df['current_price'],
            'sector': df['Sector'],
            'market_cap': df['Market capitalization'],
            'analyst_rating': df['Analyst Rating'],
            
            # Medium-term price changes
            '14d_change': df['daily_daily_weekly_14d_pct_change'],
            '22d_change': df['daily_daily_monthly_22d_pct_change'],
            '44d_change': df['daily_daily_monthly_44d_pct_change'],
            '66d_change': df['daily_daily_quarterly_66d_pct_change'],
            
            # Trend indicators
            'price_vs_sma20': df['daily_price2_sma20pct'],
            'price_vs_sma50': df['daily_price2_sma50pct'],
            'sma20_slope': df['daily_sma20slope'],
            'sma50_slope': df['daily_sma50slope'],
            'sma10_vs_sma20': df['daily_sma10vssma20'],
            'sma20_vs_sma50': df['daily_sma20vssma50'],
            
            # Position and momentum
            'price_position_52w': df['daily_price_position_52w'],
            'trend_days_10_pct': df['trend_days_10_pct'],
            'obv_trend': df['obv_trend'],
            'perfectbullish_alignment': df['daily_perfectbullishalignment'],
            
            # Market context
            'sp500_member': df['SP500'],
            'nasdaq100_member': df['NASDAQ100'],
        }
        
        medium_df = pd.DataFrame(medium_term_signals)
        medium_df['medium_term_score'] = self._calculate_medium_term_score(medium_df)
        
        return medium_df.sort_values('medium_term_score', ascending=False)
    
    def long_term_analysis(self):
        """Long-term analysis (6 months - 1 year+)"""
        df = self.merged_data.copy()
        
        long_term_signals = {
            'ticker': df['ticker'],
            'current_price': df['current_price'],
            'sector': df['Sector'],
            'industry': df['Industry'],
            'market_cap': df['Market capitalization'],
            'analyst_rating': df['Analyst Rating'],
            
            # Long-term performance
            '132d_change': df['daily_daily_quarterly_132d_pct_change'],
            '252d_change': df['daily_daily_yearly_252d_pct_change'],
            'half_year_change': df['daily_half_year_pct_change'],
            'year_change': df['daily_year_pct_change'],
            
            # Long-term trend indicators  
            'price_vs_sma200': df['daily_price2_sma200pct'],
            'price_vs_sma250': df['daily_price2_sma250pct'],
            'price_vs_sma350': df['daily_price2_sma350pct'],
            'sma200_slope': df['daily_sma200slope'],
            'sma250_slope': df['daily_sma250slope'],
            'sma350_slope': df['daily_sma350slope'],
            'sma50_vs_sma200': df['daily_sma50vssma200'],
            
            # Fundamental context
            'directional_strength': df['directional_strength'],
            '5day_low_vs_30day_high': df['daily_5day_low_vs_30day_high'],
            
            # Index memberships (proxy for quality)
            'sp500_member': df['SP500'],
            'russell1000_member': df['Russell1000'],
            'dow_member': df['DowJonesIndustrialAverage'],
        }
        
        long_df = pd.DataFrame(long_term_signals)
        long_df['long_term_score'] = self._calculate_long_term_score(long_df)
        
        return long_df.sort_values('long_term_score', ascending=False)
    
    def _calculate_short_term_score(self, df):
        """Calculate short-term composite score"""
        score = 0
        
        # Recent momentum (40% weight)
        score += np.where(df['1d_change'] > 0, 1, -1) * 0.15
        score += np.where(df['3d_change'] > 0, 1, -1) * 0.15
        score += np.where(df['5d_change'] > 0, 1, -1) * 0.10
        
        # Technical indicators (30% weight)
        score += np.where(df['rsi_14'].between(30, 70), 0.1, -0.1)  # Not oversold/overbought
        score += np.where(df['macd_histogram'] > 0, 0.1, -0.1)
        score += np.where(df['ema10_slope'] > 0, 0.1, -0.1)
        
        # Volume and volatility (20% weight)
        score += np.where(df['volume_trend'] > 0, 0.1, -0.1)
        score += np.where(df['atr_percentile'] < 80, 0.1, -0.1)  # Moderate volatility
        
        # Price action (10% weight)
        score += np.where(df['is_bullish'], 0.05, -0.05)
        score += np.where(df['at_20day_high'], 0.05, -0.05)
        
        return score
    
    def _calculate_medium_term_score(self, df):
        """Calculate medium-term composite score"""
        score = 0
        
        # Medium-term performance (40% weight)
        score += np.where(df['22d_change'] > 0, 0.15, -0.15)
        score += np.where(df['44d_change'] > 0, 0.15, -0.15)
        score += np.where(df['66d_change'] > 0, 0.10, -0.10)
        
        # Trend indicators (35% weight)
        score += np.where(df['price_vs_sma20'] > 0, 0.1, -0.1)
        score += np.where(df['price_vs_sma50'] > 0, 0.1, -0.1)
        score += np.where(df['sma20_slope'] > 0, 0.075, -0.075)
        score += np.where(df['sma50_slope'] > 0, 0.075, -0.075)
        
        # Quality indicators (25% weight)
        score += np.where(df['price_position_52w'] > 0.5, 0.1, -0.1)
        score += np.where(df['perfectbullish_alignment'], 0.1, -0.1)
        score += np.where(df['trend_days_10_pct'] > 50, 0.05, -0.05)
        
        return score
    
    def _calculate_long_term_score(self, df):
        """Calculate long-term composite score"""
        score = 0
        
        # Long-term performance (50% weight)
        score += np.where(df['252d_change'] > 0, 0.2, -0.2)
        score += np.where(df['year_change'] > 0, 0.15, -0.15)
        score += np.where(df['132d_change'] > 0, 0.15, -0.15)
        
        # Long-term trend (30% weight)
        score += np.where(df['price_vs_sma200'] > 0, 0.1, -0.1)
        score += np.where(df['sma200_slope'] > 0, 0.1, -0.1)
        score += np.where(df['sma50_vs_sma200'], 0.1, -0.1)
        
        # Quality and fundamentals (20% weight)
        rating_score = df['analyst_rating'].map({
            'Strong buy': 0.1, 'Buy': 0.05, 'Neutral': 0, 'Sell': -0.05, 'Strong sell': -0.1
        }).fillna(0)
        score += rating_score
        
        score += np.where(df['sp500_member'], 0.05, 0)
        score += np.where(df['dow_member'], 0.05, 0)
        
        return score

# Initialize the analysis framework
analyzer = MarketAnalysisFramework(basic_calc, universe)

print("Market Analysis Framework initialized successfully!")
print(f"Analyzing {len(analyzer.merged_data)} stocks with complete data.")