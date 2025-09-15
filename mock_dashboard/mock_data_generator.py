#!/usr/bin/env python3
"""
Mock Data Generator for Trading Dashboard
========================================

Generates realistic random market data mimicking the real trading system outputs.
Used for dashboard design and testing without requiring actual market data.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import json

class MockDataGenerator:
    """Generate realistic mock trading data for dashboard design"""
    
    def __init__(self, output_dir="sample_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Mock ticker universe
        self.tickers = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            # Large Cap Traditional  
            'JPM', 'JNJ', 'PG', 'V', 'UNH', 'HD', 'PFE', 'KO', 'WMT', 'DIS',
            # Mid Cap Growth
            'CRM', 'ADBE', 'PYPL', 'SHOP', 'ROKU', 'ZM', 'DOCU', 'TWLO',
            # Small Cap
            'CRWD', 'SNOW', 'PLTR', 'RBLX', 'RIVN', 'LCID', 'COIN'
        ]
        
        # Index ETFs
        self.indexes = ['SPY', 'QQQ', 'IWM', '^DJI']
        
        # Sector ETFs
        self.sectors = {
            'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
            'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary', 'XLU': 'Utilities', 'XLRE': 'Real Estate'
        }
        
        # Screener types from your system
        self.screener_types = [
            'stockbee_9m_movers', 'stockbee_weekly_movers', 'stockbee_daily_gainers',
            'qullamaggie_suite', 'volume_suite_hv_absolute', 'volume_breakout',
            'gold_launch_pad', 'rti_compression', 'adl_divergence', 'guppy_alignment',
            'atr1_screener', 'momentum_screener', 'breakout_screener'
        ]
    
    def generate_market_pulse_data(self):
        """Generate mock market pulse data matching your system output"""
        
        # GMI signals for each index
        gmi_data = {}
        for index in self.indexes:
            score = random.randint(2, 4)  # Current GMI scores 
            signal = 'GREEN' if score >= 3 else 'RED' if score <= 1 else 'NEUTRAL'
            
            gmi_data[index] = {
                'gmi_score': score,
                'gmi_signal': signal,
                'components': {
                    'short_trend': random.choice([True, False]),
                    'momentum': random.choice([True, False]), 
                    'long_trend': random.choice([True, False]),
                    'price_momentum': random.choice([True, False])
                }
            }
        
        # Distribution Days analysis
        dd_count = random.randint(0, 8)
        warning_level = 'SEVERE' if dd_count >= 6 else 'CAUTION' if dd_count >= 4 else 'NONE'
        
        # Follow-Through Days
        has_recent_ftd = random.choice([True, False])
        ftd_data = None
        if has_recent_ftd:
            ftd_data = {
                'date': (datetime.now() - timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d'),
                'gain_pct': round(random.uniform(1.5, 4.0), 1),
                'days_from_bottom': random.randint(4, 12),
                'quality': random.choice(['EXCELLENT', 'GOOD', 'FAIR'])
            }
        
        # Market breadth (new highs/lows)
        total_universe = 757  # Your current universe size
        new_highs = random.randint(20, 300)
        new_lows = random.randint(10, 150)
        net_highs = new_highs - new_lows
        
        breadth_signal = 'HEALTHY' if net_highs > 50 else 'UNHEALTHY' if net_highs < -50 else 'NEUTRAL'
        
        market_pulse = {
            'timestamp': datetime.now().isoformat(),
            'gmi_analysis': gmi_data,
            'distribution_days': {
                'count': dd_count,
                'warning_level': warning_level,
                'lookback_period': 25
            },
            'follow_through_days': ftd_data,
            'market_breadth': {
                'new_highs': new_highs,
                'new_lows': new_lows, 
                'net_highs': net_highs,
                'breadth_signal': breadth_signal,
                'universe_size': total_universe
            },
            'overall_signal': random.choice(['BULLISH', 'BEARISH', 'NEUTRAL']),
            'confidence': random.choice(['HIGH', 'MEDIUM', 'LOW'])
        }
        
        # Save to file
        with open(self.output_dir / 'market_pulse_data.json', 'w') as f:
            json.dump(market_pulse, f, indent=2)
        
        return market_pulse
    
    def generate_screener_results(self):
        """Generate mock screener results matching your system format"""
        
        results = []
        
        # Generate 20-50 random screener hits
        num_results = random.randint(20, 50)
        
        for i in range(num_results):
            ticker = random.choice(self.tickers)
            screener = random.choice(self.screener_types)
            
            # Mock signal strength and metrics
            signal_strength = random.choice(['STRONG', 'MODERATE', 'WEAK'])
            score = random.uniform(5.0, 10.0) if signal_strength == 'STRONG' else \
                   random.uniform(3.0, 7.0) if signal_strength == 'MODERATE' else \
                   random.uniform(1.0, 5.0)
            
            volume = random.randint(500000, 50000000)
            price = random.uniform(10, 500)
            
            # Setup stage
            setup_stage = random.choice(['Entry', 'Watch', 'Exit', 'Hold'])
            
            result = {
                'ticker': ticker,
                'screener_type': screener,
                'signal_strength': signal_strength,
                'score': round(score, 1),
                'current_price': round(price, 2),
                'volume': volume,
                'setup_stage': setup_stage,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'screener_results.csv', index=False)
        
        return results_df
    
    def generate_sector_performance(self):
        """Generate mock sector performance data"""
        
        sector_data = []
        
        for etf, sector_name in self.sectors.items():
            # Generate realistic daily performance
            daily_change = random.uniform(-3.0, 3.0)
            weekly_change = random.uniform(-8.0, 8.0)
            monthly_change = random.uniform(-15.0, 15.0)
            
            # Mock volume and technical data
            volume = random.randint(5000000, 50000000)
            current_price = random.uniform(30, 200)
            
            # Trend status
            trend_status = random.choice(['DARK_GREEN', 'LIGHT_GREEN', 'YELLOW', 'RED'])
            ma_position = random.choice(['Above 20MA', 'Near 20MA', 'Below 20MA'])
            
            sector_data.append({
                'etf': etf,
                'sector_name': sector_name,
                'daily_change_pct': round(daily_change, 2),
                'weekly_change_pct': round(weekly_change, 2),
                'monthly_change_pct': round(monthly_change, 2),
                'current_price': round(current_price, 2),
                'volume': volume,
                'trend_status': trend_status,
                'ma_position': ma_position
            })
        
        sector_df = pd.DataFrame(sector_data)
        sector_df.to_csv(self.output_dir / 'sector_performance.csv', index=False)
        
        return sector_df
    
    def generate_index_technical_data(self):
        """Generate mock technical analysis data for major indexes"""
        
        index_data = []
        
        for index in self.indexes:
            price = random.uniform(100, 500)
            atr_pct = random.uniform(0.8, 2.5)
            
            # MA cycle information
            cycle_type = random.choice(['BULLISH', 'BEARISH'])
            cycle_days = random.randint(1, 45)
            distance_from_ma = random.uniform(-5.0, 10.0)
            
            # Technical indicators
            rsi = random.uniform(30, 70)
            macd = random.uniform(-2.0, 2.0)
            
            index_data.append({
                'index': index,
                'current_price': round(price, 2),
                'atr_pct': round(atr_pct, 2),
                'ma_status': random.choice(['Above 20MA', 'Near 20MA', 'Below 20MA']),
                'cycle_type': cycle_type,
                'cycle_days': cycle_days,
                'distance_from_ma_pct': round(distance_from_ma, 2),
                'rsi_14': round(rsi, 1),
                'macd': round(macd, 3),
                'chillax_color': random.choice(['DARK_GREEN', 'LIGHT_GREEN', 'YELLOW', 'RED'])
            })
        
        index_df = pd.DataFrame(index_data)
        index_df.to_csv(self.output_dir / 'index_technical.csv', index=False)
        
        return index_df
    
    def generate_alerts_data(self):
        """Generate mock alert data for the alerts tab"""
        
        alerts = []
        
        # High priority alerts
        high_priority_alerts = [
            "TSLA: Volume spike +340% - investigate breakout",
            "Energy sector: +1.8% rotation signal detected", 
            "NVDA: Breaking above key resistance at $425",
            "Distribution Day WARNING: 4 days detected in SPY"
        ]
        
        # Medium priority alerts  
        medium_priority_alerts = [
            "AAPL: Approaching 20MA support at $185",
            "Gold Launch Pad setup forming in META",
            "RTI compression detected in 7 stocks",
            "Volume dry-up pattern in semiconductor sector"
        ]
        
        # Generate random selection of alerts
        num_high = random.randint(1, 3)
        num_medium = random.randint(2, 5)
        
        for i in range(num_high):
            alerts.append({
                'priority': 'HIGH',
                'type': 'WARNING',
                'message': random.choice(high_priority_alerts),
                'timestamp': datetime.now().isoformat(),
                'action_required': True
            })
        
        for i in range(num_medium):
            alerts.append({
                'priority': 'MEDIUM', 
                'type': 'OPPORTUNITY',
                'message': random.choice(medium_priority_alerts),
                'timestamp': datetime.now().isoformat(),
                'action_required': False
            })
        
        alerts_df = pd.DataFrame(alerts)
        alerts_df.to_csv(self.output_dir / 'alerts_data.csv', index=False)
        
        return alerts_df
    
    def generate_top_opportunities(self):
        """Generate mock top opportunities list"""
        
        opportunities = []
        
        # Create 10-15 top opportunities
        num_opportunities = random.randint(10, 15)
        
        for i in range(num_opportunities):
            ticker = random.choice(self.tickers)
            screener = random.choice(self.screener_types)
            
            # Generate realistic metrics
            signal_strength = random.choice(['STRONG', 'MODERATE', 'WEAK'])
            score = random.uniform(7.0, 9.5) if signal_strength == 'STRONG' else \
                   random.uniform(5.0, 7.5) if signal_strength == 'MODERATE' else \
                   random.uniform(3.0, 6.0)
            
            volume = random.randint(1000000, 50000000)
            price = random.uniform(15, 450)
            
            # Entry/exit levels
            entry_level = round(price * random.uniform(1.01, 1.05), 2)
            stop_level = round(price * random.uniform(0.92, 0.98), 2)
            target_level = round(price * random.uniform(1.08, 1.25), 2)
            
            opportunities.append({
                'ticker': ticker,
                'primary_screener': screener,
                'signal_strength': signal_strength,
                'score': round(score, 1),
                'current_price': round(price, 2),
                'volume': volume,
                'setup_stage': random.choice(['Entry', 'Watch', 'Breakout', 'Follow-through']),
                'entry_level': entry_level,
                'stop_level': stop_level,
                'target_level': target_level,
                'risk_reward': round((target_level - entry_level) / (entry_level - stop_level), 1)
            })
        
        # Sort by score (best opportunities first)
        opportunities = sorted(opportunities, key=lambda x: x['score'], reverse=True)
        
        opportunities_df = pd.DataFrame(opportunities)
        opportunities_df.to_csv(self.output_dir / 'top_opportunities.csv', index=False)
        
        return opportunities_df
    
    def generate_complete_mock_dataset(self):
        """Generate complete mock dataset for dashboard"""
        
        print("🎲 Generating mock market data...")
        
        # Generate all data components
        market_pulse = self.generate_market_pulse_data()
        screener_results = self.generate_screener_results()
        sector_performance = self.generate_sector_performance()
        index_technical = self.generate_index_technical_data()
        alerts = self.generate_alerts_data()
        opportunities = self.generate_top_opportunities()
        
        # Generate summary statistics
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'market_signal': market_pulse['overall_signal'],
            'total_screener_hits': len(screener_results),
            'high_priority_alerts': len([a for a in alerts.to_dict('records') if a['priority'] == 'HIGH']),
            'top_opportunities': len(opportunities),
            'sectors_analyzed': len(sector_performance),
            'indexes_analyzed': len(index_technical)
        }
        
        with open(self.output_dir / 'data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Mock data generated:")
        print(f"  • Market signal: {market_pulse['overall_signal']}")
        print(f"  • Screener hits: {len(screener_results)}")
        print(f"  • Top opportunities: {len(opportunities)}")
        print(f"  • High priority alerts: {summary['high_priority_alerts']}")
        print(f"  • Output directory: {self.output_dir}")
        
        return {
            'market_pulse': market_pulse,
            'screener_results': screener_results,
            'sector_performance': sector_performance,
            'index_technical': index_technical,
            'alerts': alerts,
            'opportunities': opportunities,
            'summary': summary
        }

def generate_sample_scenarios():
    """Generate multiple market scenario datasets"""
    
    scenarios = ['bullish_market', 'bearish_market', 'neutral_market', 'volatile_market']
    
    for scenario in scenarios:
        print(f"\n📊 Generating {scenario} scenario...")
        
        # Create scenario-specific output directory
        scenario_dir = Path(f"sample_data/{scenario}")
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        generator = MockDataGenerator(output_dir=scenario_dir)
        
        # Adjust data generation based on scenario
        if scenario == 'bullish_market':
            # Override with more bullish signals
            pass
        elif scenario == 'bearish_market':
            # Override with more bearish signals
            pass
        
        dataset = generator.generate_complete_mock_dataset()
        
        print(f"✅ {scenario} dataset created in {scenario_dir}")

if __name__ == "__main__":
    print("🎯 Mock Dashboard Data Generator")
    print("=" * 40)
    
    # Generate default dataset
    generator = MockDataGenerator()
    dataset = generator.generate_complete_mock_dataset()
    
    # Generate multiple scenarios
    print("\n🎬 Generating scenario datasets...")
    generate_sample_scenarios()
    
    print("\n✅ All mock data generation completed!")
    print("Ready for dashboard design and testing.")