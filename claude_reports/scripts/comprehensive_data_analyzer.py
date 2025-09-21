#!/usr/bin/env python3
"""
Comprehensive Multi-Data Source Market Analysis Engine
====================================================

This engine leverages ALL available data sources to create the most
comprehensive market analysis possible:

Data Sources:
- Basic Calculations: Technical indicators, performance metrics
- Market Breadth: NYSE, NASDAQ100, S&P500 breadth analysis
- Relative Strength: IBD-style RS across timeframes
- Percentile Rankings: Performance percentile distributions
- Stage Analysis: Market stage classification
- Ticker Universes: 269 different universe classifications

Creates advanced visualizations and insights by combining all data sources.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
import base64
import json

class ComprehensiveMarketAnalyzer:
    """
    Advanced multi-source market analysis engine
    """

    def __init__(self, results_directory):
        """
        Initialize comprehensive analyzer with all data sources
        """
        self.results_dir = Path(results_directory)
        self.output_dir = Path(__file__).parent.parent / "outputs" / "comprehensive"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("🔍 Initializing Comprehensive Market Analyzer...")
        print(f"📁 Results directory: {self.results_dir}")

        # Initialize data containers
        self.data_inventory = {}
        self.analysis_results = {}

        # Load all data sources
        self.discover_data_sources()
        self.load_core_datasets()

    def discover_data_sources(self):
        """
        Discover and catalog all available data sources
        """
        print("\n📊 Discovering available data sources...")

        data_categories = [
            'basic_calculation',
            'market_breadth',
            'rs',
            'per',
            'stage_analysis',
            'ticker_universes',
            'post_process'
        ]

        for category in data_categories:
            category_path = self.results_dir / category
            if category_path.exists():
                csv_files = list(category_path.glob("*.csv"))
                png_files = list(category_path.glob("*.png"))

                self.data_inventory[category] = {
                    'csv_files': csv_files,
                    'png_files': png_files,
                    'count': len(csv_files)
                }

                print(f"  📂 {category}: {len(csv_files)} CSV files, {len(png_files)} charts")

        total_files = sum(cat['count'] for cat in self.data_inventory.values())
        print(f"📈 Total discovered: {total_files} data files")

    def load_core_datasets(self):
        """
        Load the main datasets for analysis
        """
        print("\n📥 Loading core datasets...")

        # Basic Calculations - Technical indicators
        basic_calc_files = self.data_inventory.get('basic_calculation', {}).get('csv_files', [])
        if basic_calc_files:
            self.basic_calculations = pd.read_csv(basic_calc_files[0])
            print(f"✅ Basic calculations: {len(self.basic_calculations)} stocks")

        # Market Breadth - Multiple indices
        breadth_files = self.data_inventory.get('market_breadth', {}).get('csv_files', [])
        self.market_breadth = {}
        for file in breadth_files:
            if 'SP500' in file.name:
                self.market_breadth['SP500'] = pd.read_csv(file)
            elif 'NASDAQ100' in file.name:
                self.market_breadth['NASDAQ100'] = pd.read_csv(file)
            elif 'NYSE' in file.name:
                self.market_breadth['NYSE'] = pd.read_csv(file)

        print(f"✅ Market breadth: {len(self.market_breadth)} indices")

        # Relative Strength
        rs_files = self.data_inventory.get('rs', {}).get('csv_files', [])
        self.rs_data = {}
        for file in rs_files:
            if 'stocks' in file.name:
                self.rs_data['stocks'] = pd.read_csv(file)
            elif 'sectors' in file.name:
                self.rs_data['sectors'] = pd.read_csv(file)
            elif 'industries' in file.name:
                self.rs_data['industries'] = pd.read_csv(file)

        print(f"✅ Relative strength: {len(self.rs_data)} levels")

        # Stage Analysis
        stage_files = self.data_inventory.get('stage_analysis', {}).get('csv_files', [])
        if stage_files:
            self.stage_analysis = pd.read_csv(stage_files[0])
            print(f"✅ Stage analysis: {len(self.stage_analysis)} stocks")

        # Ticker Universes
        universe_files = self.data_inventory.get('ticker_universes', {}).get('csv_files', [])
        self.ticker_universes = {}
        for file in universe_files[:10]:  # Load top 10 universes
            universe_name = file.stem.replace('ticker_universe_', '')
            self.ticker_universes[universe_name] = pd.read_csv(file)

        print(f"✅ Ticker universes: {len(self.ticker_universes)} universes loaded")

    def create_market_overview_dashboard(self):
        """
        Create comprehensive market overview combining all data sources
        """
        print("\n📊 Creating comprehensive market overview dashboard...")

        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Market Breadth Overview', 'Sector RS Performance', 'Stage Distribution',
                'Technical Indicators', 'Universe Performance', 'Risk Analysis',
                'Momentum Leaders', 'Breadth vs Performance', 'Market Regime'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'heatmap'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'indicator'}]
            ]
        )

        # 1. Market Breadth Overview
        if self.market_breadth:
            breadth_summary = {}
            for index, df in self.market_breadth.items():
                if 'advance_decline_ratio' in df.columns:
                    breadth_summary[index] = df['advance_decline_ratio'].iloc[-1]

            fig.add_trace(
                go.Bar(x=list(breadth_summary.keys()), y=list(breadth_summary.values()),
                       name='A/D Ratio', marker_color='lightblue'),
                row=1, col=1
            )

        # 2. Sector RS Performance
        if 'sectors' in self.rs_data:
            sector_rs = self.rs_data['sectors']
            quarterly_col = None
            for col in sector_rs.columns:
                if 'quarterly' in col and 'rs_vs_QQQ' in col:
                    quarterly_col = col
                    break

            if quarterly_col:
                fig.add_trace(
                    go.Scatter(x=sector_rs['ticker'], y=sector_rs[quarterly_col],
                               mode='markers+lines', name='Sector RS',
                               marker=dict(size=12, color=sector_rs[quarterly_col], colorscale='RdYlGn')),
                    row=1, col=2
                )

        # 3. Stage Distribution
        if hasattr(self, 'stage_analysis'):
            stage_counts = self.stage_analysis['daily_sa_name'].value_counts()
            fig.add_trace(
                go.Pie(labels=stage_counts.index, values=stage_counts.values,
                       name='Stage Distribution'),
                row=1, col=3
            )

        # 4. Technical Indicators Heatmap
        if hasattr(self, 'basic_calculations'):
            # Select key technical indicators
            tech_cols = [col for col in self.basic_calculations.columns
                        if any(x in col.lower() for x in ['rsi', 'sma', 'ema', 'momentum'])][:10]

            if tech_cols:
                tech_data = self.basic_calculations[tech_cols].corr()
                fig.add_trace(
                    go.Heatmap(z=tech_data.values, x=tech_data.columns, y=tech_data.index,
                               colorscale='RdBu', name='Indicator Correlation'),
                    row=2, col=1
                )

        # 5. Universe Performance Comparison
        if self.ticker_universes:
            universe_performance = {}
            for name, df in list(self.ticker_universes.items())[:5]:
                if hasattr(self, 'basic_calculations'):
                    universe_tickers = df['ticker'].values if 'ticker' in df.columns else []
                    matching_data = self.basic_calculations[
                        self.basic_calculations['ticker'].isin(universe_tickers)
                    ]
                    if not matching_data.empty and 'returns_1d' in matching_data.columns:
                        universe_performance[name] = matching_data['returns_1d'].mean()

            if universe_performance:
                fig.add_trace(
                    go.Bar(x=list(universe_performance.keys()),
                           y=list(universe_performance.values()),
                           name='Universe Performance', marker_color='lightgreen'),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Market Intelligence Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=False
        )

        # Save dashboard
        dashboard_file = self.output_dir / "comprehensive_market_dashboard.html"
        fig.write_html(dashboard_file)
        print(f"✅ Comprehensive dashboard saved: {dashboard_file}")

        return dashboard_file

    def analyze_cross_data_correlations(self):
        """
        Analyze correlations across different data sources
        """
        print("\n🔗 Analyzing cross-data correlations...")

        correlation_insights = {}

        # Merge data sources by ticker
        if hasattr(self, 'basic_calculations') and hasattr(self, 'rs_data') and 'stocks' in self.rs_data:

            # Merge basic calculations with RS data
            rs_stocks = self.rs_data['stocks']
            merged_data = pd.merge(
                self.basic_calculations[['ticker', 'current_price', 'daily_daily_daily_1d_pct_change', 'daily_avg_volume_20']],
                rs_stocks[['ticker', 'daily_daily_daily_1d_rs_vs_QQQ', 'daily_daily_yearly_252d_rs_vs_QQQ']],
                on='ticker',
                how='inner'
            )

            if not merged_data.empty:
                # Calculate correlations
                numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
                corr_matrix = merged_data[numeric_cols].corr()

                # Create correlation heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                           square=True, fmt='.2f')
                plt.title('Cross-Data Source Correlations', fontsize=16, fontweight='bold')
                plt.tight_layout()

                correlation_file = self.output_dir / "cross_data_correlations.png"
                plt.savefig(correlation_file, dpi=300, bbox_inches='tight')
                plt.close()

                correlation_insights['file'] = correlation_file
                correlation_insights['strong_correlations'] = []

                # Find strong correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            correlation_insights['strong_correlations'].append({
                                'var1': corr_matrix.columns[i],
                                'var2': corr_matrix.columns[j],
                                'correlation': corr_value
                            })

                print(f"✅ Cross-correlations analyzed: {len(correlation_insights['strong_correlations'])} strong correlations found")

        return correlation_insights

    def create_market_regime_analysis(self):
        """
        Comprehensive market regime analysis using all data sources
        """
        print("\n🎭 Creating market regime analysis...")

        regime_data = {}

        # Market breadth regime
        if self.market_breadth:
            breadth_signals = {}
            for index, df in self.market_breadth.items():
                if 'advance_decline_ratio' in df.columns:
                    recent_ad = df['advance_decline_ratio'].tail(10).mean()
                    if recent_ad > 1.5:
                        breadth_signals[index] = 'Bullish'
                    elif recent_ad > 0.8:
                        breadth_signals[index] = 'Neutral'
                    else:
                        breadth_signals[index] = 'Bearish'

            regime_data['breadth'] = breadth_signals

        # RS regime analysis
        if 'stocks' in self.rs_data:
            rs_stocks = self.rs_data['stocks']
            rs_columns = [col for col in rs_stocks.columns if '_rs_vs_QQQ' in col]
            if rs_columns:
                avg_rs = rs_stocks[rs_columns].mean().mean()
                outperformers_pct = (rs_stocks[rs_columns].mean(axis=1) > 1.0).mean() * 100

                if avg_rs > 1.05 and outperformers_pct > 60:
                    rs_regime = 'Strong Bull Market'
                elif avg_rs > 1.0 and outperformers_pct > 50:
                    rs_regime = 'Bull Market'
                elif avg_rs > 0.95:
                    rs_regime = 'Sideways Market'
                else:
                    rs_regime = 'Bear Market'

                regime_data['relative_strength'] = {
                    'regime': rs_regime,
                    'avg_rs': avg_rs,
                    'outperformers_pct': outperformers_pct
                }

        # Stage analysis regime
        if hasattr(self, 'stage_analysis'):
            stage_distribution = self.stage_analysis['daily_sa_name'].value_counts(normalize=True) * 100

            if 'Stage 2' in stage_distribution and stage_distribution['Stage 2'] > 40:
                stage_regime = 'Growth Phase'
            elif 'Stage 4' in stage_distribution and stage_distribution['Stage 4'] > 40:
                stage_regime = 'Decline Phase'
            else:
                stage_regime = 'Transition Phase'

            regime_data['stage_analysis'] = {
                'regime': stage_regime,
                'distribution': stage_distribution.to_dict()
            }

        # Create regime visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Market Breadth Signals', 'RS Performance Distribution',
                          'Stage Analysis', 'Regime Summary'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'pie'}, {'type': 'indicator'}]]
        )

        # Market breadth signals
        if 'breadth' in regime_data:
            breadth_data = regime_data['breadth']
            colors = ['green' if v == 'Bullish' else 'red' if v == 'Bearish' else 'yellow'
                     for v in breadth_data.values()]

            fig.add_trace(
                go.Bar(x=list(breadth_data.keys()), y=[1]*len(breadth_data),
                       marker_color=colors, name='Breadth Signals'),
                row=1, col=1
            )

        # RS distribution
        if 'relative_strength' in regime_data and 'stocks' in self.rs_data:
            rs_values = self.rs_data['stocks'][rs_columns].mean(axis=1)
            fig.add_trace(
                go.Histogram(x=rs_values, nbinsx=30, name='RS Distribution'),
                row=1, col=2
            )

        # Stage distribution
        if 'stage_analysis' in regime_data:
            stage_dist = regime_data['stage_analysis']['distribution']
            fig.add_trace(
                go.Pie(labels=list(stage_dist.keys()), values=list(stage_dist.values()),
                       name='Stage Distribution'),
                row=2, col=1
            )

        # Overall regime indicator
        overall_regime = "Mixed Signals"
        if 'relative_strength' in regime_data:
            overall_regime = regime_data['relative_strength']['regime']

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=regime_data.get('relative_strength', {}).get('avg_rs', 1.0),
                title={'text': f"Market Regime: {overall_regime}"},
                gauge={
                    'axis': {'range': [0.8, 1.2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0.8, 0.95], 'color': "lightgray"},
                        {'range': [0.95, 1.05], 'color': "gray"},
                        {'range': [1.05, 1.2], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Market Regime Analysis")

        regime_file = self.output_dir / "market_regime_analysis.html"
        fig.write_html(regime_file)

        print(f"✅ Market regime analysis saved: {regime_file}")
        return regime_data, regime_file

    def create_universe_performance_analysis(self):
        """
        Analyze performance across different ticker universes
        """
        print("\n🌌 Creating universe performance analysis...")

        universe_analysis = {}

        if self.ticker_universes and hasattr(self, 'basic_calculations'):

            performance_data = []

            for universe_name, universe_df in list(self.ticker_universes.items())[:15]:  # Top 15 universes
                if 'ticker' in universe_df.columns:
                    universe_tickers = universe_df['ticker'].values

                    # Get performance data for universe tickers
                    universe_performance = self.basic_calculations[
                        self.basic_calculations['ticker'].isin(universe_tickers)
                    ]

                    if not universe_performance.empty:
                        perf_metrics = {}

                        # Calculate key metrics
                        if 'returns_1d' in universe_performance.columns:
                            perf_metrics['avg_return_1d'] = universe_performance['returns_1d'].mean()
                            perf_metrics['volatility_1d'] = universe_performance['returns_1d'].std()

                        if 'current_price' in universe_performance.columns:
                            perf_metrics['avg_price'] = universe_performance['current_price'].mean()

                        if 'volume' in universe_performance.columns:
                            perf_metrics['avg_volume'] = universe_performance['volume'].mean()

                        perf_metrics['universe'] = universe_name
                        perf_metrics['count'] = len(universe_performance)

                        performance_data.append(perf_metrics)

            if performance_data:
                universe_df = pd.DataFrame(performance_data)

                # Create universe performance visualization
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Return vs Risk', 'Universe Sizes', 'Average Returns', 'Performance Distribution')
                )

                # Risk-return scatter
                if 'avg_return_1d' in universe_df.columns and 'volatility_1d' in universe_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=universe_df['volatility_1d'],
                            y=universe_df['avg_return_1d'],
                            mode='markers+text',
                            text=universe_df['universe'],
                            textposition="top center",
                            marker=dict(size=universe_df['count']/10, color=universe_df['avg_return_1d'],
                                       colorscale='RdYlGn', showscale=True),
                            name='Universe Performance'
                        ),
                        row=1, col=1
                    )

                # Universe sizes
                fig.add_trace(
                    go.Bar(x=universe_df['universe'], y=universe_df['count'],
                           name='Universe Size', marker_color='lightblue'),
                    row=1, col=2
                )

                # Average returns
                if 'avg_return_1d' in universe_df.columns:
                    colors = ['green' if x > 0 else 'red' for x in universe_df['avg_return_1d']]
                    fig.add_trace(
                        go.Bar(x=universe_df['universe'], y=universe_df['avg_return_1d'],
                               marker_color=colors, name='Avg Returns'),
                        row=2, col=1
                    )

                # Performance distribution
                if 'avg_return_1d' in universe_df.columns:
                    fig.add_trace(
                        go.Histogram(x=universe_df['avg_return_1d'], nbinsx=20,
                                   name='Return Distribution'),
                        row=2, col=2
                    )

                fig.update_layout(height=800, title_text="Universe Performance Analysis")
                fig.update_xaxes(tickangle=45)

                universe_file = self.output_dir / "universe_performance_analysis.html"
                fig.write_html(universe_file)

                universe_analysis['data'] = universe_df
                universe_analysis['file'] = universe_file

                print(f"✅ Universe analysis saved: {universe_file}")

        return universe_analysis

    def generate_comprehensive_insights(self):
        """
        Generate comprehensive insights from all data sources
        """
        print("\n💡 Generating comprehensive market insights...")

        insights = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_sources': len(self.data_inventory),
            'total_files': sum(cat['count'] for cat in self.data_inventory.values()),
            'insights': []
        }

        # Market breadth insights
        if self.market_breadth:
            breadth_summary = {}
            for index, df in self.market_breadth.items():
                if 'advance_decline_ratio' in df.columns:
                    recent_ad = df['advance_decline_ratio'].tail(5).mean()
                    breadth_summary[index] = recent_ad

            if breadth_summary:
                best_breadth = max(breadth_summary.items(), key=lambda x: x[1])
                worst_breadth = min(breadth_summary.items(), key=lambda x: x[1])

                insights['insights'].extend([
                    f"📊 Best market breadth: {best_breadth[0]} (A/D: {best_breadth[1]:.2f})",
                    f"📉 Weakest breadth: {worst_breadth[0]} (A/D: {worst_breadth[1]:.2f})"
                ])
            else:
                insights['insights'].append("📊 Market breadth data not available for analysis")

        # Relative strength insights
        if 'stocks' in self.rs_data:
            rs_stocks = self.rs_data['stocks']
            rs_columns = [col for col in rs_stocks.columns if '_rs_vs_QQQ' in col]
            if rs_columns:
                avg_rs = rs_stocks[rs_columns].mean(axis=1)
                top_rs_stock = rs_stocks.loc[avg_rs.idxmax()]
                outperformers = (avg_rs > 1.0).sum()

                insights['insights'].extend([
                    f"🚀 Top RS performer: {top_rs_stock['ticker']} (Avg RS: {avg_rs.max():.3f})",
                    f"📈 Market leadership: {outperformers}/{len(rs_stocks)} stocks outperforming ({outperformers/len(rs_stocks)*100:.1f}%)"
                ])

        # Stage analysis insights
        if hasattr(self, 'stage_analysis'):
            stage_dist = self.stage_analysis['daily_sa_name'].value_counts()
            dominant_stage = stage_dist.index[0]
            stage_pct = stage_dist.iloc[0] / len(self.stage_analysis) * 100

            insights['insights'].append(
                f"🎭 Market stage: {dominant_stage} dominates ({stage_pct:.1f}% of stocks)"
            )

        # Technical indicator insights
        if hasattr(self, 'basic_calculations'):
            if 'rsi' in self.basic_calculations.columns:
                overbought = (self.basic_calculations['rsi'] > 70).sum()
                oversold = (self.basic_calculations['rsi'] < 30).sum()

                insights['insights'].extend([
                    f"⚡ RSI conditions: {overbought} overbought, {oversold} oversold",
                ])

        # Data completeness insight
        insights['insights'].append(
            f"📁 Data richness: {insights['total_files']} datasets across {insights['data_sources']} categories"
        )

        return insights

    def run_comprehensive_analysis(self):
        """
        Run the complete comprehensive analysis workflow
        """
        print("🚀 Starting Comprehensive Multi-Source Market Analysis")
        print("=" * 70)

        results = {}

        # 1. Create overview dashboard
        results['dashboard'] = self.create_market_overview_dashboard()

        # 2. Cross-data correlations
        results['correlations'] = self.analyze_cross_data_correlations()

        # 3. Market regime analysis
        regime_data, regime_file = self.create_market_regime_analysis()
        results['regime'] = {'data': regime_data, 'file': regime_file}

        # 4. Universe performance analysis
        results['universe_analysis'] = self.create_universe_performance_analysis()

        # 5. Generate comprehensive insights
        results['insights'] = self.generate_comprehensive_insights()

        print("\n" + "=" * 70)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {results['insights']['total_files']} data files")
        print(f"📂 All outputs saved to: {self.output_dir}")
        print("=" * 70)

        return results


def main():
    """
    Main execution function
    """
    current_dir = Path(__file__).parent.parent.parent
    results_directory = current_dir / "results"

    print(f"Results directory: {results_directory}")

    # Initialize and run comprehensive analysis
    analyzer = ComprehensiveMarketAnalyzer(results_directory)
    results = analyzer.run_comprehensive_analysis()

    print(f"\n🎉 Comprehensive analysis complete!")
    print(f"💡 Key insights: {len(results['insights']['insights'])} insights generated")
    print(f"📈 Data sources utilized: {results['insights']['data_sources']}")


if __name__ == "__main__":
    main()