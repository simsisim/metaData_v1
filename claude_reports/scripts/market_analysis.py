#!/usr/bin/env python3
"""
Comprehensive Market Analysis Report Generator
==============================================

This script analyzes relative strength (RS) and percentile data to generate
comprehensive market insights with advanced visualizations including:
- Sector and industry heatmaps
- Relative strength radar charts
- Time period performance analysis
- Machine learning pattern detection
- Top performers identification

Based on IBD-style relative strength methodology with multi-timeframe analysis.
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
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

import os
from datetime import datetime
from pathlib import Path

# Set style for better visuals
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MarketAnalysisEngine:
    """
    Advanced market analysis engine for RS and percentile data analysis
    """

    def __init__(self, data_directory):
        """
        Initialize the analysis engine

        Args:
            data_directory: Path to directory containing RS and percentile CSV files
        """
        self.data_dir = Path(data_directory)
        self.output_dir = Path(__file__).parent.parent / "outputs"
        self.reports_dir = Path(__file__).parent.parent / "reports"

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        # Data containers
        self.rs_data = {}
        self.per_data = {}
        self.analysis_results = {}

        print(f"MarketAnalysisEngine initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_data(self):
        """
        Load all RS and percentile data files
        """
        print("\n=== LOADING DATA ===")

        # Load RS data with dynamic file discovery
        rs_dir = self.data_dir / "rs"
        for level in ['stocks', 'sectors', 'industries']:
            rs_file = self._find_latest_file(rs_dir, level, 'rs')
            if rs_file:
                self.rs_data[level] = pd.read_csv(rs_file)
                print(f"✅ Loaded RS {level}: {len(self.rs_data[level])} records")
            else:
                print(f"❌ RS {level} file not found")

        # Load percentile data with dynamic file discovery
        per_dir = self.data_dir / "per"
        for level in ['stocks', 'sectors', 'industries']:
            per_file = self._find_latest_file(per_dir, level, 'per')
            if per_file:
                self.per_data[level] = pd.read_csv(per_file)
                print(f"✅ Loaded Percentile {level}: {len(self.per_data[level])} records")
            else:
                print(f"❌ Percentile {level} file not found")

        return self

    def _find_latest_file(self, directory, level, file_type):
        """Find the latest file for a given level and type with new naming convention support."""
        import glob

        if not directory.exists():
            return None

        # Patterns for different file types
        if file_type == 'rs':
            patterns = [
                # New format: rs_{benchmark}_{method}_{level}_daily_*.csv
                f"rs_*_*_{level}_daily_*.csv",
                # Legacy format: rs_{method}_{level}_daily_*.csv
                f"rs_*_{level}_daily_*.csv"
            ]
        elif file_type == 'per':
            patterns = [
                # New format: per_{benchmark}_{method}_{level}_daily_*.csv
                f"per_*_*_{level}_daily_*.csv",
                # Legacy format: per_{method}_{level}_daily_*.csv
                f"per_*_{level}_daily_*.csv"
            ]
        else:
            return None

        for pattern in patterns:
            matching_files = glob.glob(str(directory / pattern))
            if matching_files:
                # Return the most recent file path
                return max(matching_files, key=os.path.getmtime)

        return None

    def extract_timeframes(self, df):
        """
        Extract available timeframes from column names

        Args:
            df: DataFrame with RS or percentile data

        Returns:
            List of timeframe identifiers
        """
        rs_columns = [col for col in df.columns if '_rs_vs_QQQ' in col]
        timeframes = []

        for col in rs_columns:
            # Extract timeframe identifier (e.g., 'daily_daily_daily_1d', 'daily_daily_monthly_22d')
            parts = col.split('_')
            if len(parts) >= 4:
                timeframe_id = '_'.join(parts[:4])
                timeframes.append(timeframe_id)

        return sorted(list(set(timeframes)))

    def create_sector_performance_heatmap(self):
        """
        Create sector performance heatmap across multiple timeframes
        """
        print("\n=== CREATING SECTOR PERFORMANCE HEATMAP ===")

        if 'sectors' not in self.rs_data:
            print("❌ No sector data available")
            return

        df = self.rs_data['sectors'].copy()

        # Extract timeframes
        timeframes = self.extract_timeframes(df)
        print(f"Found timeframes: {timeframes}")

        # Prepare data for heatmap
        heatmap_data = []

        for timeframe in timeframes:
            rs_col = f"{timeframe}_rs_vs_QQQ"
            if rs_col in df.columns:
                for _, row in df.iterrows():
                    sector = row.get('ticker', row.get('sector', 'Unknown'))
                    rs_value = row[rs_col]

                    heatmap_data.append({
                        'Sector': sector,
                        'Timeframe': timeframe.replace('daily_daily_', '').replace('_', ' '),
                        'RS_Value': rs_value,
                        'Performance': 'Outperform' if rs_value > 1.0 else 'Underperform'
                    })

        heatmap_df = pd.DataFrame(heatmap_data)

        # Create pivot table for heatmap
        pivot_data = heatmap_df.pivot(index='Sector', columns='Timeframe', values='RS_Value')

        # Create matplotlib heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_data,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   center=1.0,
                   cbar_kws={'label': 'Relative Strength vs QQQ'})

        plt.title('Sector Relative Strength Heatmap Across Timeframes\n(Green = Outperform, Red = Underperform)',
                 fontsize=16, fontweight='bold')
        plt.xlabel('Timeframe', fontsize=12)
        plt.ylabel('Sector', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        output_file = self.output_dir / "sector_performance_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Sector heatmap saved: {output_file}")

        # Create interactive Plotly heatmap
        fig = px.imshow(pivot_data.values,
                       labels=dict(x="Timeframe", y="Sector", color="RS Value"),
                       x=pivot_data.columns,
                       y=pivot_data.index,
                       color_continuous_scale='RdYlGn',
                       color_continuous_midpoint=1.0,
                       title="Interactive Sector Relative Strength Heatmap")

        fig.update_layout(
            title_font_size=16,
            xaxis_title="Timeframe",
            yaxis_title="Sector",
            width=1000,
            height=600
        )

        # Save interactive plot
        interactive_file = self.output_dir / "sector_performance_heatmap_interactive.html"
        fig.write_html(interactive_file)
        print(f"✅ Interactive heatmap saved: {interactive_file}")

        return heatmap_df

    def create_industry_performance_analysis(self):
        """
        Create industry performance analysis with clustering
        """
        print("\n=== CREATING INDUSTRY PERFORMANCE ANALYSIS ===")

        if 'industries' not in self.rs_data:
            print("❌ No industry data available")
            return

        df = self.rs_data['industries'].copy()
        timeframes = self.extract_timeframes(df)

        # Prepare data matrix for clustering
        rs_columns = [f"{tf}_rs_vs_QQQ" for tf in timeframes if f"{tf}_rs_vs_QQQ" in df.columns]

        if len(rs_columns) == 0:
            print("❌ No RS columns found")
            return

        # Create feature matrix
        feature_matrix = df[rs_columns].fillna(1.0).values
        industry_names = df['ticker'].values if 'ticker' in df.columns else df.index.values

        # Perform K-means clustering
        n_clusters = min(3, len(industry_names))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_matrix)

        # Create radar chart for top industries
        fig = go.Figure()

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, (industry, cluster) in enumerate(zip(industry_names, clusters)):
            if i < 10:  # Show top 10 industries
                values = feature_matrix[i].tolist()
                values.append(values[0])  # Close the radar chart

                timeframe_labels = [tf.replace('daily_daily_', '').replace('_', ' ') for tf in timeframes]
                timeframe_labels.append(timeframe_labels[0])

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=timeframe_labels,
                    fill='toself',
                    name=f"{industry} (Cluster {cluster})",
                    line_color=colors[i % len(colors)]
                ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.8, 1.2]
                )),
            showlegend=True,
            title="Industry Relative Strength Radar Chart<br>Clustered by Performance Patterns",
            title_font_size=16
        )

        # Save radar chart
        radar_file = self.output_dir / "industry_radar_chart.html"
        fig.write_html(radar_file)
        print(f"✅ Industry radar chart saved: {radar_file}")

        return {'clusters': clusters, 'industries': industry_names, 'features': feature_matrix}

    def create_top_performers_analysis(self):
        """
        Identify and visualize top performers across different timeframes
        """
        print("\n=== CREATING TOP PERFORMERS ANALYSIS ===")

        if 'stocks' not in self.rs_data:
            print("❌ No stock data available")
            return

        df = self.rs_data['stocks'].copy()
        timeframes = self.extract_timeframes(df)

        # Calculate average RS across timeframes
        rs_columns = [f"{tf}_rs_vs_QQQ" for tf in timeframes if f"{tf}_rs_vs_QQQ" in df.columns]

        if len(rs_columns) == 0:
            print("❌ No RS columns found")
            return

        # Calculate multi-timeframe RS score
        df['avg_rs'] = df[rs_columns].mean(axis=1)
        df['rs_consistency'] = df[rs_columns].std(axis=1)  # Lower is more consistent

        # Get top performers
        top_performers = df.nlargest(20, 'avg_rs')[['ticker', 'sector', 'industry', 'avg_rs', 'rs_consistency'] + rs_columns]

        # Create bubble chart
        fig = px.scatter(top_performers,
                        x='avg_rs',
                        y='rs_consistency',
                        size='avg_rs',
                        color='sector',
                        hover_name='ticker',
                        hover_data=['industry'],
                        title="Top Performers: Average RS vs Consistency<br>Bubble size = Average RS score",
                        labels={
                            'avg_rs': 'Average Relative Strength',
                            'rs_consistency': 'RS Consistency (lower = more consistent)'
                        })

        fig.update_layout(
            width=1000,
            height=700,
            title_font_size=16
        )

        # Save bubble chart
        bubble_file = self.output_dir / "top_performers_bubble_chart.html"
        fig.write_html(bubble_file)
        print(f"✅ Top performers bubble chart saved: {bubble_file}")

        # Create performance distribution
        plt.figure(figsize=(15, 10))

        # Subplot 1: Top 10 performers by average RS
        plt.subplot(2, 2, 1)
        top_10 = top_performers.head(10)
        bars = plt.barh(range(len(top_10)), top_10['avg_rs'], color=plt.cm.RdYlGn(top_10['avg_rs']))
        plt.yticks(range(len(top_10)), top_10['ticker'])
        plt.xlabel('Average Relative Strength')
        plt.title('Top 10 Performers by Average RS')
        plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)

        # Subplot 2: Sector distribution
        plt.subplot(2, 2, 2)
        sector_counts = top_performers['sector'].value_counts()
        plt.pie(sector_counts.values, labels=sector_counts.index, autopct='%1.1f%%')
        plt.title('Top Performers by Sector')

        # Subplot 3: RS distribution histogram
        plt.subplot(2, 2, 3)
        plt.hist(df['avg_rs'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(df['avg_rs'].mean(), color='red', linestyle='--', label=f'Mean: {df["avg_rs"].mean():.3f}')
        plt.axvline(1.0, color='black', linestyle='-', label='Neutral (1.0)')
        plt.xlabel('Average Relative Strength')
        plt.ylabel('Number of Stocks')
        plt.title('Distribution of Average RS Scores')
        plt.legend()

        # Subplot 4: Consistency vs Performance
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(df['avg_rs'], df['rs_consistency'],
                             c=df['avg_rs'], cmap='RdYlGn', alpha=0.6)
        plt.xlabel('Average Relative Strength')
        plt.ylabel('RS Consistency (Std Dev)')
        plt.title('Performance vs Consistency')
        plt.colorbar(scatter, label='Average RS')

        plt.tight_layout()

        # Save performance analysis
        performance_file = self.output_dir / "top_performers_analysis.png"
        plt.savefig(performance_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Top performers analysis saved: {performance_file}")

        return top_performers

    def perform_machine_learning_analysis(self):
        """
        Perform machine learning analysis to identify patterns
        """
        print("\n=== PERFORMING MACHINE LEARNING ANALYSIS ===")

        if 'stocks' not in self.rs_data:
            print("❌ No stock data available")
            return

        df = self.rs_data['stocks'].copy()
        timeframes = self.extract_timeframes(df)

        # Prepare feature matrix
        rs_columns = [f"{tf}_rs_vs_QQQ" for tf in timeframes if f"{tf}_rs_vs_QQQ" in df.columns]

        if len(rs_columns) < 3:
            print("❌ Insufficient RS columns for ML analysis")
            return

        # Create feature matrix
        features = df[rs_columns].fillna(1.0)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(features_scaled)

        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # PCA plot
        scatter1 = axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1],
                                     c=clusters, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title(f'PCA Analysis\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

        # t-SNE plot
        scatter2 = axes[0, 1].scatter(tsne_result[:, 0], tsne_result[:, 1],
                                     c=clusters, cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('t-SNE Analysis')
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')

        # Cluster characteristics
        cluster_stats = []
        for i in range(4):
            cluster_mask = clusters == i
            cluster_features = features[cluster_mask]
            cluster_stats.append({
                'cluster': i,
                'count': cluster_mask.sum(),
                'avg_rs': cluster_features.mean().mean(),
                'std_rs': cluster_features.std().mean()
            })

        cluster_df = pd.DataFrame(cluster_stats)

        # Bar plot of cluster characteristics
        x = range(len(cluster_df))
        width = 0.35

        axes[1, 0].bar([i - width/2 for i in x], cluster_df['avg_rs'], width,
                      label='Average RS', alpha=0.8)
        axes[1, 0].bar([i + width/2 for i in x], cluster_df['std_rs'], width,
                      label='RS Volatility', alpha=0.8)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Cluster Characteristics')
        axes[1, 0].legend()
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'Cluster {i}' for i in range(4)])

        # Cluster size pie chart
        axes[1, 1].pie(cluster_df['count'], labels=[f'Cluster {i}' for i in range(4)], autopct='%1.1f%%')
        axes[1, 1].set_title('Cluster Size Distribution')

        plt.tight_layout()

        # Save ML analysis
        ml_file = self.output_dir / "machine_learning_analysis.png"
        plt.savefig(ml_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Machine learning analysis saved: {ml_file}")

        # Add cluster information to dataframe
        df['cluster'] = clusters
        df['pca_1'] = pca_result[:, 0]
        df['pca_2'] = pca_result[:, 1]

        return {
            'pca': pca,
            'tsne_result': tsne_result,
            'clusters': clusters,
            'cluster_stats': cluster_df,
            'features_scaled': features_scaled
        }

    def generate_market_insights(self):
        """
        Generate key market insights and trends
        """
        print("\n=== GENERATING MARKET INSIGHTS ===")

        insights = {
            'date': '2025-09-05',
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'insights': []
        }

        # Analyze sector performance
        if 'sectors' in self.rs_data:
            sector_df = self.rs_data['sectors']

            # Find best and worst performing sectors
            if len(sector_df) > 0:
                # Get latest RS data (assume it's quarterly for comprehensive view)
                quarterly_col = None
                for col in sector_df.columns:
                    if 'quarterly' in col and '_rs_vs_QQQ' in col:
                        quarterly_col = col
                        break

                if quarterly_col:
                    best_sector = sector_df.loc[sector_df[quarterly_col].idxmax()]
                    worst_sector = sector_df.loc[sector_df[quarterly_col].idxmin()]

                    insights['insights'].extend([
                        f"🏆 Best Performing Sector (Quarterly): {best_sector['ticker']} with RS {best_sector[quarterly_col]:.3f}",
                        f"📉 Worst Performing Sector (Quarterly): {worst_sector['ticker']} with RS {worst_sector[quarterly_col]:.3f}"
                    ])

        # Analyze stock performance
        if 'stocks' in self.rs_data:
            stock_df = self.rs_data['stocks']

            # Calculate average RS
            rs_columns = [col for col in stock_df.columns if '_rs_vs_QQQ' in col]
            if rs_columns:
                stock_df['avg_rs'] = stock_df[rs_columns].mean(axis=1)

                outperformers = (stock_df['avg_rs'] > 1.0).sum()
                underperformers = (stock_df['avg_rs'] < 1.0).sum()

                insights['insights'].extend([
                    f"📊 Market Breadth: {outperformers} stocks outperforming QQQ, {underperformers} underperforming",
                    f"🎯 Average Market RS: {stock_df['avg_rs'].mean():.3f}",
                    f"📈 Top Stock: {stock_df.loc[stock_df['avg_rs'].idxmax(), 'ticker']} (RS: {stock_df['avg_rs'].max():.3f})"
                ])

        # Market trend analysis
        if 'stocks' in self.rs_data and 'sectors' in self.rs_data:
            stock_avg = self.rs_data['stocks'][[col for col in self.rs_data['stocks'].columns if '_rs_vs_QQQ' in col]].mean().mean()
            sector_avg = self.rs_data['sectors'][[col for col in self.rs_data['sectors'].columns if '_rs_vs_QQQ' in col]].mean().mean()

            if stock_avg > 1.05:
                trend = "🚀 Strong Bull Market"
            elif stock_avg > 1.0:
                trend = "📈 Moderate Bull Market"
            elif stock_avg > 0.95:
                trend = "➡️ Sideways Market"
            else:
                trend = "📉 Bear Market Conditions"

            insights['insights'].append(f"🎭 Market Trend: {trend} (Overall RS: {stock_avg:.3f})")

        self.analysis_results['insights'] = insights
        return insights

    def create_comprehensive_report(self):
        """
        Create a comprehensive HTML report with all visualizations
        """
        print("\n=== CREATING COMPREHENSIVE REPORT ===")

        # Generate insights
        insights = self.generate_market_insights()

        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Market Analysis Report - {insights['date']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-top: 10px;
        }}
        .insights {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .insights h2 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .insight-item {{
            background-color: white;
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #6c757d;
            margin-top: 5px;
        }}
        .methodology {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Comprehensive Market Analysis Report</h1>
            <div class="subtitle">Relative Strength & Performance Analysis • {insights['date']}</div>
            <div class="subtitle">Generated: {insights['analysis_timestamp']}</div>
        </div>

        <div class="insights">
            <h2>🔍 Key Market Insights</h2>
            {"".join([f'<div class="insight-item">{insight}</div>' for insight in insights['insights']])}
        </div>

        <div class="section">
            <h2>🗺️ Sector Performance Heatmap</h2>
            <p>This heatmap shows relative strength performance across sectors and timeframes.
            Green indicates outperformance vs QQQ (RS > 1.0), while red indicates underperformance.</p>
            <div class="chart-container">
                <img src="../outputs/sector_performance_heatmap.png" alt="Sector Performance Heatmap">
            </div>
            <p><strong>Interpretation:</strong> Look for consistently green sectors across timeframes for strong momentum,
            and red sectors for potential rotation opportunities.</p>
        </div>

        <div class="section">
            <h2>🎯 Top Performers Analysis</h2>
            <p>Comprehensive analysis of top-performing stocks based on multi-timeframe relative strength scores.</p>
            <div class="chart-container">
                <img src="../outputs/top_performers_analysis.png" alt="Top Performers Analysis">
            </div>
            <p><strong>Key Metrics:</strong> Average RS measures overall performance, while consistency (lower std dev)
            indicates reliable performance across timeframes.</p>
        </div>

        <div class="section">
            <h2>🤖 Machine Learning Pattern Analysis</h2>
            <p>Advanced clustering analysis using PCA and t-SNE to identify hidden patterns in market behavior.</p>
            <div class="chart-container">
                <img src="../outputs/machine_learning_analysis.png" alt="Machine Learning Analysis">
            </div>
            <p><strong>Clusters Identified:</strong> Stocks are grouped by similar performance patterns,
            helping identify regime changes and rotation opportunities.</p>
        </div>

        <div class="methodology">
            <h3>📈 Methodology & Data Sources</h3>
            <p><strong>Relative Strength Calculation:</strong> IBD-style relative strength comparing individual securities to QQQ benchmark across multiple timeframes (1-day to 252-day periods).</p>
            <p><strong>Timeframes Analyzed:</strong> Daily (1d, 3d, 5d), Weekly (7d, 14d), Monthly (22d, 44d), Quarterly (66d, 132d), Yearly (252d)</p>
            <p><strong>Universe:</strong> Combined ticker selection (choice 2-5) including major indices components</p>
            <p><strong>Data Date:</strong> September 5, 2025</p>
            <p><strong>Analysis Tools:</strong> Python ecosystem including pandas, scikit-learn, plotly, and seaborn for comprehensive statistical analysis</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(self.rs_data.get('stocks', []))}</div>
                <div class="stat-label">Stocks Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self.rs_data.get('sectors', []))}</div>
                <div class="stat-label">Sectors Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self.rs_data.get('industries', []))}</div>
                <div class="stat-label">Industries Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">9</div>
                <div class="stat-label">Timeframes</div>
            </div>
        </div>

        <div class="footer">
            <p>📊 Report generated by Advanced Market Analysis Engine</p>
            <p>🔬 Combining traditional technical analysis with machine learning insights</p>
            <p>⚠️ This analysis is for educational purposes. Past performance does not guarantee future results.</p>
        </div>
    </div>
</body>
</html>
"""

        # Save HTML report
        report_file = self.reports_dir / f"comprehensive_market_analysis_{insights['date'].replace('-', '')}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Comprehensive report saved: {report_file}")
        return report_file

    def run_complete_analysis(self):
        """
        Run the complete market analysis workflow
        """
        print("🚀 Starting Comprehensive Market Analysis")
        print("=" * 50)

        # Load data
        self.load_data()

        # Create all visualizations
        self.create_sector_performance_heatmap()
        self.create_industry_performance_analysis()
        self.create_top_performers_analysis()
        self.perform_machine_learning_analysis()

        # Generate comprehensive report
        report_file = self.create_comprehensive_report()

        print("\n" + "=" * 50)
        print("✅ ANALYSIS COMPLETE!")
        print(f"📄 Main Report: {report_file}")
        print(f"📂 All outputs: {self.output_dir}")
        print("=" * 50)

        return report_file


def main():
    """
    Main execution function
    """
    # Set up paths
    current_dir = Path(__file__).parent.parent.parent
    data_directory = current_dir / "results"

    print(f"Data directory: {data_directory}")

    # Initialize and run analysis
    analyzer = MarketAnalysisEngine(data_directory)
    report_file = analyzer.run_complete_analysis()

    print(f"\n🎉 Analysis complete! Open {report_file} to view the comprehensive report.")


if __name__ == "__main__":
    main()