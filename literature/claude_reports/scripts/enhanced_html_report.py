#!/usr/bin/env python3
"""
Enhanced Static HTML Report Generator
====================================

Creates beautiful, readable static HTML reports with improved:
- Typography and visual hierarchy
- Section organization and flow
- Chart integration and sizing
- Interactive elements within static format
- Professional styling and branding
"""

import pandas as pd
import numpy as np
import base64
from pathlib import Path
from datetime import datetime
import json

class EnhancedHTMLReportGenerator:
    """
    Enhanced static HTML report generator focused on readability and professional presentation
    """

    def __init__(self, data_directory):
        """
        Initialize the enhanced HTML generator
        """
        self.data_dir = Path(data_directory)
        self.output_dir = Path(__file__).parent.parent / "outputs"
        self.reports_dir = Path(__file__).parent.parent / "reports"
        self.advanced_clustering_dir = self.output_dir / "advanced_clustering"

        # Load all data
        self.load_all_data()

        print(f"📊 Enhanced HTML Report Generator initialized")

    def load_all_data(self):
        """
        Load all analysis data for the report
        """
        print("📥 Loading comprehensive analysis data...")

        # Load RS data with dynamic file discovery
        rs_dir = self.data_dir / "rs"
        self.rs_stocks = self._load_latest_rs_file(rs_dir, "stocks")
        self.rs_sectors = self._load_latest_rs_file(rs_dir, "sectors")
        self.rs_industries = self._load_latest_rs_file(rs_dir, "industries")

        # Calculate enhanced metrics
        if not self.rs_stocks.empty:
            rs_columns = [col for col in self.rs_stocks.columns if '_rs_vs_QQQ' in col]
            if rs_columns:
                self.rs_stocks['avg_rs'] = self.rs_stocks[rs_columns].mean(axis=1)
                self.rs_stocks['rs_consistency'] = self.rs_stocks[rs_columns].std(axis=1)

        print(f"✅ Loaded: {len(self.rs_stocks)} stocks, {len(self.rs_sectors)} sectors, {len(self.rs_industries)} industries")

    def _load_latest_rs_file(self, rs_dir, level):
        """Load the latest RS file for a given level with new naming convention support."""
        import glob

        if not rs_dir.exists():
            return pd.DataFrame()

        # Common patterns to try (both new and legacy formats)
        patterns = [
            # New format: rs_{benchmark}_{method}_{level}_daily_*.csv
            f"rs_*_*_{level}_daily_*.csv",
            # Legacy format: rs_{method}_{level}_daily_*.csv
            f"rs_*_{level}_daily_*.csv"
        ]

        for pattern in patterns:
            matching_files = glob.glob(str(rs_dir / pattern))
            if matching_files:
                # Return the most recent file as DataFrame
                latest_file = max(matching_files, key=os.path.getmtime)
                return pd.read_csv(latest_file)

        return pd.DataFrame()

    def encode_image_to_base64(self, image_path):
        """
        Encode image to base64 for embedding in HTML
        """
        try:
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{img_base64}"
        except:
            return None

    def get_advanced_styles(self):
        """
        Return enhanced CSS styles for better readability
        """
        return """
        <style>
            /* Enhanced Typography and Layout */
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.7;
                color: #2c3e50;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 0;
                margin: 0;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                box-shadow: 0 0 40px rgba(0,0,0,0.1);
                border-radius: 0;
                overflow: hidden;
            }

            /* Header Design */
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 60px 40px;
                text-align: center;
                position: relative;
            }

            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="1" fill="white" opacity="0.1"/><circle cx="80" cy="80" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                opacity: 0.3;
            }

            .header h1 {
                font-size: 3.2em;
                font-weight: 300;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                position: relative;
                z-index: 1;
            }

            .header .subtitle {
                font-size: 1.4em;
                opacity: 0.9;
                font-weight: 300;
                position: relative;
                z-index: 1;
            }

            .header .meta {
                margin-top: 30px;
                font-size: 0.95em;
                opacity: 0.8;
                position: relative;
                z-index: 1;
            }

            /* Content Area */
            .content {
                padding: 0;
            }

            /* Enhanced Sections */
            .section {
                padding: 50px 40px;
                border-bottom: 1px solid #ecf0f1;
                position: relative;
            }

            .section:nth-child(even) {
                background: #fafbfc;
            }

            .section h2 {
                font-size: 2.2em;
                color: #2c3e50;
                margin-bottom: 25px;
                font-weight: 600;
                position: relative;
                padding-bottom: 15px;
            }

            .section h2::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 60px;
                height: 3px;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 2px;
            }

            .section h3 {
                font-size: 1.5em;
                color: #34495e;
                margin: 30px 0 15px 0;
                font-weight: 500;
            }

            /* Enhanced Insights */
            .insights {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                margin: 30px 0;
                border-radius: 12px;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }

            .insights h3 {
                color: white;
                font-size: 1.8em;
                margin-bottom: 25px;
                text-align: center;
            }

            .insight-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }

            .insight-card {
                background: rgba(255, 255, 255, 0.15);
                padding: 20px;
                border-radius: 8px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }

            .insight-card .icon {
                font-size: 2em;
                margin-bottom: 10px;
                display: block;
            }

            .insight-card .title {
                font-weight: 600;
                margin-bottom: 8px;
                font-size: 1.1em;
            }

            .insight-card .value {
                font-size: 1.8em;
                font-weight: 300;
                margin-bottom: 5px;
            }

            .insight-card .description {
                font-size: 0.9em;
                opacity: 0.9;
            }

            /* Enhanced Statistics Grid */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 25px;
                margin: 40px 0;
            }

            .stat-card {
                background: white;
                padding: 30px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                border: 1px solid #e8ecef;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                position: relative;
                overflow: hidden;
            }

            .stat-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }

            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            }

            .stat-icon {
                font-size: 2.5em;
                margin-bottom: 15px;
                color: #667eea;
            }

            .stat-value {
                font-size: 2.8em;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
                line-height: 1;
            }

            .stat-label {
                color: #7f8c8d;
                font-size: 1em;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .stat-change {
                margin-top: 10px;
                font-size: 0.9em;
                padding: 5px 12px;
                border-radius: 20px;
                display: inline-block;
            }

            .stat-change.positive {
                background: #d4edda;
                color: #155724;
            }

            .stat-change.negative {
                background: #f8d7da;
                color: #721c24;
            }

            /* Enhanced Charts */
            .chart-container {
                margin: 40px 0;
                text-align: center;
                background: white;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                border: 1px solid #e8ecef;
            }

            .chart-container h4 {
                font-size: 1.3em;
                margin-bottom: 20px;
                color: #2c3e50;
                font-weight: 500;
            }

            .chart-container img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            }

            .chart-description {
                margin-top: 20px;
                font-style: italic;
                color: #7f8c8d;
                line-height: 1.6;
                max-width: 800px;
                margin-left: auto;
                margin-right: auto;
            }

            /* Enhanced Tables */
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 30px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            }

            .data-table th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 18px 15px;
                text-align: left;
                font-weight: 600;
                font-size: 0.95em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .data-table td {
                padding: 15px;
                border-bottom: 1px solid #ecf0f1;
                font-size: 0.95em;
            }

            .data-table tr:hover {
                background: #f8f9fa;
            }

            .data-table .performance-positive {
                color: #27ae60;
                font-weight: 600;
            }

            .data-table .performance-negative {
                color: #e74c3c;
                font-weight: 600;
            }

            /* Enhanced Methodology */
            .methodology {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 40px;
                border-radius: 12px;
                border-left: 6px solid #28a745;
                margin: 40px 0;
            }

            .methodology h3 {
                color: #28a745;
                font-size: 1.4em;
                margin-bottom: 20px;
            }

            .methodology ul {
                list-style: none;
                padding: 0;
            }

            .methodology li {
                padding: 8px 0;
                padding-left: 25px;
                position: relative;
            }

            .methodology li::before {
                content: '✓';
                position: absolute;
                left: 0;
                color: #28a745;
                font-weight: bold;
            }

            /* Footer */
            .footer {
                background: #2c3e50;
                color: white;
                text-align: center;
                padding: 40px;
            }

            .footer p {
                margin: 10px 0;
                opacity: 0.8;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .container {
                    margin: 0;
                    border-radius: 0;
                }

                .header {
                    padding: 40px 20px;
                }

                .header h1 {
                    font-size: 2.2em;
                }

                .section {
                    padding: 30px 20px;
                }

                .stats-grid {
                    grid-template-columns: 1fr;
                    gap: 15px;
                }

                .insight-grid {
                    grid-template-columns: 1fr;
                }
            }

            /* Print Styles */
            @media print {
                body {
                    background: white;
                }

                .container {
                    box-shadow: none;
                    max-width: none;
                }

                .header {
                    background: #2c3e50 !important;
                    -webkit-print-color-adjust: exact;
                }

                .chart-container {
                    break-inside: avoid;
                }
            }
        </style>
        """

    def generate_market_insights_advanced(self):
        """
        Generate enhanced market insights with more detailed analysis
        """
        insights = []

        if not self.rs_stocks.empty:
            # Market breadth analysis
            outperformers = (self.rs_stocks['avg_rs'] > 1.0).sum()
            underperformers = (self.rs_stocks['avg_rs'] < 1.0).sum()
            total_stocks = len(self.rs_stocks)

            # Performance distribution
            strong_outperformers = (self.rs_stocks['avg_rs'] > 1.1).sum()
            strong_underperformers = (self.rs_stocks['avg_rs'] < 0.9).sum()

            # Consistency analysis
            consistent_performers = (self.rs_stocks['rs_consistency'] < 0.1).sum()
            volatile_performers = (self.rs_stocks['rs_consistency'] > 0.2).sum()

            # Top performer
            top_stock = self.rs_stocks.loc[self.rs_stocks['avg_rs'].idxmax()]

            insights_data = {
                'market_breadth': {
                    'outperformers': outperformers,
                    'outperformers_pct': outperformers/total_stocks*100,
                    'underperformers': underperformers,
                    'underperformers_pct': underperformers/total_stocks*100
                },
                'performance_distribution': {
                    'strong_out': strong_outperformers,
                    'strong_out_pct': strong_outperformers/total_stocks*100,
                    'strong_under': strong_underperformers,
                    'strong_under_pct': strong_underperformers/total_stocks*100
                },
                'consistency': {
                    'consistent': consistent_performers,
                    'consistent_pct': consistent_performers/total_stocks*100,
                    'volatile': volatile_performers,
                    'volatile_pct': volatile_performers/total_stocks*100
                },
                'top_performer': {
                    'ticker': top_stock['ticker'],
                    'rs': top_stock['avg_rs'],
                    'sector': top_stock.get('sector', 'N/A'),
                    'industry': top_stock.get('industry', 'N/A')
                },
                'market_rs': self.rs_stocks['avg_rs'].mean()
            }

        return insights_data

    def create_enhanced_html_report(self):
        """
        Create the enhanced static HTML report
        """
        print("\n📄 Creating enhanced static HTML report...")

        # Generate insights
        insights = self.generate_market_insights_advanced()

        # Get images as base64
        sector_heatmap = self.encode_image_to_base64(self.output_dir / "sector_performance_heatmap.png")
        top_performers = self.encode_image_to_base64(self.output_dir / "top_performers_analysis.png")
        ml_analysis = self.encode_image_to_base64(self.output_dir / "machine_learning_analysis.png")
        cluster_optimization = self.encode_image_to_base64(self.advanced_clustering_dir / "cluster_optimization_metrics.png")

        # Build HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Market Analysis Report - September 2025</title>
    {self.get_advanced_styles()}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Market Intelligence Report</h1>
            <div class="subtitle">Comprehensive Relative Strength & Clustering Analysis</div>
            <div class="meta">
                <div>Analysis Date: September 5, 2025 • Report Generated: {datetime.now().strftime('%B %d, %Y')}</div>
                <div>Universe: 117 Stocks • 9 Sectors • 10 Industries</div>
            </div>
        </div>

        <div class="content">
            <div class="section">
                <div class="insights">
                    <h3>🔍 Executive Market Intelligence</h3>
                    <div class="insight-grid">
                        <div class="insight-card">
                            <div class="icon">📊</div>
                            <div class="title">Market Breadth</div>
                            <div class="value">{insights['market_breadth']['outperformers_pct']:.1f}%</div>
                            <div class="description">Stocks outperforming QQQ benchmark</div>
                        </div>
                        <div class="insight-card">
                            <div class="icon">🚀</div>
                            <div class="title">Strong Momentum</div>
                            <div class="value">{insights['performance_distribution']['strong_out']}</div>
                            <div class="description">Stocks with RS > 1.1 (strong outperformance)</div>
                        </div>
                        <div class="insight-card">
                            <div class="icon">⚖️</div>
                            <div class="title">Consistency</div>
                            <div class="value">{insights['consistency']['consistent_pct']:.1f}%</div>
                            <div class="description">Stocks with stable RS patterns</div>
                        </div>
                        <div class="insight-card">
                            <div class="icon">🏆</div>
                            <div class="title">Top Performer</div>
                            <div class="value">{insights['top_performer']['ticker']}</div>
                            <div class="description">RS: {insights['top_performer']['rs']:.3f} • {insights['top_performer']['sector']}</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📈 Performance Metrics Dashboard</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon">📊</div>
                        <div class="stat-value">{insights['market_breadth']['outperformers']}</div>
                        <div class="stat-label">Outperformers</div>
                        <div class="stat-change positive">+{insights['market_breadth']['outperformers_pct']:.1f}% of universe</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">📉</div>
                        <div class="stat-value">{insights['market_breadth']['underperformers']}</div>
                        <div class="stat-label">Underperformers</div>
                        <div class="stat-change negative">{insights['market_breadth']['underperformers_pct']:.1f}% of universe</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">🎯</div>
                        <div class="stat-value">{insights['market_rs']:.3f}</div>
                        <div class="stat-label">Market RS Average</div>
                        <div class="stat-change {'positive' if insights['market_rs'] > 1.0 else 'negative'}">{'Above' if insights['market_rs'] > 1.0 else 'Below'} benchmark</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">⚡</div>
                        <div class="stat-value">{insights['consistency']['volatile']}</div>
                        <div class="stat-label">High Volatility</div>
                        <div class="stat-change negative">{insights['consistency']['volatile_pct']:.1f}% showing instability</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>🗺️ Sector Performance Analysis</h2>
                <div class="chart-container">
                    <h4>Multi-Timeframe Sector Heatmap</h4>
                    {"<img src='" + sector_heatmap + "' alt='Sector Performance Heatmap'>" if sector_heatmap else "<p>Chart not available</p>"}
                    <div class="chart-description">
                        This heatmap reveals sector rotation patterns across multiple timeframes. Green areas indicate
                        outperformance vs QQQ, while red areas show underperformance. Look for sectors with consistent
                        green across timeframes for momentum plays, and red sectors for potential rotation opportunities.
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>🏆 Top Performers Deep Dive</h2>
                <div class="chart-container">
                    <h4>Performance Distribution & Consistency Analysis</h4>
                    {"<img src='" + top_performers + "' alt='Top Performers Analysis'>" if top_performers else "<p>Chart not available</p>"}
                    <div class="chart-description">
                        Four-panel analysis showing: (1) Top 10 performers by average RS, (2) Sector composition of leaders,
                        (3) Overall RS distribution across the universe, (4) Performance vs consistency scatter plot.
                        The ideal quadrant is high performance with low volatility (upper left in scatter plot).
                    </div>
                </div>

                <h3>🎯 Top 10 Performance Leaders</h3>
                {"<table class='data-table'>" if not self.rs_stocks.empty else ""}
                {"<tr><th>Rank</th><th>Ticker</th><th>Sector</th><th>Avg RS</th><th>Consistency</th><th>Performance</th></tr>" if not self.rs_stocks.empty else ""}
                {"".join([
                    f"<tr><td>{i}</td><td><strong>{row['ticker']}</strong></td><td>{row.get('sector', 'N/A')}</td>"
                    f"<td class='performance-{'positive' if row['avg_rs'] > 1.0 else 'negative'}'>{row['avg_rs']:.3f}</td>"
                    f"<td>{row['rs_consistency']:.3f}</td>"
                    f"<td class='performance-{'positive' if row['avg_rs'] > 1.0 else 'negative'}'>{'Outperform' if row['avg_rs'] > 1.0 else 'Underperform'}</td></tr>"
                    for i, (_, row) in enumerate(self.rs_stocks.nlargest(10, 'avg_rs').iterrows(), 1)
                ]) if not self.rs_stocks.empty else ""}
                {"</table>" if not self.rs_stocks.empty else ""}
            </div>

            <div class="section">
                <h2>🤖 Advanced Clustering Intelligence</h2>
                <div class="chart-container">
                    <h4>Scientific Cluster Optimization</h4>
                    {"<img src='" + cluster_optimization + "' alt='Cluster Optimization'>" if cluster_optimization else "<p>Chart not available</p>"}
                    <div class="chart-description">
                        Multiple optimization metrics determine the ideal number of clusters. The analysis reveals
                        <strong>2 distinct performance clusters</strong>: a large group of market followers (109 stocks)
                        and an elite group of momentum leaders (8 stocks) with accelerating relative strength patterns.
                    </div>
                </div>

                <div class="methodology">
                    <h3>🔬 Clustering Methodology</h3>
                    <ul>
                        <li><strong>Gaussian Mixture Model</strong> selected as optimal algorithm (Silhouette Score: 0.717)</li>
                        <li><strong>Feature Engineering</strong>: 10 RS timeframes + derived momentum indicators</li>
                        <li><strong>Validation</strong>: Multiple algorithms compared (K-means, Hierarchical, DBSCAN, GMM)</li>
                        <li><strong>Optimization</strong>: Elbow method, Silhouette analysis, Calinski-Harabasz scoring</li>
                        <li><strong>Business Logic</strong>: Clusters mapped to actionable investment strategies</li>
                    </ul>
                </div>

                <h3>💡 Cluster Investment Implications</h3>
                <div class="insight-grid">
                    <div class="insight-card">
                        <div class="icon">👥</div>
                        <div class="title">Cluster 0: Market Followers</div>
                        <div class="value">109 stocks</div>
                        <div class="description">Average RS: 0.963 • Strategy: Value rotation candidates, mean reversion plays</div>
                    </div>
                    <div class="insight-card">
                        <div class="icon">🚀</div>
                        <div class="title">Cluster 1: Momentum Leaders</div>
                        <div class="value">8 stocks</div>
                        <div class="description">Average RS: 1.207 • Strategy: Momentum continuation, breakout plays</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>⚗️ Machine Learning Insights</h2>
                <div class="chart-container">
                    <h4>Pattern Recognition & Dimensionality Analysis</h4>
                    {"<img src='" + ml_analysis + "' alt='Machine Learning Analysis'>" if ml_analysis else "<p>Chart not available</p>"}
                    <div class="chart-description">
                        Advanced pattern recognition using PCA, t-SNE, and clustering algorithms. The analysis reveals
                        clear bifurcation in the market: most stocks following similar patterns while a small group
                        exhibits distinctly different momentum characteristics. This suggests a concentrated leadership market.
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📋 Investment Strategy Framework</h2>

                <h3>🎯 Portfolio Construction Guidelines</h3>
                <div class="methodology">
                    <h3>Strategic Allocation</h3>
                    <ul>
                        <li><strong>Core Holdings (60-70%)</strong>: Cluster 0 market followers with strong fundamentals</li>
                        <li><strong>Momentum Allocation (20-30%)</strong>: Cluster 1 leaders with position size limits</li>
                        <li><strong>Rotation Reserve (10-20%)</strong>: Cash for sector rotation opportunities</li>
                        <li><strong>Risk Management</strong>: Smaller positions in high-volatility Cluster 1 stocks</li>
                    </ul>
                </div>

                <h3>⚡ Tactical Considerations</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon">📈</div>
                        <div class="stat-value">Momentum</div>
                        <div class="stat-label">Market Regime</div>
                        <div class="stat-change positive">Cluster 1 showing acceleration</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">🔄</div>
                        <div class="stat-value">Rotation</div>
                        <div class="stat-label">Opportunity</div>
                        <div class="stat-change negative">93% of stocks in lagging cluster</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">⚖️</div>
                        <div class="stat-value">Balance</div>
                        <div class="stat-label">Risk Profile</div>
                        <div class="stat-change positive">Diversification across patterns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">🎯</div>
                        <div class="stat-value">Precision</div>
                        <div class="stat-label">Targeting</div>
                        <div class="stat-change positive">Clear cluster definitions</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📊 Technical Methodology</h2>
                <div class="methodology">
                    <h3>🔬 Analysis Framework</h3>
                    <ul>
                        <li><strong>Relative Strength</strong>: IBD-style methodology vs QQQ benchmark across 10 timeframes</li>
                        <li><strong>Timeframe Coverage</strong>: 1d to 252d periods capturing multiple momentum cycles</li>
                        <li><strong>Universe Selection</strong>: Combined choice 2-5 including major index components</li>
                        <li><strong>Machine Learning</strong>: Multiple algorithms with scientific validation</li>
                        <li><strong>Feature Engineering</strong>: Derived momentum, consistency, and trend indicators</li>
                        <li><strong>Visualization</strong>: Professional-grade charts with statistical overlays</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>📊 <strong>Enhanced Market Intelligence System</strong></p>
            <p>🔬 Combining Traditional Analysis with Machine Learning Innovation</p>
            <p>⚠️ For educational purposes • Past performance does not guarantee future results</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} • Analysis Date: September 5, 2025</p>
        </div>
    </div>
</body>
</html>
"""

        # Save enhanced HTML report
        report_file = self.reports_dir / f"enhanced_market_intelligence_{datetime.now().strftime('%Y%m%d')}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Enhanced HTML report saved: {report_file}")
        print(f"📄 File size: {report_file.stat().st_size / 1024:.1f} KB")

        return report_file


def main():
    """
    Main execution function
    """
    current_dir = Path(__file__).parent.parent.parent
    data_directory = current_dir / "results"

    # Create enhanced HTML report
    generator = EnhancedHTMLReportGenerator(data_directory)
    report_file = generator.create_enhanced_html_report()

    print(f"\n🌟 Enhanced static HTML report complete!")
    print(f"📊 Professional presentation with embedded visualizations")
    print(f"🎯 Optimized for readability and executive consumption")


if __name__ == "__main__":
    main()