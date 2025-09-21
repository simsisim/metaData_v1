#!/usr/bin/env python3
"""
Advanced Unsupervised Clustering Analysis for Market Data
========================================================

This script implements sophisticated clustering algorithms to discover
hidden patterns in relative strength data, providing more actionable
insights than basic PCA analysis.

Clustering Algorithms:
- K-means with optimal cluster selection (Elbow + Silhouette)
- Hierarchical clustering with dendrograms
- DBSCAN for outlier detection
- Gaussian Mixture Models for soft clustering
- Time series clustering for pattern recognition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Time series clustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime

class AdvancedClusteringEngine:
    """
    Advanced clustering analysis engine for financial market data
    """

    def __init__(self, data_directory):
        """
        Initialize the clustering engine
        """
        self.data_dir = Path(data_directory)
        self.output_dir = Path(__file__).parent.parent / "outputs" / "advanced_clustering"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.load_market_data()
        print(f"🔬 Advanced Clustering Engine initialized")
        print(f"📊 Data: {len(self.stocks_data)} stocks loaded")

    def load_market_data(self):
        """
        Load and prepare market data for clustering analysis
        """
        print("📥 Loading market data...")

        # Load RS data
        rs_file = self.data_dir / "rs" / "rs_ibd_stocks_daily_2-5_20250905.csv"
        if rs_file.exists():
            self.stocks_data = pd.read_csv(rs_file)
            print(f"✅ Loaded {len(self.stocks_data)} stocks")
        else:
            raise FileNotFoundError(f"RS data file not found: {rs_file}")

        # Extract RS features for clustering
        self.rs_columns = [col for col in self.stocks_data.columns if '_rs_vs_QQQ' in col]
        print(f"📈 Found {len(self.rs_columns)} RS timeframes")

        # Create feature matrix
        self.feature_matrix = self.stocks_data[self.rs_columns].fillna(1.0)
        self.tickers = self.stocks_data['ticker'].values

        # Add derived features
        self.feature_matrix['rs_mean'] = self.feature_matrix.mean(axis=1)
        self.feature_matrix['rs_std'] = self.feature_matrix[self.rs_columns].std(axis=1)
        self.feature_matrix['rs_trend'] = (
            self.feature_matrix[[col for col in self.rs_columns if 'yearly' in col]].mean(axis=1) -
            self.feature_matrix[[col for col in self.rs_columns if 'daily_1d' in col]].mean(axis=1)
        )

        print(f"🔧 Created feature matrix: {self.feature_matrix.shape}")

    def find_optimal_clusters(self, max_clusters=10):
        """
        Find optimal number of clusters using multiple metrics
        """
        print("\n🎯 Finding optimal number of clusters...")

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.feature_matrix)

        # Test different cluster numbers
        cluster_range = range(2, max_clusters + 1)
        metrics = {
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }

        for k in cluster_range:
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)

            # Calculate metrics
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(features_scaled, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(features_scaled, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(features_scaled, labels))

        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cluster Optimization Metrics', fontsize=16, fontweight='bold')

        # Elbow curve
        axes[0, 0].plot(cluster_range, metrics['inertia'], 'bo-')
        axes[0, 0].set_title('Elbow Method (Lower is better)')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].grid(True, alpha=0.3)

        # Silhouette score
        axes[0, 1].plot(cluster_range, metrics['silhouette'], 'go-')
        axes[0, 1].set_title('Silhouette Score (Higher is better)')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].grid(True, alpha=0.3)

        # Calinski-Harabasz score
        axes[1, 0].plot(cluster_range, metrics['calinski_harabasz'], 'ro-')
        axes[1, 0].set_title('Calinski-Harabasz Score (Higher is better)')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('CH Score')
        axes[1, 0].grid(True, alpha=0.3)

        # Davies-Bouldin score
        axes[1, 1].plot(cluster_range, metrics['davies_bouldin'], 'mo-')
        axes[1, 1].set_title('Davies-Bouldin Score (Lower is better)')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('DB Score')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_optimization_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Find optimal k
        best_silhouette_k = cluster_range[np.argmax(metrics['silhouette'])]
        best_ch_k = cluster_range[np.argmax(metrics['calinski_harabasz'])]
        best_db_k = cluster_range[np.argmin(metrics['davies_bouldin'])]

        print(f"📊 Optimal clusters by metric:")
        print(f"  • Silhouette Score: {best_silhouette_k} clusters")
        print(f"  • Calinski-Harabasz: {best_ch_k} clusters")
        print(f"  • Davies-Bouldin: {best_db_k} clusters")

        # Use majority vote or silhouette as primary
        optimal_k = best_silhouette_k
        print(f"🎯 Selected optimal clusters: {optimal_k}")

        return optimal_k, metrics, features_scaled

    def perform_multiple_clustering_algorithms(self, features_scaled, optimal_k):
        """
        Apply multiple clustering algorithms and compare results
        """
        print(f"\n🔬 Applying multiple clustering algorithms with k={optimal_k}...")

        clustering_results = {}

        # 1. K-Means
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clustering_results['K-Means'] = kmeans.fit_predict(features_scaled)

        # 2. Hierarchical Clustering
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        clustering_results['Hierarchical'] = hierarchical.fit_predict(features_scaled)

        # 3. Gaussian Mixture Model
        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        clustering_results['GMM'] = gmm.fit_predict(features_scaled)

        # 4. DBSCAN (automatic cluster detection)
        # Tune eps parameter
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(features_scaled)
        distances, indices = neighbors_fit.kneighbors(features_scaled)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # Use knee point as eps
        eps = np.percentile(distances, 90)
        dbscan = DBSCAN(eps=eps, min_samples=5)
        clustering_results['DBSCAN'] = dbscan.fit_predict(features_scaled)

        # Calculate silhouette scores for comparison
        algorithm_scores = {}
        for algo_name, labels in clustering_results.items():
            if len(set(labels)) > 1 and -1 not in labels:  # Valid clustering
                score = silhouette_score(features_scaled, labels)
                algorithm_scores[algo_name] = score
            else:
                algorithm_scores[algo_name] = -1  # Invalid clustering

        print("📊 Algorithm Comparison (Silhouette Scores):")
        for algo, score in algorithm_scores.items():
            print(f"  • {algo}: {score:.3f}")

        return clustering_results, algorithm_scores

    def create_advanced_cluster_visualization(self, features_scaled, clustering_results):
        """
        Create comprehensive cluster visualization
        """
        print("\n🎨 Creating advanced cluster visualizations...")

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)

        # t-SNE for non-linear visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(features_scaled)

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('K-Means (PCA)', 'K-Means (t-SNE)',
                          'Hierarchical (PCA)', 'DBSCAN (PCA)'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        algorithms_to_plot = ['K-Means', 'Hierarchical', 'DBSCAN']
        plot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        # Plot K-Means with PCA
        kmeans_labels = clustering_results['K-Means']
        fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0], y=pca_result[:, 1],
                mode='markers',
                marker=dict(color=kmeans_labels, colorscale='viridis', size=8),
                text=[f"{ticker}<br>Cluster: {label}" for ticker, label in zip(self.tickers, kmeans_labels)],
                hovertemplate='%{text}<extra></extra>',
                name='K-Means'
            ),
            row=1, col=1
        )

        # Plot K-Means with t-SNE
        fig.add_trace(
            go.Scatter(
                x=tsne_result[:, 0], y=tsne_result[:, 1],
                mode='markers',
                marker=dict(color=kmeans_labels, colorscale='viridis', size=8),
                text=[f"{ticker}<br>Cluster: {label}" for ticker, label in zip(self.tickers, kmeans_labels)],
                hovertemplate='%{text}<extra></extra>',
                name='K-Means (t-SNE)'
            ),
            row=1, col=2
        )

        # Plot Hierarchical with PCA
        hier_labels = clustering_results['Hierarchical']
        fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0], y=pca_result[:, 1],
                mode='markers',
                marker=dict(color=hier_labels, colorscale='plasma', size=8),
                text=[f"{ticker}<br>Cluster: {label}" for ticker, label in zip(self.tickers, hier_labels)],
                hovertemplate='%{text}<extra></extra>',
                name='Hierarchical'
            ),
            row=2, col=1
        )

        # Plot DBSCAN with PCA
        dbscan_labels = clustering_results['DBSCAN']
        fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0], y=pca_result[:, 1],
                mode='markers',
                marker=dict(color=dbscan_labels, colorscale='cividis', size=8),
                text=[f"{ticker}<br>Cluster: {label}" for ticker, label in zip(self.tickers, dbscan_labels)],
                hovertemplate='%{text}<extra></extra>',
                name='DBSCAN'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title='Advanced Clustering Algorithm Comparison',
            height=800,
            showlegend=False
        )

        # Save interactive plot
        fig.write_html(self.output_dir / 'advanced_clustering_comparison.html')
        print(f"✅ Interactive clustering comparison saved")

        return pca_result, tsne_result

    def analyze_cluster_characteristics(self, clustering_results, optimal_k):
        """
        Analyze the characteristics of each cluster
        """
        print(f"\n📊 Analyzing cluster characteristics...")

        # Use K-Means results as primary
        primary_labels = clustering_results['K-Means']

        # Create analysis dataframe
        analysis_df = self.stocks_data.copy()
        analysis_df['cluster'] = primary_labels
        analysis_df['rs_mean'] = self.feature_matrix['rs_mean']
        analysis_df['rs_std'] = self.feature_matrix['rs_std']
        analysis_df['rs_trend'] = self.feature_matrix['rs_trend']

        # Calculate cluster statistics
        cluster_stats = []

        for cluster_id in range(optimal_k):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]

            if len(cluster_data) > 0:
                stats = {
                    'Cluster': f"Cluster {cluster_id}",
                    'Count': len(cluster_data),
                    'Avg_RS': cluster_data['rs_mean'].mean(),
                    'RS_Volatility': cluster_data['rs_std'].mean(),
                    'RS_Trend': cluster_data['rs_trend'].mean(),
                    'Top_Sectors': cluster_data['sector'].value_counts().head(3).to_dict(),
                    'Sample_Stocks': cluster_data['ticker'].head(5).tolist()
                }
                cluster_stats.append(stats)

        # Create cluster characteristics table
        stats_df = pd.DataFrame([{
            'Cluster': stats['Cluster'],
            'Size': stats['Count'],
            'Avg RS': f"{stats['Avg_RS']:.3f}",
            'Volatility': f"{stats['RS_Volatility']:.3f}",
            'Trend': f"{stats['RS_Trend']:.3f}",
            'Top Sectors': ', '.join(list(stats['Top_Sectors'].keys())[:2]),
            'Sample Stocks': ', '.join(stats['Sample_Stocks'][:3])
        } for stats in cluster_stats])

        print("\n🎯 Cluster Characteristics:")
        print(stats_df.to_string(index=False))

        # Create cluster interpretation
        interpretations = []
        for stats in cluster_stats:
            avg_rs = stats['Avg_RS']
            volatility = stats['RS_Volatility']
            trend = stats['RS_Trend']

            if avg_rs > 1.05:
                performance = "Strong Outperformers"
            elif avg_rs > 1.0:
                performance = "Modest Outperformers"
            elif avg_rs > 0.95:
                performance = "Market Performers"
            else:
                performance = "Underperformers"

            if volatility < 0.05:
                consistency = "Highly Consistent"
            elif volatility < 0.1:
                consistency = "Moderately Consistent"
            else:
                consistency = "Volatile"

            if trend > 0.05:
                momentum = "Accelerating"
            elif trend > -0.05:
                momentum = "Stable"
            else:
                momentum = "Decelerating"

            interpretation = f"{performance} • {consistency} • {momentum}"
            interpretations.append(interpretation)

        # Add interpretations to stats
        for i, interpretation in enumerate(interpretations):
            cluster_stats[i]['Interpretation'] = interpretation

        print("\n💡 Cluster Interpretations:")
        for stats in cluster_stats:
            print(f"  • {stats['Cluster']}: {stats['Interpretation']}")
            print(f"    └─ {stats['Count']} stocks, Top sectors: {list(stats['Top_Sectors'].keys())[:2]}")

        return cluster_stats, analysis_df

    def create_cluster_heatmap(self, analysis_df, cluster_stats):
        """
        Create detailed cluster heatmap
        """
        print("\n🔥 Creating cluster performance heatmap...")

        # Prepare data for heatmap
        heatmap_data = []

        for cluster_stat in cluster_stats:
            cluster_id = int(cluster_stat['Cluster'].split()[-1])
            cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]

            # Calculate timeframe performance for this cluster
            timeframe_performance = {}
            for col in self.rs_columns:
                timeframe_name = col.replace('daily_daily_', '').replace('_rs_vs_QQQ', '').replace('_', ' ')
                avg_performance = cluster_data[col].mean()
                timeframe_performance[timeframe_name] = avg_performance

            for timeframe, performance in timeframe_performance.items():
                heatmap_data.append({
                    'Cluster': cluster_stat['Cluster'],
                    'Timeframe': timeframe,
                    'Performance': performance,
                    'Interpretation': cluster_stat['Interpretation']
                })

        heatmap_df = pd.DataFrame(heatmap_data)

        # Create pivot table
        pivot_df = heatmap_df.pivot(index='Cluster', columns='Timeframe', values='Performance')

        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            zmid=1.0,
            text=np.round(pivot_df.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='Cluster: %{y}<br>Timeframe: %{x}<br>RS: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title='Cluster Performance Heatmap<br>Relative Strength by Timeframe and Cluster',
            xaxis_title="Timeframe",
            yaxis_title="Cluster",
            width=1000,
            height=500
        )

        fig.write_html(self.output_dir / 'cluster_performance_heatmap.html')
        print(f"✅ Cluster heatmap saved")

        return heatmap_df

    def create_dendogram_analysis(self, features_scaled):
        """
        Create hierarchical clustering dendrogram
        """
        print("\n🌳 Creating dendrogram analysis...")

        # Calculate linkage matrix
        linkage_matrix = linkage(features_scaled, method='ward')

        # Create dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix,
                  labels=self.tickers,
                  leaf_rotation=90,
                  leaf_font_size=8)
        plt.title('Hierarchical Clustering Dendrogram\nWard Linkage Method', fontsize=14, fontweight='bold')
        plt.xlabel('Stock Ticker')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Dendrogram saved")

    def generate_clustering_insights_report(self, cluster_stats, algorithm_scores):
        """
        Generate comprehensive clustering insights report
        """
        print("\n📝 Generating clustering insights report...")

        insights = []

        # Algorithm performance insights
        best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])
        insights.append(f"🏆 Best performing algorithm: {best_algorithm[0]} (Silhouette: {best_algorithm[1]:.3f})")

        # Cluster composition insights
        largest_cluster = max(cluster_stats, key=lambda x: x['Count'])
        smallest_cluster = min(cluster_stats, key=lambda x: x['Count'])

        insights.append(f"📊 Largest cluster: {largest_cluster['Cluster']} with {largest_cluster['Count']} stocks")
        insights.append(f"📊 Smallest cluster: {smallest_cluster['Cluster']} with {smallest_cluster['Count']} stocks")

        # Performance insights
        best_performing_cluster = max(cluster_stats, key=lambda x: x['Avg_RS'])
        worst_performing_cluster = min(cluster_stats, key=lambda x: x['Avg_RS'])

        insights.append(f"🚀 Best performing: {best_performing_cluster['Cluster']} (Avg RS: {best_performing_cluster['Avg_RS']:.3f})")
        insights.append(f"📉 Worst performing: {worst_performing_cluster['Cluster']} (Avg RS: {worst_performing_cluster['Avg_RS']:.3f})")

        # Volatility insights
        most_consistent_cluster = min(cluster_stats, key=lambda x: x['RS_Volatility'])
        most_volatile_cluster = max(cluster_stats, key=lambda x: x['RS_Volatility'])

        insights.append(f"📈 Most consistent: {most_consistent_cluster['Cluster']} (Volatility: {most_consistent_cluster['RS_Volatility']:.3f})")
        insights.append(f"⚡ Most volatile: {most_volatile_cluster['Cluster']} (Volatility: {most_volatile_cluster['RS_Volatility']:.3f})")

        # Actionable insights
        insights.append("\n💡 Investment Implications:")
        for stats in cluster_stats:
            if stats['Avg_RS'] > 1.05 and stats['RS_Volatility'] < 0.1:
                insights.append(f"  • {stats['Cluster']}: Strong momentum with consistency - potential core holdings")
            elif stats['Avg_RS'] > 1.0 and stats['RS_Trend'] > 0.05:
                insights.append(f"  • {stats['Cluster']}: Accelerating momentum - watch for breakouts")
            elif stats['Avg_RS'] < 0.95 and stats['RS_Trend'] < -0.05:
                insights.append(f"  • {stats['Cluster']}: Declining momentum - consider rotation opportunities")

        print("\n🔍 KEY CLUSTERING INSIGHTS:")
        for insight in insights:
            print(insight)

        return insights

    def run_complete_clustering_analysis(self):
        """
        Run the complete advanced clustering analysis
        """
        print("🚀 Starting Advanced Clustering Analysis")
        print("=" * 60)

        # Step 1: Find optimal clusters
        optimal_k, metrics, features_scaled = self.find_optimal_clusters()

        # Step 2: Apply multiple algorithms
        clustering_results, algorithm_scores = self.perform_multiple_clustering_algorithms(features_scaled, optimal_k)

        # Step 3: Create visualizations
        pca_result, tsne_result = self.create_advanced_cluster_visualization(features_scaled, clustering_results)

        # Step 4: Analyze characteristics
        cluster_stats, analysis_df = self.analyze_cluster_characteristics(clustering_results, optimal_k)

        # Step 5: Create heatmap
        heatmap_df = self.create_cluster_heatmap(analysis_df, cluster_stats)

        # Step 6: Create dendrogram
        self.create_dendogram_analysis(features_scaled)

        # Step 7: Generate insights
        insights = self.generate_clustering_insights_report(cluster_stats, algorithm_scores)

        print("\n" + "=" * 60)
        print("✅ ADVANCED CLUSTERING ANALYSIS COMPLETE!")
        print(f"📂 All outputs saved to: {self.output_dir}")
        print("=" * 60)

        return {
            'optimal_k': optimal_k,
            'cluster_stats': cluster_stats,
            'insights': insights,
            'algorithm_scores': algorithm_scores
        }


def main():
    """
    Main execution function
    """
    current_dir = Path(__file__).parent.parent.parent
    data_directory = current_dir / "results"

    # Run advanced clustering analysis
    engine = AdvancedClusteringEngine(data_directory)
    results = engine.run_complete_clustering_analysis()

    print(f"\n🎉 Advanced clustering analysis complete!")
    print(f"📊 Found {results['optimal_k']} optimal clusters")
    print(f"🏆 Best algorithm: {max(results['algorithm_scores'].items(), key=lambda x: x[1])[0]}")


if __name__ == "__main__":
    main()