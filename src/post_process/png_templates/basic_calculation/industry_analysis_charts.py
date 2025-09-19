#!/usr/bin/env python3
"""
Industry Analysis Charts for Basic Calculation Data
===================================================

Industry-focused analysis charts including:
- Industry performance rankings within sectors
- Cross-industry comparison analysis
- Industry concentration and diversity metrics
- Top performers by industry
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style for consistent charts
plt.style.use('default')
sns.set_palette("tab20")

def create_industry_performance_ranking(df: pd.DataFrame, output_path: str,
                                      performance_col: str = 'daily_daily_yearly_252d_pct_change',
                                      top_n: int = 20, title_suffix: str = "") -> bool:
    """
    Create industry performance ranking chart showing top N industries.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_col: Performance column to analyze
        top_n: Number of top industries to display
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['industry', performance_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Calculate industry performance statistics
        industry_stats = df.groupby('industry').agg({
            performance_col: ['mean', 'count', 'std'],
            'ticker': 'count'
        }).round(2)

        # Flatten column names
        industry_stats.columns = ['_'.join(col).strip() for col in industry_stats.columns]

        # Filter industries with at least 2 stocks
        industry_stats = industry_stats[industry_stats[f'{performance_col}_count'] >= 2]

        if industry_stats.empty:
            logger.warning("No industry data available with sufficient stock count")
            return False

        # Sort by mean performance and get top N
        top_industries = industry_stats.nlargest(top_n, f'{performance_col}_mean')

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

        # 1. Top Industry Performance Ranking
        perf_data = top_industries[f'{performance_col}_mean'].sort_values(ascending=True)
        colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(perf_data)))

        bars1 = ax1.barh(range(len(perf_data)), perf_data.values, color=colors1)

        ax1.set_yticks(range(len(perf_data)))
        ax1.set_yticklabels([label[:30] + '...' if len(label) > 30 else label
                            for label in perf_data.index], fontsize=9)
        ax1.set_xlabel('Mean Performance (%)', fontsize=11)
        ax1.set_title(f'Top {top_n} Industry Performance Rankings {title_suffix}',
                     fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, value in enumerate(perf_data.values):
            ax1.text(value + (max(perf_data.values) * 0.01), i, f'{value:.1f}%',
                    va='center', fontsize=8)

        # 2. Industry Stock Count vs Performance
        ax2.scatter(top_industries[f'{performance_col}_count'], top_industries[f'{performance_col}_mean'],
                   s=top_industries[f'{performance_col}_std']*10, alpha=0.7, c=range(len(top_industries)),
                   cmap='viridis')

        ax2.set_xlabel('Number of Stocks in Industry', fontsize=11)
        ax2.set_ylabel('Mean Performance (%)', fontsize=11)
        ax2.set_title('Industry Size vs Performance (bubble size = volatility)', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

        # Add industry labels for top performers
        for i, (industry, row) in enumerate(top_industries.head(10).iterrows()):
            ax2.annotate(industry[:20] + '...' if len(industry) > 20 else industry,
                        (row[f'{performance_col}_count'], row[f'{performance_col}_mean']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

        # 3. Industry Performance Distribution
        # Show distribution of all industry performances
        all_industry_perf = df.groupby('industry')[performance_col].mean()

        ax3.hist(all_industry_perf.dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(all_industry_perf.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {all_industry_perf.mean():.1f}%')
        ax3.axvline(all_industry_perf.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {all_industry_perf.median():.1f}%')

        ax3.set_xlabel('Industry Performance (%)', fontsize=11)
        ax3.set_ylabel('Number of Industries', fontsize=11)
        ax3.set_title('Distribution of Industry Performance', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Industry Risk-Return Profile
        # Filter for industries with reasonable data
        risk_return_data = industry_stats[industry_stats[f'{performance_col}_count'] >= 3]

        if not risk_return_data.empty:
            ax4.scatter(risk_return_data[f'{performance_col}_std'], risk_return_data[f'{performance_col}_mean'],
                       s=risk_return_data[f'{performance_col}_count']*3, alpha=0.7,
                       c=risk_return_data[f'{performance_col}_mean'], cmap='RdYlGn')

            ax4.set_xlabel('Performance Volatility (Std Dev %)', fontsize=11)
            ax4.set_ylabel('Mean Performance (%)', fontsize=11)
            ax4.set_title('Industry Risk-Return Profile (size = stock count)', fontsize=12, fontweight='bold')
            ax4.grid(alpha=0.3)

            # Add quadrant lines
            ax4.axhline(y=risk_return_data[f'{performance_col}_mean'].median(),
                       color='gray', linestyle='--', alpha=0.5)
            ax4.axvline(x=risk_return_data[f'{performance_col}_std'].median(),
                       color='gray', linestyle='--', alpha=0.5)

            # Label best risk-adjusted performers (high return, low risk)
            high_return = risk_return_data[f'{performance_col}_mean'] > risk_return_data[f'{performance_col}_mean'].quantile(0.75)
            low_risk = risk_return_data[f'{performance_col}_std'] < risk_return_data[f'{performance_col}_std'].quantile(0.5)
            best_performers = risk_return_data[high_return & low_risk]

            for industry, row in best_performers.iterrows():
                ax4.annotate(industry[:15] + '...' if len(industry) > 15 else industry,
                            (row[f'{performance_col}_std'], row[f'{performance_col}_mean']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.3))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created industry performance ranking: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating industry performance ranking: {e}")
        return False

def create_industry_sector_comparison(df: pd.DataFrame, output_path: str,
                                    performance_col: str = 'daily_daily_yearly_252d_pct_change',
                                    title_suffix: str = "") -> bool:
    """
    Create comparison between industries within their respective sectors.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_col: Performance column to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['sector', 'industry', performance_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Calculate sector and industry performance
        sector_perf = df.groupby('sector')[performance_col].mean()
        industry_perf = df.groupby(['sector', 'industry'])[performance_col].mean().reset_index()

        if industry_perf.empty:
            logger.warning("No industry-sector data available")
            return False

        # Calculate industry performance relative to sector
        industry_perf['sector_performance'] = industry_perf['sector'].map(sector_perf)
        industry_perf['relative_performance'] = industry_perf[performance_col] - industry_perf['sector_performance']

        # Get top sectors by number of industries
        sector_industry_count = industry_perf['sector'].value_counts()
        top_sectors = sector_industry_count.head(4).index.tolist()

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for i, sector in enumerate(top_sectors):
            if i >= 4:
                break

            sector_data = industry_perf[industry_perf['sector'] == sector].copy()
            sector_data = sector_data.sort_values('relative_performance', ascending=True)

            # Color industries based on relative performance
            colors = ['red' if x < 0 else 'green' for x in sector_data['relative_performance']]

            bars = axes[i].barh(range(len(sector_data)), sector_data['relative_performance'],
                               color=colors, alpha=0.7)

            axes[i].set_yticks(range(len(sector_data)))
            axes[i].set_yticklabels([label[:25] + '...' if len(label) > 25 else label
                                   for label in sector_data['industry']], fontsize=9)
            axes[i].set_xlabel('Relative Performance vs Sector (%)', fontsize=10)
            axes[i].set_title(f'{sector}\n(Sector avg: {sector_data["sector_performance"].iloc[0]:.1f}%)',
                             fontsize=11, fontweight='bold')
            axes[i].grid(axis='x', alpha=0.3)
            axes[i].axvline(x=0, color='black', linestyle='-', linewidth=1)

            # Add value labels
            for j, value in enumerate(sector_data['relative_performance']):
                axes[i].text(value + (max(abs(sector_data['relative_performance'])) * 0.02), j,
                           f'{value:+.1f}%', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created industry-sector comparison: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating industry-sector comparison: {e}")
        return False

def create_industry_concentration_analysis(df: pd.DataFrame, output_path: str,
                                         title_suffix: str = "") -> bool:
    """
    Create analysis of industry concentration and diversity.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['sector', 'industry', 'market_cap']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Calculate industry statistics
        industry_stats = df.groupby('industry').agg({
            'ticker': 'count',
            'market_cap': ['sum', 'mean', 'median'],
            'sector': 'first'
        })

        # Flatten column names
        industry_stats.columns = ['_'.join(col) if col[1] else col[0] for col in industry_stats.columns]

        if industry_stats.empty:
            logger.warning("No industry concentration data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Industry Size by Stock Count
        stock_count = industry_stats['ticker'].sort_values(ascending=False).head(20)

        colors1 = plt.cm.viridis(np.linspace(0, 1, len(stock_count)))
        bars1 = ax1.barh(range(len(stock_count)), stock_count.values, color=colors1)

        ax1.set_yticks(range(len(stock_count)))
        ax1.set_yticklabels([label[:25] + '...' if len(label) > 25 else label
                           for label in stock_count.index], fontsize=9)
        ax1.set_xlabel('Number of Stocks', fontsize=11)
        ax1.set_title(f'Top 20 Industries by Stock Count {title_suffix}', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, value in enumerate(stock_count.values):
            ax1.text(value + (max(stock_count.values) * 0.01), i, f'{int(value)}',
                    va='center', fontsize=9)

        # 2. Industry Market Cap Distribution
        # Convert market cap to billions for readability
        industry_stats['market_cap_sum_billions'] = industry_stats['market_cap_sum'] / 1e9

        market_cap_top = industry_stats.nlargest(15, 'market_cap_sum')['market_cap_sum_billions']

        colors2 = plt.cm.plasma(np.linspace(0, 1, len(market_cap_top)))
        bars2 = ax2.barh(range(len(market_cap_top)), market_cap_top.values, color=colors2)

        ax2.set_yticks(range(len(market_cap_top)))
        ax2.set_yticklabels([label[:25] + '...' if len(label) > 25 else label
                           for label in market_cap_top.index], fontsize=9)
        ax2.set_xlabel('Total Market Cap (Billions USD)', fontsize=11)
        ax2.set_title('Top 15 Industries by Market Capitalization', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # 3. Industry Concentration by Sector
        sector_diversity = industry_stats.groupby('sector_')['ticker'].agg(['count', 'sum']).reset_index()
        sector_diversity['avg_stocks_per_industry'] = sector_diversity['sum'] / sector_diversity['count']
        sector_diversity = sector_diversity.sort_values('count', ascending=True)

        bars3 = ax3.barh(range(len(sector_diversity)), sector_diversity['count'],
                        color='lightcoral', alpha=0.7, label='Number of Industries')

        # Add secondary axis for average stocks per industry
        ax3_twin = ax3.twinx()
        line3 = ax3_twin.plot(sector_diversity['avg_stocks_per_industry'].values,
                             range(len(sector_diversity)), 'bo-', alpha=0.8,
                             label='Avg Stocks per Industry')

        ax3.set_yticks(range(len(sector_diversity)))
        ax3.set_yticklabels(sector_diversity['sector_'], fontsize=10)
        ax3.set_xlabel('Number of Industries', fontsize=11)
        ax3_twin.set_xlabel('Average Stocks per Industry', fontsize=11)
        ax3.set_title('Industry Diversity by Sector', fontsize=12, fontweight='bold')

        ax3.legend(loc='lower right')
        ax3_twin.legend(loc='upper right')
        ax3.grid(axis='x', alpha=0.3)

        # 4. Market Cap vs Stock Count Scatter
        # Filter industries with reasonable data
        scatter_data = industry_stats[industry_stats['ticker'] >= 2]

        ax4.scatter(scatter_data['ticker'], scatter_data['market_cap_sum_billions'],
                   s=100, alpha=0.7, c=range(len(scatter_data)), cmap='tab20')

        ax4.set_xlabel('Number of Stocks in Industry', fontsize=11)
        ax4.set_ylabel('Total Market Cap (Billions USD)', fontsize=11)
        ax4.set_title('Industry Size: Stock Count vs Market Cap', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)

        # Add labels for interesting outliers
        # High market cap, few stocks (concentrated industries)
        high_cap_low_count = scatter_data[
            (scatter_data['market_cap_sum_billions'] > scatter_data['market_cap_sum_billions'].quantile(0.8)) &
            (scatter_data['ticker'] < scatter_data['ticker'].quantile(0.5))
        ]

        for industry, row in high_cap_low_count.iterrows():
            ax4.annotate(industry[:20] + '...' if len(industry) > 20 else industry,
                        (row['ticker'], row['market_cap_sum_billions']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.3))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created industry concentration analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating industry concentration analysis: {e}")
        return False

def create_industry_top_performers_grid(df: pd.DataFrame, output_path: str,
                                      performance_col: str = 'daily_daily_yearly_252d_pct_change',
                                      title_suffix: str = "") -> bool:
    """
    Create grid showing top performing stocks by industry.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_col: Performance column to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['industry', 'ticker', performance_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Get top industries by average performance
        industry_perf = df.groupby('industry')[performance_col].mean().sort_values(ascending=False)
        top_industries = industry_perf.head(12).index.tolist()  # Top 12 for 4x3 grid

        if len(top_industries) < 4:
            logger.warning("Not enough industries for grid display")
            return False

        # Create grid layout
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        axes = axes.flatten()

        for i, industry in enumerate(top_industries):
            if i >= rows * cols:
                break

            # Get top 5 performers in this industry
            industry_data = df[df['industry'] == industry]
            top_stocks = industry_data.nlargest(5, performance_col)

            if top_stocks.empty:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{industry[:30]}...', fontsize=10, fontweight='bold')
                continue

            # Create bar chart for this industry
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_stocks)))
            bars = axes[i].barh(range(len(top_stocks)), top_stocks[performance_col], color=colors)

            axes[i].set_yticks(range(len(top_stocks)))
            axes[i].set_yticklabels(top_stocks['ticker'], fontsize=9)
            axes[i].set_xlabel('Performance (%)', fontsize=9)

            # Truncate long industry names
            industry_title = industry[:30] + '...' if len(industry) > 30 else industry
            axes[i].set_title(f'{industry_title}\n(Avg: {industry_perf[industry]:.1f}%)',
                             fontsize=10, fontweight='bold')
            axes[i].grid(axis='x', alpha=0.3)

            # Add value labels
            for j, value in enumerate(top_stocks[performance_col]):
                axes[i].text(value + (max(top_stocks[performance_col]) * 0.02), j,
                           f'{value:.1f}%', va='center', fontsize=8)

        # Hide unused subplots
        for i in range(len(top_industries), rows * cols):
            axes[i].set_visible(False)

        plt.suptitle(f'Top Performing Stocks by Industry {title_suffix}',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created industry top performers grid: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating industry top performers grid: {e}")
        return False