#!/usr/bin/env python3
"""
Performance Charts for Basic Calculation Data
=============================================

Multi-timeframe performance analysis charts including:
- Top performers by timeframe
- Performance distribution histograms
- Multi-timeframe comparison charts
- Performance correlation analysis
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
sns.set_palette("husl")

def create_top_performers_chart(df: pd.DataFrame, timeframe_col: str, output_path: str,
                               top_n: int = 10, title_suffix: str = "") -> bool:
    """
    Create horizontal bar chart of top N performers for a specific timeframe.

    Args:
        df: DataFrame with basic_calculation data
        timeframe_col: Column name for performance data (e.g., 'daily_daily_yearly_252d_pct_change')
        output_path: Path to save PNG file
        top_n: Number of top performers to show
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if timeframe_col not in df.columns:
            logger.warning(f"Column {timeframe_col} not found in DataFrame")
            return False

        # Get top performers
        top_performers = df.nlargest(top_n, timeframe_col)

        if top_performers.empty:
            logger.warning(f"No data found for {timeframe_col}")
            return False

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create horizontal bar chart
        bars = ax.barh(range(len(top_performers)),
                      top_performers[timeframe_col],
                      color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_performers))))

        # Customize chart
        ax.set_yticks(range(len(top_performers)))
        ax.set_yticklabels(top_performers['ticker'], fontsize=10)
        ax.set_xlabel('Performance (%)', fontsize=12)
        ax.set_title(f'Top {top_n} Performers - {timeframe_col.replace("daily_", "").replace("_", " ").title()} {title_suffix}',
                    fontsize=14, fontweight='bold')

        # Add value labels on bars
        for i, (idx, row) in enumerate(top_performers.iterrows()):
            value = row[timeframe_col]
            ax.text(value + 0.1, i, f'{value:.1f}%',
                   va='center', fontsize=9)

        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created top performers chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating top performers chart: {e}")
        return False

def create_performance_distribution_chart(df: pd.DataFrame, timeframe_col: str, output_path: str,
                                        title_suffix: str = "") -> bool:
    """
    Create performance distribution histogram with statistics.

    Args:
        df: DataFrame with basic_calculation data
        timeframe_col: Column name for performance data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if timeframe_col not in df.columns:
            logger.warning(f"Column {timeframe_col} not found in DataFrame")
            return False

        # Filter out NaN values
        data = df[timeframe_col].dropna()

        if data.empty:
            logger.warning(f"No valid data found for {timeframe_col}")
            return False

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}%')
        ax1.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.1f}%')
        ax1.set_xlabel('Performance (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Performance Distribution - {timeframe_col.replace("daily_", "").replace("_", " ").title()}',
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Box plot
        ax2.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('Performance (%)', fontsize=12)
        ax2.set_title('Performance Distribution Box Plot', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

        # Add statistics text
        stats_text = f"""Statistics:
Count: {len(data):,}
Mean: {data.mean():.1f}%
Median: {data.median():.1f}%
Std Dev: {data.std():.1f}%
Min: {data.min():.1f}%
Max: {data.max():.1f}%"""

        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created performance distribution chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating performance distribution chart: {e}")
        return False

def create_multi_timeframe_comparison_chart(df: pd.DataFrame, output_path: str,
                                          title_suffix: str = "") -> bool:
    """
    Create scatter plots comparing different timeframe performances.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define timeframe columns
        timeframe_cols = [
            'daily_daily_daily_1d_pct_change',
            'daily_5period_pct_change',
            'daily_daily_weekly_7d_pct_change',
            'daily_daily_monthly_22d_pct_change',
            'daily_daily_quarterly_66d_pct_change',
            'daily_daily_yearly_252d_pct_change'
        ]

        # Filter available columns
        available_cols = [col for col in timeframe_cols if col in df.columns]

        if len(available_cols) < 2:
            logger.warning("Need at least 2 timeframe columns for comparison")
            return False

        # Create correlation matrix
        corr_data = df[available_cols].corr()

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Correlation heatmap
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   ax=ax1, square=True, fmt='.2f')
        ax1.set_title('Performance Correlation Matrix', fontsize=12, fontweight='bold')

        # 2. Short vs Long term scatter
        if 'daily_daily_daily_1d_pct_change' in available_cols and 'daily_daily_yearly_252d_pct_change' in available_cols:
            ax2.scatter(df['daily_daily_daily_1d_pct_change'], df['daily_daily_yearly_252d_pct_change'],
                       alpha=0.6, s=30)
            ax2.set_xlabel('1-Day Performance (%)', fontsize=11)
            ax2.set_ylabel('1-Year Performance (%)', fontsize=11)
            ax2.set_title('Short-term vs Long-term Performance', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)

            # Add trend line
            z = np.polyfit(df['daily_daily_daily_1d_pct_change'].dropna(),
                          df['daily_daily_yearly_252d_pct_change'].dropna(), 1)
            p = np.poly1d(z)
            ax2.plot(df['daily_daily_daily_1d_pct_change'], p(df['daily_daily_daily_1d_pct_change']),
                    "r--", alpha=0.8)

        # 3. Medium term comparison
        if 'daily_daily_monthly_22d_pct_change' in available_cols and 'daily_daily_quarterly_66d_pct_change' in available_cols:
            ax3.scatter(df['daily_daily_monthly_22d_pct_change'], df['daily_daily_quarterly_66d_pct_change'],
                       alpha=0.6, s=30, color='orange')
            ax3.set_xlabel('Monthly Performance (%)', fontsize=11)
            ax3.set_ylabel('Quarterly Performance (%)', fontsize=11)
            ax3.set_title('Monthly vs Quarterly Performance', fontsize=12, fontweight='bold')
            ax3.grid(alpha=0.3)

        # 4. Performance ranking comparison
        if len(available_cols) >= 3:
            # Create performance rankings for top 3 timeframes
            ranking_data = []
            for col in available_cols[:3]:
                col_short = col.replace('daily_', '').replace('_pct_change', '')
                top_10 = df.nlargest(10, col)['ticker'].tolist()
                ranking_data.extend([(ticker, rank+1, col_short) for rank, ticker in enumerate(top_10)])

            ranking_df = pd.DataFrame(ranking_data, columns=['Ticker', 'Rank', 'Timeframe'])

            # Create ranking comparison chart
            for i, timeframe in enumerate(ranking_df['Timeframe'].unique()):
                tf_data = ranking_df[ranking_df['Timeframe'] == timeframe]
                ax4.scatter([i] * len(tf_data), tf_data['Rank'], alpha=0.7, s=50)

            ax4.set_xlabel('Timeframe', fontsize=11)
            ax4.set_ylabel('Ranking Position', fontsize=11)
            ax4.set_title('Top Performer Rankings Across Timeframes', fontsize=12, fontweight='bold')
            ax4.set_xticks(range(len(ranking_df['Timeframe'].unique())))
            ax4.set_xticklabels(ranking_df['Timeframe'].unique(), rotation=45)
            ax4.invert_yaxis()
            ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created multi-timeframe comparison chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating multi-timeframe comparison chart: {e}")
        return False

def create_performance_momentum_chart(df: pd.DataFrame, output_path: str,
                                    title_suffix: str = "") -> bool:
    """
    Create chart showing performance vs momentum indicators.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Required columns
        required_cols = ['daily_daily_yearly_252d_pct_change', 'daily_rsi_14', 'daily_momentum_20']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid data for performance momentum analysis")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Performance vs RSI
        scatter1 = ax1.scatter(valid_data['daily_rsi_14'], valid_data['daily_daily_yearly_252d_pct_change'],
                              alpha=0.6, s=30, c=valid_data['daily_momentum_20'],
                              cmap='RdYlGn', vmin=-50, vmax=50)
        ax1.set_xlabel('RSI (14-day)', fontsize=11)
        ax1.set_ylabel('1-Year Performance (%)', fontsize=11)
        ax1.set_title('Performance vs RSI (colored by Momentum)', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Momentum')

        # Add RSI zones
        ax1.axvline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax1.axvline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax1.legend()

        # 2. Performance vs Momentum
        ax2.scatter(valid_data['daily_momentum_20'], valid_data['daily_daily_yearly_252d_pct_change'],
                   alpha=0.6, s=30, color='purple')
        ax2.set_xlabel('Momentum (20-day)', fontsize=11)
        ax2.set_ylabel('1-Year Performance (%)', fontsize=11)
        ax2.set_title('Performance vs Momentum', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

        # Add trend line
        z = np.polyfit(valid_data['daily_momentum_20'].dropna(),
                      valid_data['daily_daily_yearly_252d_pct_change'].dropna(), 1)
        p = np.poly1d(z)
        ax2.plot(valid_data['daily_momentum_20'], p(valid_data['daily_momentum_20']),
                "r--", alpha=0.8)

        # 3. RSI Distribution by Performance Quartiles
        performance_quartiles = pd.qcut(valid_data['daily_daily_yearly_252d_pct_change'],
                                      q=4, labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)'])

        quartile_data = [valid_data[performance_quartiles == q]['daily_rsi_14'].dropna()
                        for q in performance_quartiles.cat.categories]

        ax3.boxplot(quartile_data, labels=performance_quartiles.cat.categories)
        ax3.set_xlabel('Performance Quartiles', fontsize=11)
        ax3.set_ylabel('RSI (14-day)', fontsize=11)
        ax3.set_title('RSI Distribution by Performance Quartiles', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)

        # 4. Momentum Distribution by Performance Quartiles
        momentum_data = [valid_data[performance_quartiles == q]['daily_momentum_20'].dropna()
                        for q in performance_quartiles.cat.categories]

        ax4.boxplot(momentum_data, labels=performance_quartiles.cat.categories)
        ax4.set_xlabel('Performance Quartiles', fontsize=11)
        ax4.set_ylabel('Momentum (20-day)', fontsize=11)
        ax4.set_title('Momentum Distribution by Performance Quartiles', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created performance momentum chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating performance momentum chart: {e}")
        return False