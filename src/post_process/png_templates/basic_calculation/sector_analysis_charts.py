#!/usr/bin/env python3
"""
Sector Analysis Charts for Basic Calculation Data
=================================================

Sector-focused analysis charts including:
- Sector performance heatmaps
- Sector ranking charts
- Cross-sector comparison analysis
- Sector rotation patterns
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
sns.set_palette("Set2")

def create_sector_performance_heatmap(df: pd.DataFrame, timeframe_cols: list, output_path: str,
                                    title_suffix: str = "") -> bool:
    """
    Create heatmap showing sector performance across multiple timeframes.

    Args:
        df: DataFrame with basic_calculation data
        timeframe_cols: List of performance column names
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if 'sector' not in df.columns:
            logger.warning("Sector column not found in DataFrame")
            return False

        # Filter available timeframe columns
        available_cols = [col for col in timeframe_cols if col in df.columns]

        if not available_cols:
            logger.warning("No timeframe columns found")
            return False

        # Group by sector and calculate mean performance
        sector_performance = df.groupby('sector')[available_cols].mean()

        if sector_performance.empty:
            logger.warning("No sector performance data available")
            return False

        # Create simplified column names for display
        display_cols = []
        for col in available_cols:
            if '1d' in col:
                display_cols.append('1 Day')
            elif '5d' in col or '5period' in col:
                display_cols.append('5 Days')
            elif '7d' in col:
                display_cols.append('1 Week')
            elif '22d' in col:
                display_cols.append('1 Month')
            elif '66d' in col:
                display_cols.append('1 Quarter')
            elif '252d' in col:
                display_cols.append('1 Year')
            else:
                display_cols.append(col.replace('daily_', '').replace('_pct_change', ''))

        sector_performance.columns = display_cols

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # 1. Heatmap
        sns.heatmap(sector_performance, annot=True, cmap='RdYlGn', center=0,
                   ax=ax1, fmt='.1f', cbar_kws={'label': 'Performance (%)'})
        ax1.set_title(f'Sector Performance Across Timeframes {title_suffix}',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Timeframe', fontsize=12)
        ax1.set_ylabel('Sector', fontsize=12)

        # Rotate x-axis labels for better readability
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)

        # 2. Sector ranking chart (using longest timeframe available)
        longest_timeframe = available_cols[-1]  # Assume last column is longest timeframe
        longest_display = display_cols[-1]

        sector_ranking = df.groupby('sector')[longest_timeframe].mean().sort_values(ascending=True)

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sector_ranking)))
        bars = ax2.barh(range(len(sector_ranking)), sector_ranking.values, color=colors)

        ax2.set_yticks(range(len(sector_ranking)))
        ax2.set_yticklabels(sector_ranking.index, fontsize=10)
        ax2.set_xlabel(f'Performance (%)', fontsize=12)
        ax2.set_title(f'Sector Ranking - {longest_display} Performance {title_suffix}',
                     fontsize=14, fontweight='bold')

        # Add value labels on bars
        for i, (sector, value) in enumerate(sector_ranking.items()):
            ax2.text(value + (max(sector_ranking.values) * 0.01), i, f'{value:.1f}%',
                    va='center', fontsize=9)

        ax2.grid(axis='x', alpha=0.3)
        ax2.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created sector performance heatmap: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating sector performance heatmap: {e}")
        return False

def create_sector_comparison_chart(df: pd.DataFrame, output_path: str,
                                 performance_col: str = 'daily_daily_yearly_252d_pct_change',
                                 title_suffix: str = "") -> bool:
    """
    Create comprehensive sector comparison with multiple metrics.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_col: Performance column to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['sector', performance_col, 'daily_rsi_14', 'atr_pct']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Group by sector and calculate statistics
        sector_stats = df.groupby('sector').agg({
            performance_col: ['mean', 'median', 'std', 'count'],
            'daily_rsi_14': 'mean',
            'atr_pct': 'mean'
        }).round(2)

        # Flatten column names
        sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns]

        if sector_stats.empty:
            logger.warning("No sector data available for comparison")
            return False

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mean Performance by Sector
        mean_perf = sector_stats[f'{performance_col}_mean'].sort_values(ascending=True)
        colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(mean_perf)))
        bars1 = ax1.barh(range(len(mean_perf)), mean_perf.values, color=colors1)

        ax1.set_yticks(range(len(mean_perf)))
        ax1.set_yticklabels(mean_perf.index, fontsize=9)
        ax1.set_xlabel('Mean Performance (%)', fontsize=11)
        ax1.set_title(f'Mean Sector Performance {title_suffix}', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, value in enumerate(mean_perf.values):
            ax1.text(value + (max(mean_perf.values) * 0.01), i, f'{value:.1f}%',
                    va='center', fontsize=8)

        # 2. Performance Volatility (Standard Deviation)
        volatility = sector_stats[f'{performance_col}_std'].sort_values(ascending=True)
        colors2 = plt.cm.Reds(np.linspace(0.3, 0.9, len(volatility)))
        bars2 = ax2.barh(range(len(volatility)), volatility.values, color=colors2)

        ax2.set_yticks(range(len(volatility)))
        ax2.set_yticklabels(volatility.index, fontsize=9)
        ax2.set_xlabel('Performance Volatility (%)', fontsize=11)
        ax2.set_title(f'Sector Performance Volatility {title_suffix}', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # 3. Risk-Return Scatter (Performance vs Volatility)
        ax3.scatter(sector_stats[f'{performance_col}_std'], sector_stats[f'{performance_col}_mean'],
                   s=sector_stats[f'{performance_col}_count']*2, alpha=0.7,
                   c=sector_stats['daily_rsi_14_mean'], cmap='coolwarm')

        ax3.set_xlabel('Performance Volatility (%)', fontsize=11)
        ax3.set_ylabel('Mean Performance (%)', fontsize=11)
        ax3.set_title('Sector Risk-Return Profile (size=count, color=RSI)', fontsize=12, fontweight='bold')

        # Add sector labels
        for sector, row in sector_stats.iterrows():
            ax3.annotate(sector, (row[f'{performance_col}_std'], row[f'{performance_col}_mean']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

        ax3.grid(alpha=0.3)

        # 4. Sector Count and Average RSI
        count_data = sector_stats[f'{performance_col}_count'].sort_values(ascending=True)
        rsi_data = sector_stats.loc[count_data.index, 'daily_rsi_14_mean']

        # Create twin axis for RSI
        ax4_twin = ax4.twinx()

        bars4 = ax4.barh(range(len(count_data)), count_data.values, alpha=0.7, color='skyblue', label='Stock Count')
        line4 = ax4_twin.plot(rsi_data.values, range(len(rsi_data)), 'ro-', alpha=0.8, label='Avg RSI')

        ax4.set_yticks(range(len(count_data)))
        ax4.set_yticklabels(count_data.index, fontsize=9)
        ax4.set_xlabel('Number of Stocks', fontsize=11)
        ax4_twin.set_xlabel('Average RSI', fontsize=11)
        ax4.set_title('Sector Stock Count vs Average RSI', fontsize=12, fontweight='bold')

        # Add legends
        ax4.legend(loc='lower right')
        ax4_twin.legend(loc='upper right')
        ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created sector comparison chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating sector comparison chart: {e}")
        return False

def create_sector_rotation_analysis(df: pd.DataFrame, output_path: str,
                                  title_suffix: str = "") -> bool:
    """
    Create sector rotation analysis showing momentum shifts across timeframes.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Required columns for rotation analysis
        rotation_cols = {
            'Short-term (1D)': 'daily_daily_daily_1d_pct_change',
            'Medium-term (1M)': 'daily_daily_monthly_22d_pct_change',
            'Long-term (1Y)': 'daily_daily_yearly_252d_pct_change'
        }

        # Filter available columns
        available_rotation = {k: v for k, v in rotation_cols.items() if v in df.columns}

        if len(available_rotation) < 2:
            logger.warning("Need at least 2 timeframes for rotation analysis")
            return False

        if 'sector' not in df.columns:
            logger.warning("Sector column not found")
            return False

        # Calculate sector rankings for each timeframe
        sector_rankings = {}
        for timeframe_name, col in available_rotation.items():
            sector_perf = df.groupby('sector')[col].mean().sort_values(ascending=False)
            sector_rankings[timeframe_name] = {sector: rank+1 for rank, sector in enumerate(sector_perf.index)}

        # Create DataFrame for ranking comparison
        ranking_df = pd.DataFrame(sector_rankings)

        if ranking_df.empty:
            logger.warning("No ranking data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Ranking change heatmap
        sns.heatmap(ranking_df, annot=True, cmap='RdYlGn_r', ax=ax1, fmt='.0f',
                   cbar_kws={'label': 'Ranking Position'})
        ax1.set_title(f'Sector Rankings Across Timeframes {title_suffix}',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Timeframe', fontsize=11)
        ax1.set_ylabel('Sector', fontsize=11)

        # 2. Ranking change analysis (if we have 3 timeframes)
        if len(available_rotation) >= 3:
            timeframes = list(ranking_df.columns)
            # Calculate ranking changes
            short_to_medium = ranking_df[timeframes[1]] - ranking_df[timeframes[0]]
            medium_to_long = ranking_df[timeframes[2]] - ranking_df[timeframes[1]]

            # Plot ranking changes
            x_pos = np.arange(len(ranking_df.index))
            width = 0.35

            bars1 = ax2.bar(x_pos - width/2, short_to_medium, width, label=f'{timeframes[0]} → {timeframes[1]}',
                           alpha=0.8)
            bars2 = ax2.bar(x_pos + width/2, medium_to_long, width, label=f'{timeframes[1]} → {timeframes[2]}',
                           alpha=0.8)

            ax2.set_xlabel('Sector', fontsize=11)
            ax2.set_ylabel('Ranking Change', fontsize=11)
            ax2.set_title('Sector Ranking Changes (negative = improvement)', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(ranking_df.index, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # 3. Performance momentum plot
        if len(available_rotation) >= 2:
            timeframes = list(available_rotation.keys())
            col1, col2 = list(available_rotation.values())[:2]

            sector_perf1 = df.groupby('sector')[col1].mean()
            sector_perf2 = df.groupby('sector')[col2].mean()

            ax3.scatter(sector_perf1, sector_perf2, s=100, alpha=0.7)

            # Add sector labels
            for sector in sector_perf1.index:
                ax3.annotate(sector, (sector_perf1[sector], sector_perf2[sector]),
                            xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

            ax3.set_xlabel(f'{timeframes[0]} Performance (%)', fontsize=11)
            ax3.set_ylabel(f'{timeframes[1]} Performance (%)', fontsize=11)
            ax3.set_title(f'Sector Performance: {timeframes[0]} vs {timeframes[1]}', fontsize=12, fontweight='bold')
            ax3.grid(alpha=0.3)

            # Add quadrant lines
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # 4. Sector momentum consistency (standard deviation of rankings)
        ranking_std = ranking_df.std(axis=1).sort_values(ascending=True)
        colors4 = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(ranking_std)))  # Green for consistent, red for volatile

        bars4 = ax4.barh(range(len(ranking_std)), ranking_std.values, color=colors4)

        ax4.set_yticks(range(len(ranking_std)))
        ax4.set_yticklabels(ranking_std.index, fontsize=10)
        ax4.set_xlabel('Ranking Volatility (Std Dev)', fontsize=11)
        ax4.set_title('Sector Ranking Consistency (lower = more consistent)', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, value in enumerate(ranking_std.values):
            ax4.text(value + (max(ranking_std.values) * 0.01), i, f'{value:.1f}',
                    va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created sector rotation analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating sector rotation analysis: {e}")
        return False

def create_sector_technical_analysis(df: pd.DataFrame, output_path: str,
                                    title_suffix: str = "") -> bool:
    """
    Create sector-level technical analysis charts.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['sector', 'daily_rsi_14', 'daily_momentum_20', 'daily_price_position_52w']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns for technical analysis: {missing_cols}")
            return False

        # Group by sector and calculate technical indicators
        sector_technical = df.groupby('sector')[['daily_rsi_14', 'daily_momentum_20', 'daily_price_position_52w']].mean()

        if sector_technical.empty:
            logger.warning("No sector technical data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. RSI by Sector
        rsi_data = sector_technical['daily_rsi_14'].sort_values(ascending=True)
        colors_rsi = ['red' if x > 70 else 'green' if x < 30 else 'orange' for x in rsi_data.values]

        bars1 = ax1.barh(range(len(rsi_data)), rsi_data.values, color=colors_rsi)

        ax1.set_yticks(range(len(rsi_data)))
        ax1.set_yticklabels(rsi_data.index, fontsize=10)
        ax1.set_xlabel('Average RSI (14-day)', fontsize=11)
        ax1.set_title(f'Sector RSI Analysis {title_suffix}', fontsize=12, fontweight='bold')

        # Add RSI zones
        ax1.axvline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax1.axvline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, value in enumerate(rsi_data.values):
            ax1.text(value + 1, i, f'{value:.0f}', va='center', fontsize=9)

        # 2. Momentum by Sector
        momentum_data = sector_technical['daily_momentum_20'].sort_values(ascending=True)
        colors_momentum = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(momentum_data)))

        bars2 = ax2.barh(range(len(momentum_data)), momentum_data.values, color=colors_momentum)

        ax2.set_yticks(range(len(momentum_data)))
        ax2.set_yticklabels(momentum_data.index, fontsize=10)
        ax2.set_xlabel('Average Momentum (20-day)', fontsize=11)
        ax2.set_title(f'Sector Momentum Analysis {title_suffix}', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)

        # 3. 52-Week Position by Sector
        position_data = sector_technical['daily_price_position_52w'].sort_values(ascending=True)
        colors_position = plt.cm.plasma(np.linspace(0.2, 0.8, len(position_data)))

        bars3 = ax3.barh(range(len(position_data)), position_data.values, color=colors_position)

        ax3.set_yticks(range(len(position_data)))
        ax3.set_yticklabels(position_data.index, fontsize=10)
        ax3.set_xlabel('Average 52-Week Position', fontsize=11)
        ax3.set_title(f'Sector 52-Week Position Analysis {title_suffix}', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # Add position zones
        ax3.axvline(0.8, color='green', linestyle='--', alpha=0.7, label='Near High (0.8)')
        ax3.axvline(0.2, color='red', linestyle='--', alpha=0.7, label='Near Low (0.2)')
        ax3.legend()

        # 4. Technical indicator correlation scatter
        ax4.scatter(sector_technical['daily_rsi_14'], sector_technical['daily_momentum_20'],
                   s=sector_technical['daily_price_position_52w']*200, alpha=0.7,
                   c=sector_technical['daily_price_position_52w'], cmap='viridis')

        # Add sector labels
        for sector, row in sector_technical.iterrows():
            ax4.annotate(sector, (row['daily_rsi_14'], row['daily_momentum_20']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

        ax4.set_xlabel('Average RSI (14-day)', fontsize=11)
        ax4.set_ylabel('Average Momentum (20-day)', fontsize=11)
        ax4.set_title('Sector Technical Indicators (size/color = 52W position)', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)

        # Add quadrant lines
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created sector technical analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating sector technical analysis: {e}")
        return False