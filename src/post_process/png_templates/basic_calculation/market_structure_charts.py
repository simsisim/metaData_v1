#!/usr/bin/env python3
"""
Market Structure Charts for Basic Calculation Data
==================================================

Market structure analysis charts including:
- Market cap analysis
- Exchange distribution
- Analyst rating analysis
- Market concentration metrics
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
sns.set_palette("Set3")

def create_market_cap_analysis(df: pd.DataFrame, output_path: str,
                              performance_col: str = 'daily_daily_yearly_252d_pct_change',
                              title_suffix: str = "") -> bool:
    """
    Create market capitalization analysis charts.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_col: Performance column to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['market_cap', performance_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid market cap data available")
            return False

        # Create market cap categories
        valid_data = valid_data.copy()
        valid_data['market_cap_billions'] = valid_data['market_cap'] / 1e9

        # Define market cap categories
        valid_data['cap_category'] = pd.cut(valid_data['market_cap'],
                                          bins=[0, 2e9, 10e9, 50e9, 200e9, np.inf],
                                          labels=['Micro Cap', 'Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap'])

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Market Cap Distribution
        ax1.hist(np.log10(valid_data['market_cap_billions']), bins=40, alpha=0.7,
                color='lightblue', edgecolor='black')
        ax1.set_xlabel('Market Cap (Log10 Billions USD)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Market Cap Distribution {title_suffix}', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)

        # Add vertical lines for category boundaries
        boundaries = [0.002, 2, 10, 50, 200]  # in billions
        boundary_labels = ['Micro', 'Small', 'Mid', 'Large', 'Mega']
        colors_bound = ['red', 'orange', 'yellow', 'green', 'blue']

        for boundary, label, color in zip(boundaries, boundary_labels, colors_bound):
            if boundary > 0:
                ax1.axvline(np.log10(boundary), color=color, linestyle='--', alpha=0.7, label=label)

        ax1.legend(title='Cap Categories')

        # 2. Performance by Market Cap Category
        cap_performance = valid_data.groupby('cap_category')[performance_col].agg(['mean', 'median', 'std'])

        # Remove NaN categories
        cap_performance = cap_performance.dropna()

        if not cap_performance.empty:
            x_pos = np.arange(len(cap_performance.index))

            bars = ax2.bar(x_pos, cap_performance['mean'], alpha=0.8,
                          color=['red', 'orange', 'yellow', 'green', 'blue'][:len(cap_performance)],
                          yerr=cap_performance['std'], capsize=5)

            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(cap_performance.index, rotation=45)
            ax2.set_ylabel('Mean Performance (%)', fontsize=11)
            ax2.set_title('Performance by Market Cap Category', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)

            # Add value labels
            for i, value in enumerate(cap_performance['mean']):
                ax2.text(i, value + cap_performance['std'].iloc[i] + 1, f'{value:.1f}%',
                        ha='center', fontsize=9)

        # 3. Market Cap vs Performance Scatter
        # Use log scale for better visualization
        ax3.scatter(valid_data['market_cap_billions'], valid_data[performance_col],
                   alpha=0.6, s=30, c=valid_data['market_cap_billions'], cmap='viridis')

        ax3.set_xscale('log')
        ax3.set_xlabel('Market Cap (Billions USD, Log Scale)', fontsize=11)
        ax3.set_ylabel('Performance (%)', fontsize=11)
        ax3.set_title('Market Cap vs Performance', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('Market Cap (Billions)', fontsize=10)

        # 4. Market Cap Category Distribution
        cap_counts = valid_data['cap_category'].value_counts()

        colors4 = ['red', 'orange', 'yellow', 'green', 'blue'][:len(cap_counts)]
        wedges, texts, autotexts = ax4.pie(cap_counts.values, labels=cap_counts.index,
                                          autopct='%1.1f%%', colors=colors4, startangle=90)

        ax4.set_title(f'Market Cap Category Distribution\n(Total: {len(valid_data):,} stocks)',
                     fontsize=12, fontweight='bold')

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created market cap analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating market cap analysis: {e}")
        return False

def create_exchange_analysis(df: pd.DataFrame, output_path: str,
                           performance_col: str = 'daily_daily_yearly_252d_pct_change',
                           title_suffix: str = "") -> bool:
    """
    Create exchange distribution and performance analysis.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_col: Performance column to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['exchange', performance_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid exchange data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Exchange Distribution
        exchange_counts = valid_data['exchange'].value_counts()

        colors1 = plt.cm.Set2(np.linspace(0, 1, len(exchange_counts)))
        wedges, texts, autotexts = ax1.pie(exchange_counts.values, labels=exchange_counts.index,
                                          autopct='%1.1f%%', colors=colors1, startangle=90)

        ax1.set_title(f'Exchange Distribution {title_suffix}\n(Total: {len(valid_data):,} stocks)',
                     fontsize=12, fontweight='bold')

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')

        # 2. Performance by Exchange
        exchange_performance = valid_data.groupby('exchange')[performance_col].agg(['mean', 'median', 'std', 'count'])

        bars = ax2.bar(range(len(exchange_performance)), exchange_performance['mean'],
                      alpha=0.8, color=colors1[:len(exchange_performance)],
                      yerr=exchange_performance['std'], capsize=5)

        ax2.set_xticks(range(len(exchange_performance)))
        ax2.set_xticklabels(exchange_performance.index)
        ax2.set_ylabel('Mean Performance (%)', fontsize=11)
        ax2.set_title('Performance by Exchange', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (exchange, row) in enumerate(exchange_performance.iterrows()):
            ax2.text(i, row['mean'] + row['std'] + 1,
                    f'{row["mean"]:.1f}%\n(n={int(row["count"])})',
                    ha='center', fontsize=9)

        # 3. Exchange Performance Distribution
        exchanges = valid_data['exchange'].unique()
        exchange_data = [valid_data[valid_data['exchange'] == ex][performance_col].dropna()
                        for ex in exchanges]

        box_plot = ax3.boxplot(exchange_data, labels=exchanges, patch_artist=True)

        # Color boxes
        for patch, color in zip(box_plot['boxes'], colors1):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_xlabel('Exchange', fontsize=11)
        ax3.set_ylabel('Performance (%)', fontsize=11)
        ax3.set_title('Performance Distribution by Exchange', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)

        # 4. Exchange vs Market Cap (if available)
        if 'market_cap' in valid_data.columns:
            exchange_market_cap = valid_data.groupby('exchange')['market_cap'].agg(['mean', 'median', 'sum'])
            exchange_market_cap['mean_billions'] = exchange_market_cap['mean'] / 1e9
            exchange_market_cap['median_billions'] = exchange_market_cap['median'] / 1e9

            x_pos = np.arange(len(exchange_market_cap.index))
            width = 0.35

            bars1 = ax4.bar(x_pos - width/2, exchange_market_cap['mean_billions'], width,
                           label='Mean Market Cap', alpha=0.8, color='skyblue')
            bars2 = ax4.bar(x_pos + width/2, exchange_market_cap['median_billions'], width,
                           label='Median Market Cap', alpha=0.8, color='lightcoral')

            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(exchange_market_cap.index)
            ax4.set_ylabel('Market Cap (Billions USD)', fontsize=11)
            ax4.set_title('Market Cap by Exchange', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created exchange analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating exchange analysis: {e}")
        return False

def create_analyst_rating_analysis(df: pd.DataFrame, output_path: str,
                                 performance_col: str = 'daily_daily_yearly_252d_pct_change',
                                 title_suffix: str = "") -> bool:
    """
    Create analyst rating distribution and performance analysis.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_col: Performance column to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['analyst rating', performance_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid analyst rating data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Analyst Rating Distribution
        rating_counts = valid_data['analyst rating'].value_counts()

        # Define colors for different ratings
        rating_color_map = {
            'Strong buy': 'darkgreen',
            'Buy': 'green',
            'Neutral': 'yellow',
            'Sell': 'orange',
            'Strong sell': 'red'
        }

        colors1 = [rating_color_map.get(rating, 'gray') for rating in rating_counts.index]

        wedges, texts, autotexts = ax1.pie(rating_counts.values, labels=rating_counts.index,
                                          autopct='%1.1f%%', colors=colors1, startangle=90)

        ax1.set_title(f'Analyst Rating Distribution {title_suffix}\n(Total: {len(valid_data):,} stocks)',
                     fontsize=12, fontweight='bold')

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')

        # 2. Performance by Analyst Rating
        rating_performance = valid_data.groupby('analyst rating')[performance_col].agg(['mean', 'median', 'std', 'count'])

        colors2 = [rating_color_map.get(rating, 'gray') for rating in rating_performance.index]

        bars = ax2.bar(range(len(rating_performance)), rating_performance['mean'],
                      alpha=0.8, color=colors2,
                      yerr=rating_performance['std'], capsize=5)

        ax2.set_xticks(range(len(rating_performance)))
        ax2.set_xticklabels(rating_performance.index, rotation=45)
        ax2.set_ylabel('Mean Performance (%)', fontsize=11)
        ax2.set_title('Performance by Analyst Rating', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (rating, row) in enumerate(rating_performance.iterrows()):
            ax2.text(i, row['mean'] + row['std'] + 1,
                    f'{row["mean"]:.1f}%\n(n={int(row["count"])})',
                    ha='center', fontsize=9)

        # 3. Rating Performance Distribution
        ratings = valid_data['analyst rating'].unique()
        rating_data = [valid_data[valid_data['analyst rating'] == rating][performance_col].dropna()
                      for rating in ratings]

        box_plot = ax3.boxplot(rating_data, labels=ratings, patch_artist=True)

        # Color boxes according to rating
        for patch, rating in zip(box_plot['boxes'], ratings):
            patch.set_facecolor(rating_color_map.get(rating, 'gray'))
            patch.set_alpha(0.7)

        ax3.set_xlabel('Analyst Rating', fontsize=11)
        ax3.set_ylabel('Performance (%)', fontsize=11)
        ax3.set_title('Performance Distribution by Rating', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(alpha=0.3)

        # 4. Rating by Sector (if available)
        if 'sector' in valid_data.columns:
            # Create rating distribution by sector
            rating_sector = valid_data.groupby('sector')['analyst rating'].value_counts(normalize=True).unstack()
            rating_sector = rating_sector.fillna(0) * 100  # Convert to percentages

            # Select top sectors by stock count
            top_sectors = valid_data['sector'].value_counts().head(8).index
            rating_sector_top = rating_sector.loc[top_sectors]

            rating_sector_top.plot(kind='barh', stacked=True, ax=ax4,
                                  color=[rating_color_map.get(col, 'gray') for col in rating_sector_top.columns])

            ax4.set_xlabel('Percentage of Stocks', fontsize=11)
            ax4.set_ylabel('Sector', fontsize=11)
            ax4.set_title('Analyst Rating Distribution by Sector (%)', fontsize=12, fontweight='bold')
            ax4.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created analyst rating analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating analyst rating analysis: {e}")
        return False

def create_market_concentration_analysis(df: pd.DataFrame, output_path: str,
                                       title_suffix: str = "") -> bool:
    """
    Create market concentration and diversity analysis.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['sector', 'market_cap']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid concentration data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Market Cap Concentration by Sector
        sector_market_cap = valid_data.groupby('sector').agg({
            'market_cap': ['sum', 'count'],
            'ticker': 'count'
        })

        # Flatten column names
        sector_market_cap.columns = ['_'.join(col) for col in sector_market_cap.columns]
        sector_market_cap['market_cap_sum_billions'] = sector_market_cap['market_cap_sum'] / 1e9

        # Sort by total market cap
        sector_market_cap = sector_market_cap.sort_values('market_cap_sum_billions', ascending=True)

        colors1 = plt.cm.viridis(np.linspace(0, 1, len(sector_market_cap)))
        bars = ax1.barh(range(len(sector_market_cap)), sector_market_cap['market_cap_sum_billions'],
                       color=colors1)

        ax1.set_yticks(range(len(sector_market_cap)))
        ax1.set_yticklabels(sector_market_cap.index, fontsize=10)
        ax1.set_xlabel('Total Market Cap (Billions USD)', fontsize=11)
        ax1.set_title(f'Market Cap Concentration by Sector {title_suffix}', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, value in enumerate(sector_market_cap['market_cap_sum_billions']):
            ax1.text(value + max(sector_market_cap['market_cap_sum_billions']) * 0.01, i,
                    f'${value:.0f}B', va='center', fontsize=9)

        # 2. Stock Count vs Market Cap by Sector
        ax2.scatter(sector_market_cap['ticker_count'], sector_market_cap['market_cap_sum_billions'],
                   s=100, alpha=0.7, c=range(len(sector_market_cap)), cmap='plasma')

        ax2.set_xlabel('Number of Stocks', fontsize=11)
        ax2.set_ylabel('Total Market Cap (Billions USD)', fontsize=11)
        ax2.set_title('Sector Concentration: Stock Count vs Market Cap', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

        # Add sector labels
        for sector, row in sector_market_cap.iterrows():
            ax2.annotate(sector[:15], (row['ticker_count'], row['market_cap_sum_billions']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

        # 3. Market Cap Herfindahl Index by Sector
        # Calculate concentration within each sector
        sector_hhi = []
        sector_names = []

        for sector in valid_data['sector'].unique():
            sector_data = valid_data[valid_data['sector'] == sector]
            if len(sector_data) > 1:
                # Calculate market share within sector
                total_sector_cap = sector_data['market_cap'].sum()
                sector_data = sector_data.copy()
                sector_data['market_share'] = sector_data['market_cap'] / total_sector_cap

                # Calculate HHI (sum of squared market shares)
                hhi = (sector_data['market_share'] ** 2).sum()
                sector_hhi.append(hhi)
                sector_names.append(sector)

        if sector_hhi:
            hhi_df = pd.DataFrame({'sector': sector_names, 'hhi': sector_hhi})
            hhi_df = hhi_df.sort_values('hhi', ascending=True)

            colors3 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(hhi_df)))
            bars = ax3.barh(range(len(hhi_df)), hhi_df['hhi'], color=colors3)

            ax3.set_yticks(range(len(hhi_df)))
            ax3.set_yticklabels(hhi_df['sector'], fontsize=10)
            ax3.set_xlabel('Herfindahl Index (concentration)', fontsize=11)
            ax3.set_title('Market Concentration Within Sectors', fontsize=12, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)

            # Add interpretation lines
            ax3.axvline(0.15, color='green', linestyle='--', alpha=0.7, label='Competitive (0.15)')
            ax3.axvline(0.25, color='orange', linestyle='--', alpha=0.7, label='Moderately Concentrated (0.25)')
            ax3.legend()

        # 4. Top Companies by Market Cap
        if len(valid_data) > 10:
            top_companies = valid_data.nlargest(15, 'market_cap')[['ticker', 'market_cap', 'sector']]
            top_companies['market_cap_billions'] = top_companies['market_cap'] / 1e9

            # Color by sector
            sectors = top_companies['sector'].unique()
            sector_colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))
            color_map = dict(zip(sectors, sector_colors))
            colors4 = [color_map[sector] for sector in top_companies['sector']]

            bars = ax4.barh(range(len(top_companies)), top_companies['market_cap_billions'],
                           color=colors4, alpha=0.8)

            ax4.set_yticks(range(len(top_companies)))
            ax4.set_yticklabels(top_companies['ticker'], fontsize=10)
            ax4.set_xlabel('Market Cap (Billions USD)', fontsize=11)
            ax4.set_title('Top 15 Companies by Market Cap', fontsize=12, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (idx, row) in enumerate(top_companies.iterrows()):
                ax4.text(row['market_cap_billions'] + max(top_companies['market_cap_billions']) * 0.01, i,
                        f'${row["market_cap_billions"]:.0f}B', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created market concentration analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating market concentration analysis: {e}")
        return False