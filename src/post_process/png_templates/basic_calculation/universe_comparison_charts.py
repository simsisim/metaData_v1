#!/usr/bin/env python3
"""
Universe Comparison Charts for Basic Calculation Data
=====================================================

Universe/index comparison charts including:
- Index membership analysis
- Performance comparison across universes
- Universe overlap analysis
- Sector representation in different universes
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
sns.set_palette("tab10")

def create_index_membership_analysis(df: pd.DataFrame, output_path: str,
                                    performance_col: str = 'daily_daily_yearly_252d_pct_change',
                                    title_suffix: str = "") -> bool:
    """
    Create index membership analysis charts.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_col: Performance column to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find boolean index columns
        index_cols = [col for col in df.columns if col in [
            'SP500', 'NASDAQ100', 'Russell1000', 'Russell3000', 'DowJonesIndustrialAverage',
            'NASDAQComposite', 'SP100', 'Russell2000'
        ]]

        if not index_cols:
            logger.warning("No index membership columns found")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Index Membership Counts
        index_counts = []
        index_names = []

        for col in index_cols:
            count = df[col].sum() if col in df.columns else 0
            index_counts.append(count)
            # Clean up index names for display
            clean_name = col.replace('DowJones', 'Dow ').replace('NASDAQ', 'NASDAQ ')
            index_names.append(clean_name)

        # Sort by count
        sorted_data = sorted(zip(index_names, index_counts), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_counts = zip(*sorted_data)

        colors1 = plt.cm.Set2(np.linspace(0, 1, len(sorted_names)))
        bars = ax1.barh(range(len(sorted_names)), sorted_counts, color=colors1, alpha=0.8)

        ax1.set_yticks(range(len(sorted_names)))
        ax1.set_yticklabels(sorted_names, fontsize=10)
        ax1.set_xlabel('Number of Stocks', fontsize=11)
        ax1.set_title(f'Index Membership Counts {title_suffix}', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, count in enumerate(sorted_counts):
            ax1.text(count + max(sorted_counts) * 0.01, i, str(int(count)), va='center', fontsize=9)

        # 2. Performance by Index Membership (top 6 indices)
        if performance_col in df.columns:
            top_indices = sorted_data[:6]  # Top 6 indices by membership
            perf_data = []

            for index_name, _ in top_indices:
                # Find original column name
                orig_col = None
                for col in index_cols:
                    clean_col = col.replace('DowJones', 'Dow ').replace('NASDAQ', 'NASDAQ ')
                    if clean_col == index_name:
                        orig_col = col
                        break

                if orig_col and orig_col in df.columns:
                    index_stocks = df[df[orig_col] == True][performance_col].dropna()
                    non_index_stocks = df[df[orig_col] == False][performance_col].dropna()

                    if not index_stocks.empty and not non_index_stocks.empty:
                        perf_data.append({
                            'index': index_name,
                            'in_index': index_stocks.mean(),
                            'not_in_index': non_index_stocks.mean(),
                            'difference': index_stocks.mean() - non_index_stocks.mean()
                        })

            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                x_pos = np.arange(len(perf_df))
                width = 0.35

                bars1 = ax2.bar(x_pos - width/2, perf_df['in_index'], width, label='In Index', alpha=0.8)
                bars2 = ax2.bar(x_pos + width/2, perf_df['not_in_index'], width, label='Not in Index', alpha=0.8)

                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(perf_df['index'], rotation=45, ha='right')
                ax2.set_ylabel('Mean Performance (%)', fontsize=11)
                ax2.set_title('Performance: Index vs Non-Index Stocks', fontsize=12, fontweight='bold')
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)

        # 3. Index Overlap Analysis
        if len(index_cols) >= 2:
            # Calculate overlap between indices
            overlap_matrix = pd.DataFrame(index=index_cols, columns=index_cols)

            for i, col1 in enumerate(index_cols):
                for j, col2 in enumerate(index_cols):
                    if col1 in df.columns and col2 in df.columns:
                        overlap = ((df[col1] == True) & (df[col2] == True)).sum()
                        col1_total = (df[col1] == True).sum()
                        if col1_total > 0:
                            overlap_pct = (overlap / col1_total) * 100
                            overlap_matrix.loc[col1, col2] = overlap_pct
                        else:
                            overlap_matrix.loc[col1, col2] = 0

            # Convert to numeric and fill NaN
            overlap_matrix = overlap_matrix.astype(float).fillna(0)

            if not overlap_matrix.empty:
                # Clean up labels for heatmap
                clean_labels = [col.replace('DowJones', 'Dow').replace('NASDAQ', 'NASDAQ')[:15]
                               for col in overlap_matrix.index]

                overlap_matrix.index = clean_labels
                overlap_matrix.columns = clean_labels

                sns.heatmap(overlap_matrix, annot=True, cmap='Blues', ax=ax3, fmt='.0f',
                           cbar_kws={'label': 'Overlap %'})
                ax3.set_title('Index Overlap Analysis (%)', fontsize=12, fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
                ax3.tick_params(axis='y', rotation=0)

        # 4. Multi-Index Membership
        if len(index_cols) >= 3:
            # Count how many indices each stock belongs to
            df_indices = df[index_cols].fillna(False)
            membership_counts = df_indices.sum(axis=1)

            membership_dist = membership_counts.value_counts().sort_index()

            colors4 = plt.cm.viridis(np.linspace(0, 1, len(membership_dist)))
            bars = ax4.bar(membership_dist.index, membership_dist.values, color=colors4, alpha=0.8)

            ax4.set_xlabel('Number of Indices Stock Belongs To', fontsize=11)
            ax4.set_ylabel('Number of Stocks', fontsize=11)
            ax4.set_title('Multi-Index Membership Distribution', fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)

            # Add value labels
            for i, (idx, value) in enumerate(membership_dist.items()):
                ax4.text(idx, value + max(membership_dist.values) * 0.01, str(value),
                        ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created index membership analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating index membership analysis: {e}")
        return False

def create_universe_performance_comparison(df: pd.DataFrame, output_path: str,
                                         performance_cols: list = None,
                                         title_suffix: str = "") -> bool:
    """
    Create performance comparison across different universes/timeframes.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        performance_cols: List of performance columns to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if performance_cols is None:
            performance_cols = [
                'daily_daily_daily_1d_pct_change',
                'daily_daily_weekly_7d_pct_change',
                'daily_daily_monthly_22d_pct_change',
                'daily_daily_quarterly_66d_pct_change',
                'daily_daily_yearly_252d_pct_change'
            ]

        # Filter available performance columns
        available_perf_cols = [col for col in performance_cols if col in df.columns]

        if not available_perf_cols:
            logger.warning("No performance columns found")
            return False

        # Find universe columns
        universe_cols = [col for col in df.columns if col in [
            'SP500', 'NASDAQ100', 'Russell1000', 'Russell3000', 'SP100'
        ]]

        if not universe_cols:
            logger.warning("No universe columns found")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Performance Heatmap by Universe and Timeframe
        performance_matrix = []
        universe_names = []

        for universe in universe_cols:
            if universe in df.columns:
                universe_stocks = df[df[universe] == True]
                universe_perf = []

                for perf_col in available_perf_cols:
                    if perf_col in universe_stocks.columns:
                        mean_perf = universe_stocks[perf_col].mean()
                        universe_perf.append(mean_perf)
                    else:
                        universe_perf.append(np.nan)

                performance_matrix.append(universe_perf)
                universe_names.append(universe.replace('NASDAQ', 'NASDAQ '))

        if performance_matrix:
            perf_df = pd.DataFrame(performance_matrix, columns=available_perf_cols, index=universe_names)

            # Clean column names for display
            clean_col_names = []
            for col in available_perf_cols:
                if '1d' in col:
                    clean_col_names.append('1 Day')
                elif '7d' in col:
                    clean_col_names.append('1 Week')
                elif '22d' in col:
                    clean_col_names.append('1 Month')
                elif '66d' in col:
                    clean_col_names.append('1 Quarter')
                elif '252d' in col:
                    clean_col_names.append('1 Year')
                else:
                    clean_col_names.append(col[-10:])

            perf_df.columns = clean_col_names

            sns.heatmap(perf_df, annot=True, cmap='RdYlGn', center=0, ax=ax1, fmt='.1f',
                       cbar_kws={'label': 'Performance %'})
            ax1.set_title(f'Universe Performance by Timeframe {title_suffix}', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)

        # 2. Performance Distribution by Universe (using longest timeframe)
        if available_perf_cols:
            longest_timeframe = available_perf_cols[-1]  # Assume last is longest
            universe_data = []
            universe_labels = []

            for universe in universe_cols:
                if universe in df.columns:
                    universe_stocks = df[df[universe] == True][longest_timeframe].dropna()
                    if not universe_stocks.empty:
                        universe_data.append(universe_stocks)
                        universe_labels.append(universe.replace('NASDAQ', 'NASDAQ '))

            if universe_data:
                box_plot = ax2.boxplot(universe_data, labels=universe_labels, patch_artist=True)

                # Color boxes
                colors2 = plt.cm.Set3(np.linspace(0, 1, len(universe_data)))
                for patch, color in zip(box_plot['boxes'], colors2):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax2.set_xlabel('Universe', fontsize=11)
                ax2.set_ylabel(f'Performance % ({longest_timeframe.split("_")[-2]})', fontsize=11)
                ax2.set_title('Performance Distribution by Universe', fontsize=12, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(alpha=0.3)

        # 3. Universe Size vs Performance
        if available_perf_cols:
            universe_stats = []

            for universe in universe_cols:
                if universe in df.columns:
                    universe_stocks = df[df[universe] == True]
                    size = len(universe_stocks)
                    perf = universe_stocks[available_perf_cols[-1]].mean()  # Use longest timeframe

                    if not np.isnan(perf):
                        universe_stats.append({
                            'universe': universe.replace('NASDAQ', 'NASDAQ '),
                            'size': size,
                            'performance': perf
                        })

            if universe_stats:
                stats_df = pd.DataFrame(universe_stats)

                ax3.scatter(stats_df['size'], stats_df['performance'], s=100, alpha=0.7,
                           c=range(len(stats_df)), cmap='viridis')

                for i, row in stats_df.iterrows():
                    ax3.annotate(row['universe'], (row['size'], row['performance']),
                                xytext=(5, 5), textcoords='offset points', fontsize=9)

                ax3.set_xlabel('Universe Size (Number of Stocks)', fontsize=11)
                ax3.set_ylabel('Mean Performance (%)', fontsize=11)
                ax3.set_title('Universe Size vs Performance', fontsize=12, fontweight='bold')
                ax3.grid(alpha=0.3)

        # 4. Sector Representation in Universes
        if 'sector' in df.columns:
            # Select top 3 universes by size
            top_universes = []
            for universe in universe_cols:
                if universe in df.columns:
                    size = (df[universe] == True).sum()
                    top_universes.append((universe, size))

            top_universes = sorted(top_universes, key=lambda x: x[1], reverse=True)[:3]

            if top_universes:
                sector_data = []

                for universe, _ in top_universes:
                    universe_sectors = df[df[universe] == True]['sector'].value_counts(normalize=True) * 100
                    sector_data.append(universe_sectors)

                # Combine sector data
                all_sectors = set()
                for sector_series in sector_data:
                    all_sectors.update(sector_series.index)

                # Create matrix
                sector_matrix = []
                universe_labels_clean = []

                for universe, _ in top_universes:
                    universe_sectors = df[df[universe] == True]['sector'].value_counts(normalize=True) * 100
                    row = [universe_sectors.get(sector, 0) for sector in all_sectors]
                    sector_matrix.append(row)
                    universe_labels_clean.append(universe.replace('NASDAQ', 'NASDAQ '))

                if sector_matrix:
                    sector_df = pd.DataFrame(sector_matrix, columns=list(all_sectors), index=universe_labels_clean)

                    # Select top sectors by average representation
                    sector_means = sector_df.mean(axis=0).sort_values(ascending=False)
                    top_sectors = sector_means.head(10).index

                    sector_df_top = sector_df[top_sectors]

                    sns.heatmap(sector_df_top, annot=True, cmap='Blues', ax=ax4, fmt='.1f',
                               cbar_kws={'label': 'Sector %'})
                    ax4.set_title('Sector Representation in Major Universes (%)', fontsize=12, fontweight='bold')
                    ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created universe performance comparison: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating universe performance comparison: {e}")
        return False

def create_universe_sector_analysis(df: pd.DataFrame, output_path: str,
                                  title_suffix: str = "") -> bool:
    """
    Create detailed sector analysis within different universes.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if 'sector' not in df.columns:
            logger.warning("Sector column not found")
            return False

        # Find universe columns
        universe_cols = [col for col in df.columns if col in [
            'SP500', 'NASDAQ100', 'Russell1000', 'DowJonesIndustrialAverage'
        ]]

        if not universe_cols:
            logger.warning("No universe columns found")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

        # 1. Sector Distribution Comparison
        sector_comparison = {}

        for universe in universe_cols:
            if universe in df.columns:
                universe_sectors = df[df[universe] == True]['sector'].value_counts()
                sector_comparison[universe.replace('DowJones', 'Dow ')] = universe_sectors

        if sector_comparison:
            comparison_df = pd.DataFrame(sector_comparison).fillna(0)

            comparison_df.plot(kind='bar', ax=ax1, alpha=0.8, width=0.8)
            ax1.set_xlabel('Sector', fontsize=11)
            ax1.set_ylabel('Number of Stocks', fontsize=11)
            ax1.set_title(f'Sector Distribution Across Universes {title_suffix}', fontsize=12, fontweight='bold')
            ax1.legend(title='Universe', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)

        # 2. Sector Concentration by Universe
        if len(universe_cols) >= 2:
            concentration_data = []

            for universe in universe_cols[:4]:  # Top 4 universes
                if universe in df.columns:
                    universe_stocks = df[df[universe] == True]
                    sector_counts = universe_stocks['sector'].value_counts()

                    # Calculate Herfindahl index for sector concentration
                    total_stocks = len(universe_stocks)
                    if total_stocks > 0:
                        sector_shares = sector_counts / total_stocks
                        hhi = (sector_shares ** 2).sum()
                        concentration_data.append({
                            'universe': universe.replace('DowJones', 'Dow ').replace('NASDAQ', 'NASDAQ '),
                            'hhi': hhi,
                            'dominant_sector': sector_counts.index[0] if not sector_counts.empty else 'None',
                            'dominant_pct': (sector_counts.iloc[0] / total_stocks * 100) if not sector_counts.empty else 0
                        })

            if concentration_data:
                conc_df = pd.DataFrame(concentration_data)

                colors = ['red' if x > 0.2 else 'orange' if x > 0.15 else 'green' for x in conc_df['hhi']]
                bars = ax2.bar(range(len(conc_df)), conc_df['hhi'], color=colors, alpha=0.8)

                ax2.set_xticks(range(len(conc_df)))
                ax2.set_xticklabels(conc_df['universe'], rotation=45)
                ax2.set_ylabel('Herfindahl Index', fontsize=11)
                ax2.set_title('Sector Concentration by Universe', fontsize=12, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)

                # Add interpretation lines
                ax2.axhline(0.15, color='green', linestyle='--', alpha=0.7, label='Diverse (0.15)')
                ax2.axhline(0.20, color='orange', linestyle='--', alpha=0.7, label='Concentrated (0.20)')
                ax2.legend()

                # Add value labels
                for i, (idx, row) in enumerate(conc_df.iterrows()):
                    ax2.text(i, row['hhi'] + 0.01, f'{row["hhi"]:.2f}', ha='center', fontsize=9)

        # 3. Technology Sector Deep Dive (if available)
        tech_universes = []
        tech_counts = []

        for universe in universe_cols:
            if universe in df.columns:
                tech_stocks = df[(df[universe] == True) & (df['sector'] == 'Electronic technology')]
                tech_count = len(tech_stocks)
                total_universe = (df[universe] == True).sum()

                if total_universe > 0:
                    tech_pct = (tech_count / total_universe) * 100
                    tech_universes.append(universe.replace('DowJones', 'Dow ').replace('NASDAQ', 'NASDAQ '))
                    tech_counts.append(tech_pct)

        if tech_universes:
            colors3 = plt.cm.Blues(np.linspace(0.4, 0.9, len(tech_universes)))
            bars = ax3.bar(range(len(tech_universes)), tech_counts, color=colors3, alpha=0.8)

            ax3.set_xticks(range(len(tech_universes)))
            ax3.set_xticklabels(tech_universes, rotation=45)
            ax3.set_ylabel('Technology Stocks (%)', fontsize=11)
            ax3.set_title('Technology Sector Representation', fontsize=12, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)

            # Add value labels
            for i, value in enumerate(tech_counts):
                ax3.text(i, value + max(tech_counts) * 0.01, f'{value:.1f}%', ha='center', fontsize=9)

        # 4. Sector Leaders by Universe
        if 'market_cap' in df.columns:
            # Find largest company in each sector for each universe
            sector_leaders = {}

            for universe in universe_cols[:2]:  # Use top 2 universes
                if universe in df.columns:
                    universe_stocks = df[df[universe] == True]
                    leaders = universe_stocks.loc[universe_stocks.groupby('sector')['market_cap'].idxmax()]

                    sector_leaders[universe.replace('DowJones', 'Dow ').replace('NASDAQ', 'NASDAQ ')] = leaders

            if sector_leaders:
                # Create comparison of market caps
                leader_data = []

                for universe, leaders in sector_leaders.items():
                    for _, stock in leaders.iterrows():
                        leader_data.append({
                            'universe': universe,
                            'sector': stock['sector'],
                            'ticker': stock['ticker'],
                            'market_cap_billions': stock['market_cap'] / 1e9
                        })

                if leader_data:
                    leader_df = pd.DataFrame(leader_data)

                    # Pivot for comparison
                    pivot_df = leader_df.pivot_table(index='sector', columns='universe',
                                                   values='market_cap_billions', fill_value=0)

                    pivot_df.plot(kind='bar', ax=ax4, alpha=0.8, width=0.8)
                    ax4.set_xlabel('Sector', fontsize=11)
                    ax4.set_ylabel('Market Cap of Sector Leader (Billions)', fontsize=11)
                    ax4.set_title('Sector Leaders Market Cap by Universe', fontsize=12, fontweight='bold')
                    ax4.legend(title='Universe')
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created universe sector analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating universe sector analysis: {e}")
        return False