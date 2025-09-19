#!/usr/bin/env python3
"""
Technical Analysis Charts for Basic Calculation Data
====================================================

Technical analysis charts including:
- RSI analysis and distributions
- MACD signal analysis
- Momentum indicators
- Moving average alignments
- Price position analysis
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
sns.set_palette("viridis")

def create_rsi_analysis_chart(df: pd.DataFrame, output_path: str,
                             title_suffix: str = "") -> bool:
    """
    Create comprehensive RSI analysis charts.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['daily_rsi_14', 'sector']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=['daily_rsi_14'])

        if valid_data.empty:
            logger.warning("No valid RSI data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. RSI Distribution
        ax1.hist(valid_data['daily_rsi_14'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(valid_data['daily_rsi_14'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_data["daily_rsi_14"].mean():.1f}')
        ax1.axvline(valid_data['daily_rsi_14'].median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {valid_data["daily_rsi_14"].median():.1f}')

        # Add RSI zones
        ax1.axvline(70, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Overbought (70)')
        ax1.axvline(30, color='blue', linestyle=':', linewidth=2, alpha=0.7, label='Oversold (30)')

        ax1.set_xlabel('RSI (14-day)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'RSI Distribution {title_suffix}', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. RSI by Sector
        sector_rsi = valid_data.groupby('sector')['daily_rsi_14'].mean().sort_values(ascending=True)

        colors = ['red' if x > 70 else 'blue' if x < 30 else 'green' for x in sector_rsi.values]
        bars = ax2.barh(range(len(sector_rsi)), sector_rsi.values, color=colors, alpha=0.8)

        ax2.set_yticks(range(len(sector_rsi)))
        ax2.set_yticklabels(sector_rsi.index, fontsize=10)
        ax2.set_xlabel('Average RSI (14-day)', fontsize=11)
        ax2.set_title('RSI by Sector', fontsize=12, fontweight='bold')

        # Add RSI zones
        ax2.axvline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axvline(30, color='blue', linestyle='--', alpha=0.7, label='Oversold')
        ax2.axvline(50, color='gray', linestyle='-', alpha=0.5, label='Neutral')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, value in enumerate(sector_rsi.values):
            ax2.text(value + 1, i, f'{value:.0f}', va='center', fontsize=9)

        # 3. RSI Zone Analysis
        rsi_zones = pd.cut(valid_data['daily_rsi_14'],
                          bins=[0, 30, 70, 100],
                          labels=['Oversold (<30)', 'Neutral (30-70)', 'Overbought (>70)'])

        zone_counts = rsi_zones.value_counts()
        colors3 = ['blue', 'green', 'red']

        wedges, texts, autotexts = ax3.pie(zone_counts.values, labels=zone_counts.index,
                                          autopct='%1.1f%%', colors=colors3, startangle=90)

        ax3.set_title(f'RSI Zone Distribution\n(Total Stocks: {len(valid_data):,})',
                     fontsize=12, fontweight='bold')

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')

        # 4. RSI vs Performance (if available)
        if 'daily_daily_yearly_252d_pct_change' in valid_data.columns:
            performance_data = valid_data.dropna(subset=['daily_daily_yearly_252d_pct_change'])

            ax4.scatter(performance_data['daily_rsi_14'], performance_data['daily_daily_yearly_252d_pct_change'],
                       alpha=0.6, s=30)

            ax4.set_xlabel('RSI (14-day)', fontsize=11)
            ax4.set_ylabel('1-Year Performance (%)', fontsize=11)
            ax4.set_title('RSI vs Performance Relationship', fontsize=12, fontweight='bold')
            ax4.grid(alpha=0.3)

            # Add RSI zone lines
            ax4.axvline(70, color='red', linestyle='--', alpha=0.5)
            ax4.axvline(30, color='blue', linestyle='--', alpha=0.5)

            # Add trend line
            if len(performance_data) > 10:
                z = np.polyfit(performance_data['daily_rsi_14'].dropna(),
                              performance_data['daily_daily_yearly_252d_pct_change'].dropna(), 1)
                p = np.poly1d(z)
                ax4.plot(performance_data['daily_rsi_14'], p(performance_data['daily_rsi_14']),
                        "r--", alpha=0.8, label='Trend')
                ax4.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created RSI analysis chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating RSI analysis chart: {e}")
        return False

def create_macd_analysis_chart(df: pd.DataFrame, output_path: str,
                              title_suffix: str = "") -> bool:
    """
    Create MACD signal analysis charts.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['daily_macd', 'daily_macd_signal', 'daily_macd_histogram']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid MACD data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. MACD Histogram Distribution
        ax1.hist(valid_data['daily_macd_histogram'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(valid_data['daily_macd_histogram'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_data["daily_macd_histogram"].mean():.2f}')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero Line')

        ax1.set_xlabel('MACD Histogram', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'MACD Histogram Distribution {title_suffix}', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. MACD vs Signal Scatter
        ax2.scatter(valid_data['daily_macd'], valid_data['daily_macd_signal'],
                   alpha=0.6, s=30, c=valid_data['daily_macd_histogram'], cmap='RdYlGn')

        ax2.set_xlabel('MACD Line', fontsize=11)
        ax2.set_ylabel('MACD Signal Line', fontsize=11)
        ax2.set_title('MACD vs Signal (color = histogram)', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

        # Add diagonal line (MACD = Signal)
        min_val = min(valid_data['daily_macd'].min(), valid_data['daily_macd_signal'].min())
        max_val = max(valid_data['daily_macd'].max(), valid_data['daily_macd_signal'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='MACD = Signal')
        ax2.legend()

        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('MACD Histogram', fontsize=10)

        # 3. MACD Signal Categories
        # Classify MACD signals
        valid_data = valid_data.copy()
        valid_data['macd_signal_category'] = 'Neutral'
        valid_data.loc[valid_data['daily_macd_histogram'] > 0, 'macd_signal_category'] = 'Bullish'
        valid_data.loc[valid_data['daily_macd_histogram'] < 0, 'macd_signal_category'] = 'Bearish'

        signal_counts = valid_data['macd_signal_category'].value_counts()
        colors3 = ['green', 'red', 'gray']

        wedges, texts, autotexts = ax3.pie(signal_counts.values, labels=signal_counts.index,
                                          autopct='%1.1f%%', colors=colors3, startangle=90)

        ax3.set_title(f'MACD Signal Distribution\n(Total Stocks: {len(valid_data):,})',
                     fontsize=12, fontweight='bold')

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')

        # 4. MACD by Sector
        if 'sector' in valid_data.columns:
            sector_macd = valid_data.groupby('sector')['daily_macd_histogram'].mean().sort_values(ascending=True)

            colors4 = ['red' if x < 0 else 'green' for x in sector_macd.values]
            bars = ax4.barh(range(len(sector_macd)), sector_macd.values, color=colors4, alpha=0.8)

            ax4.set_yticks(range(len(sector_macd)))
            ax4.set_yticklabels(sector_macd.index, fontsize=10)
            ax4.set_xlabel('Average MACD Histogram', fontsize=11)
            ax4.set_title('MACD Momentum by Sector', fontsize=12, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            ax4.axvline(0, color='black', linestyle='-', linewidth=1)

            # Add value labels
            for i, value in enumerate(sector_macd.values):
                ax4.text(value + (max(abs(sector_macd.values)) * 0.02), i,
                        f'{value:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created MACD analysis chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating MACD analysis chart: {e}")
        return False

def create_momentum_analysis_chart(df: pd.DataFrame, output_path: str,
                                  title_suffix: str = "") -> bool:
    """
    Create momentum indicators analysis charts.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['daily_momentum_20', 'daily_price_position_52w']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid momentum data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Momentum Distribution
        ax1.hist(valid_data['daily_momentum_20'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax1.axvline(valid_data['daily_momentum_20'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_data["daily_momentum_20"].mean():.1f}')
        ax1.axvline(valid_data['daily_momentum_20'].median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {valid_data["daily_momentum_20"].median():.1f}')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero')

        ax1.set_xlabel('Momentum (20-day)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Momentum Distribution {title_suffix}', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. 52-Week Position Distribution
        ax2.hist(valid_data['daily_price_position_52w'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(valid_data['daily_price_position_52w'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_data["daily_price_position_52w"].mean():.2f}')
        ax2.axvline(0.5, color='gray', linestyle='-', linewidth=1, label='Mid-range')

        # Add position zones
        ax2.axvline(0.8, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Near High (0.8)')
        ax2.axvline(0.2, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Near Low (0.2)')

        ax2.set_xlabel('52-Week Price Position (0=Low, 1=High)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('52-Week Position Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Momentum vs 52-Week Position
        ax3.scatter(valid_data['daily_momentum_20'], valid_data['daily_price_position_52w'],
                   alpha=0.6, s=30)

        ax3.set_xlabel('Momentum (20-day)', fontsize=11)
        ax3.set_ylabel('52-Week Position', fontsize=11)
        ax3.set_title('Momentum vs 52-Week Position', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)

        # Add quadrant lines
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Add trend line
        if len(valid_data) > 10:
            z = np.polyfit(valid_data['daily_momentum_20'].dropna(),
                          valid_data['daily_price_position_52w'].dropna(), 1)
            p = np.poly1d(z)
            ax3.plot(valid_data['daily_momentum_20'], p(valid_data['daily_momentum_20']),
                    "r--", alpha=0.8, label='Trend')
            ax3.legend()

        # 4. Momentum Categories by Sector
        if 'sector' in valid_data.columns:
            # Create momentum categories
            valid_data = valid_data.copy()
            valid_data['momentum_category'] = pd.cut(valid_data['daily_momentum_20'],
                                                   bins=[-np.inf, -10, 10, np.inf],
                                                   labels=['Negative', 'Neutral', 'Positive'])

            # Calculate percentage in each category by sector
            momentum_by_sector = valid_data.groupby('sector')['momentum_category'].value_counts(normalize=True).unstack()
            momentum_by_sector = momentum_by_sector.fillna(0) * 100  # Convert to percentages

            momentum_by_sector.plot(kind='barh', stacked=True, ax=ax4,
                                  color=['red', 'gray', 'green'], alpha=0.8)

            ax4.set_xlabel('Percentage of Stocks', fontsize=11)
            ax4.set_ylabel('Sector', fontsize=11)
            ax4.set_title('Momentum Categories by Sector (%)', fontsize=12, fontweight='bold')
            ax4.legend(title='Momentum', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created momentum analysis chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating momentum analysis chart: {e}")
        return False

def create_moving_average_analysis(df: pd.DataFrame, output_path: str,
                                 title_suffix: str = "") -> bool:
    """
    Create moving average alignment and trend analysis.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check for moving average columns
        ma_cols = [col for col in df.columns if 'sma' in col.lower() and 'slope' in col.lower()]
        alignment_cols = [col for col in df.columns if 'alignment' in col.lower()]
        position_cols = [col for col in df.columns if 'vs' in col.lower() and ('sma' in col.lower() or 'ema' in col.lower())]

        if not ma_cols and not alignment_cols and not position_cols:
            logger.warning("No moving average data columns found")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. SMA Slope Analysis (if available)
        if ma_cols:
            # Use first available SMA slope column
            slope_col = ma_cols[0]
            valid_slope = df[slope_col].dropna()

            if not valid_slope.empty:
                ax1.hist(valid_slope, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                ax1.axvline(valid_slope.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {valid_slope.mean():.2f}')
                ax1.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero (Flat)')

                ax1.set_xlabel(f'{slope_col.replace("daily_", "").replace("_", " ").title()}', fontsize=11)
                ax1.set_ylabel('Frequency', fontsize=11)
                ax1.set_title(f'Moving Average Slope Distribution {title_suffix}', fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(alpha=0.3)

        # 2. Price vs MA Position Analysis
        if position_cols:
            # Use multiple position columns if available
            position_data = df[position_cols[:3]].dropna()  # Use first 3 columns

            if not position_data.empty:
                for i, col in enumerate(position_data.columns):
                    ax2.hist(position_data[col], bins=30, alpha=0.6, label=col.replace('daily_', '').replace('_', ' '),
                            histtype='step', linewidth=2)

                ax2.set_xlabel('Price vs Moving Average (%)', fontsize=11)
                ax2.set_ylabel('Frequency', fontsize=11)
                ax2.set_title('Price Position vs Moving Averages', fontsize=12, fontweight='bold')
                ax2.legend()
                ax2.grid(alpha=0.3)
                ax2.axvline(0, color='black', linestyle='-', linewidth=1)

        # 3. MA Alignment Analysis (if available)
        if alignment_cols:
            alignment_col = alignment_cols[0]
            alignment_data = df[alignment_col].dropna()

            if not alignment_data.empty:
                alignment_counts = alignment_data.value_counts()

                colors3 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(alignment_counts)))
                wedges, texts, autotexts = ax3.pie(alignment_counts.values, labels=alignment_counts.index,
                                                  autopct='%1.1f%%', colors=colors3, startangle=90)

                ax3.set_title(f'Moving Average Alignment\n(Total: {len(alignment_data):,})',
                             fontsize=12, fontweight='bold')

                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_weight('bold')

        # 4. Trend Strength Analysis
        # Combine multiple indicators for trend strength
        trend_indicators = []

        # Add slope indicators
        for col in ma_cols[:2]:  # Use first 2 slope columns
            if col in df.columns:
                trend_indicators.append(df[col])

        # Add position indicators
        for col in position_cols[:2]:  # Use first 2 position columns
            if col in df.columns:
                trend_indicators.append(df[col])

        if trend_indicators:
            trend_df = pd.concat(trend_indicators, axis=1)
            trend_df = trend_df.dropna()

            if not trend_df.empty:
                # Calculate composite trend score
                normalized_df = trend_df.copy()
                for col in trend_df.columns:
                    normalized_df[col] = (trend_df[col] - trend_df[col].mean()) / trend_df[col].std()

                trend_score = normalized_df.mean(axis=1)

                # Create trend categories
                trend_categories = pd.cut(trend_score,
                                        bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
                                        labels=['Strong Bear', 'Weak Bear', 'Neutral', 'Weak Bull', 'Strong Bull'])

                category_counts = trend_categories.value_counts()
                colors4 = ['darkred', 'red', 'gray', 'green', 'darkgreen']

                bars = ax4.bar(range(len(category_counts)), category_counts.values, color=colors4, alpha=0.8)

                ax4.set_xticks(range(len(category_counts)))
                ax4.set_xticklabels(category_counts.index, rotation=45)
                ax4.set_ylabel('Number of Stocks', fontsize=11)
                ax4.set_title('Trend Strength Distribution', fontsize=12, fontweight='bold')
                ax4.grid(axis='y', alpha=0.3)

                # Add value labels
                for i, value in enumerate(category_counts.values):
                    ax4.text(i, value + max(category_counts.values) * 0.01, str(value),
                            ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created moving average analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating moving average analysis: {e}")
        return False