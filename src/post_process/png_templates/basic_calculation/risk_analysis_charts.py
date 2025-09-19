#!/usr/bin/env python3
"""
Risk Analysis Charts for Basic Calculation Data
===============================================

Risk-focused analysis charts including:
- Risk-return scatter plots
- Volatility analysis
- Drawdown and recovery analysis
- VaR and risk metrics
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
sns.set_palette("coolwarm")

def create_risk_return_scatter(df: pd.DataFrame, output_path: str,
                              return_col: str = 'daily_daily_yearly_252d_pct_change',
                              risk_col: str = 'atr_pct',
                              title_suffix: str = "") -> bool:
    """
    Create comprehensive risk-return scatter plot analysis.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        return_col: Return column to analyze
        risk_col: Risk proxy column to analyze
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = [return_col, risk_col, 'sector', 'market_cap']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=[return_col, risk_col])

        if valid_data.empty:
            logger.warning("No valid risk-return data available")
            return False

        # Create market cap categories
        valid_data = valid_data.copy()
        valid_data['market_cap_category'] = pd.cut(valid_data['market_cap'],
                                                  bins=[0, 2e9, 10e9, 50e9, np.inf],
                                                  labels=['Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap'])

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Main Risk-Return Scatter (colored by sector)
        sectors = valid_data['sector'].dropna().unique()
        if len(sectors) == 0:
            logger.warning("No valid sectors found for risk-return scatter")
            return False

        colors = plt.cm.tab20(np.linspace(0, 1, len(sectors)))
        sector_colors = dict(zip(sectors, colors))

        for sector in sectors:
            if pd.isna(sector):
                continue
            sector_data = valid_data[valid_data['sector'] == sector]
            if not sector_data.empty:
                ax1.scatter(sector_data[risk_col], sector_data[return_col],
                           alpha=0.6, s=50, label=sector, color=sector_colors[sector])

        ax1.set_xlabel(f'Risk ({risk_col})', fontsize=11)
        ax1.set_ylabel(f'Return ({return_col.replace("daily_", "").replace("_", " ")})', fontsize=11)
        ax1.set_title(f'Risk-Return Profile by Sector {title_suffix}', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)

        # Add quadrant lines
        ax1.axhline(y=valid_data[return_col].median(), color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=valid_data[risk_col].median(), color='gray', linestyle='--', alpha=0.5)

        # Add legend with smaller font
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 2. Risk-Return by Market Cap
        cap_categories = valid_data['market_cap_category'].dropna().unique()
        cap_colors = plt.cm.viridis(np.linspace(0, 1, len(cap_categories)))

        for i, category in enumerate(cap_categories):
            cat_data = valid_data[valid_data['market_cap_category'] == category]
            ax2.scatter(cat_data[risk_col], cat_data[return_col],
                       alpha=0.7, s=60, label=category, color=cap_colors[i])

        ax2.set_xlabel(f'Risk ({risk_col})', fontsize=11)
        ax2.set_ylabel(f'Return ({return_col.replace("daily_", "").replace("_", " ")})', fontsize=11)
        ax2.set_title('Risk-Return Profile by Market Cap', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()

        # 3. Risk Distribution by Sector
        limited_sectors = sectors[:8]  # Limit to top 8 sectors
        sector_risk_data = []
        valid_sector_labels = []

        for sector in limited_sectors:
            sector_data = valid_data[valid_data['sector'] == sector][risk_col].dropna()
            if len(sector_data) > 0:
                sector_risk_data.append(sector_data)
                valid_sector_labels.append(str(sector)[:15])

        if len(sector_risk_data) > 0:
            box_plot = ax3.boxplot(sector_risk_data, labels=valid_sector_labels,
                                  patch_artist=True)

            # Color boxes - ensure we have enough colors
            box_colors = colors[:len(sector_risk_data)]
            for patch, color in zip(box_plot['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            ax3.text(0.5, 0.5, 'No valid sector risk data',
                    transform=ax3.transAxes, ha='center', va='center')

        ax3.set_xlabel('Sector', fontsize=11)
        ax3.set_ylabel(f'Risk ({risk_col})', fontsize=11)
        ax3.set_title('Risk Distribution by Sector', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(alpha=0.3)

        # 4. Efficient Frontier Approximation
        # Create risk-return bins for portfolio-like analysis
        try:
            risk_bins = pd.cut(valid_data[risk_col], bins=10)
            frontier_data = valid_data.groupby(risk_bins, observed=False).agg({
                return_col: ['mean', 'max', 'min'],
                risk_col: 'mean'
            }).reset_index()

            # Flatten column names safely
            new_columns = []
            for col in frontier_data.columns:
                if isinstance(col, tuple) and len(col) > 1 and col[1]:
                    new_columns.append(f'{col[0]}_{col[1]}')
                else:
                    new_columns.append(str(col[0]) if isinstance(col, tuple) else str(col))
            frontier_data.columns = new_columns

            # Find the correct column names
            risk_mean_col = None
            return_max_col = None
            return_mean_col = None
            return_min_col = None

            for col in frontier_data.columns:
                if risk_col in str(col) and 'mean' in str(col):
                    risk_mean_col = col
                elif return_col in str(col) and 'max' in str(col):
                    return_max_col = col
                elif return_col in str(col) and 'mean' in str(col):
                    return_mean_col = col
                elif return_col in str(col) and 'min' in str(col):
                    return_min_col = col

            if all([risk_mean_col, return_max_col, return_mean_col, return_min_col]):
                ax4.plot(frontier_data[risk_mean_col], frontier_data[return_max_col],
                        'go-', alpha=0.8, label='Max Return per Risk Level', markersize=6)
                ax4.plot(frontier_data[risk_mean_col], frontier_data[return_mean_col],
                        'bo-', alpha=0.8, label='Mean Return per Risk Level', markersize=6)
                ax4.fill_between(frontier_data[risk_mean_col],
                                frontier_data[return_min_col],
                                frontier_data[return_max_col],
                                alpha=0.2, color='blue')
            else:
                ax4.text(0.5, 0.5, 'Unable to create frontier plot',
                        transform=ax4.transAxes, ha='center', va='center')
        except Exception as e:
            logger.warning(f"Error creating efficient frontier: {e}")
            ax4.text(0.5, 0.5, 'Frontier plot unavailable',
                    transform=ax4.transAxes, ha='center', va='center')

        ax4.set_xlabel(f'Risk Level ({risk_col})', fontsize=11)
        ax4.set_ylabel(f'Return ({return_col.replace("daily_", "").replace("_", " ")})', fontsize=11)
        ax4.set_title('Efficient Frontier Approximation', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created risk-return scatter: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating risk-return scatter: {e}")
        return False

def create_volatility_analysis(df: pd.DataFrame, output_path: str,
                              title_suffix: str = "") -> bool:
    """
    Create comprehensive volatility analysis charts.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['atr_pct', 'daily_rsi_14', 'sector']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid volatility data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. ATR Distribution
        ax1.hist(valid_data['atr_pct'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(valid_data['atr_pct'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_data["atr_pct"].mean():.2f}%')
        ax1.axvline(valid_data['atr_pct'].median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {valid_data["atr_pct"].median():.2f}%')

        ax1.set_xlabel('Average True Range (%)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'ATR Distribution {title_suffix}', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Add percentile lines
        for percentile in [90, 95, 99]:
            pct_value = np.percentile(valid_data['atr_pct'], percentile)
            ax1.axvline(pct_value, color='orange', linestyle=':', alpha=0.7,
                       label=f'{percentile}th: {pct_value:.2f}%')

        # 2. Volatility by Sector
        sector_volatility = valid_data.groupby('sector')['atr_pct'].agg(['mean', 'median', 'std']).sort_values('mean', ascending=True)

        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sector_volatility)))
        bars = ax2.barh(range(len(sector_volatility)), sector_volatility['mean'], color=colors, alpha=0.8)

        ax2.set_yticks(range(len(sector_volatility)))
        ax2.set_yticklabels(sector_volatility.index, fontsize=10)
        ax2.set_xlabel('Mean ATR (%)', fontsize=11)
        ax2.set_title('Volatility by Sector', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, value in enumerate(sector_volatility['mean']):
            ax2.text(value + 0.1, i, f'{value:.1f}%', va='center', fontsize=9)

        # 3. Volatility vs RSI
        ax3.scatter(valid_data['daily_rsi_14'], valid_data['atr_pct'], alpha=0.6, s=30)
        ax3.set_xlabel('RSI (14-day)', fontsize=11)
        ax3.set_ylabel('ATR (%)', fontsize=11)
        ax3.set_title('Volatility vs RSI Relationship', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)

        # Add trend line
        z = np.polyfit(valid_data['daily_rsi_14'].dropna(), valid_data['atr_pct'].dropna(), 1)
        p = np.poly1d(z)
        ax3.plot(valid_data['daily_rsi_14'], p(valid_data['daily_rsi_14']), "r--", alpha=0.8)

        # Add RSI zones
        ax3.axvline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax3.axvline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax3.legend()

        # 4. High Volatility Stocks Analysis
        high_vol_threshold = np.percentile(valid_data['atr_pct'], 90)
        high_vol_stocks = valid_data[valid_data['atr_pct'] > high_vol_threshold]

        if not high_vol_stocks.empty:
            high_vol_sectors = high_vol_stocks['sector'].value_counts()

            colors4 = plt.cm.Set3(np.linspace(0, 1, len(high_vol_sectors)))
            wedges, texts, autotexts = ax4.pie(high_vol_sectors.values, labels=high_vol_sectors.index,
                                              autopct='%1.1f%%', colors=colors4, startangle=90)

            ax4.set_title(f'High Volatility Stocks by Sector\n(ATR > {high_vol_threshold:.1f}%, n={len(high_vol_stocks)})',
                         fontsize=12, fontweight='bold')

            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created volatility analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating volatility analysis: {e}")
        return False

def create_drawdown_analysis(df: pd.DataFrame, output_path: str,
                           title_suffix: str = "") -> bool:
    """
    Create drawdown and recovery analysis charts.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        required_cols = ['daily_distance_from_ATH_pct', 'daily_distance_from_ATL_pct',
                        'daily_ATH_ATL_position_pct', 'sector']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        # Filter valid data
        valid_data = df.dropna(subset=required_cols)

        if valid_data.empty:
            logger.warning("No valid drawdown data available")
            return False

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Distance from ATH Distribution
        ax1.hist(valid_data['daily_distance_from_ATH_pct'], bins=50, alpha=0.7,
                color='lightcoral', edgecolor='black')
        ax1.axvline(valid_data['daily_distance_from_ATH_pct'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_data["daily_distance_from_ATH_pct"].mean():.1f}%')
        ax1.axvline(valid_data['daily_distance_from_ATH_pct'].median(), color='darkred', linestyle='--', linewidth=2,
                   label=f'Median: {valid_data["daily_distance_from_ATH_pct"].median():.1f}%')

        ax1.set_xlabel('Distance from All-Time High (%)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Distance from ATH Distribution {title_suffix}', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Distance from ATL Distribution
        ax2.hist(valid_data['daily_distance_from_ATL_pct'], bins=50, alpha=0.7,
                color='lightgreen', edgecolor='black')
        ax2.axvline(valid_data['daily_distance_from_ATL_pct'].mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_data["daily_distance_from_ATL_pct"].mean():.1f}%')
        ax2.axvline(valid_data['daily_distance_from_ATL_pct'].median(), color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Median: {valid_data["daily_distance_from_ATL_pct"].median():.1f}%')

        ax2.set_xlabel('Distance from All-Time Low (%)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distance from ATL Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. 52-Week Position by Sector
        sector_position = valid_data.groupby('sector')['daily_ATH_ATL_position_pct'].mean().sort_values(ascending=True)

        colors3 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sector_position)))
        bars = ax3.barh(range(len(sector_position)), sector_position.values, color=colors3)

        ax3.set_yticks(range(len(sector_position)))
        ax3.set_yticklabels(sector_position.index, fontsize=10)
        ax3.set_xlabel('Average 52-Week Position (0=Low, 1=High)', fontsize=11)
        ax3.set_title('Sector 52-Week Position Analysis', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # Add position zone lines
        ax3.axvline(0.2, color='red', linestyle='--', alpha=0.7, label='Near Low')
        ax3.axvline(0.8, color='green', linestyle='--', alpha=0.7, label='Near High')
        ax3.legend()

        # Add value labels
        for i, value in enumerate(sector_position.values):
            ax3.text(value + 0.01, i, f'{value:.2f}', va='center', fontsize=9)

        # 4. ATH vs ATL Distance Scatter
        ax4.scatter(valid_data['daily_distance_from_ATH_pct'], valid_data['daily_distance_from_ATL_pct'],
                   alpha=0.6, s=30, c=valid_data['daily_ATH_ATL_position_pct'], cmap='RdYlGn')

        ax4.set_xlabel('Distance from ATH (%)', fontsize=11)
        ax4.set_ylabel('Distance from ATL (%)', fontsize=11)
        ax4.set_title('ATH vs ATL Distance (color = 52W position)', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('52-Week Position', fontsize=10)

        # Identify interesting quadrants
        # Stocks near ATH and far from ATL (strong performers)
        near_ath = valid_data['daily_distance_from_ATH_pct'] > -10
        far_atl = valid_data['daily_distance_from_ATL_pct'] > 50
        strong_performers = valid_data[near_ath & far_atl]

        if not strong_performers.empty:
            ax4.scatter(strong_performers['daily_distance_from_ATH_pct'],
                       strong_performers['daily_distance_from_ATL_pct'],
                       s=100, facecolors='none', edgecolors='blue', linewidth=2,
                       label=f'Strong Performers (n={len(strong_performers)})')
            ax4.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created drawdown analysis: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating drawdown analysis: {e}")
        return False

def create_risk_metrics_dashboard(df: pd.DataFrame, output_path: str,
                                title_suffix: str = "") -> bool:
    """
    Create comprehensive risk metrics dashboard.

    Args:
        df: DataFrame with basic_calculation data
        output_path: Path to save PNG file
        title_suffix: Additional text for chart title

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Calculate additional risk metrics
        risk_metrics = df.copy()

        # Sharpe ratio approximation (return / volatility)
        if 'daily_daily_yearly_252d_pct_change' in df.columns and 'atr_pct' in df.columns:
            risk_metrics['sharpe_approx'] = risk_metrics['daily_daily_yearly_252d_pct_change'] / risk_metrics['atr_pct']

        # Risk score (combination of multiple factors)
        risk_factors = ['atr_pct', 'daily_distance_from_ATH_pct']
        available_factors = [col for col in risk_factors if col in df.columns]

        if available_factors:
            # Normalize risk factors and create composite score
            for factor in available_factors:
                risk_metrics[f'{factor}_norm'] = (risk_metrics[factor] - risk_metrics[factor].mean()) / risk_metrics[factor].std()

            risk_metrics['risk_score'] = risk_metrics[[f'{factor}_norm' for factor in available_factors]].mean(axis=1)

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Risk Score Distribution
        if 'risk_score' in risk_metrics.columns:
            valid_risk = risk_metrics['risk_score'].dropna()

            ax1.hist(valid_risk, bins=40, alpha=0.7, color='orange', edgecolor='black')
            ax1.axvline(valid_risk.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {valid_risk.mean():.2f}')
            ax1.axvline(valid_risk.median(), color='blue', linestyle='--', linewidth=2,
                       label=f'Median: {valid_risk.median():.2f}')

            ax1.set_xlabel('Composite Risk Score', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.set_title(f'Risk Score Distribution {title_suffix}', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)

        # 2. Sharpe Ratio Analysis
        if 'sharpe_approx' in risk_metrics.columns:
            valid_sharpe = risk_metrics['sharpe_approx'].dropna()
            # Remove extreme outliers for better visualization
            valid_sharpe = valid_sharpe[(valid_sharpe > np.percentile(valid_sharpe, 1)) &
                                       (valid_sharpe < np.percentile(valid_sharpe, 99))]

            ax2.hist(valid_sharpe, bins=40, alpha=0.7, color='purple', edgecolor='black')
            ax2.axvline(valid_sharpe.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {valid_sharpe.mean():.2f}')
            ax2.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero')

            ax2.set_xlabel('Sharpe Ratio Approximation', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Sharpe Ratio Distribution', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)

        # 3. Risk-Adjusted Returns by Sector
        if 'sharpe_approx' in risk_metrics.columns and 'sector' in risk_metrics.columns:
            sector_sharpe = risk_metrics.groupby('sector')['sharpe_approx'].mean().sort_values(ascending=True)
            # Remove extreme values for visualization
            sector_sharpe = sector_sharpe[(sector_sharpe > -10) & (sector_sharpe < 10)]

            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sector_sharpe)))
            bars = ax3.barh(range(len(sector_sharpe)), sector_sharpe.values, color=colors)

            ax3.set_yticks(range(len(sector_sharpe)))
            ax3.set_yticklabels(sector_sharpe.index, fontsize=10)
            ax3.set_xlabel('Average Sharpe Ratio', fontsize=11)
            ax3.set_title('Risk-Adjusted Returns by Sector', fontsize=12, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            ax3.axvline(0, color='black', linestyle='-', linewidth=1)

        # 4. Risk Categories Analysis
        if 'atr_pct' in risk_metrics.columns:
            # Create risk categories
            risk_metrics['risk_category'] = pd.cut(risk_metrics['atr_pct'],
                                                  bins=[0, 2, 4, 6, np.inf],
                                                  labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])

            risk_category_counts = risk_metrics['risk_category'].value_counts()

            colors4 = ['green', 'yellow', 'orange', 'red']
            wedges, texts, autotexts = ax4.pie(risk_category_counts.values,
                                              labels=risk_category_counts.index,
                                              autopct='%1.1f%%', colors=colors4, startangle=90)

            ax4.set_title('Portfolio Risk Distribution\n(Based on ATR %)', fontsize=12, fontweight='bold')

            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created risk metrics dashboard: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating risk metrics dashboard: {e}")
        return False