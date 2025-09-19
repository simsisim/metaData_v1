#!/usr/bin/env python3
"""
Overview V1 PDF Template
=======================

Comprehensive daily market analysis report implementing the 4-step workflow:
1. Performance Analysis - Multi-timeframe metrics calculation
2. Sector/Industry Analysis - Group analysis and rotation detection
3. Visualization - Automated chart generation
4. PDF Assembly - Professional report layout

Features:
- Executive summary with key findings
- Multi-timeframe performance analysis (1D, 7D, 22D, 66D, 252D)
- Sector rotation detection (STRONG_IN, ROTATING_OUT, NEUTRAL)
- Comprehensive chart suite (bar, heatmap, scatter, tornado, radar)
- Investment recommendations per timeframe
- Professional PDF layout with structured sections
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

logger = logging.getLogger(__name__)

def generate_pdf(df: pd.DataFrame, pdf_path: str, metadata: dict = None) -> bool:
    """
    Generate comprehensive market overview PDF report following 4-step workflow.

    Args:
        df: DataFrame with basic_calculation data
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize the market analysis engine
        analyzer = MarketOverviewAnalyzer(df, metadata)

        # Step 1: Performance Analysis - Multi-timeframe metrics calculation
        performance_analysis = analyzer.calculate_performance_metrics()

        # Step 2: Sector/Industry Analysis - Group analysis and rotation detection
        sector_analysis = analyzer.analyze_sectors_industries()

        # Step 3: Visualization - Automated chart generation
        chart_paths = analyzer.generate_visualizations()

        # Step 4: PDF Assembly - Professional report layout
        success = analyzer.assemble_pdf(pdf_path, performance_analysis, sector_analysis, chart_paths)

        if success:
            logger.info(f"Successfully generated Overview V1 PDF: {pdf_path}")
            return True
        else:
            logger.error("Failed to generate Overview V1 PDF")
            return False

    except Exception as e:
        logger.error(f"Error generating Overview V1 PDF: {e}")
        return False

class MarketOverviewAnalyzer:
    """Main analyzer class implementing the 4-step workflow."""

    def __init__(self, df: pd.DataFrame, metadata: dict = None):
        self.df = df
        self.metadata = metadata or {}
        self.timeframes = {
            '1D': 'daily_daily_daily_1d_pct_change',
            '7D': 'daily_daily_weekly_7d_pct_change',
            '22D': 'daily_daily_monthly_22d_pct_change',
            '66D': 'daily_daily_quarterly_66d_pct_change',
            '252D': 'daily_daily_yearly_252d_pct_change'
        }

        # Filter timeframes to available columns
        self.available_timeframes = {
            label: col for label, col in self.timeframes.items()
            if col in df.columns
        }

        # Use permanent PNG directory
        self.png_dir = Path("results/post_process")
        self.png_dir.mkdir(parents=True, exist_ok=True)

    def calculate_performance_metrics(self) -> dict:
        """Step 1: Calculate multi-timeframe performance metrics."""
        logger.info("Step 1: Calculating performance metrics")

        analysis = {}

        # Calculate momentum score (document-specified weighting)
        if len(self.available_timeframes) >= 2:
            timeframe_cols = list(self.available_timeframes.values())

            # Create weights based on number of timeframes (shorter term gets less weight)
            if len(timeframe_cols) == 5:
                weights = [0.05, 0.15, 0.25, 0.25, 0.3]  # 1D, 7D, 22D, 66D, 252D
            elif len(timeframe_cols) == 4:
                weights = [0.1, 0.2, 0.3, 0.4]
            elif len(timeframe_cols) == 3:
                weights = [0.2, 0.3, 0.5]
            else:  # 2 timeframes
                weights = [0.3, 0.7]

            # Ensure weights match number of columns
            weights = weights[:len(timeframe_cols)]
            weights = np.array(weights) / sum(weights)  # Normalize

            momentum_data = self.df[timeframe_cols].fillna(0)
            self.df['momentum_score'] = (momentum_data * weights).sum(axis=1)

            analysis['momentum_calculated'] = True
        else:
            self.df['momentum_score'] = 0
            analysis['momentum_calculated'] = False

        # Calculate trend consistency
        timeframe_cols = list(self.available_timeframes.values())
        positive_counts = (self.df[timeframe_cols] > 0).sum(axis=1)
        self.df['trend_consistency'] = positive_counts / len(timeframe_cols) * 100

        # Calculate summary statistics
        analysis['summary_stats'] = {}
        for label, col in self.available_timeframes.items():
            if col in self.df.columns:
                data = self.df[col].dropna()
                analysis['summary_stats'][label] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'positive_count': (data > 0).sum(),
                    'total_count': len(data)
                }

        # Identify top performers
        analysis['top_performers'] = {}
        for label, col in self.available_timeframes.items():
            if col in self.df.columns:
                top_10 = self.df.nlargest(10, col)[['ticker', col]].copy()
                analysis['top_performers'][label] = top_10

        # Identify momentum leaders
        if 'momentum_score' in self.df.columns:
            analysis['momentum_leaders'] = self.df.nlargest(15, 'momentum_score')[['ticker', 'momentum_score', 'trend_consistency']].copy()

        return analysis

    def analyze_sectors_industries(self) -> dict:
        """Step 2: Analyze sectors and industries with rotation detection."""
        logger.info("Step 2: Analyzing sectors and industries")

        analysis = {}

        # Sector analysis
        if 'sector' in self.df.columns:
            sector_perf = self.df.groupby('sector')[list(self.available_timeframes.values())].mean()

            # Apply rotation detection logic from document
            analysis['sector_rotation'] = self._detect_rotation_signals(sector_perf)
            analysis['sector_performance'] = sector_perf
            analysis['sector_rankings'] = self._rank_groups(sector_perf)

        # Enhanced Industry analysis
        if 'industry' in self.df.columns:
            analysis.update(self._analyze_industries_enhanced())

        return analysis

    def _detect_rotation_signals(self, group_perf: pd.DataFrame) -> pd.DataFrame:
        """Detect sector rotation signals based on document logic."""
        rotation_signals = group_perf.copy()
        rotation_signals['rotation_signal'] = 'NEUTRAL'

        # Get short-term columns (1D and 7D if available)
        short_term_cols = []
        for label in ['1D', '7D']:
            if label in self.available_timeframes:
                col = self.available_timeframes[label]
                if col in group_perf.columns:
                    short_term_cols.append(col)

        if len(short_term_cols) >= 2:
            col1, col2 = short_term_cols[0], short_term_cols[1]

            # STRONG_IN: positive 1D AND 7D
            strong_in_mask = (group_perf[col1] > 0) & (group_perf[col2] > 0)
            rotation_signals.loc[strong_in_mask, 'rotation_signal'] = 'STRONG_IN'

            # ROTATING_OUT: negative 1D AND 7D
            rotating_out_mask = (group_perf[col1] < 0) & (group_perf[col2] < 0)
            rotation_signals.loc[rotating_out_mask, 'rotation_signal'] = 'ROTATING_OUT'

        return rotation_signals

    def _rank_groups(self, group_perf: pd.DataFrame) -> dict:
        """Rank groups by different timeframes."""
        rankings = {}

        for label, col in self.available_timeframes.items():
            if col in group_perf.columns:
                ranked = group_perf.sort_values(col, ascending=False)
                rankings[label] = ranked[[col]].copy()

        return rankings

    def _analyze_industries_enhanced(self) -> dict:
        """Enhanced industry analysis with market cap weighting and leader/laggard identification."""
        logger.info("Performing enhanced industry analysis")

        analysis = {}

        # Check if we have market_cap column for weighting
        has_market_cap = 'market_cap' in self.df.columns

        # 1. Market cap weighted industry performance
        if has_market_cap:
            analysis['industry_performance'] = self._calculate_weighted_industry_performance()
        else:
            # Fallback to simple mean if no market cap data
            analysis['industry_performance'] = self.df.groupby('industry')[list(self.available_timeframes.values())].mean()

        # 2. Industry momentum scoring and classification
        analysis['industry_momentum'] = self._calculate_industry_momentum(analysis['industry_performance'])
        analysis['industry_classification'] = self._classify_industry_performance(analysis['industry_performance'])

        # 3. Industry rotation signals
        analysis['industry_rotation'] = self._detect_rotation_signals(analysis['industry_performance'])

        # 4. Intra-industry leader/laggard analysis
        analysis['industry_leaders_laggards'] = self._identify_industry_leaders_laggards()

        # 5. Performance attribution and risk metrics
        analysis['industry_risk_metrics'] = self._calculate_industry_risk_metrics()

        # 6. Traditional rankings for backward compatibility
        analysis['industry_rankings'] = self._rank_groups(analysis['industry_performance'])

        return analysis

    def _calculate_weighted_industry_performance(self) -> pd.DataFrame:
        """Calculate market cap weighted industry performance."""
        timeframe_cols = list(self.available_timeframes.values())

        def weighted_mean(group):
            weights = group['market_cap'].fillna(0)
            if weights.sum() == 0:
                return group[timeframe_cols].mean()
            return (group[timeframe_cols].multiply(weights, axis=0)).sum() / weights.sum()

        return self.df.groupby('industry').apply(weighted_mean)

    def _calculate_industry_momentum(self, industry_perf: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced momentum scores for industries."""
        momentum_df = industry_perf.copy()

        # Multi-timeframe momentum scoring
        timeframe_cols = list(self.available_timeframes.values())
        if len(timeframe_cols) >= 2:
            # Momentum weights: shorter term gets less weight
            if len(timeframe_cols) == 5:
                weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # 1D, 7D, 22D, 66D, 252D
            elif len(timeframe_cols) == 4:
                weights = [0.15, 0.25, 0.3, 0.3]
            elif len(timeframe_cols) == 3:
                weights = [0.2, 0.3, 0.5]
            else:  # 2 timeframes
                weights = [0.3, 0.7]

            weights = weights[:len(timeframe_cols)]
            weights = np.array(weights) / sum(weights)

            momentum_data = industry_perf[timeframe_cols].fillna(0)
            momentum_df['momentum_score'] = (momentum_data * weights).sum(axis=1)
        else:
            momentum_df['momentum_score'] = 0

        return momentum_df

    def _classify_industry_performance(self, industry_perf: pd.DataFrame) -> pd.DataFrame:
        """Classify industries into LEADERS, EMERGING, DECLINING, LAGGARDS."""
        classification_df = industry_perf.copy()

        # Use longest timeframe available for classification
        long_term_col = None
        for label in ['252D', '66D', '22D', '7D', '1D']:
            if label in self.available_timeframes:
                long_term_col = self.available_timeframes[label]
                break

        if long_term_col and long_term_col in industry_perf.columns:
            performance = industry_perf[long_term_col]

            # Calculate percentiles
            p75 = performance.quantile(0.75)
            p50 = performance.quantile(0.50)
            p25 = performance.quantile(0.25)

            # Classify based on percentiles
            classification_df['classification'] = 'NEUTRAL'
            classification_df.loc[performance >= p75, 'classification'] = 'LEADERS'
            classification_df.loc[(performance >= p50) & (performance < p75), 'classification'] = 'EMERGING'
            classification_df.loc[(performance >= p25) & (performance < p50), 'classification'] = 'DECLINING'
            classification_df.loc[performance < p25, 'classification'] = 'LAGGARDS'
        else:
            classification_df['classification'] = 'NEUTRAL'

        return classification_df

    def _identify_industry_leaders_laggards(self) -> dict:
        """Identify top 3 performers and bottom 2 laggards within each industry."""
        leaders_laggards = {}

        # Use longest timeframe for leader/laggard identification
        long_term_col = None
        for label in ['252D', '66D', '22D', '7D', '1D']:
            if label in self.available_timeframes:
                long_term_col = self.available_timeframes[label]
                break

        if not long_term_col or long_term_col not in self.df.columns:
            return leaders_laggards

        for industry, industry_data in self.df.groupby('industry'):
            if len(industry_data) < 3:  # Skip industries with too few stocks
                continue

            # Sort by performance
            sorted_data = industry_data.sort_values(long_term_col, ascending=False)

            # Top 3 performers
            leaders = sorted_data.head(3)[['ticker', long_term_col, 'market_cap']].copy() if 'market_cap' in self.df.columns else sorted_data.head(3)[['ticker', long_term_col]].copy()

            # Bottom 2 laggards
            laggards = sorted_data.tail(2)[['ticker', long_term_col, 'market_cap']].copy() if 'market_cap' in self.df.columns else sorted_data.tail(2)[['ticker', long_term_col]].copy()

            # Calculate performance gap
            if len(leaders) > 0 and len(laggards) > 0:
                performance_gap = leaders[long_term_col].iloc[0] - laggards[long_term_col].iloc[-1]
            else:
                performance_gap = 0

            leaders_laggards[industry] = {
                'leaders': leaders,
                'laggards': laggards,
                'performance_gap': performance_gap,
                'stock_count': len(industry_data)
            }

        return leaders_laggards

    def _calculate_industry_risk_metrics(self) -> dict:
        """Calculate risk and dispersion metrics for industries."""
        risk_metrics = {}

        # Use longest timeframe for risk calculation
        long_term_col = None
        for label in ['252D', '66D', '22D', '7D', '1D']:
            if label in self.available_timeframes:
                long_term_col = self.available_timeframes[label]
                break

        if not long_term_col or long_term_col not in self.df.columns:
            return risk_metrics

        for industry, industry_data in self.df.groupby('industry'):
            if len(industry_data) < 3:
                continue

            performance = industry_data[long_term_col].dropna()
            if len(performance) == 0:
                continue

            risk_metrics[industry] = {
                'performance_std': performance.std(),
                'performance_range': performance.max() - performance.min(),
                'performance_iqr': performance.quantile(0.75) - performance.quantile(0.25),
                'concentration_ratio': self._calculate_gini_coefficient(performance),
                'stock_count': len(performance)
            }

        return risk_metrics

    def _calculate_gini_coefficient(self, values: pd.Series) -> float:
        """Calculate Gini coefficient for concentration analysis."""
        try:
            # Convert to numpy array and sort
            sorted_values = np.sort(values.values)
            n = len(sorted_values)

            if n == 0:
                return 0.0

            # Calculate Gini coefficient
            cumsum = np.cumsum(sorted_values)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] != 0 else 0.0

            return max(0.0, min(1.0, gini))  # Ensure between 0 and 1
        except:
            return 0.0

    def generate_visualizations(self) -> dict:
        """Step 3: Generate comprehensive chart suite."""
        logger.info("Step 3: Generating visualizations")

        chart_paths = {}

        # A. Bar Charts - Top performers
        chart_paths['performance_bars'] = self._create_performance_bar_charts()

        # B. Heatmaps - Sector performance across timeframes
        chart_paths['sector_heatmap'] = self._create_sector_heatmap()

        # C. Scatter Plots - Momentum vs long-term performance
        chart_paths['momentum_scatter'] = self._create_momentum_scatter()

        # D. Tornado Charts - Sector dispersion
        chart_paths['tornado_chart'] = self._create_tornado_chart()

        # E. Performance Distribution
        chart_paths['distribution_chart'] = self._create_distribution_chart()

        # F. Enhanced Industry Analysis Charts
        chart_paths['industry_performance_matrix'] = self._create_industry_performance_matrix()
        chart_paths['industry_momentum_bubble'] = self._create_industry_momentum_bubble()
        chart_paths['industry_dispersion_box'] = self._create_industry_dispersion_box()
        chart_paths['leader_laggard_comparison'] = self._create_leader_laggard_comparison()
        chart_paths['industry_rotation_flow'] = self._create_industry_rotation_flow()

        return chart_paths

    def _create_performance_bar_charts(self) -> str:
        """Create bar charts for top performers."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            # Create bar chart for each available timeframe (up to 4)
            for i, (label, col) in enumerate(list(self.available_timeframes.items())[:4]):
                if i >= 4:
                    break

                ax = axes[i]
                top_10 = self.df.nlargest(10, col)

                colors_list = plt.cm.RdYlGn(np.linspace(0.3, 1, len(top_10)))
                bars = ax.barh(range(len(top_10)), top_10[col], color=colors_list)

                ax.set_yticks(range(len(top_10)))
                ax.set_yticklabels(top_10['ticker'])
                ax.set_xlabel('Performance (%)')
                ax.set_title(f'Top 10 Performers - {label}', fontweight='bold')
                ax.grid(axis='x', alpha=0.3)

                # Add value labels
                for j, (bar, value) in enumerate(zip(bars, top_10[col])):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                           f'{value:.1f}%', va='center', fontsize=8)

            # Hide unused subplots
            for i in range(len(self.available_timeframes), 4):
                axes[i].set_visible(False)

            plt.tight_layout()
            chart_path = self.png_dir / "performance_bar_charts.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating performance bar charts: {e}")
            return ""

    def _create_sector_heatmap(self) -> str:
        """Create sector performance heatmap across timeframes."""
        try:
            if 'sector' not in self.df.columns:
                return ""

            # Get sector performance data
            sector_perf = self.df.groupby('sector')[list(self.available_timeframes.values())].mean()

            if sector_perf.empty:
                return ""

            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))

            # Use timeframe labels for columns
            display_data = sector_perf.copy()
            display_data.columns = list(self.available_timeframes.keys())

            sns.heatmap(display_data, annot=True, fmt='.1f', cmap='RdYlGn',
                       center=0, cbar_kws={'label': 'Performance (%)'}, ax=ax)

            ax.set_title('Sector Performance Heatmap Across Timeframes', fontsize=14, fontweight='bold')
            ax.set_xlabel('Timeframe')
            ax.set_ylabel('Sector')

            plt.tight_layout()
            chart_path = self.png_dir / "sector_heatmap.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating sector heatmap: {e}")
            return ""

    def _create_momentum_scatter(self) -> str:
        """Create scatter plot of momentum vs long-term performance."""
        try:
            if 'momentum_score' not in self.df.columns:
                return ""

            # Get long-term performance column (252D or longest available)
            long_term_col = None
            for label in ['252D', '66D', '22D']:
                if label in self.available_timeframes:
                    long_term_col = self.available_timeframes[label]
                    break

            if not long_term_col:
                return ""

            fig, ax = plt.subplots(figsize=(12, 8))

            # Create scatter plot colored by sector if available
            if 'sector' in self.df.columns:
                sectors = self.df['sector'].unique()
                colors_list = plt.cm.tab20(np.linspace(0, 1, len(sectors)))
                sector_colors = dict(zip(sectors, colors_list))

                for sector in sectors:
                    sector_data = self.df[self.df['sector'] == sector]
                    ax.scatter(sector_data['momentum_score'], sector_data[long_term_col],
                             label=sector, alpha=0.7, color=sector_colors[sector])

                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.scatter(self.df['momentum_score'], self.df[long_term_col], alpha=0.7)

            ax.set_xlabel('Momentum Score')
            ax.set_ylabel(f'Long-term Performance (%) - {list(self.available_timeframes.keys())[-1]}')
            ax.set_title('Momentum vs Long-term Performance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add quadrant lines
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            plt.tight_layout()
            chart_path = self.png_dir / "momentum_scatter.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating momentum scatter: {e}")
            return ""

    def _create_tornado_chart(self) -> str:
        """Create tornado chart showing sector dispersion."""
        try:
            if 'sector' not in self.df.columns:
                return ""

            # Use longest timeframe available for dispersion analysis
            long_term_col = None
            for label in ['252D', '66D', '22D', '7D', '1D']:
                if label in self.available_timeframes:
                    long_term_col = self.available_timeframes[label]
                    break

            if not long_term_col:
                return ""

            # Calculate sector dispersion (min/max range)
            sector_stats = self.df.groupby('sector')[long_term_col].agg(['min', 'max', 'mean']).reset_index()
            sector_stats['range'] = sector_stats['max'] - sector_stats['min']
            sector_stats = sector_stats.sort_values('mean', ascending=True)

            fig, ax = plt.subplots(figsize=(12, 8))

            # Create tornado chart
            y_pos = range(len(sector_stats))

            # Plot ranges
            for i, (_, row) in enumerate(sector_stats.iterrows()):
                width = row['range']
                left = row['min']
                color = 'green' if row['mean'] > 0 else 'red'

                ax.barh(i, width, left=left, alpha=0.7, color=color)

                # Add mean marker
                ax.plot(row['mean'], i, 'o', color='black', markersize=6)

            ax.set_yticks(y_pos)
            ax.set_yticklabels([s[:20] + '...' if len(s) > 20 else s for s in sector_stats['sector']])
            ax.set_xlabel('Performance Range (%)')
            ax.set_title('Sector Performance Dispersion (Tornado Chart)', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)

            plt.tight_layout()
            chart_path = self.png_dir / "tornado_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating tornado chart: {e}")
            return ""

    def _create_distribution_chart(self) -> str:
        """Create performance distribution chart."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            # Create distribution for each timeframe (up to 4)
            for i, (label, col) in enumerate(list(self.available_timeframes.items())[:4]):
                if i >= 4:
                    break

                ax = axes[i]
                data = self.df[col].dropna()

                # Histogram
                ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}%')
                ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.1f}%')

                ax.set_xlabel('Performance (%)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Performance Distribution - {label}', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for i in range(len(self.available_timeframes), 4):
                axes[i].set_visible(False)

            plt.tight_layout()
            chart_path = self.png_dir / "distribution_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating distribution chart: {e}")
            return ""

    def _create_industry_performance_matrix(self) -> str:
        """Create industry performance matrix heatmap across timeframes."""
        try:
            if 'industry' not in self.df.columns:
                return ""

            # Get industry performance data (market cap weighted if available)
            has_market_cap = 'market_cap' in self.df.columns
            if has_market_cap:
                industry_perf = self._calculate_weighted_industry_performance()
            else:
                industry_perf = self.df.groupby('industry')[list(self.available_timeframes.values())].mean()

            if industry_perf.empty:
                return ""

            # Limit to top 15 industries for readability
            long_term_col = list(self.available_timeframes.values())[-1]
            top_industries = industry_perf.nlargest(15, long_term_col)

            # Create heatmap
            fig, ax = plt.subplots(figsize=(14, 10))

            # Use timeframe labels for columns
            display_data = top_industries.copy()
            display_data.columns = list(self.available_timeframes.keys())

            # Create heatmap with custom colormap
            sns.heatmap(display_data, annot=True, fmt='.1f', cmap='RdYlGn',
                       center=0, cbar_kws={'label': 'Performance (%)'}, ax=ax,
                       linewidths=0.5)

            ax.set_title('Industry Performance Matrix (Top 15 Industries)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Timeframe')
            ax.set_ylabel('Industry')

            # Truncate long industry names
            yticklabels = [label.get_text()[:25] + '...' if len(label.get_text()) > 25
                          else label.get_text() for label in ax.get_yticklabels()]
            ax.set_yticklabels(yticklabels, rotation=0)

            plt.tight_layout()
            chart_path = self.png_dir / "industry_performance_matrix.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating industry performance matrix: {e}")
            return ""

    def _create_industry_momentum_bubble(self) -> str:
        """Create industry momentum bubble chart."""
        try:
            if 'industry' not in self.df.columns:
                return ""

            # Calculate industry momentum and performance
            has_market_cap = 'market_cap' in self.df.columns
            if has_market_cap:
                industry_perf = self._calculate_weighted_industry_performance()
            else:
                industry_perf = self.df.groupby('industry')[list(self.available_timeframes.values())].mean()

            industry_momentum = self._calculate_industry_momentum(industry_perf)
            industry_rotation = self._detect_rotation_signals(industry_perf)

            if industry_momentum.empty:
                return ""

            # Get short and long term columns
            short_term_col = None
            long_term_col = None

            for label in ['7D', '1D']:
                if label in self.available_timeframes:
                    short_term_col = self.available_timeframes[label]
                    break

            for label in ['252D', '66D', '22D']:
                if label in self.available_timeframes:
                    long_term_col = self.available_timeframes[label]
                    break

            if not short_term_col or not long_term_col:
                return ""

            # Calculate bubble sizes (stock count per industry)
            industry_counts = self.df.groupby('industry').size()

            fig, ax = plt.subplots(figsize=(14, 10))

            # Color mapping for rotation signals
            colors = {'STRONG_IN': 'green', 'ROTATING_OUT': 'red', 'NEUTRAL': 'gray'}

            for industry in industry_momentum.index:
                x = industry_momentum.loc[industry, short_term_col] if short_term_col in industry_momentum.columns else 0
                y = industry_momentum.loc[industry, long_term_col] if long_term_col in industry_momentum.columns else 0

                size = industry_counts.get(industry, 1) * 20  # Scale bubble size
                rotation_signal = industry_rotation.loc[industry, 'rotation_signal'] if industry in industry_rotation.index else 'NEUTRAL'
                color = colors.get(rotation_signal, 'gray')

                ax.scatter(x, y, s=size, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)

                # Add industry label for significant industries
                if size > 100:  # Only label larger industries
                    ax.annotate(industry[:15] + '...' if len(industry) > 15 else industry,
                               (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)

            ax.set_xlabel(f'Short-term Performance (%) - {list(self.available_timeframes.keys())[list(self.available_timeframes.values()).index(short_term_col)]}')
            ax.set_ylabel(f'Long-term Performance (%) - {list(self.available_timeframes.keys())[list(self.available_timeframes.values()).index(long_term_col)]}')
            ax.set_title('Industry Momentum Bubble Chart', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add quadrant lines
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Create legend
            legend_elements = [plt.scatter([], [], s=100, color=color, label=signal, alpha=0.6, edgecolors='black')
                             for signal, color in colors.items()]
            ax.legend(handles=legend_elements, loc='upper left')

            plt.tight_layout()
            chart_path = self.png_dir / "industry_momentum_bubble.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating industry momentum bubble: {e}")
            return ""

    def _create_industry_dispersion_box(self) -> str:
        """Create box plots showing performance dispersion within industries."""
        try:
            if 'industry' not in self.df.columns:
                return ""

            # Use longest timeframe for dispersion analysis
            long_term_col = None
            for label in ['252D', '66D', '22D', '7D', '1D']:
                if label in self.available_timeframes:
                    long_term_col = self.available_timeframes[label]
                    break

            if not long_term_col:
                return ""

            # Get top 10 industries by average performance
            industry_means = self.df.groupby('industry')[long_term_col].mean().sort_values(ascending=False)
            top_industries = industry_means.head(10).index.tolist()

            # Prepare data for box plots
            box_data = []
            box_labels = []

            for industry in top_industries:
                industry_data = self.df[self.df['industry'] == industry][long_term_col].dropna()
                if len(industry_data) >= 3:  # Only include industries with enough data
                    box_data.append(industry_data.values)
                    box_labels.append(industry[:20] + '...' if len(industry) > 20 else industry)

            if not box_data:
                return ""

            fig, ax = plt.subplots(figsize=(14, 8))

            # Create box plots
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)

            # Color box plots based on median performance
            colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xlabel('Industry')
            ax.set_ylabel(f'Performance (%) - {list(self.available_timeframes.keys())[list(self.available_timeframes.values()).index(long_term_col)]}')
            ax.set_title('Industry Performance Dispersion (Top 10 Industries)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            chart_path = self.png_dir / "industry_dispersion_box.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating industry dispersion box: {e}")
            return ""

    def _create_leader_laggard_comparison(self) -> str:
        """Create leader-laggard comparison charts for top industries."""
        try:
            if 'industry' not in self.df.columns:
                return ""

            # Get leaders/laggards data
            leaders_laggards = self._identify_industry_leaders_laggards()

            if not leaders_laggards:
                return ""

            # Select top 6 industries by stock count for comparison
            sorted_industries = sorted(leaders_laggards.items(),
                                     key=lambda x: x[1]['stock_count'], reverse=True)[:6]

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, (industry, data) in enumerate(sorted_industries):
                if i >= 6:
                    break

                ax = axes[i]

                # Prepare data for plotting
                leaders_df = data['leaders']
                laggards_df = data['laggards']

                # Get performance column name
                perf_col = [col for col in leaders_df.columns if 'pct_change' in col][0]

                # Combine leaders and laggards
                plot_data = []
                plot_labels = []
                colors_list = []

                # Add leaders (green shades)
                for idx, (_, row) in enumerate(leaders_df.iterrows()):
                    plot_data.append(row[perf_col])
                    plot_labels.append(f"{row['ticker']} (L)")
                    colors_list.append(plt.cm.Greens(0.5 + 0.2 * idx))

                # Add laggards (red shades)
                for idx, (_, row) in enumerate(laggards_df.iterrows()):
                    plot_data.append(row[perf_col])
                    plot_labels.append(f"{row['ticker']} (Lag)")
                    colors_list.append(plt.cm.Reds(0.5 + 0.2 * idx))

                # Create bar chart
                bars = ax.bar(range(len(plot_data)), plot_data, color=colors_list, alpha=0.7, edgecolor='black')

                # Add value labels on bars
                for bar, value in zip(bars, plot_data):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                           f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

                ax.set_xticks(range(len(plot_labels)))
                ax.set_xticklabels(plot_labels, rotation=45, ha='right')
                ax.set_ylabel('Performance (%)')
                ax.set_title(f'{industry[:25]}...' if len(industry) > 25 else industry, fontweight='bold', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Hide unused subplots
            for i in range(len(sorted_industries), 6):
                axes[i].set_visible(False)

            plt.suptitle('Leaders vs Laggards by Industry (Top 6 Industries)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            chart_path = self.png_dir / "leader_laggard_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating leader-laggard comparison: {e}")
            return ""

    def _create_industry_rotation_flow(self) -> str:
        """Create industry rotation flow visualization."""
        try:
            if 'industry' not in self.df.columns:
                return ""

            # Get industry rotation signals
            has_market_cap = 'market_cap' in self.df.columns
            if has_market_cap:
                industry_perf = self._calculate_weighted_industry_performance()
            else:
                industry_perf = self.df.groupby('industry')[list(self.available_timeframes.values())].mean()

            industry_rotation = self._detect_rotation_signals(industry_perf)
            industry_classification = self._classify_industry_performance(industry_perf)

            if industry_rotation.empty:
                return ""

            # Count industries by rotation signal and classification
            rotation_counts = industry_rotation['rotation_signal'].value_counts()
            classification_counts = industry_classification['classification'].value_counts()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Rotation signals pie chart
            colors_rotation = {'STRONG_IN': 'green', 'ROTATING_OUT': 'red', 'NEUTRAL': 'gray'}
            rotation_colors = [colors_rotation.get(signal, 'gray') for signal in rotation_counts.index]

            wedges1, texts1, autotexts1 = ax1.pie(rotation_counts.values, labels=rotation_counts.index,
                                                  autopct='%1.1f%%', colors=rotation_colors,
                                                  startangle=90)
            ax1.set_title('Industry Rotation Signals', fontsize=14, fontweight='bold')

            # Classification pie chart
            colors_class = {'LEADERS': 'darkgreen', 'EMERGING': 'lightgreen',
                           'DECLINING': 'orange', 'LAGGARDS': 'darkred', 'NEUTRAL': 'gray'}
            class_colors = [colors_class.get(cls, 'gray') for cls in classification_counts.index]

            wedges2, texts2, autotexts2 = ax2.pie(classification_counts.values, labels=classification_counts.index,
                                                  autopct='%1.1f%%', colors=class_colors,
                                                  startangle=90)
            ax2.set_title('Industry Classification', fontsize=14, fontweight='bold')

            # Style the text
            for autotext in autotexts1 + autotexts2:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            plt.suptitle('Industry Rotation Flow Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            chart_path = self.png_dir / "industry_rotation_flow.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error creating industry rotation flow: {e}")
            return ""

    def assemble_pdf(self, pdf_path: str, performance_analysis: dict,
                    sector_analysis: dict, chart_paths: dict) -> bool:
        """Step 4: Assemble professional PDF report."""
        logger.info("Step 4: Assembling PDF report")

        try:
            # Create PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                  rightMargin=0.5*inch, leftMargin=0.5*inch,
                                  topMargin=0.75*inch, bottomMargin=0.5*inch)

            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                       fontSize=18, spaceAfter=20, textColor=colors.darkblue,
                                       alignment=TA_CENTER)

            # Build content
            content = []

            # Cover Page & Executive Summary
            content.extend(self._create_cover_page(styles, performance_analysis, sector_analysis))
            content.append(PageBreak())

            # Multi-Timeframe Performance Section
            content.extend(self._create_performance_section(styles, performance_analysis, chart_paths))
            content.append(PageBreak())

            # Top Stocks/Leaders Section
            content.extend(self._create_leaders_section(styles, performance_analysis, chart_paths))
            content.append(PageBreak())

            # Sector Analysis Section
            content.extend(self._create_sector_section(styles, sector_analysis, chart_paths))
            content.append(PageBreak())

            # Enhanced Industry Analysis Section (3 pages)
            content.extend(self._create_enhanced_industry_section(styles, sector_analysis, chart_paths))
            content.append(PageBreak())

            # Intra-Industry Leaders & Laggards Section
            content.extend(self._create_industry_leaders_section(styles, sector_analysis, chart_paths))
            content.append(PageBreak())

            # Industry Risk & Attribution Section
            content.extend(self._create_industry_risk_section(styles, sector_analysis, chart_paths))
            content.append(PageBreak())

            # Momentum & Rotation Analysis
            content.extend(self._create_momentum_section(styles, performance_analysis, chart_paths))
            content.append(PageBreak())

            # Enhanced Investment Recommendations (with industry insights)
            content.extend(self._create_recommendations_section(styles, performance_analysis, sector_analysis))

            # Build PDF
            doc.build(content)
            return True

        except Exception as e:
            logger.error(f"Error assembling PDF: {e}")
            return False

    def _create_cover_page(self, styles, performance_analysis: dict, sector_analysis: dict) -> list:
        """Create cover page with executive summary."""
        content = []

        # Title
        title_text = f"Daily Market Analysis Report"
        if self.metadata and 'original_filename' in self.metadata:
            title_text += f" - {datetime.now().strftime('%Y-%m-%d')}"

        content.append(Paragraph(title_text, styles['Title']))
        content.append(Spacer(1, 0.4*inch))

        # Executive Summary
        content.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
        content.append(Spacer(1, 0.2*inch))

        # Generate executive insights
        insights = self._generate_executive_insights(performance_analysis, sector_analysis)

        for insight in insights:
            content.append(Paragraph(f"• {insight}", styles['Normal']))
            content.append(Spacer(1, 0.1*inch))

        content.append(Spacer(1, 0.3*inch))

        # Key metrics summary
        content.append(Paragraph("<b>Key Market Metrics</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        metrics_table = self._create_key_metrics_table(performance_analysis)
        if metrics_table:
            content.append(metrics_table)

        return content

    def _create_performance_section(self, styles, performance_analysis: dict, chart_paths: dict) -> list:
        """Create multi-timeframe performance section."""
        content = []

        content.append(Paragraph("<b>Multi-Timeframe Performance Analysis</b>", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Performance distribution chart
        if chart_paths.get('distribution_chart'):
            content.append(Paragraph("<b>Performance Distributions</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['distribution_chart'], width=7*inch, height=5*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Summary statistics table
        content.extend(self._create_summary_stats_table(performance_analysis, styles))

        return content

    def _create_leaders_section(self, styles, performance_analysis: dict, chart_paths: dict) -> list:
        """Create top stocks/leaders section."""
        content = []

        content.append(Paragraph("<b>Top Stocks & Market Leaders</b>", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Performance bar charts
        if chart_paths.get('performance_bars'):
            content.append(Paragraph("<b>Top Performers by Timeframe</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['performance_bars'], width=7*inch, height=5*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Momentum leaders table
        if 'momentum_leaders' in performance_analysis:
            content.extend(self._create_momentum_leaders_table(performance_analysis['momentum_leaders'], styles))

        return content

    def _create_sector_section(self, styles, sector_analysis: dict, chart_paths: dict) -> list:
        """Create sector/industry analysis section."""
        content = []

        content.append(Paragraph("<b>Sector & Industry Analysis</b>", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Sector heatmap
        if chart_paths.get('sector_heatmap'):
            content.append(Paragraph("<b>Sector Performance Heatmap</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['sector_heatmap'], width=7*inch, height=5*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Tornado chart
        if chart_paths.get('tornado_chart'):
            content.append(Paragraph("<b>Sector Dispersion Analysis</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['tornado_chart'], width=7*inch, height=5*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Rotation signals table
        if 'sector_rotation' in sector_analysis:
            content.extend(self._create_rotation_signals_table(sector_analysis['sector_rotation'], styles))

        return content

    def _create_momentum_section(self, styles, performance_analysis: dict, chart_paths: dict) -> list:
        """Create momentum & rotation analysis section."""
        content = []

        content.append(Paragraph("<b>Momentum & Rotation Analysis</b>", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Momentum scatter plot
        if chart_paths.get('momentum_scatter'):
            content.append(Paragraph("<b>Momentum vs Long-term Performance</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['momentum_scatter'], width=7*inch, height=5*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        return content

    def _create_recommendations_section(self, styles, performance_analysis: dict, sector_analysis: dict) -> list:
        """Create investment recommendations section."""
        content = []

        content.append(Paragraph("<b>Investment Recommendations</b>", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Generate recommendations
        recommendations = self._generate_investment_recommendations(performance_analysis, sector_analysis)

        for timeframe, recs in recommendations.items():
            content.append(Paragraph(f"<b>{timeframe} Recommendations:</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            for rec in recs:
                content.append(Paragraph(f"• {rec}", styles['Normal']))
                content.append(Spacer(1, 0.05*inch))

            content.append(Spacer(1, 0.2*inch))

        return content

    def _create_enhanced_industry_section(self, styles, sector_analysis: dict, chart_paths: dict) -> list:
        """Create enhanced industry performance overview section."""
        content = []

        content.append(Paragraph("<b>Enhanced Industry Performance Analysis</b>", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Industry performance matrix heatmap
        if chart_paths.get('industry_performance_matrix'):
            content.append(Paragraph("<b>Industry Performance Matrix</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['industry_performance_matrix'], width=7*inch, height=5*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Industry momentum bubble chart
        if chart_paths.get('industry_momentum_bubble'):
            content.append(Paragraph("<b>Industry Momentum & Rotation Signals</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['industry_momentum_bubble'], width=7*inch, height=5*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Industry classification and rotation summary
        if 'industry_classification' in sector_analysis and 'industry_rotation' in sector_analysis:
            content.extend(self._create_industry_summary_tables(sector_analysis, styles))

        return content

    def _create_industry_leaders_section(self, styles, sector_analysis: dict, chart_paths: dict) -> list:
        """Create intra-industry leaders & laggards section."""
        content = []

        content.append(Paragraph("<b>Intra-Industry Leaders & Laggards Analysis</b>", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Leaders vs laggards comparison
        if chart_paths.get('leader_laggard_comparison'):
            content.append(Paragraph("<b>Leaders vs Laggards by Industry</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['leader_laggard_comparison'], width=8*inch, height=6*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Industry dispersion analysis
        if chart_paths.get('industry_dispersion_box'):
            content.append(Paragraph("<b>Performance Dispersion Within Industries</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['industry_dispersion_box'], width=7*inch, height=4*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Leaders/laggards detailed table
        if 'industry_leaders_laggards' in sector_analysis:
            content.extend(self._create_leaders_laggards_table(sector_analysis['industry_leaders_laggards'], styles))

        return content

    def _create_industry_risk_section(self, styles, sector_analysis: dict, chart_paths: dict) -> list:
        """Create industry risk & attribution analysis section."""
        content = []

        content.append(Paragraph("<b>Industry Risk & Attribution Analysis</b>", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Industry rotation flow
        if chart_paths.get('industry_rotation_flow'):
            content.append(Paragraph("<b>Industry Rotation Flow Analysis</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            img = RLImage(chart_paths['industry_rotation_flow'], width=7*inch, height=4*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Risk metrics table
        if 'industry_risk_metrics' in sector_analysis:
            content.extend(self._create_industry_risk_table(sector_analysis['industry_risk_metrics'], styles))

        return content

    def _create_industry_summary_tables(self, sector_analysis: dict, styles) -> list:
        """Create industry classification and rotation summary tables."""
        content = []

        # Industry classification summary
        if 'industry_classification' in sector_analysis:
            classification_data = sector_analysis['industry_classification']
            classification_counts = classification_data['classification'].value_counts()

            content.append(Paragraph("<b>Industry Classification Summary</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            # Create classification table
            table_data = [['Classification', 'Count', 'Percentage']]
            total_industries = len(classification_data)

            for classification, count in classification_counts.items():
                percentage = (count / total_industries) * 100 if total_industries > 0 else 0
                table_data.append([classification, str(count), f"{percentage:.1f}%"])

            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            content.append(table)
            content.append(Spacer(1, 0.2*inch))

        return content

    def _create_leaders_laggards_table(self, leaders_laggards: dict, styles) -> list:
        """Create detailed leaders/laggards table."""
        content = []

        content.append(Paragraph("<b>Top Industry Performance Gaps</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        # Sort industries by performance gap
        sorted_industries = sorted(leaders_laggards.items(),
                                 key=lambda x: x[1]['performance_gap'], reverse=True)[:10]

        table_data = [['Industry', 'Top Performer', 'Performance', 'Bottom Performer', 'Performance', 'Gap (%)']]

        for industry, data in sorted_industries:
            if not data['leaders'].empty and not data['laggards'].empty:
                top_performer = data['leaders'].iloc[0]
                bottom_performer = data['laggards'].iloc[-1]

                # Get performance column
                perf_cols = [col for col in top_performer.index if 'pct_change' in col]
                if perf_cols:
                    perf_col = perf_cols[0]
                    top_perf = top_performer[perf_col]
                    bottom_perf = bottom_performer[perf_col]
                    gap = data['performance_gap']

                    table_data.append([
                        industry[:20] + '...' if len(industry) > 20 else industry,
                        top_performer['ticker'],
                        f"{top_perf:.1f}%",
                        bottom_performer['ticker'],
                        f"{bottom_perf:.1f}%",
                        f"{gap:.1f}%"
                    ])

        if len(table_data) > 1:  # Only create table if we have data
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            content.append(table)
            content.append(Spacer(1, 0.2*inch))

        return content

    def _create_industry_risk_table(self, risk_metrics: dict, styles) -> list:
        """Create industry risk metrics table."""
        content = []

        content.append(Paragraph("<b>Industry Risk Concentration Metrics</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        # Sort industries by performance standard deviation (risk)
        sorted_industries = sorted(risk_metrics.items(),
                                 key=lambda x: x[1]['performance_std'], reverse=True)[:10]

        table_data = [['Industry', 'Std Dev (%)', 'Range (%)', 'IQR (%)', 'Gini Coeff', 'Stock Count']]

        for industry, metrics in sorted_industries:
            table_data.append([
                industry[:20] + '...' if len(industry) > 20 else industry,
                f"{metrics['performance_std']:.1f}",
                f"{metrics['performance_range']:.1f}",
                f"{metrics['performance_iqr']:.1f}",
                f"{metrics['concentration_ratio']:.2f}",
                str(metrics['stock_count'])
            ])

        if len(table_data) > 1:  # Only create table if we have data
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            content.append(table)
            content.append(Spacer(1, 0.2*inch))

        return content

    def _generate_executive_insights(self, performance_analysis: dict, sector_analysis: dict) -> list:
        """Generate executive summary insights."""
        insights = []

        try:
            # Market breadth insight
            total_stocks = len(self.df)
            insights.append(f"Analyzed {total_stocks} stocks across {len(self.available_timeframes)} timeframes")

            # Top performer insight
            if 'momentum_leaders' in performance_analysis and not performance_analysis['momentum_leaders'].empty:
                top_leader = performance_analysis['momentum_leaders'].iloc[0]
                insights.append(f"Top momentum leader: {top_leader['ticker']} with {top_leader['momentum_score']:.1f} momentum score")

            # Sector rotation insight
            if 'sector_rotation' in sector_analysis:
                rotation_data = sector_analysis['sector_rotation']
                strong_sectors = rotation_data[rotation_data['rotation_signal'] == 'STRONG_IN'].index.tolist()
                weak_sectors = rotation_data[rotation_data['rotation_signal'] == 'ROTATING_OUT'].index.tolist()

                if strong_sectors:
                    insights.append(f"Strong inflow sectors: {', '.join(strong_sectors[:3])}")
                if weak_sectors:
                    insights.append(f"Rotating out sectors: {', '.join(weak_sectors[:3])}")

            # Industry analysis insights
            if 'industry_classification' in sector_analysis:
                classification_data = sector_analysis['industry_classification']
                leaders = classification_data[classification_data['classification'] == 'LEADERS']
                laggards = classification_data[classification_data['classification'] == 'LAGGARDS']
                insights.append(f"Industry classification: {len(leaders)} leaders, {len(laggards)} laggards")

            if 'industry_leaders_laggards' in sector_analysis:
                leaders_laggards = sector_analysis['industry_leaders_laggards']
                top_gaps = sorted(leaders_laggards.items(), key=lambda x: x[1]['performance_gap'], reverse=True)[:3]
                if top_gaps:
                    top_industry = top_gaps[0][0]
                    top_gap = top_gaps[0][1]['performance_gap']
                    insights.append(f"Highest intra-industry gap: {top_industry[:25]}{'...' if len(top_industry) > 25 else ''} ({top_gap:.1f}%)")

            if 'industry_rotation' in sector_analysis:
                industry_rotation = sector_analysis['industry_rotation']
                strong_industries = industry_rotation[industry_rotation['rotation_signal'] == 'STRONG_IN'].index.tolist()
                if strong_industries:
                    insights.append(f"Strong momentum industries: {', '.join(strong_industries[:2])}")

            # Performance distribution insight
            if 'summary_stats' in performance_analysis:
                long_term_stats = None
                for timeframe in ['252D', '66D', '22D']:
                    if timeframe in performance_analysis['summary_stats']:
                        long_term_stats = performance_analysis['summary_stats'][timeframe]
                        break

                if long_term_stats:
                    positive_pct = (long_term_stats['positive_count'] / long_term_stats['total_count']) * 100
                    insights.append(f"Market breadth: {positive_pct:.0f}% of stocks showing positive long-term performance")

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Analysis completed successfully")

        return insights

    def _create_key_metrics_table(self, performance_analysis: dict) -> Table:
        """Create key metrics summary table."""
        try:
            if 'summary_stats' not in performance_analysis:
                return None

            table_data = [['Timeframe', 'Mean (%)', 'Median (%)', 'Std Dev (%)', 'Winners/Total']]

            for timeframe, stats in performance_analysis['summary_stats'].items():
                winner_ratio = f"{stats['positive_count']}/{stats['total_count']}"
                table_data.append([
                    timeframe,
                    f"{stats['mean']:.1f}",
                    f"{stats['median']:.1f}",
                    f"{stats['std']:.1f}",
                    winner_ratio
                ])

            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            return table

        except Exception as e:
            logger.error(f"Error creating metrics table: {e}")
            return None

    def _create_summary_stats_table(self, performance_analysis: dict, styles) -> list:
        """Create detailed summary statistics table."""
        content = []

        metrics_table = self._create_key_metrics_table(performance_analysis)
        if metrics_table:
            content.append(Paragraph("<b>Performance Statistics Summary</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            content.append(metrics_table)
            content.append(Spacer(1, 0.2*inch))

        return content

    def _create_momentum_leaders_table(self, momentum_leaders: pd.DataFrame, styles) -> list:
        """Create momentum leaders table."""
        content = []

        content.append(Paragraph("<b>Top Momentum Leaders</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        table_data = [['Rank', 'Ticker', 'Momentum Score', 'Trend Consistency (%)']]

        for idx, (_, row) in enumerate(momentum_leaders.head(10).iterrows(), 1):
            table_data.append([
                str(idx),
                row['ticker'],
                f"{row['momentum_score']:.2f}",
                f"{row['trend_consistency']:.0f}%"
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        content.append(table)
        content.append(Spacer(1, 0.2*inch))

        return content

    def _create_rotation_signals_table(self, sector_rotation: pd.DataFrame, styles) -> list:
        """Create sector rotation signals table."""
        content = []

        content.append(Paragraph("<b>Sector Rotation Signals</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        # Group by rotation signal
        signals_summary = sector_rotation.groupby('rotation_signal').size()

        summary_text = f"<b>Signal Distribution:</b> "
        for signal, count in signals_summary.items():
            summary_text += f"{signal}: {count} sectors, "
        summary_text = summary_text.rstrip(', ')

        content.append(Paragraph(summary_text, styles['Normal']))
        content.append(Spacer(1, 0.1*inch))

        # Detailed table
        table_data = [['Sector', 'Signal', 'Short-term Trend']]

        for sector, row in sector_rotation.iterrows():
            signal = row['rotation_signal']

            # Get short-term trend (average of 1D and 7D if available)
            short_term_cols = []
            for label in ['1D', '7D']:
                if label in self.available_timeframes:
                    col = self.available_timeframes[label]
                    if col in row.index:
                        short_term_cols.append(row[col])

            if short_term_cols:
                short_term_avg = np.mean(short_term_cols)
                trend_text = f"{short_term_avg:.1f}%"
            else:
                trend_text = "N/A"

            table_data.append([
                sector[:25] + '...' if len(sector) > 25 else sector,
                signal,
                trend_text
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        content.append(table)
        content.append(Spacer(1, 0.2*inch))

        return content

    def _generate_investment_recommendations(self, performance_analysis: dict, sector_analysis: dict) -> dict:
        """Generate timeframe-specific investment recommendations."""
        recommendations = {}

        try:
            # Short-term recommendations (1D-7D)
            short_term_recs = []
            if 'momentum_leaders' in performance_analysis and not performance_analysis['momentum_leaders'].empty:
                top_momentum = performance_analysis['momentum_leaders'].head(3)['ticker'].tolist()
                short_term_recs.append(f"Focus on momentum leaders: {', '.join(top_momentum)}")

            if 'sector_rotation' in sector_analysis:
                strong_sectors = sector_analysis['sector_rotation'][
                    sector_analysis['sector_rotation']['rotation_signal'] == 'STRONG_IN'
                ].index.tolist()
                if strong_sectors:
                    short_term_recs.append(f"Consider exposure to strong inflow sectors: {', '.join(strong_sectors[:2])}")

            # Industry-specific short-term recommendations
            if 'industry_rotation' in sector_analysis:
                strong_industries = sector_analysis['industry_rotation'][
                    sector_analysis['industry_rotation']['rotation_signal'] == 'STRONG_IN'
                ].index.tolist()
                if strong_industries:
                    short_term_recs.append(f"Strong momentum industries: {', '.join(strong_industries[:2])}")

            if 'industry_leaders_laggards' in sector_analysis:
                leaders_laggards = sector_analysis['industry_leaders_laggards']
                for industry, data in list(leaders_laggards.items())[:2]:  # Top 2 industries
                    if not data['leaders'].empty:
                        top_performer = data['leaders'].iloc[0]['ticker']
                        short_term_recs.append(f"{industry[:15]}{'...' if len(industry) > 15 else ''} leader: {top_performer}")

            recommendations['Short-term (1-7 days)'] = short_term_recs

            # Medium-term recommendations (22D-66D)
            medium_term_recs = []
            if '22D' in performance_analysis.get('top_performers', {}):
                monthly_leaders = performance_analysis['top_performers']['22D'].head(3)['ticker'].tolist()
                medium_term_recs.append(f"Monthly trend leaders: {', '.join(monthly_leaders)}")

            # Industry classification insights for medium-term
            if 'industry_classification' in sector_analysis:
                classification_data = sector_analysis['industry_classification']
                emerging_industries = classification_data[classification_data['classification'] == 'EMERGING'].index.tolist()
                if emerging_industries:
                    medium_term_recs.append(f"Emerging industries to watch: {', '.join(emerging_industries[:2])}")

            recommendations['Medium-term (1-3 months)'] = medium_term_recs

            # Long-term recommendations (252D)
            long_term_recs = []
            if '252D' in performance_analysis.get('top_performers', {}):
                yearly_leaders = performance_analysis['top_performers']['252D'].head(3)['ticker'].tolist()
                long_term_recs.append(f"Long-term leadership positions: {', '.join(yearly_leaders)}")

            # Industry leaders for long-term positioning
            if 'industry_classification' in sector_analysis:
                classification_data = sector_analysis['industry_classification']
                leader_industries = classification_data[classification_data['classification'] == 'LEADERS'].index.tolist()
                if leader_industries:
                    long_term_recs.append(f"Leading industries for long-term positioning: {', '.join(leader_industries[:2])}")

            # Risk management insights
            if 'industry_risk_metrics' in sector_analysis:
                risk_metrics = sector_analysis['industry_risk_metrics']
                low_risk_industries = sorted(risk_metrics.items(), key=lambda x: x[1]['performance_std'])[:2]
                if low_risk_industries:
                    industry_names = [industry[:15] + '...' if len(industry) > 15 else industry
                                    for industry, _ in low_risk_industries]
                    long_term_recs.append(f"Lower volatility industries: {', '.join(industry_names)}")

            recommendations['Long-term (6+ months)'] = long_term_recs

            # Industry-specific risk recommendations
            industry_risk_recs = []
            if 'industry_risk_metrics' in sector_analysis:
                risk_metrics = sector_analysis['industry_risk_metrics']
                high_risk_industries = sorted(risk_metrics.items(),
                                            key=lambda x: x[1]['performance_std'], reverse=True)[:3]
                if high_risk_industries:
                    for industry, metrics in high_risk_industries:
                        if metrics['concentration_ratio'] > 0.5:  # High concentration
                            industry_risk_recs.append(f"Monitor {industry[:20]}{'...' if len(industry) > 20 else ''}: high concentration risk")

            if industry_risk_recs:
                recommendations['Risk Management'] = industry_risk_recs

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations['General'] = ["Maintain diversified portfolio based on analysis"]

        return recommendations


