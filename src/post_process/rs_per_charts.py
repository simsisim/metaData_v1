"""
RS/PER Advanced Visualization Engine
===================================

Creates comprehensive charts for RS/PER analysis including:
- Multi-Timeframe RS Heatmap (Top 20 Stocks)
- Sector Relative Rotation Graph (RRG)
- Industry Relative Rotation Graph (RRG)
- Leadership Strength Bar Chart
- Momentum Pattern Scatter Plot
- Elite Stocks Radar Chart

All charts are optimized for professional PDF reports with consistent styling.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Set matplotlib backend for server environments
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class RSPERChartGenerator:
    """
    Advanced chart generator for RS/PER analysis.
    Creates professional-quality visualizations for comprehensive market analysis.
    """

    def __init__(self, config=None, output_dir: str = None):
        """Initialize the chart generator."""
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('charts')
        self.output_dir.mkdir(exist_ok=True)

        # Set professional styling
        self._setup_chart_styling()

        # Color schemes for different chart types
        self.color_schemes = {
            'heatmap': 'RdYlGn',
            'sector_colors': {
                'Technology': '#2E8B57',
                'Healthcare': '#4169E1',
                'Finance': '#DAA520',
                'Energy': '#DC143C',
                'Consumer': '#9932CC',
                'Industrial': '#FF8C00',
                'Materials': '#8B4513',
                'Utilities': '#708090',
                'Real Estate': '#228B22',
                'Communication': '#4682B4'
            },
            'quadrant_colors': {
                'leading_improving': '#00FF00',
                'leading_weakening': '#FFFF00',
                'lagging_improving': '#87CEEB',
                'lagging_weakening': '#FF6347'
            },
            'momentum_colors': {
                'ACCELERATING': '#00FF00',
                'BUILDING': '#90EE90',
                'CONSOLIDATING': '#FFFF00',
                'MIXED': '#FFA500',
                'WEAKENING': '#FF6347',
                'DECELERATING': '#FF0000'
            }
        }

    def _setup_chart_styling(self):
        """Set up professional chart styling."""
        # Set style parameters
        plt.style.use('default')
        sns.set_palette("husl")

        # Configure matplotlib parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })

    def generate_all_charts(self, analysis_results: Dict, date_str: str = None) -> Dict[str, str]:
        """
        Generate all charts for the comprehensive RS/PER report.

        Args:
            analysis_results: Complete analysis results from RSPERAnalyzer
            date_str: Date string for file naming

        Returns:
            Dictionary mapping chart names to file paths
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')

        logger.info("Generating all RS/PER charts")

        charts = {}

        try:
            # A. Multi-Timeframe RS Heatmap (Top 20 Stocks)
            if analysis_results.get('stocks_analysis'):
                charts['heatmap_rs'] = self.create_rs_heatmap(
                    analysis_results['stocks_analysis'], date_str
                )

            # B. Sector Relative Rotation Graph (RRG)
            if analysis_results.get('sectors_analysis'):
                charts['sector_rrg'] = self.create_sector_rrg(
                    analysis_results['sectors_analysis'], date_str
                )

            # C. Industry Relative Rotation Graph (RRG)
            if analysis_results.get('industries_analysis'):
                charts['industry_rrg'] = self.create_industry_rrg(
                    analysis_results['industries_analysis'], date_str
                )

            # D. Leadership Strength Bar Chart
            if analysis_results.get('stocks_analysis'):
                charts['leadership_strength'] = self.create_leadership_chart(
                    analysis_results['stocks_analysis'], date_str
                )

            # E. Momentum Pattern Scatter Plot
            if analysis_results.get('stocks_analysis'):
                charts['momentum_patterns'] = self.create_momentum_scatter(
                    analysis_results['stocks_analysis'], date_str
                )

            # F. Elite Stocks Radar Chart
            if analysis_results.get('stocks_analysis'):
                charts['elite_radar'] = self.create_elite_radar(
                    analysis_results['stocks_analysis'], date_str
                )

            logger.info(f"Generated {len(charts)} charts successfully")
            return charts

        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            raise

    def create_rs_heatmap(self, stocks_analysis: Dict, date_str: str) -> str:
        """Create Multi-Timeframe RS Heatmap for top 20 stocks."""
        try:
            logger.info("Creating RS heatmap")

            # Get top 20 stocks by composite strength
            top_stocks = stocks_analysis['top_performers'][:20]

            if not top_stocks:
                logger.warning("No stocks available for heatmap")
                return None

            # Prepare heatmap data
            heatmap_data = []
            timeframes = ['3d', '5d', '7d', '14d', '22d', '44d', '66d', '132d', '252d']

            for stock in top_stocks:
                row_data = {'ticker': stock['ticker']}
                for tf in timeframes:
                    row_data[tf] = stock['rs_values'].get(tf, 1.0)
                heatmap_data.append(row_data)

            # Create DataFrame
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_df = heatmap_df.set_index('ticker')

            # Create heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(
                heatmap_df,
                cmap=self.color_schemes['heatmap'],
                center=1.0,
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Relative Strength vs QQQ'},
                linewidths=0.5
            )

            plt.title(f"Top 20 Stocks Multi-Timeframe Relative Strength Heatmap\nDate: {date_str}")
            plt.xlabel("Timeframes")
            plt.ylabel("Stocks (Ranked by Composite Strength)")
            plt.tight_layout()

            # Save chart
            filename = f"heatmap_rs_{date_str}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"RS heatmap saved: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create RS heatmap: {e}")
            plt.close()
            return None

    def create_sector_rrg(self, sectors_analysis: Dict, date_str: str) -> str:
        """Create Sector Relative Rotation Graph (RRG)."""
        try:
            logger.info("Creating sector RRG")

            sectors_data = sectors_analysis['individual_analysis']

            if not sectors_data:
                logger.warning("No sectors data available for RRG")
                return None

            plt.figure(figsize=(12, 10))

            # Plot sectors
            for sector in sectors_data:
                rs_strength = sector['composite_strength']
                momentum = sector['momentum_medium']
                sector_name = sector['sector']

                # Get color for sector
                color = self._get_sector_color(sector_name)

                plt.scatter(rs_strength, momentum, s=150, c=color, alpha=0.7, edgecolors='black')

                # Add sector label
                plt.annotate(
                    self._get_short_name(sector_name, 12),
                    (rs_strength, momentum),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8
                )

            # Add quadrant lines
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            plt.axvline(x=1.0, color='black', linestyle='-', alpha=0.3, linewidth=1)

            # Add quadrant labels
            self._add_rrg_quadrant_labels()

            plt.xlabel("Sector RS Strength vs QQQ")
            plt.ylabel("Sector Momentum (14d vs 44d %)")
            plt.title(f"Sector Relative Rotation Graph (RRG)\nSector Positioning and Momentum Analysis - {date_str}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save chart
            filename = f"sector_rrg_{date_str}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Sector RRG saved: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create sector RRG: {e}")
            plt.close()
            return None

    def create_industry_rrg(self, industries_analysis: Dict, date_str: str) -> str:
        """Create Industry Relative Rotation Graph (RRG)."""
        try:
            logger.info("Creating industry RRG")

            industries_data = industries_analysis['individual_analysis']

            if not industries_data:
                logger.warning("No industries data available for RRG")
                return None

            plt.figure(figsize=(14, 12))

            # Plot industries
            for industry in industries_data:
                rs_strength = industry['composite_strength']
                momentum = industry['momentum_medium']
                industry_name = industry['industry']
                market_cap_weight = industry.get('market_cap_weight', 1.0)

                # Color based on classification
                color = self._get_industry_color(industry['classification'])

                # Size based on market cap weight
                size = max(50, market_cap_weight * 100)

                plt.scatter(rs_strength, momentum, s=size, c=color, alpha=0.7, edgecolors='black')

                # Add industry label
                short_name = industry.get('industry_short_name', self._get_short_name(industry_name, 10))
                plt.annotate(
                    short_name,
                    (rs_strength, momentum),
                    xytext=(3, 3),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )

            # Add quadrant lines
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            plt.axvline(x=1.0, color='black', linestyle='-', alpha=0.3, linewidth=1)

            # Add quadrant labels
            self._add_rrg_quadrant_labels()

            plt.xlabel("Industry RS Strength vs QQQ")
            plt.ylabel("Industry Momentum (14d vs 44d %)")
            plt.title(f"Industry Relative Rotation Graph (RRG)\nIndustry Positioning and Momentum Analysis - {date_str}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save chart
            filename = f"industry_rrg_{date_str}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Industry RRG saved: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create industry RRG: {e}")
            plt.close()
            return None

    def create_leadership_chart(self, stocks_analysis: Dict, date_str: str) -> str:
        """Create Leadership Strength Bar Chart."""
        try:
            logger.info("Creating leadership strength chart")

            # Get top leaders by trend consistency
            all_stocks = stocks_analysis['individual_analysis']
            top_leaders = sorted(all_stocks, key=lambda x: x['trend_consistency'], reverse=True)[:15]

            if not top_leaders:
                logger.warning("No stocks available for leadership chart")
                return None

            # Prepare data
            tickers = [stock['ticker'] for stock in top_leaders]
            leadership_strength = [stock['trend_consistency'] for stock in top_leaders]
            classifications = [stock['classification'] for stock in top_leaders]

            # Color mapping for classifications
            color_map = {
                'ELITE_LEADER': '#00FF00',
                'STRONG_PERFORMER': '#90EE90',
                'ABOVE_AVERAGE': '#FFFF00',
                'AVERAGE': '#FFA500',
                'WEAK_LAGGARD': '#FF6347'
            }
            colors = [color_map.get(cls, '#CCCCCC') for cls in classifications]

            # Create horizontal bar chart
            plt.figure(figsize=(12, 8))
            bars = plt.barh(tickers, leadership_strength, color=colors, alpha=0.8, edgecolor='black')

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, leadership_strength)):
                plt.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=9)

            plt.xlabel("Leadership Strength (%)")
            plt.ylabel("Stocks")
            plt.title(f"Stock Leadership Strength Across Multiple Timeframes\nTop 15 Most Consistent Leaders - {date_str}")
            plt.grid(True, alpha=0.3, axis='x')

            # Add legend
            unique_classifications = list(set(classifications))
            legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map.get(cls, '#CCCCCC'))
                             for cls in unique_classifications]
            plt.legend(legend_elements, unique_classifications, loc='lower right')

            plt.tight_layout()

            # Save chart
            filename = f"leadership_strength_{date_str}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Leadership chart saved: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create leadership chart: {e}")
            plt.close()
            return None

    def create_momentum_scatter(self, stocks_analysis: Dict, date_str: str) -> str:
        """Create Momentum Pattern Scatter Plot."""
        try:
            logger.info("Creating momentum scatter plot")

            stocks_data = stocks_analysis['individual_analysis']

            if not stocks_data:
                logger.warning("No stocks data available for momentum scatter")
                return None

            # Prepare data
            momentum_medium = [stock['momentum_medium'] for stock in stocks_data]
            momentum_long = [stock['momentum_long'] for stock in stocks_data]
            patterns = [stock['momentum_pattern'] for stock in stocks_data]
            composite_strength = [stock['composite_strength'] for stock in stocks_data]

            # Color mapping for patterns
            pattern_colors = [self.color_schemes['momentum_colors'].get(pattern, '#CCCCCC')
                            for pattern in patterns]

            # Size based on composite strength
            sizes = [max(20, strength * 50) for strength in composite_strength]

            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(momentum_medium, momentum_long, c=pattern_colors, s=sizes,
                                alpha=0.6, edgecolors='black', linewidth=0.5)

            # Add quadrant lines
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Add quadrant labels
            plt.text(5, 5, 'Accelerating\n(Both Positive)', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
            plt.text(-5, 5, 'Mixed\n(Med-, Long+)', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            plt.text(5, -5, 'Mixed\n(Med+, Long-)', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
            plt.text(-5, -5, 'Decelerating\n(Both Negative)', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

            plt.xlabel("Medium-term Momentum (14d vs 44d %)")
            plt.ylabel("Long-term Momentum (66d vs 252d %)")
            plt.title(f"Advanced Momentum Pattern Analysis\nBubble Size = Composite Strength - {date_str}")
            plt.grid(True, alpha=0.3)

            # Add legend for patterns
            unique_patterns = list(set(patterns))
            legend_elements = [plt.scatter([], [], c=self.color_schemes['momentum_colors'].get(pattern, '#CCCCCC'),
                                         s=100, label=pattern) for pattern in unique_patterns]
            plt.legend(handles=legend_elements, title='Momentum Patterns', loc='upper left')

            plt.tight_layout()

            # Save chart
            filename = f"momentum_patterns_{date_str}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Momentum scatter saved: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create momentum scatter: {e}")
            plt.close()
            return None

    def create_elite_radar(self, stocks_analysis: Dict, date_str: str) -> str:
        """Create Elite Stocks Radar Chart."""
        try:
            logger.info("Creating elite stocks radar chart")

            # Get top 6 elite performers
            all_stocks = stocks_analysis['individual_analysis']
            elite_stocks = [stock for stock in all_stocks
                          if stock['classification'] in ['ELITE_LEADER', 'STRONG_PERFORMER']]

            if len(elite_stocks) < 3:
                # Fallback to top performers by composite strength
                elite_stocks = sorted(all_stocks, key=lambda x: x['composite_strength'], reverse=True)[:6]

            if not elite_stocks:
                logger.warning("No stocks available for elite radar chart")
                return None

            elite_stocks = elite_stocks[:6]  # Limit to 6 for readability

            # Setup radar chart
            timeframes = ['3d', '5d', '7d', '14d', '22d', '44d', '66d', '132d', '252d']
            angles = np.linspace(0, 2 * np.pi, len(timeframes), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

            # Plot each elite stock
            colors = plt.cm.Set3(np.linspace(0, 1, len(elite_stocks)))

            for i, stock in enumerate(elite_stocks):
                # Get percentile values for this stock
                percentile_values = list(stock['percentile_values'].values())
                percentile_values += percentile_values[:1]  # Complete the circle

                ax.plot(angles, percentile_values, 'o-', linewidth=2, label=stock['ticker'],
                       color=colors[i], alpha=0.8)
                ax.fill(angles, percentile_values, alpha=0.1, color=colors[i])

            # Customize radar chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(timeframes)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20th', '40th', '60th', '80th', '100th'])
            ax.grid(True)

            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            plt.title(f"Elite Stocks Multi-Timeframe Percentile Performance\nTop 6 Elite Performers - {date_str}",
                     y=1.08, size=14)

            plt.tight_layout()

            # Save chart
            filename = f"elite_radar_{date_str}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Elite radar chart saved: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create elite radar chart: {e}")
            plt.close()
            return None

    def _add_rrg_quadrant_labels(self):
        """Add quadrant labels to RRG charts."""
        # Get current axis limits for positioning
        xlim = plt.xlim()
        ylim = plt.ylim()

        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.15
        y_pos_high = ylim[1] - (ylim[1] - ylim[0]) * 0.15
        y_pos_low = ylim[0] + (ylim[1] - ylim[0]) * 0.15

        x_pos_left = xlim[0] + (xlim[1] - xlim[0]) * 0.15

        # Quadrant labels
        plt.text(x_pos, y_pos_high, 'Leading\n(Improving)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7), fontsize=10)
        plt.text(x_pos, y_pos_low, 'Leading\n(Weakening)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7), fontsize=10)
        plt.text(x_pos_left, y_pos_high, 'Lagging\n(Improving)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7), fontsize=10)
        plt.text(x_pos_left, y_pos_low, 'Lagging\n(Weakening)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7), fontsize=10)

    def _get_sector_color(self, sector_name: str) -> str:
        """Get color for sector based on name."""
        sector_keywords = {
            'Technology': ['tech', 'software', 'computer', 'internet', 'semiconductor'],
            'Healthcare': ['health', 'medical', 'pharma', 'biotech', 'drug'],
            'Finance': ['bank', 'financial', 'insurance', 'real estate'],
            'Energy': ['energy', 'oil', 'gas', 'petroleum'],
            'Consumer': ['consumer', 'retail', 'food', 'beverage'],
            'Industrial': ['industrial', 'manufacturing', 'aerospace', 'defense'],
            'Materials': ['materials', 'mining', 'chemical', 'metal'],
            'Utilities': ['utilities', 'electric', 'water', 'gas'],
            'Communication': ['telecom', 'communication', 'media']
        }

        sector_lower = sector_name.lower()
        for sector_type, keywords in sector_keywords.items():
            if any(keyword in sector_lower for keyword in keywords):
                return self.color_schemes['sector_colors'].get(sector_type, '#808080')

        return '#808080'  # Default gray

    def _get_industry_color(self, classification: str) -> str:
        """Get color for industry based on classification."""
        color_map = {
            'LEADING': '#00FF00',
            'OUTPERFORMING': '#90EE90',
            'NEUTRAL': '#FFFF00',
            'LAGGING': '#FF6347'
        }
        return color_map.get(classification, '#CCCCCC')

    def _get_short_name(self, name: str, max_length: int) -> str:
        """Get shortened name for display."""
        if len(name) <= max_length:
            return name

        # Common abbreviations
        abbreviations = {
            'technology': 'Tech',
            'information': 'Info',
            'services': 'Svc',
            'software': 'SW',
            'manufacturing': 'Mfg',
            'pharmaceuticals': 'Pharma',
            'biotechnology': 'Biotech',
            'telecommunications': 'Telecom',
            'semiconductors': 'Semis'
        }

        short_name = name.lower()
        for full, abbrev in abbreviations.items():
            short_name = short_name.replace(full, abbrev)

        if len(short_name) > max_length:
            short_name = short_name[:max_length-2] + '..'

        return short_name.title()

    def create_chart_summary(self, charts: Dict[str, str], date_str: str) -> str:
        """Create a summary document of all generated charts."""
        summary_content = f"""
# RS/PER Charts Summary - {date_str}

Generated charts for comprehensive multi-timeframe market analysis:

## Chart Files Generated:
"""

        for chart_name, filepath in charts.items():
            if filepath:
                filename = Path(filepath).name
                summary_content += f"- **{chart_name}**: {filename}\n"

        summary_content += f"""

## Chart Descriptions:

1. **heatmap_rs**: Multi-timeframe RS strength heatmap for top 20 stocks
2. **sector_rrg**: Sector relative rotation graph showing positioning and momentum
3. **industry_rrg**: Industry relative rotation graph for granular analysis
4. **leadership_strength**: Bar chart of most consistent leaders across timeframes
5. **momentum_patterns**: Scatter plot of momentum patterns with composite strength
6. **elite_radar**: Radar chart of elite performers' percentile rankings

## Usage:
These charts are designed for integration into PDF reports using the rs_per_template.
All charts are saved at 300 DPI resolution for professional presentation quality.
"""

        # Save summary
        summary_file = self.output_dir / f"charts_summary_{date_str}.md"
        with open(summary_file, 'w') as f:
            f.write(summary_content)

        return str(summary_file)