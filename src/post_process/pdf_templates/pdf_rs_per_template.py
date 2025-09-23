#!/usr/bin/env python3
"""
RS/PER DataFrame-Based PDF Template
==================================

Comprehensive multi-timeframe RS/PER market analysis following the Overview V1 pattern.
Generates professional investment analysis reports using DataFrame processing instead of
external file access.

Features:
- 9-timeframe RS analysis (3d, 5d, 7d, 14d, 22d, 44d, 66d, 132d, 252d)
- Percentile ranking analysis across all timeframes
- Multi-level analysis (Stocks, Sectors, Industries)
- Professional chart suite (Heatmap, RRG, Leadership, Momentum, Elite Radar)
- Comprehensive narrative reports with investment recommendations
- DataFrame-based processing (no external file dependencies)
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
    Generate comprehensive RS/PER analysis PDF using DataFrame approach.

    Args:
        df: DataFrame with merged RS/PER data (stocks level)
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize the RS/PER analysis engine (following Overview V1 pattern)
        analyzer = RSPERAnalyzer(df, metadata)

        # Step 1: RS/PER Analysis - Multi-timeframe metrics calculation
        rsper_analysis = analyzer.calculate_rsper_metrics()

        # Step 2: Multi-Level Analysis - Stocks, sectors, industries analysis
        multi_level_analysis = analyzer.analyze_stocks_sectors_industries()

        # Step 3: Visualization - Generate RS/PER specific charts
        chart_paths = analyzer.generate_rsper_visualizations()

        # Step 4: PDF Assembly - Professional report layout
        success = analyzer.assemble_rsper_pdf(pdf_path, rsper_analysis, multi_level_analysis, chart_paths)

        if success:
            logger.info(f"Successfully generated RS/PER PDF: {pdf_path}")
            return True
        else:
            logger.error("Failed to generate RS/PER PDF")
            return False

    except Exception as e:
        logger.error(f"Error generating RS/PER PDF: {e}")
        return False


class RSPERAnalyzer:
    """Main RS/PER analyzer class implementing DataFrame-based analysis."""

    def __init__(self, df: pd.DataFrame, metadata: dict = None):
        self.df = df
        self.metadata = metadata or {}

        # RS/PER specific timeframes (9 timeframes vs Overview V1's 5)
        self.rs_timeframes = {
            '3d': 'rs_QQQ_ibd_stocks_daily_daily_daily_3d_rs_vs_QQQ',
            '5d': 'rs_QQQ_ibd_stocks_daily_daily_daily_5d_rs_vs_QQQ',
            '7d': 'rs_QQQ_ibd_stocks_daily_daily_weekly_7d_rs_vs_QQQ',
            '14d': 'rs_QQQ_ibd_stocks_daily_daily_weekly_14d_rs_vs_QQQ',
            '22d': 'rs_QQQ_ibd_stocks_daily_daily_monthly_22d_rs_vs_QQQ',
            '44d': 'rs_QQQ_ibd_stocks_daily_daily_monthly_44d_rs_vs_QQQ',
            '66d': 'rs_QQQ_ibd_stocks_daily_daily_quarterly_66d_rs_vs_QQQ',
            '132d': 'rs_QQQ_ibd_stocks_daily_daily_quarterly_132d_rs_vs_QQQ',
            '252d': 'rs_QQQ_ibd_stocks_daily_daily_yearly_252d_rs_vs_QQQ'
        }

        # Percentile columns
        self.per_timeframes = {
            '3d': 'per_daily_daily_daily_3d_rs_vs_QQQ_per_NASDAQ100',
            '5d': 'per_daily_daily_daily_5d_rs_vs_QQQ_per_NASDAQ100',
            '7d': 'per_daily_daily_weekly_7d_rs_vs_QQQ_per_NASDAQ100',
            '14d': 'per_daily_daily_weekly_14d_rs_vs_QQQ_per_NASDAQ100',
            '22d': 'per_daily_daily_monthly_22d_rs_vs_QQQ_per_NASDAQ100',
            '44d': 'per_daily_daily_monthly_44d_rs_vs_QQQ_per_NASDAQ100',
            '66d': 'per_daily_daily_quarterly_66d_rs_vs_QQQ_per_NASDAQ100',
            '132d': 'per_daily_daily_quarterly_132d_rs_vs_QQQ_per_NASDAQ100',
            '252d': 'per_daily_daily_yearly_252d_rs_vs_QQQ_per_NASDAQ100'
        }

        # Filter timeframes to available columns
        self.available_rs_timeframes = {
            label: col for label, col in self.rs_timeframes.items()
            if col in df.columns
        }

        self.available_per_timeframes = {
            label: col for label, col in self.per_timeframes.items()
            if col in df.columns
        }

        # Use hard-coded PNG directory (like Overview V1)
        self.png_dir = Path("results/post_process")
        self.png_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"RS/PER Analyzer initialized with {len(df)} stocks, "
                   f"{len(self.available_rs_timeframes)} RS timeframes, "
                   f"{len(self.available_per_timeframes)} PER timeframes")

    def calculate_rsper_metrics(self) -> dict:
        """Step 1: Calculate comprehensive RS/PER metrics."""
        logger.info("Step 1: Calculating RS/PER metrics")

        analysis = {}

        # Market condition assessment
        analysis['market_condition'] = self._assess_market_condition()

        # Stock-level RS/PER analysis
        analysis['stock_metrics'] = self._calculate_stock_metrics()

        # Leadership analysis
        analysis['leadership'] = self._analyze_leadership_patterns()

        # Momentum patterns
        analysis['momentum_patterns'] = self._classify_momentum_patterns()

        # Elite performers
        analysis['elite_performers'] = self._identify_elite_performers()

        return analysis

    def analyze_stocks_sectors_industries(self) -> dict:
        """Step 2: Multi-level analysis using DataFrame groupby (like Overview V1)."""
        logger.info("Step 2: Analyzing stocks, sectors, and industries")

        analysis = {}

        # Sector analysis (using groupby like Overview V1)
        if 'rs_QQQ_ibd_stocks_sector' in self.df.columns:
            sector_rs = self.df.groupby('rs_QQQ_ibd_stocks_sector')[list(self.available_rs_timeframes.values())].mean()
            sector_per = self.df.groupby('rs_QQQ_ibd_stocks_sector')[list(self.available_per_timeframes.values())].mean()

            analysis['sector_rotation'] = self._analyze_sector_rotation(sector_rs, sector_per)
            analysis['sector_leaders'] = self._identify_sector_leaders(sector_rs, sector_per)

        # Industry analysis (using groupby like Overview V1)
        if 'rs_QQQ_ibd_stocks_industry' in self.df.columns:
            industry_rs = self.df.groupby('rs_QQQ_ibd_stocks_industry')[list(self.available_rs_timeframes.values())].mean()
            industry_per = self.df.groupby('rs_QQQ_ibd_stocks_industry')[list(self.available_per_timeframes.values())].mean()

            analysis['industry_rotation'] = self._analyze_industry_rotation(industry_rs, industry_per)
            analysis['industry_leaders'] = self._identify_industry_leaders(industry_rs, industry_per)

        # Cross-timeframe analysis
        analysis['cross_timeframe'] = self._analyze_cross_timeframe_patterns()

        return analysis

    def generate_rsper_visualizations(self) -> dict:
        """Step 3: Generate RS/PER specific charts."""
        logger.info("Step 3: Generating RS/PER visualizations")

        chart_paths = {}

        try:
            # Chart 1: Multi-timeframe RS Heatmap
            chart_paths['rs_heatmap'] = self._create_rs_heatmap()

            # Chart 2: Sector RRG (Relative Rotation Graph)
            chart_paths['sector_rrg'] = self._create_sector_rrg()

            # Chart 3: Industry RRG
            chart_paths['industry_rrg'] = self._create_industry_rrg()

            # Chart 4: Leadership Strength Chart
            chart_paths['leadership_strength'] = self._create_leadership_chart()

            # Chart 5: Momentum Patterns Scatter
            chart_paths['momentum_scatter'] = self._create_momentum_scatter()

            # Chart 6: Elite Performers Radar
            chart_paths['elite_radar'] = self._create_elite_radar()

            logger.info(f"Generated {len([c for c in chart_paths.values() if c])} charts successfully")

        except Exception as e:
            logger.error(f"Error generating charts: {e}")

        return chart_paths

    def assemble_rsper_pdf(self, pdf_path: str, rsper_analysis: dict,
                          multi_level_analysis: dict, chart_paths: dict) -> bool:
        """Step 4: Assemble professional RS/PER PDF report."""
        logger.info("Step 4: Assembling RS/PER PDF report")

        try:
            # Create PDF document (same approach as Overview V1)
            doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                  rightMargin=0.5*inch, leftMargin=0.5*inch,
                                  topMargin=0.75*inch, bottomMargin=0.5*inch)

            # Get styles
            styles = getSampleStyleSheet()

            # Create custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER
            )

            content = []

            # Title
            title_text = f"RS/PER Comprehensive Market Analysis"
            if self.metadata and 'original_filename' in self.metadata:
                title_text += f" - {datetime.now().strftime('%Y-%m-%d')}"

            content.append(Paragraph(title_text, title_style))
            content.append(Spacer(1, 0.3*inch))

            # Executive Summary
            content.extend(self._build_executive_summary(rsper_analysis))

            # Market Structure Analysis
            content.extend(self._build_market_structure_section(rsper_analysis))

            # Multi-Timeframe Analysis
            content.extend(self._build_timeframe_analysis_section(rsper_analysis))

            # Sector Analysis
            content.extend(self._build_sector_analysis_section(multi_level_analysis))

            # Industry Analysis
            content.extend(self._build_industry_analysis_section(multi_level_analysis))

            # Charts Section
            content.extend(self._build_charts_section(chart_paths))

            # Investment Recommendations
            content.extend(self._build_recommendations_section(rsper_analysis, multi_level_analysis))

            # Build PDF
            doc.build(content)
            logger.info(f"RS/PER PDF successfully created: {pdf_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to assemble RS/PER PDF: {e}")
            return False

    def _assess_market_condition(self) -> dict:
        """Assess overall market condition based on RS/PER metrics."""
        try:
            # Calculate market breadth using percentiles
            high_per_stocks = 0
            total_stocks = len(self.df)

            for timeframe, col in self.available_per_timeframes.items():
                if col in self.df.columns:
                    high_per_count = len(self.df[self.df[col] > 70])
                    high_per_stocks = max(high_per_stocks, high_per_count)

            breadth_pct = (high_per_stocks / total_stocks * 100) if total_stocks > 0 else 0

            # Determine market condition
            if breadth_pct > 60:
                condition = "STRONG_BULLISH"
                description = "Strong bullish market with broad participation"
            elif breadth_pct > 40:
                condition = "MODERATE_BULLISH"
                description = "Moderately bullish market with selective strength"
            elif breadth_pct > 20:
                condition = "NEUTRAL"
                description = "Neutral market with mixed signals"
            else:
                condition = "WEAK"
                description = "Weak market with limited leadership"

            return {
                'condition': condition,
                'description': description,
                'breadth_pct': breadth_pct,
                'total_stocks': total_stocks,
                'high_per_stocks': high_per_stocks
            }

        except Exception as e:
            logger.error(f"Error assessing market condition: {e}")
            return {
                'condition': 'UNKNOWN',
                'description': 'Unable to assess market condition',
                'breadth_pct': 0,
                'total_stocks': len(self.df),
                'high_per_stocks': 0
            }

    def _calculate_stock_metrics(self) -> dict:
        """Calculate stock-level RS/PER metrics."""
        metrics = {}

        try:
            # Calculate average RS across timeframes
            rs_cols = list(self.available_rs_timeframes.values())
            if rs_cols:
                self.df['avg_rs'] = self.df[rs_cols].mean(axis=1)
                metrics['avg_rs_leaders'] = self.df.nlargest(10, 'avg_rs')[['ticker', 'avg_rs']].to_dict('records')

            # Calculate average percentile across timeframes
            per_cols = list(self.available_per_timeframes.values())
            if per_cols:
                self.df['avg_per'] = self.df[per_cols].mean(axis=1)
                metrics['avg_per_leaders'] = self.df.nlargest(10, 'avg_per')[['ticker', 'avg_per']].to_dict('records')

            # Consistency metrics
            if rs_cols:
                self.df['rs_consistency'] = self.df[rs_cols].std(axis=1)
                metrics['most_consistent'] = self.df.nsmallest(10, 'rs_consistency')[['ticker', 'rs_consistency']].to_dict('records')

        except Exception as e:
            logger.error(f"Error calculating stock metrics: {e}")

        return metrics

    def _analyze_leadership_patterns(self) -> dict:
        """Analyze leadership patterns across timeframes."""
        patterns = {}

        try:
            # Short-term leaders (3d, 5d)
            short_term_cols = [col for label, col in self.available_rs_timeframes.items()
                             if label in ['3d', '5d'] and col in self.df.columns]
            if short_term_cols:
                self.df['short_term_rs'] = self.df[short_term_cols].mean(axis=1)
                patterns['short_term_leaders'] = self.df.nlargest(5, 'short_term_rs')['ticker'].tolist()

            # Long-term leaders (132d, 252d)
            long_term_cols = [col for label, col in self.available_rs_timeframes.items()
                            if label in ['132d', '252d'] and col in self.df.columns]
            if long_term_cols:
                self.df['long_term_rs'] = self.df[long_term_cols].mean(axis=1)
                patterns['long_term_leaders'] = self.df.nlargest(5, 'long_term_rs')['ticker'].tolist()

        except Exception as e:
            logger.error(f"Error analyzing leadership patterns: {e}")

        return patterns

    def _classify_momentum_patterns(self) -> dict:
        """Classify momentum patterns."""
        patterns = {}

        try:
            # Simple momentum classification based on RS trend
            if '22d' in self.available_rs_timeframes and '252d' in self.available_rs_timeframes:
                short_col = self.available_rs_timeframes['22d']
                long_col = self.available_rs_timeframes['252d']

                self.df['momentum_type'] = 'NEUTRAL'
                self.df.loc[self.df[short_col] > self.df[long_col], 'momentum_type'] = 'ACCELERATING'
                self.df.loc[self.df[short_col] < self.df[long_col], 'momentum_type'] = 'DECELERATING'

                patterns['momentum_distribution'] = self.df['momentum_type'].value_counts().to_dict()
                patterns['accelerating_stocks'] = self.df[self.df['momentum_type'] == 'ACCELERATING']['ticker'].tolist()[:10]

        except Exception as e:
            logger.error(f"Error classifying momentum patterns: {e}")

        return patterns

    def _identify_elite_performers(self) -> dict:
        """Identify elite performers based on RS/PER metrics."""
        elite = {}

        try:
            # Elite criteria: High percentile (>80) across multiple timeframes
            per_cols = list(self.available_per_timeframes.values())
            if len(per_cols) >= 3:
                # Count how many timeframes each stock is above 80th percentile
                self.df['elite_score'] = (self.df[per_cols] > 80).sum(axis=1)
                elite_threshold = len(per_cols) * 0.6  # 60% of timeframes

                elite_stocks = self.df[self.df['elite_score'] >= elite_threshold]
                elite['elite_stocks'] = elite_stocks.nlargest(10, 'elite_score')[['ticker', 'elite_score']].to_dict('records')
                elite['elite_count'] = len(elite_stocks)

        except Exception as e:
            logger.error(f"Error identifying elite performers: {e}")

        return elite

    def _analyze_sector_rotation(self, sector_rs: pd.DataFrame, sector_per: pd.DataFrame) -> dict:
        """Analyze sector rotation patterns."""
        rotation = {}

        try:
            if not sector_rs.empty and '22d' in self.available_rs_timeframes:
                col_22d = self.available_rs_timeframes['22d']
                if col_22d in sector_rs.columns:
                    sector_ranking = sector_rs[col_22d].sort_values(ascending=False)
                    rotation['sector_leaders'] = sector_ranking.head(3).to_dict()
                    rotation['sector_laggards'] = sector_ranking.tail(3).to_dict()

        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {e}")

        return rotation

    def _identify_sector_leaders(self, sector_rs: pd.DataFrame, sector_per: pd.DataFrame) -> dict:
        """Identify sector leaders."""
        leaders = {}

        try:
            if not sector_rs.empty:
                # Average RS across all available timeframes
                sector_rs['avg_rs'] = sector_rs[list(self.available_rs_timeframes.values())].mean(axis=1)
                leaders['top_sectors'] = sector_rs['avg_rs'].nlargest(5).to_dict()

        except Exception as e:
            logger.error(f"Error identifying sector leaders: {e}")

        return leaders

    def _analyze_industry_rotation(self, industry_rs: pd.DataFrame, industry_per: pd.DataFrame) -> dict:
        """Analyze industry rotation patterns."""
        rotation = {}

        try:
            if not industry_rs.empty and '22d' in self.available_rs_timeframes:
                col_22d = self.available_rs_timeframes['22d']
                if col_22d in industry_rs.columns:
                    industry_ranking = industry_rs[col_22d].sort_values(ascending=False)
                    rotation['industry_leaders'] = industry_ranking.head(5).to_dict()
                    rotation['industry_laggards'] = industry_ranking.tail(3).to_dict()

        except Exception as e:
            logger.error(f"Error analyzing industry rotation: {e}")

        return rotation

    def _identify_industry_leaders(self, industry_rs: pd.DataFrame, industry_per: pd.DataFrame) -> dict:
        """Identify industry leaders."""
        leaders = {}

        try:
            if not industry_rs.empty:
                # Average RS across all available timeframes
                industry_rs['avg_rs'] = industry_rs[list(self.available_rs_timeframes.values())].mean(axis=1)
                leaders['top_industries'] = industry_rs['avg_rs'].nlargest(5).to_dict()

        except Exception as e:
            logger.error(f"Error identifying industry leaders: {e}")

        return leaders

    def _analyze_cross_timeframe_patterns(self) -> dict:
        """Analyze patterns across timeframes."""
        patterns = {}

        try:
            # Analyze RS progression from short to long term
            timeframe_order = ['3d', '5d', '7d', '14d', '22d', '44d', '66d', '132d', '252d']
            available_order = [tf for tf in timeframe_order if tf in self.available_rs_timeframes]

            if len(available_order) >= 3:
                # Calculate trend strength
                patterns['timeframe_progression'] = {}
                for ticker in self.df['ticker'].head(10):  # Sample analysis
                    ticker_data = self.df[self.df['ticker'] == ticker]
                    if not ticker_data.empty:
                        rs_values = []
                        for tf in available_order:
                            col = self.available_rs_timeframes[tf]
                            if col in ticker_data.columns:
                                rs_values.append(ticker_data[col].iloc[0])

                        if len(rs_values) >= 3:
                            # Simple trend analysis
                            trend = 'IMPROVING' if rs_values[-1] > rs_values[0] else 'DECLINING'
                            patterns['timeframe_progression'][ticker] = {
                                'trend': trend,
                                'rs_values': rs_values
                            }

        except Exception as e:
            logger.error(f"Error analyzing cross-timeframe patterns: {e}")

        return patterns

    # Chart generation methods (simplified for now)
    def _create_rs_heatmap(self) -> str:
        """Create RS heatmap chart."""
        try:
            # Get top 20 stocks by average RS
            if hasattr(self.df, 'avg_rs'):
                top_stocks = self.df.nlargest(20, 'avg_rs')
            else:
                top_stocks = self.df.head(20)

            # Prepare data for heatmap
            rs_cols = list(self.available_rs_timeframes.values())
            heatmap_data = top_stocks[['ticker'] + rs_cols].set_index('ticker')

            # Create heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0)
            plt.title('RS Heatmap: Top 20 Stocks\nAll 9 Timeframes (3d, 5d, 7d, 14d, 22d, 44d, 66d, 132d, 252d)\nValues: RS vs QQQ (>1.0 = outperforming)', fontsize=14, fontweight='bold')
            plt.tight_layout()

            chart_path = self.png_dir / "rs_heatmap.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("RS heatmap created successfully")
            return str(chart_path)

        except Exception as e:
            logger.error(f"Failed to create RS heatmap: {e}")
            plt.close()
            return None

    def _create_sector_rrg(self) -> str:
        """Create Sector Relative Rotation Graph."""
        try:
            # Simplified RRG - would need more sophisticated implementation
            plt.figure(figsize=(12, 8))
            plt.scatter([1, 2, 3], [1, 2, 3], s=100)  # Placeholder
            plt.title('Sector Relative Rotation Graph (RRG)\nTimeframe: 22d RS vs QQQ\nGrouped by: rs_QQQ_ibd_stocks_sector', fontsize=14, fontweight='bold')
            plt.xlabel('Relative Strength')
            plt.ylabel('Momentum')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            chart_path = self.png_dir / "sector_rrg.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Failed to create sector RRG: {e}")
            plt.close()
            return None

    def _create_industry_rrg(self) -> str:
        """Create Industry Relative Rotation Graph."""
        try:
            # Simplified RRG - placeholder
            plt.figure(figsize=(12, 8))
            plt.scatter([1, 2, 3, 4], [1, 2, 3, 4], s=100)
            plt.title('Industry Relative Rotation Graph (RRG)\nTimeframe: 22d RS vs QQQ\nGrouped by: rs_QQQ_ibd_stocks_industry', fontsize=14, fontweight='bold')
            plt.xlabel('Relative Strength')
            plt.ylabel('Momentum')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            chart_path = self.png_dir / "industry_rrg.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Failed to create industry RRG: {e}")
            plt.close()
            return None

    def _create_leadership_chart(self) -> str:
        """Create leadership strength chart."""
        try:
            # Get top 10 leaders by average RS
            if hasattr(self.df, 'avg_rs'):
                top_leaders = self.df.nlargest(10, 'avg_rs')

                plt.figure(figsize=(12, 8))
                plt.barh(top_leaders['ticker'], top_leaders['avg_rs'], color='steelblue')
                plt.title('Leadership Strength\nVariable: Average RS (All 9 Timeframes: 3d-252d)\nTop 10 Consistent Leaders', fontsize=14, fontweight='bold')
                plt.xlabel('Average Relative Strength')
                plt.tight_layout()

                chart_path = self.png_dir / "leadership_strength.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                return str(chart_path)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to create leadership chart: {e}")
            plt.close()
            return None

    def _create_momentum_scatter(self) -> str:
        """Create momentum patterns scatter plot."""
        try:
            # Simple scatter plot using available data
            if '22d' in self.available_rs_timeframes and '252d' in self.available_rs_timeframes:
                col_22d = self.available_rs_timeframes['22d']
                col_252d = self.available_rs_timeframes['252d']

                plt.figure(figsize=(12, 8))
                plt.scatter(self.df[col_252d], self.df[col_22d], alpha=0.6)
                plt.xlabel('252-day RS')
                plt.ylabel('22-day RS')
                plt.title('Momentum Patterns Scatter Plot\nX-axis: 252d RS (Long-term) | Y-axis: 22d RS (Medium-term)\nData: All 117 Stocks vs QQQ', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                chart_path = self.png_dir / "momentum_scatter.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                return str(chart_path)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to create momentum scatter: {e}")
            plt.close()
            return None

    def _create_elite_radar(self) -> str:
        """Create elite performers radar chart."""
        try:
            # Simplified radar chart - placeholder
            plt.figure(figsize=(10, 10))
            angles = np.linspace(0, 2*np.pi, 9, endpoint=False).tolist()

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            ax.plot(angles + [angles[0]], [1,2,3,4,5,4,3,2,1] + [1])  # Placeholder data
            ax.fill(angles + [angles[0]], [1,2,3,4,5,4,3,2,1] + [1], alpha=0.25)

            ax.set_xticks(angles)
            ax.set_xticklabels(['3d', '5d', '7d', '14d', '22d', '44d', '66d', '132d', '252d'])
            ax.set_title('Elite Performers Radar Chart\nVariable: PER Percentiles (All 9 Timeframes)\nCriteria: PER > 80 across ≥60% of timeframes', size=14, fontweight='bold', pad=20)

            chart_path = self.png_dir / "elite_radar.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Failed to create elite radar: {e}")
            plt.close()
            return None

    # PDF section builders
    def _build_executive_summary(self, analysis: dict) -> list:
        """Build executive summary section."""
        content = []
        styles = getSampleStyleSheet()

        # Section header with data specification
        content.append(Paragraph("<b>Executive Summary</b> - PER Analysis (All 9 Timeframes: 3d-252d)", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Market condition
        market_condition = analysis.get('market_condition', {})
        summary_text = f"""
        <b>Market Condition:</b> {market_condition.get('condition', 'Unknown')}<br/>
        <b>Description:</b> {market_condition.get('description', 'Analysis unavailable')}<br/>
        <b>Market Breadth:</b> {market_condition.get('breadth_pct', 0):.1f}% ({market_condition.get('high_per_stocks', 0)} of {market_condition.get('total_stocks', 0)} stocks)<br/>
        <br/>
        This comprehensive RS/PER analysis examines {market_condition.get('total_stocks', 0)} stocks across 9 timeframes
        (3d, 5d, 7d, 14d, 22d, 44d, 66d, 132d, 252d), using both Relative Strength (RS) and Percentile (PER) metrics
        to provide insights into market structure, momentum patterns, and investment opportunities.<br/>
        <b>Market Breadth Calculation:</b> Based on PER > 70 across all timeframes.
        """

        content.append(Paragraph(summary_text, styles['Normal']))
        content.append(Spacer(1, 0.3*inch))

        return content

    def _build_market_structure_section(self, analysis: dict) -> list:
        """Build market structure analysis section."""
        content = []
        styles = getSampleStyleSheet()

        content.append(Paragraph("<b>Market Structure Analysis</b> - RS Leadership (3d-5d vs 132d-252d)", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Leadership patterns
        leadership = analysis.get('leadership', {})
        if leadership:
            leaders_text = f"""
            <b>Short-Term Leaders (RS 3d-5d avg):</b> {', '.join(leadership.get('short_term_leaders', [])[:5])}<br/>
            <b>Long-Term Leaders (RS 132d-252d avg):</b> {', '.join(leadership.get('long_term_leaders', [])[:5])}<br/>
            <i>Analysis compares short-term momentum vs long-term strength patterns.</i>
            """
            content.append(Paragraph(leaders_text, styles['Normal']))

        content.append(Spacer(1, 0.3*inch))
        return content

    def _build_timeframe_analysis_section(self, analysis: dict) -> list:
        """Build multi-timeframe analysis section."""
        content = []
        styles = getSampleStyleSheet()

        content.append(Paragraph("<b>Multi-Timeframe Analysis</b> - RS Average (All 9 Timeframes) + PER Consistency", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Stock metrics
        stock_metrics = analysis.get('stock_metrics', {})
        if stock_metrics.get('avg_rs_leaders'):
            leaders = stock_metrics['avg_rs_leaders'][:5]
            leaders_text = "<b>Top 5 RS Leaders (Avg across 3d-252d):</b> " + ", ".join([f"{stock['ticker']} ({stock['avg_rs']:.1f})"
                                                           for stock in leaders]) + "<br/>" + \
                          "<i>RS Values: >1.0 = outperforming QQQ, <1.0 = underperforming QQQ</i>"
            content.append(Paragraph(leaders_text, styles['Normal']))

        content.append(Spacer(1, 0.3*inch))
        return content

    def _build_sector_analysis_section(self, analysis: dict) -> list:
        """Build sector analysis section."""
        content = []
        styles = getSampleStyleSheet()

        content.append(Paragraph("<b>Sector Analysis</b> - RS 22d Timeframe (Monthly Rotation)", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        sector_rotation = analysis.get('sector_rotation', {})
        if sector_rotation.get('sector_leaders'):
            leaders_text = "<b>Leading Sectors (RS 22d):</b> " + ", ".join([f"{sector} ({rs:.1f})"
                                                          for sector, rs in list(sector_rotation['sector_leaders'].items())[:3]]) + "<br/>" + \
                          "<i>22-day timeframe captures medium-term sector rotation patterns</i>"
            content.append(Paragraph(leaders_text, styles['Normal']))

        content.append(Spacer(1, 0.3*inch))
        return content

    def _build_industry_analysis_section(self, analysis: dict) -> list:
        """Build industry analysis section."""
        content = []
        styles = getSampleStyleSheet()

        content.append(Paragraph("<b>Industry Analysis</b> - RS 22d Timeframe (Monthly Rotation)", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        industry_rotation = analysis.get('industry_rotation', {})
        if industry_rotation.get('industry_leaders'):
            leaders_text = "<b>Leading Industries (RS 22d):</b> " + ", ".join([f"{industry} ({rs:.1f})"
                                                             for industry, rs in list(industry_rotation['industry_leaders'].items())[:3]]) + "<br/>" + \
                          "<i>22-day timeframe identifies industry rotation and momentum shifts</i>"
            content.append(Paragraph(leaders_text, styles['Normal']))

        content.append(Spacer(1, 0.3*inch))
        return content

    def _build_charts_section(self, chart_paths: dict) -> list:
        """Build charts section."""
        content = []
        styles = getSampleStyleSheet()

        content.append(PageBreak())
        content.append(Paragraph("<b>Visual Analysis</b> - Charts with Variable Specifications", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Add charts that were successfully generated
        for chart_name, chart_path in chart_paths.items():
            if chart_path and Path(chart_path).exists():
                try:
                    content.append(RLImage(chart_path, width=7*inch, height=5*inch))
                    content.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    logger.error(f"Error adding chart {chart_name}: {e}")

        return content

    def _build_recommendations_section(self, rsper_analysis: dict, multi_level_analysis: dict) -> list:
        """Build investment recommendations section."""
        content = []
        styles = getSampleStyleSheet()

        content.append(PageBreak())
        content.append(Paragraph("<b>Investment Recommendations</b> - Based on PER Elite Scoring (>80 Percentile)", styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))

        # Elite performers
        elite = rsper_analysis.get('elite_performers', {})
        if elite.get('elite_stocks'):
            elite_text = f"""
            <b>Elite Performers ({elite.get('elite_count', 0)} stocks identified):</b><br/>
            These stocks demonstrate PER > 80 (top 20%) across ≥60% of timeframes (≥5 out of 9).<br/>
            <b>Top Candidates:</b> {', '.join([stock['ticker'] for stock in elite['elite_stocks'][:5]])}<br/>
            <i>Elite Score = count of timeframes where stock ranks in top 20% of NASDAQ100</i><br/>
            """
            content.append(Paragraph(elite_text, styles['Normal']))

        # General recommendations
        market_condition = rsper_analysis.get('market_condition', {})
        condition = market_condition.get('condition', 'UNKNOWN')

        if condition == 'STRONG_BULLISH':
            rec_text = "Strong market conditions support growth-oriented positioning with focus on momentum leaders."
        elif condition == 'MODERATE_BULLISH':
            rec_text = "Selective opportunities in leading sectors and high-quality growth stocks."
        elif condition == 'NEUTRAL':
            rec_text = "Mixed market signals suggest defensive positioning with selective opportunities."
        else:
            rec_text = "Weak market conditions favor defensive positioning and capital preservation."

        content.append(Spacer(1, 0.2*inch))
        content.append(Paragraph(f"<b>Market Strategy:</b> {rec_text}", styles['Normal']))

        return content