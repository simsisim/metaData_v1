"""
RS/PER PDF Assembly Engine
==========================

Creates professional PDF reports for comprehensive multi-timeframe RS/PER analysis.
Uses ReportLab for high-quality PDF generation with professional styling.

Report Structure:
1. Cover Page & Executive Summary
2. Market Structure Assessment
3. Multi-Timeframe RS Analysis (with heatmap)
4. Sector Rotation Analysis (with sector RRG)
5. Industry Rotation Analysis (with industry RRG)
6. Momentum Pattern Analysis (with scatter plot)
7. Leadership Analysis (with bar chart)
8. Elite Performance Analysis (with radar chart)
9. Trading Strategies & Investment Recommendations
"""

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, Spacer, PageBreak
from reportlab.platypus import TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.frames import Frame

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RSPERPDFBuilder:
    """
    Professional PDF builder for RS/PER comprehensive analysis reports.
    """

    def __init__(self, config=None, output_dir: str = None):
        """Initialize the PDF builder."""
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('reports')
        self.output_dir.mkdir(exist_ok=True)

        # Setup styles
        self.styles = self._setup_styles()

        # Page settings
        self.pagesize = letter
        self.margin = 0.75 * inch

    def _setup_styles(self) -> Dict:
        """Setup custom paragraph styles for professional formatting."""
        styles = getSampleStyleSheet()

        # Custom styles
        custom_styles = {
            'CustomTitle': ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                textColor=colors.darkblue,
                spaceAfter=30,
                alignment=TA_CENTER
            ),
            'SectionHeader': ParagraphStyle(
                'SectionHeader',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.darkblue,
                spaceBefore=20,
                spaceAfter=15,
                borderWidth=2,
                borderColor=colors.darkblue,
                borderPadding=10
            ),
            'SubHeader': ParagraphStyle(
                'SubHeader',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.darkgreen,
                spaceBefore=15,
                spaceAfter=10
            ),
            'Body': ParagraphStyle(
                'Body',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=12,
                alignment=TA_JUSTIFY
            ),
            'BulletPoint': ParagraphStyle(
                'BulletPoint',
                parent=styles['Normal'],
                fontSize=10,
                leftIndent=20,
                spaceAfter=6
            ),
            'KeyInsight': ParagraphStyle(
                'KeyInsight',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.darkred,
                leftIndent=15,
                spaceAfter=8,
                borderWidth=1,
                borderColor=colors.lightgrey,
                borderPadding=8,
                backColor=colors.lightgrey
            ),
            'ChartCaption': ParagraphStyle(
                'ChartCaption',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=TA_CENTER,
                spaceAfter=15
            ),
            'Footer': ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
        }

        # Create dictionary with both default and custom styles
        # Use a safer approach to merge styles
        all_styles = {}

        # Add default styles by name
        for style_name in ['Normal', 'Title', 'Heading1', 'Heading2', 'Heading3', 'Bullet', 'Code']:
            if style_name in styles:
                all_styles[style_name] = styles[style_name]

        # Add custom styles
        all_styles.update(custom_styles)

        return all_styles

    def build_pdf(self, report_sections: Dict, charts: Dict[str, str],
                  analysis_results: Dict, date_str: str = None) -> str:
        """
        Build complete PDF report.

        Args:
            report_sections: Generated report sections from RSPERReportGenerator
            charts: Chart file paths from RSPERChartGenerator
            analysis_results: Analysis results for data tables
            date_str: Date string for filename

        Returns:
            Path to generated PDF file
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')

        logger.info(f"Building RS/PER PDF report for {date_str}")

        try:
            # Generate filename
            filename = f"comprehensive_rs_per_market_analysis_{date_str}.pdf"
            filepath = self.output_dir / filename

            # Create PDF document
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=self.pagesize,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin
            )

            # Build story (content elements)
            story = []

            # 1. Cover Page & Executive Summary
            story.extend(self._create_cover_page(report_sections, date_str))
            story.append(PageBreak())

            # 2. Market Structure Assessment
            story.extend(self._create_market_structure_section(report_sections, analysis_results))
            story.append(PageBreak())

            # 3. Multi-Timeframe RS Analysis
            story.extend(self._create_rs_analysis_section(
                report_sections, charts.get('heatmap_rs'), analysis_results
            ))
            story.append(PageBreak())

            # 4. Sector Rotation Analysis
            story.extend(self._create_sector_rotation_section(
                report_sections, charts.get('sector_rrg'), analysis_results
            ))
            story.append(PageBreak())

            # 5. Industry Rotation Analysis
            story.extend(self._create_industry_rotation_section(
                report_sections, charts.get('industry_rrg'), analysis_results
            ))
            story.append(PageBreak())

            # 6. Momentum Pattern Analysis
            story.extend(self._create_momentum_analysis_section(
                report_sections, charts.get('momentum_patterns'), analysis_results
            ))
            story.append(PageBreak())

            # 7. Leadership Analysis
            story.extend(self._create_leadership_section(
                report_sections, charts.get('leadership_strength'), analysis_results
            ))
            story.append(PageBreak())

            # 8. Elite Performance Analysis
            story.extend(self._create_elite_analysis_section(
                report_sections, charts.get('elite_radar'), analysis_results
            ))
            story.append(PageBreak())

            # 9. Trading Strategies & Investment Recommendations
            story.extend(self._create_strategies_section(report_sections, analysis_results))

            # Build PDF
            doc.build(story)

            logger.info(f"PDF report successfully created: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"PDF building failed: {e}")
            raise

    def _create_cover_page(self, report_sections: Dict, date_str: str) -> List:
        """Create cover page and executive summary."""
        elements = []

        # Title
        elements.append(Paragraph(
            "Comprehensive Multi-Timeframe Market Analysis Report",
            self.styles['CustomTitle']
        ))

        # Subtitle
        elements.append(Paragraph(
            f"RS/PER Analysis - {date_str}",
            self.styles['SubHeader']
        ))

        elements.append(Spacer(1, 0.5 * inch))

        # Executive Summary
        exec_summary = report_sections.get('executive_summary')
        if exec_summary:
            elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            elements.append(Paragraph(exec_summary.content, self.styles['Body']))

            # Key insights
            if exec_summary.key_insights:
                elements.append(Paragraph("Key Insights:", self.styles['SubHeader']))
                for insight in exec_summary.key_insights:
                    elements.append(Paragraph(f"• {insight}", self.styles['KeyInsight']))

        return elements

    def _create_market_structure_section(self, report_sections: Dict, analysis_results: Dict) -> List:
        """Create market structure assessment section."""
        elements = []

        elements.append(Paragraph("Market Structure Assessment", self.styles['SectionHeader']))

        market_structure = report_sections.get('market_structure')
        if market_structure:
            elements.append(Paragraph(market_structure.content, self.styles['Body']))

        # Add market condition summary table
        market_condition = analysis_results.get('market_condition')
        if market_condition:
            elements.append(self._create_market_condition_table(market_condition))

        return elements

    def _create_rs_analysis_section(self, report_sections: Dict, heatmap_path: str,
                                   analysis_results: Dict) -> List:
        """Create multi-timeframe RS analysis section."""
        elements = []

        elements.append(Paragraph("Multi-Timeframe Relative Strength Analysis", self.styles['SectionHeader']))

        # Add heatmap chart
        if heatmap_path and Path(heatmap_path).exists():
            elements.append(self._create_chart_with_caption(
                heatmap_path,
                "Multi-Timeframe RS Heatmap: Top 20 stocks ranked by composite strength across 9 timeframes",
                width=6*inch, height=4*inch
            ))

        # RS analysis content
        rs_analysis = report_sections.get('rs_analysis')
        if rs_analysis:
            elements.append(Paragraph(rs_analysis.content, self.styles['Body']))

        # Top performers table
        stocks_analysis = analysis_results.get('stocks_analysis', {})
        top_performers = stocks_analysis.get('top_performers', [])
        if top_performers:
            elements.append(self._create_top_performers_table(top_performers[:10]))

        return elements

    def _create_sector_rotation_section(self, report_sections: Dict, sector_rrg_path: str,
                                       analysis_results: Dict) -> List:
        """Create sector rotation analysis section."""
        elements = []

        elements.append(Paragraph("Sector Rotation Analysis", self.styles['SectionHeader']))

        # Add sector RRG chart
        if sector_rrg_path and Path(sector_rrg_path).exists():
            elements.append(self._create_chart_with_caption(
                sector_rrg_path,
                "Sector Relative Rotation Graph: Sector positioning and momentum analysis",
                width=5.5*inch, height=4.5*inch
            ))

        # Sector rotation content
        sector_rotation = report_sections.get('sector_rotation')
        if sector_rotation:
            elements.append(Paragraph(sector_rotation.content, self.styles['Body']))

        # Sector summary table
        sectors_analysis = analysis_results.get('sectors_analysis', {})
        if sectors_analysis:
            elements.append(self._create_sector_summary_table(sectors_analysis))

        return elements

    def _create_industry_rotation_section(self, report_sections: Dict, industry_rrg_path: str,
                                         analysis_results: Dict) -> List:
        """Create industry rotation analysis section."""
        elements = []

        elements.append(Paragraph("Industry Rotation Analysis", self.styles['SectionHeader']))

        # Add industry RRG chart
        if industry_rrg_path and Path(industry_rrg_path).exists():
            elements.append(self._create_chart_with_caption(
                industry_rrg_path,
                "Industry Relative Rotation Graph: Granular industry positioning and momentum analysis",
                width=6*inch, height=5*inch
            ))

        # Industry rotation content
        industry_rotation = report_sections.get('industry_rotation')
        if industry_rotation:
            elements.append(Paragraph(industry_rotation.content, self.styles['Body']))

        # Industry quadrant summary
        industries_analysis = analysis_results.get('industries_analysis', {})
        if industries_analysis:
            elements.append(self._create_industry_quadrant_table(industries_analysis))

        return elements

    def _create_momentum_analysis_section(self, report_sections: Dict, momentum_chart_path: str,
                                         analysis_results: Dict) -> List:
        """Create momentum pattern analysis section."""
        elements = []

        elements.append(Paragraph("Momentum Pattern Analysis", self.styles['SectionHeader']))

        # Add momentum scatter plot
        if momentum_chart_path and Path(momentum_chart_path).exists():
            elements.append(self._create_chart_with_caption(
                momentum_chart_path,
                "Momentum Pattern Scatter Plot: Medium-term vs Long-term momentum with composite strength sizing",
                width=5.5*inch, height=4.5*inch
            ))

        # Momentum analysis content
        momentum_analysis = report_sections.get('momentum_analysis')
        if momentum_analysis:
            elements.append(Paragraph(momentum_analysis.content, self.styles['Body']))

        return elements

    def _create_leadership_section(self, report_sections: Dict, leadership_chart_path: str,
                                  analysis_results: Dict) -> List:
        """Create leadership analysis section."""
        elements = []

        elements.append(Paragraph("Leadership Analysis", self.styles['SectionHeader']))

        # Add leadership chart
        if leadership_chart_path and Path(leadership_chart_path).exists():
            elements.append(self._create_chart_with_caption(
                leadership_chart_path,
                "Leadership Strength Chart: Most consistent leaders across multiple timeframes",
                width=5.5*inch, height=4*inch
            ))

        # Leadership analysis content
        leadership_analysis = report_sections.get('leadership_analysis')
        if leadership_analysis:
            elements.append(Paragraph(leadership_analysis.content, self.styles['Body']))

        return elements

    def _create_elite_analysis_section(self, report_sections: Dict, elite_radar_path: str,
                                      analysis_results: Dict) -> List:
        """Create elite performance analysis section."""
        elements = []

        elements.append(Paragraph("Elite Performance Analysis", self.styles['SectionHeader']))

        # Add elite radar chart
        if elite_radar_path and Path(elite_radar_path).exists():
            elements.append(self._create_chart_with_caption(
                elite_radar_path,
                "Elite Stocks Radar Chart: Multi-timeframe percentile performance of top performers",
                width=5.5*inch, height=5.5*inch
            ))

        # Elite analysis content
        elite_analysis = report_sections.get('elite_analysis')
        if elite_analysis:
            elements.append(Paragraph(elite_analysis.content, self.styles['Body']))

        return elements

    def _create_strategies_section(self, report_sections: Dict, analysis_results: Dict) -> List:
        """Create trading strategies and recommendations section."""
        elements = []

        elements.append(Paragraph("Trading Strategies & Investment Recommendations", self.styles['SectionHeader']))

        # Trading strategies content
        trading_strategies = report_sections.get('trading_strategies')
        if trading_strategies:
            elements.append(Paragraph(trading_strategies.content, self.styles['Body']))

        # Add risk management summary
        elements.append(self._create_risk_summary_table(analysis_results))

        return elements

    def _create_chart_with_caption(self, chart_path: str, caption: str,
                                   width: float = 6*inch, height: float = 4*inch) -> KeepTogether:
        """Create chart with caption, kept together."""
        elements = []

        try:
            # Add chart image
            img = Image(chart_path, width=width, height=height)
            elements.append(img)

            # Add caption
            elements.append(Paragraph(caption, self.styles['ChartCaption']))

        except Exception as e:
            logger.warning(f"Failed to add chart {chart_path}: {e}")
            elements.append(Paragraph(f"[Chart unavailable: {Path(chart_path).name}]", self.styles['ChartCaption']))

        return KeepTogether(elements)

    def _create_market_condition_table(self, market_condition) -> Table:
        """Create market condition summary table."""
        data = [
            ['Metric', 'Value', 'Percentage'],
            ['Total Stocks Analyzed', f"{market_condition.total_stocks:,}", '100.0%'],
            ['Strong RS Stocks (>1.05)', f"{market_condition.strong_rs_stocks:,}", f"{market_condition.market_breadth_pct:.1f}%"],
            ['Consistent Performers', f"{market_condition.consistent_stocks:,}", f"{market_condition.consistency_breadth_pct:.1f}%"],
            ['Elite Performers (90th+)', f"{market_condition.elite_stocks:,}", f"{market_condition.elite_breadth_pct:.1f}%"],
            ['Market Condition', market_condition.condition.replace('_', ' ').title(), ''],
        ]

        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        return table

    def _create_top_performers_table(self, top_performers: List[Dict]) -> Table:
        """Create top performers table."""
        data = [['Rank', 'Ticker', 'Composite Strength', 'Trend Consistency %', 'Classification']]

        for i, stock in enumerate(top_performers, 1):
            data.append([
                str(i),
                stock['ticker'],
                f"{stock['composite_strength']:.3f}",
                f"{stock['trend_consistency']:.1f}%",
                stock['classification']
            ])

        table = Table(data, colWidths=[0.5*inch, 1*inch, 1.2*inch, 1.3*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        return table

    def _create_sector_summary_table(self, sectors_analysis: Dict) -> Table:
        """Create sector summary table."""
        individual_analysis = sectors_analysis.get('individual_analysis', [])

        if not individual_analysis:
            return Paragraph("No sector data available.", self.styles['Body'])

        data = [['Sector', 'RS Strength', 'Momentum %', 'Classification', 'Rotation Signal']]

        for sector in individual_analysis[:8]:  # Limit to top 8 for space
            data.append([
                sector['sector'][:20],  # Truncate long names
                f"{sector['composite_strength']:.3f}",
                f"{sector['momentum_medium']:.1f}%",
                sector['classification'],
                sector['rotation_signal']
            ])

        table = Table(data, colWidths=[2*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        return table

    def _create_industry_quadrant_table(self, industries_analysis: Dict) -> Table:
        """Create industry quadrant distribution table."""
        quadrant_analysis = industries_analysis.get('quadrant_analysis', {})

        if not quadrant_analysis:
            return Paragraph("No industry quadrant data available.", self.styles['Body'])

        data = [
            ['Quadrant', 'Count', 'Description'],
            ['Leading & Improving', str(len(quadrant_analysis.get('leading_improving', []))), 'Buy/Overweight'],
            ['Leading & Weakening', str(len(quadrant_analysis.get('leading_weakening', []))), 'Profit Taking'],
            ['Lagging & Improving', str(len(quadrant_analysis.get('lagging_improving', []))), 'Emerging Opportunities'],
            ['Lagging & Weakening', str(len(quadrant_analysis.get('lagging_weakening', []))), 'Avoid/Underweight']
        ]

        table = Table(data, colWidths=[2*inch, 1*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        return table

    def _create_risk_summary_table(self, analysis_results: Dict) -> Table:
        """Create risk management summary table."""
        market_condition = analysis_results.get('market_condition')

        if not market_condition:
            return Paragraph("Risk summary unavailable.", self.styles['Body'])

        # Determine risk parameters based on market condition
        if market_condition.market_breadth_pct > 50:
            risk_profile = "Aggressive"
            max_position = "5-8%"
            portfolio_beta = "1.0-1.3"
            stop_loss = "8-12%"
        elif market_condition.market_breadth_pct > 30:
            risk_profile = "Moderate"
            max_position = "3-5%"
            portfolio_beta = "0.8-1.1"
            stop_loss = "6-10%"
        else:
            risk_profile = "Conservative"
            max_position = "2-3%"
            portfolio_beta = "0.5-0.8"
            stop_loss = "5-8%"

        data = [
            ['Risk Parameter', 'Recommendation'],
            ['Risk Profile', risk_profile],
            ['Max Position Size', max_position],
            ['Portfolio Beta Target', portfolio_beta],
            ['Stop Loss Levels', stop_loss],
            ['Cash Allocation', '10-20%']
        ]

        table = Table(data, colWidths=[2.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        return table