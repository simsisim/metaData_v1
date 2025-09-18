"""
GMI/GMI2 Enhanced PDF Report Generator
====================================

Generates professional PDF reports combining GMI and GMI2 analysis with enhanced design.
Based on the market breadth reporting architecture but optimized for GMI/GMI2 signals.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging

from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)


class GMI_GMI2_ReportGenerator:
    """
    Generate enhanced GMI/GMI2 PDF reports with professional design.

    Features:
    - Combined GMI + GMI2 analysis
    - Enhanced visual design
    - Signal strength indicators
    - Professional color scheme
    - Market direction assessment
    """

    def __init__(self, user_config=None):
        self.user_config = user_config

        # Enhanced color scheme
        self.colors = {
            'bullish': colors.Color(0.0, 0.5, 0.0),      # Dark Green
            'bearish': colors.Color(0.7, 0.0, 0.0),      # Dark Red
            'neutral': colors.Color(0.8, 0.6, 0.0),      # Orange
            'header': colors.Color(0.1, 0.2, 0.4),       # Dark Blue
            'background': colors.Color(0.95, 0.95, 0.95), # Light Gray
            'text': colors.black
        }

        # GMI component descriptions
        self.gmi_components = {
            'r1': 'Wishing Wealth 10-Day Successful New High Index ≥50%',
            'r2': 'At least 100 New Highs Today (6,000+ US Stocks)',
            'r3': 'Wishing Wealth Daily QQQ Index Positive',
            'r4': 'Wishing Wealth Daily SPY Index Positive',
            'r5': 'Wishing Wealth Weekly QQQ Index Positive',
            'r6': 'IBD Mutual Fund Index > 50-Day Average'
        }

        # GMI2 component descriptions
        self.gmi2_components = {
            'r1': 'More US New Highs than Lows Today',
            'r2': 'QQQ Closed Above 10-Week Average',
            'r3': 'QQQ Closed Above 4-Week Average',
            'r4': 'QQQ Closed Above 10-Day Average',
            'r5': 'QQQ 4-Week Average > 10W/30W Average',
            'r6': 'QQQ Daily 10.4 Stochastic <20',
            'r7': 'QQQ Daily 12.26.9 MACD Histogram Rising or Black',
            'r8': 'QQQ 10.4.4 Daily Stochastic Fast>Slow or Above 80',
            'r9': 'QQQ Daily 10.1 Stochastic <=20'
        }

    def generate_report(self, gmi_csv_path: Path, gmi2_csv_path: Path, output_path: Path,
                       index_symbol: str = 'SPY') -> bool:
        """
        Generate enhanced GMI/GMI2 PDF report.

        Args:
            gmi_csv_path: Path to GMI analysis CSV file
            gmi2_csv_path: Path to GMI2 analysis CSV file
            output_path: Path for output PDF file
            index_symbol: Index symbol being analyzed

        Returns:
            bool: True if report generated successfully
        """
        try:
            # Load data
            gmi_data = pd.read_csv(gmi_csv_path) if gmi_csv_path.exists() else None
            gmi2_data = pd.read_csv(gmi2_csv_path) if gmi2_csv_path.exists() else None

            if gmi_data is None and gmi2_data is None:
                logger.error(f"No GMI or GMI2 data found for {index_symbol}")
                return False

            # Create enhanced PDF
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=landscape(A4),
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.7*inch,
                bottomMargin=0.5*inch
            )

            # Build report content
            story = []

            # Add title section
            story.extend(self._create_title_section(gmi_data, gmi2_data, index_symbol))

            # Add executive summary
            story.extend(self._create_executive_summary(gmi_data, gmi2_data))

            # Add GMI analysis section
            if gmi_data is not None:
                story.extend(self._create_gmi_section(gmi_data))

            # Add GMI2 analysis section
            if gmi2_data is not None:
                story.extend(self._create_gmi2_section(gmi2_data))

            # Add market assessment
            story.extend(self._create_market_assessment(gmi_data, gmi2_data))

            # Build PDF
            doc.build(story)

            logger.info(f"Enhanced GMI/GMI2 report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating GMI/GMI2 report: {e}")
            return False

    def _create_title_section(self, gmi_data: pd.DataFrame, gmi2_data: pd.DataFrame,
                             index_symbol: str) -> List:
        """Create enhanced title section."""
        story = []
        styles = getSampleStyleSheet()

        # Main title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=20,
            textColor=self.colors['header'],
            alignment=TA_CENTER,
            spaceAfter=12
        )

        story.append(Paragraph(
            f"<b>WISHING WEALTH MARKET PULSE ANALYSIS ({index_symbol})</b>",
            title_style
        ))

        # Date subtitle
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=self.colors['text'],
            alignment=TA_CENTER,
            spaceAfter=20
        )

        # Get date from available data
        analysis_date = None
        if gmi_data is not None and not gmi_data.empty:
            analysis_date = gmi_data.iloc[0]['date']
        elif gmi2_data is not None and not gmi2_data.empty:
            analysis_date = gmi2_data.iloc[0]['date']

        if analysis_date:
            story.append(Paragraph(
                f"As of Close {analysis_date}",
                date_style
            ))

        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_executive_summary(self, gmi_data: pd.DataFrame, gmi2_data: pd.DataFrame) -> List:
        """Create executive summary section."""
        story = []
        styles = getSampleStyleSheet()

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.colors['header'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        story.append(Paragraph("<b>EXECUTIVE SUMMARY</b>", header_style))

        # Create summary table
        summary_data = []

        # GMI Summary
        if gmi_data is not None and not gmi_data.empty:
            gmi_row = gmi_data.iloc[0]
            gmi_score = int(gmi_row['daily_gmi_score'])
            gmi_signal = gmi_row['daily_gmi_signal']
            gmi_color = self._get_signal_color(gmi_signal)

            summary_data.append([
                'GMI (General Market Index)',
                f'{gmi_score}/6',
                gmi_signal,
                self._get_signal_interpretation(gmi_signal)
            ])

        # GMI2 Summary
        if gmi2_data is not None and not gmi2_data.empty:
            gmi2_row = gmi2_data.iloc[0]
            gmi2_score = int(gmi2_row['daily_gmi2_score'])
            gmi2_signal = gmi2_row['daily_gmi2_signal']

            summary_data.append([
                'GMI2 (Enhanced Market Index)',
                f'{gmi2_score}/9',
                gmi2_signal,
                self._get_signal_interpretation(gmi2_signal)
            ])

        # Overall Assessment
        overall_signal = self._calculate_overall_signal(gmi_data, gmi2_data)
        summary_data.append([
            'Overall Market Direction',
            '-',
            overall_signal,
            self._get_signal_interpretation(overall_signal)
        ])

        if summary_data:
            summary_table = Table(summary_data, colWidths=[2.5*inch, 1*inch, 1*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['header']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            # Add header row
            header_data = [['Analysis Type', 'Score', 'Signal', 'Interpretation']]
            header_table = Table(header_data, colWidths=[2.5*inch, 1*inch, 1*inch, 3*inch])
            header_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['header']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ]))

            story.append(header_table)
            story.append(summary_table)

        story.append(Spacer(1, 0.3*inch))
        return story

    def _create_gmi_section(self, gmi_data: pd.DataFrame) -> List:
        """Create detailed GMI analysis section."""
        story = []
        styles = getSampleStyleSheet()

        if gmi_data.empty:
            return story

        gmi_row = gmi_data.iloc[0]

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.colors['header'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        gmi_score = int(gmi_row['daily_gmi_score'])
        story.append(Paragraph(f"<b>GMI ANALYSIS: {gmi_score}/6</b>", header_style))

        # Create detailed component table
        component_data = []
        for i in range(1, 7):
            component_key = f'r{i}'
            component_value = gmi_row[f'daily_gmi_{component_key}']
            status = "YES" if component_value == 1.0 else "NO"
            color = self.colors['bullish'] if component_value == 1.0 else self.colors['bearish']

            component_data.append([
                f"{i}.",
                self.gmi_components[component_key],
                status
            ])

        gmi_table = Table(component_data, colWidths=[0.3*inch, 5.5*inch, 0.8*inch])
        gmi_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        # Color code the YES/NO responses
        for i, row in enumerate(component_data):
            if row[2] == "YES":
                gmi_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (-1, i), (-1, i), self.colors['bullish']),
                    ('FONTNAME', (-1, i), (-1, i), 'Helvetica-Bold'),
                ]))
            else:
                gmi_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (-1, i), (-1, i), self.colors['bearish']),
                    ('FONTNAME', (-1, i), (-1, i), 'Helvetica-Bold'),
                ]))

        story.append(gmi_table)
        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_gmi2_section(self, gmi2_data: pd.DataFrame) -> List:
        """Create detailed GMI2 analysis section."""
        story = []
        styles = getSampleStyleSheet()

        if gmi2_data.empty:
            return story

        gmi2_row = gmi2_data.iloc[0]

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.colors['header'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        gmi2_score = int(gmi2_row['daily_gmi2_score'])
        story.append(Paragraph(f"<b>GMI2 ANALYSIS: {gmi2_score}/9</b>", header_style))

        # Create detailed component table
        component_data = []
        for i in range(1, 10):
            component_key = f'r{i}'
            component_value = gmi2_row[f'daily_gmi2_{component_key}']
            status = "YES" if component_value == 1.0 else "NO"

            component_data.append([
                f"{i}.",
                self.gmi2_components[component_key],
                status
            ])

        gmi2_table = Table(component_data, colWidths=[0.3*inch, 5.5*inch, 0.8*inch])
        gmi2_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        # Color code the YES/NO responses
        for i, row in enumerate(component_data):
            if row[2] == "YES":
                gmi2_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (-1, i), (-1, i), self.colors['bullish']),
                    ('FONTNAME', (-1, i), (-1, i), 'Helvetica-Bold'),
                ]))
            else:
                gmi2_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (-1, i), (-1, i), self.colors['bearish']),
                    ('FONTNAME', (-1, i), (-1, i), 'Helvetica-Bold'),
                ]))

        story.append(gmi2_table)
        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_market_assessment(self, gmi_data: pd.DataFrame, gmi2_data: pd.DataFrame) -> List:
        """Create overall market assessment section."""
        story = []
        styles = getSampleStyleSheet()

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.colors['header'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        story.append(Paragraph("<b>MARKET ASSESSMENT & RECOMMENDATIONS</b>", header_style))

        # Calculate combined assessment
        assessment = self._generate_assessment(gmi_data, gmi2_data)

        # Create assessment content
        content_style = ParagraphStyle(
            'Assessment',
            parent=styles['Normal'],
            fontSize=12,
            textColor=self.colors['text'],
            alignment=TA_LEFT,
            spaceAfter=10
        )

        for item in assessment:
            story.append(Paragraph(item, content_style))

        return story

    def _get_signal_color(self, signal: str) -> colors.Color:
        """Get color for signal type."""
        if signal == 'GREEN':
            return self.colors['bullish']
        elif signal == 'RED':
            return self.colors['bearish']
        else:
            return self.colors['neutral']

    def _get_signal_interpretation(self, signal: str) -> str:
        """Get interpretation for signal."""
        interpretations = {
            'GREEN': 'Bullish - Market conditions favorable',
            'RED': 'Bearish - Market conditions unfavorable',
            'YELLOW': 'Neutral - Mixed market conditions',
            'ORANGE': 'Caution - Transitional conditions'
        }
        return interpretations.get(signal, 'Unknown signal')

    def _calculate_overall_signal(self, gmi_data: pd.DataFrame, gmi2_data: pd.DataFrame) -> str:
        """Calculate overall market signal from GMI + GMI2."""
        signals = []

        if gmi_data is not None and not gmi_data.empty:
            signals.append(gmi_data.iloc[0]['daily_gmi_signal'])

        if gmi2_data is not None and not gmi2_data.empty:
            signals.append(gmi2_data.iloc[0]['daily_gmi2_signal'])

        if not signals:
            return 'UNKNOWN'

        # Simple consensus logic
        green_count = signals.count('GREEN')
        red_count = signals.count('RED')

        if green_count > red_count:
            return 'GREEN'
        elif red_count > green_count:
            return 'RED'
        else:
            return 'YELLOW'

    def _generate_assessment(self, gmi_data: pd.DataFrame, gmi2_data: pd.DataFrame) -> List[str]:
        """Generate market assessment and recommendations."""
        assessment = []

        # Overall signal assessment
        overall = self._calculate_overall_signal(gmi_data, gmi2_data)

        if overall == 'GREEN':
            assessment.append("<b>📈 BULLISH OUTLOOK:</b> Both GMI and GMI2 indicators suggest favorable market conditions.")
            assessment.append("• <b>Recommended Action:</b> Consider increasing equity exposure")
            assessment.append("• <b>Risk Level:</b> Low to Moderate")
        elif overall == 'RED':
            assessment.append("<b>📉 BEARISH OUTLOOK:</b> Market indicators suggest unfavorable conditions.")
            assessment.append("• <b>Recommended Action:</b> Consider reducing equity exposure or defensive positioning")
            assessment.append("• <b>Risk Level:</b> High")
        else:
            assessment.append("<b>⚠️ MIXED SIGNALS:</b> Market indicators show conflicting signals.")
            assessment.append("• <b>Recommended Action:</b> Maintain current allocation, monitor closely")
            assessment.append("• <b>Risk Level:</b> Moderate")

        # Add specific recommendations
        assessment.append("")
        assessment.append("<b>Key Monitoring Points:</b>")
        assessment.append("• Watch for changes in new highs/lows ratio")
        assessment.append("• Monitor QQQ technical indicators for trend changes")
        assessment.append("• Track mutual fund index momentum")

        return assessment


def generate_gmi_gmi2_reports(market_pulse_results_dir: Path, output_dir: Path, user_config) -> List[str]:
    """
    Generate GMI/GMI2 reports for all available analysis results.

    Args:
        market_pulse_results_dir: Directory containing GMI/GMI2 CSV files
        output_dir: Directory for output PDF reports
        user_config: User configuration

    Returns:
        List of generated report file paths
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_generator = GMI_GMI2_ReportGenerator(user_config)

        generated_reports = []

        # Find all GMI files and match with GMI2 files
        gmi_files = list(market_pulse_results_dir.glob('gmi_*.csv'))

        for gmi_file in gmi_files:
            # Extract index symbol and details from filename
            # Format: gmi_SPY_2-5_daily_20250905.csv
            parts = gmi_file.stem.split('_')
            if len(parts) >= 4:
                index_symbol = parts[1]
                choice = parts[2]
                timeframe = parts[3]
                date = parts[4] if len(parts) > 4 else 'latest'

                # Find corresponding GMI2 file
                gmi2_file = market_pulse_results_dir / f'gmi2_{index_symbol}_{choice}_{timeframe}_{date}.csv'

                # Generate report
                report_name = f'gmi_gmi2_{index_symbol}_{choice}_{timeframe}_{date}.pdf'
                report_path = output_dir / report_name

                success = report_generator.generate_report(
                    gmi_csv_path=gmi_file,
                    gmi2_csv_path=gmi2_file if gmi2_file.exists() else None,
                    output_path=report_path,
                    index_symbol=index_symbol
                )

                if success:
                    generated_reports.append(str(report_path))
                    logger.info(f"Generated GMI/GMI2 report: {report_name}")

        return generated_reports

    except Exception as e:
        logger.error(f"Error generating GMI/GMI2 reports: {e}")
        return []