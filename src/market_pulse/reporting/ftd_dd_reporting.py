"""
FTD/DD Enhanced PDF Report Generator
===================================

Generates professional PDF reports combining Follow-Through Days and Distribution Days analysis.
Based on William J. O'Neil's market timing methodology with enhanced design using ReportLab.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
import json

from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)


class FTDDistributionReportGenerator:
    """
    Generate enhanced FTD/DD PDF reports with professional design.

    Features:
    - Combined FTD + DD analysis
    - Chart integration without bottom text
    - Signal strength/severity indicators
    - Professional color scheme
    - Market direction assessment based on O'Neil methodology
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
            'text': colors.black,
            'ftd_green': colors.Color(0.2, 0.8, 0.2),    # Bright Green
            'dd_red': colors.Color(0.8, 0.2, 0.2)        # Bright Red
        }

        # FTD strength definitions
        self.ftd_strengths = {
            'Strong': 'Exceptional volume surge (>50%) with significant price advance (>2.0%)',
            'Moderate': 'Good volume increase (>35%) with solid price advance (>1.5%)',
            'Weak': 'Minimal criteria met - watch for confirmation'
        }

        # DD severity definitions
        self.dd_severities = {
            'Severe': 'Heavy institutional selling with significant decline (>1.0%)',
            'Moderate': 'Notable selling pressure with moderate decline (>0.5%)',
            'Mild': 'Light distribution - monitor for accumulation'
        }

    def generate_report(self, ftd_dd_csv_path: Path, chart_path: Path, signals_path: Path,
                       output_path: Path, index_symbol: str = 'SPY') -> bool:
        """
        Generate enhanced FTD/DD PDF report.

        Args:
            ftd_dd_csv_path: Path to FTD/DD analysis CSV file
            chart_path: Path to FTD/DD chart PNG file
            signals_path: Path to FTD/DD signals JSON file
            output_path: Path for output PDF file
            index_symbol: Index symbol being analyzed

        Returns:
            bool: True if report generated successfully
        """
        try:
            # Load data
            ftd_dd_data = pd.read_csv(ftd_dd_csv_path) if ftd_dd_csv_path.exists() else None
            signals_data = None

            if signals_path and signals_path.exists():
                with open(signals_path, 'r') as f:
                    signals_data = json.load(f)

            if ftd_dd_data is None:
                logger.error(f"No FTD/DD data found for {index_symbol}")
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
            story.extend(self._create_title_section(ftd_dd_data, index_symbol))

            # Add executive summary
            story.extend(self._create_executive_summary(ftd_dd_data))

            # Add chart (if available)
            if chart_path and chart_path.exists():
                story.extend(self._create_chart_section(chart_path))

            # Add FTD analysis section
            if signals_data:
                story.extend(self._create_ftd_section(signals_data))

            # Add DD analysis section
            if signals_data:
                story.extend(self._create_dd_section(signals_data))

            # Add market assessment
            story.extend(self._create_market_assessment(ftd_dd_data, signals_data))

            # Build PDF
            doc.build(story)

            logger.info(f"Enhanced FTD/DD report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating FTD/DD report: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _create_title_section(self, ftd_dd_data: pd.DataFrame, index_symbol: str) -> List:
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
            f"<b>FOLLOW-THROUGH DAYS & DISTRIBUTION DAYS ANALYSIS ({index_symbol})</b>",
            title_style
        ))

        # Subtitle
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=self.colors['text'],
            alignment=TA_CENTER,
            spaceAfter=20
        )

        story.append(Paragraph(
            "William J. O'Neil Market Timing Methodology",
            subtitle_style
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

        # Get date from data
        if not ftd_dd_data.empty:
            analysis_date = ftd_dd_data.iloc[0]['date']
            story.append(Paragraph(
                f"As of Close {analysis_date}",
                date_style
            ))

        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_executive_summary(self, ftd_dd_data: pd.DataFrame) -> List:
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

        if ftd_dd_data.empty:
            return story

        # Get data
        row = ftd_dd_data.iloc[0]

        # Create summary table
        summary_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Market State', row['market_state'], self._get_market_state_interpretation(row['market_state'])],
            ['Confidence Level', row['confidence'], self._get_confidence_interpretation(row['confidence'])],
            ['Recent FTDs', f"{row['recent_ftd_count']} (last {row.get('recent_activity_period', 25)} days)",
             self._get_ftd_interpretation(row['recent_ftd_count'])],
            ['Recent DDs', f"{row['recent_dd_count']} (last {row.get('recent_activity_period', 25)} days)",
             self._get_dd_interpretation(row['recent_dd_count'])],
            ['FTD/DD Ratio', f"{row['ftd_dd_ratio']:.2f}", self._get_ratio_interpretation(row['ftd_dd_ratio'])],
            ['Signal Strength', row['signal_strength'], row['latest_signal']]
        ]

        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['header']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (2, 1), (2, -1), 'LEFT'),  # Left-align interpretation column
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        return story

    def _create_chart_section(self, chart_path: Path) -> List:
        """Create chart section with embedded PNG."""
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

        story.append(Paragraph("<b>TECHNICAL ANALYSIS CHART</b>", header_style))

        try:
            # Add chart image - optimal scaling for 16:10 aspect ratio chart
            chart_image = Image(str(chart_path), width=8*inch, height=5*inch)
            story.append(chart_image)
            story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            logger.error(f"Error adding chart to report: {e}")
            story.append(Paragraph("Chart could not be loaded", styles['Normal']))

        return story

    def _create_ftd_section(self, signals_data: Dict) -> List:
        """Create detailed FTD analysis section."""
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

        ftd_signals = signals_data.get('follow_through_days', {}).get('signals', [])
        ftd_count = len(ftd_signals)

        story.append(Paragraph(f"<b>FOLLOW-THROUGH DAYS ANALYSIS: {ftd_count} Recent Signals</b>", header_style))

        # Add FTD description
        description_style = ParagraphStyle(
            'Description',
            parent=styles['Normal'],
            fontSize=11,
            textColor=self.colors['text'],
            alignment=TA_LEFT,
            spaceAfter=10
        )

        story.append(Paragraph(
            "<b>Follow-Through Days</b> indicate institutional buying and potential market trend continuation. "
            "They require significant price advance with above-average volume.",
            description_style
        ))

        if ftd_signals:
            # Create FTD table
            ftd_data = [['#', 'Date', 'Price Change', 'Volume Ratio', 'Strength', 'Close Price', 'Days Ago']]

            for signal in ftd_signals[:10]:  # Show latest 10
                ftd_data.append([
                    str(signal['sequence_number']),
                    signal['date'],
                    f"{signal['price_change_pct']:.2f}%",
                    f"{signal['volume_ratio']:.2f}x",
                    signal['strength'],
                    f"${signal['close_price']:.2f}",
                    str(signal['days_ago'])
                ])

            ftd_table = Table(ftd_data, colWidths=[0.3*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 0.8*inch])
            ftd_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['ftd_green']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))

            story.append(ftd_table)
        else:
            story.append(Paragraph("No recent Follow-Through Days detected.", styles['Normal']))

        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_dd_section(self, signals_data: Dict) -> List:
        """Create detailed DD analysis section."""
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

        dd_signals = signals_data.get('distribution_days', {}).get('signals', [])
        dd_count = len(dd_signals)

        story.append(Paragraph(f"<b>DISTRIBUTION DAYS ANALYSIS: {dd_count} Recent Signals</b>", header_style))

        # Add DD description
        description_style = ParagraphStyle(
            'Description',
            parent=styles['Normal'],
            fontSize=11,
            textColor=self.colors['text'],
            alignment=TA_LEFT,
            spaceAfter=10
        )

        story.append(Paragraph(
            "<b>Distribution Days</b> indicate institutional selling and potential market weakness. "
            "They show price decline with above-average volume, suggesting professional selling.",
            description_style
        ))

        if dd_signals:
            # Create DD table
            dd_data = [['#', 'Date', 'Price Change', 'Volume Ratio', 'Severity', 'Close Price', 'Days Ago']]

            for signal in dd_signals[:10]:  # Show latest 10
                dd_data.append([
                    str(signal['sequence_number']),
                    signal['date'],
                    f"{signal['price_change_pct']:.2f}%",
                    f"{signal['volume_ratio']:.2f}x",
                    signal['severity'],
                    f"${signal['close_price']:.2f}",
                    str(signal['days_ago'])
                ])

            dd_table = Table(dd_data, colWidths=[0.3*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 0.8*inch])
            dd_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['dd_red']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))

            story.append(dd_table)
        else:
            story.append(Paragraph("No recent Distribution Days detected.", styles['Normal']))

        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_market_assessment(self, ftd_dd_data: pd.DataFrame, signals_data: Dict) -> List:
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

        story.append(Paragraph("<b>MARKET ASSESSMENT & O'NEIL METHODOLOGY</b>", header_style))

        if ftd_dd_data.empty:
            return story

        row = ftd_dd_data.iloc[0]

        # Generate assessment based on O'Neil methodology
        assessment = self._generate_oneill_assessment(row, signals_data)

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

    def _get_market_state_interpretation(self, state: str) -> str:
        """Get interpretation for market state."""
        interpretations = {
            'Confirmed Uptrend': 'Strong bullish conditions - Multiple FTDs confirm uptrend',
            'Uptrend': 'Bullish conditions - Uptrend in progress',
            'Under Pressure': 'Bearish conditions - Accumulating distribution',
            'Correction': 'Strong bearish conditions - Heavy distribution',
            'Neutral': 'Mixed conditions - No clear trend direction'
        }
        return interpretations.get(state, 'Unknown market state')

    def _get_confidence_interpretation(self, confidence: str) -> str:
        """Get interpretation for confidence level."""
        interpretations = {
            'High': 'Strong signal reliability - Clear market direction',
            'Moderate': 'Good signal reliability - Moderate conviction',
            'Low': 'Weak signal reliability - Mixed signals'
        }
        return interpretations.get(confidence, 'Unknown confidence')

    def _get_ftd_interpretation(self, count: int) -> str:
        """Get interpretation for FTD count."""
        if count >= 2:
            return 'Strong bullish indication'
        elif count == 1:
            return 'Moderate bullish indication'
        else:
            return 'No recent bullish confirmation'

    def _get_dd_interpretation(self, count: int) -> str:
        """Get interpretation for DD count."""
        if count >= 5:
            return 'Strong bearish indication'
        elif count >= 3:
            return 'Moderate bearish indication'
        elif count >= 1:
            return 'Some distribution pressure'
        else:
            return 'No significant distribution'

    def _get_ratio_interpretation(self, ratio: float) -> str:
        """Get interpretation for FTD/DD ratio."""
        if ratio >= 2.0:
            return 'Strong bullish dominance'
        elif ratio >= 1.0:
            return 'Bullish bias'
        elif ratio >= 0.5:
            return 'Bearish bias'
        else:
            return 'Strong bearish dominance'

    def _generate_oneill_assessment(self, row: pd.DataFrame, signals_data: Dict) -> List[str]:
        """Generate market assessment based on O'Neil methodology."""
        assessment = []

        market_state = row['market_state']
        ftd_count = row['recent_ftd_count']
        dd_count = row['recent_dd_count']
        ratio = row['ftd_dd_ratio']

        # Market direction assessment
        if market_state in ['Confirmed Uptrend', 'Uptrend']:
            assessment.append("<b>📈 BULLISH MARKET OUTLOOK</b>")
            assessment.append("• <b>Market Direction:</b> Follow-Through Days confirm institutional buying")
            assessment.append("• <b>Recommended Action:</b> Consider increasing equity exposure")
            assessment.append("• <b>Risk Level:</b> Low to Moderate")
        elif market_state in ['Under Pressure', 'Correction']:
            assessment.append("<b>📉 BEARISH MARKET OUTLOOK</b>")
            assessment.append("• <b>Market Direction:</b> Distribution Days indicate institutional selling")
            assessment.append("• <b>Recommended Action:</b> Consider reducing equity exposure or defensive positioning")
            assessment.append("• <b>Risk Level:</b> High")
        else:
            assessment.append("<b>⚠️ NEUTRAL MARKET OUTLOOK</b>")
            assessment.append("• <b>Market Direction:</b> Mixed signals - no clear institutional bias")
            assessment.append("• <b>Recommended Action:</b> Maintain current allocation, monitor closely")
            assessment.append("• <b>Risk Level:</b> Moderate")

        assessment.append("")
        assessment.append("<b>William J. O'Neil Key Principles:</b>")
        assessment.append("• <b>Follow-Through Days:</b> Confirm new uptrends and institutional buying")
        assessment.append("• <b>Distribution Days:</b> Signal potential market tops and selling pressure")
        assessment.append("• <b>Volume Analysis:</b> Price without volume is meaningless")
        assessment.append("• <b>Cluster Analysis:</b> Multiple signals in short timeframe increase reliability")

        assessment.append("")
        assessment.append("<b>Current Market Metrics:</b>")
        assessment.append(f"• Recent FTD Count: {ftd_count} - {self._get_ftd_interpretation(ftd_count)}")
        assessment.append(f"• Recent DD Count: {dd_count} - {self._get_dd_interpretation(dd_count)}")
        assessment.append(f"• FTD/DD Ratio: {ratio:.2f} - {self._get_ratio_interpretation(ratio)}")

        return assessment


def generate_ftd_dd_reports(market_pulse_results_dir: Path, output_dir: Path, user_config) -> List[str]:
    """
    Generate FTD/DD reports for all available analysis results.

    Args:
        market_pulse_results_dir: Directory containing FTD/DD CSV files
        output_dir: Directory for output PDF reports
        user_config: User configuration

    Returns:
        List of generated report file paths
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_generator = FTDDistributionReportGenerator(user_config)

        generated_reports = []

        # Find all FTD/DD files
        ftd_dd_files = list(market_pulse_results_dir.glob('ftd_dd_*.csv'))

        for ftd_dd_file in ftd_dd_files:
            # Extract details from filename
            # Format: ftd_dd_SPY_2-5_daily_20250905.csv
            parts = ftd_dd_file.stem.split('_')
            if len(parts) >= 5:
                index_symbol = parts[2]
                choice = parts[3]
                timeframe = parts[4]
                date = parts[5] if len(parts) > 5 else 'latest'

                # Find corresponding chart and signals files
                chart_file = market_pulse_results_dir / f'ftd_dd_chart_{index_symbol}_{choice}_{timeframe}_{date}.png'
                signals_file = market_pulse_results_dir / f'ftd_dd_signals_{index_symbol}_{choice}_{timeframe}_{date}.json'

                # Generate report
                report_name = f'ftd_dd_{index_symbol}_{choice}_{timeframe}_{date}.pdf'
                report_path = output_dir / report_name

                success = report_generator.generate_report(
                    ftd_dd_csv_path=ftd_dd_file,
                    chart_path=chart_file if chart_file.exists() else None,
                    signals_path=signals_file if signals_file.exists() else None,
                    output_path=report_path,
                    index_symbol=index_symbol
                )

                if success:
                    generated_reports.append(str(report_path))
                    logger.info(f"Generated FTD/DD report: {report_name}")

        return generated_reports

    except Exception as e:
        logger.error(f"Error generating FTD/DD reports: {e}")
        return []