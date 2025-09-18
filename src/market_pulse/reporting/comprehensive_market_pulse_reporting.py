"""
Comprehensive Market Pulse Report Generator
==========================================

Combines all market pulse indicators (Chillax MAs, MA Cycles, FTD/DD) into unified reports.
Leverages existing PNG charts and individual PDF reports for efficient document generation.

File naming: market_pulse_[INDEX]_[CHOICE]_[TIMEFRAME]_[DATE].pdf
Examples: market_pulse_SPY_2-5_daily_20250905.pdf
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
import subprocess
import tempfile
import os

from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.frames import Frame

logger = logging.getLogger(__name__)


class ComprehensiveMarketPulseReportGenerator:
    """
    Generate comprehensive market pulse PDF reports combining all indicators.

    Features:
    - Combines Chillax MAs, MA Cycles, and FTD/DD analysis
    - Leverages existing PNG charts and PDF reports
    - Professional ReportLab-based design
    - One unified report per index/timeframe combination
    """

    def __init__(self, user_config=None):
        self.user_config = user_config

        # Enhanced color scheme
        self.colors = {
            'primary': colors.Color(0.1, 0.2, 0.4),       # Dark Blue
            'secondary': colors.Color(0.2, 0.4, 0.6),     # Medium Blue
            'accent': colors.Color(0.8, 0.6, 0.0),        # Orange
            'bullish': colors.Color(0.0, 0.5, 0.0),       # Dark Green
            'bearish': colors.Color(0.7, 0.0, 0.0),       # Dark Red
            'neutral': colors.Color(0.5, 0.5, 0.5),       # Gray
            'background': colors.Color(0.95, 0.95, 0.95), # Light Gray
            'text': colors.black
        }

    def generate_comprehensive_report(self, index: str, choice: str, timeframe: str,
                                    data_date: str, market_pulse_dir: Path,
                                    reports_dir: Path, output_path: Path) -> bool:
        """
        Generate comprehensive market pulse report for a specific index/timeframe.

        Args:
            index: Index symbol (SPY, QQQ, IWM)
            choice: Ticker choice (2-5)
            timeframe: Data timeframe (daily, weekly)
            data_date: Data date (20250905)
            market_pulse_dir: Directory containing PNG charts
            reports_dir: Directory containing individual PDF reports
            output_path: Path for output PDF file

        Returns:
            bool: True if report generated successfully
        """
        try:
            logger.info(f"Generating comprehensive market pulse report for {index} - {timeframe}")

            # Collect available components
            components = self._collect_available_components(
                index, choice, timeframe, data_date, market_pulse_dir, reports_dir
            )

            if not components['has_any_components']:
                logger.warning(f"No market pulse components found for {index} - {timeframe}")
                return False

            # Create PDF document
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
            story.extend(self._create_title_section(index, timeframe, data_date))

            # Add executive summary
            story.extend(self._create_executive_summary(components, index, timeframe))

            # Add Chillax MAs section (if available)
            if components['chillax']['available']:
                story.extend(self._create_chillax_section(components['chillax'], index))

            # Add MA Cycles section (if available)
            if components['ma_cycles']['available']:
                story.extend(self._create_ma_cycles_section(components['ma_cycles'], index))

            # Add FTD/DD section (if available)
            if components['ftd_dd']['available']:
                story.extend(self._create_ftd_dd_section(components['ftd_dd'], index))

            # Add combined market assessment
            story.extend(self._create_combined_market_assessment(components, index))

            # Build PDF
            doc.build(story)

            logger.info(f"Comprehensive market pulse report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating comprehensive market pulse report: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _collect_available_components(self, index: str, choice: str, timeframe: str,
                                    data_date: str, market_pulse_dir: Path,
                                    reports_dir: Path) -> Dict[str, Any]:
        """
        Collect all available market pulse components for the specified parameters.

        Returns:
            Dictionary containing paths and availability status for each component
        """
        components = {
            'index': index,
            'choice': choice,
            'timeframe': timeframe,
            'data_date': data_date,
            'has_any_components': False,
            'chillax': {'available': False, 'chart_path': None},
            'ma_cycles': {'available': False, 'chart_path': None},
            'ftd_dd': {'available': False, 'chart_path': None, 'report_path': None, 'signals_path': None}
        }

        # Check for Chillax MAs chart
        chillax_chart = market_pulse_dir / f"chillax_mas_{index}_{choice}_{timeframe}_{data_date}.png"
        if chillax_chart.exists():
            components['chillax']['available'] = True
            components['chillax']['chart_path'] = chillax_chart
            components['has_any_components'] = True
            logger.info(f"Found Chillax chart: {chillax_chart.name}")

        # Check for MA Cycles chart
        ma_cycles_chart = market_pulse_dir / f"ma_cycles_{index}_{choice}_{timeframe}_{data_date}.png"
        if ma_cycles_chart.exists():
            components['ma_cycles']['available'] = True
            components['ma_cycles']['chart_path'] = ma_cycles_chart
            components['has_any_components'] = True
            logger.info(f"Found MA Cycles chart: {ma_cycles_chart.name}")

        # Check for FTD/DD components
        ftd_dd_chart = market_pulse_dir / f"ftd_dd_chart_{index}_{choice}_{timeframe}_{data_date}.png"
        ftd_dd_report = reports_dir / f"ftd_dd_{index}_{choice}_{timeframe}_{data_date}.pdf"
        ftd_dd_signals = market_pulse_dir / f"ftd_dd_signals_{index}_{choice}_{timeframe}_{data_date}.json"

        if ftd_dd_chart.exists():
            components['ftd_dd']['available'] = True
            components['ftd_dd']['chart_path'] = ftd_dd_chart
            components['has_any_components'] = True
            logger.info(f"Found FTD/DD chart: {ftd_dd_chart.name}")

            if ftd_dd_report.exists():
                components['ftd_dd']['report_path'] = ftd_dd_report
                logger.info(f"Found FTD/DD report: {ftd_dd_report.name}")

            if ftd_dd_signals.exists():
                components['ftd_dd']['signals_path'] = ftd_dd_signals
                logger.info(f"Found FTD/DD signals: {ftd_dd_signals.name}")

        return components

    def _create_title_section(self, index: str, timeframe: str, data_date: str) -> List:
        """Create comprehensive report title section."""
        story = []
        styles = getSampleStyleSheet()

        # Main title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=22,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            spaceAfter=12
        )

        story.append(Paragraph(
            f"<b>COMPREHENSIVE MARKET PULSE ANALYSIS ({index})</b>",
            title_style
        ))

        # Subtitle
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=self.colors['secondary'],
            alignment=TA_CENTER,
            spaceAfter=10
        )

        timeframe_display = timeframe.title()
        story.append(Paragraph(
            f"Chillax MAs • MA Cycles • Follow-Through Days & Distribution Days",
            subtitle_style
        ))

        # Date and timeframe
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=self.colors['text'],
            alignment=TA_CENTER,
            spaceAfter=20
        )

        # Format date
        formatted_date = f"{data_date[:4]}-{data_date[4:6]}-{data_date[6:8]}"
        story.append(Paragraph(
            f"{timeframe_display} Analysis • As of {formatted_date}",
            date_style
        ))

        story.append(Spacer(1, 0.3*inch))
        return story

    def _create_executive_summary(self, components: Dict, index: str, timeframe: str) -> List:
        """Create executive summary of all available components."""
        story = []
        styles = getSampleStyleSheet()

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        story.append(Paragraph("<b>EXECUTIVE SUMMARY</b>", header_style))

        # Create summary table
        summary_data = [
            ['Component', 'Status', 'Analysis Available']
        ]

        # Chillax MAs
        chillax_status = "✅ Active" if components['chillax']['available'] else "⏸️ Not Available"
        chillax_analysis = "Moving Average Signals & Display SMAs" if components['chillax']['available'] else "N/A"
        summary_data.append(['Chillax Moving Averages', chillax_status, chillax_analysis])

        # MA Cycles
        ma_cycles_status = "✅ Active" if components['ma_cycles']['available'] else "⏸️ Not Available"
        ma_cycles_analysis = "Bull/Bear Cycle Analysis & Statistics" if components['ma_cycles']['available'] else "N/A"
        summary_data.append(['MA Cycles Analysis', ma_cycles_status, ma_cycles_analysis])

        # FTD/DD
        ftd_dd_status = "✅ Active" if components['ftd_dd']['available'] else "⏸️ Not Available"
        ftd_dd_analysis = "O'Neil Follow-Through & Distribution Days" if components['ftd_dd']['available'] else "N/A"
        summary_data.append(['FTD/DD Analysis', ftd_dd_status, ftd_dd_analysis])

        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (2, 1), (2, -1), 'LEFT'),  # Left-align analysis column
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        return story

    def _create_chillax_section(self, chillax_component: Dict, index: str) -> List:
        """Create Chillax Moving Averages section."""
        story = []
        styles = getSampleStyleSheet()

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        story.append(Paragraph("<b>CHILLAX MOVING AVERAGES ANALYSIS</b>", header_style))

        # Description
        description_style = ParagraphStyle(
            'Description',
            parent=styles['Normal'],
            fontSize=11,
            textColor=self.colors['text'],
            alignment=TA_LEFT,
            spaceAfter=10
        )

        story.append(Paragraph(
            "<b>Chillax Moving Averages</b> provide trend confirmation through multiple SMA analysis. "
            "The system evaluates analysis SMAs for signal generation and display SMAs for visual confirmation.",
            description_style
        ))

        # Add chart
        if chillax_component['chart_path']:
            try:
                chart_image = Image(str(chillax_component['chart_path']), width=8*inch, height=5*inch)
                story.append(chart_image)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.error(f"Error adding Chillax chart: {e}")
                story.append(Paragraph("Chillax chart could not be loaded", styles['Normal']))

        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_ma_cycles_section(self, ma_cycles_component: Dict, index: str) -> List:
        """Create MA Cycles Analysis section."""
        story = []
        styles = getSampleStyleSheet()

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        story.append(Paragraph("<b>MOVING AVERAGE CYCLES ANALYSIS</b>", header_style))

        # Description
        description_style = ParagraphStyle(
            'Description',
            parent=styles['Normal'],
            fontSize=11,
            textColor=self.colors['text'],
            alignment=TA_LEFT,
            spaceAfter=10
        )

        story.append(Paragraph(
            "<b>MA Cycles Analysis</b> tracks price movement relative to key moving averages, "
            "identifying bull and bear market cycles with statistical analysis of cycle lengths and patterns.",
            description_style
        ))

        # Add chart
        if ma_cycles_component['chart_path']:
            try:
                chart_image = Image(str(ma_cycles_component['chart_path']), width=8*inch, height=5*inch)
                story.append(chart_image)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.error(f"Error adding MA Cycles chart: {e}")
                story.append(Paragraph("MA Cycles chart could not be loaded", styles['Normal']))

        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_ftd_dd_section(self, ftd_dd_component: Dict, index: str) -> List:
        """Create FTD/DD Analysis section by embedding standalone FTD/DD PDF report."""
        story = []
        styles = getSampleStyleSheet()

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        story.append(Paragraph("<b>FOLLOW-THROUGH DAYS & DISTRIBUTION DAYS ANALYSIS</b>", header_style))

        # Check if standalone FTD/DD PDF report exists
        if ftd_dd_component.get('report_path') and Path(ftd_dd_component['report_path']).exists():
            try:
                # Embed the standalone FTD/DD PDF content as images
                embedded_content = self._embed_pdf_as_images(ftd_dd_component['report_path'])
                story.extend(embedded_content)
                logger.info(f"Successfully embedded FTD/DD PDF report: {ftd_dd_component['report_path']}")

            except Exception as e:
                logger.error(f"Error embedding FTD/DD PDF report: {e}")
                # Fallback to basic description
                story.append(Paragraph(
                    "Detailed FTD/DD analysis report could not be embedded. "
                    f"Please refer to the standalone report: {Path(ftd_dd_component['report_path']).name}",
                    styles['Normal']
                ))
        else:
            # Fallback content when no standalone report is available
            story.append(Paragraph(
                "<b>William J. O'Neil Market Timing Methodology</b> - Follow-Through Days confirm market uptrends "
                "while Distribution Days signal institutional selling pressure. This analysis provides critical "
                "market timing signals for trend identification.",
                styles['Normal']
            ))

            # Add chart if available as fallback
            if ftd_dd_component.get('chart_path'):
                try:
                    chart_image = Image(str(ftd_dd_component['chart_path']), width=8*inch, height=5*inch)
                    story.append(chart_image)
                except Exception as e:
                    logger.error(f"Error adding FTD/DD chart: {e}")
                    story.append(Paragraph("FTD/DD chart could not be loaded", styles['Normal']))

        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_combined_market_assessment(self, components: Dict, index: str) -> List:
        """Create combined market assessment from all available components."""
        story = []
        styles = getSampleStyleSheet()

        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            spaceAfter=15
        )

        story.append(Paragraph("<b>COMBINED MARKET ASSESSMENT</b>", header_style))

        # Create assessment content
        content_style = ParagraphStyle(
            'Assessment',
            parent=styles['Normal'],
            fontSize=12,
            textColor=self.colors['text'],
            alignment=TA_LEFT,
            spaceAfter=10
        )

        assessment_points = []

        # Component availability summary
        active_components = []
        if components['chillax']['available']:
            active_components.append("Chillax MAs")
        if components['ma_cycles']['available']:
            active_components.append("MA Cycles")
        if components['ftd_dd']['available']:
            active_components.append("FTD/DD Analysis")

        assessment_points.append(f"<b>📊 Active Components:</b> {', '.join(active_components) if active_components else 'None'}")

        # Market pulse methodology
        assessment_points.append("")
        assessment_points.append("<b>🎯 Market Pulse Methodology:</b>")
        assessment_points.append("• <b>Chillax MAs:</b> Trend confirmation through multiple timeframe moving average analysis")
        assessment_points.append("• <b>MA Cycles:</b> Bull/bear cycle identification and statistical pattern analysis")
        assessment_points.append("• <b>FTD/DD:</b> Institutional buying/selling pressure using O'Neil methodology")

        # Analysis recommendations
        assessment_points.append("")
        assessment_points.append("<b>📋 Analysis Recommendations:</b>")
        assessment_points.append("• Review each component for comprehensive market understanding")
        assessment_points.append("• Cross-validate signals across multiple indicators")
        assessment_points.append("• Consider timeframe consistency in decision making")
        assessment_points.append("• Monitor signal clustering for increased reliability")

        # Technical notes
        assessment_points.append("")
        assessment_points.append("<b>🔧 Technical Notes:</b>")
        assessment_points.append("• All charts use consistent data sources and timeframes")
        assessment_points.append("• Individual component reports available for detailed analysis")
        assessment_points.append("• Signal parameters follow established market timing methodologies")

        for point in assessment_points:
            story.append(Paragraph(point, content_style))

        return story

    def _embed_pdf_as_images(self, pdf_path: str) -> List:
        """
        Convert PDF pages to images and embed them in the report.

        Args:
            pdf_path: Path to the PDF file to embed

        Returns:
            List of ReportLab elements (Images) to add to the story
        """
        story = []

        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return story

            # Create temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)

                # Convert PDF to PNG images using pdftoppm
                # Command: pdftoppm -png -r 150 input.pdf output_prefix
                output_prefix = temp_dir_path / "page"
                cmd = [
                    'pdftoppm',
                    '-png',
                    '-r', '150',  # 150 DPI for good quality
                    str(pdf_path),
                    str(output_prefix)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    logger.error(f"PDF conversion failed: {result.stderr}")
                    return story

                # Find all generated PNG files and sort them
                png_files = sorted(temp_dir_path.glob("page-*.png"))

                if not png_files:
                    logger.warning(f"No PNG files generated from PDF: {pdf_path}")
                    return story

                logger.info(f"Generated {len(png_files)} images from PDF: {pdf_path.name}")

                # Add each page as an image to the story
                for i, png_file in enumerate(png_files):
                    try:
                        # Calculate optimal size to fit on page
                        # Letter size is 8.5" x 11", with margins we have about 7.5" x 9.5"
                        max_width = 7.5 * inch
                        max_height = 9.5 * inch

                        # Create image with automatic sizing
                        img = Image(str(png_file))

                        # Scale to fit within page bounds while maintaining aspect ratio
                        img_width, img_height = img.imageWidth, img.imageHeight

                        # Calculate scale factor
                        width_scale = max_width / img_width
                        height_scale = max_height / img_height
                        scale = min(width_scale, height_scale, 1.0)  # Don't upscale

                        # Apply scaling
                        final_width = img_width * scale
                        final_height = img_height * scale

                        img.drawWidth = final_width
                        img.drawHeight = final_height

                        story.append(img)

                        # Add page break between pages (except for the last page)
                        if i < len(png_files) - 1:
                            story.append(PageBreak())

                        logger.debug(f"Added page {i+1}/{len(png_files)} from embedded PDF")

                    except Exception as e:
                        logger.error(f"Error processing PNG file {png_file}: {e}")
                        continue

                logger.info(f"Successfully embedded {len(png_files)} pages from PDF")

        except subprocess.TimeoutExpired:
            logger.error(f"PDF conversion timeout for: {pdf_path}")
        except Exception as e:
            logger.error(f"Error embedding PDF {pdf_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return story


def generate_comprehensive_market_pulse_reports(market_pulse_dir: Path, reports_dir: Path,
                                              output_dir: Path, user_config) -> List[str]:
    """
    Generate comprehensive market pulse reports for all available combinations.

    Args:
        market_pulse_dir: Directory containing PNG charts
        reports_dir: Directory containing individual reports
        output_dir: Directory for comprehensive reports
        user_config: User configuration

    Returns:
        List of generated comprehensive report file paths
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_generator = ComprehensiveMarketPulseReportGenerator(user_config)

        generated_reports = []

        # Find all unique combinations from available files
        combinations = set()

        # Scan for Chillax files
        for file in market_pulse_dir.glob('chillax_mas_*.png'):
            parts = file.stem.split('_')
            if len(parts) >= 5:
                index, choice, timeframe, date = parts[2], parts[3], parts[4], parts[5]
                combinations.add((index, choice, timeframe, date))

        # Scan for MA Cycles files
        for file in market_pulse_dir.glob('ma_cycles_*.png'):
            parts = file.stem.split('_')
            if len(parts) >= 5:
                index, choice, timeframe, date = parts[2], parts[3], parts[4], parts[5]
                combinations.add((index, choice, timeframe, date))

        # Scan for FTD/DD files
        for file in market_pulse_dir.glob('ftd_dd_chart_*.png'):
            parts = file.stem.split('_')
            if len(parts) >= 6:
                index, choice, timeframe, date = parts[3], parts[4], parts[5], parts[6]
                combinations.add((index, choice, timeframe, date))

        logger.info(f"Found {len(combinations)} unique index/timeframe combinations")

        # Generate comprehensive report for each combination
        for index, choice, timeframe, date in sorted(combinations):
            try:
                # Generate comprehensive report
                report_name = f'market_pulse_{index}_{choice}_{timeframe}_{date}.pdf'
                report_path = output_dir / report_name

                success = report_generator.generate_comprehensive_report(
                    index=index,
                    choice=choice,
                    timeframe=timeframe,
                    data_date=date,
                    market_pulse_dir=market_pulse_dir,
                    reports_dir=reports_dir,
                    output_path=report_path
                )

                if success:
                    generated_reports.append(str(report_path))
                    logger.info(f"Generated comprehensive report: {report_name}")

            except Exception as e:
                logger.error(f"Error generating comprehensive report for {index}-{timeframe}: {e}")

        return generated_reports

    except Exception as e:
        logger.error(f"Error generating comprehensive market pulse reports: {e}")
        return []