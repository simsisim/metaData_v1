#!/usr/bin/env python3
"""
Dedicated Stage Analysis Report Generator.
Creates PNG charts and PDF reports specifically for stage analysis data.
Based on the original stage_analysis_helper/PP_stage_analysis.py pattern.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image as RLImage
from reportlab.lib.pagesizes import letter, landscape, A3
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import logging

logger = logging.getLogger(__name__)

class StageAnalysisReportGenerator:
    """
    Dedicated generator for stage analysis reports.
    Creates PNG pie charts and comprehensive PDF reports matching the original pattern.
    """

    def __init__(self, output_dir_reports="results/reports"):
        """
        Initialize the stage analysis report generator.

        Args:
            output_dir_reports: Directory for PDF output files
        """
        self.output_dir_reports = Path(output_dir_reports)
        self.output_dir_reports.mkdir(parents=True, exist_ok=True)

    def _detect_timeframe_columns(self, df: pd.DataFrame):
        """
        Detect timeframe-specific column names based on available columns.

        Returns:
            Tuple of (sa_name_col, sa_code_col, sa_color_col)
        """
        columns = df.columns.tolist()

        # Check for timeframe-specific columns
        if 'daily_sa_name' in columns:
            return 'daily_sa_name', 'daily_sa_code', 'daily_sa_color_code'
        elif 'weekly_sa_name' in columns:
            return 'weekly_sa_name', 'weekly_sa_code', 'weekly_sa_color_code'
        elif 'monthly_sa_name' in columns:
            return 'monthly_sa_name', 'monthly_sa_code', 'monthly_sa_color_code'
        else:
            raise ValueError(f"Could not find stage analysis columns in: {columns}")

    def create_stage_pie_chart(self, csv_path: str) -> str:
        """
        Create stage analysis pie chart from CSV data.

        Args:
            csv_path: Path to stage analysis CSV file

        Returns:
            Path to generated PNG file
        """
        try:
            # Read CSV data
            df = pd.read_csv(csv_path)

            # Generate PNG filename from CSV
            csv_stem = Path(csv_path).stem
            png_path = Path(csv_path).parent / f"{csv_stem}_pie_chart.png"

            # Detect timeframe and get appropriate column names
            sa_name_col, sa_code_col, sa_color_col = self._detect_timeframe_columns(df)

            # Prepare data for pie chart
            stage_counts = df[sa_name_col].value_counts().reset_index()
            stage_counts.columns = ['stage', 'count']

            # Color mapping for stages (matching original)
            color_map = {
                'Bullish Trend': '#388e3c',
                'Bullish Fade': '#FF9800',
                'Bearish Trend': '#F44336',
                'Bearish Confirmation': '#D32F2F',
                'Launch Pad': '#C2A7D4',
                'Pullback': '#D4C464',
                'Mean Reversion': '#C0AF53',
                'Upward Pivot': '#9E9E9E',
                'Undefined': '#757575',
                'Breakout Confirmation': '#8BC34A',
                'Fade Confirmation': '#FF5722'
            }

            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

            colors = [color_map.get(stage, '#999999') for stage in stage_counts['stage']]

            wedges, texts, autotexts = ax.pie(
                stage_counts['count'],
                labels=stage_counts['stage'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )

            ax.set_title('Market Stage Analysis Distribution', fontsize=16, fontweight='bold', pad=20)

            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)
                autotext.set_weight('bold')

            plt.tight_layout()
            plt.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=200)
            plt.close(fig)
            plt.clf()

            logger.info(f"Generated stage analysis pie chart: {png_path}")
            return str(png_path)

        except Exception as e:
            logger.error(f"Error creating stage pie chart from {csv_path}: {e}")
            raise

    def create_stage_report(self, csv_path: str, png_path: str) -> str:
        """
        Create comprehensive stage analysis PDF report.

        Args:
            csv_path: Path to stage analysis CSV file
            png_path: Path to pie chart PNG file

        Returns:
            Path to generated PDF file
        """
        try:
            # Read CSV data
            df = pd.read_csv(csv_path)

            # Generate PDF filename from CSV
            csv_stem = Path(csv_path).stem
            pdf_path = self.output_dir_reports / f"{csv_stem}.pdf"

            # Create PDF document with A3 landscape for maximum table width
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=landscape(A3),
                rightMargin=14, leftMargin=14, topMargin=24, bottomMargin=24
            )

            # Build story elements
            story = self._build_stage_analysis_story(df, png_path)

            # Build PDF
            doc.build(story)

            logger.info(f"Generated stage analysis PDF report: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Error creating stage analysis PDF for {csv_path}: {e}")
            raise

    def _build_stage_analysis_story(self, df: pd.DataFrame, png_path: str):
        """Build the PDF story elements for stage analysis."""
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Market Stage Analysis", styles['Title']))
        story.append(Spacer(1, 14))

        # Pie chart section
        story.append(Paragraph("Market Stage Analysis Distribution", styles['Heading2']))
        if Path(png_path).exists():
            # Increase height for better proportions and less stretching
            story.append(RLImage(png_path, width=5*inch, height=5*inch))
        else:
            story.append(Paragraph("Chart not available", styles['Normal']))
        story.append(Spacer(1, 18))

        # Stage Table by Market Stage (columnar format)
        story.append(Paragraph("Stage Table by Market Stage", styles['Heading2']))
        story.append(Spacer(1, 12))

        # Detect timeframe and get appropriate column names
        sa_name_col, sa_code_col, sa_color_col = self._detect_timeframe_columns(df)

        # Get unique stages and color codes for table
        stage_triplets = df[[sa_name_col, sa_code_col, sa_color_col]].drop_duplicates().values.tolist()

        # Build columns for each stage
        columns = []
        for sa_name, sa_code, sa_color in stage_triplets:
            tickers = df[df[sa_name_col] == sa_name]['ticker'].tolist()
            col = [sa_name, sa_code] + tickers
            columns.append((col, sa_color))

        if columns:
            # Pad columns to same length
            max_len = max(len(col) for col, _ in columns)
            for col, _ in columns:
                col += [''] * (max_len - len(col))

            # Transpose to create table data (rows)
            table_data = list(map(list, zip(*[col for col, _ in columns])))
            num_stages = len(columns)

            # Color mapping for backgrounds (exact match to original)
            CODE_TO_COLOR = {
                'green_light': colors.HexColor('#C8E6C9'),
                'orange_light': colors.HexColor('#FFE0B2'),
                'purple': colors.HexColor('#E1BEE7'),
                'red_light': colors.HexColor('#FFCDD2'),
                'red': colors.HexColor('#FFEBEE'),
                'yellow': colors.HexColor('#FFF9C4'),
                'yellow_light': colors.HexColor('#FFFDE7'),
                'gray': colors.HexColor('#F5F5F5'),
                'gray_light': colors.HexColor('#FAFAFA'),
                'lime': colors.HexColor('#F1F8E9'),
                'orange_dark': colors.HexColor('#FFE0B2'),
            }

            # Get background colors for each column
            col_bg_colors = [CODE_TO_COLOR.get(color_code, colors.white) for _, color_code in columns]

            # Create table style with improved formatting for landscape layout
            table_style = TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,1), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),  # Slightly smaller font for better fit
                ('WORDWRAP', (0,0), (-1,-1), True)  # Enable word wrapping
            ])

            # Add background colors for each column
            for col_idx, bg_color in enumerate(col_bg_colors):
                table_style.add('BACKGROUND', (col_idx, 0), (col_idx, max_len-1), bg_color)

            # Calculate dynamic column widths based on A3 landscape page width
            # A3 landscape: ~16.5" - margins = ~16" usable width
            available_width = 16 * inch
            min_col_width = 0.7 * inch  # Minimum readable width
            max_col_width = 2.0 * inch  # Maximum to prevent overly wide columns

            # Calculate optimal column width
            optimal_width = available_width / num_stages
            col_width = max(min_col_width, min(optimal_width, max_col_width))

            col_widths = [col_width] * num_stages
            stage_table = Table(table_data, colWidths=col_widths)
            stage_table.setStyle(table_style)
            story.append(stage_table)

        return story

    def process_stage_analysis(self, csv_path: str):
        """
        Complete processing: Stage Analysis CSV → PNG + PDF.

        Args:
            csv_path: Path to stage analysis CSV file

        Returns:
            Tuple of (png_path, pdf_path)
        """
        try:
            logger.info(f"Processing stage analysis: {csv_path}")

            # Generate PNG pie chart
            png_path = self.create_stage_pie_chart(csv_path)

            # Generate PDF report
            pdf_path = self.create_stage_report(csv_path, png_path)

            logger.info(f"Successfully processed stage analysis: PNG={png_path}, PDF={pdf_path}")
            return png_path, pdf_path

        except Exception as e:
            logger.error(f"Error processing stage analysis {csv_path}: {e}")
            raise

def process_stage_analysis_csv(csv_path: str, output_dir_reports: str = "results/reports"):
    """
    Convenience function to process a stage analysis CSV file.

    Args:
        csv_path: Path to stage analysis CSV file
        output_dir_reports: Output directory for PDF files

    Returns:
        Tuple of (png_path, pdf_path)
    """
    generator = StageAnalysisReportGenerator(output_dir_reports)
    return generator.process_stage_analysis(csv_path)