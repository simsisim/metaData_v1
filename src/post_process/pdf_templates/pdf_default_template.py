#!/usr/bin/env python3
"""
Default PDF Template for Post-Process Workflow
==============================================

Generic PDF template that works with any filtered DataFrame.
Provides basic data summary and statistics.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import logging

logger = logging.getLogger(__name__)

def generate_pdf(df: pd.DataFrame, output_pdf_path: str, metadata: dict = None) -> bool:
    """
    Generate generic PDF report from filtered DataFrame.

    Args:
        df: Filtered and sorted DataFrame
        output_pdf_path: Where to save the PDF
        metadata: Processing context and statistics

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Generating default PDF report: {output_pdf_path}")

        # Create PDF document
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=landscape(letter),
            rightMargin=14, leftMargin=14, topMargin=24, bottomMargin=24
        )

        # Build story elements
        story = _build_default_story(df, metadata)

        # Build PDF
        doc.build(story)

        logger.info(f"Successfully generated default PDF: {output_pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating default PDF: {e}")
        return False

def _build_default_story(df: pd.DataFrame, metadata: dict = None):
    """Build the PDF story elements for default template."""
    styles = getSampleStyleSheet()
    story = []

    # Title with metadata context
    if metadata:
        title_text = f"Data Summary Report ({metadata.get('original_rows', 0)} → {metadata.get('filtered_rows', 0)} rows)"
    else:
        title_text = f"Data Summary Report ({len(df)} rows)"

    story.append(Paragraph(title_text, styles['Title']))
    story.append(Spacer(1, 14))

    # Add processing info if available
    if metadata:
        info_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        info_text += f"Source: {metadata.get('original_filename', 'Unknown')}<br/>"
        info_text += f"File ID: {metadata.get('file_id', 'N/A')}<br/>"
        info_text += f"Filters Applied: {len(metadata.get('filter_operations', []))}<br/>"
        info_text += f"Sorts Applied: {len(metadata.get('sort_operations', []))}"
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 12))

    # Data Summary Section
    story.append(Paragraph("Data Summary", styles['Heading2']))
    story.append(Spacer(1, 8))

    # Basic statistics
    summary_data = [
        ['Metric', 'Value'],
        ['Total Rows', str(len(df))],
        ['Total Columns', str(len(df.columns))],
        ['Memory Usage', f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"]
    ]

    # Add column info
    if len(df.columns) <= 20:  # Only show columns if not too many
        summary_data.append(['Columns', ', '.join(df.columns[:10]) + ('...' if len(df.columns) > 10 else '')])

    summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 18))

    # Filter Summary if available
    if metadata and metadata.get('filter_operations'):
        story.append(Paragraph("Applied Filters", styles['Heading2']))
        story.append(Spacer(1, 8))

        filter_data = [['Column', 'Condition', 'Value', 'Logic']]
        for filter_op in metadata['filter_operations']:
            filter_data.append([
                str(filter_op.get('Column', '')),
                str(filter_op.get('Condition', '')),
                str(filter_op.get('Value', '')),
                str(filter_op.get('Logic', ''))
            ])

        filter_table = Table(filter_data, colWidths=[2*inch, 1.5*inch, 2*inch, 1*inch])
        filter_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ]))
        story.append(filter_table)
        story.append(Spacer(1, 18))

    # Sample Data (first 10 rows, first 8 columns for readability)
    if not df.empty:
        story.append(Paragraph("Sample Data (First 10 rows)", styles['Heading2']))
        story.append(Spacer(1, 8))

        # Limit columns for PDF readability
        display_cols = df.columns[:8].tolist()
        sample_df = df[display_cols].head(10)

        # Convert to table data
        table_data = [display_cols]  # Headers
        for _, row in sample_df.iterrows():
            table_data.append([str(val)[:20] + ('...' if len(str(val)) > 20 else '') for val in row.values])

        # Calculate column widths
        col_width = (10 * inch) / len(display_cols)
        col_widths = [col_width] * len(display_cols)

        data_table = Table(table_data, colWidths=col_widths)
        data_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('WORDWRAP', (0,0), (-1,-1), True)
        ]))
        story.append(data_table)

    return story