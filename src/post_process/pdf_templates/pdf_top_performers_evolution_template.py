#!/usr/bin/env python3
"""
Top Performers Evolution PDF Template
====================================

Multi-timeframe performance tracking template that identifies top leaders
across different periods and tracks their evolution with technical indicators.

Features:
- Multi-timeframe performance tables
- Evolution tracking matrices
- Technical strength analysis
- Performance consistency scoring
- Visual heatmaps and charts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

def generate_pdf(df: pd.DataFrame, pdf_path: str, metadata: dict = None) -> bool:
    """
    Generate Top Performers Evolution PDF report.

    Args:
        df: DataFrame with basic_calculation data
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=0.75*inch, bottomMargin=0.5*inch)

        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                   fontSize=16, spaceAfter=20, textColor=colors.darkblue,
                                   alignment=TA_CENTER)

        # Build content
        content = []

        # Title
        title_text = f"Top Performers Evolution Analysis"
        if metadata and 'original_filename' in metadata:
            title_text += f" - {metadata['original_filename']}"

        content.append(Paragraph(title_text, title_style))
        content.append(Spacer(1, 0.3*inch))

        # Executive Summary
        content.extend(_create_executive_summary(df, styles))
        content.append(PageBreak())

        # Multi-Timeframe Leaders Table
        content.extend(_create_multi_timeframe_leaders_section(df, styles))
        content.append(PageBreak())

        # Performance Evolution Heatmap
        content.extend(_create_performance_evolution_heatmap_section(df, styles))
        content.append(PageBreak())

        # Technical Strength Analysis
        content.extend(_create_technical_strength_section(df, styles))

        # Build PDF
        doc.build(content)

        logger.info(f"Successfully generated Top Performers Evolution PDF: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating Top Performers Evolution PDF: {e}")
        return False

def _get_timeframe_columns(df: pd.DataFrame) -> dict:
    """Get available timeframe performance columns."""
    timeframe_mapping = {
        '1D': 'daily_daily_daily_1d_pct_change',
        '5D': 'daily_daily_daily_5d_pct_change',
        '1W': 'daily_daily_weekly_7d_pct_change',
        '2W': 'daily_daily_weekly_14d_pct_change',
        '1M': 'daily_daily_monthly_22d_pct_change',
        '2M': 'daily_daily_monthly_44d_pct_change',
        '1Q': 'daily_daily_quarterly_66d_pct_change',
        '2Q': 'daily_daily_quarterly_132d_pct_change',
        'YTD': 'daily_daily_yearly_252d_pct_change'
    }

    # Filter to only available columns
    available_timeframes = {}
    for label, col in timeframe_mapping.items():
        if col in df.columns:
            available_timeframes[label] = col

    return available_timeframes

def _calculate_top_performers(df: pd.DataFrame, timeframes: dict, top_n: int = 20) -> pd.DataFrame:
    """Calculate top performers across all timeframes."""

    # Calculate ranks for each timeframe
    rank_data = df[['ticker']].copy()

    for label, col in timeframes.items():
        if col in df.columns and not df[col].isna().all():
            rank_data[f'{label}_rank'] = df[col].rank(ascending=False, na_option='bottom')
            rank_data[f'{label}_pct'] = df[col]

    # Calculate average rank (lower is better)
    rank_cols = [col for col in rank_data.columns if col.endswith('_rank')]
    if rank_cols:
        rank_data['avg_rank'] = rank_data[rank_cols].mean(axis=1, skipna=True)

        # Calculate consistency score (how often in top quartile)
        consistency_scores = []
        for idx, row in rank_data.iterrows():
            ranks = [row[col] for col in rank_cols if not pd.isna(row[col])]
            if ranks:
                top_quartile_count = sum(1 for r in ranks if r <= len(df) * 0.25)
                consistency_scores.append((top_quartile_count / len(ranks)) * 100)
            else:
                consistency_scores.append(0)

        rank_data['consistency_score'] = consistency_scores

        # Get top performers by average rank
        top_performers = rank_data.nsmallest(top_n, 'avg_rank')

        return top_performers

    return pd.DataFrame()

def _create_executive_summary(df: pd.DataFrame, styles) -> list:
    """Create executive summary section."""
    content = []

    content.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
    content.append(Spacer(1, 0.2*inch))

    # Get timeframes and calculate top performers
    timeframes = _get_timeframe_columns(df)

    if not timeframes:
        content.append(Paragraph("No timeframe performance data available.", styles['Normal']))
        return content

    top_performers = _calculate_top_performers(df, timeframes, top_n=10)

    if top_performers.empty:
        content.append(Paragraph("No top performers data available.", styles['Normal']))
        return content

    # Create summary table
    table_data = [['Rank', 'Ticker', 'Avg Rank', 'Consistency Score (%)', 'Best Timeframe', 'Best Performance (%)']]

    for idx, (_, row) in enumerate(top_performers.head(10).iterrows(), 1):
        ticker = row['ticker']
        avg_rank = f"{row['avg_rank']:.1f}"
        consistency = f"{row['consistency_score']:.0f}%"

        # Find best performing timeframe
        pct_cols = [(col, row[col]) for col in row.index if col.endswith('_pct') and not pd.isna(row[col])]
        if pct_cols:
            best_timeframe, best_pct = max(pct_cols, key=lambda x: x[1])
            best_tf_label = best_timeframe.replace('_pct', '')
            best_performance = f"{best_pct:.1f}%"
        else:
            best_tf_label = "N/A"
            best_performance = "N/A"

        table_data.append([str(idx), ticker, avg_rank, consistency, best_tf_label, best_performance])

    # Create table
    summary_table = Table(table_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    content.append(summary_table)
    content.append(Spacer(1, 0.3*inch))

    # Key insights
    if not top_performers.empty:
        top_ticker = top_performers.iloc[0]['ticker']
        top_consistency = top_performers.iloc[0]['consistency_score']

        insights_text = f"""
        <b>Key Insights:</b><br/>
        • <b>{top_ticker}</b> leads with highest consistency across timeframes<br/>
        • Top performer maintains <b>{top_consistency:.0f}%</b> consistency score<br/>
        • Analysis covers {len(timeframes)} different time periods<br/>
        • {len(df)} total stocks analyzed for performance evolution
        """

        content.append(Paragraph(insights_text, styles['Normal']))

    return content

def _create_multi_timeframe_leaders_section(df: pd.DataFrame, styles) -> list:
    """Create detailed multi-timeframe leaders section."""
    content = []

    content.append(Paragraph("<b>Multi-Timeframe Leaders Analysis</b>", styles['Heading2']))
    content.append(Spacer(1, 0.2*inch))

    timeframes = _get_timeframe_columns(df)
    if not timeframes:
        content.append(Paragraph("No timeframe data available.", styles['Normal']))
        return content

    top_performers = _calculate_top_performers(df, timeframes, top_n=15)

    if top_performers.empty:
        content.append(Paragraph("No top performers data available.", styles['Normal']))
        return content

    # Create comprehensive table with all timeframes
    headers = ['Ticker']
    for tf_label in timeframes.keys():
        headers.extend([f'{tf_label} %', f'{tf_label} Rank'])
    headers.extend(['Avg Rank', 'Consistency'])

    table_data = [headers]

    for _, row in top_performers.head(15).iterrows():
        row_data = [row['ticker']]

        for tf_label in timeframes.keys():
            pct_val = row.get(f'{tf_label}_pct', 0)
            rank_val = row.get(f'{tf_label}_rank', 999)

            if pd.isna(pct_val):
                row_data.extend(['N/A', 'N/A'])
            else:
                row_data.extend([f"{pct_val:.1f}", f"{int(rank_val)}"])

        row_data.extend([f"{row['avg_rank']:.1f}", f"{row['consistency_score']:.0f}%"])
        table_data.append(row_data)

    # Calculate dynamic column widths
    num_timeframes = len(timeframes)
    ticker_width = 0.5 * inch
    summary_width = 0.4 * inch  # Avg Rank + Consistency
    available_width = 7.0 * inch - ticker_width - (2 * summary_width)
    col_width = available_width / (num_timeframes * 2)  # 2 columns per timeframe

    # Build column widths array
    col_widths = [ticker_width]
    for _ in range(num_timeframes):
        col_widths.extend([col_width, col_width])
    col_widths.extend([summary_width, summary_width])

    perf_table = Table(table_data, colWidths=col_widths)

    # Table style with very small fonts to fit all data
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 5),  # Very small header font
        ('FONTSIZE', (0, 1), (-1, -1), 4),  # Very small data font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 1),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),  # Thinner grid lines
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]

    perf_table.setStyle(TableStyle(table_style))
    content.append(perf_table)

    return content

def _create_performance_evolution_heatmap_section(df: pd.DataFrame, styles) -> list:
    """Create performance evolution heatmap section."""
    content = []

    content.append(Paragraph("<b>Performance Evolution Heatmap</b>", styles['Heading2']))
    content.append(Spacer(1, 0.2*inch))

    timeframes = _get_timeframe_columns(df)
    if not timeframes:
        content.append(Paragraph("No timeframe data available for heatmap.", styles['Normal']))
        return content

    top_performers = _calculate_top_performers(df, timeframes, top_n=20)

    if top_performers.empty:
        content.append(Paragraph("No performance data available for heatmap.", styles['Normal']))
        return content

    # Create heatmap data
    heatmap_data = []
    tickers = []

    for _, row in top_performers.head(20).iterrows():
        ticker = row['ticker']
        tickers.append(ticker)

        perf_row = []
        for tf_label in timeframes.keys():
            pct_val = row.get(f'{tf_label}_pct', 0)
            perf_row.append(pct_val if not pd.isna(pct_val) else 0)

        heatmap_data.append(perf_row)

    if heatmap_data:
        # Use permanent PNG directory
        png_dir = Path("results/post_process")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Create heatmap
        heatmap_path = png_dir / "performance_evolution_heatmap.png"
        if _create_heatmap_chart(heatmap_data, tickers, list(timeframes.keys()), str(heatmap_path)):
            content.append(Paragraph("<b>Performance Evolution Heatmap</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            img = RLImage(str(heatmap_path), width=6*inch, height=4*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

    return content

def _create_technical_strength_section(df: pd.DataFrame, styles) -> list:
    """Create technical strength analysis section."""
    content = []

    content.append(Paragraph("<b>Technical Strength Analysis</b>", styles['Heading2']))
    content.append(Spacer(1, 0.2*inch))

    timeframes = _get_timeframe_columns(df)
    top_performers = _calculate_top_performers(df, timeframes, top_n=15)

    if top_performers.empty:
        content.append(Paragraph("No technical data available.", styles['Normal']))
        return content

    # Technical indicators to include
    tech_indicators = {
        'RSI': 'daily_rsi_14',
        'MACD': 'daily_macd',
        'Distance SMA20': 'daily_price2_sma20pct',
        'Distance SMA50': 'daily_price2_sma50pct',
        'Price Position 52W': 'daily_price_position_52w'
    }

    # Create technical strength table
    headers = ['Ticker', 'Current Price']
    for tech_name in tech_indicators.keys():
        if tech_indicators[tech_name] in df.columns:
            headers.append(tech_name)
    headers.append('Tech Score')

    table_data = [headers]

    for _, perf_row in top_performers.head(15).iterrows():
        ticker = perf_row['ticker']

        # Get ticker data from original dataframe
        ticker_data = df[df['ticker'] == ticker]
        if ticker_data.empty:
            continue

        ticker_data = ticker_data.iloc[0]

        row_data = [ticker]

        # Current price
        current_price = ticker_data.get('current_price', 0)
        row_data.append(f"${current_price:.2f}" if not pd.isna(current_price) else "N/A")

        # Technical indicators
        tech_values = []
        for tech_name, tech_col in tech_indicators.items():
            if tech_col in df.columns:
                tech_val = ticker_data.get(tech_col, 0)
                if not pd.isna(tech_val):
                    if 'pct' in tech_col or 'position' in tech_col:
                        row_data.append(f"{tech_val:.1f}%")
                        tech_values.append(abs(tech_val) if tech_val > 0 else 0)
                    else:
                        row_data.append(f"{tech_val:.2f}")
                        # Normalize RSI and MACD for scoring
                        if 'rsi' in tech_col:
                            tech_values.append(tech_val if 30 <= tech_val <= 70 else abs(50 - tech_val))
                        else:
                            tech_values.append(abs(tech_val))
                else:
                    row_data.append("N/A")

        # Calculate composite technical score
        if tech_values:
            tech_score = np.mean(tech_values)
            row_data.append(f"{tech_score:.0f}")
        else:
            row_data.append("N/A")

        table_data.append(row_data)

    # Create technical table
    tech_table = Table(table_data)
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    content.append(tech_table)

    return content

def _create_heatmap_chart(data: list, tickers: list, timeframes: list, output_path: str) -> bool:
    """Create performance evolution heatmap chart."""
    try:
        # Convert to numpy array
        data_array = np.array(data)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap with custom colormap
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        im = ax.imshow(data_array, cmap=cmap, aspect='auto')

        # Set ticks and labels
        ax.set_xticks(range(len(timeframes)))
        ax.set_xticklabels(timeframes, rotation=45)
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance (%)', rotation=270, labelpad=15)

        # Add text annotations
        for i in range(len(tickers)):
            for j in range(len(timeframes)):
                text = ax.text(j, i, f'{data_array[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=6)

        # Set title and labels
        ax.set_title('Top Performers Evolution Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Periods', fontsize=12)
        ax.set_ylabel('Top Performers', fontsize=12)

        # Adjust layout
        plt.tight_layout()

        # Save chart
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created performance heatmap: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        return False