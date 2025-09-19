#!/usr/bin/env python3
"""
Sector Analysis PDF Template for Basic Calculation Data
=======================================================

Sector-focused analysis PDF template:
- Detailed sector performance analysis
- Cross-sector comparisons
- Sector rotation patterns
- Technical analysis by sector
- Top performers within sectors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Import chart generation functions
from ..png_templates.basic_calculation import (
    create_sector_performance_heatmap, create_sector_comparison_chart,
    create_sector_rotation_analysis, create_sector_technical_analysis,
    create_top_performers_chart
)

logger = logging.getLogger(__name__)

def generate_pdf(df: pd.DataFrame, pdf_path: str, metadata: dict = None) -> bool:
    """
    Generate sector-focused analysis PDF from basic_calculation data.

    Args:
        df: Filtered DataFrame from post-process workflow
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Generating sector analysis PDF: {pdf_path}")

        # Create document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                               rightMargin=0.5*inch, leftMargin=0.5*inch,
                               topMargin=0.5*inch, bottomMargin=0.5*inch)

        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=6
        )

        # Create permanent PNG directory
        png_dir = Path("results/post_process")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Get base filename from PDF path
        pdf_name = Path(pdf_path).stem

        # Build content
        content = []

        # Determine if this is a single sector analysis or multi-sector
        target_sector = _detect_target_sector(df, metadata)

        if target_sector:
            # Single sector deep dive
            content.extend(_create_single_sector_analysis(df, target_sector, metadata, title_style, heading_style, styles, png_dir, pdf_name))
        else:
            # Multi-sector comparison
            content.extend(_create_multi_sector_analysis(df, metadata, title_style, heading_style, styles, png_dir, pdf_name))

        # Build PDF
        doc.build(content)

        logger.info(f"Successfully generated sector analysis PDF: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating sector analysis PDF: {e}")
        return False

def _detect_target_sector(df: pd.DataFrame, metadata: dict) -> str:
    """Detect if analysis is focused on a specific sector."""
    try:
        if metadata and 'filter_operations' in metadata:
            for filter_op in metadata['filter_operations']:
                if filter_op.get('Column') == 'sector' and filter_op.get('Condition') == 'equals':
                    return filter_op.get('Value')

        # Check if 90%+ of stocks are from single sector
        if 'sector' in df.columns:
            sector_counts = df['sector'].value_counts()
            if len(sector_counts) > 0:
                dominant_sector_pct = sector_counts.iloc[0] / len(df)
                if dominant_sector_pct > 0.9:
                    return sector_counts.index[0]

        return None

    except Exception as e:
        logger.warning(f"Error detecting target sector: {e}")
        return None

def _create_single_sector_analysis(df: pd.DataFrame, target_sector: str, metadata: dict,
                                 title_style, heading_style, styles, png_dir: Path, pdf_name: str) -> list:
    """Create single sector deep-dive analysis."""
    content = []

    # Title page
    content.append(Paragraph(f"{target_sector} Sector Analysis", title_style))
    content.append(Spacer(1, 0.3*inch))

    # Sector overview
    sector_data = df[df['sector'] == target_sector] if 'sector' in df.columns else df
    content.extend(_create_sector_overview(sector_data, target_sector, styles, png_dir, pdf_name))
    content.append(PageBreak())

    # Performance analysis
    content.append(Paragraph("Performance Analysis", heading_style))
    content.extend(_create_sector_performance_analysis(sector_data, target_sector, styles, png_dir, pdf_name))
    content.append(PageBreak())

    # Industry breakdown within sector
    if 'industry' in df.columns:
        content.append(Paragraph("Industry Breakdown", heading_style))
        content.extend(_create_industry_breakdown_analysis(sector_data, target_sector, styles, png_dir, pdf_name))
        content.append(PageBreak())

    # Technical analysis
    content.append(Paragraph("Technical Analysis", heading_style))
    content.extend(_create_sector_technical_deep_dive(sector_data, target_sector, styles, png_dir, pdf_name))
    content.append(PageBreak())

    # Top performers
    content.append(Paragraph("Top Performers", heading_style))
    content.extend(_create_sector_top_performers(sector_data, target_sector, styles, png_dir, pdf_name))

    return content

def _create_multi_sector_analysis(df: pd.DataFrame, metadata: dict,
                                title_style, heading_style, styles, png_dir: Path, pdf_name: str) -> list:
    """Create multi-sector comparison analysis."""
    content = []

    # Title page
    content.append(Paragraph("Sector Comparison Analysis", title_style))
    content.append(Spacer(1, 0.3*inch))

    generation_date = datetime.now().strftime("%B %d, %Y")

    # Handle metadata timestamp (could be string or datetime)
    if metadata and 'processing_timestamp' in metadata:
        timestamp = metadata['processing_timestamp']
        if isinstance(timestamp, str):
            data_date = timestamp
        else:
            data_date = timestamp.strftime("%Y-%m-%d")
    else:
        data_date = "Unknown"
    subtitle = f"""
    <para align=center>
    <b>Cross-Sector Performance Analysis</b><br/>
    Generated: {generation_date}<br/>
    Total Stocks: {len(df):,}<br/>
    Sectors Analyzed: {df['sector'].nunique() if 'sector' in df.columns else 'N/A'}
    </para>
    """
    content.append(Paragraph(subtitle, styles['Normal']))
    content.append(PageBreak())

    # Sector performance overview
    content.append(Paragraph("Sector Performance Overview", heading_style))
    content.extend(_create_sector_performance_overview(df, styles))
    content.append(PageBreak())

    # Sector comparison charts
    content.append(Paragraph("Sector Performance Analysis", heading_style))
    content.extend(_create_sector_comparison_section(df, styles, png_dir, pdf_name))
    content.append(PageBreak())

    # Sector rotation analysis
    content.append(Paragraph("Sector Rotation Analysis", heading_style))
    content.extend(_create_sector_rotation_section(df, styles, png_dir, pdf_name))
    content.append(PageBreak())

    # Technical analysis by sector
    content.append(Paragraph("Technical Analysis by Sector", heading_style))
    content.extend(_create_multi_sector_technical_analysis(df, styles, png_dir, pdf_name))

    return content

def _create_sector_overview(df: pd.DataFrame, sector_name: str, styles, png_dir: Path, pdf_name: str) -> list:
    """Create sector overview section."""
    content = []

    # Key statistics
    stats = _calculate_sector_statistics(df)

    overview_text = f"""
    <b>Sector Overview: {sector_name}</b><br/><br/>
    This analysis covers {len(df):,} stocks in the {sector_name} sector, representing
    ${stats.get('total_market_cap', 0):.1f}B in total market capitalization.
    """

    content.append(Paragraph(overview_text, styles['Normal']))
    content.append(Spacer(1, 0.2*inch))

    # Statistics table
    if stats:
        stats_data = [
            ['Metric', 'Value'],
            ['Number of Stocks', f"{len(df):,}"],
            ['Total Market Cap', f"${stats.get('total_market_cap', 0):.1f}B"],
            ['Average Market Cap', f"${stats.get('avg_market_cap', 0):.1f}B"],
            ['Average 1Y Performance', f"{stats.get('avg_performance', 0):.1f}%"],
            ['Median 1Y Performance', f"{stats.get('median_performance', 0):.1f}%"],
            ['Performance Std Dev', f"{stats.get('perf_std', 0):.1f}%"],
            ['Average RSI', f"{stats.get('avg_rsi', 0):.0f}"],
            ['Average Volatility (ATR)', f"{stats.get('avg_volatility', 0):.1f}%"]
        ]

        stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(stats_table)

    return content

def _create_sector_performance_analysis(df: pd.DataFrame, sector_name: str, styles, png_dir: Path, pdf_name: str) -> list:
    """Create sector performance analysis with charts."""
    content = []

    # Use permanent PNG directory
    temp_path = png_dir

    # Top performers in sector
    top_perf_path = temp_path / f"{pdf_name}_sector_top_performers.png"
    if create_top_performers_chart(df, 'daily_daily_yearly_252d_pct_change',
                                 str(top_perf_path), top_n=15,
                                 title_suffix=f"in {sector_name}"):
        content.append(Paragraph("<b>Top Performers in Sector</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(top_perf_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_industry_breakdown_analysis(df: pd.DataFrame, sector_name: str, styles, png_dir: Path, pdf_name: str) -> list:
    """Create industry breakdown within sector."""
    content = []

    if 'industry' in df.columns:
        # Industry performance table
        industry_stats = df.groupby('industry').agg({
            'ticker': 'count',
            'daily_daily_yearly_252d_pct_change': ['mean', 'median', 'std'],
            'market_cap': 'sum'
        }).round(2)

        # Flatten column names
        industry_stats.columns = ['_'.join(col) if col[1] else col[0] for col in industry_stats.columns]
        industry_stats = industry_stats.sort_values('daily_daily_yearly_252d_pct_change_mean', ascending=False)

        if not industry_stats.empty:
            content.append(Paragraph(f"<b>Industry Performance within {sector_name}</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            # Create table data
            table_data = [['Industry', 'Stocks', 'Avg Return (%)', 'Market Cap ($B)']]

            for industry, row in industry_stats.head(10).iterrows():  # Top 10 industries
                industry_name = industry[:30] + '...' if len(industry) > 30 else industry
                stocks = int(row['ticker_count'])  # Fixed: use correct flattened column name
                avg_return = row['daily_daily_yearly_252d_pct_change_mean']
                market_cap = row['market_cap_sum'] / 1e9

                table_data.append([industry_name, str(stocks), f"{avg_return:.1f}%", f"${market_cap:.1f}B"])

            industry_table = Table(table_data, colWidths=[2.5*inch, 0.8*inch, 1*inch, 1*inch])
            industry_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(industry_table)

    return content

def _create_sector_technical_deep_dive(df: pd.DataFrame, sector_name: str, styles, png_dir: Path, pdf_name: str) -> list:
    """Create technical analysis deep dive for sector."""
    content = []

    # Technical summary
    tech_summary = _generate_technical_summary(df, sector_name)
    content.append(Paragraph(tech_summary, styles['Normal']))
    content.append(Spacer(1, 0.2*inch))

    return content

def _create_sector_top_performers(df: pd.DataFrame, sector_name: str, styles, png_dir: Path, pdf_name: str) -> list:
    """Create top performers section for sector."""
    content = []

    if 'daily_daily_yearly_252d_pct_change' in df.columns:
        top_performers = df.nlargest(20, 'daily_daily_yearly_252d_pct_change')

        if not top_performers.empty:
            # Create top performers table
            table_data = [['Rank', 'Ticker', '1Y Return (%)', 'Market Cap ($B)', 'RSI']]

            for i, (idx, stock) in enumerate(top_performers.iterrows(), 1):
                ticker = stock['ticker']
                performance = stock['daily_daily_yearly_252d_pct_change']
                market_cap = stock.get('market_cap', 0) / 1e9
                rsi = stock.get('daily_rsi_14', 0)

                table_data.append([
                    str(i),
                    ticker,
                    f"{performance:.1f}%",
                    f"${market_cap:.1f}B",
                    f"{rsi:.0f}"
                ])

            performers_table = Table(table_data, colWidths=[0.5*inch, 1*inch, 1*inch, 1.2*inch, 0.8*inch])
            performers_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(performers_table)

    return content

def _create_sector_performance_overview(df: pd.DataFrame, styles) -> list:
    """Create sector performance overview table."""
    content = []

    if 'sector' in df.columns and 'daily_daily_yearly_252d_pct_change' in df.columns:
        sector_performance = df.groupby('sector').agg({
            'ticker': 'count',
            'daily_daily_yearly_252d_pct_change': ['mean', 'median', 'std'],
            'market_cap': 'sum',
            'daily_rsi_14': 'mean'
        }).round(2)

        # Flatten column names
        sector_performance.columns = ['_'.join(col) if col[1] else col[0] for col in sector_performance.columns]
        sector_performance = sector_performance.sort_values('daily_daily_yearly_252d_pct_change_mean', ascending=False)

        if not sector_performance.empty:
            # Create table
            table_data = [['Sector', 'Stocks', 'Avg Return (%)', 'Volatility (%)', 'Market Cap ($B)', 'Avg RSI']]

            for sector, row in sector_performance.iterrows():
                sector_name = sector[:25] + '...' if len(sector) > 25 else sector
                stocks = int(row['ticker_count'])  # Fixed: use correct flattened column name
                avg_return = row['daily_daily_yearly_252d_pct_change_mean']
                volatility = row['daily_daily_yearly_252d_pct_change_std']
                market_cap = row['market_cap_sum'] / 1e9
                avg_rsi = row['daily_rsi_14_mean']

                table_data.append([
                    sector_name,
                    str(stocks),
                    f"{avg_return:.1f}%",
                    f"{volatility:.1f}%",
                    f"${market_cap:.1f}B",
                    f"{avg_rsi:.0f}"
                ])

            sector_table = Table(table_data, colWidths=[2*inch, 0.6*inch, 0.8*inch, 0.8*inch, 1*inch, 0.6*inch])
            sector_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(sector_table)

    return content

def _create_sector_comparison_section(df: pd.DataFrame, styles, png_dir: Path, pdf_name: str) -> list:
    """Create sector comparison charts section."""
    content = []

    # Use permanent PNG directory
    temp_path = png_dir

    charts_created = []

    # Sector comparison chart
    sector_comp_path = temp_path / f"{pdf_name}_sector_comparison.png"
    if create_sector_comparison_chart(df, str(sector_comp_path)):
        charts_created.append((str(sector_comp_path), "Sector Comparison Analysis"))

    # Sector heatmap
    heatmap_path = temp_path / f"{pdf_name}_sector_heatmap.png"
    timeframe_cols = ['daily_daily_daily_1d_pct_change', 'daily_daily_weekly_7d_pct_change',
                     'daily_daily_monthly_22d_pct_change', 'daily_daily_yearly_252d_pct_change']
    if create_sector_performance_heatmap(df, timeframe_cols, str(heatmap_path)):
        charts_created.append((str(heatmap_path), "Sector Performance Heatmap"))

    # Add charts to content
    for chart_path, chart_title in charts_created:
        content.append(Paragraph(f"<b>{chart_title}</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(chart_path, width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_sector_rotation_section(df: pd.DataFrame, styles, png_dir: Path, pdf_name: str) -> list:
    """Create sector rotation analysis section."""
    content = []

    # Use permanent PNG directory
    temp_path = png_dir

    # Sector rotation chart
    rotation_path = temp_path / f"{pdf_name}_sector_rotation.png"
    if create_sector_rotation_analysis(df, str(rotation_path)):
        content.append(Paragraph("<b>Sector Rotation Analysis</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(rotation_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_multi_sector_technical_analysis(df: pd.DataFrame, styles, png_dir: Path, pdf_name: str) -> list:
    """Create technical analysis section for multiple sectors."""
    content = []

    # Use permanent PNG directory
    temp_path = png_dir

    # Sector technical analysis
    tech_path = temp_path / f"{pdf_name}_sector_technical.png"
    if create_sector_technical_analysis(df, str(tech_path)):
        content.append(Paragraph("<b>Technical Analysis by Sector</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(tech_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _calculate_sector_statistics(df: pd.DataFrame) -> dict:
    """Calculate sector-specific statistics."""
    try:
        stats = {}

        if 'market_cap' in df.columns:
            stats['total_market_cap'] = df['market_cap'].sum() / 1e9
            stats['avg_market_cap'] = df['market_cap'].mean() / 1e9

        if 'daily_daily_yearly_252d_pct_change' in df.columns:
            perf_data = df['daily_daily_yearly_252d_pct_change'].dropna()
            stats['avg_performance'] = perf_data.mean()
            stats['median_performance'] = perf_data.median()
            stats['perf_std'] = perf_data.std()

        if 'daily_rsi_14' in df.columns:
            stats['avg_rsi'] = df['daily_rsi_14'].mean()

        if 'atr_pct' in df.columns:
            stats['avg_volatility'] = df['atr_pct'].mean()

        return stats

    except Exception as e:
        logger.warning(f"Error calculating sector statistics: {e}")
        return {}

def _generate_technical_summary(df: pd.DataFrame, sector_name: str) -> str:
    """Generate technical analysis summary text."""
    try:
        summary_parts = []

        # RSI analysis
        if 'daily_rsi_14' in df.columns:
            avg_rsi = df['daily_rsi_14'].mean()
            overbought_pct = (df['daily_rsi_14'] > 70).mean() * 100
            oversold_pct = (df['daily_rsi_14'] < 30).mean() * 100

            rsi_condition = "neutral"
            if avg_rsi > 60:
                rsi_condition = "slightly overbought"
            elif avg_rsi < 40:
                rsi_condition = "slightly oversold"

            summary_parts.append(f"The {sector_name} sector shows {rsi_condition} conditions with an average RSI of {avg_rsi:.0f}. "
                                f"{overbought_pct:.0f}% of stocks are overbought (RSI > 70) and {oversold_pct:.0f}% are oversold (RSI < 30).")

        # Momentum analysis
        if 'daily_momentum_20' in df.columns:
            avg_momentum = df['daily_momentum_20'].mean()
            positive_momentum_pct = (df['daily_momentum_20'] > 0).mean() * 100

            momentum_trend = "positive" if avg_momentum > 0 else "negative"
            summary_parts.append(f"Momentum indicators show a {momentum_trend} trend with {positive_momentum_pct:.0f}% of stocks displaying positive momentum.")

        # Volatility analysis
        if 'atr_pct' in df.columns:
            avg_volatility = df['atr_pct'].mean()
            summary_parts.append(f"Average volatility (ATR) in the sector is {avg_volatility:.1f}%.")

        if not summary_parts:
            summary_parts.append(f"Technical analysis for the {sector_name} sector indicates mixed conditions across momentum and volatility indicators.")

        return " ".join(summary_parts)

    except Exception as e:
        logger.warning(f"Error generating technical summary: {e}")
        return f"Technical analysis summary for {sector_name} sector."