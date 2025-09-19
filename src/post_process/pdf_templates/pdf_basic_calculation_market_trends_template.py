#!/usr/bin/env python3
"""
Market Trends PDF Template for Basic Calculation Data
=====================================================

Comprehensive market trends analysis PDF combining multiple chart types:
- Performance analysis across timeframes
- Sector and industry analysis
- Risk-return analysis
- Technical indicators
- Market structure insights
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
    create_top_performers_chart, create_performance_distribution_chart,
    create_multi_timeframe_comparison_chart, create_performance_momentum_chart,
    create_sector_performance_heatmap, create_sector_comparison_chart,
    create_industry_performance_ranking, create_risk_return_scatter,
    create_volatility_analysis, create_rsi_analysis_chart,
    create_market_cap_analysis, create_exchange_analysis,
    create_index_membership_analysis
)

logger = logging.getLogger(__name__)

def generate_pdf(df: pd.DataFrame, pdf_path: str, metadata: dict = None) -> bool:
    """
    Generate comprehensive market trends PDF from basic_calculation data.

    Args:
        df: Filtered DataFrame from post-process workflow
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Generating market trends PDF: {pdf_path}")

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

        # Title page
        content.extend(_create_title_section(df, metadata, title_style, styles))
        content.append(PageBreak())

        # Executive summary
        content.extend(_create_executive_summary(df, metadata, heading_style, styles))
        content.append(PageBreak())

        # Performance analysis section
        content.extend(_create_performance_section(df, metadata, heading_style, styles, png_dir, pdf_name))
        content.append(PageBreak())

        # Sector and industry analysis
        content.extend(_create_sector_industry_section(df, metadata, heading_style, styles, png_dir, pdf_name))
        content.append(PageBreak())

        # Risk analysis section
        content.extend(_create_risk_analysis_section(df, metadata, heading_style, styles, png_dir, pdf_name))
        content.append(PageBreak())

        # Technical analysis section
        content.extend(_create_technical_analysis_section(df, metadata, heading_style, styles, png_dir, pdf_name))
        content.append(PageBreak())

        # Market structure section
        content.extend(_create_market_structure_section(df, metadata, heading_style, styles, png_dir, pdf_name))

        # Build PDF
        doc.build(content)

        logger.info(f"Successfully generated market trends PDF: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating market trends PDF: {e}")
        return False

def _create_title_section(df: pd.DataFrame, metadata: dict, title_style, styles) -> list:
    """Create title page content."""
    content = []

    # Main title
    content.append(Paragraph("Market Trends Analysis Report", title_style))
    content.append(Spacer(1, 0.3*inch))

    # Subtitle with data info
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
    <b>Comprehensive Market Analysis</b><br/>
    Generated: {generation_date}<br/>
    Data Date: {data_date}<br/>
    Total Stocks Analyzed: {len(df):,}
    </para>
    """
    content.append(Paragraph(subtitle, styles['Normal']))
    content.append(Spacer(1, 0.5*inch))

    # Key statistics table
    key_stats = _calculate_key_statistics(df)
    if key_stats:
        content.append(Paragraph("<b>Key Market Statistics</b>", styles['Heading2']))
        content.append(Spacer(1, 0.1*inch))

        stats_data = [
            ['Metric', 'Value'],
            ['Total Market Cap', f"${key_stats.get('total_market_cap', 0):.1f}T"],
            ['Average Performance (1Y)', f"{key_stats.get('avg_performance', 0):.1f}%"],
            ['Median Performance (1Y)', f"{key_stats.get('median_performance', 0):.1f}%"],
            ['Number of Sectors', f"{key_stats.get('sector_count', 0)}"],
            ['Number of Industries', f"{key_stats.get('industry_count', 0)}"],
            ['Stocks Above 52W High', f"{key_stats.get('near_high_pct', 0):.1f}%"],
            ['Average RSI', f"{key_stats.get('avg_rsi', 0):.0f}"]
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

def _create_executive_summary(df: pd.DataFrame, metadata: dict, heading_style, styles) -> list:
    """Create executive summary section."""
    content = []

    content.append(Paragraph("Executive Summary", heading_style))
    content.append(Spacer(1, 0.1*inch))

    # Generate summary insights
    summary_text = _generate_summary_insights(df)
    for paragraph in summary_text:
        content.append(Paragraph(paragraph, styles['Normal']))
        content.append(Spacer(1, 0.1*inch))

    return content

def _create_performance_section(df: pd.DataFrame, metadata: dict, heading_style, styles, png_dir: Path, pdf_name: str) -> list:
    """Create performance analysis section with charts."""
    content = []

    content.append(Paragraph("Performance Analysis", heading_style))
    content.append(Spacer(1, 0.1*inch))

    # Generate performance charts
    charts_created = []

    # 1. Top performers chart
    top_perf_path = png_dir / f"{pdf_name}_top_performers.png"
    if create_top_performers_chart(df, 'daily_daily_yearly_252d_pct_change',
                                 str(top_perf_path), top_n=15):
        charts_created.append((str(top_perf_path), "Top 15 Performers (1-Year)"))

    # 2. Performance distribution
    perf_dist_path = png_dir / f"{pdf_name}_performance_distribution.png"
    if create_performance_distribution_chart(df, 'daily_daily_yearly_252d_pct_change',
                                            str(perf_dist_path)):
        charts_created.append((str(perf_dist_path), "Performance Distribution Analysis"))

    # 3. Multi-timeframe comparison
    multi_tf_path = png_dir / f"{pdf_name}_multi_timeframe.png"
    if create_multi_timeframe_comparison_chart(df, str(multi_tf_path)):
        charts_created.append((str(multi_tf_path), "Multi-Timeframe Performance Comparison"))

    # Add charts to content
    for chart_path, chart_title in charts_created:
        content.append(Paragraph(f"<b>{chart_title}</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        # Add chart image
        img = RLImage(chart_path, width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_sector_industry_section(df: pd.DataFrame, metadata: dict, heading_style, styles, png_dir: Path, pdf_name: str) -> list:
    """Create sector and industry analysis section."""
    content = []

    content.append(Paragraph("Sector & Industry Analysis", heading_style))
    content.append(Spacer(1, 0.1*inch))

    # Add basic text content for now
    content.append(Paragraph("Sector and industry analysis charts will be implemented in future versions.", styles['Normal']))
    content.append(Spacer(1, 0.2*inch))

    return content

def _create_risk_analysis_section(df: pd.DataFrame, metadata: dict, heading_style, styles, png_dir: Path, pdf_name: str) -> list:
    """Create risk analysis section."""
    content = []

    content.append(Paragraph("Risk Analysis", heading_style))
    content.append(Spacer(1, 0.1*inch))

    # Add basic text content for now
    content.append(Paragraph("Risk analysis charts will be implemented in future versions.", styles['Normal']))
    content.append(Spacer(1, 0.2*inch))

    return content

def _create_technical_analysis_section(df: pd.DataFrame, metadata: dict, heading_style, styles, png_dir: Path, pdf_name: str) -> list:
    """Create technical analysis section."""
    content = []
    content.append(Paragraph("Technical Analysis", heading_style))
    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("Technical analysis charts will be implemented in future versions.", styles['Normal']))
    content.append(Spacer(1, 0.2*inch))
    return content

def _create_market_structure_section(df: pd.DataFrame, metadata: dict, heading_style, styles, png_dir: Path, pdf_name: str) -> list:
    """Create market structure analysis section."""
    content = []
    content.append(Paragraph("Market Structure Analysis", heading_style))
    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("Market structure analysis charts will be implemented in future versions.", styles['Normal']))
    content.append(Spacer(1, 0.2*inch))
    return content

def _calculate_key_statistics(df: pd.DataFrame) -> dict:
    """Calculate key statistics for the title page."""
    try:
        stats = {}

        # Market cap statistics
        if 'market_cap' in df.columns:
            total_market_cap = df['market_cap'].sum() / 1e12  # Convert to trillions
            stats['total_market_cap'] = total_market_cap

        # Performance statistics
        if 'daily_daily_yearly_252d_pct_change' in df.columns:
            perf_data = df['daily_daily_yearly_252d_pct_change'].dropna()
            stats['avg_performance'] = perf_data.mean()
            stats['median_performance'] = perf_data.median()

        # Sector and industry counts
        if 'sector' in df.columns:
            stats['sector_count'] = df['sector'].nunique()
        if 'industry' in df.columns:
            stats['industry_count'] = df['industry'].nunique()

        # 52-week position statistics
        if 'daily_price_position_52w' in df.columns:
            near_high = (df['daily_price_position_52w'] > 0.8).sum()
            stats['near_high_pct'] = (near_high / len(df)) * 100

        # RSI statistics
        if 'daily_rsi_14' in df.columns:
            stats['avg_rsi'] = df['daily_rsi_14'].mean()

        return stats

    except Exception as e:
        logger.warning(f"Error calculating key statistics: {e}")
        return {}

def _generate_summary_insights(df: pd.DataFrame) -> list:
    """Generate executive summary text based on data analysis."""
    insights = []

    try:
        # Market performance insight
        if 'daily_daily_yearly_252d_pct_change' in df.columns:
            perf_data = df['daily_daily_yearly_252d_pct_change'].dropna()
            avg_perf = perf_data.mean()
            positive_pct = (perf_data > 0).mean() * 100

            insights.append(f"Market Performance: The average 1-year return across all analyzed stocks is {avg_perf:.1f}%, "
                           f"with {positive_pct:.0f}% of stocks showing positive returns.")

        # Sector performance insight
        if 'sector' in df.columns and 'daily_daily_yearly_252d_pct_change' in df.columns:
            sector_perf = df.groupby('sector')['daily_daily_yearly_252d_pct_change'].mean()
            best_sector = sector_perf.idxmax()
            best_perf = sector_perf.max()
            worst_sector = sector_perf.idxmin()
            worst_perf = sector_perf.min()

            insights.append(f"Sector Leadership: {best_sector} leads with {best_perf:.1f}% average performance, "
                           f"while {worst_sector} lags at {worst_perf:.1f}%.")

        # Risk insight
        if 'atr_pct' in df.columns:
            avg_volatility = df['atr_pct'].mean()
            high_vol_pct = (df['atr_pct'] > avg_volatility * 1.5).mean() * 100

            insights.append(f"Market Volatility: Average volatility (ATR) is {avg_volatility:.1f}%, "
                           f"with {high_vol_pct:.0f}% of stocks showing elevated volatility levels.")

        # Technical insight
        if 'daily_rsi_14' in df.columns:
            avg_rsi = df['daily_rsi_14'].mean()
            overbought_pct = (df['daily_rsi_14'] > 70).mean() * 100
            oversold_pct = (df['daily_rsi_14'] < 30).mean() * 100

            insights.append(f"Technical Conditions: Market RSI averages {avg_rsi:.0f}, "
                           f"with {overbought_pct:.0f}% of stocks overbought and {oversold_pct:.0f}% oversold.")

        if not insights:
            insights.append("This comprehensive market analysis provides insights into performance trends, "
                           "sector dynamics, risk characteristics, and technical conditions across the analyzed universe.")

    except Exception as e:
        logger.warning(f"Error generating summary insights: {e}")
        insights.append("Market analysis covers performance, sector trends, risk metrics, and technical indicators.")

    return insights




