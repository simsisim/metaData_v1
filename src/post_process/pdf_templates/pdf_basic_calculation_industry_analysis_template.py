#!/usr/bin/env python3
"""
Industry Analysis PDF Template for Basic Calculation Data
=========================================================

Industry-focused analysis PDF template:
- Industry performance rankings
- Industry vs sector comparisons
- Industry concentration analysis
- Top performers by industry
- Industry technical characteristics
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
    create_industry_performance_ranking, create_industry_sector_comparison,
    create_industry_concentration_analysis, create_industry_top_performers_grid
)

logger = logging.getLogger(__name__)

def generate_pdf(df: pd.DataFrame, pdf_path: str, metadata: dict = None) -> bool:
    """
    Generate industry-focused analysis PDF from basic_calculation data.

    Args:
        df: Filtered DataFrame from post-process workflow
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Generating industry analysis PDF: {pdf_path}")

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

        # Build content
        content = []

        # Determine if this is a single industry analysis or multi-industry
        target_industry = _detect_target_industry(df, metadata)

        if target_industry:
            # Single industry deep dive
            content.extend(_create_single_industry_analysis(df, target_industry, metadata, title_style, heading_style, styles))
        else:
            # Multi-industry comparison
            content.extend(_create_multi_industry_analysis(df, metadata, title_style, heading_style, styles))

        # Build PDF
        doc.build(content)

        logger.info(f"Successfully generated industry analysis PDF: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating industry analysis PDF: {e}")
        return False

def _detect_target_industry(df: pd.DataFrame, metadata: dict) -> str:
    """Detect if analysis is focused on a specific industry."""
    try:
        if metadata and 'filter_operations' in metadata:
            for filter_op in metadata['filter_operations']:
                if filter_op.get('Column') == 'industry' and filter_op.get('Condition') == 'equals':
                    return filter_op.get('Value')

        # Check if 90%+ of stocks are from single industry
        if 'industry' in df.columns:
            industry_counts = df['industry'].value_counts()
            if len(industry_counts) > 0:
                dominant_industry_pct = industry_counts.iloc[0] / len(df)
                if dominant_industry_pct > 0.9:
                    return industry_counts.index[0]

        return None

    except Exception as e:
        logger.warning(f"Error detecting target industry: {e}")
        return None

def _create_single_industry_analysis(df: pd.DataFrame, target_industry: str, metadata: dict,
                                   title_style, heading_style, styles) -> list:
    """Create single industry deep-dive analysis."""
    content = []

    # Title page
    industry_title = target_industry[:50] + "..." if len(target_industry) > 50 else target_industry
    content.append(Paragraph(f"{industry_title} Industry Analysis", title_style))
    content.append(Spacer(1, 0.3*inch))

    # Industry overview
    industry_data = df[df['industry'] == target_industry] if 'industry' in df.columns else df
    content.extend(_create_industry_overview(industry_data, target_industry, styles))
    content.append(PageBreak())

    # Performance analysis
    content.append(Paragraph("Performance Analysis", heading_style))
    content.extend(_create_industry_performance_analysis(industry_data, target_industry, styles))
    content.append(PageBreak())

    # Competitive landscape
    content.append(Paragraph("Competitive Landscape", heading_style))
    content.extend(_create_competitive_landscape_analysis(industry_data, target_industry, styles))
    content.append(PageBreak())

    # Risk and valuation metrics
    content.append(Paragraph("Risk and Valuation", heading_style))
    content.extend(_create_industry_risk_analysis(industry_data, target_industry, styles))
    content.append(PageBreak())

    # Top companies
    content.append(Paragraph("Leading Companies", heading_style))
    content.extend(_create_industry_leaders_analysis(industry_data, target_industry, styles))

    return content

def _create_multi_industry_analysis(df: pd.DataFrame, metadata: dict,
                                  title_style, heading_style, styles) -> list:
    """Create multi-industry comparison analysis."""
    content = []

    # Title page
    content.append(Paragraph("Industry Analysis Report", title_style))
    content.append(Spacer(1, 0.3*inch))

    generation_date = datetime.now().strftime("%B %d, %Y")
    subtitle = f"""
    <para align=center>
    <b>Industry Performance & Analysis</b><br/>
    Generated: {generation_date}<br/>
    Total Stocks: {len(df):,}<br/>
    Industries Analyzed: {df['industry'].nunique() if 'industry' in df.columns else 'N/A'}
    </para>
    """
    content.append(Paragraph(subtitle, styles['Normal']))
    content.append(PageBreak())

    # Industry performance overview
    content.append(Paragraph("Industry Performance Rankings", heading_style))
    content.extend(_create_industry_rankings_section(df, styles))
    content.append(PageBreak())

    # Industry vs sector analysis
    content.append(Paragraph("Industry vs Sector Analysis", heading_style))
    content.extend(_create_industry_sector_section(df, styles))
    content.append(PageBreak())

    # Industry concentration
    content.append(Paragraph("Industry Concentration Analysis", heading_style))
    content.extend(_create_industry_concentration_section(df, styles))
    content.append(PageBreak())

    # Top performers by industry
    content.append(Paragraph("Top Performers by Industry", heading_style))
    content.extend(_create_top_performers_by_industry_section(df, styles))

    return content

def _create_industry_overview(df: pd.DataFrame, industry_name: str, styles) -> list:
    """Create industry overview section."""
    content = []

    # Key statistics
    stats = _calculate_industry_statistics(df)

    overview_text = f"""
    <b>Industry Overview: {industry_name}</b><br/><br/>
    This analysis covers {len(df):,} stocks in the {industry_name} industry, representing
    ${stats.get('total_market_cap', 0):.1f}B in total market capitalization.
    """

    content.append(Paragraph(overview_text, styles['Normal']))
    content.append(Spacer(1, 0.2*inch))

    # Statistics table
    if stats:
        stats_data = [
            ['Metric', 'Value'],
            ['Number of Companies', f"{len(df):,}"],
            ['Total Market Cap', f"${stats.get('total_market_cap', 0):.1f}B"],
            ['Largest Company Market Cap', f"${stats.get('largest_company_cap', 0):.1f}B"],
            ['Average Market Cap', f"${stats.get('avg_market_cap', 0):.1f}B"],
            ['Median Market Cap', f"${stats.get('median_market_cap', 0):.1f}B"],
            ['Industry HHI (Concentration)', f"{stats.get('hhi', 0):.3f}"],
            ['Average 1Y Performance', f"{stats.get('avg_performance', 0):.1f}%"],
            ['Performance Range', f"{stats.get('perf_range', 0):.1f}%"],
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

def _create_industry_performance_analysis(df: pd.DataFrame, industry_name: str, styles) -> list:
    """Create industry performance analysis section."""
    content = []

    # Performance summary
    if 'daily_daily_yearly_252d_pct_change' in df.columns:
        perf_data = df['daily_daily_yearly_252d_pct_change'].dropna()

        if not perf_data.empty:
            avg_perf = perf_data.mean()
            median_perf = perf_data.median()
            std_perf = perf_data.std()
            min_perf = perf_data.min()
            max_perf = perf_data.max()
            positive_pct = (perf_data > 0).mean() * 100

            perf_summary = f"""
            <b>Performance Summary:</b><br/>
            The {industry_name} industry shows an average 1-year return of {avg_perf:.1f}% with a median of {median_perf:.1f}%.
            Performance ranges from {min_perf:.1f}% to {max_perf:.1f}%, with {positive_pct:.0f}% of companies showing positive returns.
            The standard deviation of {std_perf:.1f}% indicates {'high' if std_perf > 50 else 'moderate' if std_perf > 25 else 'low'} volatility within the industry.
            """

            content.append(Paragraph(perf_summary, styles['Normal']))
            content.append(Spacer(1, 0.2*inch))

    # Performance distribution table
    if 'daily_daily_yearly_252d_pct_change' in df.columns:
        # Create performance quartiles
        perf_quartiles = df['daily_daily_yearly_252d_pct_change'].quantile([0.25, 0.5, 0.75])

        quartile_data = [
            ['Performance Quartile', 'Threshold', 'Companies in Quartile'],
            ['Top 25%', f"> {perf_quartiles[0.75]:.1f}%", f"{((df['daily_daily_yearly_252d_pct_change'] > perf_quartiles[0.75]).sum())}"],
            ['2nd Quartile', f"{perf_quartiles[0.5]:.1f}% to {perf_quartiles[0.75]:.1f}%", f"{((df['daily_daily_yearly_252d_pct_change'] >= perf_quartiles[0.5]) & (df['daily_daily_yearly_252d_pct_change'] <= perf_quartiles[0.75])).sum()}"],
            ['3rd Quartile', f"{perf_quartiles[0.25]:.1f}% to {perf_quartiles[0.5]:.1f}%", f"{((df['daily_daily_yearly_252d_pct_change'] >= perf_quartiles[0.25]) & (df['daily_daily_yearly_252d_pct_change'] < perf_quartiles[0.5])).sum()}"],
            ['Bottom 25%', f"< {perf_quartiles[0.25]:.1f}%", f"{((df['daily_daily_yearly_252d_pct_change'] < perf_quartiles[0.25]).sum())}"]
        ]

        quartile_table = Table(quartile_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
        quartile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(quartile_table)

    return content

def _create_competitive_landscape_analysis(df: pd.DataFrame, industry_name: str, styles) -> list:
    """Create competitive landscape analysis."""
    content = []

    if 'market_cap' in df.columns:
        # Market share analysis
        total_industry_cap = df['market_cap'].sum()
        df_sorted = df.sort_values('market_cap', ascending=False)

        # Calculate market concentration
        top_3_share = df_sorted.head(3)['market_cap'].sum() / total_industry_cap * 100
        top_5_share = df_sorted.head(5)['market_cap'].sum() / total_industry_cap * 100
        top_10_share = df_sorted.head(10)['market_cap'].sum() / total_industry_cap * 100

        concentration_text = f"""
        <b>Market Concentration:</b><br/>
        The top 3 companies control {top_3_share:.1f}% of the industry's market capitalization.<br/>
        Top 5 companies: {top_5_share:.1f}% market share<br/>
        Top 10 companies: {top_10_share:.1f}% market share<br/><br/>
        Industry concentration is {'very high' if top_3_share > 75 else 'high' if top_3_share > 50 else 'moderate' if top_3_share > 25 else 'low'}.
        """

        content.append(Paragraph(concentration_text, styles['Normal']))
        content.append(Spacer(1, 0.2*inch))

        # Top companies table
        top_companies = df_sorted.head(10)

        company_data = [['Rank', 'Ticker', 'Market Cap ($B)', 'Market Share (%)', '1Y Return (%)']]

        for i, (idx, company) in enumerate(top_companies.iterrows(), 1):
            market_cap = company['market_cap'] / 1e9
            market_share = (company['market_cap'] / total_industry_cap) * 100
            return_1y = company.get('daily_daily_yearly_252d_pct_change', 0)

            company_data.append([
                str(i),
                company['ticker'],
                f"${market_cap:.1f}B",
                f"{market_share:.1f}%",
                f"{return_1y:.1f}%"
            ])

        company_table = Table(company_data, colWidths=[0.5*inch, 1*inch, 1.2*inch, 1*inch, 1*inch])
        company_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(company_table)

    return content

def _create_industry_risk_analysis(df: pd.DataFrame, industry_name: str, styles) -> list:
    """Create industry risk analysis section."""
    content = []

    risk_metrics = {}

    # Volatility analysis
    if 'atr_pct' in df.columns:
        avg_atr = df['atr_pct'].mean()
        risk_metrics['avg_volatility'] = avg_atr

    # Beta analysis (if available)
    # RSI analysis
    if 'daily_rsi_14' in df.columns:
        avg_rsi = df['daily_rsi_14'].mean()
        overbought_pct = (df['daily_rsi_14'] > 70).mean() * 100
        oversold_pct = (df['daily_rsi_14'] < 30).mean() * 100
        risk_metrics.update({
            'avg_rsi': avg_rsi,
            'overbought_pct': overbought_pct,
            'oversold_pct': oversold_pct
        })

    # 52-week position
    if 'daily_price_position_52w' in df.columns:
        avg_position = df['daily_price_position_52w'].mean()
        near_high_pct = (df['daily_price_position_52w'] > 0.8).mean() * 100
        near_low_pct = (df['daily_price_position_52w'] < 0.2).mean() * 100
        risk_metrics.update({
            'avg_52w_position': avg_position,
            'near_high_pct': near_high_pct,
            'near_low_pct': near_low_pct
        })

    if risk_metrics:
        risk_text = f"""
        <b>Risk Profile:</b><br/>
        Average volatility (ATR): {risk_metrics.get('avg_volatility', 0):.1f}%<br/>
        Average RSI: {risk_metrics.get('avg_rsi', 0):.0f} ({'Overbought territory' if risk_metrics.get('avg_rsi', 50) > 70 else 'Oversold territory' if risk_metrics.get('avg_rsi', 50) < 30 else 'Neutral territory'})<br/>
        Companies overbought: {risk_metrics.get('overbought_pct', 0):.0f}%<br/>
        Companies oversold: {risk_metrics.get('oversold_pct', 0):.0f}%<br/>
        Average 52-week position: {risk_metrics.get('avg_52w_position', 0):.2f}<br/>
        Companies near 52-week highs: {risk_metrics.get('near_high_pct', 0):.0f}%<br/>
        Companies near 52-week lows: {risk_metrics.get('near_low_pct', 0):.0f}%
        """

        content.append(Paragraph(risk_text, styles['Normal']))

    return content

def _create_industry_leaders_analysis(df: pd.DataFrame, industry_name: str, styles) -> list:
    """Create industry leaders analysis."""
    content = []

    if 'daily_daily_yearly_252d_pct_change' in df.columns:
        # Top performers by return
        top_performers = df.nlargest(15, 'daily_daily_yearly_252d_pct_change')

        if not top_performers.empty:
            content.append(Paragraph("<b>Top Performing Companies</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            performer_data = [['Rank', 'Ticker', '1Y Return (%)', 'Market Cap ($B)', 'RSI', '52W Position']]

            for i, (idx, stock) in enumerate(top_performers.iterrows(), 1):
                ticker = stock['ticker']
                performance = stock['daily_daily_yearly_252d_pct_change']
                market_cap = stock.get('market_cap', 0) / 1e9
                rsi = stock.get('daily_rsi_14', 0)
                position_52w = stock.get('daily_price_position_52w', 0)

                performer_data.append([
                    str(i),
                    ticker,
                    f"{performance:.1f}%",
                    f"${market_cap:.1f}B",
                    f"{rsi:.0f}",
                    f"{position_52w:.2f}"
                ])

            performer_table = Table(performer_data, colWidths=[0.5*inch, 1*inch, 1*inch, 1*inch, 0.7*inch, 0.8*inch])
            performer_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(performer_table)

    return content

def _create_industry_rankings_section(df: pd.DataFrame, styles) -> list:
    """Create industry rankings section with charts."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Industry performance ranking chart
    ranking_path = temp_path / "industry_ranking.png"
    if create_industry_performance_ranking(df, str(ranking_path)):
        content.append(Paragraph("<b>Industry Performance Rankings</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(ranking_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_industry_sector_section(df: pd.DataFrame, styles) -> list:
    """Create industry vs sector analysis section."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Industry sector comparison chart
    comparison_path = temp_path / "industry_sector_comparison.png"
    if create_industry_sector_comparison(df, str(comparison_path)):
        content.append(Paragraph("<b>Industry vs Sector Performance</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(comparison_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_industry_concentration_section(df: pd.DataFrame, styles) -> list:
    """Create industry concentration analysis section."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Industry concentration chart
    concentration_path = temp_path / "industry_concentration.png"
    if create_industry_concentration_analysis(df, str(concentration_path)):
        content.append(Paragraph("<b>Industry Concentration Analysis</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(concentration_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_top_performers_by_industry_section(df: pd.DataFrame, styles) -> list:
    """Create top performers by industry section."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Top performers grid chart
    performers_path = temp_path / "industry_top_performers.png"
    if create_industry_top_performers_grid(df, str(performers_path)):
        content.append(Paragraph("<b>Top Performers by Industry</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(performers_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _calculate_industry_statistics(df: pd.DataFrame) -> dict:
    """Calculate industry-specific statistics."""
    try:
        stats = {}

        if 'market_cap' in df.columns:
            market_caps = df['market_cap'].dropna()
            stats['total_market_cap'] = market_caps.sum() / 1e9
            stats['avg_market_cap'] = market_caps.mean() / 1e9
            stats['median_market_cap'] = market_caps.median() / 1e9
            stats['largest_company_cap'] = market_caps.max() / 1e9

            # Calculate HHI (Herfindahl-Hirschman Index) for concentration
            if len(market_caps) > 1:
                total_cap = market_caps.sum()
                market_shares = market_caps / total_cap
                hhi = (market_shares ** 2).sum()
                stats['hhi'] = hhi

        if 'daily_daily_yearly_252d_pct_change' in df.columns:
            perf_data = df['daily_daily_yearly_252d_pct_change'].dropna()
            if not perf_data.empty:
                stats['avg_performance'] = perf_data.mean()
                stats['median_performance'] = perf_data.median()
                stats['perf_std'] = perf_data.std()
                stats['perf_range'] = perf_data.max() - perf_data.min()

        if 'daily_rsi_14' in df.columns:
            stats['avg_rsi'] = df['daily_rsi_14'].mean()

        if 'atr_pct' in df.columns:
            stats['avg_volatility'] = df['atr_pct'].mean()

        return stats

    except Exception as e:
        logger.warning(f"Error calculating industry statistics: {e}")
        return {}