#!/usr/bin/env python3
"""
Universe Analysis PDF Template for Basic Calculation Data
=========================================================

Universe/index-focused analysis PDF template:
- Index membership analysis
- Universe performance comparisons
- Cross-universe overlap analysis
- Sector representation within universes
- Top performers by universe
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
    create_index_membership_analysis, create_universe_performance_comparison,
    create_universe_sector_analysis
)

logger = logging.getLogger(__name__)

def generate_pdf(df: pd.DataFrame, pdf_path: str, metadata: dict = None) -> bool:
    """
    Generate universe/index-focused analysis PDF from basic_calculation data.

    Args:
        df: Filtered DataFrame from post-process workflow
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Generating universe analysis PDF: {pdf_path}")

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

        # Determine target universe if specific filter applied
        target_universe = _detect_target_universe(df, metadata)

        if target_universe:
            # Single universe analysis
            content.extend(_create_single_universe_analysis(df, target_universe, metadata, title_style, heading_style, styles))
        else:
            # Multi-universe comparison
            content.extend(_create_multi_universe_analysis(df, metadata, title_style, heading_style, styles))

        # Build PDF
        doc.build(content)

        logger.info(f"Successfully generated universe analysis PDF: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating universe analysis PDF: {e}")
        return False

def _detect_target_universe(df: pd.DataFrame, metadata: dict) -> str:
    """Detect if analysis is focused on a specific universe/index."""
    try:
        # Check for universe-specific filters
        universe_columns = ['SP500', 'NASDAQ100', 'Russell1000', 'Russell3000', 'DowJonesIndustrialAverage']

        if metadata and 'filter_operations' in metadata:
            for filter_op in metadata['filter_operations']:
                column = filter_op.get('Column')
                if column in universe_columns and filter_op.get('Condition') == 'equals' and filter_op.get('Value') == 'TRUE':
                    return column

        # Check if 90%+ of stocks belong to single universe
        for universe in universe_columns:
            if universe in df.columns:
                universe_pct = (df[universe] == True).mean()
                if universe_pct > 0.9:
                    return universe

        return None

    except Exception as e:
        logger.warning(f"Error detecting target universe: {e}")
        return None

def _create_single_universe_analysis(df: pd.DataFrame, target_universe: str, metadata: dict,
                                   title_style, heading_style, styles) -> list:
    """Create single universe deep-dive analysis."""
    content = []

    # Clean universe name for display
    universe_display = _clean_universe_name(target_universe)

    # Title page
    content.append(Paragraph(f"{universe_display} Analysis", title_style))
    content.append(Spacer(1, 0.3*inch))

    # Universe overview
    universe_data = df[df[target_universe] == True] if target_universe in df.columns else df
    content.extend(_create_universe_overview(universe_data, universe_display, styles))
    content.append(PageBreak())

    # Performance analysis
    content.append(Paragraph("Performance Analysis", heading_style))
    content.extend(_create_universe_performance_analysis(universe_data, universe_display, styles))
    content.append(PageBreak())

    # Sector composition
    content.append(Paragraph("Sector Composition", heading_style))
    content.extend(_create_universe_sector_composition(universe_data, universe_display, styles))
    content.append(PageBreak())

    # Top performers
    content.append(Paragraph("Top Performers", heading_style))
    content.extend(_create_universe_top_performers(universe_data, universe_display, styles))
    content.append(PageBreak())

    # Risk characteristics
    content.append(Paragraph("Risk Characteristics", heading_style))
    content.extend(_create_universe_risk_analysis(universe_data, universe_display, styles))

    return content

def _create_multi_universe_analysis(df: pd.DataFrame, metadata: dict,
                                  title_style, heading_style, styles) -> list:
    """Create multi-universe comparison analysis."""
    content = []

    # Title page
    content.append(Paragraph("Universe Comparison Analysis", title_style))
    content.append(Spacer(1, 0.3*inch))

    generation_date = datetime.now().strftime("%B %d, %Y")
    subtitle = f"""
    <para align=center>
    <b>Index & Universe Performance Analysis</b><br/>
    Generated: {generation_date}<br/>
    Total Securities: {len(df):,}
    </para>
    """
    content.append(Paragraph(subtitle, styles['Normal']))
    content.append(PageBreak())

    # Universe membership overview
    content.append(Paragraph("Universe Membership Overview", heading_style))
    content.extend(_create_membership_overview_section(df, styles))
    content.append(PageBreak())

    # Performance comparison
    content.append(Paragraph("Performance Comparison", heading_style))
    content.extend(_create_performance_comparison_section(df, styles))
    content.append(PageBreak())

    # Universe characteristics
    content.append(Paragraph("Universe Characteristics", heading_style))
    content.extend(_create_universe_characteristics_section(df, styles))
    content.append(PageBreak())

    # Sector analysis across universes
    content.append(Paragraph("Sector Analysis Across Universes", heading_style))
    content.extend(_create_cross_universe_sector_section(df, styles))

    return content

def _create_universe_overview(df: pd.DataFrame, universe_name: str, styles) -> list:
    """Create universe overview section."""
    content = []

    # Key statistics
    stats = _calculate_universe_statistics(df)

    overview_text = f"""
    <b>Universe Overview: {universe_name}</b><br/><br/>
    This analysis covers {len(df):,} securities in the {universe_name} universe, representing
    ${stats.get('total_market_cap', 0):.1f}T in total market capitalization.
    """

    content.append(Paragraph(overview_text, styles['Normal']))
    content.append(Spacer(1, 0.2*inch))

    # Statistics table
    if stats:
        stats_data = [
            ['Metric', 'Value'],
            ['Number of Securities', f"{len(df):,}"],
            ['Total Market Cap', f"${stats.get('total_market_cap', 0):.1f}T"],
            ['Average Market Cap', f"${stats.get('avg_market_cap', 0):.1f}B"],
            ['Largest Component Weight', f"{stats.get('largest_weight', 0):.1f}%"],
            ['Top 10 Components Weight', f"{stats.get('top10_weight', 0):.1f}%"],
            ['Number of Sectors', f"{stats.get('sector_count', 0)}"],
            ['Largest Sector Weight', f"{stats.get('largest_sector_weight', 0):.1f}%"],
            ['Average 1Y Performance', f"{stats.get('avg_performance', 0):.1f}%"],
            ['Performance Std Dev', f"{stats.get('perf_std', 0):.1f}%"],
            ['Average RSI', f"{stats.get('avg_rsi', 0):.0f}"],
            ['Average Beta Proxy', f"{stats.get('avg_beta', 0):.2f}"]
        ]

        stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
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

def _create_universe_performance_analysis(df: pd.DataFrame, universe_name: str, styles) -> list:
    """Create universe performance analysis section."""
    content = []

    # Performance summary
    if 'daily_daily_yearly_252d_pct_change' in df.columns:
        perf_data = df['daily_daily_yearly_252d_pct_change'].dropna()

        if not perf_data.empty:
            avg_perf = perf_data.mean()
            median_perf = perf_data.median()
            positive_pct = (perf_data > 0).mean() * 100
            top_quartile_threshold = perf_data.quantile(0.75)

            perf_summary = f"""
            <b>Performance Summary:</b><br/>
            The {universe_name} universe shows an average 1-year return of {avg_perf:.1f}% with a median of {median_perf:.1f}%.
            {positive_pct:.0f}% of components generated positive returns, with the top quartile threshold at {top_quartile_threshold:.1f}%.
            """

            content.append(Paragraph(perf_summary, styles['Normal']))
            content.append(Spacer(1, 0.2*inch))

    # Performance distribution table
    if 'daily_daily_yearly_252d_pct_change' in df.columns:
        perf_ranges = [
            (float('inf'), 50, 'Exceptional (>50%)'),
            (50, 25, 'Strong (25-50%)'),
            (25, 10, 'Good (10-25%)'),
            (10, 0, 'Positive (0-10%)'),
            (0, -10, 'Modest Decline (0 to -10%)'),
            (-10, float('-inf'), 'Significant Decline (<-10%)')
        ]

        perf_dist_data = [['Performance Range', 'Count', 'Percentage']]

        for upper, lower, label in perf_ranges:
            if upper == float('inf'):
                count = (df['daily_daily_yearly_252d_pct_change'] > lower).sum()
            elif lower == float('-inf'):
                count = (df['daily_daily_yearly_252d_pct_change'] < upper).sum()
            else:
                count = ((df['daily_daily_yearly_252d_pct_change'] > lower) &
                        (df['daily_daily_yearly_252d_pct_change'] <= upper)).sum()

            percentage = (count / len(df)) * 100
            perf_dist_data.append([label, str(count), f"{percentage:.1f}%"])

        perf_table = Table(perf_dist_data, colWidths=[2.5*inch, 1*inch, 1*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(perf_table)

    return content

def _create_universe_sector_composition(df: pd.DataFrame, universe_name: str, styles) -> list:
    """Create sector composition analysis."""
    content = []

    if 'sector' in df.columns and 'market_cap' in df.columns:
        # Calculate sector weights by market cap
        sector_weights = df.groupby('sector', observed=False).agg({
            'market_cap': 'sum',
            'ticker': 'count'
        })

        total_market_cap = df['market_cap'].sum()
        sector_weights['weight_pct'] = (sector_weights['market_cap'] / total_market_cap) * 100
        sector_weights = sector_weights.sort_values('weight_pct', ascending=False)

        if not sector_weights.empty:
            content.append(Paragraph(f"<b>Sector Composition by Market Cap Weight</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            sector_data = [['Sector', 'Companies', 'Weight (%)', 'Market Cap ($B)']]

            for sector, row in sector_weights.iterrows():
                sector_name = sector[:30] + '...' if len(sector) > 30 else sector
                companies = int(row['ticker_count'])  # Fixed: use correct flattened column name
                weight = row['weight_pct']
                market_cap = row['market_cap'] / 1e9

                sector_data.append([
                    sector_name,
                    str(companies),
                    f"{weight:.1f}%",
                    f"${market_cap:.0f}B"
                ])

            sector_table = Table(sector_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1.2*inch])
            sector_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(sector_table)

    return content

def _create_universe_top_performers(df: pd.DataFrame, universe_name: str, styles) -> list:
    """Create top performers section."""
    content = []

    if 'daily_daily_yearly_252d_pct_change' in df.columns:
        top_performers = df.nlargest(20, 'daily_daily_yearly_252d_pct_change')

        if not top_performers.empty:
            content.append(Paragraph(f"<b>Top 20 Performers in {universe_name}</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            performer_data = [['Rank', 'Ticker', 'Sector', '1Y Return (%)', 'Market Cap ($B)', 'Weight (%)']]

            total_market_cap = df['market_cap'].sum() if 'market_cap' in df.columns else 1

            for i, (idx, stock) in enumerate(top_performers.iterrows(), 1):
                ticker = stock['ticker']
                sector = stock.get('sector', 'N/A')[:20]
                performance = stock['daily_daily_yearly_252d_pct_change']
                market_cap = stock.get('market_cap', 0) / 1e9
                weight = (stock.get('market_cap', 0) / total_market_cap) * 100 if total_market_cap > 0 else 0

                performer_data.append([
                    str(i),
                    ticker,
                    sector,
                    f"{performance:.1f}%",
                    f"${market_cap:.1f}B",
                    f"{weight:.2f}%"
                ])

            performer_table = Table(performer_data, colWidths=[0.4*inch, 0.8*inch, 1.5*inch, 1*inch, 1*inch, 0.8*inch])
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

def _create_universe_risk_analysis(df: pd.DataFrame, universe_name: str, styles) -> list:
    """Create universe risk analysis section."""
    content = []

    risk_summary = _generate_universe_risk_summary(df, universe_name)
    content.append(Paragraph(risk_summary, styles['Normal']))

    return content

def _create_membership_overview_section(df: pd.DataFrame, styles) -> list:
    """Create membership overview section with charts."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Index membership analysis chart
    membership_path = temp_path / "index_membership.png"
    if create_index_membership_analysis(df, str(membership_path)):
        content.append(Paragraph("<b>Index Membership Analysis</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(membership_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_performance_comparison_section(df: pd.DataFrame, styles) -> list:
    """Create performance comparison section."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Universe performance comparison chart
    comparison_path = temp_path / "universe_performance.png"
    if create_universe_performance_comparison(df, str(comparison_path)):
        content.append(Paragraph("<b>Universe Performance Comparison</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(comparison_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_universe_characteristics_section(df: pd.DataFrame, styles) -> list:
    """Create universe characteristics comparison table."""
    content = []

    # Find universe columns
    universe_cols = [col for col in df.columns if col in [
        'SP500', 'NASDAQ100', 'Russell1000', 'Russell3000', 'DowJonesIndustrialAverage'
    ]]

    if universe_cols and 'market_cap' in df.columns:
        universe_stats = {}

        for universe in universe_cols:
            universe_stocks = df[df[universe] == True]
            if len(universe_stocks) > 0:
                stats = {
                    'count': len(universe_stocks),
                    'avg_market_cap': universe_stocks['market_cap'].mean() / 1e9,
                    'total_market_cap': universe_stocks['market_cap'].sum() / 1e12,
                    'avg_performance': universe_stocks.get('daily_daily_yearly_252d_pct_change', pd.Series()).mean(),
                    'sectors': universe_stocks['sector'].nunique() if 'sector' in df.columns else 0
                }
                universe_stats[_clean_universe_name(universe)] = stats

        if universe_stats:
            content.append(Paragraph("<b>Universe Characteristics Comparison</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            char_data = [['Universe', 'Count', 'Avg Cap ($B)', 'Total Cap ($T)', 'Avg Return (%)', 'Sectors']]

            for universe, stats in universe_stats.items():
                char_data.append([
                    universe,
                    str(stats['count']),
                    f"${stats['avg_market_cap']:.1f}B",
                    f"${stats['total_market_cap']:.1f}T",
                    f"{stats['avg_performance']:.1f}%",
                    str(stats['sectors'])
                ])

            char_table = Table(char_data, colWidths=[1.5*inch, 0.8*inch, 1*inch, 1*inch, 1*inch, 0.8*inch])
            char_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(char_table)

    return content

def _create_cross_universe_sector_section(df: pd.DataFrame, styles) -> list:
    """Create cross-universe sector analysis section."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Universe sector analysis chart
    sector_path = temp_path / "universe_sector_analysis.png"
    if create_universe_sector_analysis(df, str(sector_path)):
        content.append(Paragraph("<b>Sector Analysis Across Universes</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(sector_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _calculate_universe_statistics(df: pd.DataFrame) -> dict:
    """Calculate universe-specific statistics."""
    try:
        stats = {}

        if 'market_cap' in df.columns:
            market_caps = df['market_cap'].dropna()
            stats['total_market_cap'] = market_caps.sum() / 1e12  # Trillions
            stats['avg_market_cap'] = market_caps.mean() / 1e9  # Billions

            # Largest component weight
            if len(market_caps) > 0:
                largest_weight = (market_caps.max() / market_caps.sum()) * 100
                stats['largest_weight'] = largest_weight

                # Top 10 weight
                top_10_weight = (market_caps.nlargest(10).sum() / market_caps.sum()) * 100
                stats['top10_weight'] = top_10_weight

        if 'sector' in df.columns:
            stats['sector_count'] = df['sector'].nunique()

            # Largest sector weight
            if 'market_cap' in df.columns:
                sector_weights = df.groupby('sector')['market_cap'].sum()
                largest_sector_weight = (sector_weights.max() / df['market_cap'].sum()) * 100
                stats['largest_sector_weight'] = largest_sector_weight

        if 'daily_daily_yearly_252d_pct_change' in df.columns:
            perf_data = df['daily_daily_yearly_252d_pct_change'].dropna()
            if not perf_data.empty:
                stats['avg_performance'] = perf_data.mean()
                stats['perf_std'] = perf_data.std()

        if 'daily_rsi_14' in df.columns:
            stats['avg_rsi'] = df['daily_rsi_14'].mean()

        if 'atr_pct' in df.columns:
            stats['avg_beta'] = df['atr_pct'].mean() / 2.0  # Simplified beta proxy

        return stats

    except Exception as e:
        logger.warning(f"Error calculating universe statistics: {e}")
        return {}

def _generate_universe_risk_summary(df: pd.DataFrame, universe_name: str) -> str:
    """Generate risk summary for universe."""
    try:
        risk_elements = []

        # Volatility assessment
        if 'atr_pct' in df.columns:
            avg_vol = df['atr_pct'].mean()
            vol_level = "high" if avg_vol > 4 else "moderate" if avg_vol > 2 else "low"
            risk_elements.append(f"average volatility is {vol_level} at {avg_vol:.1f}%")

        # Position assessment
        if 'daily_price_position_52w' in df.columns:
            avg_position = df['daily_price_position_52w'].mean()
            position_desc = "elevated" if avg_position > 0.7 else "moderate" if avg_position > 0.3 else "depressed"
            risk_elements.append(f"price levels are {position_desc} (avg 52W position: {avg_position:.2f})")

        # RSI assessment
        if 'daily_rsi_14' in df.columns:
            avg_rsi = df['daily_rsi_14'].mean()
            rsi_desc = "overbought" if avg_rsi > 60 else "oversold" if avg_rsi < 40 else "neutral"
            risk_elements.append(f"technical conditions appear {rsi_desc} (avg RSI: {avg_rsi:.0f})")

        if risk_elements:
            return f"<b>Risk Profile:</b> The {universe_name} universe shows {', '.join(risk_elements)}."
        else:
            return f"<b>Risk Profile:</b> Risk characteristics for {universe_name} require further analysis."

    except Exception as e:
        logger.warning(f"Error generating risk summary: {e}")
        return f"<b>Risk Profile:</b> Risk analysis for {universe_name}."

def _clean_universe_name(universe_col: str) -> str:
    """Clean universe column name for display."""
    name_map = {
        'SP500': 'S&P 500',
        'NASDAQ100': 'NASDAQ 100',
        'Russell1000': 'Russell 1000',
        'Russell3000': 'Russell 3000',
        'DowJonesIndustrialAverage': 'Dow Jones Industrial Average',
        'SP100': 'S&P 100',
        'NASDAQComposite': 'NASDAQ Composite'
    }
    return name_map.get(universe_col, universe_col)