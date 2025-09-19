#!/usr/bin/env python3
"""
Risk Analysis PDF Template for Basic Calculation Data
=====================================================

Risk-focused analysis PDF template:
- Comprehensive risk-return analysis
- Volatility analysis and metrics
- Drawdown and recovery patterns
- Risk distribution by sectors/industries
- Value at Risk and stress testing insights
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
    create_risk_return_scatter, create_volatility_analysis,
    create_drawdown_analysis, create_risk_metrics_dashboard
)

logger = logging.getLogger(__name__)

def generate_pdf(df: pd.DataFrame, pdf_path: str, metadata: dict = None) -> bool:
    """
    Generate risk-focused analysis PDF from basic_calculation data.

    Args:
        df: Filtered DataFrame from post-process workflow
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Generating risk analysis PDF: {pdf_path}")

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

        # Title page
        content.extend(_create_risk_title_section(df, metadata, title_style, styles))
        content.append(PageBreak())

        # Risk executive summary
        content.append(Paragraph("Risk Executive Summary", heading_style))
        content.extend(_create_risk_executive_summary(df, metadata, styles))
        content.append(PageBreak())

        # Risk-return analysis
        content.append(Paragraph("Risk-Return Analysis", heading_style))
        content.extend(_create_risk_return_section(df, styles))
        content.append(PageBreak())

        # Volatility analysis
        content.append(Paragraph("Volatility Analysis", heading_style))
        content.extend(_create_volatility_section(df, styles))
        content.append(PageBreak())

        # Drawdown analysis
        content.append(Paragraph("Drawdown & Recovery Analysis", heading_style))
        content.extend(_create_drawdown_section(df, styles))
        content.append(PageBreak())

        # Risk metrics dashboard
        content.append(Paragraph("Risk Metrics Dashboard", heading_style))
        content.extend(_create_risk_dashboard_section(df, styles))
        content.append(PageBreak())

        # Sector risk analysis
        content.append(Paragraph("Risk by Sector", heading_style))
        content.extend(_create_sector_risk_section(df, styles))
        content.append(PageBreak())

        # Risk recommendations
        content.append(Paragraph("Risk Management Insights", heading_style))
        content.extend(_create_risk_recommendations(df, styles))

        # Build PDF
        doc.build(content)

        logger.info(f"Successfully generated risk analysis PDF: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating risk analysis PDF: {e}")
        return False

def _create_risk_title_section(df: pd.DataFrame, metadata: dict, title_style, styles) -> list:
    """Create risk analysis title page."""
    content = []

    # Main title
    content.append(Paragraph("Risk Analysis Report", title_style))
    content.append(Spacer(1, 0.3*inch))

    # Subtitle with data info
    generation_date = datetime.now().strftime("%B %d, %Y")
    data_date = metadata.get('processing_timestamp', datetime.now()).strftime("%Y-%m-%d") if metadata else "Unknown"

    subtitle = f"""
    <para align=center>
    <b>Comprehensive Risk Assessment</b><br/>
    Generated: {generation_date}<br/>
    Data Date: {data_date}<br/>
    Total Securities Analyzed: {len(df):,}
    </para>
    """
    content.append(Paragraph(subtitle, styles['Normal']))
    content.append(Spacer(1, 0.5*inch))

    # Key risk statistics
    risk_stats = _calculate_risk_statistics(df)
    if risk_stats:
        content.append(Paragraph("<b>Key Risk Metrics</b>", styles['Heading2']))
        content.append(Spacer(1, 0.1*inch))

        risk_data = [
            ['Risk Metric', 'Value', 'Interpretation'],
            ['Average Volatility (ATR)', f"{risk_stats.get('avg_volatility', 0):.1f}%", _interpret_volatility(risk_stats.get('avg_volatility', 0))],
            ['High Volatility Securities', f"{risk_stats.get('high_vol_pct', 0):.1f}%", "Stocks with ATR > 5%"],
            ['Average Distance from ATH', f"{risk_stats.get('avg_ath_distance', 0):.1f}%", _interpret_ath_distance(risk_stats.get('avg_ath_distance', 0))],
            ['Securities Near 52W Lows', f"{risk_stats.get('near_low_pct', 0):.1f}%", "Within 20% of 52-week low"],
            ['Portfolio Beta Proxy', f"{risk_stats.get('beta_proxy', 0):.2f}", _interpret_beta(risk_stats.get('beta_proxy', 0))],
            ['Risk Score (Composite)', f"{risk_stats.get('risk_score', 0):.2f}", _interpret_risk_score(risk_stats.get('risk_score', 0))],
            ['Value at Risk (95%)', f"{risk_stats.get('var_95', 0):.1f}%", "Worst expected loss (95% confidence)"]
        ]

        risk_table = Table(risk_data, colWidths=[2*inch, 1*inch, 2*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        content.append(risk_table)

    return content

def _create_risk_executive_summary(df: pd.DataFrame, metadata: dict, styles) -> list:
    """Create risk executive summary section."""
    content = []

    # Generate risk insights
    risk_insights = _generate_risk_insights(df)
    for insight in risk_insights:
        content.append(Paragraph(insight, styles['Normal']))
        content.append(Spacer(1, 0.1*inch))

    return content

def _create_risk_return_section(df: pd.DataFrame, styles) -> list:
    """Create risk-return analysis section with charts."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Risk-return scatter plot
    risk_return_path = temp_path / "risk_return_scatter.png"
    if create_risk_return_scatter(df, str(risk_return_path)):
        content.append(Paragraph("<b>Risk-Return Scatter Analysis</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(risk_return_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    # Risk-return table by quartiles
    content.extend(_create_risk_return_table(df, styles))

    return content

def _create_volatility_section(df: pd.DataFrame, styles) -> list:
    """Create volatility analysis section."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Volatility analysis chart
    volatility_path = temp_path / "volatility_analysis.png"
    if create_volatility_analysis(df, str(volatility_path)):
        content.append(Paragraph("<b>Volatility Distribution Analysis</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(volatility_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    # Volatility summary table
    content.extend(_create_volatility_summary_table(df, styles))

    return content

def _create_drawdown_section(df: pd.DataFrame, styles) -> list:
    """Create drawdown analysis section."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Drawdown analysis chart
    drawdown_path = temp_path / "drawdown_analysis.png"
    if create_drawdown_analysis(df, str(drawdown_path)):
        content.append(Paragraph("<b>Drawdown Analysis</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(drawdown_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_risk_dashboard_section(df: pd.DataFrame, styles) -> list:
    """Create risk metrics dashboard section."""
    content = []

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)
    temp_path = png_dir

    # Risk metrics dashboard
    dashboard_path = temp_path / "risk_dashboard.png"
    if create_risk_metrics_dashboard(df, str(dashboard_path)):
        content.append(Paragraph("<b>Risk Metrics Dashboard</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(dashboard_path), width=6*inch, height=4.5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_sector_risk_section(df: pd.DataFrame, styles) -> list:
    """Create sector risk analysis section."""
    content = []

    if 'sector' in df.columns and 'atr_pct' in df.columns:
        # Sector risk analysis
        sector_risk = df.groupby('sector', observed=False).agg({
            'atr_pct': ['mean', 'median', 'std'],
            'daily_distance_from_ATH_pct': 'mean',
            'daily_price_position_52w': 'mean',
            'ticker': 'count'
        }).round(2)

        # Flatten column names
        sector_risk.columns = ['_'.join(col) if col[1] else col[0] for col in sector_risk.columns]
        sector_risk = sector_risk.sort_values('atr_pct_mean', ascending=False)

        if not sector_risk.empty:
            content.append(Paragraph("<b>Risk Profile by Sector</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            # Create sector risk table
            risk_data = [['Sector', 'Stocks', 'Avg Volatility (%)', 'Avg ATH Distance (%)', '52W Position', 'Risk Level']]

            for sector, row in sector_risk.iterrows():
                sector_name = sector[:25] + '...' if len(sector) > 25 else sector
                stock_count = int(row['ticker_count'])  # Fixed: use correct flattened column name
                avg_volatility = row['atr_pct_mean']
                ath_distance = row['daily_distance_from_ATH_pct_mean']
                position_52w = row['daily_price_position_52w_mean']

                # Determine risk level
                risk_level = "Low"
                if avg_volatility > 5:
                    risk_level = "High"
                elif avg_volatility > 3:
                    risk_level = "Medium"

                risk_data.append([
                    sector_name,
                    str(stock_count),
                    f"{avg_volatility:.1f}%",
                    f"{ath_distance:.1f}%",
                    f"{position_52w:.2f}",
                    risk_level
                ])

            sector_risk_table = Table(risk_data, colWidths=[2*inch, 0.6*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch])
            sector_risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(sector_risk_table)

    return content

def _create_risk_recommendations(df: pd.DataFrame, styles) -> list:
    """Create risk management recommendations."""
    content = []

    recommendations = _generate_risk_recommendations(df)

    content.append(Paragraph("<b>Risk Management Recommendations</b>", styles['Heading3']))
    content.append(Spacer(1, 0.1*inch))

    for i, rec in enumerate(recommendations, 1):
        content.append(Paragraph(f"<b>{i}. {rec['title']}</b>", styles['Normal']))
        content.append(Paragraph(rec['description'], styles['Normal']))
        content.append(Spacer(1, 0.1*inch))

    return content

def _create_risk_return_table(df: pd.DataFrame, styles) -> list:
    """Create risk-return quartile analysis table."""
    content = []

    if 'atr_pct' in df.columns and 'daily_daily_yearly_252d_pct_change' in df.columns:
        # Create risk quartiles
        risk_quartiles = pd.qcut(df['atr_pct'], q=4, labels=['Low Risk', 'Medium-Low', 'Medium-High', 'High Risk'])

        quartile_analysis = df.groupby(risk_quartiles, observed=False).agg({
            'daily_daily_yearly_252d_pct_change': ['mean', 'median', 'std'],
            'atr_pct': ['mean', 'min', 'max'],
            'ticker': 'count'
        }).round(2)

        # Flatten column names
        quartile_analysis.columns = ['_'.join(col) for col in quartile_analysis.columns]

        if not quartile_analysis.empty:
            content.append(Paragraph("<b>Risk-Return by Quartiles</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            table_data = [['Risk Level', 'Stocks', 'Avg Return (%)', 'Return Std (%)', 'Avg Risk (%)', 'Risk Range (%)']]

            for risk_level, row in quartile_analysis.iterrows():
                stock_count = int(row['ticker_count'])
                avg_return = row['daily_daily_yearly_252d_pct_change_mean']
                return_std = row['daily_daily_yearly_252d_pct_change_std']
                avg_risk = row['atr_pct_mean']
                risk_min = row['atr_pct_min']
                risk_max = row['atr_pct_max']

                table_data.append([
                    str(risk_level),
                    str(stock_count),
                    f"{avg_return:.1f}%",
                    f"{return_std:.1f}%",
                    f"{avg_risk:.1f}%",
                    f"{risk_min:.1f}% - {risk_max:.1f}%"
                ])

            quartile_table = Table(table_data, colWidths=[1.2*inch, 0.7*inch, 1*inch, 1*inch, 1*inch, 1.3*inch])
            quartile_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(quartile_table)

    return content

def _create_volatility_summary_table(df: pd.DataFrame, styles) -> list:
    """Create volatility summary statistics table."""
    content = []

    if 'atr_pct' in df.columns:
        vol_data = df['atr_pct'].dropna()

        if not vol_data.empty:
            vol_stats = {
                'Mean': vol_data.mean(),
                'Median': vol_data.median(),
                'Std Dev': vol_data.std(),
                '25th Percentile': vol_data.quantile(0.25),
                '75th Percentile': vol_data.quantile(0.75),
                '95th Percentile': vol_data.quantile(0.95),
                'Min': vol_data.min(),
                'Max': vol_data.max()
            }

            content.append(Paragraph("<b>Volatility Statistics</b>", styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))

            vol_table_data = [['Statistic', 'Value']]
            for stat, value in vol_stats.items():
                vol_table_data.append([stat, f"{value:.2f}%"])

            vol_table = Table(vol_table_data, colWidths=[2*inch, 1.5*inch])
            vol_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(vol_table)

    return content

def _calculate_risk_statistics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive risk statistics."""
    try:
        stats = {}

        # Volatility statistics
        if 'atr_pct' in df.columns:
            vol_data = df['atr_pct'].dropna()
            stats['avg_volatility'] = vol_data.mean()
            stats['high_vol_pct'] = (vol_data > 5.0).mean() * 100  # Stocks with ATR > 5%

        # Drawdown statistics
        if 'daily_distance_from_ATH_pct' in df.columns:
            ath_data = df['daily_distance_from_ATH_pct'].dropna()
            stats['avg_ath_distance'] = ath_data.mean()

        # 52-week position
        if 'daily_price_position_52w' in df.columns:
            position_data = df['daily_price_position_52w'].dropna()
            stats['near_low_pct'] = (position_data < 0.2).mean() * 100

        # Beta proxy (correlation with market if available)
        # For now, use volatility as a proxy
        if 'atr_pct' in df.columns:
            stats['beta_proxy'] = df['atr_pct'].mean() / 2.0  # Simplified beta estimate

        # Composite risk score
        risk_factors = []
        if 'atr_pct' in df.columns:
            risk_factors.append(df['atr_pct'] / df['atr_pct'].max())  # Normalized volatility
        if 'daily_distance_from_ATH_pct' in df.columns:
            risk_factors.append(abs(df['daily_distance_from_ATH_pct']) / 100)  # Normalized drawdown

        if risk_factors:
            composite_risk = sum(risk_factors) / len(risk_factors)
            stats['risk_score'] = composite_risk.mean()

        # Value at Risk (95% confidence level)
        if 'daily_daily_yearly_252d_pct_change' in df.columns:
            returns = df['daily_daily_yearly_252d_pct_change'].dropna()
            if not returns.empty:
                stats['var_95'] = abs(np.percentile(returns, 5))  # 5th percentile (95% VaR)

        return stats

    except Exception as e:
        logger.warning(f"Error calculating risk statistics: {e}")
        return {}

def _generate_risk_insights(df: pd.DataFrame) -> list:
    """Generate risk analysis insights."""
    insights = []

    try:
        # Volatility insight
        if 'atr_pct' in df.columns:
            avg_vol = df['atr_pct'].mean()
            high_vol_pct = (df['atr_pct'] > 5.0).mean() * 100

            vol_assessment = "moderate"
            if avg_vol > 4:
                vol_assessment = "elevated"
            elif avg_vol < 2:
                vol_assessment = "low"

            insights.append(f"Market volatility is currently {vol_assessment} with an average ATR of {avg_vol:.1f}%. "
                           f"{high_vol_pct:.0f}% of securities exhibit high volatility (>5% ATR), indicating "
                           f"{'significant' if high_vol_pct > 25 else 'moderate' if high_vol_pct > 10 else 'limited'} "
                           f"risk concentration.")

        # Position insight
        if 'daily_price_position_52w' in df.columns:
            avg_position = df['daily_price_position_52w'].mean()
            near_high_pct = (df['daily_price_position_52w'] > 0.8).mean() * 100
            near_low_pct = (df['daily_price_position_52w'] < 0.2).mean() * 100

            insights.append(f"The market is positioned at an average of {avg_position:.1f} on the 52-week range. "
                           f"{near_high_pct:.0f}% of stocks are near 52-week highs while {near_low_pct:.0f}% "
                           f"are near lows, suggesting {'bullish' if near_high_pct > near_low_pct else 'bearish'} sentiment.")

        # Sector risk insight
        if 'sector' in df.columns and 'atr_pct' in df.columns:
            sector_risk = df.groupby('sector')['atr_pct'].mean()
            riskiest_sector = sector_risk.idxmax()
            safest_sector = sector_risk.idxmin()

            insights.append(f"Sector risk analysis reveals {riskiest_sector} as the most volatile sector "
                           f"({sector_risk.max():.1f}% avg ATR) while {safest_sector} appears most stable "
                           f"({sector_risk.min():.1f}% avg ATR).")

        if not insights:
            insights.append("Risk analysis indicates mixed market conditions with varying volatility levels across sectors and securities.")

    except Exception as e:
        logger.warning(f"Error generating risk insights: {e}")
        insights.append("Risk analysis provides insights into market volatility, position dynamics, and sector-specific risk characteristics.")

    return insights

def _generate_risk_recommendations(df: pd.DataFrame) -> list:
    """Generate risk management recommendations."""
    recommendations = []

    try:
        # Volatility-based recommendation
        if 'atr_pct' in df.columns:
            avg_vol = df['atr_pct'].mean()
            high_vol_pct = (df['atr_pct'] > 5.0).mean() * 100

            if high_vol_pct > 20:
                recommendations.append({
                    'title': 'High Volatility Environment',
                    'description': f'With {high_vol_pct:.0f}% of securities showing high volatility, consider position sizing adjustments and increased use of stop-losses. Focus on quality names with lower volatility profiles.'
                })
            elif avg_vol < 2:
                recommendations.append({
                    'title': 'Low Volatility Environment',
                    'description': 'Current low volatility environment may indicate complacency. Consider preparing for potential volatility expansion and maintaining adequate hedging strategies.'
                })

        # Position-based recommendation
        if 'daily_price_position_52w' in df.columns:
            near_high_pct = (df['daily_price_position_52w'] > 0.8).mean() * 100

            if near_high_pct > 30:
                recommendations.append({
                    'title': 'Elevated Price Levels',
                    'description': f'With {near_high_pct:.0f}% of stocks near 52-week highs, exercise increased caution on new positions. Consider taking profits on extended positions and maintaining tighter risk management.'
                })

        # Sector diversification recommendation
        if 'sector' in df.columns and 'atr_pct' in df.columns:
            sector_risk_spread = df.groupby('sector')['atr_pct'].mean().std()

            if sector_risk_spread > 1.5:
                recommendations.append({
                    'title': 'Sector Risk Diversification',
                    'description': 'Significant volatility differences across sectors suggest opportunities for risk-adjusted allocation. Consider overweighting lower-volatility sectors for defensive positioning.'
                })

        # Default recommendations
        if not recommendations:
            recommendations.extend([
                {
                    'title': 'Portfolio Diversification',
                    'description': 'Maintain adequate diversification across sectors, market caps, and risk profiles to manage overall portfolio volatility.'
                },
                {
                    'title': 'Risk Monitoring',
                    'description': 'Implement regular risk monitoring processes including volatility tracking, correlation analysis, and stress testing scenarios.'
                }
            ])

    except Exception as e:
        logger.warning(f"Error generating risk recommendations: {e}")

    return recommendations[:5]  # Limit to 5 recommendations

# Helper functions for risk interpretation
def _interpret_volatility(vol: float) -> str:
    """Interpret volatility level."""
    if vol > 5:
        return "High"
    elif vol > 3:
        return "Moderate"
    elif vol > 1:
        return "Low"
    else:
        return "Very Low"

def _interpret_ath_distance(distance: float) -> str:
    """Interpret distance from all-time high."""
    abs_distance = abs(distance)
    if abs_distance > 50:
        return "Deep drawdown"
    elif abs_distance > 25:
        return "Moderate drawdown"
    elif abs_distance > 10:
        return "Minor drawdown"
    else:
        return "Near highs"

def _interpret_beta(beta: float) -> str:
    """Interpret beta proxy."""
    if beta > 1.5:
        return "High beta"
    elif beta > 0.8:
        return "Market beta"
    else:
        return "Low beta"

def _interpret_risk_score(score: float) -> str:
    """Interpret composite risk score."""
    if score > 0.7:
        return "High risk"
    elif score > 0.4:
        return "Moderate risk"
    else:
        return "Low risk"