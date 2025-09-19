#!/usr/bin/env python3
"""
Top Performers Evolution V1 PDF Template
=======================================

Step-by-step workflow template for identifying and analyzing top stock leaders
with individual timeframe analysis, evolution tracking, and technical context.

Workflow Steps:
1. Identify top leaders per time period (individual charts)
2. Track performance evolution of top performers
3. Add technical indicator context
4. Organize data into summary tables
5. Visualize trends and patterns
6. Analyze patterns and generate insights
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
    Generate Top Performers Evolution V1 PDF report with step-by-step workflow.

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

        # Title and Executive Summary
        title_text = f"Top Performers Evolution Analysis V1"
        if metadata and 'original_filename' in metadata:
            title_text += f" - {metadata['original_filename']}"

        content.append(Paragraph(title_text, title_style))
        content.append(Spacer(1, 0.3*inch))
        content.extend(_create_workflow_overview(styles))
        content.append(PageBreak())

        # Step 1: Individual Time Period Leaders
        content.extend(_create_individual_period_sections(df, styles))

        # Step 2: Evolution Tracking
        content.extend(_create_evolution_tracking_sections(df, styles))

        # Step 4: Technical Context
        content.extend(_create_technical_context_section(df, styles))

        # Step 5: Consolidated Summary
        content.extend(_create_consolidated_summary_section(df, styles))

        # Step 6: Advanced Visualizations
        content.extend(_create_advanced_visualizations_section(df, styles))

        # Step 7: Pattern Analysis & Insights
        content.extend(_create_insights_analysis_section(df, styles))

        # Build PDF
        doc.build(content)

        logger.info(f"Successfully generated Top Performers Evolution V1 PDF: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating Top Performers Evolution V1 PDF: {e}")
        return False

def _get_timeframe_columns(df: pd.DataFrame) -> dict:
    """Get available timeframe performance columns."""
    timeframe_mapping = {
        'Daily (1D)': 'daily_daily_daily_1d_pct_change',
        'Weekly (1W)': 'daily_daily_weekly_7d_pct_change',
        'Monthly (1M)': 'daily_daily_monthly_22d_pct_change',
        'Quarterly (1Q)': 'daily_daily_quarterly_66d_pct_change',
        'Year-to-Date': 'daily_daily_yearly_252d_pct_change'
    }

    # Filter to only available columns
    available_timeframes = {}
    for label, col in timeframe_mapping.items():
        if col in df.columns:
            available_timeframes[label] = col

    return available_timeframes

def _create_workflow_overview(styles) -> list:
    """Create workflow overview section."""
    content = []

    content.append(Paragraph("<b>Analysis Workflow Overview</b>", styles['Heading2']))
    content.append(Spacer(1, 0.2*inch))

    workflow_text = """
    <b>Step-by-Step Analysis Process:</b><br/>
    <b>Step 1:</b> Identify Top Leaders per Time Period - Individual analysis and charts for each timeframe<br/>
    <b>Step 2:</b> Track Performance Evolution - Individual stock journey analysis across all timeframes<br/>
    <b>Step 4:</b> Add Technical Indicator Context - RSI, MACD, and moving average analysis<br/>
    <b>Step 5:</b> Consolidated Summary Tables - Master view of all top performers<br/>
    <b>Step 6:</b> Advanced Visualizations - Heatmaps and pattern recognition<br/>
    <b>Step 7:</b> Pattern Analysis & Insights - Actionable insights and recommendations
    """

    content.append(Paragraph(workflow_text, styles['Normal']))
    content.append(Spacer(1, 0.3*inch))

    return content

def _create_individual_period_sections(df: pd.DataFrame, styles) -> list:
    """Step 1: Create individual time period analysis sections."""
    content = []

    content.append(Paragraph("<b>Step 1: Top Leaders by Time Period</b>", styles['Heading1']))
    content.append(Spacer(1, 0.2*inch))

    timeframes = _get_timeframe_columns(df)

    if not timeframes:
        content.append(Paragraph("No timeframe performance data available.", styles['Normal']))
        return content

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each timeframe individually
    for period_name, period_col in timeframes.items():
        content.append(PageBreak())
        content.append(Paragraph(f"<b>{period_name} Leaders</b>", styles['Heading2']))
        content.append(Spacer(1, 0.2*inch))

        # Get top 20 performers for this period
        period_data = df[df[period_col].notna()].copy()
        if period_data.empty:
            content.append(Paragraph(f"No data available for {period_name}.", styles['Normal']))
            continue

        top_performers = period_data.nlargest(20, period_col)

        # Create individual period chart
        chart_filename = f"period_leaders_{period_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        chart_path = png_dir / chart_filename

        if _create_period_bar_chart(top_performers, period_col, period_name, str(chart_path)):
            img = RLImage(str(chart_path), width=6*inch, height=4*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))

        # Create period summary table
        content.extend(_create_period_summary_table(top_performers, period_col, period_name, styles))

    return content

def _create_evolution_tracking_sections(df: pd.DataFrame, styles) -> list:
    """Step 2: Create evolution tracking sections for top performers."""
    content = []

    content.append(PageBreak())
    content.append(Paragraph("<b>Step 2: Performance Evolution Tracking</b>", styles['Heading1']))
    content.append(Spacer(1, 0.2*inch))

    timeframes = _get_timeframe_columns(df)
    if not timeframes:
        content.append(Paragraph("No timeframe data available for evolution tracking.", styles['Normal']))
        return content

    # Get unified list of all top performers (top 10 from each period)
    all_top_performers = set()
    for period_name, period_col in timeframes.items():
        period_data = df[df[period_col].notna()]
        if not period_data.empty:
            top_10 = period_data.nlargest(10, period_col)['ticker'].tolist()
            all_top_performers.update(top_10)

    if not all_top_performers:
        content.append(Paragraph("No top performers identified for evolution tracking.", styles['Normal']))
        return content

    content.append(Paragraph(f"<b>Tracking {len(all_top_performers)} Unique Top Performers</b>", styles['Heading3']))
    content.append(Spacer(1, 0.2*inch))

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)

    # Create evolution chart for unified top performers
    evolution_chart_path = png_dir / "performance_evolution_tracking.png"
    if _create_evolution_tracking_chart(df, list(all_top_performers), timeframes, str(evolution_chart_path)):
        content.append(Paragraph("<b>Performance Evolution Across All Timeframes</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(evolution_chart_path), width=7*inch, height=5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    # Create evolution summary table
    content.extend(_create_evolution_summary_table(df, list(all_top_performers), timeframes, styles))

    return content

def _create_technical_context_section(df: pd.DataFrame, styles) -> list:
    """Step 4: Create technical indicator context section."""
    content = []

    content.append(PageBreak())
    content.append(Paragraph("<b>Step 4: Technical Indicator Context</b>", styles['Heading1']))
    content.append(Spacer(1, 0.2*inch))

    # Get top performers for technical analysis
    timeframes = _get_timeframe_columns(df)
    all_top_performers = set()

    for period_name, period_col in timeframes.items():
        period_data = df[df[period_col].notna()]
        if not period_data.empty:
            top_10 = period_data.nlargest(10, period_col)['ticker'].tolist()
            all_top_performers.update(top_10)

    if not all_top_performers:
        content.append(Paragraph("No top performers for technical analysis.", styles['Normal']))
        return content

    # Technical indicators analysis
    content.extend(_create_technical_indicators_table(df, list(all_top_performers), styles))

    return content

def _create_consolidated_summary_section(df: pd.DataFrame, styles) -> list:
    """Step 5: Create consolidated summary section."""
    content = []

    content.append(PageBreak())
    content.append(Paragraph("<b>Step 5: Consolidated Summary</b>", styles['Heading1']))
    content.append(Spacer(1, 0.2*inch))

    timeframes = _get_timeframe_columns(df)
    if not timeframes:
        content.append(Paragraph("No data available for consolidated summary.", styles['Normal']))
        return content

    # Get unified top performers
    all_top_performers = set()
    for period_name, period_col in timeframes.items():
        period_data = df[df[period_col].notna()]
        if not period_data.empty:
            top_performers = period_data.nlargest(15, period_col)['ticker'].tolist()
            all_top_performers.update(top_performers)

    if not all_top_performers:
        content.append(Paragraph("No top performers for consolidated summary.", styles['Normal']))
        return content

    # Create master summary table
    content.extend(_create_master_summary_table(df, list(all_top_performers), timeframes, styles))

    return content

def _create_advanced_visualizations_section(df: pd.DataFrame, styles) -> list:
    """Step 6: Create advanced visualizations section."""
    content = []

    content.append(PageBreak())
    content.append(Paragraph("<b>Step 6: Advanced Visualizations</b>", styles['Heading1']))
    content.append(Spacer(1, 0.2*inch))

    timeframes = _get_timeframe_columns(df)
    if not timeframes:
        content.append(Paragraph("No data available for advanced visualizations.", styles['Normal']))
        return content

    # Use permanent PNG directory
    png_dir = Path("results/post_process")
    png_dir.mkdir(parents=True, exist_ok=True)

    # Create comprehensive heatmap
    heatmap_path = png_dir / "comprehensive_performance_heatmap.png"
    if _create_comprehensive_heatmap(df, timeframes, str(heatmap_path)):
        content.append(Paragraph("<b>Comprehensive Performance Heatmap</b>", styles['Heading3']))
        content.append(Spacer(1, 0.1*inch))

        img = RLImage(str(heatmap_path), width=7*inch, height=5*inch)
        content.append(img)
        content.append(Spacer(1, 0.2*inch))

    return content

def _create_insights_analysis_section(df: pd.DataFrame, styles) -> list:
    """Step 7: Create pattern analysis and insights section."""
    content = []

    content.append(PageBreak())
    content.append(Paragraph("<b>Step 7: Pattern Analysis & Insights</b>", styles['Heading1']))
    content.append(Spacer(1, 0.2*inch))

    timeframes = _get_timeframe_columns(df)
    if not timeframes:
        content.append(Paragraph("No data available for pattern analysis.", styles['Normal']))
        return content

    # Analyze patterns and generate insights
    insights = _analyze_performance_patterns(df, timeframes)

    content.append(Paragraph("<b>Key Insights & Patterns</b>", styles['Heading3']))
    content.append(Spacer(1, 0.2*inch))

    for insight in insights:
        content.append(Paragraph(f"• {insight}", styles['Normal']))
        content.append(Spacer(1, 0.1*inch))

    return content

# Helper functions for chart creation and data analysis

def _create_period_bar_chart(top_performers: pd.DataFrame, period_col: str, period_name: str, output_path: str) -> bool:
    """Create bar chart for individual time period leaders."""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        tickers = top_performers['ticker'].head(20)
        performances = top_performers[period_col].head(20)

        # Create color gradient based on performance
        colors_list = plt.cm.RdYlGn(np.linspace(0.3, 1, len(performances)))

        bars = ax.bar(range(len(tickers)), performances, color=colors_list)

        # Customize chart
        ax.set_xlabel('Stocks', fontsize=12)
        ax.set_ylabel('Performance (%)', fontsize=12)
        ax.set_title(f'{period_name} Top Performers', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, performances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        logger.error(f"Error creating period bar chart: {e}")
        return False

def _create_period_summary_table(top_performers: pd.DataFrame, period_col: str, period_name: str, styles) -> list:
    """Create summary table for individual period."""
    content = []

    # Create top 15 summary table
    table_data = [['Rank', 'Ticker', 'Performance (%)', 'Current Price', 'Market Cap']]

    for idx, (_, row) in enumerate(top_performers.head(15).iterrows(), 1):
        ticker = row['ticker']
        performance = f"{row[period_col]:.1f}%"
        price = f"${row.get('current_price', 0):.2f}" if 'current_price' in row else "N/A"
        market_cap = f"${row.get('market_cap', 0)/1e9:.1f}B" if 'market_cap' in row else "N/A"

        table_data.append([str(idx), ticker, performance, price, market_cap])

    period_table = Table(table_data)
    period_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    content.append(period_table)
    content.append(Spacer(1, 0.2*inch))

    return content

def _create_evolution_tracking_chart(df: pd.DataFrame, top_performers: list, timeframes: dict, output_path: str) -> bool:
    """Create evolution tracking chart for top performers."""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot performance evolution for each stock
        colors_list = plt.cm.tab20(np.linspace(0, 1, len(top_performers)))

        for i, ticker in enumerate(top_performers[:10]):  # Limit to top 10 for readability
            stock_data = df[df['ticker'] == ticker]
            if stock_data.empty:
                continue

            stock_data = stock_data.iloc[0]

            performances = []
            timeframe_labels = []

            for tf_name, tf_col in timeframes.items():
                if tf_col in stock_data.index and not pd.isna(stock_data[tf_col]):
                    performances.append(stock_data[tf_col])
                    timeframe_labels.append(tf_name.split('(')[0].strip())

            if performances:
                ax.plot(timeframe_labels, performances, marker='o',
                       linewidth=2, label=ticker, color=colors_list[i])

        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Performance (%)', fontsize=12)
        ax.set_title('Top Performers Evolution Across Timeframes', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        logger.error(f"Error creating evolution tracking chart: {e}")
        return False

def _create_evolution_summary_table(df: pd.DataFrame, top_performers: list, timeframes: dict, styles) -> list:
    """Create evolution summary table."""
    content = []

    content.append(Paragraph("<b>Evolution Summary Table</b>", styles['Heading3']))
    content.append(Spacer(1, 0.1*inch))

    # Create headers
    headers = ['Ticker'] + [tf.split('(')[0].strip() for tf in timeframes.keys()] + ['Pattern']

    table_data = [headers]

    for ticker in top_performers[:15]:  # Top 15 for table
        stock_data = df[df['ticker'] == ticker]
        if stock_data.empty:
            continue

        stock_data = stock_data.iloc[0]
        row_data = [ticker]

        performances = []
        for tf_name, tf_col in timeframes.items():
            if tf_col in stock_data.index and not pd.isna(stock_data[tf_col]):
                perf = stock_data[tf_col]
                row_data.append(f"{perf:.1f}%")
                performances.append(perf)
            else:
                row_data.append("N/A")

        # Classify pattern
        pattern = _classify_evolution_pattern(performances) if performances else "Unknown"
        row_data.append(pattern)

        table_data.append(row_data)

    evolution_table = Table(table_data)
    evolution_table.setStyle(TableStyle([
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

    content.append(evolution_table)
    content.append(Spacer(1, 0.2*inch))

    return content

def _create_technical_indicators_table(df: pd.DataFrame, top_performers: list, styles) -> list:
    """Create technical indicators analysis table."""
    content = []

    content.append(Paragraph("<b>Technical Indicators Analysis</b>", styles['Heading3']))
    content.append(Spacer(1, 0.2*inch))

    # Technical indicators columns
    tech_cols = {
        'RSI': 'daily_rsi_14',
        'MACD': 'daily_macd',
        'Price vs SMA20': 'daily_price2_sma20pct',
        'Price vs SMA50': 'daily_price2_sma50pct',
        '52W Position': 'daily_price_position_52w'
    }

    headers = ['Ticker', 'Current Price'] + list(tech_cols.keys()) + ['Technical Score']
    table_data = [headers]

    for ticker in top_performers[:15]:
        stock_data = df[df['ticker'] == ticker]
        if stock_data.empty:
            continue

        stock_data = stock_data.iloc[0]
        row_data = [ticker]

        # Current price
        current_price = stock_data.get('current_price', 0)
        row_data.append(f"${current_price:.2f}" if not pd.isna(current_price) else "N/A")

        # Technical indicators
        tech_values = []
        for tech_name, tech_col in tech_cols.items():
            if tech_col in stock_data.index and not pd.isna(stock_data[tech_col]):
                tech_val = stock_data[tech_col]
                if 'pct' in tech_col or 'position' in tech_col:
                    row_data.append(f"{tech_val:.1f}%")
                else:
                    row_data.append(f"{tech_val:.2f}")
                tech_values.append(abs(tech_val))
            else:
                row_data.append("N/A")

        # Calculate technical score
        if tech_values:
            tech_score = np.mean(tech_values)
            row_data.append(f"{tech_score:.0f}")
        else:
            row_data.append("N/A")

        table_data.append(row_data)

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

def _create_master_summary_table(df: pd.DataFrame, top_performers: list, timeframes: dict, styles) -> list:
    """Create master summary table with all data."""
    content = []

    content.append(Paragraph("<b>Master Summary - All Top Performers</b>", styles['Heading3']))
    content.append(Spacer(1, 0.2*inch))

    # Limit timeframes for table width
    limited_timeframes = dict(list(timeframes.items())[:4])

    headers = ['Ticker'] + [tf.split('(')[0].strip() for tf in limited_timeframes.keys()] + ['RSI', 'Tech Score']
    table_data = [headers]

    for ticker in top_performers[:20]:
        stock_data = df[df['ticker'] == ticker]
        if stock_data.empty:
            continue

        stock_data = stock_data.iloc[0]
        row_data = [ticker]

        # Performance data
        for tf_name, tf_col in limited_timeframes.items():
            if tf_col in stock_data.index and not pd.isna(stock_data[tf_col]):
                row_data.append(f"{stock_data[tf_col]:.1f}%")
            else:
                row_data.append("N/A")

        # Technical data
        rsi = stock_data.get('daily_rsi_14', 0)
        row_data.append(f"{rsi:.0f}" if not pd.isna(rsi) else "N/A")

        # Technical score (simplified)
        tech_score = rsi if not pd.isna(rsi) else 0
        row_data.append(f"{tech_score:.0f}")

        table_data.append(row_data)

    master_table = Table(table_data)
    master_table.setStyle(TableStyle([
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

    content.append(master_table)

    return content

def _create_comprehensive_heatmap(df: pd.DataFrame, timeframes: dict, output_path: str) -> bool:
    """Create comprehensive performance heatmap."""
    try:
        # Get top performers across all timeframes
        all_performers = set()
        for tf_name, tf_col in timeframes.items():
            period_data = df[df[tf_col].notna()]
            if not period_data.empty:
                top_performers = period_data.nlargest(10, tf_col)['ticker'].tolist()
                all_performers.update(top_performers)

        if not all_performers:
            return False

        # Create heatmap data
        heatmap_data = []
        tickers = list(all_performers)[:20]  # Limit to 20 for readability

        for ticker in tickers:
            stock_data = df[df['ticker'] == ticker]
            if stock_data.empty:
                continue

            stock_data = stock_data.iloc[0]
            row_data = []

            for tf_name, tf_col in timeframes.items():
                if tf_col in stock_data.index and not pd.isna(stock_data[tf_col]):
                    row_data.append(stock_data[tf_col])
                else:
                    row_data.append(0)

            heatmap_data.append(row_data)

        if not heatmap_data:
            return False

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto')

        # Set labels
        ax.set_xticks(range(len(timeframes)))
        ax.set_xticklabels([tf.split('(')[0].strip() for tf in timeframes.keys()], rotation=45)
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance (%)', rotation=270, labelpad=15)

        # Add value annotations
        for i in range(len(tickers)):
            for j in range(len(timeframes)):
                text = ax.text(j, i, f'{heatmap_array[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=6)

        ax.set_title('Comprehensive Performance Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        logger.error(f"Error creating comprehensive heatmap: {e}")
        return False

def _classify_evolution_pattern(performances: list) -> str:
    """Classify evolution pattern based on performance data."""
    if not performances or len(performances) < 2:
        return "Insufficient Data"

    # Calculate trend
    trend = np.polyfit(range(len(performances)), performances, 1)[0]

    # Calculate volatility
    volatility = np.std(performances)

    if trend > 5 and volatility < 10:
        return "Consistent Accelerator"
    elif trend > 2:
        return "Momentum Leader"
    elif trend < -2:
        return "Declining Leader"
    elif volatility > 20:
        return "Volatile Performer"
    else:
        return "Stable Performer"

def _analyze_performance_patterns(df: pd.DataFrame, timeframes: dict) -> list:
    """Analyze patterns and generate insights."""
    insights = []

    try:
        # Get top performers for each timeframe
        period_leaders = {}
        for tf_name, tf_col in timeframes.items():
            period_data = df[df[tf_col].notna()]
            if not period_data.empty:
                top_10 = period_data.nlargest(10, tf_col)
                period_leaders[tf_name] = {
                    'tickers': top_10['ticker'].tolist(),
                    'avg_performance': top_10[tf_col].mean(),
                    'top_performance': top_10[tf_col].max()
                }

        # Analyze consistency
        all_tickers = set()
        for leaders in period_leaders.values():
            all_tickers.update(leaders['tickers'])

        consistent_leaders = []
        for ticker in all_tickers:
            appearances = sum(1 for leaders in period_leaders.values() if ticker in leaders['tickers'])
            if appearances >= len(timeframes) * 0.6:  # Appears in 60%+ of timeframes
                consistent_leaders.append(ticker)

        if consistent_leaders:
            insights.append(f"Consistent leaders appearing across multiple timeframes: {', '.join(consistent_leaders[:5])}")

        # Analyze performance trends
        best_period = max(period_leaders.items(), key=lambda x: x[1]['avg_performance'])
        insights.append(f"Strongest average performance period: {best_period[0]} with {best_period[1]['avg_performance']:.1f}% average")

        # Top absolute performer
        top_absolute = max(period_leaders.items(), key=lambda x: x[1]['top_performance'])
        insights.append(f"Highest individual performance: {top_absolute[1]['top_performance']:.1f}% in {top_absolute[0]} period")

        # Market breadth analysis
        total_performers = len(all_tickers)
        insights.append(f"Market leadership breadth: {total_performers} unique stocks identified as top performers")

        return insights

    except Exception as e:
        logger.error(f"Error analyzing performance patterns: {e}")
        return ["Error analyzing performance patterns"]