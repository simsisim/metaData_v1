"""
Market Breadth PDF Report Generator
==================================

Generates comprehensive PDF reports combining market breadth PNG charts with tabular analysis
using enhanced CSV data with signal strength calculations and threshold classifications.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

from reportlab.lib.pagesizes import letter, A4, LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

logger = logging.getLogger(__name__)


class BreadthReportGenerator:
    """
    Generate comprehensive market breadth PDF reports.

    Features:
    - Market breadth PNG chart embedding
    - Threshold documentation tables
    - Signal strength classification tables
    - Latest N days analysis with real signal data from CSV
    """

    def __init__(self, user_config=None):
        self.user_config = user_config
        self.breadth_thresholds_data = [
            ["Threshold Name", "Default Value", "Signal Type", "Condition & Effect"],
            ["market_breadth_strong_ma_breadth_threshold", "80", "Breadth Thrust (Bullish)", "Overall breadth score above this triggers a bullish thrust"],
            ["market_breadth_weak_ma_breadth_threshold", "20", "Breadth Deterioration (Bearish)", "Overall breadth score below this triggers bearish signal"],
            ["market_breadth_daily_252day_new_highs_threshold", "100", "New Highs Expansion (Bullish)", "Net 52-week new highs above threshold triggers bullish signal"],
            ["market_breadth_strong_ad_ratio_threshold", "2", "Advance/Decline Thrust (Bullish)", "Advance/Decline ratio above threshold triggers bullish thrust"]
        ]

        self.signal_strength_data = [
            ["Signal Strength", "Criteria"],
            ["Strong Bullish", "≥ 3 bullish signals with net score ≥ 2"],
            ["Moderate Bullish", "≥ 2 bullish signals with net score ≥ 1"],
            ["Weak Signal (Bullish)", "Any bullish signals but low net score"],
            ["Strong Bearish", "≥ 3 bearish signals with net score ≤ -2"],
            ["Moderate Bearish", "≥ 2 bearish signals with net score ≤ -1"],
            ["Weak Signal (Bearish)", "Any bearish signals but low net score"],
            ["Neutral", "No significant signals"]
        ]

    def is_enabled(self) -> bool:
        """Check if market breadth reporting is enabled."""
        if not self.user_config:
            return True  # Default enabled if no config
        return getattr(self.user_config, 'market_breadth_report_enable', True)

    def get_template_type(self) -> str:
        """Get the configured template type."""
        if not self.user_config:
            return 'standard'
        return getattr(self.user_config, 'market_breadth_report_template_type', 'standard')

    def generate_report(self, csv_path: Path, png_path: Path, output_path: Path,
                       latest_days: int = 10) -> bool:
        """
        Generate comprehensive market breadth PDF report.

        Args:
            csv_path: Path to market breadth CSV file with enhanced signal data
            png_path: Path to market breadth PNG chart
            output_path: Path for output PDF file
            latest_days: Number of recent trading days to analyze

        Returns:
            bool: True if report generated successfully
        """
        try:
            # Check if reporting is enabled
            if not self.is_enabled():
                logger.info("Market breadth reporting is disabled")
                return False

            logger.info(f"Generating market breadth report: {output_path}")
            logger.info(f"Using template type: {self.get_template_type()}")

            # Load and process CSV data
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return False

            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

            # Create PDF document with landscape orientation for better table fit
            doc = SimpleDocTemplate(str(output_path), pagesize=landscape(LETTER),
                                  rightMargin=36, leftMargin=36,
                                  topMargin=36, bottomMargin=36)

            # Build story (PDF content)
            story = []
            styles = getSampleStyleSheet()

            # Add report sections
            self._add_title_section(story, styles, csv_path)
            self._add_chart_section(story, png_path)
            self._add_current_analysis_section(story, styles, df, latest_days)
            self._add_documentation_section(story, styles)

            # Build PDF
            doc.build(story)

            logger.info(f"Successfully generated report: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False

    def _add_title_section(self, story: List, styles, csv_path: Path):
        """Add report title and metadata."""
        story.append(Paragraph("Market Breadth Analysis Report", styles["Title"]))
        story.append(Spacer(1, 12))

        # Extract metadata from filename
        filename = csv_path.stem
        parts = filename.replace('market_breadth_', '').split('_')
        universe = parts[0] if parts else "Unknown"

        metadata_text = f"<b>Universe:</b> {universe}<br/><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/><b>Data Source:</b> {csv_path.name}"
        story.append(Paragraph(metadata_text, styles["BodyText"]))
        story.append(Spacer(1, 24))

    def _add_chart_section(self, story: List, png_path: Path):
        """Add market breadth PNG chart."""
        if png_path and png_path.exists():
            try:
                # Add chart with proper scaling for landscape
                img = Image(str(png_path), width=9*inch, height=6*inch)
                story.append(img)
                story.append(Spacer(1, 24))
            except Exception as e:
                logger.warning(f"Could not embed chart {png_path}: {e}")
                story.append(Paragraph(f"<i>Chart not available: {png_path}</i>", styles["BodyText"]))
                story.append(Spacer(1, 12))
        else:
            logger.warning(f"PNG chart not found: {png_path}")
            story.append(Paragraph("<i>Chart not available</i>", styles["BodyText"]))
            story.append(Spacer(1, 12))

    def _add_documentation_section(self, story: List, styles):
        """Add threshold and signal strength documentation tables."""
        # Breadth Thresholds Table
        story.append(Paragraph("<b>Market Breadth Thresholds</b>", styles["Heading2"]))
        story.append(Paragraph("These thresholds define when bull or bear market breadth signals are triggered.", styles["BodyText"]))
        story.append(Spacer(1, 12))

        threshold_table = Table(self.breadth_thresholds_data,
                               colWidths=[2.8*inch, 0.8*inch, 1.8*inch, 4.2*inch],
                               hAlign='LEFT')
        threshold_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(threshold_table)
        story.append(Spacer(1, 24))

        # Signal Strength Classification Table
        story.append(Paragraph("<b>Signal Strength Classification</b>", styles["Heading2"]))
        story.append(Paragraph("Classification based on counts of bullish/bearish indicators and net scores.", styles["BodyText"]))
        story.append(Spacer(1, 12))

        signal_table = Table(self.signal_strength_data,
                            colWidths=[2*inch, 8*inch],
                            hAlign='LEFT')
        signal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(signal_table)
        story.append(Spacer(1, 24))

    def _add_current_analysis_section(self, story: List, styles, df: pd.DataFrame, latest_days: int):
        """Add current market analysis with latest N days."""
        story.append(Paragraph(f"<b>Latest {latest_days} Trading Days Analysis</b>", styles["Heading2"]))
        story.append(Paragraph("Current market breadth conditions with signal strength classifications.", styles["BodyText"]))
        story.append(Spacer(1, 12))

        # Get latest data
        latest_data = self._extract_latest_data(df, latest_days)

        if latest_data.empty:
            story.append(Paragraph("<i>No recent data available for analysis.</i>", styles["BodyText"]))
            return

        # Create analysis table
        analysis_table_data = self._create_analysis_table_data(latest_data)

        analysis_table = Table(analysis_table_data,
                              colWidths=[0.7*inch, 0.5*inch, 1.0*inch, 0.6*inch, 0.4*inch, 0.4*inch, 1.0*inch, 2.8*inch],
                              hAlign='LEFT')

        # Apply styling with color coding for signal strength
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]

        # Add color coding for signal strength
        for i, row_data in enumerate(analysis_table_data[1:], 1):  # Skip header
            signal_strength = row_data[6]  # Signal strength column
            if 'Strong Bullish' in signal_strength:
                table_style.append(('BACKGROUND', (6, i), (6, i), colors.darkgreen))
                table_style.append(('TEXTCOLOR', (6, i), (6, i), colors.whitesmoke))
            elif 'Moderate Bullish' in signal_strength:
                table_style.append(('BACKGROUND', (6, i), (6, i), colors.lightgreen))
            elif 'Strong Bearish' in signal_strength:
                table_style.append(('BACKGROUND', (6, i), (6, i), colors.darkred))
                table_style.append(('TEXTCOLOR', (6, i), (6, i), colors.whitesmoke))
            elif 'Moderate Bearish' in signal_strength:
                table_style.append(('BACKGROUND', (6, i), (6, i), colors.lightcoral))
            elif 'Weak Signal' in signal_strength:
                table_style.append(('BACKGROUND', (6, i), (6, i), colors.lightyellow))

        analysis_table.setStyle(TableStyle(table_style))
        story.append(analysis_table)
        story.append(Spacer(1, 12))

        # Add summary statistics
        self._add_summary_statistics(story, styles, latest_data)

    def _extract_latest_data(self, df: pd.DataFrame, latest_days: int) -> pd.DataFrame:
        """Extract latest N days with required columns."""
        try:
            # Get the latest data
            latest_df = df.tail(latest_days).copy()

            # Check for required columns with proper prefixes
            required_columns = []
            for col in df.columns:
                if any(suffix in col for suffix in [
                    '_ad_ratio', '_pct_above_ma_20', '_pct_above_ma_50', '_pct_above_ma_200',
                    '_net_long_new_highs', '_total_bullish_signals', '_total_bearish_signals',
                    '_net_signal_score', '_signal_strength', '_threshold_classification'
                ]):
                    required_columns.append(col)

            # Always include date column
            if 'date' in df.columns:
                required_columns.insert(0, 'date')

            # Filter to available columns
            available_columns = [col for col in required_columns if col in df.columns]

            if not available_columns:
                logger.warning("No required analysis columns found in CSV")
                return pd.DataFrame()

            return latest_df[available_columns]

        except Exception as e:
            logger.error(f"Error extracting latest data: {e}")
            return pd.DataFrame()

    def _create_analysis_table_data(self, df: pd.DataFrame) -> List[List[str]]:
        """Create formatted table data for analysis section."""
        try:
            # Table headers
            headers = ["Date", "AD Ratio", "MA Breadth %", "Net Highs", "Bull Sigs", "Bear Sigs", "Signal Strength", "Threshold Class"]
            table_data = [headers]

            # Find column names (they may have prefixes)
            date_col = self._find_column(df, 'date')
            ad_ratio_col = self._find_column(df, '_ad_ratio')
            ma20_col = self._find_column(df, '_pct_above_ma_20')
            ma50_col = self._find_column(df, '_pct_above_ma_50')
            ma200_col = self._find_column(df, '_pct_above_ma_200')
            net_highs_col = self._find_column(df, '_net_long_new_highs')
            bullish_col = self._find_column(df, '_total_bullish_signals')
            bearish_col = self._find_column(df, '_total_bearish_signals')
            strength_col = self._find_column(df, '_signal_strength')
            threshold_col = self._find_column(df, '_threshold_classification')

            # Process each row
            for _, row in df.iterrows():
                date_str = str(row[date_col])[:10] if date_col else "N/A"
                ad_ratio = f"{float(row[ad_ratio_col]):.2f}" if ad_ratio_col and pd.notna(row[ad_ratio_col]) else "N/A"

                # Format MA breadth as "20/50/200"
                ma_breadth_parts = []
                for col in [ma20_col, ma50_col, ma200_col]:
                    if col and pd.notna(row[col]):
                        ma_breadth_parts.append(f"{float(row[col]):.0f}")
                    else:
                        ma_breadth_parts.append("--")
                ma_breadth = "/".join(ma_breadth_parts)

                net_highs = f"{int(row[net_highs_col])}" if net_highs_col and pd.notna(row[net_highs_col]) else "N/A"
                bullish = f"{int(row[bullish_col])}" if bullish_col and pd.notna(row[bullish_col]) else "N/A"
                bearish = f"{int(row[bearish_col])}" if bearish_col and pd.notna(row[bearish_col]) else "N/A"
                strength = str(row[strength_col]) if strength_col else "N/A"
                threshold = str(row[threshold_col]) if threshold_col else "N/A"

                table_data.append([date_str, ad_ratio, ma_breadth, net_highs, bullish, bearish, strength, threshold])

            return table_data

        except Exception as e:
            logger.error(f"Error creating analysis table: {e}")
            return [["Error creating table data"]]

    def _find_column(self, df: pd.DataFrame, suffix: str) -> Optional[str]:
        """Find column with given suffix."""
        for col in df.columns:
            if col.endswith(suffix):
                return col
        return None

    def _add_summary_statistics(self, story: List, styles, df: pd.DataFrame):
        """Add summary statistics section."""
        strength_col = self._find_column(df, '_signal_strength')

        if strength_col:
            strength_counts = df[strength_col].value_counts()
            summary_text = f"<b>Signal Strength Summary (Last {len(df)} days):</b><br/>"
            for strength, count in strength_counts.items():
                summary_text += f"• {strength}: {count} days<br/>"

            story.append(Paragraph(summary_text, styles["BodyText"]))


def main():
    """Command line interface for generating breadth reports."""
    import argparse
    from src.config import Config

    parser = argparse.ArgumentParser(description='Generate Market Breadth PDF Report')
    parser.add_argument('--csv', required=True, help='Path to market breadth CSV file')
    parser.add_argument('--png', help='Path to market breadth PNG chart')
    parser.add_argument('--output', required=True, help='Output PDF file path')
    parser.add_argument('--days', type=int, default=10, help='Number of recent days to analyze')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Generate report
    generator = BreadthReportGenerator()
    csv_path = Path(args.csv)
    png_path = Path(args.png) if args.png else None
    output_path = Path(args.output)

    success = generator.generate_report(csv_path, png_path, output_path, args.days)

    if success:
        print(f"✅ Report generated successfully: {output_path}")
    else:
        print("❌ Report generation failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())