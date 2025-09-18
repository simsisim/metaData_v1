import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image as RLImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import logging

logger = logging.getLogger(__name__)

class UniversalPngPdfGenerator:
    """
    Universal generator that creates PNG charts and PDF reports from CSV files.
    Follows the stage analysis pattern: CSV → PNG + PDF (with embedded PNG).
    """

    def __init__(self, output_dir_reports="results/reports"):
        """
        Initialize the generator.

        Args:
            output_dir_reports: Directory for PDF output files
        """
        self.output_dir_reports = Path(output_dir_reports)
        self.output_dir_reports.mkdir(parents=True, exist_ok=True)

        # Chart configuration
        self.chart_config = {
            'figsize': (12, 8),
            'dpi': 200,
            'style': 'whitegrid',
            'color_palette': 'Set2'
        }

        # PDF configuration
        self.pdf_config = {
            'pagesize': letter,
            'margins': {'right': 14, 'left': 14, 'top': 24, 'bottom': 24}
        }

    def detect_chart_type(self, csv_path: str, df: pd.DataFrame) -> str:
        """
        Detect chart type based on CSV filename and content.

        Args:
            csv_path: Path to CSV file
            df: DataFrame content

        Returns:
            Chart type string
        """
        filename = Path(csv_path).stem.lower()

        if 'stage_analysis' in filename:
            return 'stage_analysis'
        elif 'ftd_dd' in filename or 'distribution' in filename:
            return 'ftd_dd'
        elif 'ma_cycles' in filename:
            return 'ma_cycles'
        elif 'chillax' in filename:
            return 'chillax_ma'
        elif 'breadth' in filename:
            return 'market_breadth'
        elif 'screener' in filename:
            return 'screener'
        else:
            return 'generic'

    def generate_png_from_csv(self, csv_path: str) -> str:
        """
        Generate PNG chart from CSV data.

        Args:
            csv_path: Path to input CSV file

        Returns:
            Path to generated PNG file
        """
        try:
            # Read CSV data
            df = pd.read_csv(csv_path)

            # Detect chart type
            chart_type = self.detect_chart_type(csv_path, df)

            # Generate base filename from CSV
            csv_stem = Path(csv_path).stem
            png_path = Path(csv_path).parent / f"{csv_stem}.png"

            # Create chart based on type
            if chart_type == 'stage_analysis':
                self._create_stage_analysis_chart(df, png_path)
            elif chart_type == 'ftd_dd':
                self._create_ftd_dd_chart(df, png_path)
            elif chart_type == 'ma_cycles':
                self._create_ma_cycles_chart(df, png_path)
            elif chart_type == 'chillax_ma':
                self._create_chillax_chart(df, png_path)
            elif chart_type == 'market_breadth':
                self._create_breadth_chart(df, png_path)
            else:
                self._create_generic_chart(df, png_path)

            logger.info(f"Generated PNG chart: {png_path}")
            return str(png_path)

        except Exception as e:
            logger.error(f"Error generating PNG from CSV {csv_path}: {e}")
            raise

    def _create_stage_analysis_chart(self, df: pd.DataFrame, png_path: Path):
        """Create stage analysis pie chart."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        plt.style.use('default')

        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

        # Assuming stage analysis has 'daily_sa_name' column
        if 'daily_sa_name' in df.columns:
            stage_counts = df['daily_sa_name'].value_counts()

            # Enhanced color map with more stages
            colors_map = {
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

            colors = [colors_map.get(stage, '#999999') for stage in stage_counts.index]

            wedges, texts, autotexts = ax.pie(stage_counts.values, labels=stage_counts.index,
                                            autopct='%1.1f%%', colors=colors, startangle=90)

            ax.set_title('Market Stage Analysis Distribution', fontsize=16, fontweight='bold', pad=20)

            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
        else:
            ax.text(0.5, 0.5, 'No stage data available', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=200)
        plt.close(fig)  # Explicitly close figure
        plt.clf()      # Clear figure

    def _create_ftd_dd_chart(self, df: pd.DataFrame, png_path: Path):
        """Create FTD/DD signal chart."""
        plt.style.use('default')

        fig, ax = plt.subplots(figsize=self.chart_config['figsize'], dpi=self.chart_config['dpi'])

        # Basic FTD/DD visualization
        if 'date' in df.columns and 'signal_type' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

            ftd_data = df[df['signal_type'] == 'FTD']
            dd_data = df[df['signal_type'] == 'DD']

            ax.scatter(ftd_data['date'], ftd_data.get('price_change_pct', 0),
                      c='green', label='Follow Through Days', s=100, alpha=0.7)
            ax.scatter(dd_data['date'], dd_data.get('price_change_pct', 0),
                      c='red', label='Distribution Days', s=100, alpha=0.7)

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price Change %', fontsize=12)
            ax.set_title('Follow Through Days & Distribution Days Analysis', fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No FTD/DD data available', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_ma_cycles_chart(self, df: pd.DataFrame, png_path: Path):
        """Create MA cycles chart."""
        plt.style.use('default')

        fig, ax = plt.subplots(figsize=self.chart_config['figsize'], dpi=self.chart_config['dpi'])

        # Basic MA cycles visualization
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

            # Plot available MA columns
            ma_columns = [col for col in df.columns if 'ma' in col.lower() or 'moving' in col.lower()]

            for col in ma_columns[:5]:  # Limit to first 5 MA columns
                if df[col].notna().any():
                    ax.plot(df['date'], df[col], label=col, linewidth=2)

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Moving Average Cycles Analysis', fontsize=16, fontweight='bold')
            if ma_columns:
                ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No MA cycles data available', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_chillax_chart(self, df: pd.DataFrame, png_path: Path):
        """Create Chillax MA chart."""
        plt.style.use('default')

        fig, ax = plt.subplots(figsize=self.chart_config['figsize'], dpi=self.chart_config['dpi'])

        # Basic Chillax visualization
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

            # Plot price and chillax indicators
            if 'close' in df.columns:
                ax.plot(df['date'], df['close'], label='Price', linewidth=2, color='black')

            chillax_columns = [col for col in df.columns if 'chillax' in col.lower()]
            colors = ['blue', 'red', 'green', 'orange', 'purple']

            for i, col in enumerate(chillax_columns[:5]):
                if df[col].notna().any():
                    ax.plot(df['date'], df[col], label=col, linewidth=2, color=colors[i % len(colors)])

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.set_title('Chillax Moving Averages Analysis', fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Chillax data available', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_breadth_chart(self, df: pd.DataFrame, png_path: Path):
        """Create market breadth chart."""
        plt.style.use('default')

        fig, ax = plt.subplots(figsize=self.chart_config['figsize'], dpi=self.chart_config['dpi'])

        # Basic breadth visualization
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

            breadth_columns = [col for col in df.columns if any(term in col.lower()
                             for term in ['advance', 'decline', 'breadth', 'ratio'])]

            for col in breadth_columns[:3]:  # Limit to first 3 breadth columns
                if df[col].notna().any():
                    ax.plot(df['date'], df[col], label=col, linewidth=2)

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Market Breadth Analysis', fontsize=16, fontweight='bold')
            if breadth_columns:
                ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No breadth data available', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_generic_chart(self, df: pd.DataFrame, png_path: Path):
        """Create generic chart for unknown data types."""
        plt.style.use('default')

        fig, ax = plt.subplots(figsize=self.chart_config['figsize'], dpi=self.chart_config['dpi'])

        # Basic visualization
        numeric_columns = df.select_dtypes(include=['number']).columns[:5]

        if len(numeric_columns) > 0:
            for col in numeric_columns:
                ax.plot(df.index, df[col], label=col, linewidth=2)

            ax.set_xlabel('Index', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Data Analysis', fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No numeric data for visualization', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', facecolor='white')
        plt.close()

    def generate_pdf_report(self, csv_path: str, png_path: str) -> str:
        """
        Generate PDF report from CSV data and PNG chart.

        Args:
            csv_path: Path to input CSV file
            png_path: Path to PNG chart file

        Returns:
            Path to generated PDF file
        """
        try:
            # Read CSV data
            df = pd.read_csv(csv_path)

            # Generate PDF filename from CSV
            csv_stem = Path(csv_path).stem
            pdf_path = self.output_dir_reports / f"{csv_stem}.pdf"

            # Detect chart type for appropriate content
            chart_type = self.detect_chart_type(csv_path, df)

            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=self.pdf_config['pagesize'],
                **self.pdf_config['margins']
            )

            # Build story elements
            story = self._build_pdf_story(df, png_path, chart_type, csv_stem)

            # Build PDF
            doc.build(story)

            logger.info(f"Generated PDF report: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Error generating PDF report for {csv_path}: {e}")
            raise

    def _build_pdf_story(self, df: pd.DataFrame, png_path: str, chart_type: str, csv_stem: str):
        """Build the PDF story elements."""
        styles = getSampleStyleSheet()
        story = []

        # Title based on chart type
        title_map = {
            'stage_analysis': 'Market Stage Analysis',
            'ftd_dd': 'Follow Through Days & Distribution Days Analysis',
            'ma_cycles': 'Moving Average Cycles Analysis',
            'chillax_ma': 'Chillax Moving Averages Analysis',
            'market_breadth': 'Market Breadth Analysis',
            'screener': 'Stock Screener Results',
            'generic': 'Data Analysis Report'
        }

        title = title_map.get(chart_type, 'Analysis Report')
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 14))

        # Chart section
        story.append(Paragraph(f"{title} Chart", styles['Heading2']))
        if Path(png_path).exists():
            # Calculate optimal image size
            img_width = min(6 * inch, 500)
            img_height = img_width * 0.6  # Maintain aspect ratio
            story.append(RLImage(png_path, width=img_width, height=img_height))
        else:
            story.append(Paragraph("Chart not available", styles['Normal']))
        story.append(Spacer(1, 18))

        # Data summary section
        story.append(Paragraph("Data Summary", styles['Heading2']))
        story.append(Spacer(1, 12))

        # Basic statistics
        summary_data = [
            ['Metric', 'Value'],
            ['Total Records', str(len(df))],
            ['Date Range', self._get_date_range(df)],
            ['Columns', str(len(df.columns))],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]

        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10)
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 18))

        # Chart-specific data tables
        if chart_type == 'stage_analysis' and 'daily_sa_name' in df.columns:
            story.extend(self._create_stage_analysis_tables(df, styles))
        elif chart_type == 'ftd_dd':
            story.extend(self._create_ftd_dd_tables(df, styles))

        return story

    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Get date range from DataFrame."""
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            try:
                dates = pd.to_datetime(df[date_columns[0]], errors='coerce').dropna()
                if len(dates) > 0:
                    return f"{dates.min().date()} to {dates.max().date()}"
            except:
                pass
        return "Not available"

    def _create_stage_analysis_tables(self, df: pd.DataFrame, styles):
        """Create stage analysis specific tables matching original format."""
        story_elements = []

        # Stage distribution table (summary)
        story_elements.append(Paragraph("Stage Distribution", styles['Heading3']))

        stage_counts = df['daily_sa_name'].value_counts().reset_index()
        stage_counts.columns = ['Stage', 'Count']
        stage_counts['Percentage'] = (stage_counts['Count'] / len(df) * 100).round(1)

        table_data = [['Stage', 'Count', 'Percentage']]
        for _, row in stage_counts.iterrows():
            table_data.append([row['Stage'], str(row['Count']), f"{row['Percentage']}%"])

        table = Table(table_data, colWidths=[2*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9)
        ]))

        story_elements.append(table)
        story_elements.append(Spacer(1, 18))

        # Stage Table by Market Stage (columnar format like original)
        story_elements.append(Paragraph("Stage Table by Market Stage", styles['Heading3']))
        story_elements.append(Spacer(1, 12))

        # Get unique stages and color codes for table
        stage_triplets = df[['daily_sa_name', 'daily_sa_code', 'daily_sa_color_code']].drop_duplicates().values.tolist()

        # Build columns for each stage
        columns = []
        for sa_name, sa_code, sa_color in stage_triplets:
            tickers = df[df['daily_sa_name'] == sa_name]['ticker'].tolist()
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

            # Color mapping for backgrounds (matching CSV color codes)
            CODE_TO_COLOR = {
                'green_light': colors.HexColor('#C8E6C9'),    # Bullish Trend
                'orange_light': colors.HexColor('#FFE0B2'),   # Bullish Fade
                'purple': colors.HexColor('#E1BEE7'),         # Launch Pad
                'red_light': colors.HexColor('#FFCDD2'),      # Bearish Trend
                'red': colors.HexColor('#FFEBEE'),            # Bearish Confirmation
                'yellow': colors.HexColor('#FFF9C4'),         # Mean Reversion
                'yellow_light': colors.HexColor('#FFFDE7'),   # Pullback
                'gray': colors.HexColor('#F5F5F5'),           # Undefined
                'gray_light': colors.HexColor('#FAFAFA'),     # Upward Pivot
                'lime': colors.HexColor('#F1F8E9'),           # Breakout Confirmation
                'orange_dark': colors.HexColor('#FFE0B2'),    # Fade Confirmation
            }

            # Get background colors for each column
            col_bg_colors = [CODE_TO_COLOR.get(color_code, colors.white) for _, color_code in columns]

            # Create table style
            table_style = TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,1), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 8)
            ])

            # Add background colors for each column
            for col_idx, bg_color in enumerate(col_bg_colors):
                table_style.add('BACKGROUND', (col_idx, 0), (col_idx, max_len-1), bg_color)

            # Calculate column widths dynamically
            col_width = min(60, 400 / num_stages)  # Max 400pt total width
            col_widths = [col_width] * num_stages

            stage_table = Table(table_data, colWidths=col_widths)
            stage_table.setStyle(table_style)
            story_elements.append(stage_table)

        return story_elements

    def _create_ftd_dd_tables(self, df: pd.DataFrame, styles):
        """Create FTD/DD specific tables."""
        story_elements = []

        if 'signal_type' in df.columns:
            # Signal summary
            story_elements.append(Paragraph("Signal Summary", styles['Heading3']))

            signal_counts = df['signal_type'].value_counts().reset_index()
            signal_counts.columns = ['Signal Type', 'Count']

            table_data = [['Signal Type', 'Count']]
            for _, row in signal_counts.iterrows():
                table_data.append([row['Signal Type'], str(row['Count'])])

            table = Table(table_data, colWidths=[2*inch, 1*inch])
            table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9)
            ]))

            story_elements.append(table)

        return story_elements

    def process_csv_file(self, csv_path: str):
        """
        Complete processing: CSV → PNG + PDF.

        Args:
            csv_path: Path to input CSV file

        Returns:
            Tuple of (png_path, pdf_path)
        """
        try:
            logger.info(f"Processing CSV file: {csv_path}")

            # Generate PNG chart
            png_path = self.generate_png_from_csv(csv_path)

            # Generate PDF report
            pdf_path = self.generate_pdf_report(csv_path, png_path)

            logger.info(f"Successfully processed {csv_path} → PNG: {png_path}, PDF: {pdf_path}")
            return png_path, pdf_path

        except Exception as e:
            logger.error(f"Error processing CSV file {csv_path}: {e}")
            raise

def process_csv_to_png_pdf(csv_path: str, output_dir_reports: str = "results/reports"):
    """
    Convenience function to process a single CSV file.

    Args:
        csv_path: Path to CSV file
        output_dir_reports: Output directory for PDF files

    Returns:
        Tuple of (png_path, pdf_path)
    """
    generator = UniversalPngPdfGenerator(output_dir_reports)
    return generator.process_csv_file(csv_path)