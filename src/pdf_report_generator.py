"""
PDF Report Generator
===================

Automated PDF report generation for financial analysis data.
Creates comprehensive reports combining data tables and tornado charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Image, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    
logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generates automated PDF reports from CSV data with embedded charts."""
    
    def __init__(self, config, user_config):
        """Initialize PDF report generator."""
        self.config = config
        self.user_config = user_config
        self.output_dir = config.directories['RESULTS_DIR'] / 'reports'
        self.output_dir.mkdir(exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. Install with: pip install reportlab")
    
    def generate_index_analysis_report(self, csv_file_path: str) -> Optional[str]:
        """
        Generate PDF report from index percentage analysis CSV.
        
        Args:
            csv_file_path: Path to SP500_NASDAQ100_Percentage_Analysis CSV file
            
        Returns:
            Path to generated PDF report or None if failed
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not installed - cannot generate PDF reports")
            return None
            
        try:
            csv_path = Path(csv_file_path)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return None
                
            # Extract timestamp from filename for output naming
            timestamp = self._extract_timestamp_from_filename(csv_path.name)
            
            # Load data
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV data: {len(df)} rows, {len(df.columns)} columns")
            
            # Generate tornado chart using existing implementation
            chart_path = self._generate_tornado_chart(str(csv_path), self.output_dir)
            
            # Generate PDF report
            pdf_path = self._generate_pdf_report(df, chart_path, csv_path.stem, timestamp)
            
            logger.info(f"PDF report generated: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return None
    
    def _extract_timestamp_from_filename(self, filename: str) -> str:
        """Extract timestamp from filename like SP500_NASDAQ100_Percentage_Analysis_daily_20250904_125556.csv"""
        parts = filename.split('_')
        if len(parts) >= 2:
            # Look for date_time pattern
            for i, part in enumerate(parts):
                if len(part) == 8 and part.isdigit():  # YYYYMMDD
                    if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():  # HHMMSS
                        return f"{part}_{parts[i + 1]}"
                    return part
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _generate_tornado_chart(self, csv_file_path: str, output_dir: Path) -> Optional[str]:
        """Generate tornado chart using existing implementation from index_overview_file."""
        try:
            # Import the existing tornado chart function
            from .index_overview_file import generate_tornado_chart_from_analysis
            
            # Use the existing implementation that creates the correct tornado chart
            chart_path = generate_tornado_chart_from_analysis(csv_file_path, output_dir)
            
            if chart_path:
                logger.info(f"Tornado chart generated using existing implementation: {chart_path}")
                return chart_path
            else:
                logger.warning("Tornado chart generation failed")
                return None
                
        except Exception as e:
            logger.error(f"Error generating tornado chart: {e}")
            return None
    
    def _generate_pdf_report(self, df: pd.DataFrame, chart_path: Optional[str], 
                           base_name: str, timestamp: str) -> Optional[str]:
        """
        Generate PDF report with data table and tornado chart.
        
        Args:
            df: DataFrame with analysis data
            chart_path: Path to tornado chart image
            base_name: Base filename
            timestamp: Timestamp for filename
            
        Returns:
            Path to generated PDF
        """
        if not REPORTLAB_AVAILABLE:
            return None
            
        try:
            # Create PDF filename
            pdf_filename = f"{base_name}_Report_{timestamp}.pdf"
            pdf_path = self.output_dir / pdf_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle('CustomTitle', fontSize=16, alignment=1, spaceAfter=20)
            title = Paragraph(f"Index Percentage Analysis Report<br/>{timestamp[:8]}", title_style)
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Data table
            table_data = self._prepare_table_data(df)
            table = Table(table_data)
            
            # Style the table
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Add tornado chart if available
            if chart_path and Path(chart_path).exists():
                story.append(Paragraph("Tornado Chart Analysis", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                # Add chart image
                img = Image(chart_path, width=7*inch, height=5*inch)
                story.append(img)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            return None
    
    def _prepare_table_data(self, df: pd.DataFrame) -> List[List[str]]:
        """Prepare DataFrame data for ReportLab table format."""
        table_data = []
        
        # Headers
        headers = list(df.columns)
        table_data.append(headers)
        
        # Data rows
        for _, row in df.iterrows():
            row_data = []
            for col in df.columns:
                value = row[col]
                
                # Format different data types
                if pd.isna(value):
                    row_data.append("")
                elif isinstance(value, (int, float)):
                    if col.endswith('%'):
                        row_data.append(f"{value:.1f}%")
                    else:
                        row_data.append(f"{value:.2f}")
                else:
                    row_data.append(str(value))
            
            table_data.append(row_data)
        
        return table_data
    
    def process_latest_analysis_files(self) -> List[str]:
        """
        Process all recent analysis files and generate PDF reports.
        
        Returns:
            List of generated PDF report paths
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not installed - cannot generate PDF reports")
            return []
            
        try:
            # Find recent percentage analysis files
            overview_dir = self.config.directories['RESULTS_DIR'] / 'overview'
            
            if not overview_dir.exists():
                logger.warning(f"Overview directory not found: {overview_dir}")
                return []
            
            # Find percentage analysis CSV files
            analysis_files = list(overview_dir.glob("*Percentage_Analysis_*.csv"))
            
            if not analysis_files:
                logger.warning("No percentage analysis files found")
                return []
            
            # Sort by modification time (newest first)
            analysis_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            generated_reports = []
            
            # Process each file (limit to most recent 3 for performance)
            for csv_file in analysis_files[:3]:
                logger.info(f"Processing analysis file: {csv_file.name}")
                
                pdf_path = self.generate_index_analysis_report(str(csv_file))
                if pdf_path:
                    generated_reports.append(pdf_path)
            
            logger.info(f"Generated {len(generated_reports)} PDF reports")
            return generated_reports
            
        except Exception as e:
            logger.error(f"Failed to process analysis files: {e}")
            return []


def generate_pdf_reports_for_latest_analysis(config, user_config) -> List[str]:
    """
    Convenience function to generate PDF reports for latest analysis files.
    
    Args:
        config: Config object
        user_config: User configuration
        
    Returns:
        List of generated PDF report paths
    """
    generator = PDFReportGenerator(config, user_config)
    return generator.process_latest_analysis_files()


def generate_pdf_report_from_csv(csv_file_path: str, config, user_config) -> Optional[str]:
    """
    Generate single PDF report from CSV file.
    
    Args:
        csv_file_path: Path to CSV file
        config: Config object  
        user_config: User configuration
        
    Returns:
        Path to generated PDF or None if failed
    """
    generator = PDFReportGenerator(config, user_config)
    return generator.generate_index_analysis_report(csv_file_path)