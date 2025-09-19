"""
Report Generator
================

Generates comprehensive reports using data from CSV files directly.
Automatically finds latest files by scanning the results directory.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, A3, landscape, portrait
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates automated reports using CSV files directly."""
    
    def __init__(self, config, user_config):
        """Initialize report generator."""
        self.config = config
        self.user_config = user_config
        self.output_dir = Path(getattr(user_config, 'report_output_dir', 'results/reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results directory for CSV files
        self.results_dir = config.directories.get('RESULTS_DIR')
        
        # Report configuration
        self.template_type = getattr(user_config, 'report_template_type', 'market_analysis')
        self.page_size = getattr(user_config, 'report_page_size', 'A4_landscape')
        self.include_charts = getattr(user_config, 'report_include_charts', True)
        self.file_dates_auto = getattr(user_config, 'report_file_dates_auto', True)
        self.file_dates_manual = getattr(user_config, 'report_file_dates_manual', '')
        self.format = getattr(user_config, 'report_format', 'PDF')
        
        # Report sections configuration
        self.sections = {
            'basic_stats': getattr(user_config, 'report_sections_basic_stats', True),
            'percentage_analysis': getattr(user_config, 'report_sections_percentage_analysis', True),
            'rs_analysis': getattr(user_config, 'report_sections_rs_analysis', True),
            'tornado_charts': getattr(user_config, 'report_sections_tornado_charts', True),
            'summary': getattr(user_config, 'report_sections_summary', True)
        }
        
        self.max_tickers_display = getattr(user_config, 'report_max_tickers_display', 20)
        self.include_metadata = getattr(user_config, 'report_include_metadata', True)
    
    def generate_report(self, data_dates: Optional[Dict[str, Dict[str, str]]] = None) -> Optional[Path]:
        """
        Generate comprehensive report using centralized data dates.
        
        Args:
            data_dates: Centralized data dates from main.py (optional - will auto-discover if None)
            
        Returns:
            Path to generated report or None if failed
        """
        try:
            logger.info(f"Starting report generation with template: {self.template_type}")
            print(f"📄 Generating Report: {self.template_type}")
            
            # Determine file dates to use
            if self.file_dates_auto or not self.file_dates_manual:
                file_dates = self._get_latest_file_dates(data_dates)
            else:
                file_dates = self._parse_manual_file_dates()
            
            if not file_dates:
                logger.error("No file dates available for report generation")
                return None
            
            print(f"  📅 Using data dates: {file_dates}")
            
            # Load data for each timeframe
            report_data = {}
            for timeframe, date_info in file_dates.items():
                data = self._load_data_for_timeframe(timeframe, date_info['data_date'])
                if data:
                    report_data[timeframe] = data
            
            if not report_data:
                logger.error("No data loaded for report generation")
                return None
            
            # Generate report based on template type
            if self.template_type == 'indexes_overview':
                logger.warning("indexes_overview template is deprecated (index_overview module removed)")
                print(f"⚠️  Template 'indexes_overview' is deprecated. Using 'market_analysis' instead.")
                self.template_type = 'market_analysis'
                return self._generate_market_analysis_report(report_data, file_dates)
            elif self.template_type == 'market_analysis':
                return self._generate_market_analysis_report(report_data, file_dates)
            elif self.template_type == 'full_analysis':
                return self._generate_full_analysis_report(report_data, file_dates)
            else:
                logger.error(f"Unknown report template: {self.template_type}")
                logger.error("Available templates: market_analysis, full_analysis")
                return None
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            print(f"  ❌ Report generation failed: {e}")
            return None
    
    def _get_latest_file_dates(self, data_dates: Optional[Dict] = None) -> Dict[str, Dict[str, str]]:
        """Get latest file dates from centralized system or CSV file scanning."""
        if data_dates:
            # Use centralized data dates from main.py
            return data_dates
        
        # Fall back to scanning CSV files for latest dates
        latest_files = {}
        for timeframe in ['daily', 'weekly', 'monthly']:
            # Scan for basic calculation files
            pattern = f'basic_calculation_*_{timeframe}_*.csv'
            basic_calc_dir = self.config.directories['BASIC_CALCULATION_DIR']
            matching_files = list(basic_calc_dir.glob(pattern))
            
            if matching_files:
                # Extract date from filename and find the latest
                latest_file = max(matching_files, key=lambda f: f.name)
                # Extract date from filename like: basic_calculation_daily_0_20250829.csv
                try:
                    date_part = latest_file.stem.split('_')[-1]
                    if len(date_part) == 8 and date_part.isdigit():
                        latest_files[timeframe] = {
                            'data_date': date_part,
                            'formatted_date': self._format_date(date_part)
                        }
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse date from filename: {latest_file.name}")
        
        return latest_files
    
    def _parse_manual_file_dates(self) -> Dict[str, Dict[str, str]]:
        """Parse manual file dates from configuration."""
        if not self.file_dates_manual:
            return {}
        
        try:
            # Format: "20250829:20250825:20250829" for daily:weekly:monthly
            date_parts = self.file_dates_manual.split(':')
            timeframes = ['daily', 'weekly', 'monthly']
            
            file_dates = {}
            for i, timeframe in enumerate(timeframes):
                if i < len(date_parts) and date_parts[i].strip():
                    data_date = date_parts[i].strip()
                    file_dates[timeframe] = {
                        'data_date': data_date,
                        'formatted_date': self._format_date(data_date)
                    }
            
            return file_dates
            
        except Exception as e:
            logger.error(f"Failed to parse manual file dates: {e}")
            return {}
    
    def _format_date(self, date_str: str) -> str:
        """Format YYYYMMDD to YYYY-MM-DD."""
        if len(date_str) >= 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str
    
    def _find_file_with_naming_variants(self, directory: Path, base_pattern: str, timeframe: str, data_date: str) -> Optional[Path]:
        """Find file trying both hyphen and underscore naming conventions."""
        original_choice = str(self.user_config.ticker_choice)

        # Try both naming conventions
        for choice_variant in [original_choice, original_choice.replace('-', '_')]:
            file_path = directory / base_pattern.format(
                choice=choice_variant,
                timeframe=timeframe,
                data_date=data_date
            )
            if file_path.exists():
                return file_path
        return None

    def _load_data_for_timeframe(self, timeframe: str, data_date: str) -> Dict[str, pd.DataFrame]:
        """Load all relevant data files for a timeframe."""
        data = {}

        try:
            results_dir = self.config.directories['RESULTS_DIR']
            overview_dir = results_dir / 'overview'

            # Load counts analysis (try both naming conventions)
            counts_file = self._find_file_with_naming_variants(
                overview_dir, 'index_counts_{choice}_{timeframe}_{data_date}.csv', timeframe, data_date
            )
            if counts_file and counts_file.exists():
                data['counts'] = pd.read_csv(counts_file)
            
            # Load percentage change analysis - sectors and indexes (try both naming conventions)
            pctchg_sectors_file = self._find_file_with_naming_variants(
                overview_dir, 'pctChgRS_pctChg_sectors_{choice}_{timeframe}_{data_date}.csv', timeframe, data_date
            )
            pctchg_indexes_file = self._find_file_with_naming_variants(
                overview_dir, 'pctChgRS_pctChg_indexes_{choice}_{timeframe}_{data_date}.csv', timeframe, data_date
            )

            # Combine sectors and indexes for legacy compatibility, or load separately
            pctchg_data_list = []
            if pctchg_sectors_file and pctchg_sectors_file.exists():
                sectors_data = pd.read_csv(pctchg_sectors_file)
                sectors_data['analysis_type'] = 'sectors'
                pctchg_data_list.append(sectors_data)
            if pctchg_indexes_file and pctchg_indexes_file.exists():
                indexes_data = pd.read_csv(pctchg_indexes_file)
                indexes_data['analysis_type'] = 'indexes'
                pctchg_data_list.append(indexes_data)
            
            if pctchg_data_list:
                data['pctchg'] = pd.concat(pctchg_data_list, ignore_index=True)
            
            # Load RS analysis - sectors and indexes (try both naming conventions)
            rs_sectors_file = self._find_file_with_naming_variants(
                overview_dir, 'pctChgRS_rs_sectors_{choice}_{timeframe}_{data_date}.csv', timeframe, data_date
            )
            rs_indexes_file = self._find_file_with_naming_variants(
                overview_dir, 'pctChgRS_rs_indexes_{choice}_{timeframe}_{data_date}.csv', timeframe, data_date
            )

            # Combine sectors and indexes for legacy compatibility, or load separately
            rs_data_list = []
            if rs_sectors_file and rs_sectors_file.exists():
                sectors_rs_data = pd.read_csv(rs_sectors_file)
                sectors_rs_data['analysis_type'] = 'sectors'
                rs_data_list.append(sectors_rs_data)
            if rs_indexes_file and rs_indexes_file.exists():
                indexes_rs_data = pd.read_csv(rs_indexes_file)
                indexes_rs_data['analysis_type'] = 'indexes'
                rs_data_list.append(indexes_rs_data)
            
            if rs_data_list:
                data['rs'] = pd.concat(rs_data_list, ignore_index=True)
            
            # Check for tornado chart
            tornado_file = overview_dir / f'index_tornado_{timeframe}_{data_date}.png'
            if tornado_file.exists():
                data['tornado_chart'] = tornado_file
            
            # Load basic calculations if needed (try both naming conventions)
            basic_calc_file = self._find_file_with_naming_variants(
                self.config.directories['BASIC_CALCULATION_DIR'],
                'basic_calculation_{choice}_{timeframe}_{data_date}.csv',
                timeframe, data_date
            )
            if basic_calc_file and basic_calc_file.exists():
                data['basic_calc'] = pd.read_csv(basic_calc_file)
                
            logger.info(f"Loaded {len(data)} data files for {timeframe} ({data_date})")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data for {timeframe} ({data_date}): {e}")
            return {}
    
    # NOTE: _generate_indexes_overview_report method removed
    # indexes_overview template deprecated due to removal of index_overview module
    
    def _create_metadata_section(self, file_dates: Dict, styles) -> List:
        """Create metadata section showing data dates and file info."""
        elements = []
        
        elements.append(Paragraph("Data Sources & Timestamps", styles['Heading2']))
        
        # Create metadata table
        metadata_data = [['Timeframe', 'Data Date', 'Formatted Date']]
        for timeframe, date_info in file_dates.items():
            metadata_data.append([
                timeframe.title(),
                date_info['data_date'],
                date_info['formatted_date']
            ])
        
        metadata_table = Table(metadata_data)
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(metadata_table)
        elements.append(Paragraph(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        return elements
    
    def _create_summary_section(self, report_data: Dict, styles) -> List:
        """Create executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", styles['Heading2']))
        
        # Analyze key metrics across timeframes
        summary_points = []
        
        for timeframe, data in report_data.items():
            if 'counts' in data:
                counts_df = data['counts']
                # Find key insights from counts data
                # This would be customized based on specific metrics
                summary_points.append(f"• {timeframe.title()}: {len(counts_df)} key metrics analyzed")
        
        for point in summary_points:
            elements.append(Paragraph(point, styles['Normal']))
        
        return elements
    
    def _create_basic_stats_section(self, report_data: Dict, styles) -> List:
        """Create basic statistics section."""
        elements = []
        
        elements.append(Paragraph("Index Statistics Overview", styles['Heading2']))
        
        for timeframe, data in report_data.items():
            if 'counts' in data:
                elements.append(Paragraph(f"{timeframe.title()} Analysis", styles['Heading3']))
                
                counts_df = data['counts']
                if not counts_df.empty:
                    # Create table from counts data (first few rows)
                    display_rows = min(10, len(counts_df))
                    table_data = [counts_df.columns.tolist()]
                    
                    for _, row in counts_df.head(display_rows).iterrows():
                        table_data.append([str(val) for val in row.tolist()])
                    
                    counts_table = Table(table_data)
                    counts_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    elements.append(counts_table)
                    elements.append(Spacer(1, 12))
        
        return elements
    
    def _create_tornado_charts_section(self, report_data: Dict, styles) -> List:
        """Create tornado charts section."""
        elements = []
        
        elements.append(Paragraph("Market Breadth Analysis (Tornado Charts)", styles['Heading2']))
        
        for timeframe, data in report_data.items():
            if 'tornado_chart' in data:
                elements.append(Paragraph(f"{timeframe.title()} Market Breadth", styles['Heading3']))
                
                # Add tornado chart image
                chart_path = data['tornado_chart']
                try:
                    # Determine appropriate image size based on page size
                    if self.page_size == 'A3_landscape':
                        img_width = 10 * inch
                        img_height = 6 * inch
                    else:
                        img_width = 8 * inch
                        img_height = 5 * inch
                    
                    chart_img = Image(str(chart_path), width=img_width, height=img_height)
                    elements.append(chart_img)
                    elements.append(Spacer(1, 12))
                    
                except Exception as e:
                    logger.warning(f"Failed to add tornado chart for {timeframe}: {e}")
                    elements.append(Paragraph(f"Chart unavailable for {timeframe}", styles['Normal']))
        
        return elements
    
    def _create_percentage_analysis_section(self, report_data: Dict, styles) -> List:
        """Create percentage analysis section."""
        elements = []
        
        elements.append(Paragraph("Percentage Change Analysis", styles['Heading2']))
        
        for timeframe, data in report_data.items():
            if 'pctchg' in data:
                elements.append(Paragraph(f"{timeframe.title()} Performance Metrics", styles['Heading3']))
                
                pctchg_df = data['pctchg']
                if not pctchg_df.empty:
                    # Create table from percentage data
                    display_rows = min(self.max_tickers_display, len(pctchg_df))
                    table_data = [pctchg_df.columns.tolist()]
                    
                    for _, row in pctchg_df.head(display_rows).iterrows():
                        table_data.append([str(val) for val in row.tolist()])
                    
                    pctchg_table = Table(table_data)
                    pctchg_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    elements.append(pctchg_table)
                    elements.append(Spacer(1, 12))
        
        return elements
    
    def _create_rs_analysis_section(self, report_data: Dict, styles) -> List:
        """Create relative strength analysis section."""
        elements = []
        
        elements.append(Paragraph("Relative Strength Analysis", styles['Heading2']))
        
        for timeframe, data in report_data.items():
            if 'rs' in data:
                elements.append(Paragraph(f"{timeframe.title()} Relative Strength Metrics", styles['Heading3']))
                
                rs_df = data['rs']
                if not rs_df.empty:
                    # Create table from RS data
                    display_rows = min(self.max_tickers_display, len(rs_df))
                    table_data = [rs_df.columns.tolist()]
                    
                    for _, row in rs_df.head(display_rows).iterrows():
                        table_data.append([str(val) for val in row.tolist()])
                    
                    rs_table = Table(table_data)
                    rs_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    elements.append(rs_table)
                    elements.append(Spacer(1, 12))
        
        return elements
    
    def _generate_market_analysis_report(self, report_data: Dict, file_dates: Dict) -> Path:
        """Generate market analysis report using basic_calculation data."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'market_analysis_report_{timestamp}.pdf'

        # Determine page size
        if self.page_size == 'A4_portrait':
            page_size = A4
        elif self.page_size == 'A4_landscape':
            page_size = landscape(A4)
        elif self.page_size == 'A3_landscape':
            page_size = landscape(A3)
        else:
            page_size = landscape(A4)  # default

        doc = SimpleDocTemplate(str(output_file), pagesize=page_size)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )

        story.append(Paragraph("Market Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # Metadata section
        if self.include_metadata:
            story.extend(self._create_metadata_section(file_dates, styles))
            story.append(Spacer(1, 20))

        # Summary section
        if self.sections['summary']:
            story.extend(self._create_summary_section(report_data, styles))
            story.append(Spacer(1, 20))

        # Basic statistics section
        if self.sections['basic_stats']:
            story.extend(self._create_basic_stats_section(report_data, styles))
            story.append(Spacer(1, 20))

        # Tornado charts section
        if self.sections['tornado_charts'] and self.include_charts:
            story.extend(self._create_tornado_charts_section(report_data, styles))
            story.append(PageBreak())

        # Build PDF
        doc.build(story)

        logger.info(f"Market analysis report generated: {output_file}")
        print(f"  📄 Report generated: {output_file.name}")

        return output_file
    
    def _generate_full_analysis_report(self, report_data: Dict, file_dates: Dict) -> Path:
        """Generate full analysis report (placeholder for future expansion).""" 
        logger.info("Full analysis report template not yet implemented")
        return None