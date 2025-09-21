"""
RS/PER Main Processor
====================

Main orchestrator for comprehensive RS/PER analysis and PDF report generation.
Integrates all components: data loading, analysis, visualization, and PDF creation.

This is the entry point for the rs_per_template in the post-processing system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .rs_per_data_loader import RSPERDataLoader
from .rs_per_analyzer import RSPERAnalyzer
from .rs_per_charts import RSPERChartGenerator
from .rs_per_report_generator import RSPERReportGenerator
from .rs_per_pdf_builder import RSPERPDFBuilder

logger = logging.getLogger(__name__)


class RSPERProcessor:
    """
    Main processor for comprehensive RS/PER analysis and reporting.

    Orchestrates the complete workflow:
    1. Load RS and PER data files
    2. Perform multi-timeframe analysis
    3. Generate visualizations (6 chart types)
    4. Create narrative report content
    5. Assemble professional PDF report
    """

    def __init__(self, config, user_config=None):
        """Initialize the RS/PER processor."""
        self.config = config
        self.user_config = user_config

        # Initialize components
        self.data_loader = RSPERDataLoader(config)
        self.analyzer = RSPERAnalyzer(config)

        # Setup output directories
        self.charts_dir = config.directories['RESULTS_DIR'] / 'rs_per_charts'
        self.reports_dir = config.directories['RESULTS_DIR'] / 'rs_per_reports'

        self.charts_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        self.chart_generator = RSPERChartGenerator(config, str(self.charts_dir))
        self.report_generator = RSPERReportGenerator(config)
        self.pdf_builder = RSPERPDFBuilder(config, str(self.reports_dir))

    def process_rs_per_analysis(self, date_str: str = None,
                               filtered_data: pd.DataFrame = None) -> Dict:
        """
        Process complete RS/PER analysis and generate PDF report.

        Args:
            date_str: Date string in YYYYMMDD format
            filtered_data: Pre-filtered data (for post-processing integration)

        Returns:
            Results dictionary with file paths and analysis summary
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')

        logger.info(f"Starting RS/PER analysis processing for {date_str}")

        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading RS/PER data")
            data = self.data_loader.load_daily_data(date_str)

            if not data or data.get('validation_status') != 'success':
                logger.error("Data loading failed or validation unsuccessful")
                return self._create_error_result("Data loading failed")

            # Step 2: Perform multi-timeframe analysis
            logger.info("Step 2: Performing multi-timeframe analysis")
            analysis_results = self.analyzer.perform_multi_timeframe_analysis(data)

            if not analysis_results:
                logger.error("Analysis failed")
                return self._create_error_result("Analysis failed")

            # Step 3: Generate visualizations
            logger.info("Step 3: Generating visualizations")
            charts = self.chart_generator.generate_all_charts(analysis_results, date_str)

            if not charts:
                logger.warning("Chart generation failed, continuing without charts")
                charts = {}

            # Step 4: Generate report content
            logger.info("Step 4: Generating report content")
            report_sections = self.report_generator.generate_comprehensive_report(
                analysis_results, charts, date_str
            )

            # Step 5: Create PDF report
            logger.info("Step 5: Creating PDF report")
            pdf_path = self.pdf_builder.build_pdf(
                report_sections, charts, analysis_results, date_str
            )

            # Step 6: Generate summary
            logger.info("Step 6: Generating processing summary")
            processing_summary = self._create_processing_summary(
                data, analysis_results, charts, pdf_path, date_str
            )

            logger.info("RS/PER analysis processing completed successfully")

            return {
                'success': True,
                'date': date_str,
                'pdf_path': pdf_path,
                'charts': charts,
                'analysis_results': analysis_results,
                'processing_summary': processing_summary,
                'data_quality': data.get('validation_status'),
                'timeframes_processed': data.get('timeframes', [])
            }

        except Exception as e:
            logger.error(f"RS/PER processing failed: {e}")
            return self._create_error_result(f"Processing failed: {e}")

    def validate_rs_per_data_availability(self, date_str: str = None) -> Dict:
        """
        Validate that required RS and PER data files are available.

        Args:
            date_str: Date string to check

        Returns:
            Validation results dictionary
        """
        if date_str is None:
            date_str = self.data_loader._get_latest_date()

        logger.info(f"Validating RS/PER data availability for {date_str}")

        validation_results = {
            'date': date_str,
            'data_available': False,
            'rs_files_found': [],
            'per_files_found': [],
            'missing_files': [],
            'data_quality_score': 0.0,
            'recommendations': []
        }

        try:
            # Check for required RS files
            rs_patterns = {
                'rs_ibd_stocks': f'rs_QQQ_ibd_stocks_daily_2-5_{date_str}.csv',
                'rs_ibd_sectors': f'rs_QQQ_ibd_sectors_daily_2-5_{date_str}.csv',
                'rs_ibd_industries': f'rs_QQQ_ibd_industries_daily_2-5_{date_str}.csv'
            }

            rs_dir = self.config.directories['RESULTS_DIR'] / 'rs'
            for file_type, filename in rs_patterns.items():
                file_path = rs_dir / filename
                if file_path.exists():
                    validation_results['rs_files_found'].append(filename)
                else:
                    validation_results['missing_files'].append(filename)

            # Check for required PER files
            per_patterns = {
                'per_stocks': f'per_QQQ_NASDAQ100_ibd_stocks_daily_2-5_{date_str}.csv',
                'per_sectors': f'per_QQQ_NASDAQ100_ibd_sectors_daily_2-5_{date_str}.csv',
                'per_industries': f'per_QQQ_NASDAQ100_ibd_industries_daily_2-5_{date_str}.csv'
            }

            per_dir = self.config.directories['RESULTS_DIR'] / 'per'
            for file_type, filename in per_patterns.items():
                file_path = per_dir / filename
                if file_path.exists():
                    validation_results['per_files_found'].append(filename)
                else:
                    validation_results['missing_files'].append(filename)

            # Calculate data quality score
            total_required = len(rs_patterns) + len(per_patterns)
            files_found = len(validation_results['rs_files_found']) + len(validation_results['per_files_found'])
            validation_results['data_quality_score'] = (files_found / total_required) * 100

            # Determine if data is available for processing
            validation_results['data_available'] = (
                len(validation_results['rs_files_found']) >= 2 and  # At least 2 RS files
                len(validation_results['per_files_found']) >= 1     # At least 1 PER file
            )

            # Generate recommendations
            if not validation_results['data_available']:
                validation_results['recommendations'].append(
                    "Insufficient data files for RS/PER analysis. Run RS and PER generation first."
                )

            if validation_results['data_quality_score'] < 100:
                validation_results['recommendations'].append(
                    f"Data completeness: {validation_results['data_quality_score']:.1f}%. "
                    f"Missing files: {', '.join(validation_results['missing_files'])}"
                )

            logger.info(f"Data validation completed. Quality score: {validation_results['data_quality_score']:.1f}%")
            return validation_results

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            validation_results['recommendations'].append(f"Validation error: {e}")
            return validation_results

    def get_available_analysis_dates(self, lookback_days: int = 30) -> List[str]:
        """Get list of dates with available RS/PER data for analysis."""
        return self.data_loader.get_available_dates(lookback_days)

    def _create_processing_summary(self, data: Dict, analysis_results: Dict,
                                 charts: Dict, pdf_path: str, date_str: str) -> Dict:
        """Create comprehensive processing summary."""
        summary = {
            'processing_date': datetime.now().isoformat(),
            'analysis_date': date_str,
            'data_summary': {
                'timeframes_analyzed': len(data.get('timeframes', [])),
                'data_sources': list(data.get('rs_data', {}).keys()) + list(data.get('per_data', {}).keys()),
                'data_quality': data.get('validation_status')
            },
            'analysis_summary': {},
            'visualization_summary': {
                'charts_generated': len([path for path in charts.values() if path]),
                'chart_types': list(charts.keys()),
                'charts_failed': len([path for path in charts.values() if not path])
            },
            'output_files': {
                'pdf_report': pdf_path,
                'charts_directory': str(self.charts_dir),
                'reports_directory': str(self.reports_dir)
            }
        }

        # Add analysis-specific summary
        market_condition = analysis_results.get('market_condition')
        if market_condition:
            summary['analysis_summary'] = {
                'market_condition': market_condition.condition,
                'total_stocks': market_condition.total_stocks,
                'market_breadth_pct': market_condition.market_breadth_pct,
                'elite_performers': market_condition.elite_stocks
            }

        # Add stocks analysis summary
        stocks_analysis = analysis_results.get('stocks_analysis', {})
        if stocks_analysis:
            top_performers = stocks_analysis.get('top_performers', [])
            summary['analysis_summary']['top_performers_count'] = len(top_performers)

            if top_performers:
                summary['analysis_summary']['top_performer'] = {
                    'ticker': top_performers[0]['ticker'],
                    'composite_strength': top_performers[0]['composite_strength'],
                    'classification': top_performers[0]['classification']
                }

        return summary

    def _create_error_result(self, error_message: str) -> Dict:
        """Create standardized error result."""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'pdf_path': None,
            'charts': {},
            'analysis_results': None,
            'processing_summary': None
        }

    def create_analysis_report_summary(self, results: Dict) -> str:
        """Create a text summary of the analysis results."""
        if not results.get('success'):
            return f"RS/PER Analysis Failed: {results.get('error', 'Unknown error')}"

        processing_summary = results.get('processing_summary', {})
        analysis_summary = processing_summary.get('analysis_summary', {})

        summary_text = f"""
RS/PER COMPREHENSIVE ANALYSIS SUMMARY
=====================================
Analysis Date: {results.get('date')}
Processing Time: {processing_summary.get('processing_date')}

MARKET CONDITION: {analysis_summary.get('market_condition', 'Unknown')}
Total Stocks Analyzed: {analysis_summary.get('total_stocks', 0):,}
Market Breadth: {analysis_summary.get('market_breadth_pct', 0):.1f}%
Elite Performers: {analysis_summary.get('elite_performers', 0)}

TOP PERFORMER: {analysis_summary.get('top_performer', {}).get('ticker', 'N/A')}
- Composite Strength: {analysis_summary.get('top_performer', {}).get('composite_strength', 0):.3f}
- Classification: {analysis_summary.get('top_performer', {}).get('classification', 'N/A')}

OUTPUTS GENERATED:
- PDF Report: {Path(results.get('pdf_path', '')).name}
- Charts Generated: {processing_summary.get('visualization_summary', {}).get('charts_generated', 0)}/6
- Timeframes Analyzed: {processing_summary.get('data_summary', {}).get('timeframes_analyzed', 0)}/9

DATA QUALITY: {processing_summary.get('data_summary', {}).get('data_quality', 'Unknown')}
"""

        return summary_text


def run_rs_per_analysis(config, user_config=None, date_str: str = None,
                       filtered_data: pd.DataFrame = None) -> Dict:
    """
    Standalone function to run complete RS/PER analysis.

    This is the main entry point for the rs_per_template in post-processing.

    Args:
        config: Config object
        user_config: User configuration object
        date_str: Date string for analysis
        filtered_data: Pre-filtered data (not used in RS/PER analysis)

    Returns:
        Results dictionary
    """
    logger.info("Starting standalone RS/PER analysis")

    try:
        processor = RSPERProcessor(config, user_config)

        # Validate data availability first
        validation = processor.validate_rs_per_data_availability(date_str)

        if not validation['data_available']:
            logger.warning(f"Insufficient data for RS/PER analysis: {validation['recommendations']}")
            return processor._create_error_result(
                f"Insufficient data. Quality score: {validation['data_quality_score']:.1f}%"
            )

        # Run analysis
        results = processor.process_rs_per_analysis(date_str, filtered_data)

        # Log summary
        if results.get('success'):
            summary = processor.create_analysis_report_summary(results)
            logger.info(f"RS/PER analysis completed:\n{summary}")

        return results

    except Exception as e:
        logger.error(f"Standalone RS/PER analysis failed: {e}")
        return {
            'success': False,
            'error': f"Analysis failed: {e}",
            'timestamp': datetime.now().isoformat()
        }


def validate_rs_per_requirements(config) -> Dict:
    """
    Validate that all required components are available for RS/PER analysis.

    Returns:
        Validation results with recommendations
    """
    logger.info("Validating RS/PER analysis requirements")

    validation = {
        'requirements_met': True,
        'missing_components': [],
        'recommendations': [],
        'system_ready': False
    }

    try:
        # Check if RS/PER directories exist
        rs_dir = config.directories['RESULTS_DIR'] / 'rs'
        per_dir = config.directories['RESULTS_DIR'] / 'per'

        if not rs_dir.exists():
            validation['missing_components'].append('RS results directory')
            validation['recommendations'].append('Run RS analysis first to generate RS data files')

        if not per_dir.exists():
            validation['missing_components'].append('PER results directory')
            validation['recommendations'].append('Run PER analysis first to generate percentile data files')

        # Check for recent data files
        processor = RSPERProcessor(config)
        available_dates = processor.get_available_analysis_dates(lookback_days=7)

        if not available_dates:
            validation['missing_components'].append('Recent RS/PER data files')
            validation['recommendations'].append('Generate recent RS and PER data files for analysis')
        else:
            latest_date = available_dates[0]
            date_validation = processor.validate_rs_per_data_availability(latest_date)

            if not date_validation['data_available']:
                validation['missing_components'].append('Complete data set for latest date')
                validation['recommendations'].extend(date_validation['recommendations'])

        # Final assessment
        validation['requirements_met'] = len(validation['missing_components']) == 0
        validation['system_ready'] = validation['requirements_met'] and len(available_dates) > 0

        if validation['system_ready']:
            validation['recommendations'].append('System ready for RS/PER analysis')

        logger.info(f"Requirements validation completed. System ready: {validation['system_ready']}")
        return validation

    except Exception as e:
        logger.error(f"Requirements validation failed: {e}")
        validation['requirements_met'] = False
        validation['system_ready'] = False
        validation['recommendations'].append(f"Validation error: {e}")
        return validation