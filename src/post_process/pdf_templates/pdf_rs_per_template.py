"""
RS/PER Template for Post-Process PDF Generation
==============================================

PDF template for comprehensive multi-timeframe RS/PER market analysis.
This template handles RS and percentile data to generate professional
investment analysis reports.

Note: This template bypasses the standard filtered DataFrame approach
and directly accesses RS/PER data files to perform its own analysis.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def generate_pdf(df: pd.DataFrame, pdf_path: str, metadata: Dict = None) -> bool:
    """
    Generate comprehensive RS/PER analysis PDF.

    Args:
        df: Filtered DataFrame (not used for RS/PER - we use direct file access)
        pdf_path: Output PDF file path
        metadata: Processing metadata from post-process workflow

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Starting RS/PER template PDF generation")

        # Import RS/PER processor
        from ..rs_per_processor import run_rs_per_analysis

        # Get config from metadata or create minimal config
        config = _get_config_from_metadata(metadata)

        # Extract date from pdf_path for analysis
        date_str = _extract_date_from_path(pdf_path)

        # Run complete RS/PER analysis
        logger.info(f"Running RS/PER analysis for date: {date_str}")
        results = run_rs_per_analysis(
            config=config,
            user_config=None,
            date_str=date_str
        )

        if not results.get('success'):
            logger.error(f"RS/PER analysis failed: {results.get('error')}")
            return _create_error_pdf(pdf_path, results.get('error', 'Analysis failed'))

        # The analysis already creates its own PDF, so we need to move/copy it
        generated_pdf_path = results.get('pdf_path')

        if generated_pdf_path and Path(generated_pdf_path).exists():
            # Move the generated PDF to the expected location
            source_pdf = Path(generated_pdf_path)
            target_pdf = Path(pdf_path)

            if source_pdf != target_pdf:
                # Copy the file to the expected location
                import shutil
                shutil.copy2(source_pdf, target_pdf)
                logger.info(f"Moved RS/PER PDF from {source_pdf} to {target_pdf}")

            # Log success summary
            processing_summary = results.get('processing_summary', {})
            analysis_summary = processing_summary.get('analysis_summary', {})

            logger.info(f"RS/PER PDF generation successful:")
            logger.info(f"  Market Condition: {analysis_summary.get('market_condition', 'Unknown')}")
            logger.info(f"  Stocks Analyzed: {analysis_summary.get('total_stocks', 0):,}")
            logger.info(f"  Charts Generated: {processing_summary.get('visualization_summary', {}).get('charts_generated', 0)}/6")
            logger.info(f"  Output PDF: {target_pdf}")

            return True

        else:
            logger.error("RS/PER analysis completed but no PDF was generated")
            return _create_error_pdf(pdf_path, "PDF generation failed in RS/PER analysis")

    except ImportError as e:
        logger.error(f"Failed to import RS/PER processor: {e}")
        return _create_error_pdf(pdf_path, f"RS/PER module import failed: {e}")

    except Exception as e:
        logger.error(f"RS/PER template failed: {e}")
        return _create_error_pdf(pdf_path, f"Template execution failed: {e}")


def _get_config_from_metadata(metadata: Dict) -> object:
    """Extract or create configuration object from metadata."""
    if metadata and 'config' in metadata:
        return metadata['config']

    # Create minimal config object
    logger.warning("No config in metadata, creating minimal config")

    class MinimalConfig:
        def __init__(self):
            from pathlib import Path
            base_dir = Path.cwd()
            self.directories = {
                'RESULTS_DIR': base_dir / 'results',
                'TICKERS_DIR': base_dir / 'data' / 'tickers'
            }

    return MinimalConfig()


def _extract_date_from_path(pdf_path: str) -> Optional[str]:
    """Extract date string from PDF path."""
    try:
        # Look for YYYYMMDD pattern in filename
        import re
        filename = Path(pdf_path).stem
        date_match = re.search(r'(\d{8})', filename)

        if date_match:
            return date_match.group(1)

        # Fallback to current date
        return datetime.now().strftime('%Y%m%d')

    except Exception as e:
        logger.warning(f"Failed to extract date from path {pdf_path}: {e}")
        return datetime.now().strftime('%Y%m%d')


def _create_error_pdf(pdf_path: str, error_message: str) -> bool:
    """Create a simple error PDF when RS/PER analysis fails."""
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch

        logger.info(f"Creating error PDF: {pdf_path}")

        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("RS/PER Analysis Report", styles['Title']))
        story.append(Spacer(1, 0.5 * inch))

        # Error message
        story.append(Paragraph("Analysis Error", styles['Heading1']))
        story.append(Paragraph(f"The RS/PER analysis could not be completed:", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(error_message, styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

        # Recommendations
        story.append(Paragraph("Recommendations:", styles['Heading2']))
        recommendations = [
            "Ensure RS analysis has been run and RS files are available",
            "Ensure PER analysis has been run and PER files are available",
            "Check that BASIC flag is set to TRUE to enable RS/PER processing",
            "Verify that the date specified has corresponding data files",
            "Check log files for detailed error information"
        ]

        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))

        # Footer
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                              styles['Normal']))

        doc.build(story)
        logger.info(f"Error PDF created: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create error PDF: {e}")
        return False


def validate_rs_per_requirements(config) -> Dict:
    """
    Validate that RS/PER data is available for analysis.

    Args:
        config: Configuration object

    Returns:
        Validation results dictionary
    """
    try:
        from ..rs_per_processor import validate_rs_per_requirements
        return validate_rs_per_requirements(config)

    except ImportError:
        return {
            'requirements_met': False,
            'missing_components': ['RS/PER processor module'],
            'recommendations': ['Install RS/PER analysis components'],
            'system_ready': False
        }

    except Exception as e:
        return {
            'requirements_met': False,
            'missing_components': ['Unknown validation error'],
            'recommendations': [f'Check system configuration: {e}'],
            'system_ready': False
        }