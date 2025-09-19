#!/usr/bin/env python3
"""
Post-Process PDF Generator Dispatcher
====================================

Dispatches PDF generation based on PDF_type using appropriate templates.
Designed to work with filtered DataFrames from post-process workflow.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def generate_post_process_pdfs(df: pd.DataFrame, pdf_type: str, csv_output_path: str, metadata: dict = None) -> str:
    """
    Generate PDF from filtered DataFrame using specified template.

    Args:
        df: Already filtered and sorted DataFrame
        pdf_type: Template identifier (e.g., 'stage_analysis', 'default')
        csv_output_path: Path where CSV was saved (for PDF naming)
        metadata: Rich processing context from post-process workflow

    Returns:
        Path to generated PDF file

    Raises:
        Exception: If PDF generation fails
    """
    try:
        # Derive PDF path from CSV path
        csv_path = Path(csv_output_path)
        pdf_path = csv_path.with_suffix('.pdf')

        logger.info(f"Generating PDF with template '{pdf_type}': {pdf_path}")

        # Get appropriate template function
        from .pdf_templates import get_template
        template_func = get_template(pdf_type)

        # Generate PDF using template
        success = template_func(df, str(pdf_path), metadata)

        if success:
            logger.info(f"Successfully generated PDF: {pdf_path}")
            return str(pdf_path)
        else:
            raise Exception(f"Template '{pdf_type}' returned failure status")

    except Exception as e:
        logger.error(f"Failed to generate PDF with template '{pdf_type}': {e}")
        raise

def extract_template_from_config(file_ops: pd.DataFrame) -> str:
    """
    Extract explicit template or fall back to PDF_type auto-selection.

    Args:
        file_ops: DataFrame containing operations for this file group

    Returns:
        Template name string, defaults to 'default' if not specified
    """
    # 1. Check for explicit Template column first (NEW)
    if 'Template' in file_ops.columns:
        templates = file_ops['Template'].dropna()
        if not templates.empty:
            template = str(templates.iloc[0]).strip()
            if template and template.lower() != 'auto':
                return template  # Direct template specification

    # 2. Fall back to PDF_type (EXISTING LOGIC - backward compatible)
    if 'PDF_type' in file_ops.columns:
        pdf_types = file_ops['PDF_type'].dropna()
        if not pdf_types.empty:
            return str(pdf_types.iloc[0]).strip()

    return 'default'

# Backward compatibility alias
def extract_pdf_type_from_config(file_ops: pd.DataFrame) -> str:
    """Backward compatibility alias for extract_template_from_config."""
    return extract_template_from_config(file_ops)

def is_pdf_enabled(file_ops: pd.DataFrame) -> bool:
    """
    Check if PDF generation is enabled via PDF_enable column.

    Args:
        file_ops: DataFrame containing operations for this file group

    Returns:
        True if PDF generation is enabled, False otherwise
    """
    if 'PDF_enable' not in file_ops.columns:
        return True  # Default to enabled if column missing (backward compatibility)

    # Get all PDF_enable values for this group
    enable_values = file_ops['PDF_enable'].dropna()

    if enable_values.empty:
        return True  # Default to enabled if all values are empty

    # Check if any row enables PDF (conservative approach)
    for value in enable_values:
        value_str = str(value).upper().strip()
        if value_str == 'TRUE':
            return True

    # If we have explicit values and none are TRUE, then it's disabled
    return False


def should_generate_pdf(file_ops: pd.DataFrame) -> bool:
    """
    Determine if PDF generation should be performed based on configuration.
    Checks both PDF_type existence and PDF_enable status.

    Args:
        file_ops: DataFrame containing operations for this file group

    Returns:
        True if PDF should be generated, False otherwise
    """
    # Check if PDF_type column exists and has values
    has_pdf_type = 'PDF_type' in file_ops.columns and not file_ops['PDF_type'].dropna().empty

    # Check if PDF generation is enabled
    pdf_enabled = is_pdf_enabled(file_ops)

    return has_pdf_type and pdf_enabled