"""
PDF Templates for Post-Process Workflow
======================================

Template registry and dispatcher for generating PDFs from filtered DataFrames.
Each template handles specific data types and generates appropriate visualizations.
"""

def get_template(pdf_type: str):
    """
    Get the appropriate PDF generation function based on template type.

    Args:
        pdf_type: Template identifier (e.g., 'stage_analysis', 'default')

    Returns:
        Function that generates PDF from DataFrame
    """
    template_map = {
        'stage_analysis': 'pdf_stage_analysis_template',
        'default': 'pdf_default_template'
    }

    module_name = template_map.get(pdf_type, 'pdf_default_template')

    try:
        if module_name == 'pdf_stage_analysis_template':
            from .pdf_stage_analysis_template import generate_pdf
        else:
            from .pdf_default_template import generate_pdf
        return generate_pdf
    except ImportError as e:
        # Fallback to default template
        from .pdf_default_template import generate_pdf
        return generate_pdf