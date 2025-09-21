"""
PDF Templates for Post-Process Workflow
======================================

Template registry and dispatcher for generating PDFs from filtered DataFrames.
Each template handles specific data types and generates appropriate visualizations.
"""

import logging

logger = logging.getLogger(__name__)

def get_template(template_name: str):
    """
    Get the appropriate PDF generation function based on template type.

    Args:
        template_name: Template identifier (e.g., 'stage_analysis', 'market_trends', 'sector_analysis', etc.)

    Returns:
        Function that generates PDF from DataFrame
    """
    # Validate template exists before processing
    valid_templates = {
        'stage_analysis', 'default', 'market_trends', 'sector_analysis',
        'industry_analysis', 'risk_analysis', 'universe_analysis', 'basic_calculation',
        'top_performers_evolution', 'top_performers_evolution_v1', 'overview_v1',
        'rs_per_template'
    }

    if template_name not in valid_templates:
        logger.warning(f"Unknown template '{template_name}', falling back to auto-selection")
        template_name = 'basic_calculation'  # Trigger auto-selection
    template_map = {
        # Legacy templates
        'stage_analysis': 'pdf_stage_analysis_template',
        'default': 'pdf_default_template',

        # Basic calculation templates
        'market_trends': 'pdf_basic_calculation_market_trends_template',
        'sector_analysis': 'pdf_basic_calculation_sector_analysis_template',
        'industry_analysis': 'pdf_basic_calculation_industry_analysis_template',
        'risk_analysis': 'pdf_basic_calculation_risk_analysis_template',
        'universe_analysis': 'pdf_basic_calculation_universe_analysis_template',

        # Advanced analysis templates
        'top_performers_evolution': 'pdf_top_performers_evolution_template',
        'top_performers_evolution_v1': 'pdf_top_performers_evolution_v1_template',
        'overview_v1': 'pdf_overview_v1_template',
        'rs_per_template': 'pdf_rs_per_template',

        # Auto-selection aliases
        'basic_calculation': 'auto_select_basic_calculation'
    }

    module_name = template_map.get(template_name, 'pdf_default_template')

    try:
        # Handle auto-selection
        if module_name == 'auto_select_basic_calculation':
            # Return the auto-selection function instead of direct template
            return auto_select_basic_calculation_template

        # Legacy templates
        elif module_name == 'pdf_stage_analysis_template':
            from .pdf_stage_analysis_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_default_template':
            from .pdf_default_template import generate_pdf
            return generate_pdf

        # Basic calculation templates
        elif module_name == 'pdf_basic_calculation_market_trends_template':
            from .pdf_basic_calculation_market_trends_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_basic_calculation_sector_analysis_template':
            from .pdf_basic_calculation_sector_analysis_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_basic_calculation_industry_analysis_template':
            from .pdf_basic_calculation_industry_analysis_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_basic_calculation_risk_analysis_template':
            from .pdf_basic_calculation_risk_analysis_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_basic_calculation_universe_analysis_template':
            from .pdf_basic_calculation_universe_analysis_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_top_performers_evolution_template':
            from .pdf_top_performers_evolution_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_top_performers_evolution_v1_template':
            from .pdf_top_performers_evolution_v1_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_overview_v1_template':
            from .pdf_overview_v1_template import generate_pdf
            return generate_pdf
        elif module_name == 'pdf_rs_per_template':
            from .pdf_rs_per_template import generate_pdf
            return generate_pdf

        else:
            # Fallback to default template
            from .pdf_default_template import generate_pdf
            return generate_pdf

    except ImportError as e:
        logger.warning(f"Failed to import template '{module_name}': {e}")
        # Fallback to default template
        from .pdf_default_template import generate_pdf
        return generate_pdf

def auto_select_basic_calculation_template(df, pdf_path: str, metadata: dict = None):
    """
    Auto-select the most appropriate basic_calculation template based on filter patterns.

    Args:
        df: DataFrame with basic_calculation data
        pdf_path: Output PDF file path
        metadata: Rich context from post-process workflow including filter operations

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine best template based on filter patterns
        selected_template = _determine_template_from_filters(df, metadata)

        logger.info(f"Auto-selected template: {selected_template}")

        # Get the selected template function
        template_func = get_template(selected_template)

        # Generate PDF using selected template
        return template_func(df, pdf_path, metadata)

    except Exception as e:
        logger.error(f"Error in auto-selection: {e}")
        # Fallback to market trends template
        from .pdf_basic_calculation_market_trends_template import generate_pdf
        return generate_pdf(df, pdf_path, metadata)

def _determine_template_from_filters(df, metadata: dict) -> str:
    """
    Determine the best template based on filter operations and data characteristics.

    Args:
        df: DataFrame with data
        metadata: Rich context including filter operations

    Returns:
        str: Template name to use
    """
    try:
        if not metadata or 'filter_operations' not in metadata:
            return 'market_trends'  # Default comprehensive template

        filter_ops = metadata['filter_operations']

        # Check for specific filter patterns
        for filter_op in filter_ops:
            column = filter_op.get('Column', '')
            condition = filter_op.get('Condition', '')
            value = filter_op.get('Value', '')

            # Sector-specific filter
            if column == 'sector' and condition == 'equals':
                return 'sector_analysis'

            # Industry-specific filter
            if column == 'industry' and condition == 'equals':
                return 'industry_analysis'

            # Universe/index filters
            universe_columns = ['SP500', 'NASDAQ100', 'Russell1000', 'Russell3000', 'DowJonesIndustrialAverage']
            if column in universe_columns and condition == 'equals' and str(value).upper() == 'TRUE':
                return 'universe_analysis'

            # Risk-related filters
            risk_columns = ['atr_pct', 'daily_distance_from_ATH_pct', 'daily_rsi_14']
            if column in risk_columns:
                return 'risk_analysis'

        # Data-driven selection based on predominant characteristics
        if 'sector' in df.columns:
            # If 90%+ from single sector, use sector template
            sector_counts = df['sector'].value_counts()
            if len(sector_counts) > 0 and (sector_counts.iloc[0] / len(df)) > 0.9:
                return 'sector_analysis'

        if 'industry' in df.columns:
            # If 90%+ from single industry, use industry template
            industry_counts = df['industry'].value_counts()
            if len(industry_counts) > 0 and (industry_counts.iloc[0] / len(df)) > 0.9:
                return 'industry_analysis'

        # Check for universe concentration
        universe_columns = ['SP500', 'NASDAQ100', 'Russell1000', 'Russell3000']
        for universe_col in universe_columns:
            if universe_col in df.columns:
                universe_pct = (df[universe_col] == True).mean()
                if universe_pct > 0.9:
                    return 'universe_analysis'

        # Default to comprehensive market trends
        return 'market_trends'

    except Exception as e:
        logger.warning(f"Error in template selection: {e}")
        return 'market_trends'