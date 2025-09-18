"""
Post-Process Module
==================

Modular post-processing system for filtering, sorting, and generating reports
from existing CSV output files without modifying core calculation modules.

Components:
- return_file_info: Maps logical names to physical files, finds latest by date
- post_process_workflow: Orchestrates multi-file processing workflow
- generate_post_process_pdfs: PDF report generation (future implementation)
"""

from .return_file_info import get_latest_file, get_logical_file_mapping
from .post_process_workflow import run_post_processing

__version__ = "1.0.0"
__all__ = [
    "get_latest_file",
    "get_logical_file_mapping",
    "run_post_processing"
]