"""
PRE_PROCESS Module - TradingView Data Transformation
===================================================

Independent module for processing TradingView CSV files into standardized format.
Executes before BASIC phase as part of the main pipeline.

Main Components:
- PreProcessManager: Main orchestrator
- PreProcessConfig: Configuration loader
- FilePatternMatcher: Pattern-based file discovery
- TickerExtractor: Extract ticker from filenames
- ColumnStandardizer: TradingView → Standard format conversion
- DataTransformer: Core transformation logic
"""

from .pre_process_manager import PreProcessManager

def run_pre_process_analysis():
    """
    Main entry point for PRE_PROCESS module.
    Called from main.py when PRE_PROCESS = TRUE.
    """
    manager = PreProcessManager()
    return manager.run_pre_processing()

__all__ = ['run_pre_process_analysis', 'PreProcessManager']