"""
Sustainability Ratios Overview Charts Module
==========================================

Generates visual charts for overview analysis by calling existing panel submodule functions.
Does NOT modify panel submodule functionality - just uses it with different CSV input.

This module:
- Calls existing SRProcessor with user_charts_display.csv instead of user_data_panel.csv
- Uses identical chart generation: same SRProcessor, same create_multi_panel_chart()
- Produces identical quality charts using proven panel submodule code
- Zero changes to existing panel submodule functionality
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any

from ...config import Config
from ...user_defined_data import UserConfiguration
from ..sr_calculations import SRProcessor

logger = logging.getLogger(__name__)


class OverviewChartsProcessor:
    """
    Generates overview charts by calling existing panel submodule functions.
    No changes to panel submodule - just different CSV input.
    """

    def __init__(self, config: Config, user_config: UserConfiguration, timeframe: str):
        """
        Initialize overview charts processor.

        Args:
            config: System configuration
            user_config: User configuration
            timeframe: Processing timeframe ('daily', 'weekly', 'monthly')
        """
        self.config = config
        self.user_config = user_config
        self.timeframe = timeframe
        self.chart_files = []

    def run_overview_charts_analysis(self, overview_results: Dict[str, Any] = None) -> List[str]:
        """
        Generate overview charts by calling existing panel submodule with different CSV.

        Args:
            overview_results: Results from overview values analysis (not used)

        Returns:
            List of generated chart file paths
        """
        try:
            logger.info(f"Running overview charts analysis for {self.timeframe} timeframe...")

            # Get CSV filename from configuration
            csv_filename = getattr(self.user_config, 'sr_overview_charts_display_panel', 'user_charts_display.csv')

            # Check if path includes directory separators
            if '/' in csv_filename or '\\' in csv_filename:
                # Full path specified - use as-is (could be absolute or relative)
                csv_path = Path(csv_filename)
                if not csv_path.is_absolute():
                    csv_path = Path(self.config.base_dir) / csv_path
            else:
                # Just filename - file is in parent directory
                csv_path = Path(self.config.base_dir) / csv_filename

            if not csv_path.exists():
                logger.warning(f"Overview charts CSV not found: {csv_path}")
                return []

            # Load our overview CSV using same parser as panel submodule
            from ..sr_config_reader import parse_panel_config
            panel_configs = parse_panel_config(str(csv_path))
            logger.info(f"Loaded {len(panel_configs)} overview panel configurations")

            if not panel_configs:
                logger.warning("No panel configurations loaded from overview CSV")
                return []

            # Create SRProcessor instance (same as panel submodule)
            processor = SRProcessor(self.config, self.user_config, self.timeframe)

            # Override the panel configs with our overview configs (prevent loading from SR_EB/user_data_panel.csv)
            processor.panel_configs = panel_configs

            # Process all row configurations using the correct working method
            # This method handles everything: loading data, processing panels, generating charts
            panel_success = processor.process_all_row_configurations()

            if panel_success:
                # Get chart files from results (same as panel submodule)
                chart_files = []
                if hasattr(processor, 'results') and 'chart_paths' in processor.results:
                    chart_files = list(processor.results['chart_paths'].values())
                    chart_files = [f for f in chart_files if f and os.path.exists(f)]

                self.chart_files = chart_files
                logger.info(f"Generated {len(chart_files)} overview charts using panel submodule")
                return chart_files
            else:
                logger.warning("Panel processing failed")
                return []

        except Exception as e:
            logger.error(f"Error in overview charts analysis: {e}")
            import traceback
            traceback.print_exc()
            return []