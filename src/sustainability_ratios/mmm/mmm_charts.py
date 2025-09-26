"""
MMM Chart Generation Module
============================

Generate charts for MMM gap analysis using the existing SR panel system.
Integrates with the chart_type system (line, candle, no_drawing).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class MmmChartsProcessor:
    """
    Process MMM chart generation using existing SR panel system.
    """

    def __init__(self, config, user_config, timeframe):
        """
        Initialize MMM charts processor.

        Args:
            config: System configuration
            user_config: User configuration
            timeframe: Processing timeframe ('daily', 'weekly', 'monthly')
        """
        self.config = config
        self.user_config = user_config
        self.timeframe = timeframe

    def run_gap_charts_analysis(self, gap_results: Dict):
        """
        Generate charts for gap analysis results.

        Args:
            gap_results: Results from gap calculation

        Returns:
            List of generated chart file paths
        """
        try:
            logger.info(f"Starting MMM chart generation for {self.timeframe} timeframe...")

            # Get chart configuration
            chart_config_file = getattr(self.user_config, 'sr_mmm_gaps_charts_display_panel', 'user_data_sr_mmm.csv')
            chart_history = getattr(self.user_config, 'sr_mmm_gaps_charts_display_history', 30)

            # Check if chart config file exists
            if not self._validate_chart_config(chart_config_file):
                logger.info("MMM chart config file not found, creating default configuration...")
                self._create_default_chart_config(chart_config_file, gap_results.get('tickers_processed', []))

            # Generate charts using SR panel system
            chart_files = self._generate_charts_using_sr_system(
                chart_config_file,
                chart_history,
                gap_results
            )

            logger.info(f"MMM chart generation completed: {len(chart_files)} charts generated")
            return chart_files

        except Exception as e:
            logger.error(f"Error in MMM chart generation: {e}")
            return []

    def _validate_chart_config(self, config_file: str) -> bool:
        """
        Validate that chart configuration file exists.

        Args:
            config_file: Path to chart configuration file

        Returns:
            True if file exists, False otherwise
        """
        try:
            # Check if path includes directory separators
            if '/' in config_file or '\\' in config_file:
                # Full path specified - use as-is (could be absolute or relative)
                config_path = Path(config_file)
                if not config_path.is_absolute():
                    config_path = Path(self.config.base_dir) / config_path
            else:
                # Just filename - file is in parent directory
                config_path = Path(self.config.base_dir) / config_file

            return config_path.exists()
        except Exception:
            return False

    def _resolve_config_path(self, config_file: str) -> Path:
        """
        Resolve configuration file path using consistent logic.

        Args:
            config_file: Configuration file name or path

        Returns:
            Resolved Path object
        """
        # Check if path includes directory separators
        if '/' in config_file or '\\' in config_file:
            # Full path specified - use as-is (could be absolute or relative)
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = Path(self.config.base_dir) / config_path
        else:
            # Just filename - file is in parent directory
            config_path = Path(self.config.base_dir) / config_file

        return config_path

    def _create_default_chart_config(self, config_file: str, tickers: List[str]):
        """
        Create default chart configuration for MMM analysis.

        Args:
            config_file: Path to create configuration file
            tickers: List of processed tickers
        """
        try:
            logger.info(f"Creating default MMM chart configuration: {config_file}")

            # Create default configuration entries for each ticker
            config_entries = []

            # Header
            config_entries.append("#file_name_id,chart_type,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index")

            # Create entries for each processed ticker
            for ticker in tickers:
                # Main gap analysis chart - using gap data file
                gap_ticker = f"{ticker}_gap"

                # Configuration for gap analysis:
                # Panel_1: Gap data with moving average overlay
                # Panel_1_index: Opening gap histogram
                # Panel_2_index: AdjustClose_woGap analysis
                config_entries.append(f"{ticker}_gaps,line,{gap_ticker},,,,,,\"A_HISTOGRAM(gap)\",\"B_LINE(AdjustClose_woGap)\",,,")

            # Write configuration file
            config_path = self._resolve_config_path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                f.write('\n'.join(config_entries))

            logger.info(f"Default MMM chart configuration created: {config_path}")

        except Exception as e:
            logger.error(f"Error creating default chart configuration: {e}")

    def _generate_charts_using_sr_system(self, config_file: str, chart_history: int, gap_results: Dict) -> List[str]:
        """
        Generate charts using the existing SR panel system following Overview module approach.

        Args:
            config_file: Chart configuration file
            chart_history: Number of periods to display
            gap_results: Gap calculation results (unused - for backward compatibility)

        Returns:
            List of generated chart file paths
        """
        try:
            logger.info(f"Starting MMM chart generation using SR system...")

            # Import SR components following Overview approach
            from ..sr_config_reader import parse_panel_config
            from ..sr_calculations import SRProcessor

            chart_files = []

            # Resolve configuration file path
            config_path = self._resolve_config_path(config_file)
            logger.info(f"Reading MMM chart configuration from: {config_path}")

            # Read panel configuration for MMM charts (same as Overview)
            try:
                panels_config = parse_panel_config(str(config_path))
                if not panels_config:
                    logger.warning("No panels configured in MMM chart config file")
                    return []

                logger.info(f"Found {len(panels_config)} panel configurations for MMM charts")

            except Exception as e:
                logger.warning(f"Error reading MMM chart config, using fallback: {e}")
                # Get MMM output directory for fallback charts
                from ..sr_output_manager import get_sr_output_manager
                output_manager = get_sr_output_manager()
                charts_dir = output_manager.get_submodule_dir('mmm') / 'charts'
                charts_dir.mkdir(parents=True, exist_ok=True)
                return self._generate_fallback_charts(gap_results, charts_dir)

            # Use SRProcessor with explicit config loading (same pattern as Overview)
            processor = SRProcessor(self.config, self.user_config, self.timeframe)

            # Override the panel configs with our MMM configs (prevent loading from default location)
            processor.panel_configs = panels_config

            # Process all row configurations using the correct working method
            # This method handles everything: loading data, processing panels, generating charts
            panel_success = processor.process_all_row_configurations()

            if panel_success:
                # Get chart files from results (same as Overview module)
                if hasattr(processor, 'results') and 'chart_paths' in processor.results:
                    chart_files = processor.results['chart_paths']
                    logger.info(f"✅ Generated {len(chart_files)} MMM charts")
                    for chart_file in chart_files:
                        logger.info(f"   - {chart_file}")
                else:
                    logger.warning("No chart paths found in processor results")

            else:
                logger.warning("Panel processing failed")

            return chart_files

        except Exception as e:
            logger.error(f"Error in SR system chart generation: {e}")
            import traceback
            traceback.print_exc()
            return []


    def _generate_single_chart(self, panel_config: Dict, chart_data: Dict,
                             charts_dir: Path, file_name_id: str, chart_history: int) -> Optional[str]:
        """
        Generate a single chart using SR dashboard system.

        Args:
            panel_config: Panel configuration
            chart_data: Chart data dictionary
            charts_dir: Charts output directory
            file_name_id: File name identifier
            chart_history: Number of periods to display

        Returns:
            Path to generated chart file or None if failed
        """
        try:
            # For now, create a simplified chart until full SR integration
            # This is a placeholder that will be enhanced when integrated with SR dashboard

            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime, timedelta

            # Extract data
            result = chart_data['result']
            gap = result.get('gap')
            adjust_close_wo_gap = result.get('AdjustClose_woGap')

            if gap is None or len(gap) == 0:
                return None

            # Limit to chart history
            if chart_history and chart_history > 0:
                gap = gap.tail(chart_history)
                adjust_close_wo_gap = adjust_close_wo_gap.tail(chart_history)

            # Create chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f'MMM Gap Analysis: {file_name_id}', fontsize=14, fontweight='bold')

            # Plot gaps
            ax1.plot(gap.index, gap.values, label='Gap', color='blue', linewidth=1.5)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            ax1.set_ylabel('Gap ($)')
            ax1.set_title('Gap: Open[i] - Close[i-1]')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot AdjustClose_woGap
            ax2.plot(adjust_close_wo_gap.index, adjust_close_wo_gap.values,
                    label='AdjustClose_woGap', color='green', linewidth=1.5)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            ax2.set_ylabel('Price Movement ($)')
            ax2.set_title('AdjustClose_woGap: Close[i] - (Close[i-1] - Open[i])')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            # Save chart
            chart_filename = f"mmm_{file_name_id}_{self.timeframe}.png"
            chart_path = charts_dir / chart_filename

            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Error generating single chart: {e}")
            return None

    def _generate_fallback_charts(self, gap_results: Dict, charts_dir: Path) -> List[str]:
        """
        Generate fallback charts when panel system is not available.

        Args:
            gap_results: Gap calculation results
            charts_dir: Charts output directory

        Returns:
            List of generated chart file paths
        """
        try:
            logger.info("Generating MMM fallback charts...")

            chart_files = []

            # Generate simple charts for each processed ticker
            for csv_file in gap_results.get('csv_files', []):
                try:
                    # Extract ticker from filename
                    ticker = Path(csv_file).stem.replace('_gap', '')

                    # Load gap data
                    gap_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

                    if 'gap' in gap_df.columns and 'AdjustClose_woGap' in gap_df.columns:
                        chart_file = self._generate_simple_gap_chart(gap_df, ticker, charts_dir)
                        if chart_file:
                            chart_files.append(chart_file)

                except Exception as e:
                    logger.error(f"Error generating fallback chart for {csv_file}: {e}")

            return chart_files

        except Exception as e:
            logger.error(f"Error generating fallback charts: {e}")
            return []

    def _generate_simple_gap_chart(self, gap_df: pd.DataFrame, ticker: str, charts_dir: Path) -> Optional[str]:
        """
        Generate a simple gap analysis chart.

        Args:
            gap_df: Gap data DataFrame
            ticker: Ticker symbol
            charts_dir: Charts output directory

        Returns:
            Path to generated chart file or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            # Limit to recent data
            chart_history = getattr(self.user_config, 'sr_mmm_gaps_charts_display_history', 30)
            if chart_history > 0:
                gap_df = gap_df.tail(chart_history)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f'MMM Gap Analysis: {ticker}', fontsize=14, fontweight='bold')

            # Plot gaps
            gap = gap_df['gap'].dropna()
            ax1.plot(gap.index, gap.values, label='Gap', color='blue', linewidth=1.5)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

            # Add moving averages if available
            # Moving averages removed - no longer calculated
                # 5-day MA removed - no longer calculated

            ax1.set_ylabel('Gap ($)')
            ax1.set_title('Gap: Open[i] - Close[i-1]')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot AdjustClose_woGap
            adjust_close_wo_gap = gap_df['AdjustClose_woGap'].dropna()
            ax2.plot(adjust_close_wo_gap.index, adjust_close_wo_gap.values,
                    label='AdjustClose_woGap', color='green', linewidth=1.5)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            ax2.set_ylabel('Price Movement ($)')
            ax2.set_title('AdjustClose_woGap: Close[i] - (Close[i-1] - Open[i])')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            # Save chart
            chart_filename = f"mmm_gaps_{ticker}_{self.timeframe}.png"
            chart_path = charts_dir / chart_filename

            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Generated simple gap chart: {chart_path}")
            return str(chart_path)

        except Exception as e:
            logger.error(f"Error generating simple gap chart for {ticker}: {e}")
            return None