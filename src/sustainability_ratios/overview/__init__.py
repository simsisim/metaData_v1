"""
Sustainability Ratios Overview Submodule
=======================================

Overview analysis for SR module providing short-term performance insights
for indexes, sectors, and industries.

Main Components:
- OverviewValuesProcessor: Calculates percentage changes and performance metrics
- OverviewChartsProcessor: Generates visual charts and comparisons
- OverviewProcessor: Main controller integrating both components

Usage:
    from src.sustainability_ratios.overview import OverviewProcessor

    processor = OverviewProcessor(config, user_config, timeframe)
    processor.run_complete_analysis()
"""

from .overview_values import OverviewValuesProcessor
from .overview_charts import OverviewChartsProcessor

__all__ = [
    'OverviewValuesProcessor',
    'OverviewChartsProcessor',
    'OverviewProcessor'
]


class OverviewProcessor:
    """
    Main controller for SR overview submodule.
    Coordinates values calculation and chart generation.
    """

    def __init__(self, config, user_config, timeframe):
        """
        Initialize overview processor.

        Args:
            config: System configuration
            user_config: User configuration
            timeframe: Processing timeframe ('daily', 'weekly', 'monthly')
        """
        self.config = config
        self.user_config = user_config
        self.timeframe = timeframe

        # Initialize subprocessors
        self.values_processor = OverviewValuesProcessor(config, user_config, timeframe)
        self.charts_processor = OverviewChartsProcessor(config, user_config, timeframe)

        self.results = {}

    def run_complete_analysis(self):
        """
        Run complete overview analysis including values and charts.

        Returns:
            Dictionary with analysis results and file paths
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            logger.info(f"Starting complete overview analysis for {self.timeframe} timeframe...")

            analysis_results = {
                'timeframe': self.timeframe,
                'values_results': {},
                'chart_files': [],
                'csv_files': [],
                'success': False
            }

            # Step 1: Calculate overview values
            if self.user_config.sr_overview_values_enable:
                logger.info("Calculating overview values...")
                values_results = self.values_processor.run_overview_values_analysis()
                analysis_results['values_results'] = values_results

                if values_results and 'error' not in values_results:
                    # Save values to CSV
                    from pathlib import Path
                    output_dir = Path(getattr(self.user_config, 'sr_overview_output_dir',
                                            'results/sustainability_ratios/overview'))
                    filename_prefix = getattr(self.user_config, 'sr_overview_filename_prefix', 'sr_overview')

                    csv_files = self.values_processor.save_results_to_csv(output_dir, filename_prefix)
                    analysis_results['csv_files'] = csv_files

                    logger.info(f"Overview values completed: {len(csv_files)} CSV files generated")
                else:
                    logger.warning("Overview values calculation failed or returned no results")

            # Step 2: Generate overview charts (if enabled)
            if (self.user_config.sr_overview_charts_enable and
                analysis_results['values_results'] and
                'error' not in analysis_results['values_results']):

                logger.info("Generating overview charts...")
                chart_files = self.charts_processor.run_overview_charts_analysis(
                    analysis_results['values_results']
                )
                analysis_results['chart_files'] = chart_files

                logger.info(f"Overview charts completed: {len(chart_files)} chart files generated")
            else:
                logger.info("Overview charts generation skipped (disabled or no values data)")

            # Determine overall success
            analysis_results['success'] = (
                len(analysis_results.get('csv_files', [])) > 0 or
                len(analysis_results.get('chart_files', [])) > 0
            )

            total_files = len(analysis_results.get('csv_files', [])) + len(analysis_results.get('chart_files', []))
            logger.info(f"Complete overview analysis finished: {total_files} total files generated")

            self.results = analysis_results
            return analysis_results

        except Exception as e:
            logger.error(f"Error in complete overview analysis: {e}")
            return {
                'timeframe': self.timeframe,
                'error': str(e),
                'success': False
            }