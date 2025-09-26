"""
Sustainability Ratios MMM (Market Maker Manipulation) Submodule
===============================================================

MMM analysis for SR module providing gap analysis and market manipulation insights
for configured tickers.

Main Components:
- MmmGapsProcessor: Calculates gap metrics and generates gap data files
- MmmChartsProcessor: Generates visual charts using existing panel system
- MmmProcessor: Main controller integrating both components

Usage:
    from src.sustainability_ratios.mmm import MmmProcessor

    processor = MmmProcessor(config, user_config, timeframe)
    processor.run_complete_analysis()
"""

from .mmm_gaps import MmmGapsProcessor
from .mmm_charts import MmmChartsProcessor

__all__ = [
    'MmmGapsProcessor',
    'MmmChartsProcessor',
    'MmmProcessor'
]


class MmmProcessor:
    """
    Main controller for SR MMM submodule.
    Coordinates gap calculation and chart generation.
    """

    def __init__(self, config, user_config, timeframe):
        """
        Initialize MMM processor.

        Args:
            config: System configuration
            user_config: User configuration
            timeframe: Processing timeframe ('daily', 'weekly', 'monthly')
        """
        self.config = config
        self.user_config = user_config
        self.timeframe = timeframe

        # Initialize subprocessors
        self.gaps_processor = MmmGapsProcessor(config, user_config, timeframe)
        self.charts_processor = MmmChartsProcessor(config, user_config, timeframe)

        self.results = {}

    def run_complete_analysis(self):
        """
        Run complete MMM analysis including gap calculation and charts.

        Returns:
            Dictionary with analysis results and file paths
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            logger.info(f"Starting complete MMM analysis for {self.timeframe} timeframe...")

            analysis_results = {
                'timeframe': self.timeframe,
                'gap_results': {},
                'chart_files': [],
                'csv_files': [],
                'success': False
            }

            # Step 1: Calculate gap values
            if getattr(self.user_config, 'sr_mmm_gaps_values', False):
                logger.info("Calculating MMM gap values...")
                gap_results = self.gaps_processor.run_gap_analysis()
                analysis_results['gap_results'] = gap_results

                if gap_results and 'error' not in gap_results:
                    # Gap CSV files are saved within the gaps processor
                    analysis_results['csv_files'] = gap_results.get('csv_files', [])

                    logger.info(f"MMM gap values completed: {len(analysis_results['csv_files'])} CSV files generated")
                else:
                    logger.warning("MMM gap calculation failed or returned no results")

            # Step 2: Generate MMM charts (if enabled)
            if (getattr(self.user_config, 'sr_mmm_gaps_chart_enable', False) and
                analysis_results['gap_results'] and
                'error' not in analysis_results['gap_results']):

                logger.info("Generating MMM charts...")
                chart_files = self.charts_processor.run_gap_charts_analysis(
                    analysis_results['gap_results']
                )
                analysis_results['chart_files'] = chart_files

                logger.info(f"MMM charts completed: {len(chart_files)} chart files generated")
            else:
                logger.info("MMM charts generation skipped (disabled or no gap data)")

            # Determine overall success
            analysis_results['success'] = (
                len(analysis_results.get('csv_files', [])) > 0 or
                len(analysis_results.get('chart_files', [])) > 0
            )

            total_files = len(analysis_results.get('csv_files', [])) + len(analysis_results.get('chart_files', []))
            logger.info(f"Complete MMM analysis finished: {total_files} total files generated")

            self.results = analysis_results
            return analysis_results

        except Exception as e:
            logger.error(f"Error in complete MMM analysis: {e}")
            return {
                'timeframe': self.timeframe,
                'error': str(e),
                'success': False
            }