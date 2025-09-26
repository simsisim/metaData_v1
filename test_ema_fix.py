#!/usr/bin/env python3
"""
Test EMA/SMA Overlay Fix
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from src.config import Config
    from src.user_defined_data import UserConfiguration
    from src.sustainability_ratios.overview.overview_charts import OverviewChartsProcessor

    def test_ema_fix():
        """Test that EMA/SMA overlays now work after the AUTO_DETECT fix."""
        try:
            config = Config()
            user_config = UserConfiguration('user_data.csv')

            # Test processor with updated CSV that has bundled indicators
            processor = OverviewChartsProcessor(config, user_config, 'daily')
            chart_files = processor.run_overview_charts_analysis()

            logger.info(f"Generated {len(chart_files)} chart files after AUTO_DETECT fix:")
            for chart_file in chart_files:
                logger.info(f"  - {Path(chart_file).name}")

            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    if __name__ == "__main__":
        success = test_ema_fix()
        sys.exit(0 if success else 1)

except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)