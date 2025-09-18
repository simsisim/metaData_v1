"""
Simple Market Breadth Chart Generator
====================================

Clean, minimal approach to chart generation starting with SPY/QQQ only.
Used to isolate and fix the 63-day timeline filtering issue.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SimpleBreadthChart:
    """
    Minimal chart generator for debugging timeline issues.
    Starts with SPY/QQQ only, adds complexity incrementally.
    """

    def __init__(self, config=None, user_config=None):
        self.config = config
        self.user_config = user_config

    def _get_chart_display_period(self, timeframe: str = 'daily') -> int:
        """Get the configured display period for charts."""
        if not self.user_config:
            return 63  # Default for testing

        periods = {
            'daily': getattr(self.user_config, 'market_breadth_chart_history_days', 63),
            'weekly': getattr(self.user_config, 'market_breadth_chart_history_weeks', 12),
            'monthly': getattr(self.user_config, 'market_breadth_chart_history_months', 3)
        }
        return periods.get(timeframe, 63)

    def _load_spy_data(self) -> pd.DataFrame:
        """Load SPY price data."""
        try:
            daily_data_dir = self.config.get_market_data_dir('daily')
            spy_file = daily_data_dir / "SPY.csv"

            if spy_file.exists():
                spy_data = pd.read_csv(spy_file)
                logger.info(f"Loaded SPY data: {len(spy_data)} records")
                return spy_data
            else:
                logger.error(f"SPY file not found: {spy_file}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading SPY data: {e}")
            return pd.DataFrame()

    def _filter_to_display_period(self, df: pd.DataFrame, timeframe: str = 'daily') -> pd.DataFrame:
        """Filter data to configured display period."""
        if df.empty:
            return df

        display_periods = self._get_chart_display_period(timeframe)
        logger.info(f"Filtering {len(df)} records to last {display_periods} {timeframe} periods")

        # Handle both 'Date' and 'date' columns
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date')
        elif 'date' in df.columns:
            df_sorted = df.sort_values('date')
        else:
            df_sorted = df.sort_index()

        filtered_df = df_sorted.tail(display_periods)
        logger.info(f"Filtered result: {len(filtered_df)} records")

        return filtered_df

    def _load_breadth_data(self, universe: str = 'SP500') -> pd.DataFrame:
        """Load market breadth data."""
        try:
            breadth_file = Path(f"results/market_breadth/market_breadth_{universe}_2-5_daily_20250905.csv")

            if breadth_file.exists():
                breadth_data = pd.read_csv(breadth_file)
                logger.info(f"Loaded breadth data: {len(breadth_data)} records")
                return breadth_data
            else:
                logger.error(f"Breadth file not found: {breadth_file}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading breadth data: {e}")
            return pd.DataFrame()

    def create_minimal_spy_chart(self, output_path: str, timeframe: str = 'daily') -> str:
        """
        Phase 1: Create minimal SPY-only chart to test 63-day filtering.

        Args:
            output_path: Path for output PNG file
            timeframe: Data timeframe

        Returns:
            Path to generated chart or None if failed
        """
        try:
            logger.info("=== PHASE 1: Creating minimal SPY chart ===")

            # Load SPY data
            spy_data = self._load_spy_data()
            if spy_data.empty:
                logger.error("No SPY data available")
                return None

            # Filter to display period
            filtered_spy = self._filter_to_display_period(spy_data, timeframe)
            if filtered_spy.empty:
                logger.error("No data after filtering")
                return None

            # Log data details
            date_col = 'Date' if 'Date' in filtered_spy.columns else 'date'
            if date_col in filtered_spy.columns:
                logger.info(f"SPY data range: {filtered_spy[date_col].min()} to {filtered_spy[date_col].max()}")

            # Prepare data for plotting
            dates = pd.to_datetime(filtered_spy[date_col])
            close_prices = filtered_spy['Close'] if 'Close' in filtered_spy.columns else filtered_spy['close']

            logger.info(f"Plotting {len(dates)} data points")
            logger.info(f"Date range for plotting: {dates.min().date()} to {dates.max().date()}")

            # Create simple chart
            plt.style.use('default')
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)

            # Plot SPY price
            ax.plot(dates, close_prices, 'b-', linewidth=2, label='SPY Close')

            # Log xlim before manual setting
            xlim_before = ax.get_xlim()
            logger.info(f"Auto xlim before manual setting: {xlim_before}")

            # Set xlim explicitly
            ax.set_xlim(dates.min(), dates.max())

            # Log xlim after manual setting
            xlim_after = ax.get_xlim()
            logger.info(f"Manual xlim after setting: {xlim_after}")

            # Convert xlim to dates for verification
            xlim_min_date = mdates.num2date(xlim_after[0])
            xlim_max_date = mdates.num2date(xlim_after[1])
            logger.info(f"Final xlim as dates: {xlim_min_date.date()} to {xlim_max_date.date()}")

            # Format chart
            ax.set_title(f'SPY Price Chart - {timeframe.title()} (Minimal Test)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Set tight margins
            ax.margins(x=0)

            # Save chart
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Minimal SPY chart saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating minimal SPY chart: {e}")
            return None


def main():
    """Test the minimal chart generator."""
    import sys
    sys.path.append('.')
    from src.config import Config
    from src.user_defined_data import read_user_data

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    config = Config()
    user_config = read_user_data()

    chart_gen = SimpleBreadthChart(config, user_config)

    # Test Phase 1: Minimal SPY chart
    output_path = "PHASE1_MINIMAL_SPY_CHART.png"
    result = chart_gen.create_minimal_spy_chart(output_path)

    if result:
        print(f"✅ Phase 1 successful: {result}")
        print("This chart should show ONLY the last 63 trading days")
    else:
        print("❌ Phase 1 failed")


if __name__ == "__main__":
    main()