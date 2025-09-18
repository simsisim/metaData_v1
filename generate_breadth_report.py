#!/usr/bin/env python3
"""
Market Breadth Report Generation Script
======================================

Utility script to generate market breadth PDF reports from existing CSV and PNG files.
Automatically finds the latest market breadth data and generates comprehensive reports.

Usage:
    python generate_breadth_report.py --universe SP500 --timeframe daily
    python generate_breadth_report.py --csv path/to/file.csv --png path/to/chart.png
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.market_pulse.reporting.breadth_reporting import BreadthReportGenerator
from src.config import Config
from src.user_defined_data import read_user_data


def find_latest_files(universe: str, timeframe: str = 'daily') -> tuple:
    """
    Find the latest CSV and PNG files for given universe and timeframe.

    Args:
        universe: Market universe (SP500, NASDAQ100, etc.)
        timeframe: Data timeframe (daily, weekly, monthly)

    Returns:
        tuple: (csv_path, png_path) or (None, None) if not found
    """
    try:
        config = Config()
        results_dir = Path("results/market_breadth")

        if not results_dir.exists():
            print(f"❌ Market breadth results directory not found: {results_dir}")
            return None, None

        # Find matching files
        pattern = f"market_breadth_{universe}_*_{timeframe}_*.csv"
        csv_files = list(results_dir.glob(pattern))

        if not csv_files:
            print(f"❌ No CSV files found matching pattern: {pattern}")
            return None, None

        # Get the latest CSV file (by filename timestamp)
        latest_csv = sorted(csv_files)[-1]

        # Find corresponding PNG file
        png_pattern = latest_csv.stem + ".png"
        png_path = results_dir / png_pattern

        if not png_path.exists():
            print(f"⚠️  PNG chart not found: {png_path}")
            png_path = None

        return latest_csv, png_path

    except Exception as e:
        print(f"Error finding files: {e}")
        return None, None


def generate_report_for_universe(universe: str, timeframe: str = 'daily',
                                output_dir: Path = None, days: int = 10) -> bool:
    """
    Generate report for specified universe and timeframe.

    Args:
        universe: Market universe
        timeframe: Data timeframe
        output_dir: Output directory (defaults to results/reports/)
        days: Number of recent days to analyze

    Returns:
        bool: Success status
    """
    try:
        print(f"🔍 Finding latest {universe} {timeframe} data...")

        csv_path, png_path = find_latest_files(universe, timeframe)

        if not csv_path:
            return False

        print(f"📊 Using data: {csv_path.name}")
        if png_path:
            print(f"📈 Using chart: {png_path.name}")

        # Setup output directory
        if output_dir is None:
            output_dir = Path("results/reports")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename based on CSV filename (without extension)
        output_filename = csv_path.stem.replace('market_breadth_', 'breadth_report_') + '.pdf'
        output_path = output_dir / output_filename

        # Load user configuration for market breadth reporting
        user_config = read_user_data()

        # Generate report
        print(f"📄 Generating report: {output_path.name}")

        generator = BreadthReportGenerator(user_config)

        # Check if reporting is enabled
        if not generator.is_enabled():
            print("⚠️  Market breadth reporting is disabled in configuration")
            return False

        success = generator.generate_report(csv_path, png_path, output_path, days)

        if success:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ Report generated successfully!")
            print(f"   📁 Location: {output_path}")
            print(f"   📏 Size: {size_mb:.2f} MB")
            return True
        else:
            print("❌ Report generation failed")
            return False

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False


def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description='Generate Market Breadth PDF Reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --universe SP500                    # Latest SP500 daily report
  %(prog)s --universe NASDAQ100 --timeframe weekly   # NASDAQ100 weekly report
  %(prog)s --csv data.csv --png chart.png --output report.pdf  # Custom files
  %(prog)s --all-universes                     # Generate reports for all available universes
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--universe', help='Market universe (SP500, NASDAQ100, etc.)')
    mode_group.add_argument('--csv', help='Specific CSV file path')
    mode_group.add_argument('--all-universes', action='store_true',
                           help='Generate reports for all available universes')

    # Options
    parser.add_argument('--timeframe', default='daily', choices=['daily', 'weekly', 'monthly'],
                       help='Data timeframe (default: daily)')
    parser.add_argument('--png', help='Specific PNG chart path (used with --csv)')
    parser.add_argument('--output', help='Output PDF path (used with --csv)')
    parser.add_argument('--output-dir', help='Output directory for reports')
    parser.add_argument('--days', type=int, default=10,
                       help='Number of recent days to analyze (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        if args.csv:
            # Custom file mode
            csv_path = Path(args.csv)
            png_path = Path(args.png) if args.png else None
            output_path = Path(args.output) if args.output else Path(f"breadth_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

            if not csv_path.exists():
                print(f"❌ CSV file not found: {csv_path}")
                return 1

            # Load user configuration
            user_config = read_user_data()
            generator = BreadthReportGenerator(user_config)

            # Check if reporting is enabled
            if not generator.is_enabled():
                print("⚠️  Market breadth reporting is disabled in configuration")
                return 1

            success = generator.generate_report(csv_path, png_path, output_path, args.days)

            if success:
                print(f"✅ Report generated: {output_path}")
                return 0
            else:
                print("❌ Report generation failed")
                return 1

        elif args.all_universes:
            # Generate for all available universes
            print("🔍 Searching for all available market breadth data...")

            results_dir = Path("results/market_breadth")
            if not results_dir.exists():
                print(f"❌ Results directory not found: {results_dir}")
                return 1

            # Find all unique universe names
            csv_files = list(results_dir.glob("market_breadth_*_daily_*.csv"))
            universes = set()

            for csv_file in csv_files:
                parts = csv_file.stem.split('_')
                if len(parts) >= 3:
                    universe = parts[2]  # market_breadth_UNIVERSE_...
                    universes.add(universe)

            if not universes:
                print("❌ No market breadth data found")
                return 1

            print(f"📊 Found {len(universes)} universes: {', '.join(sorted(universes))}")

            success_count = 0
            for universe in sorted(universes):
                print(f"\n--- Processing {universe} ---")
                if generate_report_for_universe(universe, args.timeframe, output_dir, args.days):
                    success_count += 1

            print(f"\n✅ Generated {success_count}/{len(universes)} reports successfully")
            return 0 if success_count > 0 else 1

        else:
            # Single universe mode
            success = generate_report_for_universe(args.universe, args.timeframe, output_dir, args.days)
            return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())