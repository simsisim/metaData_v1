#!/usr/bin/env python3
"""
TradingView Watchlist Export - Standalone Script
===============================================

Export PVB screener results to TradingView watchlist format without re-running the screener.

Usage:
    # Export latest available files
    python scripts/export_watchlist_standalone.py

    # Export specific date
    python scripts/export_watchlist_standalone.py --date 20250929

    # Export specific ticker choice
    python scripts/export_watchlist_standalone.py --choice 2

    # Combine options
    python scripts/export_watchlist_standalone.py --date 20250929 --choice 2-5

Features:
    - Discovers latest PVB screener files automatically
    - Supports specific date or auto-detection
    - Handles multiple timeframes (daily, weekly, monthly)
    - Organizes by sections in TradingView format
    - Auto-splits if >1,000 symbols
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.user_defined_data import read_user_data
from src.tw_export_watchlist import export_pvb_watchlist

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for standalone watchlist export."""
    parser = argparse.ArgumentParser(
        description='Export PVB screener results to TradingView watchlist format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export latest available files
  python scripts/export_watchlist_standalone.py

  # Export specific date
  python scripts/export_watchlist_standalone.py --date 20250929

  # Export specific ticker choice
  python scripts/export_watchlist_standalone.py --choice 2-5

  # Dry run (show files without exporting)
  python scripts/export_watchlist_standalone.py --date 20250929 --dry-run
        """
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Target date (YYYYMMDD format). If omitted, uses latest available.'
    )

    parser.add_argument(
        '--choice',
        type=str,
        help='Ticker choice (e.g., "2", "2-5"). If omitted, uses value from user_data.csv'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show which files would be exported without actually exporting'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        print("📋 Loading configuration...")
        config = Config()
        user_config = read_user_data()

        # Use provided choice or default from config
        ticker_choice = args.choice if args.choice else str(user_config.ticker_choice)
        target_date = args.date  # None means auto-detect

        print(f"🎯 Ticker choice: {ticker_choice}")
        print(f"📅 Target date: {target_date if target_date else 'Latest available'}")

        # Dry run mode - just show files
        if args.dry_run:
            print(f"\n🔍 DRY RUN MODE - Discovering files...")

            from src.tw_export_watchlist import find_latest_pvb_files

            output_dir = config.directories.get('PVB_SCREENER_DIR', config.base_dir / 'results' / 'screeners' / 'pvbTW')
            files = find_latest_pvb_files(output_dir, ticker_choice, target_date)

            if not files:
                print(f"❌ No PVB screener files found")
                print(f"   Searched: {output_dir}")
                print(f"   Pattern: pvb_screener_{ticker_choice}_*")
                return 1

            print(f"\n✓ Found {len(files)} files that would be exported:")
            for file in files:
                print(f"   📄 {file.name}")

            print(f"\n💡 Remove --dry-run to actually export these files")
            return 0

        # Actual export
        print(f"\n📊 Exporting PVB screener to TradingView watchlist...")

        watchlist_files = export_pvb_watchlist(
            config=config,
            user_config=user_config,
            csv_files=None,  # Standalone mode - will discover files
            date=target_date
        )

        if watchlist_files:
            print(f"\n✅ SUCCESS! Watchlist exported")
            return 0
        else:
            print(f"\n❌ FAILED! No watchlist files created")
            return 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Export cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=args.verbose)
        print(f"\n❌ ERROR: {e}")
        if not args.verbose:
            print(f"💡 Run with --verbose for detailed error information")
        return 1


if __name__ == '__main__':
    sys.exit(main())