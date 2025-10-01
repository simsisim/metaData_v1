"""
TradingView Watchlist Export Module
===================================

Exports screener results to TradingView-compatible watchlist format.

Features:
- Exports PVB (and other) screener results to .txt format
- Supports EXCHANGE:TICKER format required by TradingView
- Organizes by timeframe and signal type (###Section syntax)
- Handles 1,000 symbol limit with auto-splitting
- Supports both integrated (current run) and standalone (file discovery) modes

Format Example:
    ###PVB_Daily_Buy
    NASDAQ:AAPL, NASDAQ:MSFT, NYSE:IBM

    ###PVB_Daily_Sell
    NASDAQ:TSLA, NASDAQ:META

Usage:
    # Integrated mode (Phase 1)
    from src.tw_export_watchlist import export_pvb_watchlist
    export_pvb_watchlist(config, user_config, csv_files=file_list)

    # Standalone mode (Phase 2)
    export_pvb_watchlist(config, user_config, date='20250929')
"""

import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TradingViewWatchlistExporter:
    """
    Exports screener results to TradingView watchlist format.

    Supports:
    - Multiple timeframes (daily, weekly, monthly)
    - Section organization by timeframe + signal type
    - Auto-splitting at 1,000 symbol limit
    - EXCHANGE:TICKER formatting
    """

    def __init__(self, output_dir: Path):
        """
        Initialize watchlist exporter.

        Args:
            output_dir: Directory where watchlist files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TradingView symbol limit per watchlist
        self.max_symbols_per_file = 1000

        logger.info(f"TradingView Watchlist Exporter initialized: {self.output_dir}")

    def export_pvb_screener(
        self,
        csv_files: List[Path],
        output_filename: str,
        include_buy: bool = True,
        include_sell: bool = True
    ) -> List[Path]:
        """
        Export PVB screener results to TradingView watchlist.

        Args:
            csv_files: List of PVB CSV file paths to process
            output_filename: Base filename for output (e.g., 'pvb_watchlist_2_20250930.txt')
            include_buy: Include Buy signals
            include_sell: Include Sell signals

        Returns:
            List of created watchlist file paths (may be multiple if split)
        """
        if not csv_files:
            logger.warning("No CSV files provided for watchlist export")
            return []

        logger.info(f"Exporting {len(csv_files)} PVB files to TradingView watchlist")

        # Parse all CSV files
        all_data = []
        for csv_file in csv_files:
            try:
                df = self._parse_csv_file(csv_file)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Error parsing {csv_file.name}: {e}")
                continue

        if not all_data:
            logger.error("No valid data found in CSV files")
            return []

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total signals from CSV files")

        # Filter signal types (include Close signals with their parent types)
        signal_filter = []
        if include_buy:
            signal_filter.extend(['Buy', 'Close_Buy'])
        if include_sell:
            signal_filter.extend(['Sell', 'Close_Sell'])

        if not signal_filter:
            logger.error("No signal types selected (both buy and sell disabled)")
            return []

        filtered_df = combined_df[combined_df['signal_type'].isin(signal_filter)]
        logger.info(f"Filtered to {len(filtered_df)} signals (types: {', '.join(signal_filter)})")

        # Group by timeframe and signal type
        sections = self._group_by_timeframe_and_signal(filtered_df)

        # Build watchlist content
        watchlist_content = self._format_watchlist_content(sections)

        # Count total symbols
        total_symbols = sum(len(tickers) for tickers in sections.values())

        # Check if splitting needed
        if total_symbols > self.max_symbols_per_file:
            logger.warning(f"Total symbols ({total_symbols}) exceeds limit ({self.max_symbols_per_file}), splitting files")
            return self._split_and_write_watchlist(watchlist_content, sections, output_filename)
        else:
            # Single file
            output_path = self.output_dir / output_filename
            self._write_watchlist_file(output_path, watchlist_content)
            logger.info(f"✓ Watchlist exported: {output_filename} ({total_symbols} symbols)")
            return [output_path]

    def _parse_csv_file(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse CSV file and extract required columns.

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with ticker, exchange, timeframe, signal_type, days_since_signal columns
        """
        try:
            # Read only required columns
            required_cols = ['ticker', 'exchange', 'timeframe', 'signal_type', 'days_since_signal']

            # Check if file exists
            if not csv_path.exists():
                logger.warning(f"File not found: {csv_path}")
                return None

            # Read CSV
            df = pd.read_csv(csv_path, usecols=lambda x: x in required_cols)

            # Verify required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {csv_path.name}: {missing_cols}")

                # If exchange missing, try to add default
                if 'exchange' in missing_cols and 'ticker' in df.columns:
                    df['exchange'] = 'NASDAQ'  # Default
                    logger.info(f"Added default exchange 'NASDAQ' for {csv_path.name}")
                    missing_cols.remove('exchange')

                # If days_since_signal missing, add default
                if 'days_since_signal' in missing_cols:
                    df['days_since_signal'] = 0
                    logger.info(f"Added default days_since_signal=0 for {csv_path.name}")
                    missing_cols.remove('days_since_signal')

                # If still missing critical columns, skip
                if missing_cols:
                    return None

            # Extract timeframe from filename if not in CSV
            if 'timeframe' not in df.columns:
                timeframe = self._extract_timeframe_from_filename(csv_path.name)
                df['timeframe'] = timeframe
                logger.debug(f"Extracted timeframe '{timeframe}' from filename")

            # Clean data
            df = df.dropna(subset=['ticker'])
            df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
            df['exchange'] = df['exchange'].fillna('NASDAQ').astype(str).str.strip().str.upper()
            df['signal_type'] = df['signal_type'].astype(str).str.strip().str.replace(' ', '_')  # Normalize: "Close Buy" → "Close_Buy"
            df['days_since_signal'] = pd.to_numeric(df['days_since_signal'], errors='coerce').fillna(0)

            logger.debug(f"Parsed {csv_path.name}: {len(df)} signals")
            return df

        except Exception as e:
            logger.error(f"Error parsing {csv_path.name}: {e}")
            return None

    def _extract_timeframe_from_filename(self, filename: str) -> str:
        """
        Extract timeframe from filename.

        Args:
            filename: CSV filename (e.g., 'pvb_screener_2_daily_20250930.csv')

        Returns:
            Timeframe string ('daily', 'weekly', 'monthly')
        """
        filename_lower = filename.lower()

        if 'daily' in filename_lower:
            return 'daily'
        elif 'weekly' in filename_lower:
            return 'weekly'
        elif 'monthly' in filename_lower:
            return 'monthly'
        else:
            logger.warning(f"Could not extract timeframe from {filename}, defaulting to 'daily'")
            return 'daily'

    def _group_by_timeframe_and_signal(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Group tickers by timeframe and signal type, sorted by days_since_signal ascending.

        Args:
            df: DataFrame with ticker, exchange, timeframe, signal_type, days_since_signal

        Returns:
            Dictionary: {section_name: [EXCHANGE:TICKER list]} sorted by days_since_signal
        """
        sections = {}

        # Group by timeframe and signal_type
        for (timeframe, signal_type), group in df.groupby(['timeframe', 'signal_type']):
            # Deduplicate tickers within section (keep first occurrence)
            unique_group = group.drop_duplicates(subset='ticker', keep='first')

            # Sort by days_since_signal ascending (most recent first)
            unique_group = unique_group.sort_values('days_since_signal', ascending=True)

            # Format as EXCHANGE:TICKER
            symbols = [
                f"{row['exchange']}:{row['ticker']}"
                for _, row in unique_group.iterrows()
            ]

            # Section name: PVB_Daily_Buy, PVB_Daily_Sell, PVB_Daily_Close_Buy, PVB_Daily_Close_Sell
            section_name = f"PVB_{timeframe.capitalize()}_{signal_type}"

            sections[section_name] = symbols

            logger.debug(f"Section '{section_name}': {len(symbols)} symbols (sorted by days_since_signal)")

        return sections

    def _format_watchlist_content(self, sections: Dict[str, List[str]]) -> str:
        """
        Format watchlist content with sections.

        Args:
            sections: Dictionary of {section_name: [symbols]}

        Returns:
            Formatted watchlist string
        """
        content_lines = []

        # Define section order: Daily → Weekly → Monthly, Buy → Sell → Close_Buy → Close_Sell
        timeframe_order = ['daily', 'weekly', 'monthly']
        signal_order = ['Buy', 'Sell', 'Close_Buy', 'Close_Sell']

        for timeframe in timeframe_order:
            for signal_type in signal_order:
                section_name = f"PVB_{timeframe.capitalize()}_{signal_type}"

                if section_name in sections:
                    symbols = sections[section_name]

                    # Add section header
                    content_lines.append(f"###{section_name}")

                    # Add symbols (comma-separated)
                    content_lines.append(", ".join(symbols))

                    # Add blank line between sections
                    content_lines.append("")

        return "\n".join(content_lines)

    def _split_and_write_watchlist(
        self,
        content: str,
        sections: Dict[str, List[str]],
        base_filename: str
    ) -> List[Path]:
        """
        Split watchlist into multiple files if exceeds symbol limit.

        Args:
            content: Full watchlist content string
            sections: Dictionary of sections
            base_filename: Base filename (e.g., 'pvb_watchlist_2_20250930.txt')

        Returns:
            List of created file paths
        """
        # Strategy: Split sections across multiple files
        created_files = []
        current_symbols = 0
        current_sections = []
        part_num = 1

        # Process sections in order
        timeframe_order = ['daily', 'weekly', 'monthly']
        signal_order = ['Buy', 'Sell', 'Close_Buy', 'Close_Sell']

        for timeframe in timeframe_order:
            for signal_type in signal_order:
                section_name = f"PVB_{timeframe.capitalize()}_{signal_type}"

                if section_name not in sections:
                    continue

                symbols = sections[section_name]
                section_symbol_count = len(symbols)

                # Check if adding this section would exceed limit
                if current_symbols + section_symbol_count > self.max_symbols_per_file and current_sections:
                    # Write current part
                    part_filename = base_filename.replace('.txt', f'_part{part_num}.txt')
                    part_path = self.output_dir / part_filename

                    part_content = self._format_sections(current_sections)
                    self._write_watchlist_file(part_path, part_content)

                    logger.info(f"✓ Watchlist part {part_num} exported: {part_filename} ({current_symbols} symbols)")
                    created_files.append(part_path)

                    # Reset for next part
                    current_sections = []
                    current_symbols = 0
                    part_num += 1

                # Add section to current part
                current_sections.append((section_name, symbols))
                current_symbols += section_symbol_count

        # Write final part
        if current_sections:
            if part_num > 1:
                part_filename = base_filename.replace('.txt', f'_part{part_num}.txt')
            else:
                part_filename = base_filename

            part_path = self.output_dir / part_filename
            part_content = self._format_sections(current_sections)
            self._write_watchlist_file(part_path, part_content)

            logger.info(f"✓ Watchlist part {part_num} exported: {part_filename} ({current_symbols} symbols)")
            created_files.append(part_path)

        return created_files

    def _format_sections(self, sections: List[Tuple[str, List[str]]]) -> str:
        """
        Format list of sections into watchlist content.

        Args:
            sections: List of (section_name, symbols) tuples

        Returns:
            Formatted content string
        """
        content_lines = []

        for section_name, symbols in sections:
            content_lines.append(f"###{section_name}")
            content_lines.append(", ".join(symbols))
            content_lines.append("")

        return "\n".join(content_lines)

    def _write_watchlist_file(self, filepath: Path, content: str) -> None:
        """
        Write watchlist content to file.

        Args:
            filepath: Output file path
            content: Watchlist content string
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.debug(f"Wrote watchlist to {filepath}")

        except Exception as e:
            logger.error(f"Error writing watchlist file {filepath}: {e}")
            raise


def find_latest_pvb_files(
    screener_dir: Path,
    ticker_choice: str,
    date: Optional[str] = None
) -> List[Path]:
    """
    Find latest PVB screener files for watchlist export (Phase 2 - Standalone mode).

    Args:
        screener_dir: Directory containing PVB CSV files
        ticker_choice: User's ticker choice (e.g., '2', '2-5')
        date: Optional target date (YYYYMMDD). If None, uses today.
              If today's files missing, auto-detects latest available.

    Returns:
        List of CSV file paths sorted by timeframe (daily, weekly, monthly)

    Examples:
        # Find today's files
        files = find_latest_pvb_files(dir, '2')

        # Find specific date
        files = find_latest_pvb_files(dir, '2', '20250929')

        # Find latest available (any date)
        files = find_latest_pvb_files(dir, '2', date=None)
    """
    screener_dir = Path(screener_dir)

    # Step 1: Determine target date
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    # Step 2: Build search pattern
    # Matches: pvb_screener_2_daily_20250930.csv
    #          pvb_screener_2-5_weekly_20250930.csv
    pattern = f'pvb_screener_{ticker_choice}_*_{date}.csv'

    # Step 3: Search for exact date match
    files = list(screener_dir.glob(pattern))

    if files:
        logger.info(f"Found {len(files)} PVB files for date {date}")
        # Sort by timeframe order: daily, weekly, monthly
        return sorted(files, key=lambda x: (
            'weekly' in x.name,  # False (daily) sorts before True (weekly/monthly)
            'monthly' in x.name
        ))

    # Step 4: No files for target date - find latest available
    logger.warning(f"No PVB files found for date {date}, searching for latest available")

    # Get all PVB files for this ticker_choice
    pattern_all = f'pvb_screener_{ticker_choice}_*.csv'
    all_files = list(screener_dir.glob(pattern_all))

    if not all_files:
        logger.error(f"No PVB screener files found for ticker_choice '{ticker_choice}' in {screener_dir}")
        return []

    # Step 5: Extract dates and find latest
    file_dates = {}
    for file in all_files:
        # Extract date from filename: pvb_screener_2_daily_20250930.csv -> 20250930
        match = re.search(r'_(\d{8})\.csv$', file.name)
        if match:
            file_date = match.group(1)
            if file_date not in file_dates:
                file_dates[file_date] = []
            file_dates[file_date].append(file)

    if not file_dates:
        logger.error("Could not parse dates from PVB filenames")
        return []

    # Step 6: Get files from latest date
    latest_date = max(file_dates.keys())
    latest_files = file_dates[latest_date]

    logger.info(f"Using latest available date: {latest_date} ({len(latest_files)} files)")

    # Step 7: Sort by timeframe
    return sorted(latest_files, key=lambda x: (
        'weekly' in x.name,
        'monthly' in x.name
    ))


def export_pvb_watchlist(
    config,
    user_config,
    csv_files: Optional[List[Path]] = None,
    date: Optional[str] = None
) -> Optional[List[Path]]:
    """
    Export PVB screener results to TradingView watchlist.

    Supports two modes:
    - Mode 1 (Integrated): Pass csv_files from current run
    - Mode 2 (Standalone): Pass date, will discover latest files

    Args:
        config: Configuration object
        user_config: User configuration object
        csv_files: Optional list of CSV files (Mode 1 - Integrated)
        date: Optional date string YYYYMMDD (Mode 2 - Standalone)

    Returns:
        List of created watchlist file paths, or None if failed
    """
    try:
        # Get output directory
        output_dir = config.directories.get('PVB_SCREENER_DIR', config.base_dir / 'results' / 'screeners' / 'pvbTW')

        # Initialize exporter
        exporter = TradingViewWatchlistExporter(output_dir=output_dir)

        # MODE 1: Use provided files (integrated with screener run)
        if csv_files:
            logger.info(f"Mode 1 (Integrated): Using {len(csv_files)} provided CSV files")
            files_to_process = csv_files

        # MODE 2: Discover files (standalone mode)
        else:
            logger.info("Mode 2 (Standalone): Discovering latest PVB files")
            files_to_process = find_latest_pvb_files(
                screener_dir=output_dir,
                ticker_choice=str(user_config.ticker_choice),
                date=date or config.current_date
            )

            if not files_to_process:
                logger.warning("No PVB screener files found for watchlist export")
                print("⚠️  No PVB files found - run screener first or check date")
                return None

            logger.info(f"Found {len(files_to_process)} PVB files to export")

        # Get user preferences
        include_buy = getattr(user_config, 'pvb_TWmodel_watchlist_include_buy', True)
        include_sell = getattr(user_config, 'pvb_TWmodel_watchlist_include_sell', True)

        # Generate output filename
        output_date = date or config.current_date
        output_filename = f'pvb_watchlist_{user_config.ticker_choice}_{output_date}.txt'

        # Export watchlist
        created_files = exporter.export_pvb_screener(
            csv_files=files_to_process,
            output_filename=output_filename,
            include_buy=include_buy,
            include_sell=include_sell
        )

        if created_files:
            print(f"\n📊 TradingView Watchlist Export Complete!")
            for file_path in created_files:
                print(f"   📁 {file_path.name}")
            print(f"   📍 Location: {output_dir}")

            if len(created_files) > 1:
                print(f"   ⚠️  Watchlist split into {len(created_files)} files (>1,000 symbols)")

        return created_files

    except Exception as e:
        logger.error(f"Error exporting watchlist: {e}")
        print(f"❌ Watchlist export failed: {e}")
        return None