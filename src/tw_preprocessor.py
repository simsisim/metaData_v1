"""
TW Preprocessor
===============
Converts raw TradingView export files from native/ into normalized daily
snapshots written to tw_files/daily/.

Native files:  all_stocks _LOHP_YYYY-MM-DD.csv
               all_etfs _LOHP_YYYY-MM-DD.csv
Output files:  tw_snapshot_YYYYMMDD.csv
               Columns: Symbol, Date, Open, High, Low, Close, Volume
"""

import re
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_COLUMN_MAP = {
    'Open 1 day':   'Open',
    'High 1 day':   'High',
    'Low 1 day':    'Low',
    'Price':        'Close',
    'Volume 1 day': 'Volume',
}

_DATE_PATTERN = re.compile(r'_LOHP_(\d{4}-\d{2}-\d{2})\.csv$', re.IGNORECASE)


class TwPreprocessor:
    """
    Scans native/ for TradingView export files, combines stocks + ETFs per date,
    normalizes column names, and writes tw_snapshot_YYYYMMDD.csv to snapshot_dir.
    Already-processed dates are skipped (idempotent).
    """

    def __init__(self, native_dir: Path, snapshot_dir: Path):
        self.native_dir = Path(native_dir)
        self.snapshot_dir = Path(snapshot_dir)

    def run(self) -> int:
        """Process all unprocessed native files. Returns number of new snapshots created."""
        if not self.native_dir.exists():
            logger.warning(f"TW native dir not found: {self.native_dir}")
            return 0

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Group native files by extracted date string
        files_by_date: dict = {}
        for f in self.native_dir.iterdir():
            m = _DATE_PATTERN.search(f.name)
            if m:
                date_str = m.group(1)  # YYYY-MM-DD
                files_by_date.setdefault(date_str, []).append(f)

        created = 0
        for date_str, files in sorted(files_by_date.items()):
            compact = date_str.replace('-', '')  # YYYYMMDD
            out_path = self.snapshot_dir / f'tw_snapshot_{compact}.csv'
            if out_path.exists():
                continue
            df = self._combine(files, date_str)
            if df is not None and not df.empty:
                df.to_csv(out_path, index=False)
                logger.info(f"Created {out_path.name} ({len(df)} symbols)")
                created += 1

        if created:
            print(f"📸 TW Preprocessor: {created} new snapshot(s) written to {self.snapshot_dir}")
        else:
            logger.debug("TW Preprocessor: no new native files to process")
        return created

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _combine(self, files, date_str: str) -> Optional[pd.DataFrame]:
        frames = []
        for f in files:
            try:
                raw = pd.read_csv(f)
                df = self._normalize(raw, date_str)
                if df is not None:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Could not read {f.name}: {e}")
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset='Symbol', keep='first')
        return combined

    def _normalize(self, raw: pd.DataFrame, date_str: str) -> Optional[pd.DataFrame]:
        if 'Symbol' not in raw.columns:
            logger.warning("No Symbol column — skipping file")
            return None
        missing = [c for c in _COLUMN_MAP if c not in raw.columns]
        if missing:
            logger.warning(f"Missing columns {missing} — skipping file")
            return None
        df = raw[['Symbol'] + list(_COLUMN_MAP.keys())].copy()
        df = df.rename(columns=_COLUMN_MAP)
        df.insert(1, 'Date', date_str)
        df = df.dropna(subset=['Symbol', 'Close'])
        df = df[df['Close'] > 0]
        return df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
