"""
Data Reader for Post-Processing Financial Market Data
==================================================

This module handles reading historical market data from local CSV files
and provides utilities for batch processing and data validation.

Based on the original data_reader.py from the old model with enhancements
for ticker info reading and better error handling.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple
import logging

logger = logging.getLogger(__name__)


class BatchDataSupplementer:
    """
    Loads YF batch files (prices_{interval}_YYYY-MM-DD.csv) and provides
    per-ticker rows to append to historical market_data DataFrames.

    Usage:
        sup = BatchDataSupplementer(batch_dir)
        sup.load(cutoff_date)                        # once, at DataReader init
        rows = sup.get_rows('AAPL', last_timestamp)  # per ticker in read_stock_data
    """

    def __init__(self, batch_dir: Path):
        self.batch_dir = Path(batch_dir)
        self._data: Dict[str, list] = {}  # {SYMBOL: [{date, Open, High, Low, Close, Volume}]}
        self._loaded = False

    def load(self, cutoff_date: pd.Timestamp) -> int:
        """Load all batch files with date > cutoff_date. Returns number of symbols loaded."""
        if not self.batch_dir.exists():
            logger.warning(f"YF batch dir not found: {self.batch_dir}")
            self._loaded = True
            return 0

        files_loaded = 0
        for f in sorted(self.batch_dir.glob('prices_*.csv')):
            # Extract date from filename: prices_1d_2026-06-12.csv → 2026-06-12
            stem_parts = f.stem.split('_')
            date_str = stem_parts[-1]  # last part is YYYY-MM-DD
            try:
                batch_ts = pd.Timestamp(date_str)
            except Exception:
                continue
            if batch_ts <= cutoff_date:
                continue
            try:
                df = pd.read_csv(f)
                required = {'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'}
                if not required.issubset(df.columns):
                    logger.warning(f"Batch file {f.name} missing columns, skipping")
                    continue
                for _, row in df.iterrows():
                    sym = str(row['Symbol']).upper()
                    self._data.setdefault(sym, []).append({
                        'date':   pd.Timestamp(row['Date']),
                        'Open':   float(row['Open'])   if pd.notna(row['Open'])   else None,
                        'High':   float(row['High'])   if pd.notna(row['High'])   else None,
                        'Low':    float(row['Low'])    if pd.notna(row['Low'])    else None,
                        'Close':  float(row['Close']),
                        'Volume': float(row['Volume']) if pd.notna(row['Volume']) else 0.0,
                    })
                files_loaded += 1
            except Exception as e:
                logger.warning(f"Could not load batch file {f.name}: {e}")

        for sym in self._data:
            self._data[sym].sort(key=lambda r: r['date'])

        self._loaded = True
        total_syms = len(self._data)
        if files_loaded:
            logger.info(
                f"BatchDataSupplementer: {files_loaded} file(s) loaded, {total_syms} symbols"
            )
        return total_syms

    def get_rows(self, ticker: str, after_ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Return OHLCV DataFrame for rows with date > after_ts, or None if none."""
        if not self._loaded or not self._data:
            return None
        sym = ticker.upper()
        rows = (
            self._data.get(sym)
            or self._data.get(sym.replace('.', '-'))  # BRK.B → BRK-B (yf format)
            or self._data.get(sym.replace('-', '.'))  # BRK-B → BRK.B (tv format)
        )
        if not rows:
            return None
        filtered = [r for r in rows if r['date'] > after_ts]
        if not filtered:
            return None
        df = pd.DataFrame(filtered).set_index('date')
        df.index.name = 'Date'
        return df


class TwSupplementer:
    """
    Loads normalized TW daily snapshots (tw_snapshot_YYYYMMDD.csv) and provides
    per-ticker rows to append to historical market_data DataFrames.

    Usage:
        sup = TwSupplementer(snapshot_dir)
        sup.load(cutoff_date)          # once, at DataReader init
        rows = sup.get_rows('AAPL', last_timestamp)  # per ticker in read_stock_data
    """

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = Path(snapshot_dir)
        self._data: Dict[str, list] = {}  # {SYMBOL: [{date, Open, High, Low, Close, Volume}]}
        self._loaded = False

    def load(self, cutoff_date: pd.Timestamp) -> int:
        """Load all snapshots with date > cutoff_date. Returns number of symbols loaded."""
        if not self.snapshot_dir.exists():
            logger.warning(f"TW snapshot dir not found: {self.snapshot_dir}")
            self._loaded = True
            return 0

        snapshots_loaded = 0
        for f in sorted(self.snapshot_dir.glob('tw_snapshot_????????.csv')):
            date_str = f.stem.split('_')[-1]  # YYYYMMDD
            try:
                snap_ts = pd.Timestamp(
                    f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
                )
            except Exception:
                continue
            if snap_ts <= cutoff_date:
                continue
            try:
                df = pd.read_csv(f)
                required = {'Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
                if not required.issubset(df.columns):
                    logger.warning(f"Snapshot {f.name} missing columns, skipping")
                    continue
                for _, row in df.iterrows():
                    sym = str(row['Symbol']).upper()
                    self._data.setdefault(sym, []).append({
                        'date':   pd.Timestamp(row['Date']),
                        'Open':   float(row['Open'])   if pd.notna(row['Open'])   else None,
                        'High':   float(row['High'])   if pd.notna(row['High'])   else None,
                        'Low':    float(row['Low'])    if pd.notna(row['Low'])    else None,
                        'Close':  float(row['Close']),
                        'Volume': float(row['Volume']) if pd.notna(row['Volume']) else 0.0,
                    })
                snapshots_loaded += 1
            except Exception as e:
                logger.warning(f"Could not load snapshot {f.name}: {e}")

        for sym in self._data:
            self._data[sym].sort(key=lambda r: r['date'])

        self._loaded = True
        total_syms = len(self._data)
        if snapshots_loaded:
            logger.info(
                f"TwSupplementer: {snapshots_loaded} snapshot(s) loaded, {total_syms} symbols"
            )
        return total_syms

    def get_rows(self, ticker: str, after_ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Return OHLCV DataFrame for rows with date > after_ts, or None if none."""
        if not self._loaded or not self._data:
            return None
        rows = self._data.get(ticker.upper())
        if rows is None:
            rows = self._data.get(ticker.replace('-', '.').upper(), [])
        filtered = [r for r in rows if r['date'] > after_ts]
        if not filtered:
            return None
        df = pd.DataFrame(filtered).set_index('date')
        df.index.name = 'Date'
        return df


class DataReader:
    """
    Reads and processes market data from local CSV files.
    
    Supports multiple timeframes (daily, weekly, monthly) and provides
    batch processing capabilities for large datasets.
    """
    
    def __init__(self, config, timeframe='daily', batch_size=100):
        """
        Initialize DataReader with configuration and timeframe.
        
        Args:
            config: Configuration object with directory paths
            timeframe: Data timeframe ('daily', 'weekly', 'monthly', 'intraday')
            batch_size: Number of tickers to process in each batch
        """
        self.config = config
        self.timeframe = timeframe
        self.batch_size = batch_size

        # Get market data directory for specified timeframe
        self.market_data_dir = config.get_market_data_dir(timeframe)
        self.tickers_dir = config.directories['TICKERS_DIR']

        # Initialize tickers list
        self.tickers = []
        self.ticker_info = None

        # YF batch supplement (daily / weekly / monthly)
        self.batch_supplementer: Optional[BatchDataSupplementer] = None
        self._init_batch_supplementer()

        # TW supplementation (daily only — tw snapshots are daily by nature)
        self.tw_supplementer: Optional[TwSupplementer] = None
        if timeframe == 'daily':
            self._init_tw_supplementer()

        logger.info(f"DataReader initialized for {timeframe} data from {self.market_data_dir}")

    def _init_tw_supplementer(self):
        """Initialize TwSupplementer if enabled in config."""
        try:
            from src.user_defined_data import read_user_data
            user_config = read_user_data()
            if not getattr(user_config, 'tw_supplement_enable', False):
                return
            snapshot_dir = self.config.directories.get('TW_SNAPSHOT_DIR')
            if not snapshot_dir:
                return
            cutoff = self._find_market_data_cutoff()
            if cutoff is None:
                logger.warning("TW supplement: could not determine market_data cutoff date")
                return
            self.tw_supplementer = TwSupplementer(snapshot_dir)
            n = self.tw_supplementer.load(cutoff)
            if n:
                logger.info(f"TW supplement active: {n} symbols available beyond {cutoff.date()}")
        except Exception as e:
            logger.warning(f"TW supplement init failed (non-fatal): {e}")

    def _init_batch_supplementer(self):
        """Initialize BatchDataSupplementer if enabled in config."""
        try:
            from src.user_defined_data import read_user_data
            user_config = read_user_data()
            if not getattr(user_config, 'yf_batch_supplement_enable', False):
                return
            batch_dir = self.config.get_batch_data_dir(self.timeframe)
            if not batch_dir or not batch_dir.exists():
                logger.warning(f"YF batch dir not found: {batch_dir}")
                return
            cutoff = self._find_market_data_cutoff()
            if cutoff is None:
                logger.warning("YF batch supplement: could not determine market_data cutoff date")
                return
            self.batch_supplementer = BatchDataSupplementer(batch_dir)
            n = self.batch_supplementer.load(cutoff)
            if n:
                logger.info(f"YF batch supplement active: {n} symbols available beyond {cutoff.date()}")
        except Exception as e:
            logger.warning(f"YF batch supplement init failed (non-fatal): {e}")

    def _find_market_data_cutoff(self) -> Optional[pd.Timestamp]:
        """Find the last date present in market_data by reading a reference ticker."""
        for ticker in ['SPY', 'AAPL', 'QQQ', 'MSFT']:
            f = self.market_data_dir / f'{ticker}.csv'
            if not f.exists():
                continue
            try:
                df = pd.read_csv(f, usecols=[0], header=0)
                df.columns = ['date_raw']
                dates = pd.to_datetime(
                    df['date_raw'].astype(str).str.split(' ').str[0], errors='coerce'
                ).dropna()
                if not dates.empty:
                    return dates.max().tz_localize(None) if dates.max().tzinfo else dates.max()
            except Exception as e:
                logger.debug(f"Could not read cutoff from {ticker}: {e}")
        return None
    
    def load_tickers_from_file(self, combined_ticker_file: str) -> List[str]:
        """
        Load tickers from combined ticker file.
        
        Args:
            combined_ticker_file: Path to combined ticker CSV file
            
        Returns:
            List of ticker symbols
        """
        try:
            df = pd.read_csv(combined_ticker_file)
            
            # Handle different possible column names (prioritize 'ticker' as standardized column)
            ticker_column = None
            for col in ['ticker', 'symbol', 'Ticker', 'Symbol']:
                if col in df.columns:
                    ticker_column = col
                    break
            
            if ticker_column is None:
                raise ValueError(f"No ticker column found in {combined_ticker_file}")
            
            self.tickers = df[ticker_column].dropna().unique().tolist()
            logger.info(f"Loaded {len(self.tickers)} tickers from {combined_ticker_file}")
            
            return self.tickers
            
        except Exception as e:
            logger.error(f"Error loading tickers from {combined_ticker_file}: {e}")
            raise
    
    def load_ticker_info(self, ticker_info_file: Optional[str] = None, user_choice: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load additional ticker information if available.
        
        Args:
            ticker_info_file: Path to ticker info file (optional)
            user_choice: User ticker choice to find choice-specific files (optional)
            
        Returns:
            DataFrame with ticker info or None if not available
        """
        if not ticker_info_file:
            # Try to find choice-specific ticker info files first
            possible_files = []
            
            # Add choice-specific files if user_choice provided
            if user_choice is not None:
                possible_files.extend([
                    f'combined_info_tickers_clean_{user_choice}.csv',
                    f'combined_info_tickers_{user_choice}.csv'
                ])
            
            # Add generic fallback files
            possible_files.extend([
                'combined_info_tickers_clean_0.csv',  # Universe file fallback
                'combined_info_tickers_0.csv',
                'tradingview_universe_bool.csv',      # Boolean universe file
                'tradingview_universe_info.csv'
            ])
            
            for filename in possible_files:
                file_path = self.tickers_dir / filename
                if file_path.exists():
                    ticker_info_file = str(file_path)
                    logger.info(f"Found ticker info file: {filename}")
                    break
        
        if ticker_info_file and Path(ticker_info_file).exists():
            try:
                self.ticker_info = pd.read_csv(ticker_info_file)
                logger.info(f"Loaded ticker info from {ticker_info_file}")
                return self.ticker_info
            except Exception as e:
                logger.warning(f"Could not load ticker info from {ticker_info_file}: {e}")
        
        logger.info("No ticker info file available")
        return None
    
    def read_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Read historical data for a single ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame with OHLCV data or None if file not found
        """
        file_path = self.market_data_dir / f"{ticker}.csv"
        
        if not file_path.exists():
            logger.debug(f"Data file not found for {ticker}: {file_path}")
            return None
        
        try:
            # Use the exact same approach as the working marketScanners_v1 version
            df = pd.read_csv(file_path, index_col='Date', parse_dates=False)
            df.index = df.index.str.split(' ').str[0]
            df.index = pd.to_datetime(df.index)
            
            # Ensure timezone-naive datetime index for consistency across all calculations
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Remove rows with invalid dates (just in case)
            df = df[df.index.notna()]
            
            # Sort by date
            df = df.sort_index()

            # Filter to business days only (exclude weekends)
            # This removes Saturday (5) and Sunday (6) data points
            df = df[df.index.weekday < 5]

            # Return standard OHLCV columns
            standard_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col for col in standard_columns if col in df.columns]

            if not available_columns:
                logger.warning(f"No standard OHLCV columns found for {ticker}")
                return df  # Return as-is

            df = df[available_columns]

            # Append YF batch rows newer than the last historical date
            if self.batch_supplementer is not None and not df.empty:
                batch_rows = self.batch_supplementer.get_rows(ticker, df.index[-1])
                if batch_rows is not None:
                    batch_cols = [c for c in available_columns if c in batch_rows.columns]
                    df = pd.concat([df, batch_rows[batch_cols]])

            # Append TW snapshot rows newer than the last historical date (or batch date)
            if self.tw_supplementer is not None and not df.empty:
                tw_rows = self.tw_supplementer.get_rows(ticker, df.index[-1])
                if tw_rows is not None:
                    tw_cols = [c for c in available_columns if c in tw_rows.columns]
                    df = pd.concat([df, tw_rows[tw_cols]])

            return df
            
        except Exception as e:
            logger.error(f"Error reading data for {ticker}: {e}")
            return None
    
    def validate_stock_data(self, ticker: str, df: pd.DataFrame, 
                          min_data_points: Optional[int] = None) -> Tuple[bool, str]:
        """
        Validate stock data quality.
        
        Args:
            ticker: Ticker symbol
            df: DataFrame with stock data
            min_data_points: Minimum number of data points required (auto-calculated if None)
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if df is None or df.empty:
            return False, "No data available"
        
        # Set minimum data points based on timeframe if not provided
        if min_data_points is None:
            timeframe_minimums = {
                'daily': 252,    # 1 year of trading days
                'weekly': 52,    # 1 year of weeks  
                'monthly': 12,   # 1 year of months
                'intraday': 100  # Minimum for intraday
            }
            min_data_points = timeframe_minimums.get(self.timeframe, 50)  # Default 50
        
        # Check for minimum data points
        if len(df) < min_data_points:
            return False, f"Insufficient data: {len(df)} < {min_data_points} ({self.timeframe})"
        
        # Check for required columns
        if 'Close' not in df.columns:
            return False, "Missing Close price column"
        
        # Check for excessive missing values
        if df['Close'].isnull().sum() > len(df) * 0.1:  # More than 10% missing
            return False, f"Too many missing Close prices: {df['Close'].isnull().sum()}"
        
        # Check for unrealistic price values
        close_prices = df['Close'].dropna()
        if (close_prices <= 0).any():
            return False, "Invalid price values (≤ 0)"
        
        # Check for extreme price movements (potential data errors)
        price_changes = close_prices.pct_change().dropna()
        extreme_moves = (abs(price_changes) > 0.5).sum()  # More than 50% change
        if extreme_moves > len(price_changes) * 0.01:  # More than 1% of days
            return False, f"Too many extreme price movements: {extreme_moves}"
        
        return True, "Valid data"
    
    def get_batches(self, tickers: Optional[List[str]] = None) -> Generator[List[str], None, None]:
        """
        Generate batches of tickers for processing.
        
        Args:
            tickers: List of tickers (uses self.tickers if None)
            
        Yields:
            Lists of ticker symbols for batch processing
        """
        if tickers is None:
            tickers = self.tickers
            
        for i in range(0, len(tickers), self.batch_size):
            yield tickers[i:i + self.batch_size]
    
    def read_batch_data(self, ticker_batch: List[str], 
                       validate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Read data for a batch of tickers.
        
        Args:
            ticker_batch: List of ticker symbols
            validate: Whether to validate data quality
            
        Returns:
            Dictionary mapping tickers to their DataFrames
        """
        batch_data = {}
        
        for ticker in ticker_batch:
            df = self.read_stock_data(ticker)
            
            if df is not None:
                if validate:
                    is_valid, reason = self.validate_stock_data(ticker, df)
                    if is_valid:
                        batch_data[ticker] = df
                    else:
                        logger.debug(f"Skipping {ticker}: {reason}")
                else:
                    batch_data[ticker] = df
            
        logger.info(f"Successfully read {len(batch_data)}/{len(ticker_batch)} tickers from batch")
        return batch_data
    
    def create_combined_dataframe(self, column='Close', 
                                exclude_patterns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a combined DataFrame with specified column for all tickers.
        
        Args:
            column: Column to extract ('Close', 'Volume', etc.)
            exclude_patterns: List of ticker patterns to exclude
            
        Returns:
            DataFrame with tickers as columns and dates as index
        """
        combined_df = pd.DataFrame()
        
        # Default exclusions for problematic index tickers
        if exclude_patterns is None:
            exclude_patterns = ['^BUK100P', '^FTSE', '^GDAXI', '^FCHI', 
                              '^STOXX50E', '^N100', '^BFX', '^HSI', '^STI']
        
        processed_count = 0
        skipped_count = 0
        
        for ticker in self.tickers:
            # Check exclusion patterns
            if any(pattern in ticker for pattern in exclude_patterns):
                logger.debug(f"Excluding ticker {ticker} due to exclusion pattern")
                skipped_count += 1
                continue
            
            df = self.read_stock_data(ticker)
            
            if df is not None and column in df.columns:
                is_valid, reason = self.validate_stock_data(ticker, df)
                
                if is_valid:
                    # Rename column to ticker name
                    ticker_series = df[column].rename(ticker)
                    combined_df = pd.concat([combined_df, ticker_series], axis=1)
                    processed_count += 1
                else:
                    logger.debug(f"Skipping {ticker}: {reason}")
                    skipped_count += 1
            else:
                skipped_count += 1
        
        logger.info(f"Combined DataFrame created: {processed_count} tickers, "
                   f"{skipped_count} skipped, shape: {combined_df.shape}")
        
        return combined_df
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of available data.
        
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'timeframe': self.timeframe,
            'total_tickers': len(self.tickers),
            'available_files': 0,
            'valid_files': 0,
            'date_range': {'start': None, 'end': None},
            'avg_data_points': 0
        }
        
        # Count available files
        if self.market_data_dir.exists():
            csv_files = list(self.market_data_dir.glob('*.csv'))
            summary['available_files'] = len(csv_files)
        
        # Sample some files to get date range and validation info
        sample_size = min(10, len(self.tickers))
        if sample_size > 0:
            valid_count = 0
            total_points = 0
            min_date = None
            max_date = None
            
            for ticker in self.tickers[:sample_size]:
                df = self.read_stock_data(ticker)
                if df is not None and not df.empty:
                    is_valid, _ = self.validate_stock_data(ticker, df)
                    if is_valid:
                        valid_count += 1
                        total_points += len(df)
                        
                        # Update date range
                        if min_date is None or df.index.min() < min_date:
                            min_date = df.index.min()
                        if max_date is None or df.index.max() > max_date:
                            max_date = df.index.max()
            
            summary['valid_files'] = valid_count
            if valid_count > 0:
                summary['avg_data_points'] = total_points // valid_count
                summary['date_range']['start'] = min_date.strftime('%Y-%m-%d') if min_date else None
                summary['date_range']['end'] = max_date.strftime('%Y-%m-%d') if max_date else None
        
        return summary
    
    def __str__(self) -> str:
        """String representation of DataReader."""
        return (f"DataReader(timeframe={self.timeframe}, "
                f"tickers={len(self.tickers)}, "
                f"batch_size={self.batch_size})")