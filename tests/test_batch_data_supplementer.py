"""
Tests for BatchDataSupplementer and its integration in DataReader.read_stock_data().

Covers:
- BatchDataSupplementer.load() — file parsing, cutoff filtering, symbol indexing
- BatchDataSupplementer.get_rows() — date filtering, ticker normalization
- DataReader.read_stock_data() — batch rows stitched after historical data
- Edge cases: missing dir, bad filenames, missing columns, unknown ticker
"""

import io
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_reader import BatchDataSupplementer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_batch_file(tmp_path: Path, date_str: str, rows: list[dict]) -> Path:
    """Write a prices_1d_{date_str}.csv into tmp_path and return the path."""
    f = tmp_path / f"prices_1d_{date_str}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    return f


# ---------------------------------------------------------------------------
# BatchDataSupplementer — unit tests
# ---------------------------------------------------------------------------

class TestBatchDataSupplementerLoad:

    def test_loads_files_after_cutoff(self, tmp_path):
        _write_batch_file(tmp_path, "2026-06-10", [
            {"Date": "2026-06-10", "Symbol": "AAPL", "Open": 190.0, "High": 192.0,
             "Low": 189.0, "Close": 191.0, "Adj Close": 191.0, "Volume": 50000.0},
            {"Date": "2026-06-10", "Symbol": "MSFT", "Open": 400.0, "High": 405.0,
             "Low": 399.0, "Close": 403.0, "Adj Close": 403.0, "Volume": 30000.0},
        ])
        _write_batch_file(tmp_path, "2026-06-11", [
            {"Date": "2026-06-11", "Symbol": "AAPL", "Open": 191.5, "High": 193.0,
             "Low": 190.0, "Close": 192.5, "Adj Close": 192.5, "Volume": 45000.0},
        ])

        sup = BatchDataSupplementer(tmp_path)
        cutoff = pd.Timestamp("2026-06-09")
        n = sup.load(cutoff)

        assert n == 2  # AAPL and MSFT
        assert len(sup._data["AAPL"]) == 2
        assert len(sup._data["MSFT"]) == 1

    def test_skips_files_on_or_before_cutoff(self, tmp_path):
        _write_batch_file(tmp_path, "2026-06-08", [
            {"Date": "2026-06-08", "Symbol": "AAPL", "Open": 185.0, "High": 186.0,
             "Low": 184.0, "Close": 185.5, "Adj Close": 185.5, "Volume": 40000.0},
        ])
        _write_batch_file(tmp_path, "2026-06-09", [
            {"Date": "2026-06-09", "Symbol": "AAPL", "Open": 186.0, "High": 187.0,
             "Low": 185.0, "Close": 186.5, "Adj Close": 186.5, "Volume": 42000.0},
        ])

        sup = BatchDataSupplementer(tmp_path)
        n = sup.load(pd.Timestamp("2026-06-09"))  # exact cutoff date excluded

        assert n == 0
        assert sup._data == {}

    def test_returns_zero_for_missing_directory(self, tmp_path):
        sup = BatchDataSupplementer(tmp_path / "nonexistent")
        n = sup.load(pd.Timestamp("2026-06-01"))
        assert n == 0

    def test_skips_file_with_bad_name(self, tmp_path):
        (tmp_path / "bad_name.csv").write_text("Date,Symbol,Open,High,Low,Close,Volume\n")
        sup = BatchDataSupplementer(tmp_path)
        n = sup.load(pd.Timestamp("2026-06-01"))
        assert n == 0

    def test_skips_file_missing_required_columns(self, tmp_path):
        # Missing 'Volume'
        f = tmp_path / "prices_1d_2026-06-10.csv"
        f.write_text("Date,Symbol,Open,High,Low,Close\n2026-06-10,AAPL,190,192,189,191\n")
        sup = BatchDataSupplementer(tmp_path)
        n = sup.load(pd.Timestamp("2026-06-09"))
        assert n == 0

    def test_rows_sorted_by_date(self, tmp_path):
        # Write files in reverse order; data should still be sorted
        _write_batch_file(tmp_path, "2026-06-12", [
            {"Date": "2026-06-12", "Symbol": "AAPL", "Open": 193.0, "High": 194.0,
             "Low": 192.0, "Close": 193.5, "Adj Close": 193.5, "Volume": 48000.0},
        ])
        _write_batch_file(tmp_path, "2026-06-10", [
            {"Date": "2026-06-10", "Symbol": "AAPL", "Open": 190.0, "High": 192.0,
             "Low": 189.0, "Close": 191.0, "Adj Close": 191.0, "Volume": 50000.0},
        ])

        sup = BatchDataSupplementer(tmp_path)
        sup.load(pd.Timestamp("2026-06-09"))

        dates = [r["date"] for r in sup._data["AAPL"]]
        assert dates == sorted(dates)


class TestBatchDataSupplementerGetRows:

    def _loaded_sup(self, tmp_path) -> BatchDataSupplementer:
        _write_batch_file(tmp_path, "2026-06-10", [
            {"Date": "2026-06-10", "Symbol": "AAPL", "Open": 190.0, "High": 192.0,
             "Low": 189.0, "Close": 191.0, "Adj Close": 191.0, "Volume": 50000.0},
            {"Date": "2026-06-10", "Symbol": "BRK-B", "Open": 480.0, "High": 482.0,
             "Low": 479.0, "Close": 481.0, "Adj Close": 481.0, "Volume": 5000.0},
        ])
        _write_batch_file(tmp_path, "2026-06-11", [
            {"Date": "2026-06-11", "Symbol": "AAPL", "Open": 191.5, "High": 193.0,
             "Low": 190.0, "Close": 192.5, "Adj Close": 192.5, "Volume": 45000.0},
        ])
        sup = BatchDataSupplementer(tmp_path)
        sup.load(pd.Timestamp("2026-06-09"))
        return sup

    def test_returns_rows_after_cutoff(self, tmp_path):
        sup = self._loaded_sup(tmp_path)
        rows = sup.get_rows("AAPL", pd.Timestamp("2026-06-10"))
        assert rows is not None
        assert len(rows) == 1
        assert rows.index[0] == pd.Timestamp("2026-06-11")

    def test_returns_all_rows_when_cutoff_before_data(self, tmp_path):
        sup = self._loaded_sup(tmp_path)
        rows = sup.get_rows("AAPL", pd.Timestamp("2026-06-09"))
        assert rows is not None
        assert len(rows) == 2

    def test_returns_none_for_unknown_ticker(self, tmp_path):
        sup = self._loaded_sup(tmp_path)
        assert sup.get_rows("ZZZZ", pd.Timestamp("2026-06-09")) is None

    def test_returns_none_when_all_rows_filtered(self, tmp_path):
        sup = self._loaded_sup(tmp_path)
        # Cutoff after all batch dates
        assert sup.get_rows("AAPL", pd.Timestamp("2026-06-12")) is None

    def test_ticker_normalization_dash_dot(self, tmp_path):
        # BRK-B stored in batch as BRK-B; look it up as BRK.B (dash→dot fallback)
        sup = self._loaded_sup(tmp_path)
        rows = sup.get_rows("BRK.B", pd.Timestamp("2026-06-09"))
        assert rows is not None
        assert len(rows) == 1

    def test_case_insensitive_ticker(self, tmp_path):
        sup = self._loaded_sup(tmp_path)
        rows = sup.get_rows("aapl", pd.Timestamp("2026-06-09"))
        assert rows is not None

    def test_returned_dataframe_has_correct_columns(self, tmp_path):
        sup = self._loaded_sup(tmp_path)
        rows = sup.get_rows("AAPL", pd.Timestamp("2026-06-09"))
        assert rows is not None
        # Adj Close is NOT stored; only OHLCV
        assert set(rows.columns) == {"Open", "High", "Low", "Close", "Volume"}
        assert rows.index.name == "Date"

    def test_returned_dataframe_index_is_timestamp(self, tmp_path):
        sup = self._loaded_sup(tmp_path)
        rows = sup.get_rows("AAPL", pd.Timestamp("2026-06-09"))
        assert rows is not None
        assert isinstance(rows.index[0], pd.Timestamp)


# ---------------------------------------------------------------------------
# Integration: DataReader.read_stock_data() stitches batch rows
# ---------------------------------------------------------------------------

class TestDataReaderBatchStitch:
    """
    Use a minimal mock Config and a real temp dir with:
      - one per-ticker historical CSV  (AAPL.csv, data up to 2026-06-09)
      - two batch files                (2026-06-10, 2026-06-11)
    Verify that read_stock_data() returns the combined data.
    """

    HISTORICAL_CSV = textwrap.dedent("""\
        Date,Open,High,Low,Close,Volume
        2026-06-05 00:00:00-04:00,308.0,312.0,307.0,310.0,60000000.0
        2026-06-08 00:00:00-04:00,310.5,314.0,309.0,313.0,55000000.0
        2026-06-09 00:00:00-04:00,312.0,316.0,311.0,315.0,50000000.0
    """)

    def _setup(self, tmp_path):
        # Historical per-ticker directory
        hist_dir = tmp_path / "daily"
        hist_dir.mkdir()
        (hist_dir / "AAPL.csv").write_text(self.HISTORICAL_CSV)

        # Batch directory
        batch_dir = tmp_path / "batch_daily"
        batch_dir.mkdir()
        _write_batch_file(batch_dir, "2026-06-10", [
            {"Date": "2026-06-10", "Symbol": "AAPL", "Open": 316.0, "High": 318.0,
             "Low": 314.0, "Close": 317.0, "Adj Close": 317.0, "Volume": 48000000.0},
        ])
        _write_batch_file(batch_dir, "2026-06-11", [
            {"Date": "2026-06-11", "Symbol": "AAPL", "Open": 317.5, "High": 319.0,
             "Low": 316.0, "Close": 318.5, "Adj Close": 318.5, "Volume": 46000000.0},
        ])

        # Mock config
        config = MagicMock()
        config.get_market_data_dir.return_value = hist_dir
        config.get_batch_data_dir.return_value = batch_dir
        config.directories = {"TICKERS_DIR": tmp_path}

        return config, batch_dir

    def test_batch_rows_appended_after_historical(self, tmp_path):
        config, batch_dir = self._setup(tmp_path)

        from src.data_reader import DataReader

        # Patch user_defined_data so batch supplement is enabled
        with patch("src.data_reader.DataReader._init_tw_supplementer"):
            with patch("src.user_defined_data.read_user_data") as mock_ud:
                ud = MagicMock()
                ud.yf_batch_supplement_enable = True
                mock_ud.return_value = ud

                dr = DataReader(config, timeframe="daily")
                # Manually set the supplementer (init already ran via __init__)
                # Re-run init to ensure it picks up the mock
                dr.batch_supplementer = BatchDataSupplementer(batch_dir)
                dr.batch_supplementer.load(pd.Timestamp("2026-06-09"))

        df = dr.read_stock_data("AAPL")

        assert df is not None
        assert len(df) == 5  # 3 historical + 2 batch
        assert df.index[-1] == pd.Timestamp("2026-06-11")
        assert df.index[-2] == pd.Timestamp("2026-06-10")
        assert df.loc[pd.Timestamp("2026-06-10"), "Close"] == pytest.approx(317.0)

    def test_no_duplicate_dates(self, tmp_path):
        config, batch_dir = self._setup(tmp_path)

        from src.data_reader import DataReader

        with patch("src.data_reader.DataReader._init_tw_supplementer"):
            with patch("src.user_defined_data.read_user_data") as mock_ud:
                ud = MagicMock()
                ud.yf_batch_supplement_enable = True
                mock_ud.return_value = ud

                dr = DataReader(config, timeframe="daily")
                dr.batch_supplementer = BatchDataSupplementer(batch_dir)
                dr.batch_supplementer.load(pd.Timestamp("2026-06-09"))

        df = dr.read_stock_data("AAPL")
        assert df is not None
        assert df.index.duplicated().sum() == 0

    def test_returns_none_for_missing_ticker_file(self, tmp_path):
        config, _ = self._setup(tmp_path)

        from src.data_reader import DataReader

        with patch("src.data_reader.DataReader._init_tw_supplementer"):
            with patch("src.user_defined_data.read_user_data") as mock_ud:
                ud = MagicMock()
                ud.yf_batch_supplement_enable = False
                mock_ud.return_value = ud

                dr = DataReader(config, timeframe="daily")

        assert dr.read_stock_data("ZZZZ") is None
