"""
SCOOTER Screener (stSCOOTER + fastSCOOTER)
==========================================

Reads the basic_calc CSV (Layer 2 output) and filters tickers by their
pre-computed SCOOTER scores. No OHLCV data needed — pure CSV filter.

stSCOOTER  — StockCharts SCTR methodology, trend leaders (slow, 60/30/10 weights)
fastSCOOTER — Inverted weights (10/30/60) + shorter periods, early momentum
Combined    — Intersection: tickers passing both (high-conviction signal)
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Set

import pandas as pd

# Maps ticker_choice ID → boolean membership column in combined_info files
_UNIVERSE_ID_TO_COLUMN = {
    1: 'SP500',
    2: 'NASDAQ100',
    3: 'NASDAQComposite',
    4: 'Russell1000',
}

logger = logging.getLogger(__name__)


class ScooterScreener:
    """Filter tickers from a basic_calc DataFrame by SCOOTER score thresholds."""

    def __init__(self, user_config):
        self.user_config = user_config

    def run_stscooter(self, basic_calc_df: pd.DataFrame) -> pd.DataFrame:
        score_col = 'stscooter_score'
        if score_col not in basic_calc_df.columns:
            logger.warning("stscooter_score column not found in basic_calc — run BASIC phase first")
            return pd.DataFrame()

        min_score  = getattr(self.user_config, 'stscooter_screener_min_score',  70)
        min_price  = getattr(self.user_config, 'stscooter_screener_min_price',  5.0)
        min_volume = getattr(self.user_config, 'stscooter_screener_min_volume', 100000)
        t_top      = getattr(self.user_config, 'stscooter_screener_threshold_top',    90)
        t_strong   = getattr(self.user_config, 'stscooter_screener_threshold_strong', 70)

        mask = basic_calc_df[score_col] >= min_score
        if 'current_price' in basic_calc_df.columns:
            mask &= basic_calc_df['current_price'] >= min_price
        if 'daily_volume_avg_20d' in basic_calc_df.columns:
            mask &= basic_calc_df['daily_volume_avg_20d'] >= min_volume

        result = basic_calc_df[mask].copy()
        result['stscooter_tier'] = result[score_col].apply(
            lambda s: 'leader' if s >= t_top else ('strong' if s >= t_strong else 'average')
        )
        return result.sort_values(score_col, ascending=False).reset_index(drop=True)

    def run_fastscooter(self, basic_calc_df: pd.DataFrame) -> pd.DataFrame:
        score_col = 'fastscooter_score'
        if score_col not in basic_calc_df.columns:
            logger.warning("fastscooter_score column not found in basic_calc — run BASIC phase first")
            return pd.DataFrame()

        min_score  = getattr(self.user_config, 'fastscooter_screener_min_score',  65)
        min_price  = getattr(self.user_config, 'fastscooter_screener_min_price',  5.0)
        min_volume = getattr(self.user_config, 'fastscooter_screener_min_volume', 100000)
        t_top      = getattr(self.user_config, 'fastscooter_screener_threshold_top',    85)
        t_strong   = getattr(self.user_config, 'fastscooter_screener_threshold_strong', 65)

        mask = basic_calc_df[score_col] >= min_score
        if 'current_price' in basic_calc_df.columns:
            mask &= basic_calc_df['current_price'] >= min_price
        if 'daily_volume_avg_20d' in basic_calc_df.columns:
            mask &= basic_calc_df['daily_volume_avg_20d'] >= min_volume

        result = basic_calc_df[mask].copy()
        result['fastscooter_tier'] = result[score_col].apply(
            lambda s: 'leader' if s >= t_top else ('strong' if s >= t_strong else 'average')
        )
        return result.sort_values(score_col, ascending=False).reset_index(drop=True)

    def run_combined(
        self,
        st_result: pd.DataFrame,
        fast_result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Intersection of tickers passing both screeners."""
        if st_result.empty or fast_result.empty:
            return pd.DataFrame()
        ticker_col = 'ticker'
        if ticker_col not in st_result.columns or ticker_col not in fast_result.columns:
            return pd.DataFrame()
        common = set(st_result[ticker_col]) & set(fast_result[ticker_col])
        combined = st_result[st_result[ticker_col].isin(common)].copy()
        fast_tiers = fast_result[[ticker_col, 'fastscooter_tier']].set_index(ticker_col)
        combined = combined.join(fast_tiers, on=ticker_col, how='left')
        combined['combined_avg_score'] = (
            combined['stscooter_score'] + combined['fastscooter_score']
        ) / 2
        if 'stscooter_pct_rank' in combined.columns and 'fastscooter_pct_rank' in fast_result.columns:
            fast_pct = fast_result[[ticker_col, 'fastscooter_pct_rank']].set_index(ticker_col)
            combined = combined.join(fast_pct, on=ticker_col, how='left')
            combined['combined_avg_pct_rank'] = (
                combined['stscooter_pct_rank'] + combined['fastscooter_pct_rank']
            ) / 2
        return combined.sort_values('combined_avg_score', ascending=False).reset_index(drop=True)


def _load_universe_tickers(config, scooter_universe: str) -> Optional[Set[str]]:
    """
    Return the set of tickers for the given scooter_universe choice string.
    Returns None if scooter_universe is empty (rank across all tickers in basic_calc).

    Format: single id "2" or dash-combined "1-2" (same convention as ticker_choice).
    Resolution order:
      1. combined_tickers_{choice}.csv  — direct file match
      2. combined_info_tickers_*.csv   — filter by boolean membership columns
    """
    if not scooter_universe or not scooter_universe.strip():
        return None

    choice = scooter_universe.strip().replace(' ', '')
    tickers_dir = Path(config.directories['TICKERS_DIR'])

    # 1 — try exact combined_tickers file
    combined_file = tickers_dir / f"combined_tickers_{choice}.csv"
    if combined_file.exists():
        tickers = set(pd.read_csv(combined_file)['ticker'].dropna().astype(str))
        logger.info(f"SCOOTER universe '{choice}': {len(tickers)} tickers from {combined_file.name}")
        return tickers

    # 2 — fallback: load a combined_info file and filter by membership columns
    try:
        ids = [int(x.strip()) for x in choice.split('-')]
    except ValueError:
        logger.warning(f"Invalid scooter_universe format '{choice}' — using all tickers in basic_calc")
        return None

    # Find the most suitable combined_info file (prefer one that covers the requested ids)
    info_file = tickers_dir / f"combined_info_tickers_{choice}.csv"
    if not info_file.exists():
        # Pick the largest combined_info file available as best-effort
        candidates = sorted(tickers_dir.glob("combined_info_tickers_[0-9]*.csv"),
                            key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            logger.warning(f"No combined_info file found for scooter_universe='{choice}' — using all tickers")
            return None
        info_file = candidates[0]
        logger.info(f"SCOOTER universe '{choice}': using {info_file.name} for membership filter")

    needed_cols = ['ticker'] + [c for c in _UNIVERSE_ID_TO_COLUMN.values()]
    usecols = lambda c: c in needed_cols
    info_df = pd.read_csv(info_file, usecols=usecols)

    mask = pd.Series(False, index=info_df.index)
    for uid in ids:
        col = _UNIVERSE_ID_TO_COLUMN.get(uid)
        if col and col in info_df.columns:
            mask |= info_df[col].astype(bool)
        else:
            logger.warning(f"SCOOTER universe: no membership column for id {uid} — skipped")

    tickers = set(info_df.loc[mask, 'ticker'].dropna().astype(str))
    logger.info(f"SCOOTER universe '{choice}': {len(tickers)} tickers via membership columns")
    return tickers


def _add_percentile_ranks(df: pd.DataFrame,
                          universe_tickers: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Add stscooter_pct_rank / fastscooter_pct_rank columns (0–99.9).

    Rank is computed within universe_tickers if provided, otherwise across all rows.
    Non-universe tickers are retained in df but get NaN for pct_rank columns.
    """
    df = df.copy()
    ticker_col = 'ticker' if 'ticker' in df.columns else None

    if universe_tickers and ticker_col:
        rank_mask = df[ticker_col].isin(universe_tickers)
        n_universe = rank_mask.sum()
        n_total = len(df)
        if n_universe == 0:
            logger.warning("SCOOTER universe filter matched 0 tickers in basic_calc — check scooter_universe setting")
        else:
            logger.info(f"SCOOTER percentile rank: {n_universe} universe tickers out of {n_total} in basic_calc")
    else:
        rank_mask = pd.Series(True, index=df.index)

    for col, rank_col in [
        ('stscooter_score',   'stscooter_pct_rank'),
        ('fastscooter_score', 'fastscooter_pct_rank'),
    ]:
        if col in df.columns:
            df[rank_col] = float('nan')
            subset = df.loc[rank_mask, col]
            df.loc[rank_mask, rank_col] = (
                subset.rank(pct=True).mul(100).clip(upper=99.9).round(1)
            )
    return df


def _save_scooter_result(df: pd.DataFrame, output_dir: Path, filename: str) -> Optional[Path]:
    if df.empty:
        logger.info(f"No tickers passed {filename} — nothing to save")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    path = output_dir / f"{filename}_{date_str}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} tickers → {path}")
    return path


def run_scooter_screeners(config, user_config) -> Dict[str, Any]:
    """
    Entry point called from main.py / screeners_streaming.py.
    Reads the latest basic_calc daily CSV, runs both SCOOTER screeners,
    saves results, returns summary dict.
    """
    from src.basic_calculations import find_latest_basic_calculation_file

    results = {'stscooter': 0, 'fastscooter': 0, 'combined': 0}

    basic_calc_file = find_latest_basic_calculation_file(
        config, 'daily', user_config.ticker_choice
    )
    if not basic_calc_file or not basic_calc_file.exists():
        logger.warning("SCOOTER screeners: no basic_calc daily file found — run BASIC phase first")
        return results

    logger.info(f"SCOOTER screeners reading: {basic_calc_file}")
    basic_calc_df = pd.read_csv(basic_calc_file)

    scooter_universe = getattr(user_config, 'scooter_universe', '').strip()
    if not scooter_universe:
        scooter_universe = str(getattr(user_config, 'ticker_choice', '')).strip()
    universe_tickers = _load_universe_tickers(config, scooter_universe)
    basic_calc_df = _add_percentile_ranks(basic_calc_df, universe_tickers)

    screener = ScooterScreener(user_config)
    st_result   = pd.DataFrame()
    fast_result = pd.DataFrame()

    if getattr(user_config, 'stscooter_screener_enable', True):
        st_result = screener.run_stscooter(basic_calc_df)
        _save_scooter_result(
            st_result,
            config.directories['STSCOOTER_SCREENER_DIR'],
            'stscooter',
        )
        results['stscooter'] = len(st_result)
        logger.info(f"stSCOOTER: {len(st_result)} tickers passed")

    if getattr(user_config, 'fastscooter_screener_enable', True):
        fast_result = screener.run_fastscooter(basic_calc_df)
        _save_scooter_result(
            fast_result,
            config.directories['FASTSCOOTER_SCREENER_DIR'],
            'fastscooter',
        )
        results['fastscooter'] = len(fast_result)
        logger.info(f"fastSCOOTER: {len(fast_result)} tickers passed")

    if (
        getattr(user_config, 'scooter_combined_enable', True)
        and not st_result.empty
        and not fast_result.empty
    ):
        combined = screener.run_combined(st_result, fast_result)
        _save_scooter_result(
            combined,
            config.directories['SCOOTER_COMBINED_DIR'],
            'scooter_combined',
        )
        results['combined'] = len(combined)
        logger.info(f"SCOOTER combined: {len(combined)} tickers in both")

    return results
