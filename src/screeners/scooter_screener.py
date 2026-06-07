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
from typing import Dict, Any, Optional

import pandas as pd

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
        fast_scores = fast_result[[ticker_col, 'fastscooter_score', 'fastscooter_tier']].set_index(ticker_col)
        combined = combined.join(fast_scores, on=ticker_col, how='left')
        combined['combined_avg_score'] = (
            combined['stscooter_score'] + combined['fastscooter_score']
        ) / 2
        return combined.sort_values('combined_avg_score', ascending=False).reset_index(drop=True)


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
