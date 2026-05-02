"""Pure scoring math — no broker / DB / network dependencies.

Mirrors :func:`nest.engine.breadth._metrics_against_baseline` but
takes already-aggregated OHLC frames rather than per-minute wide
pivots, since the yfinance daily-bar path gets today's running
OHLC server-side from Yahoo (one ``yf.download(period='2d',
interval='1d')`` call returns it).

Both surfaces (Streamlit web app + nest API's local polling worker)
import this module to compute identical breadth rows.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


_TODAY_EXTREME_BAND_BPS = 5.0
_BPS_DENOM = 10_000.0


@dataclass(frozen=True)
class BreadthRow:
    """One breadth snapshot. Field names mirror the
    ``breadth_grid`` table columns (without the ``d_`` prefix —
    callers prepend that when persisting). The composite scores
    are the headline numbers for the heatmap."""
    bullish_bo_pct: float
    above_close_pct: float
    green_range_pct: float
    bearish_bo_pct: float
    below_close_pct: float
    red_range_pct: float
    today_high_count: int
    today_low_count: int
    score_bull: float
    score_bear: float
    universe_size: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_breadth_row(
    today_ohlc: pd.DataFrame,
    yesterday_ohlc: pd.DataFrame,
    *,
    band_bps: float = _TODAY_EXTREME_BAND_BPS,
) -> BreadthRow:
    """Compute one breadth row.

    ``today_ohlc`` and ``yesterday_ohlc`` are symbol-indexed DataFrames
    with columns ``open / high / low / close``. The two frames don't
    need to share the same row set — symbols missing from either side
    just contribute zero to that side's counts.

    ``band_bps`` is the tolerance for "stock at today's extreme" (5 bps
    = ~₹0.05 on a ₹100 stock). Matches the default in the production
    breadth engine.
    """
    if today_ohlc is None or today_ohlc.empty:
        return _empty_row(universe_size=0)

    # Align both frames on the union of symbols. Symbols that exist in
    # `today` but not `yesterday` get NaN baselines → comparisons return
    # False → contribute zero to baseline-driven counts.
    universe = today_ohlc.index
    universe_size = int(len(universe))

    today_close = today_ohlc["close"].astype(float)
    today_open = today_ohlc["open"].astype(float)

    if yesterday_ohlc is None or yesterday_ohlc.empty:
        bullish_bo = above_close = green_range = 0.0
        bearish_bo = below_close = red_range = 0.0
    else:
        base_high = yesterday_ohlc["high"].reindex(universe).astype(float)
        base_low = yesterday_ohlc["low"].reindex(universe).astype(float)
        base_close = yesterday_ohlc["close"].reindex(universe).astype(float)

        # Each metric: % of universe that satisfies the condition.
        bullish_bo = float(_pct(today_close > base_high, universe_size))
        above_close = float(_pct(today_close > base_close, universe_size))
        green_range = float(_pct(today_close > today_open, universe_size))
        bearish_bo = float(_pct(today_close < base_low, universe_size))
        below_close = float(_pct(today_close < base_close, universe_size))
        red_range = float(_pct(today_close < today_open, universe_size))

    # Today-extreme counts: how many stocks are within `band_bps` of
    # their own running high (resp. low) for the day. These don't need
    # a baseline — they're a today-only check.
    today_high = today_ohlc["high"].astype(float)
    today_low = today_ohlc["low"].astype(float)

    band = band_bps / _BPS_DENOM
    high_thresh = today_high * (1.0 - band)
    low_thresh = today_low * (1.0 + band)
    today_high_count = int((today_close >= high_thresh).sum())
    today_low_count = int((today_close <= low_thresh).sum())

    score_bull = (bullish_bo + above_close + green_range) / 3.0
    score_bear = (bearish_bo + below_close + red_range) / 3.0

    return BreadthRow(
        bullish_bo_pct=bullish_bo,
        above_close_pct=above_close,
        green_range_pct=green_range,
        bearish_bo_pct=bearish_bo,
        below_close_pct=below_close,
        red_range_pct=red_range,
        today_high_count=today_high_count,
        today_low_count=today_low_count,
        score_bull=score_bull,
        score_bear=score_bear,
        universe_size=universe_size,
    )


def _pct(mask: pd.Series, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(mask.fillna(False).sum()) / denom * 100.0


def _empty_row(universe_size: int) -> BreadthRow:
    return BreadthRow(
        bullish_bo_pct=0.0, above_close_pct=0.0, green_range_pct=0.0,
        bearish_bo_pct=0.0, below_close_pct=0.0, red_range_pct=0.0,
        today_high_count=0, today_low_count=0,
        score_bull=0.0, score_bear=0.0,
        universe_size=universe_size,
    )
