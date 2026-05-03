"""Historical breadth replay — pick a date, compute per-5-min breadth.

Mirrors the WPF "Intraday Breadth" page's date-picker workflow inside
the Streamlit app. Reuses the same scoring math
(``compute_breadth_row``) so columns + values are directly comparable
to the live Daily / Weekly tabs.

Constraint: Yahoo only retains 5-minute bars for ~60 days. Older dates
return an empty frame with a clear ``debug["error"]`` reason.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf

from breadth_compute import BreadthRow, compute_breadth_row
from universe import UniverseEntry


_IST = "Asia/Kolkata"


def fetch_historical_breadth(
    target_date: date,
    universe: List[UniverseEntry],
) -> Tuple[pd.DataFrame, List[str], dict, Optional[BreadthRow]]:
    """Pull 5-minute intraday OHLC for ``target_date`` plus the prior
    daily close as baseline; replay per-5-min breadth for the day.

    Returns ``(grid_df, missing_symbols, debug, last_row)``. Empty grid
    means no usable data — `debug["error"]` carries the reason
    (holiday, too-old date, Yahoo returned empty, etc.). ``last_row``
    is the BreadthRow at end of session, used by the UI to render
    its sentiment-cards block; ``None`` when the grid is empty.
    """
    tickers_list = [u.yfinance_ticker for u in universe]
    sym_by_yt = {u.yfinance_ticker: u.symbol for u in universe}
    tickers = " ".join(tickers_list)

    debug: dict = {
        "tickers_requested": len(tickers_list),
        "target_date": str(target_date),
    }

    # Two-day intraday window: target day + 1 day buffer for safety
    # against TZ edge cases at the start/end of session.
    intraday_df = yf.download(
        tickers=tickers,
        start=target_date - timedelta(days=2),
        end=target_date + timedelta(days=1),
        interval="5m",
        progress=False,
        threads=False,
        auto_adjust=False,
    )
    # 10-day daily window for the prior-close baseline (handles long
    # weekends without missing the last trading day before target).
    daily_df = yf.download(
        tickers=tickers,
        start=target_date - timedelta(days=10),
        end=target_date + timedelta(days=1),
        interval="1d",
        progress=False,
        threads=False,
        auto_adjust=False,
    )

    if intraday_df is None or intraday_df.empty:
        return pd.DataFrame(), tickers_list, {
            **debug, "error": "Yahoo returned no 5-min bars for this date "
                              "(likely holiday/weekend or older than ~60-day retention)",
        }, None
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(), tickers_list, {
            **debug, "error": "Yahoo returned no daily bars for the baseline window",
        }, None

    # Convert intraday index to IST and slice to target_date only.
    idx = intraday_df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        intraday_df = intraday_df.tz_convert(_IST)
    intraday_day = intraday_df[intraday_df.index.date == target_date]
    if intraday_day.empty:
        return pd.DataFrame(), tickers_list, {
            **debug, "error": f"no 5-min data for {target_date} after IST slice "
                              f"(holiday or weekend)",
        }, None

    # Pick yesterday's daily row = latest daily index strictly before target.
    # Keep both forms: the date for display, the Timestamp for `.loc` lookup
    # (yfinance returns DatetimeIndex of midnight timestamps, not raw dates).
    daily_idx = pd.to_datetime(daily_df.index)
    earlier_mask = daily_idx.date < target_date
    if not earlier_mask.any():
        return pd.DataFrame(), tickers_list, {
            **debug, "error": "no prior trading day in baseline window",
        }, None
    earlier_ts = daily_idx[earlier_mask]
    yesterday_ts = earlier_ts.max()                          # actual index value
    yesterday_pos = pd.Timestamp(yesterday_ts).date()         # for display only
    debug["yesterday_baseline"] = str(yesterday_pos)

    is_multi = intraday_day.columns.nlevels > 1
    if not is_multi:
        # Single-ticker quirk — practically impossible for our 190+ universe
        # but guard anyway.
        return pd.DataFrame(), tickers_list, {
            **debug, "error": "expected multi-index intraday columns; got single-ticker shape",
        }, None

    # Wide pivot views per field — buckets × tickers.
    opens  = intraday_day["Open"]
    highs  = intraday_day["High"]
    lows   = intraday_day["Low"]
    closes = intraday_day["Close"]

    # Running OHLC needed by `compute_breadth_row`:
    # - today's open: first non-NaN open of the day per ticker.
    # - running high: cummax of highs.
    # - running low:  cummin of lows.
    # - running close: ffilled close.
    today_open_per_ticker = opens.bfill().iloc[0]
    cum_high  = highs.cummax()
    cum_low   = lows.cummin()
    cum_close = closes.ffill()

    # Yesterday baseline — pulled from the daily frame at yesterday_ts.
    daily_multi = daily_df.columns.nlevels > 1
    if not daily_multi:
        return pd.DataFrame(), tickers_list, {
            **debug, "error": "single-ticker daily shape unsupported",
        }, None
    try:
        y_open  = daily_df["Open"].loc[yesterday_ts]
        y_high  = daily_df["High"].loc[yesterday_ts]
        y_low   = daily_df["Low"].loc[yesterday_ts]
        y_close = daily_df["Close"].loc[yesterday_ts]
    except KeyError:
        # Last-ditch fallback: align by date if the precise Timestamp lookup
        # missed (some yfinance edge cases hand back tz-aware indexes).
        try:
            row_mask = pd.to_datetime(daily_df.index).date == yesterday_pos
            slice_ = daily_df[row_mask]
            if slice_.empty:
                raise KeyError
            y_open  = slice_["Open"].iloc[0]
            y_high  = slice_["High"].iloc[0]
            y_low   = slice_["Low"].iloc[0]
            y_close = slice_["Close"].iloc[0]
        except Exception:
            return pd.DataFrame(), tickers_list, {
                **debug, "error": f"daily baseline row for {yesterday_pos} not present",
            }, None

    yesterday_ohlc = pd.DataFrame({
        "open": y_open, "high": y_high, "low": y_low, "close": y_close,
    })
    # Re-index from yfinance ticker (e.g. "RELIANCE.NS") → bare symbol.
    yesterday_ohlc.index = yesterday_ohlc.index.map(lambda t: sym_by_yt.get(t, t))
    yesterday_ohlc = yesterday_ohlc.dropna()
    debug["resolved_yesterday"] = int(len(yesterday_ohlc))

    rows: list[dict] = []
    last_row: Optional[BreadthRow] = None
    for ts in intraday_day.index:
        today_ohlc = pd.DataFrame({
            "open":  today_open_per_ticker,
            "high":  cum_high.loc[ts],
            "low":   cum_low.loc[ts],
            "close": cum_close.loc[ts],
        }).dropna()
        if today_ohlc.empty:
            continue
        today_ohlc.index = today_ohlc.index.map(lambda t: sym_by_yt.get(t, t))

        row = compute_breadth_row(today_ohlc, yesterday_ohlc)
        last_row = row
        # IST hh:mm label (ts is already IST-localized after tz_convert above).
        time_label = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
        rows.append({
            "Time":          time_label,
            "Bull BO %":     row.bullish_bo_pct,
            "Abv Close %":   row.above_close_pct,
            "Green Range %": row.green_range_pct,
            "Today High#":   row.today_high_count,
            "Score Bull":    row.score_bull,
            "Bear BO %":     row.bearish_bo_pct,
            "Bel Close %":   row.below_close_pct,
            "Red Range %":   row.red_range_pct,
            "Today Low#":    row.today_low_count,
            "Score Bear":    row.score_bear,
        })

    debug["resolved_buckets"] = len(rows)
    return pd.DataFrame(rows), [], debug, last_row
