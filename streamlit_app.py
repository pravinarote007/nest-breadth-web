"""NIFTY F&O Breadth — free Streamlit dashboard.

Polls Yahoo Finance every 60s for the cash-universe daily-baseline
+ today's running OHLC, computes breadth in memory, and renders a
heatmap. No DB, no API key, no broker session — runs unattended on
Streamlit Cloud free tier.

Caveats (shown in the UI banner too):
- ~15-min delay vs reality (Yahoo's NSE feed is delayed).
- Cold start when the app wakes from idle ~30 sec.
- Coverage gaps for newly-added F&O constituents that lag Yahoo's
  catalog.
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE F&O cash session window. Outside this window the live F&O Breadth
# tab freezes (no Yahoo polling, no row appends) — historical tab is
# unaffected since it's user-triggered, on-demand.
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def _is_market_open(now_ist: datetime) -> bool:
    """True if `now_ist` falls inside the NSE intraday session
    (Mon–Fri, 09:15–15:30 IST). Holidays are not modeled — yfinance
    returns no fresh bars on holidays anyway, so the live grid simply
    stays frozen with the prior session's last poll."""
    if now_ist.weekday() >= 5:                # Sat=5, Sun=6
        return False
    t = now_ist.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE

import pandas as pd
import streamlit as st
import yfinance as yf

from breadth_compute import (
    compute_breadth_row, compute_breadth_row_extended,
    BreadthRow, ExtendedBreadthRow,
)
from direction_classifier import (
    classify_daily, classify_weekly, classify_overall, classify_option_selling,
    DirectionVerdict,
)
from universe import load_universe
from historical_breadth import fetch_historical_breadth
import local_store

UNIVERSE_CSV = Path(__file__).parent / "universe.csv"
POLL_SECONDS = 60


# --- Page setup ---------------------------------------------------------

st.set_page_config(
    page_title="BreadthBeat",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
<style>
/* Page padding tweak. */
div.block-container { padding-top: 2rem; }

/* Hide Streamlit Cloud's top-right toolbar buttons:
   the Fork / GitHub / Manage-app icons in the deployed app's header.
   Belt-and-braces: a few selectors in case Streamlit changes which
   element holds the badge. */
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.stAppDeployButton { display: none !important; }
header [data-testid="stStatusWidget"] { display: none !important; }
button[title="View fullscreen"] + button { display: none; }
/* The github-corner SVG (a.k.a. "Fork me on GitHub" ribbon) some
   Streamlit Cloud builds inject. */
a.github-fork-ribbon, a[href*="github.com"][aria-label*="GitHub"],
.viewerBadge_link__qRIco, [class*="viewerBadge"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* Lub-dub heartbeat: two quick beats, then rest. ~1.4s cycle. */
@keyframes bb-heartbeat {
  0%   { transform: scale(1);    filter: drop-shadow(0 0 0 rgba(34,197,94,0)); }
  8%   { transform: scale(1.20); filter: drop-shadow(0 0 6px rgba(34,197,94,0.55)); }
  16%  { transform: scale(1.00); }
  24%  { transform: scale(1.12); filter: drop-shadow(0 0 4px rgba(34,197,94,0.4)); }
  32%  { transform: scale(1.00); filter: drop-shadow(0 0 0 rgba(34,197,94,0)); }
  100% { transform: scale(1.00); }
}
.bb-icon {
  animation: bb-heartbeat 2.2s ease-in-out infinite;
  transform-origin: center;
  display: inline-block;
}
</style>
<div style="display:flex;align-items:center;gap:14px;
            margin-top:8px;margin-bottom:8px;">
  <svg class="bb-icon" width="56" height="40" viewBox="0 0 56 40"
       xmlns="http://www.w3.org/2000/svg" fill="none">
    <polyline points="2,20 12,20 16,8 20,32 24,4 28,30 32,16 36,24 40,20 54,20"
              stroke="#22C55E" stroke-width="2.6"
              stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <span style="font-size:42px;font-weight:300;
               color:var(--text-color, #fafafa);
               line-height:1;">BreadthBeat</span>
</div>
""",
    unsafe_allow_html=True,
)


# --- Auto-rerun every 60 seconds ---------------------------------------

try:
    from streamlit_autorefresh import st_autorefresh
    # In-session (Mon–Fri, 09:15–15:30 IST): align refreshes to wall-clock
    # minute boundaries (15:40:00, 15:41:00, …). Each rerun recomputes
    # the interval to land on the next :00 second mark; once aligned,
    # subsequent intervals are exactly 60s and stay locked.
    #
    # Off-hours: throttle to 5-min cadence. Yahoo isn't called either way
    # (fetch_breadth is gated by market_open_now), but the autorefresh
    # tick itself causes a Streamlit rerun, which we suppress to save
    # Streamlit Cloud cycles. 5 min is short enough to detect session
    # open within a couple of refresh ticks.
    #
    # NOTE: the `key` MUST vary with the interval — `streamlit_autorefresh`
    # mounts a JS setInterval timer on first render and ignores subsequent
    # `interval` prop changes for the same key. By varying the key, we
    # force a fresh component (fresh timer) on every rerun.
    _now_ist = datetime.now(IST)
    if _is_market_open(_now_ist):
        _seconds_to_next_min = 60 - _now_ist.second
        if _seconds_to_next_min <= 1:
            # Already on (or just past) the boundary — wait a full cycle
            # so the cache TTL has time to expire before refetching.
            _seconds_to_next_min += 60
        _interval_ms = _seconds_to_next_min * 1000 + 200
        _ar_key = f"poll-tick-{_now_ist.minute}-{_seconds_to_next_min}"
    else:
        _interval_ms = 5 * 60 * 1000           # 5 min
        _ar_key = f"offhours-tick-{_now_ist.hour}-{_now_ist.minute // 5}"
    st_autorefresh(interval=_interval_ms, key=_ar_key)
except ImportError:
    st.info(
        "Install `streamlit-autorefresh` (in requirements.txt) for "
        "auto-poll. Currently you'll need to refresh the page manually."
    )


# --- Data fetch (cached for 60s) ---------------------------------------

# Cache TTL must be strictly less than the polling interval — otherwise
# an aligned-to-minute-boundary rerun (which lands ~60s + 200ms after
# the previous fetch) would still hit a "fresh" cache entry and serve
# stale data. 50s gives a comfortable margin.
@st.cache_data(ttl=50, show_spinner=False)
def fetch_breadth() -> tuple[ExtendedBreadthRow, pd.DataFrame, pd.DataFrame, list[str], dict]:
    """One yf.download call → today's running + yesterday's OHLC →
    compute_breadth_row. Returns (row, today_ohlc, yesterday_ohlc,
    missing_symbols, debug_info).

    Uses field-first multi-index access (default ``group_by='column'``)
    which is more robust than ticker-first across yfinance versions:

        df["Close"]                  → wide DataFrame, columns=tickers
        df["Close"]["RELIANCE.NS"]   → Series of closes for that ticker

    Period is 5 days (not 2) so weekends + holidays still leave us
    with at least 2 trading days in the window.
    """
    universe = load_universe(UNIVERSE_CSV)
    tickers_list = [u.yfinance_ticker for u in universe]
    tickers = " ".join(tickers_list)

    df = yf.download(
        tickers=tickers,
        period="15d", interval="1d",
        progress=False, threads=False,
        auto_adjust=False,
        # default group_by="column" → field-first multi-index
    )

    debug: dict = {
        "yf_response_shape": list(df.shape) if df is not None else None,
        "yf_columns_preview": [],
        "tickers_requested": len(tickers_list),
        "yf_returned_empty": df is None or df.empty,
    }
    if df is not None and not df.empty:
        # Capture first ~10 column tuples for diagnostic display.
        cols = list(df.columns)[:10]
        debug["yf_columns_preview"] = [str(c) for c in cols]
        debug["yf_columns_nlevels"] = df.columns.nlevels

    today_rows: dict[str, tuple] = {}
    yesterday_rows: dict[str, tuple] = {}
    last_week_rows: dict[str, tuple] = {}
    missing: list[str] = []

    if df is None or df.empty:
        return (ExtendedBreadthRow(
                    daily=compute_breadth_row(pd.DataFrame(), pd.DataFrame()),
                    weekly=None,
                ),
                pd.DataFrame(), pd.DataFrame(), tickers_list, debug)

    # Detect column layout. Two possibilities:
    #   1. Multi-index, field-first: df["Close"][ticker] → Series
    #   2. Single-index (rare, single-ticker case): df["Close"] is a Series
    is_multi = df.columns.nlevels > 1

    def _series_for(ticker: str, field: str):
        if is_multi:
            try:
                level0 = df[field]
            except KeyError:
                return None
            if ticker in level0.columns:
                return level0[ticker]
            return None
        # single-index: only happens when 1 ticker was requested AND yfinance
        # collapsed the multi-index. Just return the single column.
        return df[field] if field in df.columns else None

    for u in universe:
        yf_t = u.yfinance_ticker
        close_s = _series_for(yf_t, "Close")
        open_s = _series_for(yf_t, "Open")
        high_s = _series_for(yf_t, "High")
        low_s = _series_for(yf_t, "Low")
        if close_s is None or close_s.dropna().empty:
            missing.append(u.symbol)
            continue
        # Drop NaN rows (Yahoo sometimes returns a row of NaNs for
        # holidays embedded in the period).
        idx = close_s.dropna().index
        if len(idx) == 0:
            missing.append(u.symbol)
            continue

        last_dt = idx[-1]
        try:
            today_rows[u.symbol] = (
                float(open_s.loc[last_dt]),
                float(high_s.loc[last_dt]),
                float(low_s.loc[last_dt]),
                float(close_s.loc[last_dt]),
            )
        except (KeyError, TypeError, ValueError):
            missing.append(u.symbol)
            continue

        if len(idx) >= 2:
            prev_dt = idx[-2]
            try:
                yesterday_rows[u.symbol] = (
                    float(open_s.loc[prev_dt]),
                    float(high_s.loc[prev_dt]),
                    float(low_s.loc[prev_dt]),
                    float(close_s.loc[prev_dt]),
                )
            except (KeyError, TypeError, ValueError):
                pass
        # Last-week baseline: ~5 trading days back (or whatever the
        # earliest available row is, capped at 6 back).
        if len(idx) >= 6:
            wk_dt = idx[-6]
        elif len(idx) >= 3:
            wk_dt = idx[0]
        else:
            wk_dt = None
        if wk_dt is not None:
            try:
                last_week_rows[u.symbol] = (
                    float(open_s.loc[wk_dt]),
                    float(high_s.loc[wk_dt]),
                    float(low_s.loc[wk_dt]),
                    float(close_s.loc[wk_dt]),
                )
            except (KeyError, TypeError, ValueError):
                pass

    today_ohlc = pd.DataFrame.from_dict(
        today_rows, orient="index", columns=["open", "high", "low", "close"],
    )
    yesterday_ohlc = pd.DataFrame.from_dict(
        yesterday_rows, orient="index", columns=["open", "high", "low", "close"],
    )
    last_week_ohlc = pd.DataFrame.from_dict(
        last_week_rows, orient="index", columns=["open", "high", "low", "close"],
    )
    debug["resolved_today"] = len(today_rows)
    debug["resolved_yesterday"] = len(yesterday_rows)
    debug["resolved_last_week"] = len(last_week_rows)

    ext = compute_breadth_row_extended(
        today_ohlc, yesterday_ohlc,
        last_week_ohlc if not last_week_ohlc.empty else None,
    )
    return ext, today_ohlc, yesterday_ohlc, missing, debug


# --- Color helpers (matplotlib-free) -----------------------------------


def _pct_color(v, clamp: float = 3.0) -> str:
    """Red→grey→green gradient. Clamps at ±``clamp`` percent."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if pd.isna(v):
        return ""
    clamped = max(-clamp, min(clamp, v))
    if clamped >= 0:
        t = clamped / clamp
        r = int(60 + (34 - 60) * t)
        g = int(70 + (197 - 70) * t)
        b = int(60 + (94 - 60) * t)
    else:
        t = -clamped / clamp
        r = int(60 + (239 - 60) * t)
        g = int(70 + (68 - 70) * t)
        b = int(60 + (68 - 60) * t)
    return f"background-color: rgb({r},{g},{b}); color: white"


def _score_color(v: float) -> str:
    """Same scheme but on the 0..100 score scale."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if pd.isna(v):
        return ""
    # Center at 50 → grey; anchor green/red beyond ±25 from centre.
    delta = v - 50.0
    return _pct_color(delta / 25.0 * 3.0)    # rescale to ±3 range


def _net_color(v: float) -> str:
    """For the bull−bear net column (already centered at 0)."""
    return _pct_color(v / 30.0 * 3.0)        # rescale ±30 → ±3


# --- Fetch + render ----------------------------------------------------

now_ist = datetime.now(IST)
last_poll = now_ist.strftime("%Y-%m-%d %H:%M:%S")
market_open_now = _is_market_open(now_ist)

# Off-hours: don't burn Yahoo calls or write fresh history rows. Reuse
# whatever the last successful in-session fetch produced; the F&O Breadth
# tab below renders the frozen view with a "Market closed" banner. The
# Historical Breadth tab is independent — user-triggered Compute, so
# it works any time.
if market_open_now:
    with st.spinner("Computing breadth..."):
        ext, today_ohlc, yesterday_ohlc, missing, debug = fetch_breadth()
    # Cache the latest live result so off-hours reruns can render it.
    st.session_state["live_last_ext"] = ext
    st.session_state["live_last_today_ohlc"] = today_ohlc
    st.session_state["live_last_yesterday_ohlc"] = yesterday_ohlc
    st.session_state["live_last_missing"] = missing
    st.session_state["live_last_debug"] = debug
    st.session_state["live_last_poll_ist"] = last_poll
else:
    ext = st.session_state.get("live_last_ext")
    today_ohlc = st.session_state.get("live_last_today_ohlc", pd.DataFrame())
    yesterday_ohlc = st.session_state.get("live_last_yesterday_ohlc", pd.DataFrame())
    missing = st.session_state.get("live_last_missing", [])
    debug = st.session_state.get("live_last_debug", {"market_closed": True})
    if ext is None:
        # No prior in-session live fetch (first visit outside hours).
        # Render an empty extended row so the rest of the UI doesn't blow up.
        ext = ExtendedBreadthRow(
            daily=compute_breadth_row(pd.DataFrame(), pd.DataFrame()),
            weekly=None,
        )

row = ext.daily


# --- Sentiment-cards helper (Overall / Daily / Weekly / Options) -------

def _render_verdict_card(col, title: str, verdict: DirectionVerdict) -> None:
    col.markdown(
        f"""
<div style="border:1px solid #2a2a2a;border-radius:8px;padding:10px 12px;
            background:#161616;min-height:78px;">
  <div style="font-size:11px;letter-spacing:0.08em;color:#9CA3AF;
              text-transform:uppercase;margin-bottom:6px;">{title}</div>
  <div style="background:{verdict.color};color:white;
              padding:6px 10px;border-radius:6px;font-weight:600;
              font-size:14px;display:inline-block;">{verdict.label}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_sentiment_block(ext_local: ExtendedBreadthRow) -> None:
    """Renders the 4 verdict cards + reasoning bullets. Reused by the
    live F&O Breadth tab and the Historical Breadth tab so both
    surfaces show the same WPF-style summary."""
    overall_v = classify_overall(ext_local)
    daily_v = classify_daily(ext_local.daily)
    weekly_v = classify_weekly(ext_local.weekly) if ext_local.weekly is not None else None
    options_v = classify_option_selling(ext_local.daily)

    cc1, cc2, cc3, cc4 = st.columns(4)
    _render_verdict_card(cc1, "Overall", overall_v)
    _render_verdict_card(cc2, "Daily", daily_v)
    if weekly_v is not None:
        _render_verdict_card(cc3, "Weekly", weekly_v)
    else:
        cc3.markdown(
            """
<div style="border:1px solid #2a2a2a;border-radius:8px;padding:10px 12px;
            background:#161616;min-height:78px;">
  <div style="font-size:11px;letter-spacing:0.08em;color:#9CA3AF;
              text-transform:uppercase;margin-bottom:6px;">Weekly</div>
  <div style="background:#9CA3AF;color:white;padding:6px 10px;
              border-radius:6px;font-weight:600;font-size:14px;
              display:inline-block;">N/A · need 5+ days</div>
</div>
""",
            unsafe_allow_html=True,
        )
    _render_verdict_card(cc4, "Options · Buy / Sell", options_v)

    if overall_v.reasons:
        st.markdown(
            "<div style='margin-top:8px;color:#cccccc;font-size:13px;'>"
            + "".join(f"• {r}<br/>" for r in overall_v.reasons)
            + "</div>",
            unsafe_allow_html=True,
        )


# --- Append to in-session breadth-row history --------------------------
#
# Streamlit Cloud has no persistent disk, but session_state lives for
# the user's open browser session. Each successful poll appends one
# row → the F&O tab below shows the running history while the page
# is open. State resets when the user closes the page or the app
# sleeps for >30 min.

if "weekly_history" not in st.session_state:
    st.session_state.weekly_history = []

if "breadth_history" not in st.session_state:
    st.session_state.breadth_history = []
    # Hydrate from local DuckDB on first page load (no-op on Streamlit
    # Cloud where the file doesn't exist).
    cached_df = local_store.load_today()
    if not cached_df.empty:
        for _, r in cached_df.iterrows():
            ts_val = r.get("ts")
            try:
                ts_ist = pd.Timestamp(ts_val).tz_convert(IST)
            except (TypeError, ValueError, AttributeError):
                ts_ist = pd.Timestamp(ts_val)
            st.session_state.breadth_history.append({
                "Time": ts_ist.strftime("%H:%M"),
                "Bull BO %": float(r["d_bullish_bo_pct"] or 0.0),
                "Abv Close %": float(r["d_above_close_pct"] or 0.0),
                "Green Range %": float(r["d_green_range_pct"] or 0.0),
                "Today High#": int(r["d_today_high_count"] or 0),
                "Score Bull": float(r["d_score_bull"] or 0.0),
                "Bear BO %": float(r["d_bearish_bo_pct"] or 0.0),
                "Bel Close %": float(r["d_below_close_pct"] or 0.0),
                "Red Range %": float(r["d_red_range_pct"] or 0.0),
                "Today Low#": int(r["d_today_low_count"] or 0),
                "Score Bear": float(r["d_score_bear"] or 0.0),
            })

# --- Intra-day backfill / gap-fill ----------------------------------
# Fires when the grid is empty (mid-session first open) OR when the
# latest row is more than ~6 min behind now (browser closed, reopened
# later). The 6-min threshold tolerates Yahoo's ~15-min delay without
# spuriously refetching during live polling.
#
# The Yahoo fetch is cached (5-min TTL) so multiple Streamlit reruns in
# a short window dedup to one API call.

@st.cache_data(ttl=300, show_spinner=False)
def _backfill_today_grid(target_date: date) -> pd.DataFrame:
    universe = load_universe(UNIVERSE_CSV)
    grid_df, _, _, _ = fetch_historical_breadth(target_date, universe)
    return grid_df


def _grid_gap_minutes(history: list, now_ist: datetime) -> float:
    """Minutes between now and the latest row's HH:MM. Returns +inf
    when the grid is empty so that maps to 'big gap → backfill'."""
    if not history:
        return float("inf")
    try:
        last_t = str(history[-1].get("Time", ""))
        hh, mm = last_t.split(":")
        last_dt = datetime.combine(
            now_ist.date(), dtime(int(hh), int(mm)), tzinfo=IST,
        )
        return (now_ist - last_dt).total_seconds() / 60.0
    except (ValueError, AttributeError, KeyError):
        return float("inf")


if market_open_now and _grid_gap_minutes(st.session_state.breadth_history, now_ist) > 6:
    try:
        bf_grid = _backfill_today_grid(now_ist.date())
        if not bf_grid.empty:
            existing_times = {r["Time"] for r in st.session_state.breadth_history
                              if isinstance(r.get("Time"), str)}
            for _, r in bf_grid.iterrows():
                t = str(r["Time"])
                if t in existing_times:
                    continue
                st.session_state.breadth_history.append({
                    "Time":          t,
                    "Bull BO %":     float(r["Bull BO %"]),
                    "Abv Close %":   float(r["Abv Close %"]),
                    "Green Range %": float(r["Green Range %"]),
                    "Today High#":   int(r["Today High#"]),
                    "Score Bull":    float(r["Score Bull"]),
                    "Bear BO %":     float(r["Bear BO %"]),
                    "Bel Close %":   float(r["Bel Close %"]),
                    "Red Range %":   float(r["Red Range %"]),
                    "Today Low#":    int(r["Today Low#"]),
                    "Score Bear":    float(r["Score Bear"]),
                })
            # Re-sort chronologically — the live tab reverses for
            # newest-first display, but internal order matters for the
            # last-row gap check on subsequent reruns.
            st.session_state.breadth_history.sort(key=lambda x: str(x.get("Time", "")))
    except Exception:
        # Best-effort — if Yahoo rate-limits or returns weird shape,
        # the live polls below will still populate the grid forward.
        pass

if market_open_now and not today_ohlc.empty:
    # Column order matches WPF Live Breadth (Daily tab): time, then all
    # bull-side metrics together, then all bear-side metrics together.
    history_row = {
        "Time": last_poll[-8:-3],     # HH:MM (drop ":SS")
        "Bull BO %": row.bullish_bo_pct,
        "Abv Close %": row.above_close_pct,
        "Green Range %": row.green_range_pct,
        "Today High#": row.today_high_count,
        "Score Bull": row.score_bull,
        "Bear BO %": row.bearish_bo_pct,
        "Bel Close %": row.below_close_pct,
        "Red Range %": row.red_range_pct,
        "Today Low#": row.today_low_count,
        "Score Bear": row.score_bear,
    }
    # Avoid duplicating consecutive identical polls (cache hits during
    # the 60-sec TTL window will return the same row repeatedly).
    if (not st.session_state.breadth_history
            or st.session_state.breadth_history[-1]["Time"] != history_row["Time"]):
        st.session_state.breadth_history.append(history_row)
        # Cap history at ~500 rows so a long open session doesn't bloat memory.
        if len(st.session_state.breadth_history) > 500:
            st.session_state.breadth_history = st.session_state.breadth_history[-500:]
        # Persist to local DuckDB (no-op on Streamlit Cloud).
        now_ist = datetime.now(IST)
        local_store.upsert_row(
            poll_ts=now_ist, poll_date=now_ist.date(),
            bullish_bo_pct=row.bullish_bo_pct,
            above_close_pct=row.above_close_pct,
            green_range_pct=row.green_range_pct,
            bearish_bo_pct=row.bearish_bo_pct,
            below_close_pct=row.below_close_pct,
            red_range_pct=row.red_range_pct,
            today_high_count=row.today_high_count,
            today_low_count=row.today_low_count,
            score_bull=row.score_bull,
            score_bear=row.score_bear,
            universe_size=row.universe_size,
        )

        # Weekly history (in-session only — local_store schema is daily).
        if ext.weekly is not None:
            wk = ext.weekly
            weekly_row = {
                "Time": last_poll[-8:-3],     # HH:MM (drop ":SS")
                "Bull BO %": wk.bullish_bo_pct,
                "Abv Close %": wk.above_close_pct,
                "Green Range %": wk.green_range_pct,
                "Today High#": wk.today_high_count,
                "Score Bull": wk.score_bull,
                "Bear BO %": wk.bearish_bo_pct,
                "Bel Close %": wk.below_close_pct,
                "Red Range %": wk.red_range_pct,
                "Today Low#": wk.today_low_count,
                "Score Bear": wk.score_bear,
            }
            if (not st.session_state.weekly_history
                    or st.session_state.weekly_history[-1]["Time"] != weekly_row["Time"]):
                st.session_state.weekly_history.append(weekly_row)
                if len(st.session_state.weekly_history) > 500:
                    st.session_state.weekly_history = st.session_state.weekly_history[-500:]


# --- Top-level layout: 2 tabs (F&O Breadth | Historical Breadth) ------

# Cached historical fetch — keyed on date so repeat clicks return instantly.
@st.cache_data(ttl=300, show_spinner=False)
def _cached_historical_breadth(d: date):
    universe = load_universe(UNIVERSE_CSV)
    return fetch_historical_breadth(d, universe)


tab_fno, tab_history = st.tabs([
    "📈 F&O Breadth",
    "🕒 Historical Breadth",
])


# ──────────────── F&O BREADTH (live) ────────────────
with tab_fno:
    # Off-hours banner — frozen view of the most recent in-session
    # market-hours poll. Use the Historical Breadth tab for any past
    # date.
    if not market_open_now:
        last_live_poll = st.session_state.get("live_last_poll_ist")
        if last_live_poll:
            st.warning(
                f"🔒 **Market closed.** NSE F&O session runs Mon–Fri "
                f"09:15–15:30 IST. Showing the last live poll at "
                f"**{last_live_poll[-8:-3]} IST**. Polling resumes at "
                f"the next session open."
            )
        else:
            st.info(
                "🔒 **Market closed.** NSE F&O session runs Mon–Fri "
                "09:15–15:30 IST. Live polling will start at the next "
                "session open. For past sessions use the Historical "
                "Breadth tab."
            )

    # Sentiment cards from the latest live poll.
    if not today_ohlc.empty:
        _render_sentiment_block(ext)

    if market_open_now:
        st.caption(f"Last poll: {last_poll} IST · Auto-refresh every {POLL_SECONDS}s")
    else:
        st.caption(
            f"Now: {last_poll} IST · Polling paused (off-hours)."
        )

    tab_breadth, tab_weekly, tab_symbols = st.tabs([
        "📈 Daily",
        "📅 Weekly",
        "🔢 Per-symbol direction",
    ])

# === Live: Daily history grid ==========================================
with tab_breadth:
    st.markdown("**Newest poll at the top.** One row per poll cycle. "
                "Scroll back to see how breadth evolved through the session.")

    if not st.session_state.breadth_history:
        if debug.get("yf_returned_empty"):
            st.error(
                "Yahoo Finance returned an **empty response**. Most likely "
                "causes: rate-limit, network issue, or invalid tickers. "
                "Try again in 1–2 minutes."
            )
        else:
            st.info("Waiting for first poll to complete…")
        with st.expander("Debug: yfinance response structure"):
            st.json(debug)
    else:
        history_df = pd.DataFrame(st.session_state.breadth_history[::-1])
        # Format mirrors the WPF Live Breadth grid (F2, 2-decimal).
        bull_cols = ["Bull BO %", "Abv Close %", "Green Range %", "Score Bull"]
        bear_cols = ["Bear BO %", "Bel Close %", "Red Range %", "Score Bear"]
        styled = history_df.style.format({
            **{c: "{:.2f}" for c in bull_cols + bear_cols},
        }).map(_score_color, subset=bull_cols) \
          .map(lambda v: _score_color(100 - v) if pd.notna(v) else "",
               subset=bear_cols) \
          .set_properties(subset=["Score Bull"],
                          **{"border-right": "3px solid #888"}) \
          .set_table_styles([{"selector": "th.col_heading.level0:nth-child(6)",
                              "props": [("border-right", "3px solid #888")]}],
                             overwrite=False)
        st.dataframe(styled, use_container_width=True, height=540)

        st.caption(
            f"{len(st.session_state.breadth_history)} polls in session. "
            f"Refresh / close page = state resets."
        )


# === Live: weekly breadth (today vs last-week baseline) ================
with tab_weekly:
    st.markdown(
        "**Weekly view** — same metrics, but compared against the OHLC of "
        "~5 trading days ago (last week's baseline) instead of yesterday's. "
        "Newest poll on top."
    )

    if not st.session_state.weekly_history:
        if ext.weekly is None:
            st.info(
                "Weekly baseline unavailable — Yahoo returned fewer than ~5 "
                "trading days. Try again after the first successful 15-day fetch."
            )
        else:
            st.info("Waiting for first poll to complete…")
    else:
        weekly_df = pd.DataFrame(st.session_state.weekly_history[::-1])
        bull_cols = ["Bull BO %", "Abv Close %", "Green Range %", "Score Bull"]
        bear_cols = ["Bear BO %", "Bel Close %", "Red Range %", "Score Bear"]
        styled_w = weekly_df.style.format({
            **{c: "{:.2f}" for c in bull_cols + bear_cols},
        }).map(_score_color, subset=bull_cols) \
          .map(lambda v: _score_color(100 - v) if pd.notna(v) else "",
               subset=bear_cols) \
          .set_properties(subset=["Score Bull"],
                          **{"border-right": "3px solid #888"}) \
          .set_table_styles([{"selector": "th.col_heading.level0:nth-child(6)",
                              "props": [("border-right", "3px solid #888")]}],
                             overwrite=False)
        st.dataframe(styled_w, use_container_width=True, height=540)

        st.caption(
            f"{len(st.session_state.weekly_history)} polls in session "
            f"(weekly history is in-session only — not persisted)."
        )


# ──────────────── HISTORICAL BREADTH ────────────────
with tab_history:
    st.markdown(
        "**Pick a date and hit Compute.** Replays per-5-min breadth for "
        "the F&O cash universe vs that day's prior-close baseline. "
        "Yahoo retains 5-min bars for ~60 days."
    )
    today_local = datetime.now(IST).date()
    col_d, col_b, col_pad = st.columns([2, 1, 4], vertical_alignment="bottom")
    with col_d:
        target = st.date_input(
            "Date",
            value=today_local - timedelta(days=1),
            min_value=today_local - timedelta(days=58),
            max_value=today_local,
            key="hist_date",
        )
    with col_b:
        run = st.button("Compute", type="primary", use_container_width=True,
                        key="hist_compute")

    if run:
        with st.spinner(f"Fetching 5-min breadth for {target}..."):
            grid_df_h, missing_h, debug_h, ext_h = _cached_historical_breadth(target)
        st.session_state.history_grid = grid_df_h
        st.session_state.history_meta = (target, missing_h, debug_h, ext_h)

    if "history_grid" in st.session_state:
        d_h, missing_h, debug_h, ext_h = st.session_state.history_meta
        grid_h: pd.DataFrame = st.session_state.history_grid
        if grid_h is None or grid_h.empty:
            err = debug_h.get("error", "no data returned")
            st.warning(f"No 5-min breadth available for **{d_h}** — {err}.")
        else:
            # End-of-session sentiment block — same 4 WPF-style cards as the
            # live F&O tab. Weekly card populates when we have a
            # ~5-trading-days-ago baseline; otherwise it degrades to "N/A".
            if ext_h is not None:
                _render_sentiment_block(ext_h)

            # Newest at top to match the live Daily / Weekly tabs.
            view = grid_h[::-1].reset_index(drop=True)
            bull_cols = ["Bull BO %", "Abv Close %", "Green Range %", "Score Bull"]
            bear_cols = ["Bear BO %", "Bel Close %", "Red Range %", "Score Bear"]
            styled_h = view.style.format({
                **{c: "{:.2f}" for c in bull_cols + bear_cols},
            }).map(_score_color, subset=bull_cols) \
              .map(lambda v: _score_color(100 - v) if pd.notna(v) else "",
                   subset=bear_cols) \
              .set_properties(subset=["Score Bull"],
                              **{"border-right": "3px solid #888"}) \
              .set_table_styles([{"selector": "th.col_heading.level0:nth-child(6)",
                                   "props": [("border-right", "3px solid #888")]}],
                                  overwrite=False)
            st.dataframe(styled_h, use_container_width=True, height=540)
            note = (f"{len(view)} 5-min buckets for {d_h}. "
                    f"Baseline date: {debug_h.get('yesterday_baseline', '—')}.")
            if missing_h:
                note += f" ⚠ {len(missing_h)} symbols missing."
            st.caption(note)


# === Live: per-symbol direction ========================================
with tab_symbols:
    if today_ohlc.empty or yesterday_ohlc.empty:
        st.warning(
            f"Resolved today: {debug.get('resolved_today', 0)}, "
            f"yesterday: {debug.get('resolved_yesterday', 0)}, "
            f"missing: {len(missing)}. Will populate after the next "
            f"successful poll."
        )
    else:
        aligned = today_ohlc.join(
            yesterday_ohlc, rsuffix="_y", how="inner",
        )
        aligned["pct_change"] = (
            (aligned["close"] - aligned["close_y"]) / aligned["close_y"] * 100.0
        )
        aligned = aligned.sort_values("pct_change", ascending=False)
        aligned["bias"] = aligned["pct_change"].apply(
            lambda x: "🟢" if x > 0 else ("🔴" if x < 0 else "⚪")
        )
        display = aligned[["bias", "open", "high", "low", "close", "pct_change"]].rename(
            columns={
                "bias": " ", "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "pct_change": "% chg",
            },
        )
        st.dataframe(
            display.style.format({
                "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
                "Close": "{:.2f}", "% chg": "{:+.2f}%",
            }).map(_pct_color, subset=["% chg"]),
            use_container_width=True, height=600,
        )


# Coverage details (collapsible, below tabs).
if missing:
    with st.expander(f"⚠ {len(missing)} symbols not returned by Yahoo"):
        st.write("Likely recently delisted, renamed, or newly-added F&O "
                 "constituents that lag Yahoo's catalog. The breadth "
                 "calculation skips these — universe size is reduced "
                 f"from {len(missing) + row.universe_size} to {row.universe_size} for this poll.")
        st.write(", ".join(missing))

# Disclaimer footer.
st.markdown("---")
st.caption(
    "**Disclaimer**: Free educational view powered for Traders. "
    "Not for live trading triggers. Use a broker WebSocket for "
    "real-time signals."
)
