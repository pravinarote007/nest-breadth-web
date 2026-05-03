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

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

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
    page_title="NIFTY F&O Breadth (free)",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    "<style>div.block-container{padding-top:2rem}</style>",
    unsafe_allow_html=True,
)

st.title("NIFTY F&O Breadth")


# --- Auto-rerun every 60 seconds ---------------------------------------

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=POLL_SECONDS * 1000, key="poll-tick")
except ImportError:
    st.info(
        "Install `streamlit-autorefresh` (in requirements.txt) for "
        "auto-poll. Currently you'll need to refresh the page manually."
    )


# --- Data fetch (cached for 60s) ---------------------------------------

@st.cache_data(ttl=POLL_SECONDS, show_spinner="Polling Yahoo Finance...")
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

last_poll = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

with st.spinner("Computing breadth..."):
    ext, today_ohlc, yesterday_ohlc, missing, debug = fetch_breadth()

row = ext.daily


# --- Section 1: 4 sentiment cards (Overall / Daily / Weekly / Options) ---

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


if not today_ohlc.empty:
    overall_v = classify_overall(ext)
    daily_v = classify_daily(ext.daily)
    weekly_v = classify_weekly(ext.weekly) if ext.weekly is not None else None
    options_v = classify_option_selling(ext.daily)

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

    # Combined reasoning bullets, mirroring the WPF live-breadth pane.
    bullet_lines: list[str] = []
    bullet_lines.extend(overall_v.reasons)
    if bullet_lines:
        st.markdown(
            "<div style='margin-top:8px;color:#cccccc;font-size:13px;'>"
            + "".join(f"• {r}<br/>" for r in bullet_lines)
            + "</div>",
            unsafe_allow_html=True,
        )


st.caption(f"Last poll: {last_poll} IST · Auto-refresh every {POLL_SECONDS}s")


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
                "Time": ts_ist.strftime("%H:%M:%S"),
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

if not today_ohlc.empty:
    # Column order matches WPF Live Breadth (Daily tab): time, then all
    # bull-side metrics together, then all bear-side metrics together.
    history_row = {
        "Time": last_poll[-8:],            # HH:MM:SS
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
                "Time": last_poll[-8:],
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


# --- Section 2: tabs -- F&O breadth rows | Per-symbol direction --------

tab_breadth, tab_weekly, tab_history, tab_symbols = st.tabs([
    "📈 F&O Breadth (Daily)",
    "📅 F&O Breadth (Weekly)",
    "🕒 Historical Breadth",
    "🔢 Per-symbol direction",
])


# Cached historical fetch — keyed on date so repeat clicks return instantly.
@st.cache_data(ttl=300, show_spinner=False)
def _cached_historical_breadth(d: date):
    universe = load_universe(UNIVERSE_CSV)
    return fetch_historical_breadth(d, universe)


# === TAB 1: row-per-poll breadth grid (mirrors WPF Live Breadth) =======
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


# === TAB 2: weekly breadth (today vs last-week baseline) ===============
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


# === TAB 3: historical breadth (date picker + Compute) =================
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
        )
    with col_b:
        run = st.button("Compute", type="primary", use_container_width=True,
                        key="hist_compute")

    if run:
        with st.spinner(f"Fetching 5-min breadth for {target}..."):
            grid_df_h, missing_h, debug_h = _cached_historical_breadth(target)
        st.session_state.history_grid = grid_df_h
        st.session_state.history_meta = (target, missing_h, debug_h)

    if "history_grid" in st.session_state:
        d_h, missing_h, debug_h = st.session_state.history_meta
        grid_h: pd.DataFrame = st.session_state.history_grid
        if grid_h is None or grid_h.empty:
            err = debug_h.get("error", "no data returned")
            st.warning(f"No 5-min breadth available for **{d_h}** — {err}.")
        else:
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
                    f"Universe baseline date: {debug_h.get('yesterday_baseline', '—')}.")
            if missing_h:
                note += f" ⚠ {len(missing_h)} symbols missing."
            st.caption(note)


# === TAB 4: per-symbol direction =======================================
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
