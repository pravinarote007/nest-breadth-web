"""Optional local DuckDB cache for the Streamlit app.

When run locally, the WPF "Free Breadth" worker writes breadth rows
to ``../data/breadth_yf.duckdb``. Reading that file on cold start
lets the Streamlit history grid populate instantly instead of
waiting for the first 60-sec poll cycle, and persists each poll
across page refreshes / app restarts.

On Streamlit Cloud the file doesn't exist (no persistent disk),
so all functions degrade silently to no-ops.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# DB location — search a few likely paths so this works whether the
# app is launched from green-field/, green-field/streamlit_app/, or
# anywhere else.
_CANDIDATE_PATHS = [
    Path(__file__).parent.parent / "data" / "breadth_yf.duckdb",
    Path.cwd() / "data" / "breadth_yf.duckdb",
    Path(os.environ.get("BREADTH_YF_DB", "")) if os.environ.get("BREADTH_YF_DB") else None,
]


def _resolve_db() -> Optional[Path]:
    for p in _CANDIDATE_PATHS:
        if p is not None and p.exists():
            return p
    return None


def _resolve_writable_db() -> Optional[Path]:
    """Same as `_resolve_db`, but also OK if the file doesn't yet
    exist as long as its parent directory is writable. Used by the
    persistence path so the Streamlit app can create the file the
    first time it runs locally without piggy-backing on the WPF
    worker."""
    for p in _CANDIDATE_PATHS:
        if p is None:
            continue
        if p.exists():
            return p
        if p.parent.exists():
            return p
    return None


_GRID_COLS = (
    "scope", "date", "ts",
    "d_bullish_bo_pct", "d_above_close_pct", "d_green_range_pct",
    "d_bearish_bo_pct", "d_below_close_pct", "d_red_range_pct",
    "d_today_high_count", "d_today_low_count",
    "d_score_bull", "d_score_bear",
    "universe_size", "computed_at",
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS breadth_grid (
    scope           VARCHAR NOT NULL DEFAULT 'fno_yf',
    date            DATE NOT NULL,
    ts              TIMESTAMP WITH TIME ZONE NOT NULL,
    d_bullish_bo_pct    DOUBLE,
    d_above_close_pct   DOUBLE,
    d_green_range_pct   DOUBLE,
    d_bearish_bo_pct    DOUBLE,
    d_below_close_pct   DOUBLE,
    d_red_range_pct     DOUBLE,
    d_today_high_count  INTEGER,
    d_today_low_count   INTEGER,
    d_score_bull        DOUBLE,
    d_score_bear        DOUBLE,
    universe_size       INTEGER,
    computed_at         TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (scope, date, ts)
);
"""


def load_today(d: Optional[date] = None) -> pd.DataFrame:
    """Read all breadth_grid rows for today (or `d`). Returns an
    empty DataFrame if the DB is missing, the table is missing, or
    duckdb isn't installed."""
    p = _resolve_db()
    if p is None:
        return pd.DataFrame()
    try:
        import duckdb
    except ImportError:
        return pd.DataFrame()
    target = d or date.today()
    try:
        con = duckdb.connect(str(p), read_only=True)
        try:
            df = con.execute(
                "SELECT * FROM breadth_grid WHERE date = ? ORDER BY ts",
                [target],
            ).df()
        finally:
            con.close()
        return df
    except Exception:
        # File exists but schema may not — silent degrade.
        return pd.DataFrame()


def upsert_row(
    *,
    poll_ts: datetime,
    poll_date: date,
    bullish_bo_pct: float, above_close_pct: float, green_range_pct: float,
    bearish_bo_pct: float, below_close_pct: float, red_range_pct: float,
    today_high_count: int, today_low_count: int,
    score_bull: float, score_bear: float,
    universe_size: int,
    scope: str = "fno_yf",
) -> bool:
    """Persist one breadth row. Returns True on success, False if the
    DB couldn't be written (Streamlit Cloud, missing duckdb, etc.).

    Same schema + PK as the WPF-side ``breadth_yf_store``, so both
    surfaces can write into the same file without collisions when
    they run on the same machine.
    """
    p = _resolve_writable_db()
    if p is None:
        return False
    try:
        import duckdb
    except ImportError:
        return False
    try:
        df = pd.DataFrame([{
            "scope": scope,
            "date": poll_date,
            "ts": poll_ts,
            "d_bullish_bo_pct": float(bullish_bo_pct),
            "d_above_close_pct": float(above_close_pct),
            "d_green_range_pct": float(green_range_pct),
            "d_bearish_bo_pct": float(bearish_bo_pct),
            "d_below_close_pct": float(below_close_pct),
            "d_red_range_pct": float(red_range_pct),
            "d_today_high_count": int(today_high_count),
            "d_today_low_count": int(today_low_count),
            "d_score_bull": float(score_bull),
            "d_score_bear": float(score_bear),
            "universe_size": int(universe_size),
            "computed_at": poll_ts,
        }])
        con = duckdb.connect(str(p))
        try:
            con.execute(_SCHEMA)
            con.register("incoming", df)
            col_list = ", ".join(_GRID_COLS)
            update_set = ", ".join(
                f"{c} = excluded.{c}" for c in _GRID_COLS
                if c not in ("scope", "date", "ts")
            )
            con.execute(
                f"INSERT INTO breadth_grid ({col_list}) "
                f"SELECT {col_list} FROM incoming "
                f"ON CONFLICT (scope, date, ts) DO UPDATE SET {update_set}"
            )
            con.unregister("incoming")
        finally:
            con.close()
        return True
    except Exception:
        return False
