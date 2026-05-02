"""F&O cash-universe loader — broker-agnostic.

Reads a CSV of NSE tradingsymbols. The CSV format is intentionally
minimal so it can be edited by hand and bundled into the Streamlit
deploy without dragging the existing nest data layer along.

CSV format (header required):

    symbol
    RELIANCE
    HDFCBANK
    INFY
    ...

Production NSE F&O cash-universe lives in
``data/fno_cash_universe.csv`` (used by the existing nest API).
The Streamlit deployment bundles its own copy at
``streamlit_app/universe.csv`` (mirrored from the same source).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str        # NSE tradingsymbol (uppercase, no exchange prefix)

    @property
    def yfinance_ticker(self) -> str:
        """Yahoo Finance ticker for this NSE symbol — adds the
        ``.NS`` suffix Yahoo uses for NSE listings."""
        return f"{self.symbol}.NS"


def load_universe(csv_path: str | Path) -> list[UniverseEntry]:
    """Load the universe from a CSV file. Skips blank rows and
    comments (lines starting with ``#``). Symbol column may be the
    first column or a column literally named ``symbol`` —
    detection is case-insensitive on the header.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Universe CSV not found at {path}. "
            f"Expected a CSV with at least a 'symbol' column."
        )

    out: list[UniverseEntry] = []
    seen: set[str] = set()
    sym_col = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return []

    # Header detection: if the first row's first cell is literally
    # "symbol" (case-insensitive), treat it as a header and locate
    # the symbol column. Otherwise the file has no header (matches
    # the existing data/fno_cash_universe.csv shape) — start from
    # row 0.
    first = [c.strip() for c in rows[0]] if rows else []
    has_header = bool(first) and first[0].lower() == "symbol"
    if has_header:
        for i, name in enumerate(first):
            if name.lower() == "symbol":
                sym_col = i
                break
        rows = rows[1:]

    for row in rows:
        if not row:
            continue
        if sym_col >= len(row):
            continue
        cell = row[sym_col].strip().upper()
        if not cell or cell.startswith("#"):
            continue
        if cell in seen:
            continue
        seen.add(cell)
        out.append(UniverseEntry(symbol=cell))
    return out
