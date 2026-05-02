"""Direction + conviction classifier — port of WPF
:class:`Nest.App.Services.BreadthDirectionClassifier`.

Same rules, same labels, same colour cues. Used by the Streamlit
dashboard to render the four sentiment cards (Overall / Daily /
Weekly / Options) above the data grid.

Pure / stateless — same row in, same verdict out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from breadth_compute import BreadthRow, ExtendedBreadthRow


# Hex colours that match the WPF brand palette so the heatmap-pill
# colours read identically across the desktop and web surfaces.
GREEN = "#22C55E"
RED = "#EF4444"
AMBER = "#F59E0B"
GREY = "#9CA3AF"


@dataclass(frozen=True)
class DirectionVerdict:
    """One classification at a single moment. The reasoning bullets
    re-state the inputs that drove the verdict — same structure the
    WPF DirectionVerdict record carries.
    """
    scope: str           # "Overall" / "Daily" / "Weekly" / "Option Selling"
    label: str           # e.g. "BULLISH · trend day"
    color: str           # hex string for the badge background
    reasons: list[str]


def _signed(v: float) -> str:
    return f"{v:+.1f}"


def _classify(
    *,
    scope: str,
    bull_bo: float, above_close: float, green_range: float,
    bear_bo: float, below_close: float, red_range: float,
    score_bull: float, score_bear: float,
    today_high: int, today_low: int,
) -> DirectionVerdict:
    lean = score_bull - score_bear
    reasons: list[str] = []

    # (1) Headline numbers.
    reasons.append(
        f"Score lean {_signed(lean)} (bull {score_bull:.1f} − bear {score_bear:.1f})"
    )
    reasons.append(f"Bull BO {bull_bo:.1f}%  ·  Bear BO {bear_bo:.1f}%")

    # (2) Hierarchy alignment check.
    bull_aligned = green_range >= above_close >= bull_bo
    bear_aligned = red_range >= below_close >= bear_bo
    if lean > 10 and bull_aligned:
        reasons.append(
            f"Bull hierarchy aligned: {green_range:.0f}% > open ≥ "
            f"{above_close:.0f}% > close ≥ {bull_bo:.0f}% > prior high"
        )
    elif lean > 10 and not bull_aligned and bull_bo < above_close - 15:
        reasons.append(
            f"Bullish but stuck under resistance — only {bull_bo:.0f}% past "
            f"prior high vs {above_close:.0f}% above prior close"
        )
    elif lean < -10 and bear_aligned:
        reasons.append(
            f"Bear hierarchy aligned: {red_range:.0f}% < open ≥ "
            f"{below_close:.0f}% < close ≥ {bear_bo:.0f}% < prior low"
        )
    elif lean < -10 and not bear_aligned and bear_bo < below_close - 15:
        reasons.append(
            f"Bearish but support holding — only {bear_bo:.0f}% broke prior "
            f"low vs {below_close:.0f}% below prior close"
        )

    # (3) Leadership read — today_high vs today_low.
    if today_high + today_low > 0:
        if today_high >= today_low * 3:
            reasons.append(
                f"Leadership long: {today_high} stocks at intraday highs vs "
                f"{today_low} at lows"
            )
        elif today_low >= today_high * 3:
            reasons.append(
                f"Leadership short: {today_low} stocks at intraday lows vs "
                f"{today_high} at highs"
            )
        else:
            reasons.append(
                f"Leadership thin: {today_high} highs / {today_low} lows"
            )

    # (4) Verdict table.
    if lean > 20 and bull_bo > 40:
        label, color = "BULLISH · trend day", GREEN
    elif lean > 20 and bull_bo >= 20:
        label, color = "BULL · mild", GREEN
    elif lean < -20 and bear_bo > 40:
        label, color = "BEARISH · trend day", RED
    elif lean < -20 and bear_bo >= 20:
        label, color = "BEAR · mild", RED
    elif abs(lean) <= 20 and bull_bo < 20 and bear_bo < 20:
        label, color = "CHOP — sit out", GREY
    else:
        label, color = "MIXED · stock-picker's day", AMBER

    return DirectionVerdict(scope=scope, label=label, color=color, reasons=reasons)


def classify_daily(row: BreadthRow) -> DirectionVerdict:
    return _classify(
        scope="Daily",
        bull_bo=row.bullish_bo_pct, above_close=row.above_close_pct,
        green_range=row.green_range_pct,
        bear_bo=row.bearish_bo_pct, below_close=row.below_close_pct,
        red_range=row.red_range_pct,
        score_bull=row.score_bull, score_bear=row.score_bear,
        today_high=row.today_high_count, today_low=row.today_low_count,
    )


def classify_weekly(row: BreadthRow) -> DirectionVerdict:
    return _classify(
        scope="Weekly",
        bull_bo=row.bullish_bo_pct, above_close=row.above_close_pct,
        green_range=row.green_range_pct,
        bear_bo=row.bearish_bo_pct, below_close=row.below_close_pct,
        red_range=row.red_range_pct,
        score_bull=row.score_bull, score_bear=row.score_bear,
        today_high=row.today_high_count, today_low=row.today_low_count,
    )


def classify_overall(ext: ExtendedBreadthRow) -> DirectionVerdict:
    """Top-level fusion of Daily + Weekly. Conviction comes from
    timeframe agreement; counter-trend setups carry size warnings.

    Mirrors WPF :func:`BreadthDirectionClassifier.ClassifyOverall`.
    When weekly is missing, falls back to daily-only verdict so the
    user still sees a meaningful card."""
    d = ext.daily
    w = ext.weekly
    d_lean = d.score_bull - d.score_bear
    d_strength = max(d.bullish_bo_pct, d.bearish_bo_pct)
    reasons: list[str] = [f"Daily lean {_signed(d_lean)}, max BO {d_strength:.0f}%"]

    if w is None:
        # Weekly baseline missing — degrade to daily-only.
        reasons.append("Weekly baseline unavailable — Overall = Daily verdict")
        daily_verdict = classify_daily(d)
        return DirectionVerdict(
            scope="Overall", label=daily_verdict.label,
            color=daily_verdict.color, reasons=reasons + daily_verdict.reasons[:1],
        )

    w_lean = w.score_bull - w.score_bear
    w_strength = max(w.bullish_bo_pct, w.bearish_bo_pct)
    reasons.append(f"Weekly lean {_signed(w_lean)}, max BO {w_strength:.0f}%")

    d_bull = d_lean > 10
    d_bear = d_lean < -10
    w_bull = w_lean > 10
    w_bear = w_lean < -10
    d_strong = abs(d_lean) > 20 and d_strength > 25
    w_strong = abs(w_lean) > 20 and w_strength > 25

    label: str
    color: str

    if d_bull and w_bull:
        if d_strong and w_strong:
            label, color = "HIGH CONVICTION BULL", GREEN
            reasons.append("Both timeframes in trend-bull regime — aligned momentum")
            reasons.append("Action: full size; sell puts AND/OR buy calls; ride the trend")
        else:
            label, color = "ALIGNED BULL · mild", GREEN
            reasons.append("Both lean bull but neither in trend regime")
            reasons.append("Action: bull put spread; selective long calls on dips")
    elif d_bear and w_bear:
        if d_strong and w_strong:
            label, color = "HIGH CONVICTION BEAR", RED
            reasons.append("Both timeframes in trend-bear regime — aligned momentum")
            reasons.append("Action: full size; sell calls AND/OR buy puts; ride the trend")
        else:
            label, color = "ALIGNED BEAR · mild", RED
            reasons.append("Both lean bear but neither in trend regime")
            reasons.append("Action: bear call spread; selective long puts on bounces")
    elif d_bull and w_bear:
        label, color = "COUNTER-TREND BULL", AMBER
        reasons.append("Daily up against a bearish weekly tape — fade-rally environment")
        reasons.append("Action: half size; defined-risk only; book quickly")
        reasons.append("Risk: weekly downtrend likely reasserts — don't chase")
    elif d_bear and w_bull:
        label, color = "COUNTER-TREND BEAR", AMBER
        reasons.append("Daily down inside a bullish weekly tape — fade-dip environment")
        reasons.append("Action: half size; defined-risk only; book quickly")
        reasons.append("Risk: weekly uptrend likely reasserts — don't press shorts")
    elif abs(d_lean) <= 10 and abs(w_lean) <= 10:
        label, color = "ALIGNED CHOP", GREEN
        reasons.append("Both timeframes flat — cleanest premium-sell environment")
        reasons.append("Action: full size short straddle / 1σ strangle on the index")
    elif abs(d_lean) <= 10 and w_bull:
        label, color = "DRIFT BULL (weekly)", GREEN
        reasons.append("Daily flat but weekly bullish — daily likely catches up")
        reasons.append("Action: bias longs; sell puts into weakness")
    elif abs(d_lean) <= 10 and w_bear:
        label, color = "DRIFT BEAR (weekly)", RED
        reasons.append("Daily flat but weekly bearish — daily likely catches up")
        reasons.append("Action: bias shorts; sell calls into strength")
    elif abs(w_lean) <= 10 and d_bull:
        label, color = "DAILY MOMENTUM BULL", GREEN
        reasons.append("Weekly flat but daily bullish — short-term momentum")
        reasons.append("Action: short-duration longs; expect mean reversion before weekly turns")
    elif abs(w_lean) <= 10 and d_bear:
        label, color = "DAILY MOMENTUM BEAR", RED
        reasons.append("Weekly flat but daily bearish — short-term momentum")
        reasons.append("Action: short-duration shorts; expect mean reversion")
    else:
        label, color = "MIXED · stand aside", GREY
        reasons.append("No coherent read across timeframes")
        reasons.append("Action: skip until alignment forms")

    return DirectionVerdict(scope="Overall", label=label, color=color, reasons=reasons)


def classify_option_selling(row: BreadthRow) -> DirectionVerdict:
    """Mirrors WPF :func:`ClassifyOptionSelling` — option-strategy
    recommendation keyed off daily breadth state."""
    lean = row.score_bull - row.score_bear
    bull_bo = row.bullish_bo_pct
    bear_bo = row.bearish_bo_pct
    strength = max(bull_bo, bear_bo)
    extremes = row.today_high_count + row.today_low_count
    universe = max(1, row.universe_size)
    extreme_pct = 100.0 * extremes / universe

    reasons = [
        f"Lean {_signed(lean)}, Bull BO {bull_bo:.0f}%, Bear BO {bear_bo:.0f}%",
        f"Dispersion: {extremes}/{universe} stocks at intraday extreme ({extreme_pct:.0f}%)",
    ]

    # (1) Compression squeeze — long straddle.
    if strength <= 8 and abs(lean) <= 8 and extreme_pct <= 8:
        reasons.append("Compression: both BO ≤ 8%, dispersion ≤ 8%, lean ~0 — coiled spring")
        reasons.append("Vol is cheap here; expansion almost always follows sustained compression")
        reasons.append("Setup: buy ATM straddle (or 1σ strangle for cheaper entry)")
        reasons.append("Exit trigger: first time max BO crosses 20% — take whichever leg fired")
        return DirectionVerdict("Option Selling", "BUY · long straddle (squeeze)", GREEN, reasons)

    # (2) Wide dispersion — long gamma.
    if extreme_pct >= 30:
        reasons.append(f"Dispersion {extreme_pct:.0f}% — realised vol exceeds typical IV pricing")
        reasons.append("Setup: long ATM straddle for gamma scalp, OR long ATM call/put with the strong side")
        reasons.append("DO NOT sell premium here — gamma will eat theta until dispersion narrows")
        reasons.append("Cover existing short premium positions immediately")
        return DirectionVerdict("Option Selling", "BUY · long gamma", GREEN, reasons)

    # (3) Strong directional days.
    if lean > 25 and bull_bo > 30:
        reasons.append("Strong bull tape — two playbooks fit")
        reasons.append("SELL premium: short put, or bull put spread (collect with the trend)")
        reasons.append("BUY directional: long ATM call if Bull BO just crossed +30 (early breakout)")
        reasons.append("Cover/exit trigger: Bull BO drops below 25%, or lean flips < +10")
        return DirectionVerdict(
            "Option Selling", "BULL · sell puts (or buy calls)", GREEN, reasons
        )
    if lean < -25 and bear_bo > 30:
        reasons.append("Strong bear tape — two playbooks fit")
        reasons.append("SELL premium: short call, or bear call spread (collect with the trend)")
        reasons.append("BUY directional: long ATM put if Bear BO just crossed +30 (early breakdown)")
        reasons.append("Cover/exit trigger: Bear BO drops below 25%, or lean flips > −10")
        return DirectionVerdict(
            "Option Selling", "BEAR · sell calls (or buy puts)", RED, reasons
        )

    # (4) Clean chop.
    if abs(lean) <= 10 and strength <= 15 and extreme_pct <= 10:
        reasons.append("Range-bound, no breakout cohort — pure theta environment")
        reasons.append("Setup: short ATM straddle, or 1σ strangle (sell call + sell put)")
        reasons.append("Cover trigger: any BO % crosses 20% — the universe is breaking out")
        return DirectionVerdict("Option Selling", "CHOP · sell straddle", GREEN, reasons)

    # (5) Mild tilts → defined-risk credit spreads.
    if lean > 10 and bull_bo >= 20:
        reasons.append("Bull tilt without trend strength — defined-risk premium sale")
        reasons.append("Setup: bull put spread, or put-skewed iron condor (wider call wing)")
        reasons.append("Cover trigger: Bull BO declines or lean compresses to < +5")
        return DirectionVerdict(
            "Option Selling", "MILD BULL · bull put spread", GREEN, reasons
        )
    if lean < -10 and bear_bo >= 20:
        reasons.append("Bear tilt without trend strength — defined-risk premium sale")
        reasons.append("Setup: bear call spread, or call-skewed iron condor (wider put wing)")
        reasons.append("Cover trigger: Bear BO declines or lean compresses to > −5")
        return DirectionVerdict(
            "Option Selling", "MILD BEAR · bear call spread", RED, reasons
        )

    # (6) Mixed two-way without dispersion → iron condor.
    if abs(lean) <= 20 and strength <= 25:
        reasons.append("Two-way market, no clear edge — defined-risk non-directional play")
        reasons.append("Setup: 1.5σ iron condor (short call + long further call + short put + long further put)")
        reasons.append("Cover trigger: spot approaches either short strike")
        return DirectionVerdict("Option Selling", "MIXED · iron condor", AMBER, reasons)

    # (7) Anything else — sit out.
    reasons.append("Directional pressure rising but not yet trend-strong — premium-sale edge unclear")
    reasons.append("Wait for either a true breakout (sell with it) or compression back to chop")
    return DirectionVerdict("Option Selling", "SKIP · no clean setup", GREY, reasons)
