"""Primitive series maths, written to match Pine exactly where the port depends
on it.

Two of these are places where "obviously equivalent" implementations disagree
with TradingView by enough to move a signal:

* `atr` must use Wilder's RMA seeded with an SMA, not an EMA and not a rolling
  mean. An EMA-based ATR runs a few percent tight, which shifts every
  ATR-scaled threshold in the model.
* `pivot_high` / `pivot_low` confirm `right` bars late, and the returned value
  belongs to the bar `right` back -- not to the bar you are standing on. Getting
  this wrong is lookahead bias, and it flatters a backtest badly.

VWAP and relative volume are additions, not ports; the Pine indicator has no
volume awareness at all. Reasoning for both is in the docstrings.

Stdlib only.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .bars import Bar

# ── true range / ATR ────────────────────────────────────────────────────────


def true_range(bars: list[Bar]) -> list[float]:
    """Wilder's true range. The first bar has no previous close, so its TR is
    just its own range -- which is what Pine does."""
    out: list[float] = []
    for i, b in enumerate(bars):
        if i == 0:
            out.append(b.range)
        else:
            pc = bars[i - 1].close
            out.append(max(b.high - b.low, abs(b.high - pc), abs(b.low - pc)))
    return out


def rma(values: list[float], length: int) -> list[float]:
    """Wilder's smoothing, seeded with the SMA of the first `length` values.

    Positions before the seed are `nan`, matching Pine's `na`. Callers must
    check; a `nan` threshold silently makes every comparison False, which is
    the quiet-wrong-answer failure this whole codebase is trying to avoid."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    out: list[float] = [math.nan] * len(values)
    if len(values) < length:
        return out
    seed = sum(values[:length]) / length
    out[length - 1] = seed
    alpha = 1.0 / length
    prev = seed
    for i in range(length, len(values)):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def atr(bars: list[Bar], length: int = 14) -> list[float]:
    """Average true range, Pine-compatible (`ta.atr`)."""
    return rma(true_range(bars), length)


# ── pivots ──────────────────────────────────────────────────────────────────


def pivot_high(bars: list[Bar], left: int, right: int) -> list[float | None]:
    """Pine's `ta.pivothigh`.

    `out[i]` is non-None when the bar at `i - right` is a confirmed pivot high,
    and its value is that bar's high. The offset is the entire point: a pivot
    cannot be known until `right` more bars have printed, so any strategy
    reading `out[i]` is allowed to act at bar `i` and no earlier.

    Ties use strict inequality on the left and non-strict on the right, which
    is Pine's convention -- a flat double top confirms on the later bar only."""
    n = len(bars)
    out: list[float | None] = [None] * n
    for i in range(left + right, n):
        p = i - right
        pivot = bars[p].high
        ok = True
        for j in range(p - left, p):
            if bars[j].high >= pivot:
                ok = False
                break
        if ok:
            for j in range(p + 1, p + right + 1):
                if bars[j].high > pivot:
                    ok = False
                    break
        if ok:
            out[i] = pivot
    return out


def pivot_low(bars: list[Bar], left: int, right: int) -> list[float | None]:
    """Pine's `ta.pivotlow`. See `pivot_high` for the offset convention."""
    n = len(bars)
    out: list[float | None] = [None] * n
    for i in range(left + right, n):
        p = i - right
        pivot = bars[p].low
        ok = True
        for j in range(p - left, p):
            if bars[j].low <= pivot:
                ok = False
                break
        if ok:
            for j in range(p + 1, p + right + 1):
                if bars[j].low < pivot:
                    ok = False
                    break
        if ok:
            out[i] = pivot
    return out


# ── rolling extremes ────────────────────────────────────────────────────────


def highest(bars: list[Bar], length: int) -> list[float]:
    """Rolling max of `high` over `length` bars ending at each index.

    Includes the current bar, matching `ta.highest`. Positions with fewer than
    `length` bars of history use what exists rather than returning nan -- Pine
    does the same, and a sweep test on a short warmup is still meaningful."""
    out: list[float] = []
    for i in range(len(bars)):
        lo = max(0, i - length + 1)
        out.append(max(b.high for b in bars[lo : i + 1]))
    return out


def lowest(bars: list[Bar], length: int) -> list[float]:
    out: list[float] = []
    for i in range(len(bars)):
        lo = max(0, i - length + 1)
        out.append(min(b.low for b in bars[lo : i + 1]))
    return out


# ── VWAP (addition, not a port) ─────────────────────────────────────────────


def session_vwap(bars: list[Bar]) -> tuple[list[float], list[float]]:
    """Session-anchored VWAP and its volume-weighted standard deviation.

    Returns (vwap, stdev) aligned to `bars`, both resetting at each NY session
    date. Typical price is hlc3, matching every charting package.

    Why this exists: the Pine indicator scores proximity to key opens, previous
    day high/low, and session ranges -- but not to VWAP, which is the reference
    institutional intraday execution benchmark on SPY and the level most
    algorithmic flow is actually measured against. A rejection block forming at
    VWAP is a materially different event from one forming 40bp away, and the
    model currently cannot tell those apart.

    Variance uses the shifted-data form (E[x^2] - E[x]^2). It is clamped at zero
    because float cancellation can drive it slightly negative when price barely
    moves -- exactly the deep-flat case where a naive sqrt raises."""
    vwap: list[float] = []
    stdev: list[float] = []
    cur_date: str | None = None
    sum_pv = sum_v = sum_ppv = 0.0

    for b in bars:
        d = b.session_date_ny
        if d != cur_date:
            cur_date = d
            sum_pv = sum_v = sum_ppv = 0.0
        tp = (b.high + b.low + b.close) / 3.0
        # A zero-volume bar carries no execution information, but dropping it
        # entirely would leave vwap undefined at the session's first bar on a
        # feed that reports volume late. Fall back to the typical price.
        sum_pv += tp * b.volume
        sum_v += b.volume
        sum_ppv += tp * tp * b.volume
        if sum_v > 0:
            vw = sum_pv / sum_v
            var = max(sum_ppv / sum_v - vw * vw, 0.0)
            vwap.append(vw)
            stdev.append(math.sqrt(var))
        else:
            vwap.append(tp)
            stdev.append(0.0)
    return vwap, stdev


# ── relative volume (addition, not a port) ──────────────────────────────────


def relative_volume(bars: list[Bar], lookback_sessions: int = 20) -> list[float]:
    """Volume relative to the same minute-of-day over prior sessions.

    Returns a multiple: 1.0 means "normal for this time of day", 2.5 means two
    and a half times normal. Bars without enough history return 1.0, which is
    neutral for any threshold test.

    Comparing against a flat rolling average of raw volume would be useless
    intraday -- SPY's 09:35 volume is several times its 12:15 volume every
    single day, so a flat average would mark every morning bar as exceptional
    and every lunch bar as dead. The comparison has to be like-for-like by
    minute of day, which is what this does.

    Why the model needs it: the Pine indicator never reads volume. A rejection
    block that forms on a thin lunchtime bar and one that forms on a 4x-volume
    sweep are graded identically today."""
    history: dict[int, list[float]] = defaultdict(list)
    out: list[float] = []
    for b in bars:
        m = b.minute_of_day_ny
        prior = history[m]
        if len(prior) >= 3:
            window = prior[-lookback_sessions:]
            avg = sum(window) / len(window)
            out.append(b.volume / avg if avg > 0 else 1.0)
        else:
            out.append(1.0)
        history[m].append(b.volume)
    return out
