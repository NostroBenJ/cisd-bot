"""Candidate signals, each a plain `(bars, i) -> +1 | -1 | 0` function.

Deliberately simple. Every one of these is a single well-known idea expressed
in a handful of lines, because the point of the batch is to find out WHERE
signal lives before investing in any one idea. A candidate that needs 1,500
lines before it can be tested is a candidate that will be tested too late.

Every function reads only bars at or before `i`. `verify_research.py` checks
that by truncation rather than trusting the claim.

These are price-only: the TradingView export carries no volume column, so
relative-volume and VWAP variants are absent until a feed with volume is
loaded. That is a data gap, not a judgement about their value -- volume is one
of the more promising axes and it should be added as soon as the data exists.
"""

from __future__ import annotations

from cisd.bars import Bar
from cisd.sessions import in_window, parse_session

# ── helpers ─────────────────────────────────────────────────────────────────


def _session_bars(bars: list[Bar], i: int) -> list[int]:
    """Indices of bars in the same session, up to and including `i`."""
    day = bars[i].session_date_ny
    out = []
    j = i
    while j >= 0 and bars[j].session_date_ny == day:
        out.append(j)
        j -= 1
    return list(reversed(out))


def _opening_range(bars: list[Bar], i: int, minutes: int) -> tuple[float, float, bool]:
    """(high, low, complete) of the session's first `minutes`."""
    idxs = _session_bars(bars, i)
    open_min = bars[idxs[0]].minute_of_day_ny
    rng = [k for k in idxs if bars[k].minute_of_day_ny < open_min + minutes]
    if not rng:
        return 0.0, 0.0, False
    complete = bars[i].minute_of_day_ny >= open_min + minutes
    return (max(bars[k].high for k in rng), min(bars[k].low for k in rng), complete)


def _prev_session_extremes(bars: list[Bar], i: int) -> tuple[float, float] | None:
    """Previous session's high and low, computed only from bars before `i`."""
    day = bars[i].session_date_ny
    j = i
    while j >= 0 and bars[j].session_date_ny == day:
        j -= 1
    if j < 0:
        return None
    prev_day = bars[j].session_date_ny
    hi, lo = bars[j].high, bars[j].low
    while j >= 0 and bars[j].session_date_ny == prev_day:
        hi = max(hi, bars[j].high)
        lo = min(lo, bars[j].low)
        j -= 1
    return hi, lo


# ── candidates ──────────────────────────────────────────────────────────────


def opening_range_break(bars: list[Bar], i: int, minutes: int = 30) -> int:
    """Close beyond the first 30 minutes' range, in the breakout direction."""
    hi, lo, complete = _opening_range(bars, i, minutes)
    if not complete or hi <= lo:
        return 0
    c = bars[i].close
    return 1 if c > hi else -1 if c < lo else 0


def opening_range_fade(bars: list[Bar], i: int, minutes: int = 30) -> int:
    """The mirror. Documented behaviour is that early breaks often fail."""
    return -opening_range_break(bars, i, minutes)


def gap_fade(bars: list[Bar], i: int, min_frac: float = 0.001) -> int:
    """Fade the overnight gap: short an up-gap, buy a down-gap.

    Only fires on the session's first few bars, since a gap is an opening
    phenomenon and reading it at noon is reading yesterday's news."""
    idxs = _session_bars(bars, i)
    if len(idxs) > 3:
        return 0
    prev = _prev_session_extremes(bars, i)
    if prev is None:
        return 0
    j = idxs[0] - 1
    if j < 0:
        return 0
    prev_close = bars[j].close
    open_px = bars[idxs[0]].open
    gap = (open_px - prev_close) / prev_close
    if abs(gap) < min_frac:
        return 0
    return -1 if gap > 0 else 1


def gap_continue(bars: list[Bar], i: int, min_frac: float = 0.001) -> int:
    return -gap_fade(bars, i, min_frac)


def prev_day_sweep_reclaim(bars: list[Bar], i: int) -> int:
    """The CISD kernel reduced to one rule: trade through yesterday's extreme
    and close back inside it.

    Worth testing on its own because CISD wrapped this in a hundred parameters
    and the wrapper may have been the problem rather than the premise."""
    prev = _prev_session_extremes(bars, i)
    if prev is None:
        return 0
    hi, lo = prev
    b = bars[i]
    if b.high > hi and b.close < hi:
        return -1
    if b.low < lo and b.close > lo:
        return 1
    return 0


def prev_day_break(bars: list[Bar], i: int) -> int:
    """The opposite premise: a close BEYOND yesterday's extreme continues."""
    prev = _prev_session_extremes(bars, i)
    if prev is None:
        return 0
    hi, lo = prev
    c = bars[i].close
    return 1 if c > hi else -1 if c < lo else 0


def momentum(bars: list[Bar], i: int, lookback: int = 6) -> int:
    """Sign of the net move over the last `lookback` bars."""
    if i < lookback:
        return 0
    if bars[i - lookback].session_date_ny != bars[i].session_date_ny:
        return 0
    move = bars[i].close - bars[i - lookback].close
    return 1 if move > 0 else -1 if move < 0 else 0


def mean_reversion(bars: list[Bar], i: int, lookback: int = 6) -> int:
    return -momentum(bars, i, lookback)


def range_expansion(bars: list[Bar], i: int, mult: float = 2.0, lookback: int = 20) -> int:
    """A bar whose range is unusually large, traded in its own direction."""
    if i < lookback:
        return 0
    prior = bars[i - lookback:i]
    if bars[i - lookback].session_date_ny != bars[i].session_date_ny:
        return 0
    avg = sum(b.range for b in prior) / len(prior)
    if avg <= 0 or bars[i].range < mult * avg:
        return 0
    return 1 if bars[i].is_up else -1 if bars[i].is_down else 0


def range_expansion_fade(bars: list[Bar], i: int, mult: float = 2.0, lookback: int = 20) -> int:
    return -range_expansion(bars, i, mult, lookback)


def first_bar_continuation(bars: list[Bar], i: int) -> int:
    """Follow the direction of the session's opening bar, entered later."""
    idxs = _session_bars(bars, i)
    if len(idxs) < 4 or len(idxs) > 6:
        return 0
    first = bars[idxs[0]]
    return 1 if first.is_up else -1 if first.is_down else 0


def inside_bar_break(bars: list[Bar], i: int) -> int:
    """Break of a bar that was entirely contained by its predecessor."""
    if i < 2:
        return 0
    a, b, c = bars[i - 2], bars[i - 1], bars[i]
    if a.session_date_ny != c.session_date_ny:
        return 0
    if not (b.high <= a.high and b.low >= a.low):
        return 0
    return 1 if c.close > b.high else -1 if c.close < b.low else 0


CANDIDATES = {
    "opening_range_break": opening_range_break,
    "opening_range_fade": opening_range_fade,
    "gap_fade": gap_fade,
    "gap_continue": gap_continue,
    "prev_day_sweep_reclaim": prev_day_sweep_reclaim,
    "prev_day_break": prev_day_break,
    "momentum_6": momentum,
    "mean_reversion_6": mean_reversion,
    "range_expansion": range_expansion,
    "range_expansion_fade": range_expansion_fade,
    "first_bar_continuation": first_bar_continuation,
    "inside_bar_break": inside_bar_break,
}
