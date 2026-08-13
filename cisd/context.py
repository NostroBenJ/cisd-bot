"""Market context the grader reads: dealing range, liquidity pools, fair value
gaps, opening gaps, and the draw on liquidity.

These are the "where are we" inputs, as opposed to `detect.py`'s "what just
happened". All of them are ports; the additions live in `gamma.py` and in the
volume/VWAP terms of `grade.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .bars import Bar
from .config import Config
from .sessions import in_session


# ── dealing range / premium-discount ────────────────────────────────────────


@dataclass
class DealingRange:
    """Previous day's and previous week's high/low, and where price sits in
    them. Port of the Pine's premium/discount section.

    "Discount" means below the equilibrium for a long, "premium" above it for a
    short -- i.e. favourable. Depth is blended 65/35 daily/weekly and raised to
    the power 1.5, which is the Pine's shaping: it makes shallow positioning
    count for much less than deep positioning rather than scaling linearly."""

    pdh: float = math.nan
    pdl: float = math.nan
    pwh: float = math.nan
    pwl: float = math.nan

    @property
    def daily_eq(self) -> float:
        return (self.pdh + self.pdl) / 2.0

    @property
    def daily_range(self) -> float:
        return self.pdh - self.pdl

    @property
    def weekly_eq(self) -> float:
        return (self.pwh + self.pwl) / 2.0

    @property
    def weekly_range(self) -> float:
        return self.pwh - self.pwl

    def ready(self) -> bool:
        return not (math.isnan(self.pdh) or math.isnan(self.pdl)) and self.daily_range > 0

    def favourable(self, price: float, bull: bool, eq_band_pct: float) -> bool:
        """Port of `f_favPd`. Is price on the helpful side of equilibrium?"""
        if not self.ready():
            return False
        band = self.daily_range * eq_band_pct
        eq = self.daily_eq
        return price < eq - band if bull else price > eq + band

    def depth(self, price: float, bull: bool) -> float:
        """Port of `f_pdDepth`. 0.0 at equilibrium, 1.0 at the extreme."""
        d = self._half_range_fraction(price, bull, self.daily_eq, self.daily_range)
        w = self._half_range_fraction(price, bull, self.weekly_eq, self.weekly_range)
        blended = min(max(d, 0.0), 1.0) * 0.65 + min(max(w, 0.0), 1.0) * 0.35
        return blended**1.5

    def deep_on_both(self, price: float, bull: bool) -> bool:
        """Port of `f_deepDualPd`. Unclamped, so it genuinely requires depth on
        the daily AND the weekly rather than saturating on one of them."""
        d = self._half_range_fraction(price, bull, self.daily_eq, self.daily_range)
        w = self._half_range_fraction(price, bull, self.weekly_eq, self.weekly_range)
        return d > 0.45 and w > 0.45

    @staticmethod
    def _half_range_fraction(price: float, bull: bool, eq: float, rng: float) -> float:
        if math.isnan(rng) or rng <= 0:
            return 0.0
        signed = (eq - price) if bull else (price - eq)
        return signed / (rng / 2.0)


# ── liquidity pools ─────────────────────────────────────────────────────────


@dataclass
class SessionRange:
    high: float = math.nan
    low: float = math.nan
    complete: bool = False

    def extend(self, bar: Bar) -> None:
        self.high = bar.high if math.isnan(self.high) else max(self.high, bar.high)
        self.low = bar.low if math.isnan(self.low) else min(self.low, bar.low)

    def reset(self, bar: Bar) -> None:
        self.high, self.low, self.complete = bar.high, bar.low, False


@dataclass
class LiquidityPools:
    """Tracks the levels the model treats as engineered liquidity, and whether
    they have just been swept.

    A sweep is: price traded through the level and then closed back inside it.
    Trading through without closing back is not a sweep -- it is a break, and
    the distinction is the entire premise of the rejection block."""

    cfg: Config
    ranges: dict[str, SessionRange] = field(default_factory=dict)
    hod: float = math.nan
    lod: float = math.nan
    _prev_hod: float = math.nan
    _prev_lod: float = math.nan
    _date: str | None = None
    _swept_high_at: int | None = None
    _swept_low_at: int | None = None
    _bar_index: int = 0

    def update(self, bar: Bar, index: int, dr: DealingRange) -> None:
        self._bar_index = index
        d = bar.session_date_ny
        if d != self._date:
            self._date = d
            self.hod, self.lod = bar.high, bar.low
            self._prev_hod, self._prev_lod = math.nan, math.nan
        else:
            self._prev_hod, self._prev_lod = self.hod, self.lod
            self.hod = max(self.hod, bar.high)
            self.lod = min(self.lod, bar.low)

        for name in ("asia", "london", "ny"):
            spec = _SESSION_SPECS[name]
            sr = self.ranges.setdefault(name, SessionRange())
            inside = in_session(bar, spec)
            if inside and not sr.complete and math.isnan(sr.high):
                sr.reset(bar)
            elif inside:
                sr.extend(bar)

        if self._sweep_high(bar, dr):
            self._swept_high_at = index
        if self._sweep_low(bar, dr):
            self._swept_low_at = index

    def _sweep_high(self, bar: Bar, dr: DealingRange) -> bool:
        levels = []
        if self.cfg.score_pd_sweeps and dr.ready():
            levels.append(dr.pdh)
        for sr in self.ranges.values():
            if not math.isnan(sr.high):
                levels.append(sr.high)
        if self.cfg.hunt_hod_lod and not math.isnan(self._prev_hod):
            levels.append(self._prev_hod)
        return any(bar.high > lvl and bar.close < lvl for lvl in levels)

    def _sweep_low(self, bar: Bar, dr: DealingRange) -> bool:
        levels = []
        if self.cfg.score_pd_sweeps and dr.ready():
            levels.append(dr.pdl)
        for sr in self.ranges.values():
            if not math.isnan(sr.low):
                levels.append(sr.low)
        if self.cfg.hunt_hod_lod and not math.isnan(self._prev_lod):
            levels.append(self._prev_lod)
        return any(bar.low < lvl and bar.close > lvl for lvl in levels)

    def recently_swept(self, bull: bool) -> bool:
        """A sweep in the direction that supports `bull`, inside the recency
        window. A bullish setup wants a swept LOW (liquidity taken below)."""
        at = self._swept_low_at if bull else self._swept_high_at
        if at is None:
            return False
        return (self._bar_index - at) <= self.cfg.sweep_recency_bars


_SESSION_SPECS = {"asia": "1800-0000", "london": "0300-0800", "ny": "0930-1600"}


# ── fair value gaps ─────────────────────────────────────────────────────────


@dataclass
class FairValueGap:
    top: float
    bottom: float
    direction: int  # +1 bullish, -1 bearish
    ob_top: float
    ob_bottom: float
    created_ts: int

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float, include_ob: bool) -> bool:
        if self.bottom <= price <= self.top:
            return True
        if include_ob and not math.isnan(self.ob_bottom):
            return self.ob_bottom <= price <= self.ob_top
        return False


def find_fvg(bars: list[Bar], i: int, min_points: float) -> FairValueGap | None:
    """Three-bar fair value gap ending at bar `i`.

    Bullish when bar `i`'s low sits above bar `i-2`'s high -- an untraded band
    the market skipped. The order block is bar `i-2`, the candle that initiated
    the move, which the Pine optionally folds into the zone."""
    if i < 2:
        return None
    a, c = bars[i - 2], bars[i]
    if c.low > a.high and (c.low - a.high) >= min_points:
        return FairValueGap(c.low, a.high, 1, a.high, a.low, c.ts)
    if c.high < a.low and (a.low - c.high) >= min_points:
        return FairValueGap(a.low, c.high, -1, a.high, a.low, c.ts)
    return None


@dataclass
class FvgStore:
    """Live higher-timeframe fair value gaps, pruned as they fill or go stale.

    A gap is filled when price closes beyond its far edge -- not merely touches
    it. Stale means price has walked far enough away (in ATR) that the level is
    no longer relevant; without that, the store fills with levels from days ago
    that quietly keep scoring."""

    cfg: Config
    gaps: list[FairValueGap] = field(default_factory=list)

    def add(self, gap: FairValueGap) -> None:
        self.gaps.append(gap)
        if len(self.gaps) > 40:
            self.gaps.pop(0)

    def prune(self, close: float, atr_value: float) -> None:
        if math.isnan(atr_value):
            return
        stale_distance = atr_value * self.cfg.fvg_stale_atr
        keep = []
        for g in self.gaps:
            lo = min(g.bottom, g.ob_bottom) if self.cfg.include_order_block else g.bottom
            hi = max(g.top, g.ob_top) if self.cfg.include_order_block else g.top
            filled = close < lo if g.direction == 1 else close > hi
            stale = abs(close - g.mid) > stale_distance
            if not filled and not stale:
                keep.append(g)
        self.gaps = keep

    def contains(self, price: float, bull: bool) -> bool:
        """Port of `f_ceInAoi`. Only same-direction gaps count."""
        want = 1 if bull else -1
        return any(
            g.direction == want and g.contains(price, self.cfg.include_order_block)
            for g in self.gaps
        )


# ── opening gaps ────────────────────────────────────────────────────────────


@dataclass
class OpeningGap:
    top: float
    bottom: float
    created_ts: int
    touched_top: bool = False
    touched_bottom: bool = False

    @property
    def ce(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def filled(self) -> bool:
        return self.touched_top and self.touched_bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class GapStore:
    cfg: Config
    gaps: list[OpeningGap] = field(default_factory=list)

    def add_open_gap(self, prev_close: float, open_price: float, ts: int) -> None:
        hi, lo = max(prev_close, open_price), min(prev_close, open_price)
        if (hi - lo) >= self.cfg.gap_min_points:
            self.gaps.append(OpeningGap(hi, lo, ts))

    def update(self, bar: Bar, atr_value: float) -> None:
        if math.isnan(atr_value):
            return
        stale_distance = atr_value * self.cfg.gap_stale_atr
        keep = []
        for g in self.gaps:
            g.touched_top = g.touched_top or bar.high >= g.top
            g.touched_bottom = g.touched_bottom or bar.low <= g.bottom
            if not g.filled and abs(bar.close - g.ce) <= stale_distance:
                keep.append(g)
        self.gaps = keep[-4:]

    def contains(self, price: float) -> bool:
        return any(g.contains(price) for g in self.gaps)


# ── draw on liquidity ───────────────────────────────────────────────────────


def draw_targets(
    price: float, dr: DealingRange, pools: LiquidityPools
) -> tuple[float | None, float | None]:
    """Nearest liquidity above and below -- where price is being drawn.

    Port of `f_minAbove` / `f_maxBelow`. Candidates are the previous day's and
    week's extremes plus completed session ranges. Incomplete sessions are
    excluded because a range that is still forming is not yet a target."""
    above: list[float] = []
    below: list[float] = []

    for lvl in (dr.pdh, dr.pwh):
        if not math.isnan(lvl) and lvl > price:
            above.append(lvl)
    for lvl in (dr.pdl, dr.pwl):
        if not math.isnan(lvl) and lvl < price:
            below.append(lvl)
    for sr in pools.ranges.values():
        if sr.complete:
            if not math.isnan(sr.high) and sr.high > price:
                above.append(sr.high)
            if not math.isnan(sr.low) and sr.low < price:
                below.append(sr.low)

    return (min(above) if above else None, max(below) if below else None)


def setup_type(
    bull: bool,
    ce: float,
    ext: float,
    close: float,
    bias_bull: bool | None,
    dr: DealingRange,
    pools: LiquidityPools,
    cfg: Config,
    atr_value: float,
) -> int:
    """0 = neither, 1 = continuation, 2 = reversal (terminal).

    Port of `f_setupType`. A reversal is a block whose swept extreme sits at a
    draw target -- price reached the liquidity it was seeking, so the move that
    got it there is finished. A continuation is a bias-aligned block in
    favourable premium/discount that has not yet reached its draw.

    Draw targets are measured from `close`, not from the block midpoint, which
    is what the Pine does (`dolUp`/`dolDn` are series computed from `close`).
    The distinction matters precisely in the textbook case: after a block sweeps
    the previous day's low, its midpoint sits BELOW that level, so measuring
    from the midpoint reclassifies the swept level as an upside draw and the
    setup silently stops qualifying as a reversal."""
    if not cfg.use_dol or math.isnan(atr_value):
        return 0
    band = atr_value * cfg.dol_band_atr
    up, down = draw_targets(close, dr, pools)

    if not bull and up is not None and ext >= up - band:
        return 2
    if bull and down is not None and ext <= down + band:
        return 2
    if bias_bull is None:
        return 0
    if bull and bias_bull and dr.favourable(ce, True, cfg.eq_band_pct):
        return 1
    if not bull and not bias_bull and dr.favourable(ce, False, cfg.eq_band_pct):
        return 1
    return 0
