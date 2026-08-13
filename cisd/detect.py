"""Rejection-block detection and the change-in-state-of-delivery trigger.

This is the heart of the port, so the correspondence to the Pine is spelled out
rather than left implicit.

The Pine evaluates candidate blocks at offset `o = 1` inside a
`request.security` call -- that is, on the *previous completed* higher-timeframe
bar. That offset is what keeps the original from repainting. Here the same
guarantee comes from iterating completed bars directly: `detect_rejection_block`
is only ever called with an index whose bar has closed, and it only reads bars
at or before that index. No function in this module looks forward.

The scan-back helper `_opposing_run` reproduces a subtlety in the original that
is easy to lose. Pine's loop does not stop at the first non-matching candle --
it skips leading non-matching candles, latches onto the first run it finds, and
only then stops. So `cisd_lookback = 4` does not mean "the run must be adjacent";
it means "find the first opposing run within four bars". That is reproduced
exactly, because changing it would change which signals fire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .bars import Bar
from .config import Config


@dataclass
class RejectionBlock:
    """A candidate zone, tracked from formation through signal or death.

    Naming follows the Pine so the two can be read side by side. `ce` is the
    consequent encroachment -- the midpoint, and the level price must return to.
    `ext` is the extreme (the swept wick), which is both the invalidation level
    and, for the bot, the stop."""

    timeframe: int
    bull: bool
    top: float
    bottom: float
    ce: float
    ext: float
    born_ts: int
    born_index: int
    cisd_level: float | None

    # formation qualities, fed to the grader
    strong_close: bool = False
    large_range: bool = False
    rvol: float = 1.0

    # lifecycle
    displacement_peak: float = 0.0
    displacement_extreme: float = 0.0
    confirmed: bool = False
    ever_left: bool = False
    armed: bool = False
    # Distinct from `armed`: a block can arm, disarm when price leaves again,
    # and re-arm. The counter in EngineStats tracks BLOCKS, not arming events,
    # so the funnel formed >= confirmed >= armed >= triggered stays a funnel.
    ever_armed: bool = False
    tapped: bool = False
    active: bool = True

    # scoring, filled at formation and refreshed at trigger
    score: float = 0.0
    stack: int = 0
    dol_type: int = 0
    smt_ok: bool = False
    amdx: bool = False
    reasons: dict[str, float] = field(default_factory=dict)

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def contains(self, bar: Bar) -> bool:
        """True when the bar's range overlaps the zone at all."""
        return bar.low <= self.top and bar.high >= self.bottom

    def invalidated_by(self, bar: Bar, buffer: float) -> bool:
        """Price traded through the extreme -- the block is wrong."""
        if self.bull:
            return bar.low < self.ext - buffer
        return bar.high > self.ext + buffer


def _opposing_run(
    bars: list[Bar], i: int, look: int, want_down: bool
) -> tuple[float | None, int, int | None]:
    """Find the first contiguous opposing run scanning back from `i - 1`.

    Returns (open of the run's OLDEST bar, run length, index of that bar).

    The open of the oldest bar is the whole point -- it is the price at which
    the opposing delivery series *began*, and closing back through it is what
    ICT calls the change in state. Implementations that take the open of the
    most recent bar in the run are testing something much weaker, and this is
    the single most common way the concept is coded wrong.

    Leading non-matching bars are skipped rather than terminating the scan,
    matching the Pine."""
    level: float | None = None
    length = 0
    start_idx: int | None = None
    started = False

    for k in range(1, look + 1):
        j = i - k
        if j < 0:
            break
        b = bars[j]
        matches = b.is_down if want_down else b.is_up
        if matches:
            started = True
            level = b.open
            length += 1
            start_idx = j
        elif started:
            break
    return level, length, start_idx


def _run_holds_extreme(
    bars: list[Bar], i: int, look: int, bull: bool, start_idx: int, length: int
) -> bool:
    """Did the opposing run actually drive price into the swing extreme?

    Canonical ICT anchors the change in state to the run that produced the most
    recent swing high or low. The Pine does not check this -- it accepts any
    recent opposing run, which fires on interior chop that never reached an
    extreme. Enforced only when `cisd_mode == "swing_anchored"`."""
    lo = max(0, i - look)
    window = bars[lo:i]
    if not window:
        return False
    if bull:
        extreme_off = min(range(len(window)), key=lambda k: window[k].low)
    else:
        extreme_off = max(range(len(window)), key=lambda k: window[k].high)
    extreme_idx = lo + extreme_off
    return start_idx <= extreme_idx <= start_idx + length - 1


def cisd_close(
    bars: list[Bar], i: int, bull: bool, cfg: Config
) -> tuple[bool, float | None]:
    """Did bar `i` close through the initiating open of the opposing run?

    A bullish change in state needs a preceding DOWN run whose opening price
    the current bar closes above; bearish is the mirror."""
    level, length, start_idx = _opposing_run(bars, i, cfg.cisd_lookback, want_down=bull)
    if level is None or start_idx is None or length < cfg.cisd_min_run:
        return False, None

    if cfg.cisd_mode == "swing_anchored" and not _run_holds_extreme(
        bars, i, cfg.cisd_lookback, bull, start_idx, length
    ):
        return False, level

    close = bars[i].close
    fired = close > level if bull else close < level
    return fired, level


def cisd_direction(bars: list[Bar], i: int, cfg: Config) -> int:
    """+1, -1 or 0 for the change in state at bar `i`.

    Used for the top-down bias. Unlike the Pine's `f_cisdState`, this returns
    0 when nothing fired on this bar rather than latching the last direction
    forever -- the ageing is the caller's job (`engine.BiasTracker`), which is
    what lets an unset bias be genuinely neutral instead of defaulting long."""
    look = cfg.bias_cisd_lookback
    bull_fired, _ = cisd_close(bars, i, True, _with_lookback(cfg, look))
    if bull_fired:
        return 1
    bear_fired, _ = cisd_close(bars, i, False, _with_lookback(cfg, look))
    if bear_fired:
        return -1
    return 0


def _with_lookback(cfg: Config, look: int) -> Config:
    from dataclasses import replace

    return replace(cfg, cisd_lookback=look)


def detect_rejection_block(
    bars: list[Bar],
    i: int,
    cfg: Config,
    atr_value: float,
    timeframe: int,
    rvol: float = 1.0,
) -> RejectionBlock | None:
    """Evaluate bar `i` as a rejection block. Returns None when it is not one.

    Port of Pine `f_rb`. The formation test is:

      1. real body (>= `min_body_fraction` of range),
      2. the bar swept a prior swing extreme by at least the sweep threshold,
      3. it closed back on the correct side (reclaim, optional),
      4. it left a wick beyond the body on the swept side,
      5. it closed strongly away from the swept extreme (optional).

    A bull block sits between the swept low and the body bottom; a bear block
    between the body top and the swept high. `ext` is the wick extreme, which
    becomes the stop."""
    need = max(cfg.swing_lookback, cfg.cisd_lookback) + 1
    if i < need or i >= len(bars):
        return None
    if math.isnan(atr_value):
        return None

    b = bars[i]
    if b.range <= 0 or b.body_fraction < cfg.min_body_fraction:
        return None

    prior = bars[i - cfg.swing_lookback : i]
    if not prior:
        return None
    prior_low = min(x.low for x in prior)
    prior_high = max(x.high for x in prior)

    sweep_threshold = atr_value * cfg.sweep_atr_mult if cfg.require_min_sweep else 0.0
    large_range = b.range >= atr_value * 0.8

    if b.is_up:
        swept = b.low < prior_low and (prior_low - b.low) >= sweep_threshold
        reclaimed = (not cfg.require_reclaim) or b.close > prior_low
        has_wick = b.body_bottom > b.low
        strong = b.close_position() >= cfg.strong_close_pct
        if not (swept and reclaimed and has_wick):
            return None
        if cfg.require_strong_close and not strong:
            return None
        level, _, _ = _opposing_run(bars, i, cfg.cisd_lookback, want_down=True)
        return RejectionBlock(
            timeframe=timeframe,
            bull=True,
            top=b.body_bottom,
            bottom=b.low,
            ce=(b.body_bottom + b.low) / 2.0,
            ext=b.low,
            born_ts=b.ts,
            born_index=i,
            cisd_level=level if level is not None else b.open,
            strong_close=strong,
            large_range=large_range,
            rvol=rvol,
            # Pine seeds peak at the midpoint and dispExt at the zone edge, NOT
            # at the formation bar's own extreme. Seeding from the bar's high
            # counts the formation candle itself as displacement, which
            # confirms almost every block instantly.
            displacement_peak=(b.body_bottom + b.low) / 2.0,
            displacement_extreme=b.body_bottom,
        )

    if b.is_down:
        swept = b.high > prior_high and (b.high - prior_high) >= sweep_threshold
        reclaimed = (not cfg.require_reclaim) or b.close < prior_high
        has_wick = b.high > b.body_top
        strong = (1.0 - b.close_position()) >= cfg.strong_close_pct
        if not (swept and reclaimed and has_wick):
            return None
        if cfg.require_strong_close and not strong:
            return None
        level, _, _ = _opposing_run(bars, i, cfg.cisd_lookback, want_down=False)
        return RejectionBlock(
            timeframe=timeframe,
            bull=False,
            top=b.high,
            bottom=b.body_top,
            ce=(b.high + b.body_top) / 2.0,
            ext=b.high,
            born_ts=b.ts,
            born_index=i,
            cisd_level=level if level is not None else b.open,
            strong_close=strong,
            large_range=large_range,
            rvol=rvol,
            displacement_peak=(b.high + b.body_top) / 2.0,
            displacement_extreme=b.body_top,
        )

    return None
