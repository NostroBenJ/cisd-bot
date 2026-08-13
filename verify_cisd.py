"""Verification suite for the CISD port.

Run: `python verify_cisd.py`

The house rule is that a formula has to earn trust numerically rather than by
assertion, so nothing here is a regression test against a number someone once
printed. Each check is either a hand-computed value, a structural identity the
model implies, or a property that must hold for any correct implementation.

The three that matter most:

  [9]  **Prefix stability.** Running the engine on the first N bars and on the
       first N+K bars must produce IDENTICAL signals over the shared prefix. If
       it does not, something reads the future, the backtest is fiction, and
       the live bot will behave differently from its own history. This is the
       single check that separates a tradeable ICT model from a beautiful one.

  [10] **Mirror symmetry.** Reflect every price around a constant and swap
       highs with lows: every long must become a short at the same bar with the
       same score. Any asymmetry is a bug in exactly one branch, which is the
       failure mode that hides longest because half the code is tested twice
       and half never.

  [11] **Gate monotonicity.** Tightening a gate can never produce more signals.
       Sounds trivial; catches state that leaks between blocks.
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
from dataclasses import replace
from datetime import datetime, timedelta

from cisd.bars import NY, Bar, bucket_index, resample
from cisd.config import Config, pine_bug_compat
from cisd.context import DealingRange, find_fvg
from cisd.detect import _opposing_run, cisd_close, detect_rejection_block
from cisd.engine import CisdEngine, build_period_ranges
from cisd.indicators import atr, pivot_high, pivot_low, relative_volume, rma, session_vwap, true_range
from cisd.sessions import EventCalendar, in_window, parse_session
from cisd.signal import Signal, build_targets

PASS, FAIL = 0, 0
TOL = 1e-9


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def close_to(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ── synthetic data ──────────────────────────────────────────────────────────


def _noise_bar(ts: int, price: float, rng: random.Random, vol_mult: float = 1.0) -> Bar:
    c = price + rng.gauss(0, 0.07)
    o = price
    hi = max(o, c) + abs(rng.gauss(0, 0.05))
    lo = min(o, c) - abs(rng.gauss(0, 0.05))
    return Bar(ts, o, hi, lo, c, (40_000 + rng.random() * 40_000) * vol_mult)


def _scripted_setup(t0: datetime, p: float, prior_low: float, bull: bool) -> list[Bar]:
    """The textbook sequence, constructed so the engine MUST find it.

    A pure random walk essentially never produces sweep -> displace -> leave ->
    return -> close-through in order, so testing the engine on noise alone
    passes every assertion on zero signals. This builds the canonical pattern
    explicitly, which turns the engine tests into real assertions: if the model
    cannot find a textbook setup, it is broken.

    Two constraints are non-obvious and both are properties of the model rather
    than of this generator:

    * The sweep bar's body must still be >= `min_body_fraction` of its FULL
      range, so a deeper wick demands a proportionally larger body. A rejection
      block is not a pin bar.
    * The opposing run before the trigger must be SHORT. The change-in-state
      level is the open of the run's oldest bar, so a five-bar run puts that
      level implausibly far away and the close can never reach it. Two bars,
      preceded by an opposite-coloured bar that terminates the backward scan,
      is what actually fires.

    Mirrored for `bull=False` so both branches are exercised by construction.
    """
    s = 1.0 if bull else -1.0

    def mk(minute: int, o: float, h: float, l: float, c: float, v: float) -> Bar:
        ts = int((t0 + timedelta(minutes=minute)).timestamp())
        return Bar(ts, o, h, l, c, v)

    bars: list[Bar] = []
    m = 0

    # 1 · drift toward the prior extreme, establishing a local range
    for k in range(10):
        lvl = p + s * 0.04 * (9 - k)
        bars.append(mk(m, lvl, lvl + 0.06, lvl - 0.06, lvl - s * 0.02, 50_000))
        m += 1

    # 2 · the sweep: takes out `prior_low` and closes hard back inside
    sweep_ext = prior_low - s * 0.45
    o = p - s * 0.35
    c = p + s * 0.25
    far = p + s * 0.30
    bars.append(mk(m, o,
                   max(o, c, far) if bull else max(o, c, sweep_ext),
                   min(o, c, sweep_ext) if bull else min(o, c, far),
                   c, 180_000))
    m += 1

    # 3 · displacement away from the zone
    for k in range(6):
        lvl = p + s * (0.30 + 0.12 * k)
        bars.append(mk(m, lvl, lvl + s * 0.10 if bull else lvl + 0.04,
                       lvl - 0.04 if bull else lvl + s * 0.10, lvl + s * 0.10, 90_000))
        m += 1

    # 4 · one counter-coloured bar, so the backward scan stops after two
    lvl = p + s * 0.95
    bars.append(mk(m, lvl - s * 0.05, max(lvl, lvl - s * 0.05) + 0.03,
                   min(lvl, lvl - s * 0.05) - 0.03, lvl, 70_000))
    m += 1

    # 5 · the two-bar retrace back into the zone -> armed
    r1_open = p + s * 0.10
    r1_close = p - s * 0.15
    bars.append(mk(m, r1_open, max(r1_open, r1_close) + 0.03,
                   min(r1_open, r1_close) - 0.03, r1_close, 80_000))
    m += 1
    r2_open = r1_close
    r2_close = p - s * 0.45
    r2_far = p - s * 0.55
    bars.append(mk(m, r2_open, max(r2_open, r2_close, r2_far),
                   min(r2_open, r2_close, r2_far), r2_close, 95_000))
    m += 1

    # 6 · the change-in-state close, back through r1's open
    c_open = r2_close
    c_close = p + s * 0.15
    c_far = p - s * 0.50
    bars.append(mk(m, c_open, max(c_open, c_close, c_far),
                   min(c_open, c_close, c_far), c_close, 200_000))
    return bars


def make_bars(n_days: int = 12, seed: int = 7, scripted: bool = True) -> list[Bar]:
    """Deterministic intraday bars: noise, with a scripted setup each morning.

    Setups alternate long and short day to day, and each one sweeps the
    previous session's extreme -- which is both the textbook context and what
    puts price deep enough in the dealing range for the premium/discount term
    to carry real weight."""
    rng = random.Random(seed)
    bars: list[Bar] = []
    price = 500.0
    day0 = datetime(2026, 3, 2, 9, 30, tzinfo=NY)
    prev_low = price - 2.0
    prev_high = price + 2.0
    setup_index = 0

    d = 0
    made = 0
    while made < n_days:
        start = day0 + timedelta(days=d)
        d += 1
        if start.weekday() >= 5:
            continue
        made += 1

        day_bars: list[Bar] = []
        bull = (setup_index % 2 == 0)
        # The setup must NOT land on the same minute every day. Relative volume
        # compares a bar against the same minute-of-day in prior sessions, so a
        # fixed schedule teaches it that a 200k-volume bar is normal at 09:45
        # and the volume gate then correctly rejects every scripted signal. The
        # jitter is a property of realistic data, not a workaround.
        lead = 15 + (setup_index * 7) % 40
        setup_index += 1

        for m in range(lead):
            ts = int((start + timedelta(minutes=m)).timestamp())
            price += rng.gauss(0, 0.05)
            day_bars.append(_noise_bar(ts, price, rng, 2.0 if m < 5 else 1.0))
            price = day_bars[-1].close

        if scripted:
            # walk price to just inside the extreme we are about to sweep
            anchor = (prev_low + 0.55) if bull else (prev_high - 0.55)
            seq = _scripted_setup(start + timedelta(minutes=lead), anchor,
                                  prev_low if bull else prev_high, bull)
            day_bars.extend(seq)
            price = seq[-1].close
            m = lead + len(seq)
        else:
            m = lead

        # remainder of the session as noise
        while m < 390:
            ts = int((start + timedelta(minutes=m)).timestamp())
            price += rng.gauss(0, 0.06)
            day_bars.append(_noise_bar(ts, price, rng, 2.0 if m > 360 else 1.0))
            price = day_bars[-1].close
            m += 1

        prev_low = min(b.low for b in day_bars)
        prev_high = max(b.high for b in day_bars)
        bars.extend(day_bars)

    return bars


def mirror(bars: list[Bar], centre: float = 1000.0) -> list[Bar]:
    """Reflect prices around `centre`, swapping highs and lows."""
    return [
        Bar(b.ts, 2 * centre - b.open, 2 * centre - b.low, 2 * centre - b.high,
            2 * centre - b.close, b.volume)
        for b in bars
    ]


def bar(ts: int, o: float, h: float, l: float, c: float, v: float = 1000.0) -> Bar:
    return Bar(ts, o, h, l, c, v)


# ── [1] bar aggregation ─────────────────────────────────────────────────────


def t1_aggregation() -> None:
    print("\n[1] bar aggregation")
    base = make_bars(n_days=3)
    five = resample(base, 5)
    fifteen = resample(base, 15)

    check("volume is conserved", close_to(sum(b.volume for b in five), sum(b.volume for b in base), 1e-6))
    check("5m bar count is 1/5 of 1m", abs(len(five) - len(base) / 5) <= 2,
          f"{len(five)} vs {len(base)/5:.0f}")

    ok = all(b.high >= max(b.open, b.close) and b.low <= min(b.open, b.close) for b in five)
    check("OHLC invariants hold after aggregation", ok)

    # A 5-minute bucket must equal the aggregate of exactly its 1-minute bars.
    first = five[0]
    members = [b for b in base if bucket_index(b, 5) == bucket_index(base[0], 5)]
    check("first 5m bar equals its members", (
        close_to(first.open, members[0].open)
        and close_to(first.close, members[-1].close)
        and close_to(first.high, max(m.high for m in members))
        and close_to(first.low, min(m.low for m in members))
    ))

    # Session anchoring: 09:30 must open a bucket at every timeframe, and a
    # 60-minute SPY bar must run 09:30-10:30, not 09:00-10:00.
    hourly = resample(base, 60)
    opens = {b.dt_ny.strftime("%H:%M") for b in hourly}
    check("60m bars anchor to 09:30, not the clock hour", "09:30" in opens and "10:30" in opens,
          f"got {sorted(opens)[:4]}")

    # Aggregating twice must equal aggregating once.
    via_five = resample(five, 15)
    direct = fifteen
    same = len(via_five) == len(direct) and all(
        close_to(a.open, b.open) and close_to(a.high, b.high)
        and close_to(a.low, b.low) and close_to(a.close, b.close)
        for a, b in zip(via_five, direct)
    )
    check("1m->5m->15m equals 1m->15m", same)


# ── [2] ATR / RMA against hand computation ──────────────────────────────────


def t2_atr() -> None:
    print("\n[2] ATR and Wilder smoothing")

    # rma with length 1 is a passthrough: alpha = 1, so out[i] == values[i].
    vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    r1 = rma(vals, 1)
    check("rma(length=1) is the identity", all(close_to(a, b) for a, b in zip(r1, vals)))

    # Hand-computed Wilder: seed = SMA of first 3, then alpha = 1/3.
    r3 = rma(vals, 3)
    seed = (3.0 + 1.0 + 4.0) / 3.0
    expect = [math.nan, math.nan, seed]
    prev = seed
    for v in vals[3:]:
        prev = v / 3.0 + prev * 2.0 / 3.0
        expect.append(prev)
    ok = all(
        (math.isnan(a) and math.isnan(b)) or close_to(a, b, 1e-12)
        for a, b in zip(r3, expect)
    )
    check("rma matches hand-computed Wilder smoothing", ok, f"{r3} vs {expect}")

    check("rma is nan before the seed", math.isnan(r3[0]) and math.isnan(r3[1]))

    # True range on a known series.
    bs = [bar(0, 10, 12, 9, 11), bar(60, 11, 15, 10, 14), bar(120, 14, 14.5, 8, 9)]
    tr = true_range(bs)
    check("TR[0] is the bar's own range", close_to(tr[0], 3.0))
    check("TR[1] = high-low when it dominates", close_to(tr[1], 5.0))
    # bar 3: high-low = 6.5, |high-prevclose| = 0.5, |low-prevclose| = 6.0 -> 6.5
    check("TR[2] takes the max of the three legs", close_to(tr[2], 6.5))

    # A constant-range series has ATR equal to that range once seeded.
    flat = [bar(i * 60, 100, 101, 99, 100) for i in range(40)]
    a = atr(flat, 14)
    check("ATR of a constant-range series equals that range", close_to(a[-1], 2.0, 1e-9),
          f"{a[-1]}")


# ── [3] pivots and lookahead ────────────────────────────────────────────────


def t3_pivots() -> None:
    print("\n[3] pivots")
    # Peak at index 3. With left=right=2 it confirms at index 5.
    highs = [1, 2, 3, 9, 3, 2, 1, 2, 3]
    bs = [bar(i * 60, h, h, h - 1, h) for i, h in enumerate(highs)]
    ph = pivot_high(bs, 2, 2)
    check("pivot confirms exactly `right` bars late", ph[5] == 9.0 and ph[4] is None,
          f"ph={ph}")
    check("no pivot reported before confirmation", all(p is None for p in ph[:5]))

    lows = [9, 8, 7, 1, 7, 8, 9]
    bs2 = [bar(i * 60, l + 1, l + 1, l, l) for i, l in enumerate(lows)]
    pl = pivot_low(bs2, 2, 2)
    check("pivot_low mirrors pivot_high", pl[5] == 1.0, f"pl={pl}")

    # Truncating the series must not change already-confirmed pivots.
    full = pivot_high(bs, 2, 2)
    part = pivot_high(bs[:7], 2, 2)
    check("pivots are stable under truncation", all(
        full[i] == part[i] for i in range(7)
    ))


# ── [4] the CISD trigger ────────────────────────────────────────────────────


def t4_cisd() -> None:
    print("\n[4] change in state of delivery")
    cfg = Config(cisd_lookback=4, cisd_min_run=2)

    # Three down bars (opens 110, 108, 106) then an up bar closing above 110.
    # The initiating open is the OLDEST bar's open: 110, not 106.
    bs = [
        bar(0, 110, 110.5, 107, 108),
        bar(60, 108, 108.5, 105, 106),
        bar(120, 106, 106.5, 103, 104),
        bar(180, 104, 111, 103.5, 110.5),
    ]
    level, length, start = _opposing_run(bs, 3, 4, want_down=True)
    check("run level is the OLDEST bar's open", close_to(level, 110.0), f"got {level}")
    check("run length counts every bar in the run", length == 3, f"got {length}")
    check("run start index is the oldest bar", start == 0, f"got {start}")

    fired, lvl = cisd_close(bs, 3, True, cfg)
    check("bullish CISD fires on close through the initiating open", fired and close_to(lvl, 110.0))

    # Closing above the most recent open but below the initiating one must NOT
    # fire -- this is the common mis-implementation.
    bs_short = list(bs[:3]) + [bar(180, 104, 107.5, 103.5, 107.0)]
    fired2, _ = cisd_close(bs_short, 3, True, cfg)
    check("does not fire on the most recent open alone", not fired2)

    # Run shorter than min_run is rejected.
    bs_one = [bar(0, 100, 101, 99, 100.5), bar(60, 110, 110.5, 107, 108), bar(120, 108, 112, 107, 111)]
    fired3, _ = cisd_close(bs_one, 2, True, Config(cisd_lookback=4, cisd_min_run=2))
    check("run shorter than min_run does not fire", not fired3)

    # Leading non-matching bars are SKIPPED, not treated as terminating --
    # this reproduces the Pine and is load-bearing for signal parity.
    bs_gap = [
        bar(0, 110, 110.5, 107, 108),   # down
        bar(60, 108, 108.5, 105, 106),  # down
        bar(120, 106, 112, 105.5, 111),  # up  <- interposed
        bar(180, 111, 113, 110, 112.5),  # up  (current)
    ]
    lvl_gap, len_gap, _ = _opposing_run(bs_gap, 3, 4, want_down=True)
    check("scan skips leading non-matching bars", close_to(lvl_gap, 110.0) and len_gap == 2,
          f"level={lvl_gap} len={len_gap}")

    # Bearish is the exact mirror.
    up = [
        bar(0, 100, 103, 99.5, 102),
        bar(60, 102, 105, 101.5, 104),
        bar(120, 104, 106, 99, 99.5),
    ]
    fired_b, lvl_b = cisd_close(up, 2, False, cfg)
    check("bearish CISD mirrors bullish", fired_b and close_to(lvl_b, 100.0), f"{fired_b} {lvl_b}")


# ── [5] rejection-block formation, both branches ────────────────────────────


def t5_rejection_block() -> None:
    print("\n[5] rejection-block formation")
    cfg = Config(swing_lookback=5, cisd_lookback=3, min_body_fraction=0.3,
                 require_min_sweep=False, require_strong_close=True, strong_close_pct=0.45)

    base = [bar(i * 60, 100, 100.5, 99.5, 100) for i in range(8)]

    # Bullish: sweeps below the prior low (99.5), closes strongly back up.
    # Note the body must still be >= min_body_fraction of the FULL range -- a
    # rejection block is a wick plus a real body, not a pin bar. That coupling
    # is easy to miss and it is why the sweep cannot be arbitrarily deep.
    bull_bar = bar(8 * 60, 99.0, 100.5, 98.0, 100.3)
    bs = base + [bull_bar]
    blk = detect_rejection_block(bs, 8, cfg, atr_value=1.0, timeframe=1)
    check("bull block detected on a sweep-and-reclaim", blk is not None and blk.bull)
    if blk:
        check("bull block extreme is the swept low", close_to(blk.ext, 98.0))
        check("bull block top is the body bottom", close_to(blk.top, 99.0))
        check("midpoint is between body bottom and low", close_to(blk.ce, (99.0 + 98.0) / 2))

    # Bearish: sweeps above the prior high (100.5), closes strongly back down.
    bear_bar = bar(8 * 60, 101.0, 102.0, 99.5, 99.7)
    bs2 = base + [bear_bar]
    blk2 = detect_rejection_block(bs2, 8, cfg, atr_value=1.0, timeframe=1)
    check("bear block detected on a sweep-and-reclaim", blk2 is not None and not blk2.bull)
    if blk2:
        check("bear block extreme is the swept high", close_to(blk2.ext, 102.0))
        check("bear block bottom is the body top", close_to(blk2.bottom, 101.0))

    # A weak close must be rejected when strong close is required.
    weak = bar(8 * 60, 98.2, 100.5, 98.0, 99.0)
    blk3 = detect_rejection_block(base + [weak], 8, cfg, atr_value=1.0, timeframe=1)
    check("weak close is rejected", blk3 is None)

    # No sweep -> no block.
    nosweep = bar(8 * 60, 100.0, 100.3, 99.8, 100.2)
    blk4 = detect_rejection_block(base + [nosweep], 8, cfg, atr_value=1.0, timeframe=1)
    check("bar that sweeps nothing is not a block", blk4 is None)

    # A doji has no body and cannot be a block.
    doji = bar(8 * 60, 100.0, 100.4, 98.0, 100.0)
    blk5 = detect_rejection_block(base + [doji], 8, cfg, atr_value=1.0, timeframe=1)
    check("doji is not a block", blk5 is None)

    # Sweep threshold actually gates: a 0.05 probe fails a 0.5 requirement.
    cfg_thr = replace(cfg, require_min_sweep=True, sweep_atr_mult=0.5)
    shallow = bar(8 * 60, 99.6, 100.4, 99.45, 100.3)
    blk6 = detect_rejection_block(base + [shallow], 8, cfg_thr, atr_value=1.0, timeframe=1)
    check("shallow sweep fails the distance gate", blk6 is None)


# ── [6] premium/discount and dealing range ──────────────────────────────────


def t6_dealing_range() -> None:
    print("\n[6] dealing range")
    dr = DealingRange(pdh=110.0, pdl=90.0, pwh=120.0, pwl=80.0)
    check("equilibrium is the midpoint", close_to(dr.daily_eq, 100.0))

    check("depth is zero at equilibrium", close_to(dr.depth(100.0, True), 0.0))
    check("depth is one at the daily and weekly low", close_to(dr.depth(80.0, True), 1.0, 1e-9),
          f"{dr.depth(80.0, True)}")

    # Symmetry: a long at X below equilibrium must score the same depth as a
    # short at the mirrored point above it.
    check("depth is symmetric about equilibrium",
          close_to(dr.depth(95.0, True), dr.depth(105.0, False)))

    check("discount is favourable for longs", dr.favourable(95.0, True, 0.10))
    check("discount is not favourable for shorts", not dr.favourable(95.0, False, 0.10))
    check("equilibrium band suppresses both sides", not dr.favourable(100.5, True, 0.10)
          and not dr.favourable(100.5, False, 0.10))

    check("deep_on_both needs depth on daily AND weekly", not dr.deep_on_both(94.0, True))
    check("deep_on_both fires when both are deep", dr.deep_on_both(84.0, True))

    empty = DealingRange()
    check("unset range is never favourable", not empty.favourable(100.0, True, 0.1))
    check("unset range has zero depth", close_to(empty.depth(100.0, True), 0.0))


# ── [7] VWAP against a direct weighted computation ──────────────────────────


def t7_vwap() -> None:
    print("\n[7] VWAP")
    bs = [
        bar(int(datetime(2026, 3, 2, 9, 30, tzinfo=NY).timestamp()), 100, 101, 99, 100, 1000),
        bar(int(datetime(2026, 3, 2, 9, 31, tzinfo=NY).timestamp()), 100, 103, 100, 102, 3000),
    ]
    vw, sd = session_vwap(bs)

    tp0 = (101 + 99 + 100) / 3
    tp1 = (103 + 100 + 102) / 3
    expect = (tp0 * 1000 + tp1 * 3000) / 4000
    check("VWAP equals the volume-weighted typical price", close_to(vw[1], expect, 1e-9),
          f"{vw[1]} vs {expect}")

    # Weighted standard deviation, computed the long way as an independent check.
    mean = expect
    var = (1000 * (tp0 - mean) ** 2 + 3000 * (tp1 - mean) ** 2) / 4000
    check("VWAP band matches a direct weighted stdev", close_to(sd[1], math.sqrt(var), 1e-7),
          f"{sd[1]} vs {math.sqrt(var)}")

    # Constant price -> zero dispersion, and no negative-variance sqrt blowup.
    flat = [bar(int(datetime(2026, 3, 2, 9, 30 + i, tzinfo=NY).timestamp()), 100, 100, 100, 100, 500)
            for i in range(30)]
    vwf, sdf = session_vwap(flat)
    check("constant price gives zero VWAP dispersion", close_to(sdf[-1], 0.0, 1e-9), f"{sdf[-1]}")
    check("constant price gives VWAP equal to price", close_to(vwf[-1], 100.0))

    # VWAP resets across sessions.
    d2 = [bar(int(datetime(2026, 3, 3, 9, 30, tzinfo=NY).timestamp()), 200, 200, 200, 200, 500)]
    vw2, _ = session_vwap(flat + d2)
    check("VWAP resets at the session boundary", close_to(vw2[-1], 200.0), f"{vw2[-1]}")


# ── [8] relative volume ─────────────────────────────────────────────────────


def t8_rvol() -> None:
    print("\n[8] relative volume")
    bs: list[Bar] = []
    for day in range(6):
        for m in range(3):
            ts = int((datetime(2026, 3, 2, 9, 30, tzinfo=NY) + timedelta(days=day, minutes=m)).timestamp())
            # Minute 0 is always heavy, minute 2 always light. A flat average
            # would mark every minute-0 bar exceptional; a same-minute
            # comparison must not.
            vol = 100_000 if m == 0 else 20_000
            bs.append(bar(ts, 100, 101, 99, 100, vol))
    rv = relative_volume(bs, 20)

    last_open = rv[-3]
    last_quiet = rv[-1]
    check("stable volume gives rvol near 1.0 at the open", close_to(last_open, 1.0, 1e-6),
          f"{last_open}")
    check("stable volume gives rvol near 1.0 at a quiet minute", close_to(last_quiet, 1.0, 1e-6),
          f"{last_quiet}")
    check("time-of-day is normalised away", close_to(last_open, last_quiet, 1e-6))

    # A genuine spike should register as a multiple.
    ts = int((datetime(2026, 3, 2, 9, 30, tzinfo=NY) + timedelta(days=6)).timestamp())
    spiked = bs + [bar(ts, 100, 101, 99, 100, 300_000)]
    rv2 = relative_volume(spiked, 20)
    check("a 3x volume bar reports ~3x", close_to(rv2[-1], 3.0, 1e-6), f"{rv2[-1]}")


# ── [9] prefix stability: the no-lookahead proof ────────────────────────────


def t9_prefix_stability() -> None:
    print("\n[9] prefix stability (no lookahead)")
    bars = make_bars(n_days=10)
    cutoff = int(len(bars) * 0.7)

    full = CisdEngine(Config(), symbol="SPY").run(bars)
    part = CisdEngine(Config(), symbol="SPY").run(bars[:cutoff])

    cut_ts = bars[cutoff - 1].ts
    full_prefix = [s for s in full if s.ts <= cut_ts]

    check("engine produced signals to test", len(full) > 0, f"{len(full)} signals")
    check("same signal count over the shared prefix",
          len(full_prefix) == len(part), f"full={len(full_prefix)} part={len(part)}")

    if len(full_prefix) == len(part):
        mismatches = [
            (a.ts, a.direction, b.direction, a.score, b.score)
            for a, b in zip(full_prefix, part)
            if a.ts != b.ts or a.direction != b.direction
            or not close_to(a.score, b.score, 1e-9)
            or not close_to(a.entry, b.entry, 1e-9)
            or not close_to(a.stop, b.stop, 1e-9)
        ]
        check("every prefix signal is byte-identical", not mismatches,
              f"{len(mismatches)} differ, first={mismatches[0] if mismatches else None}")

    # Second cut, to catch a boundary that happens to be clean.
    cut2 = int(len(bars) * 0.45)
    part2 = CisdEngine(Config(), symbol="SPY").run(bars[:cut2])
    prefix2 = [s for s in full if s.ts <= bars[cut2 - 1].ts]
    check("stable at a second, independent cut",
          len(prefix2) == len(part2), f"full={len(prefix2)} part={len(part2)}")


# ── [10] mirror symmetry: long and short are the same code ──────────────────


def t10_mirror_symmetry() -> None:
    print("\n[10] mirror symmetry (long/short parity)")
    cfg = replace(Config(), use_gamma=False, use_smt=False)
    bars = make_bars(n_days=10)
    flipped = mirror(bars, centre=1000.0)

    a = CisdEngine(cfg, symbol="SPY").run(bars)
    b = CisdEngine(cfg, symbol="SPY").run(flipped)

    check("mirrored run produces the same signal count",
          len(a) == len(b), f"{len(a)} vs {len(b)}")

    if len(a) == len(b) and a:
        bad = []
        for x, y in zip(a, b):
            if x.ts != y.ts:
                bad.append(("ts", x.ts, y.ts))
            elif x.direction == y.direction:
                bad.append(("direction not flipped", x.direction, y.direction))
            elif not close_to(x.score, y.score, 1e-6):
                bad.append(("score", x.score, y.score))
            elif not close_to(x.entry, 2000.0 - y.entry, 1e-6):
                bad.append(("entry", x.entry, 2000.0 - y.entry))
            elif not close_to(x.stop, 2000.0 - y.stop, 1e-6):
                bad.append(("stop", x.stop, 2000.0 - y.stop))
        check("every mirrored signal flips direction and preserves score",
              not bad, f"{len(bad)} mismatches, first={bad[0] if bad else None}")
        check("risk per share is preserved under reflection",
              all(close_to(x.risk_per_share, y.risk_per_share, 1e-6) for x, y in zip(a, b)))


# ── [11] gate monotonicity ──────────────────────────────────────────────────


def t11_gate_monotonicity() -> None:
    print("\n[11] gate monotonicity")
    bars = make_bars(n_days=10)
    base = Config()
    n_base = len(CisdEngine(base, symbol="SPY").run(bars))

    tighter_stack = len(CisdEngine(replace(base, min_stack=base.min_stack + 2), symbol="SPY").run(bars))
    check("raising min_stack cannot add signals", tighter_stack <= n_base,
          f"{tighter_stack} > {n_base}")

    tighter_grade = len(CisdEngine(replace(base, min_grade_to_signal="A+"), symbol="SPY").run(bars))
    check("requiring A+ cannot add signals", tighter_grade <= n_base,
          f"{tighter_grade} > {n_base}")

    smaller_quota = len(CisdEngine(replace(base, max_signals_per_session=1), symbol="SPY").run(bars))
    check("cutting the quota cannot add signals", smaller_quota <= n_base,
          f"{smaller_quota} > {n_base}")

    tighter_rvol = len(CisdEngine(replace(base, min_rvol=5.0), symbol="SPY").run(bars))
    check("demanding 5x volume cannot add signals", tighter_rvol <= n_base,
          f"{tighter_rvol} > {n_base}")

    narrower = len(CisdEngine(replace(base, traded_window="1000-1100"), symbol="SPY").run(bars))
    check("narrowing the traded window cannot add signals", narrower <= n_base,
          f"{narrower} > {n_base}")


# ── [12] the anchor-gate defect, demonstrated ───────────────────────────────


def t12_anchor_gate() -> None:
    print("\n[12] the Pine anchor gate is vacuous")
    bars = make_bars(n_days=10)

    # Every anchor path off, including the gap and manipulation paths, which
    # have their own switches in the Pine (`gapAnchor`, `useAMDX`).
    fixed = Config(gate_fvg=False, gate_key_open=False, gate_pd=False,
                   gate_intrinsic_any_sweep=False, track_gaps=False,
                   hunt_manipulation=False)
    pine = replace(fixed, gate_intrinsic_any_sweep=True)

    e_fixed = CisdEngine(fixed, symbol="SPY")
    e_fixed.run(bars)
    e_pine = CisdEngine(pine, symbol="SPY")
    e_pine.run(bars)

    # With every anchor switched off, the corrected build must admit nothing.
    check("all anchors off admits no blocks (fix on)",
          e_fixed.stats.blocks_formed == 0, f"{e_fixed.stats.blocks_formed} blocks")
    # The Pine admits everything anyway, because its intrinsic path is always
    # satisfied. That is the defect, reproduced.
    check("all anchors off still admits blocks (Pine behaviour)",
          e_pine.stats.blocks_formed > 0, f"{e_pine.stats.blocks_formed} blocks")

    check("the gate therefore changes the signal set",
          e_fixed.stats.signals != e_pine.stats.signals
          or e_fixed.stats.blocks_formed != e_pine.stats.blocks_formed)


# ── [13] free sweep credit, quantified ──────────────────────────────────────


def t13_free_credit() -> None:
    print("\n[13] the Pine's free sweep credit")
    cfg = Config()
    pine = replace(cfg, free_sweep_credit=True)
    credit = cfg.w_sweep * 0.4
    check("the giveaway is 6.4 points on a 16-point weight", close_to(credit, 6.4),
          f"{credit}")
    check("effective Pine base is 24.4, not the 18 the code states",
          close_to(pine.base_score + credit, 24.4), f"{pine.base_score + credit}")

    # The A threshold sits 66 - 24.4 = 41.6 above the real base, out of a
    # ceiling well over 100 -- which is why "A" is not selective in the Pine.
    ceiling = pine_bug_compat().max_possible_score()
    check("Pine score ceiling far exceeds the A threshold", ceiling > 100,
          f"ceiling={ceiling:.1f}")
    headroom = ceiling - 66.0
    check("A is reachable from many different directions", headroom > 60,
          f"headroom={headroom:.1f}")


# ── [14] bias neutrality ────────────────────────────────────────────────────


def t14_bias() -> None:
    print("\n[14] bias defaults")
    from cisd.engine import BiasTracker

    fixed = BiasTracker(Config(bias_expires=True, bias_max_age_bars=5))
    check("an unset bias is neutral, not bullish", fixed.current() is None)

    fixed.direction = 1
    fixed.set_at = 10
    fixed._now = 12
    check("a fresh bias reads through", fixed.current() is True)
    fixed._now = 100
    check("a stale bias expires to neutral", fixed.current() is None)

    latching = BiasTracker(Config(bias_expires=False))
    latching.direction = 1
    latching.set_at = 0
    latching._now = 10_000
    check("Pine-compat bias never expires", latching.current() is True)

    # Regression guard. The tracker ages in BIAS-timeframe bars; if the age
    # were ever measured against the base (1-minute) index again, every bias
    # would expire instantly and read as permanently neutral -- which looks
    # indistinguishable from a working gate until you check the labels.
    bars = make_bars(n_days=12)
    sigs = CisdEngine(Config(), symbol="SPY").run(bars)
    labels = {s.bias for s in sigs}
    check("bias is not permanently neutral across a full run",
          bool(sigs) and labels != {"neutral"}, f"labels={labels}")


# ── [15] targets and R multiples ────────────────────────────────────────────


def t15_targets() -> None:
    print("\n[15] target projection")
    # Long: extreme 99, displacement extreme 101 -> leg = 2. Entry 100, stop 99.
    ts = build_targets(entry=100.0, extreme=99.0, displacement_extreme=101.0,
                       stop=99.0, multiples=(2.0, 2.5, 4.0), bull=True)
    check("three targets projected", len(ts) == 3, f"{len(ts)}")
    check("2.0 leg multiple projects to 103", close_to(ts[0].price, 103.0), f"{ts[0].price}")
    check("4.0 leg multiple projects to 107", close_to(ts[2].price, 107.0), f"{ts[2].price}")

    # Risk is 1.0, so the 2.0 leg multiple is a 3R target -- NOT 2R. This is the
    # discrepancy the docstring warns about, verified rather than asserted.
    check("R multiple is measured from entry, not from the extreme",
          close_to(ts[0].r_multiple, 3.0), f"{ts[0].r_multiple}")

    # Short mirrors exactly.
    tss = build_targets(entry=100.0, extreme=101.0, displacement_extreme=99.0,
                        stop=101.0, multiples=(2.0,), bull=False)
    check("short target mirrors the long", close_to(tss[0].price, 97.0), f"{tss[0].price}")
    check("short R multiple mirrors the long", close_to(tss[0].r_multiple, 3.0))

    # A degenerate leg must produce nothing rather than an invented level.
    none = build_targets(100.0, 99.0, 99.0, 99.0, (2.0,), True)
    check("zero displacement leg yields no targets", none == [])

    inverted = build_targets(100.0, 99.0, 98.0, 99.0, (2.0,), True)
    check("inverted leg yields no targets", inverted == [])

    # A target already behind the entry is dropped, not booked as an instant win.
    behind = build_targets(entry=110.0, extreme=99.0, displacement_extreme=101.0,
                           stop=99.0, multiples=(2.0, 4.0), bull=True)
    check("targets behind the entry are dropped",
          all(t.price > 110.0 for t in behind), f"{[t.price for t in behind]}")


# ── [16] sessions and blackouts ─────────────────────────────────────────────


def t16_sessions() -> None:
    print("\n[16] sessions and event blackouts")
    check("0930-1600 parses to minutes", parse_session("0930-1600") == (570, 960))
    check("a 0000 end means end-of-day, not an empty window",
          parse_session("1800-0000") == (1080, 1440))

    check("in_window is half-open at the start", in_window(570, 570, 960))
    check("in_window excludes the end", not in_window(960, 570, 960))

    # The wrap-past-midnight case that a naive comparison silently drops.
    start, end = parse_session("2000-0400")
    check("wrapped window matches before midnight", in_window(1300, start, end))
    check("wrapped window matches after midnight", in_window(120, start, end))
    check("wrapped window excludes the middle of the day", not in_window(720, start, end))

    cal = EventCalendar.from_local([("2026-03-02 08:30", "CPI")], before_min=15, after_min=15)
    t = int(datetime(2026, 3, 2, 8, 30, tzinfo=NY).timestamp())
    check("blocked at the event", cal.blocked(t) == "CPI")
    check("blocked 14 minutes before", cal.blocked(t - 14 * 60) == "CPI")
    check("clear 16 minutes before", cal.blocked(t - 16 * 60) is None)
    check("clear 16 minutes after", cal.blocked(t + 16 * 60) is None)
    check("an empty calendar blocks nothing", EventCalendar().blocked(t) is None)
    check("an empty calendar reports itself as empty", EventCalendar().is_empty())


# ── [17] previous-day ranges do not leak forward ────────────────────────────


def t17_period_ranges() -> None:
    print("\n[17] previous-day / previous-week ranges")
    bars = make_bars(n_days=8)
    prev_day, prev_week = build_period_ranges(bars)

    dates = sorted({b.session_date_ny for b in bars})
    check("first session has no previous day", math.isnan(prev_day[dates[0]][0]))

    # The value for day N must equal the actual extremes of day N-1.
    d1_bars = [b for b in bars if b.session_date_ny == dates[1]]
    actual_prev = (max(b.high for b in bars if b.session_date_ny == dates[0]),
                   min(b.low for b in bars if b.session_date_ny == dates[0]))
    check("previous-day high matches the prior session", close_to(prev_day[dates[1]][0], actual_prev[0]))
    check("previous-day low matches the prior session", close_to(prev_day[dates[1]][1], actual_prev[1]))

    # And it must NOT equal the current day's own range -- the leak that would
    # make every discount reading clairvoyant.
    own_high = max(b.high for b in d1_bars)
    check("previous-day high is not the current day's high",
          not close_to(prev_day[dates[1]][0], own_high) or close_to(actual_prev[0], own_high))


# ── [18] fair value gaps ────────────────────────────────────────────────────


def t18_fvg() -> None:
    print("\n[18] fair value gaps")
    # Bullish: bar 2's low (105) is above bar 0's high (101).
    bs = [bar(0, 100, 101, 99, 100.5), bar(60, 101, 106, 100.5, 105.5), bar(120, 105.5, 107, 105, 106)]
    g = find_fvg(bs, 2, 0.1)
    check("bullish gap found", g is not None and g.direction == 1)
    if g:
        check("gap spans the untraded band", close_to(g.bottom, 101.0) and close_to(g.top, 105.0))
        check("gap midpoint is the consequent encroachment", close_to(g.mid, 103.0))
        check("order block is the initiating candle", close_to(g.ob_top, 101.0) and close_to(g.ob_bottom, 99.0))

    # Bearish mirror.
    bs2 = [bar(0, 106, 107, 105, 105.5), bar(60, 105, 105.5, 100, 100.5), bar(120, 100.5, 101, 99, 99.5)]
    g2 = find_fvg(bs2, 2, 0.1)
    check("bearish gap found", g2 is not None and g2.direction == -1)
    if g2:
        check("bearish gap spans correctly", close_to(g2.top, 105.0) and close_to(g2.bottom, 101.0))

    # Overlap means no gap.
    bs3 = [bar(0, 100, 105, 99, 104), bar(60, 104, 106, 103, 105), bar(120, 105, 107, 104, 106)]
    check("overlapping bars produce no gap", find_fvg(bs3, 2, 0.1) is None)

    # Minimum size is enforced.
    check("a gap under the minimum size is rejected", find_fvg(bs, 2, 99.0) is None)


# ── [19] engine self-consistency ────────────────────────────────────────────


def t19_engine_consistency() -> None:
    print("\n[19] engine self-consistency")
    bars = make_bars(n_days=10)
    e = CisdEngine(Config(), symbol="SPY")
    sigs = e.run(bars)
    st = e.stats

    check("signals never exceed triggers", st.signals <= st.triggers,
          f"{st.signals} > {st.triggers}")
    check("triggers never exceed armed blocks", st.triggers <= st.blocks_armed,
          f"{st.triggers} > {st.blocks_armed}")
    check("armed never exceeds confirmed", st.blocks_armed <= st.blocks_confirmed,
          f"{st.blocks_armed} > {st.blocks_confirmed}")
    check("confirmed never exceeds formed", st.blocks_confirmed <= st.blocks_formed,
          f"{st.blocks_confirmed} > {st.blocks_formed}")

    check("signals are in chronological order",
          all(sigs[i].ts <= sigs[i + 1].ts for i in range(len(sigs) - 1)))

    check("every signal has a stop on the correct side", all(
        (s.stop < s.entry) if s.is_long else (s.stop > s.entry) for s in sigs
    ))
    check("every signal has at least one target beyond the entry", all(
        s.targets and all((t.price > s.entry) if s.is_long else (t.price < s.entry) for t in s.targets)
        for s in sigs
    ))
    check("every signal has positive risk", all(s.risk_per_share > 0 for s in sigs))
    check("every signal carries its reasoning", all(s.reasons for s in sigs))

    # Quota is per session and must hold.
    by_day: dict[str, int] = {}
    for s in sigs:
        d = datetime.fromtimestamp(s.ts, NY).strftime("%Y-%m-%d")
        by_day[d] = by_day.get(d, 0) + 1
    worst = max(by_day.values()) if by_day else 0
    check("session quota is respected", worst <= Config().max_signals_per_session,
          f"max {worst} in a session")

    # Traded window is respected.
    start, end = parse_session(Config().traded_window)
    check("all signals fall inside the traded window", all(
        start <= datetime.fromtimestamp(s.ts, NY).hour * 60 + datetime.fromtimestamp(s.ts, NY).minute < end
        for s in sigs
    ))

    # R multiples recomputed independently from entry/stop/target prices.
    bad_r = [
        (s.ts, t.r_multiple, s.r_multiple_at(t.price))
        for s in sigs for t in s.targets
        if not close_to(t.r_multiple, s.r_multiple_at(t.price), 1e-6)
    ]
    check("stored R multiples match recomputation from prices", not bad_r,
          f"{len(bad_r)} mismatches")


# ── [20] determinism ────────────────────────────────────────────────────────


def t20_determinism() -> None:
    print("\n[20] determinism")
    bars = make_bars(n_days=8)
    a = CisdEngine(Config(), symbol="SPY").run(bars)
    b = CisdEngine(Config(), symbol="SPY").run(bars)
    check("two identical runs give identical signals",
          len(a) == len(b) and all(
              x.ts == y.ts and x.direction == y.direction and close_to(x.score, y.score)
              for x, y in zip(a, b)
          ))

    # A fresh engine must not carry state from a previous run.
    e = CisdEngine(Config(), symbol="SPY")
    first = e.run(bars)
    e2 = CisdEngine(Config(), symbol="SPY")
    second = e2.run(bars)
    check("engines do not share state", len(first) == len(second))


# ── report ──────────────────────────────────────────────────────────────────


def t21_risk() -> None:
    print("\n[21] risk limits")
    from cisd.risk import RiskGate, RiskLimits

    lim = RiskLimits(max_daily_loss_pct=3.0, max_risk_per_trade_pct=1.0,
                     max_trades_per_day=4, max_concurrent_positions=2,
                     kill_file="data/__nonexistent_kill__")
    g = RiskGate(lim)

    # Fails closed before initialisation -- the property that matters most for
    # an unattended bot, because a half-started process is exactly when a
    # permissive default does damage.
    dummy = Signal(ts=0, symbol="SPY", direction="long", entry=500.0, stop=498.0,
                   targets=[], grade="A", score=70, stack=3, timeframe=5,
                   setup_type="reversal", tf_agreement=2, bias="bull")
    check("uninitialised gate refuses to trade", not g.can_open(dummy, 1.0, 600))

    g.start_session(equity=10_000.0, session_date="2026-08-13")
    check("daily loss budget is 3% of equity", close_to(g.daily_loss_budget, 300.0))
    check("per-trade budget is 1% of equity", close_to(g.per_trade_budget, 100.0))

    # $1.50 contract costs $150; a $100 budget cannot buy one.
    d = g.can_open(dummy, 1.50, 600)
    check("refuses when one contract exceeds the budget", not d.allowed, d.reason)

    # $0.40 contract costs $40; $100 buys two.
    d = g.can_open(dummy, 0.40, 600)
    check("sizes to the budget", d.allowed and d.contracts == 2, f"{d.contracts} {d.reason}")

    # Absolute contract cap binds before the budget does.
    g_cap = RiskGate(replace(lim, max_contracts_per_trade=1))
    g_cap.start_session(10_000.0, "2026-08-13")
    check("absolute contract cap binds", g_cap.can_open(dummy, 0.40, 600).contracts == 1)

    # Delta sizing is never smaller than premium sizing, and is capped by
    # premium -- a delta loss cannot exceed a total loss.
    g_delta = RiskGate(replace(lim, sizing_mode="delta"))
    g_delta.start_session(10_000.0, "2026-08-13")
    n_prem = g.size(dummy, 1.00, delta=0.5)
    n_delta = g_delta.size(dummy, 1.00, delta=0.5)
    check("delta sizing is at least premium sizing", n_delta >= n_prem,
          f"delta={n_delta} premium={n_prem}")

    # Losses eat the budget, and the per-trade budget is capped by what remains.
    g.record_open(2, 80.0)
    g.record_close(-250.0)
    check("loss reduces the remaining budget", close_to(g.loss_remaining, 50.0),
          f"{g.loss_remaining}")
    check("per-trade budget is capped by the remaining daily budget",
          close_to(g.budget_for_trade(), 50.0), f"{g.budget_for_trade()}")

    g.record_open(1, 40.0)
    g.record_close(-60.0)
    check("breaching the daily loss halts the gate", g.halted_reason is not None)
    check("a halted gate refuses new trades", not g.can_open(dummy, 0.10, 600))

    # Trade count cap.
    g2 = RiskGate(lim)
    g2.start_session(10_000.0, "2026-08-13")
    for _ in range(4):
        g2.record_open(1, 10.0)
        g2.record_close(0.0)
    check("daily trade cap refuses the fifth", not g2.can_open(dummy, 0.10, 600))

    # Concurrency cap.
    g3 = RiskGate(lim)
    g3.start_session(10_000.0, "2026-08-13")
    g3.record_open(1, 10.0)
    g3.record_open(1, 10.0)
    check("concurrency cap refuses a third position", not g3.can_open(dummy, 0.10, 600))

    # Hard flat time. 11:55 = 715 minutes.
    g4 = RiskGate(lim)
    g4.start_session(10_000.0, "2026-08-13")
    check("trading allowed before the flat time", g4.can_open(dummy, 0.10, 700).allowed)
    check("must flatten at the hard time", g4.must_flatten(715) is not None)
    check("refuses new entries at the flat time", not g4.can_open(dummy, 0.10, 715))

    # A stop too far away is refused outright, not sized down.
    wide = Signal(ts=0, symbol="SPY", direction="long", entry=500.0, stop=480.0,
                  targets=[], grade="A", score=70, stack=3, timeframe=5,
                  setup_type="reversal", tf_agreement=2, bias="bull")
    check("an over-wide stop is refused", not g4.can_open(wide, 0.10, 600))

    # Unusable price fails closed rather than sizing off a nan.
    check("nan contract price is refused", not g4.can_open(dummy, float("nan"), 600))
    check("zero contract price is refused", not g4.can_open(dummy, 0.0, 600))

    # Kill file.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        kf = pathlib.Path(td) / "KILL"
        g5 = RiskGate(replace(lim, kill_file=str(kf)))
        g5.start_session(10_000.0, "2026-08-13")
        check("no kill file means trading is allowed", g5.can_open(dummy, 0.10, 600).allowed)
        kf.write_text("stop", encoding="utf-8")
        check("kill file blocks new entries", not g5.can_open(dummy, 0.10, 600))
        check("kill file forces a flatten", g5.must_flatten(600) is not None)

    check("start_session rejects zero equity",
          _raises(lambda: RiskGate(lim).start_session(0.0, "2026-08-13")))


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def t22_tv_import() -> None:
    print("\n[22] TradingView import")
    import tempfile

    from tools.tv_import import infer_timeframe_minutes, load

    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "tv.csv"
        p.write_text(
            "time,open,high,low,close,Volume,x_sig_dir,x_sig_score,x_sig_stop,x_sig_disp\n"
            "2026-08-12T09:30:00-04:00,500.0,501.0,499.5,500.5,120000,0,,,\n"
            "2026-08-12T09:31:00-04:00,500.5,502.0,500.0,501.5,90000,1,74.5,499.5,502.0\n"
            "2026-08-12T09:32:00-04:00,501.5,501.8,500.9,501.0,70000,0,,,\n",
            encoding="utf-8",
        )
        ex = load(p)
        check("bars parsed", len(ex.bars) == 3, f"{len(ex.bars)}")
        check("OHLCV read correctly", close_to(ex.bars[1].high, 502.0)
              and close_to(ex.bars[1].volume, 90000))
        check("NY wall-clock time round-trips",
              ex.bars[0].dt_ny.strftime("%H:%M") == "09:30",
              ex.bars[0].dt_ny.strftime("%H:%M"))
        check("export patch detected", ex.has_export_patch())

        rows = ex.signal_rows()
        check("one signal extracted", len(rows) == 1, f"{len(rows)}")
        if rows:
            ts, d, sc, stop, disp = rows[0]
            check("signal direction read", d == 1)
            check("signal score read", close_to(sc, 74.5))
            check("signal stop read", close_to(stop, 499.5))
        check("timeframe inferred as 1 minute", infer_timeframe_minutes(ex.bars) == 1)

        # A naive timestamp must be REJECTED, not assumed to be UTC or local --
        # guessing shifts every bar by hours and the diff then fails for a
        # reason unrelated to the port.
        q = pathlib.Path(td) / "naive.csv"
        q.write_text(
            "time,open,high,low,close\n2026-08-12T09:30:00,500,501,499,500\n",
            encoding="utf-8",
        )
        check("naive timestamps are rejected", _raises(lambda: load(q)))

        # Epoch seconds are accepted.
        r = pathlib.Path(td) / "epoch.csv"
        r.write_text("time,open,high,low,close\n1786541400,500,501,499,500\n",
                     encoding="utf-8")
        check("epoch timestamps are accepted", len(load(r).bars) == 1)

        # A missing OHLC column is an error, not a silent zero.
        s = pathlib.Path(td) / "bad.csv"
        s.write_text("time,open,close\n1786541400,500,500\n", encoding="utf-8")
        check("missing columns raise", _raises(lambda: load(s)))

        # An export without the patch must say so rather than compare to zero.
        t = pathlib.Path(td) / "nopatch.csv"
        t.write_text("time,open,high,low,close\n1786541400,500,501,499,500\n"
                     "1786541460,500,501,499,500\n", encoding="utf-8")
        ex2 = load(t)
        check("missing export patch is detected", not ex2.has_export_patch())
        check("no patch yields no signal rows", ex2.signal_rows() == [])


def report_shape() -> None:
    """Not a test -- the descriptive numbers that decide what to fix next."""
    print("\n── model shape on synthetic data ──")
    bars = make_bars(n_days=20)

    for name, cfg in (("corrected", Config()), ("pine-bug-compat", pine_bug_compat())):
        e = CisdEngine(cfg, symbol="SPY")
        sigs = e.run(bars)
        st = e.stats
        grades = ", ".join(f"{g}={n}" for g, n in sorted(st.grade_counts.items())) or "none"
        print(f"\n  {name}:")
        print(f"    bars {st.bars}  blocks {st.blocks_formed}  confirmed {st.blocks_confirmed}"
              f"  armed {st.blocks_armed}  triggers {st.triggers}  signals {st.signals}")
        print(f"    grades: {grades}")
        print(f"    score ceiling: {cfg.max_possible_score():.1f} (A threshold 66)")
        top = sorted(st.rejected.items(), key=lambda kv: -kv[1])[:5]
        if top:
            print("    top rejections: " + ", ".join(f"{k}={v}" for k, v in top))
        if sigs:
            print(f"\n    example signal:\n")
            for line in sigs[0].explain().splitlines():
                print(f"      {line}")


def main() -> int:
    # The Windows console defaults to cp1252, which cannot encode the box
    # drawing characters or the sigma in a target label. Without this the suite
    # passes and then dies printing its own results.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print("=" * 70)
    print("CISD port verification")
    print("=" * 70)

    t1_aggregation()
    t2_atr()
    t3_pivots()
    t4_cisd()
    t5_rejection_block()
    t6_dealing_range()
    t7_vwap()
    t8_rvol()
    t9_prefix_stability()
    t10_mirror_symmetry()
    t11_gate_monotonicity()
    t12_anchor_gate()
    t13_free_credit()
    t14_bias()
    t15_targets()
    t16_sessions()
    t17_period_ranges()
    t18_fvg()
    t19_engine_consistency()
    t20_determinism()
    t21_risk()
    t22_tv_import()

    print("\n" + "=" * 70)
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 70)

    if FAIL == 0:
        report_shape()
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
