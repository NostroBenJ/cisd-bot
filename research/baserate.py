"""Base rates: what does a signal-free entry return?

    python -m research.baserate data/tv_export.csv

This is the control that the CISD work never had. Three nulls are measured
across a grid of stop and target distances:

  random   -- one entry per session at a random bar in the window, random
              direction. This is the true zero-information benchmark.
  long     -- always long. SPY has drifted upward for a century, so this is
              NOT zero: any long-biased signal inherits that drift for free
              and must be scored against it, not against random.
  short    -- always short, the mirror. The gap between long and short is the
              drift, measured rather than assumed.

**Why an asymmetric structure has a non-zero base rate.** With a 0.5 ATR stop
and a 2.0 ATR target, price touches the stop far more often than the target, so
the win rate is low by construction while each win is worth 4R. Those two
effects nearly cancel, but not exactly, and the residual is a property of the
exit geometry rather than of any signal. Reading a candidate's mean R against
zero attributes that residual to the signal. Reading it against this table does
not.

The random null is averaged over many seeds, because a single draw of one
entry per session is itself noisy.
"""

from __future__ import annotations

import math
import random
import statistics
import sys

from cisd.bars import Bar
from cisd.indicators import atr
from cisd.sessions import in_window, parse_session
from research.harness import Result, evaluate_entry
from tools.tv_import import load

# Stop/target grids, in ATR at entry. Chosen to span the plausible design
# space: tight-stop/far-target (lottery), symmetric, and wide-stop/near-target
# (scalp). Each has a different base rate and that is the point.
GRID = [
    (0.5, 0.5), (0.5, 1.0), (0.5, 2.0),
    (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0),
    (2.0, 1.0), (2.0, 2.0), (2.0, 4.0),
]


def _entry_bars(bars: list[Bar], window: str) -> dict[str, list[int]]:
    """Indices inside the trading window, grouped by session."""
    start, end = parse_session(window)
    by_session: dict[str, list[int]] = {}
    for i, b in enumerate(bars):
        if in_window(b.minute_of_day_ny, start, end):
            by_session.setdefault(b.session_date_ny, []).append(i)
    return by_session


def run_null(
    bars: list[Bar],
    a: list[float],
    by_session: dict[str, list[int]],
    mode: str,
    stop_atr: float,
    target_atr: float,
    max_hold_bars: int,
    seed: int,
) -> Result:
    """One draw of a signal-free strategy.

    Every mode takes exactly one entry per session at the same randomly chosen
    bar, so `random`, `long` and `short` differ ONLY in direction. Comparing
    them therefore isolates drift rather than confounding it with entry timing.
    """
    rng = random.Random(seed)
    res = Result(f"{mode} {stop_atr}/{target_atr}")
    for _session, idxs in by_session.items():
        if not idxs:
            continue
        i = rng.choice(idxs)
        if mode == "random":
            d = rng.choice((1, -1))
        elif mode == "long":
            d = 1
        else:
            d = -1
        t = evaluate_entry(bars, i, d, a[i], stop_atr, target_atr, max_hold_bars)
        if t is not None:
            res.trades.append(t)
    res.sessions = len(by_session)
    return res


def averaged(
    bars, a, by_session, mode, stop_atr, target_atr, max_hold, seeds: int
) -> tuple[float, float, float, int]:
    """Mean R averaged over `seeds` independent draws.

    Returns (mean of means, standard error ACROSS draws, mean win rate, mean n).
    The across-draw standard error is the honest one: it captures the sampling
    variability of the whole procedure, not just of one draw's trades."""
    means, wins, ns = [], [], []
    for s in range(seeds):
        r = run_null(bars, a, by_session, mode, stop_atr, target_atr, max_hold, seed=1000 + s)
        if r.n:
            means.append(r.mean_r)
            wins.append(r.win_rate)
            ns.append(r.n)
    if not means:
        return math.nan, math.nan, math.nan, 0
    se = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else math.nan
    return statistics.mean(means), se, statistics.mean(wins), int(statistics.mean(ns))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    path = sys.argv[1] if len(sys.argv) > 1 else "data/tv_export.csv"
    window = sys.argv[2] if len(sys.argv) > 2 else "0930-1200"
    seeds = 40

    bars = load(path).bars
    a = atr(bars, 14)
    by_session = _entry_bars(bars, window)
    sessions = len(by_session)

    print("=" * 78)
    print("BASE RATES -- signal-free entries")
    print("=" * 78)
    print(f"  {len(bars)} bars, {sessions} sessions, window {window}")
    print(f"  {bars[0].dt_ny:%Y-%m-%d} to {bars[-1].dt_ny:%Y-%m-%d}")
    print(f"  one entry per session, {seeds} random draws averaged")
    print("  R is net of a 0.02 ATR round-trip cost")

    print(f"\n  {'stop/target':>12} {'mode':<8} {'n':>4} {'win':>6} {'meanR':>8} {'se':>7}")
    print("  " + "-" * 56)
    rows = {}
    for stop_atr, target_atr in GRID:
        for mode in ("random", "long", "short"):
            m, se, w, n = averaged(bars, a, by_session, mode, stop_atr, target_atr, 60, seeds)
            rows[(stop_atr, target_atr, mode)] = m
            label = f"{stop_atr}/{target_atr}" if mode == "random" else ""
            print(f"  {label:>12} {mode:<8} {n:>4} {100 * w:>5.0f}% {m:>+8.3f} {se:>7.3f}")
        print()

    print("  " + "-" * 56)
    print("\n  drift (long minus short, same entries):")
    for stop_atr, target_atr in GRID:
        d = rows[(stop_atr, target_atr, "long")] - rows[(stop_atr, target_atr, "short")]
        print(f"    {stop_atr}/{target_atr:<5} {d:+.3f} R")

    print("\n  How to use this table: a candidate signal is only interesting if")
    print("  its mean R beats the RANDOM row for the same stop/target -- and a")
    print("  long-biased candidate must beat the LONG row, because it inherits")
    print("  drift it did not earn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
