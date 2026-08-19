"""Moving averages, tested once, properly, on the full archive.

    python -m research.ema_test [window]

None of the original twelve candidates used a moving average. That was a gap in
coverage rather than a judgement, and this closes it.

**The prior is low and stated in advance.** EMA crossovers are the most-tested
family of rules in retail trading. Anything that worked on 1-minute SPY with
conventional parameters would have been competed away long before it reached
us. The value here is closing the question, not opening one.

**The parameters are pre-registered, not searched.** Conventional pairs only:
9/21, 8/21, 12/26, 20/50, plus 21 and 50 as single-EMA trend filters. Searching
fast/slow pairs would manufacture a winner -- with ~20 tests at p<0.05, one
false positive is the expectation, not the exception. The bar is |t| >= 2.5
against the random-entry null, and the Bonferroni-adjusted threshold for this
many tests is printed alongside so the reader can apply it.

Anything that clears the bar gets the concentration test immediately, because
that is what killed the last candidate that looked like this.
"""

from __future__ import annotations

import functools
import math
import statistics
import sys

from cisd.indicators import atr
from research.baserate import _entry_bars, run_null
from research.harness import HEADER, format_row, measure
from research.signals import ema_cross, ema_pullback, ema_slope, ema_trend

STRUCTURES = [(1.0, 1.0), (1.0, 2.0), (2.0, 2.0)]
BAR = 2.5


def candidates() -> dict:
    """Pre-registered, conventional parameters. This dict is not edited after
    seeing results -- that is the entire point of writing it down."""
    c = {}
    for f, s in ((9, 21), (8, 21), (12, 26), (20, 50)):
        c[f"ema_cross_{f}_{s}"] = functools.partial(ema_cross, fast=f, slow=s)
        c[f"ema_pullback_{f}_{s}"] = functools.partial(ema_pullback, fast=f, slow=s)
    for n in (21, 50):
        c[f"ema_trend_{n}"] = functools.partial(ema_trend, length=n)
        c[f"ema_slope_{n}"] = functools.partial(ema_slope, length=n)
    return c


def concentration(rs: list[float], frac: float = 0.01) -> tuple[float, float]:
    """(t before, t after) removing the best `frac` of trades."""
    def t(xs):
        if len(xs) < 2:
            return math.nan
        se = statistics.stdev(xs) / math.sqrt(len(xs))
        return statistics.mean(xs) / se if se else math.nan
    k = max(1, int(round(frac * len(rs))))
    return t(rs), t(sorted(rs)[:-k])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    from bot.feed import ArchiveFeed

    window = sys.argv[1] if len(sys.argv) > 1 else "0930-1200"
    feed = ArchiveFeed()
    bars = []
    for d in feed.sessions():
        bars.extend(feed.session_bars(d))
    if len(bars) < 50_000:
        print(f"only {len(bars)} bars available")
        return 1

    a = atr(bars, 14)
    by_session = _entry_bars(bars, window)
    cands = candidates()
    n_tests = len(cands) * len(STRUCTURES)

    print("=" * 78)
    print("MOVING AVERAGES  (pre-registered parameters, not searched)")
    print("=" * 78)
    print(f"  {len(bars):,} bars, {len(by_session)} sessions, window {window}")
    print(f"  {bars[0].session_date_ny} to {bars[-1].session_date_ny}")
    print(f"  {len(cands)} candidates x {len(STRUCTURES)} structures = {n_tests} tests")
    print(f"  bar |t| >= {BAR}; Bonferroni for {n_tests} tests needs "
          f"|t| >= {_bonferroni_t(n_tests):.2f}")

    survivors = []
    for stop_atr, target_atr in STRUCTURES:
        null = run_null(bars, a, by_session, "random", stop_atr, target_atr, 60, seed=7)
        print(f"\n  stop {stop_atr} ATR / target {target_atr} ATR"
              f"   (null mean R {null.mean_r:+.3f}, n {null.n})")
        print(HEADER)
        print("  " + "-" * 72)
        for name, fn in cands.items():
            r = measure(bars, fn, name, stop_atr=stop_atr, target_atr=target_atr,
                        window=window)
            if r.n < 50:
                print(f"  {name:<30} {r.n:>5}  too few trades to score")
                continue
            print(format_row(r, null))
            t = r.versus(null)
            if not math.isnan(t) and abs(t) >= BAR:
                survivors.append((name, stop_atr, target_atr, t, r))

    print("\n" + "=" * 78)
    if not survivors:
        print(f"  Nothing cleared |t| >= {BAR} against the null, in {n_tests} tests.")
        print("  Moving averages carry no edge on 1-minute SPY in this window.")
        print("  Closed. Do not re-test with different lengths -- that is the")
        print("  search this file exists to avoid.")
        return 0

    print(f"  {len(survivors)} of {n_tests} cleared |t| >= {BAR}. Expected by")
    print(f"  chance at p<0.05: ~{0.05 * n_tests:.1f}. Now the concentration test.")
    for name, sa, ta, t, r in survivors:
        before, after = concentration(r.rs)
        flip = "  <-- FLIPS" if (before > 0) != (after > 0) else ""
        print(f"\n  {name}  {sa}/{ta} ATR   t vs null {t:+.2f}   n {r.n}")
        print(f"    t vs zero {before:+.2f} -> {after:+.2f} "
              f"dropping the best 1%{flip}")
    return 0


def _bonferroni_t(k: int) -> float:
    """Two-sided t threshold for family-wise 0.05 over k tests, normal approx."""
    from math import log, sqrt
    p = 0.05 / k
    # Acklam-style inverse normal, adequate at these tail probabilities.
    q = p / 2.0
    t = sqrt(-2.0 * log(q))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)


if __name__ == "__main__":
    sys.exit(main())
