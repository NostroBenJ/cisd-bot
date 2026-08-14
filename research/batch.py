"""Run every candidate against the null, at a few exit structures.

    python -m research.batch data/tv_export.csv

Reports each candidate's mean R and, more importantly, its t-statistic AGAINST
the matching random-entry null rather than against zero.

**Multiple comparisons are the danger here.** Twelve candidates across three
exit structures is thirty-six tests; at a 5% threshold, roughly two will look
significant by chance alone. So the bar is set higher than usual and nothing
graduates on this table -- anything interesting must survive a held-out period
that we do not look at while iterating.
"""

from __future__ import annotations

import math
import sys

from cisd.indicators import atr
from research.baserate import _entry_bars, averaged
from research.harness import HEADER, Result, format_row, measure
from research.signals import CANDIDATES
from tools.tv_import import load

STRUCTURES = [(1.0, 1.0), (1.0, 2.0), (2.0, 2.0)]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    path = sys.argv[1] if len(sys.argv) > 1 else "data/tv_export.csv"
    window = sys.argv[2] if len(sys.argv) > 2 else "0930-1200"
    bars = load(path).bars
    a = atr(bars, 14)
    by_session = _entry_bars(bars, window)

    print("=" * 78)
    print("CANDIDATE BATCH")
    print("=" * 78)
    print(f"  {len(bars)} bars, {len(by_session)} sessions, window {window}")
    print(f"  {bars[0].dt_ny:%Y-%m-%d} to {bars[-1].dt_ny:%Y-%m-%d}")
    print(f"  {len(CANDIDATES)} candidates x {len(STRUCTURES)} structures = "
          f"{len(CANDIDATES) * len(STRUCTURES)} tests")
    print("  at p<0.05 roughly 2 of these will look significant by chance")

    survivors = []
    for stop_atr, target_atr in STRUCTURES:
        null_mean, null_se, _, _ = averaged(
            bars, a, by_session, "random", stop_atr, target_atr, 60, seeds=40)
        null = Result(f"random {stop_atr}/{target_atr}")
        # Rebuild one representative null sample so `versus` has a distribution
        from research.baserate import run_null
        null = run_null(bars, a, by_session, "random", stop_atr, target_atr, 60, seed=7)

        print(f"\n  stop {stop_atr} ATR / target {target_atr} ATR"
              f"   (null mean R {null_mean:+.3f})")
        print(HEADER)
        print("  " + "-" * 72)
        for name, fn in CANDIDATES.items():
            r = measure(bars, fn, name, stop_atr=stop_atr, target_atr=target_atr,
                        window=window)
            if r.n < 20:
                print(f"  {name:<30} {r.n:>5}  too few trades to score")
                continue
            print(format_row(r, null))
            t = r.versus(null)
            if not math.isnan(t) and abs(t) >= 2.5 and r.n >= 30:
                survivors.append((name, stop_atr, target_atr, r.mean_r, t, r.n))

    print("\n" + "=" * 78)
    if survivors:
        print("  candidates beating the null by |t| >= 2.5 with n >= 30:")
        for name, s, tg, m, t, n in sorted(survivors, key=lambda x: -abs(x[4])):
            print(f"    {name:<26} {s}/{tg}  n={n:<4} meanR {m:+.3f}  t vs null {t:+.2f}")
        print("\n  NONE of these graduate on this table. Each must be re-tested")
        print("  on a held-out period before it means anything.")
    else:
        print("  No candidate beat the null at |t| >= 2.5. That is a real result:")
        print("  it says these simple ideas carry no edge in this window on this")
        print("  sample, and it cost one afternoon instead of one week.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
