"""Verification for the research harness.

The single most important check is [1]: every candidate signal must give the
same answer when the future is removed. A signal that reads ahead produces a
beautiful batch table and no money, and it is invisible in the output.
"""

from __future__ import annotations

import math
import sys

from cisd.indicators import atr
from research.harness import evaluate_entry, measure, Result
from research.signals import CANDIDATES
from tools.tv_import import load

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    bars = load("data/tv_export.csv").bars

    print("\n[1] signals do not read the future")
    # Evaluating at index k-1 with the series truncated at k must equal
    # evaluating at k-1 with the full series available.
    probes = list(range(2000, 2000 + 400, 7))
    for name, fn in CANDIDATES.items():
        bad = 0
        for k in probes:
            if fn(bars[:k], k - 1) != fn(bars, k - 1):
                bad += 1
        check(f"{name} is causal", bad == 0, f"{bad}/{len(probes)} differ")

    print("\n[2] signals return only -1, 0 or +1")
    for name, fn in CANDIDATES.items():
        vals = {fn(bars, k) for k in probes}
        check(f"{name} range", vals <= {-1, 0, 1}, f"got {vals}")

    print("\n[3] outcome arithmetic")
    a = atr(bars, 14)
    i = 3000
    t = evaluate_entry(bars, i, 1, a[i], 1.0, 2.0, 60, cost_atr=0.0)
    if t is not None:
        if t.outcome == "target":
            check("target pays target/stop R", abs(t.r_multiple - 2.0) < 1e-9, f"{t.r_multiple}")
        elif t.outcome == "stop":
            check("stop costs exactly -1R", abs(t.r_multiple + 1.0) < 1e-9, f"{t.r_multiple}")
        else:
            check("time exit R is finite", not math.isnan(t.r_multiple))
        check("stop is below entry for a long", t.stop < t.entry)
        check("target is above entry for a long", t.target > t.entry)

    # Costs must make every outcome worse, never better.
    free = evaluate_entry(bars, i, 1, a[i], 1.0, 2.0, 60, cost_atr=0.0)
    paid = evaluate_entry(bars, i, 1, a[i], 1.0, 2.0, 60, cost_atr=0.05)
    if free and paid:
        check("costs reduce R", paid.r_multiple < free.r_multiple,
              f"{paid.r_multiple} vs {free.r_multiple}")

    print("\n[4] direction symmetry")
    # A signal and its exact negation must produce mirror-image trade counts.
    m1 = measure(bars, CANDIDATES["momentum_6"], "m", stop_atr=1.0, target_atr=1.0)
    m2 = measure(bars, CANDIDATES["mean_reversion_6"], "r", stop_atr=1.0, target_atr=1.0)
    check("a signal and its negation take the same trades", m1.n == m2.n, f"{m1.n} vs {m2.n}")
    check("their directions are opposite", all(
        x.direction == -y.direction for x, y in zip(m1.trades, m2.trades)))

    print("\n[5] harness plumbing")
    empty = Result("none")
    check("empty result reports nan, not zero", math.isnan(empty.mean_r))
    check("empty result has nan t", math.isnan(empty.t_stat))
    r = measure(bars, lambda b, i: 0, "never")
    check("a signal that never fires yields no trades", r.n == 0)
    r1 = measure(bars, CANDIDATES["gap_fade"], "g", one_per_session=True)
    r2 = measure(bars, CANDIDATES["gap_fade"], "g", one_per_session=False)
    check("one-per-session never exceeds unrestricted", r1.n <= r2.n, f"{r1.n} vs {r2.n}")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
