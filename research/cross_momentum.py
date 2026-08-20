"""Cross-sectional momentum: does ranking stocks beat holding all of them?

    python -m research.cross_momentum [cost_bp_per_side]

This is the strategy from Jegadeesh & Titman (1993, 1999): rank stocks by past
return, hold the winners, rebalance monthly. It is the ONE anomaly in this
project's reading that survived its authors' own out-of-sample test -- size and
value did not.

**The null is the equal-weighted universe, not zero.** Holding any ten of these
hundred stocks earns roughly the universe return. The question is never "does
the winner portfolio go up" -- it always does, it is long equities in a bull
market. The question is whether ranking by momentum beats picking ten at
random. So a random-decile null is drawn many times and the comparison is
PAIRED month by month, because both portfolios share the same market beta and
an unpaired test would drown the difference in market noise.

**Parameters are pre-registered:** formation windows (6,1), (12,1), (12,2) in
months, where the second number is the skip -- the standard gap that avoids
short-term reversal. Decile portfolio, monthly rebalance, equal weight. Three
tests, not a search.

**Known upward bias, unremovable with this data.** The universe is 100 US large
caps as of end-2018, chosen from memory -- and memory contains only survivors.
Every one of the hundred still trades today. A point-in-time universe would
include names that were acquired, delisted, or collapsed, and those would be
disproportionately momentum LOSERS whose absence flatters the long-only decile
less than it flatters the universe null. The direction of the bias on the
DIFFERENCE is not obvious, but the level of both is overstated.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

STORE = Path("data/equities")

# Names whose price history we cannot trust, with the reason. Checked by
# research/fetch_equities.py, which cross-checks every refresh against what is
# already on disk and refuses to overwrite a disagreement.
#
# Both of these had SPINOFFS in 2026. A spinoff is not a split, so a
# split-adjusted series carries an artificial one-day drop -- 17.8% for FDX,
# 50.9% for HON on the Alpaca feed. A 50% phantom move puts a name at the very
# top or very bottom of a momentum ranking for a full year.
#
# Excluding them is itself a mild selection effect: spinoffs are corporate
# events, so this drops a non-random slice of the universe. Two names out of a
# hundred, recorded rather than hidden. CRSP resolves it properly.
EXCLUDE = {
    "FDX": "FedEx Freight spinoff 2026-06-01; sources disagree by 18%",
    "HON": "Honeywell separation 2026-06; both feeds discontinuous",
}
DECILE = 10          # names held
NULL_SEEDS = 200     # random-decile draws
FORMATIONS = [(6, 1), (12, 1), (12, 2)]
DEFAULT_COST_BP = 5.0   # per side, on turnover


def load_prices() -> tuple[list[str], dict[str, dict[str, float]], dict[str, float]]:
    """(dates, {symbol: {date: close}}, {date: spy_close})"""
    px: dict[str, dict[str, float]] = {}
    spy: dict[str, float] = {}
    for p in sorted(STORE.glob("*.csv")):
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        series = {r["date"]: float(r["close"]) for r in rows}
        if p.stem == "_SPY":
            spy = series
        elif p.stem not in EXCLUDE:
            px[p.stem] = series
    dates = sorted(set().union(*(set(s) for s in px.values())))
    return dates, px, spy


def month_ends(dates: list[str]) -> list[str]:
    last: dict[str, str] = {}
    for d in dates:
        last[d[:7]] = d
    return [last[k] for k in sorted(last)]


def ret(px: dict[str, float], a: str, b: str) -> float:
    """Simple return from date `a` to date `b`, or nan."""
    if a not in px or b not in px or px[a] <= 0:
        return math.nan
    return px[b] / px[a] - 1.0


def paired_t(diffs: list[float]) -> float:
    if len(diffs) < 2:
        return math.nan
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return statistics.mean(diffs) / se if se else math.nan


def ann(monthly: list[float]) -> float:
    if not monthly:
        return math.nan
    g = 1.0
    for m in monthly:
        g *= (1.0 + m)
    return 100.0 * (g ** (12.0 / len(monthly)) - 1.0)


def sharpe(monthly: list[float]) -> float:
    if len(monthly) < 2:
        return math.nan
    sd = statistics.stdev(monthly)
    return (statistics.mean(monthly) / sd) * math.sqrt(12) if sd else math.nan


def turnover_cost(prev: set[str], cur: set[str], cost_bp: float) -> float:
    """Cost of moving from `prev` to `cur`, as a fraction of portfolio value.
    Both the sells and the buys are charged."""
    if not cur:
        return 0.0
    changed = len(cur - prev)
    return 2.0 * (changed / len(cur)) * (cost_bp / 1e4)


def run(mes: list[str], px, form: int, skip: int, cost_bp: float, seed: int | None):
    """One backtest. `seed` None means momentum; an int means a random decile."""
    rng = random.Random(seed) if seed is not None else None
    rets: list[float] = []
    held: set[str] = set()
    months: list[str] = []

    for k in range(form, len(mes) - 1):
        rank_end = mes[k - skip]
        rank_start = mes[k - form]
        hold_from, hold_to = mes[k], mes[k + 1]

        elig = []
        for s, series in px.items():
            if hold_from not in series or hold_to not in series:
                continue
            r = ret(series, rank_start, rank_end)
            if not math.isnan(r):
                elig.append((r, s))
        if len(elig) < 3 * DECILE:
            continue

        if rng is None:
            elig.sort(reverse=True)
            pick = {s for _, s in elig[:DECILE]}
        else:
            pick = set(rng.sample([s for _, s in elig], DECILE))

        fwd = [ret(px[s], hold_from, hold_to) for s in pick]
        fwd = [f for f in fwd if not math.isnan(f)]
        if not fwd:
            continue
        gross = statistics.mean(fwd)
        rets.append(gross - turnover_cost(held, pick, cost_bp))
        held = pick
        months.append(hold_to)
    return rets, months


def main() -> int:
    cost_bp = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_COST_BP
    dates, px, spy = load_prices()
    if len(px) < 50:
        print(f"only {len(px)} symbols in {STORE}")
        return 1
    mes = month_ends(dates)

    print("=" * 78)
    print("CROSS-SECTIONAL MOMENTUM  (pre-registered, 3 formations)")
    print("=" * 78)
    print(f"  {len(px)} symbols, {len(mes)} month-ends, {mes[0]} -> {mes[-1]}")
    print(f"  decile = {DECILE} names, equal weight, monthly rebalance")
    print(f"  costs {cost_bp:.1f}bp per side on turnover")
    print(f"  null = {NULL_SEEDS} random {DECILE}-name draws, compared PAIRED")

    # Benchmarks
    bh, held_all = [], set(px)
    for k in range(max(f for f, _ in FORMATIONS), len(mes) - 1):
        fwd = [ret(s, mes[k], mes[k + 1]) for s in px.values()]
        fwd = [f for f in fwd if not math.isnan(f)]
        if fwd:
            bh.append(statistics.mean(fwd))
    spy_r = []
    for k in range(max(f for f, _ in FORMATIONS), len(mes) - 1):
        r = ret(spy, mes[k], mes[k + 1])
        if not math.isnan(r):
            spy_r.append(r)
    print(f"\n  equal-weight universe : {ann(bh):+7.2f}%/yr  SR {sharpe(bh):+.2f}")
    print(f"  SPY buy & hold        : {ann(spy_r):+7.2f}%/yr  SR {sharpe(spy_r):+.2f}")

    print(f"\n  {'formation':<14} {'n':>4} {'ann':>9} {'SR':>6} {'vs null':>9} "
          f"{'paired t':>9} {'vs SPY':>9}")
    print("  " + "-" * 68)

    results = {}
    for form, skip in FORMATIONS:
        mom, months = run(mes, px, form, skip, cost_bp, None)
        # Null: average the random draws month by month, then pair.
        null_by_month = defaultdict(list)
        for s in range(NULL_SEEDS):
            nr, nm = run(mes, px, form, skip, cost_bp, seed=1000 + s)
            for m, v in zip(nm, nr):
                null_by_month[m].append(v)
        null = [statistics.mean(null_by_month[m]) for m in months]
        diffs = [a - b for a, b in zip(mom, null)]
        spy_pair = []
        for m in months:
            i = mes.index(m)
            r = ret(spy, mes[i - 1], m)
            spy_pair.append(r if not math.isnan(r) else 0.0)
        d_spy = [a - b for a, b in zip(mom, spy_pair)]
        label = f"{form}-{skip}"
        results[label] = (mom, null, months, diffs, d_spy)
        print(f"  {label:<14} {len(mom):>4} {ann(mom):>+8.2f}% {sharpe(mom):>+6.2f} "
              f"{ann(mom) - ann(null):>+8.2f}% {paired_t(diffs):>+9.2f} "
              f"{paired_t(d_spy):>+9.2f}")

    print("\n  A positive 'ann' with a flat paired t means the portfolio made")
    print("  money by being long equities, not by ranking them.")

    print("\n[concentration] drop the best 1% of months from the DIFFERENCE")
    for label, (mom, null, months, diffs, _) in results.items():
        k = max(1, int(round(0.01 * len(diffs))))
        cut = sorted(diffs)[:-k]
        flip = "  <-- FLIPS" if (paired_t(diffs) > 0) != (paired_t(cut) > 0) else ""
        print(f"  {label:<8} paired t {paired_t(diffs):+6.2f} -> {paired_t(cut):+6.2f}"
              f"   ({k} months removed){flip}")

    print("\n[split-half] does the edge survive in both halves")
    for label, (mom, null, months, diffs, _) in results.items():
        h = len(diffs) // 2
        a, b = diffs[:h], diffs[h:]
        same = "same sign" if (statistics.mean(a) > 0) == (statistics.mean(b) > 0) else "SIGN FLIP"
        print(f"  {label:<8} first t {paired_t(a):+6.2f}   second t {paired_t(b):+6.2f}"
              f"   {same}   (boundary {months[h]})")

    print("\n[caveats]")
    print("  Universe is 100 end-2018 US large caps and ALL 100 still trade.")
    print("  That is ~100% survivorship. Levels are overstated; the paired")
    print("  difference is the number to trust, and only relatively.")
    print(f"  {len(mes)} months is a short sample for a monthly strategy --")
    print(f"  roughly {len(mes) // 12} independent years.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
