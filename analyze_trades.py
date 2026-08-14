"""Analyse a TradingView Strategy Tester "List of Trades" export.

    python analyze_trades.py data/trades_1m_v2.csv

Reports expectancy with a standard error and a t-statistic, because a total
P&L figure alone cannot distinguish a real edge from a run of luck. Also
reports the result with the single worst trade removed -- if a strategy's
verdict flips when one outlier is dropped, the verdict was the outlier.

Exit-reason and holding-time breakdowns are here because they diagnose WHY a
result came out the way it did. A strategy whose bracket exits lose and whose
time exits win is telling you the stop is in the wrong place, not that the
entries are worthless -- though note that conditioning on "survived without
being stopped" selects for winners, so that split is a hint to investigate,
never evidence on its own.

Stdlib only.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import Counter
from datetime import datetime


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    trades = {}
    for r in rows:
        t = trades.setdefault(int(r["Trade number"]), {})
        typ = r["Type"]
        if typ.startswith("Entry"):
            t["side"] = "long" if "long" in typ else "short"
            t["entry_t"] = datetime.strptime(r["Date and time"], "%Y-%m-%d %H:%M")
            t["entry_p"] = float(r["Price USD"])
        else:
            t["exit_t"] = datetime.strptime(r["Date and time"], "%Y-%m-%d %H:%M")
            t["signal"] = r["Signal"]
        t["pnl"] = float(r["Net PnL USD"])
        t["ret"] = float(r["Return %"])
        t["comm"] = float(r["Commission USD"])
        t["bars"] = int(r["Duration (bars)"])
    out = [t for t in trades.values() if "entry_p" in t and "exit_t" in t]
    out.sort(key=lambda t: t["entry_t"])
    return out


def tstat(xs):
    if len(xs) < 2:
        return math.nan
    se = statistics.stdev(xs) / math.sqrt(len(xs))
    return statistics.mean(xs) / se if se else math.nan


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    T = load(sys.argv[1])
    n = len(T)
    pnl = [t["pnl"] for t in T]
    wins = [t for t in T if t["pnl"] > 0]
    yrs = (T[-1]["entry_t"] - T[0]["entry_t"]).days / 365.25

    print("=" * 70)
    print(f"{n} trades   {T[0]['entry_t']:%Y-%m-%d} -> {T[-1]['entry_t']:%Y-%m-%d}"
          f"   {yrs:.1f}y   {n / yrs:.1f}/yr")
    print("=" * 70)
    print(f"  win rate    {100 * len(wins) / n:.1f}%")
    print(f"  total       ${sum(pnl):,.0f}")
    print(f"  mean/trade  ${statistics.mean(pnl):+.2f}"
          f"   se ${statistics.stdev(pnl) / math.sqrt(n):.2f}"
          f"   t = {tstat(pnl):+.2f}")
    print(f"  mean return {statistics.mean([t['ret'] for t in T]):+.4f}%"
          f"   t = {tstat([t['ret'] for t in T]):+.2f}")

    worst = min(T, key=lambda t: t["pnl"])
    ex = [t["pnl"] for t in T if t is not worst]
    print(f"\n  worst trade {worst['entry_t']:%Y-%m-%d} ${worst['pnl']:,.0f}")
    print(f"  excluding it: total ${sum(ex):,.0f}  mean ${statistics.mean(ex):+.2f}"
          f"  t = {tstat(ex):+.2f}")

    print("\n  exit reason:")
    for k, c in Counter(t["signal"] for t in T).most_common():
        sub = [t["pnl"] for t in T if t["signal"] == k]
        print(f"    {k:<12}{c:>4}  mean ${statistics.mean(sub):+8.2f}  total ${sum(sub):+8.0f}")

    print("\n  by side:")
    for s in ("long", "short"):
        sub = [t for t in T if t["side"] == s]
        if not sub:
            continue
        w = sum(1 for t in sub if t["pnl"] > 0)
        print(f"    {s:<6}{len(sub):>5}  win {100 * w / len(sub):4.1f}%"
              f"  mean ${statistics.mean([t['pnl'] for t in sub]):+8.2f}")

    # A losing bracket exit means the stop was touched, so the gross loss IS
    # the risk that was on. Winners cannot be measured this way.
    stopped = [t for t in T if t["signal"] in ("L exit", "S exit") and t["pnl"] < 0]
    if stopped:
        risk = [abs(t["pnl"] + t["comm"]) / 100.0 for t in stopped]
        rp = [100 * r / t["entry_p"] for r, t in zip(risk, stopped)]
        print(f"\n  stop distance (from {len(stopped)} stopped-out trades):")
        print(f"    median ${statistics.median(risk):.2f}/share"
              f"  = {statistics.median(rp):.3f}% of spot")

    print("\n  by era:")
    lo = T[0]["entry_t"].year
    while lo <= T[-1]["entry_t"].year:
        sub = [t for t in T if lo <= t["entry_t"].year < lo + 5]
        if sub:
            w = sum(1 for t in sub if t["pnl"] > 0)
            print(f"    {lo}-{lo + 4}  {len(sub):>4} trades  win {100 * w / len(sub):4.1f}%"
                  f"  total ${sum(t['pnl'] for t in sub):+8.0f}")
        lo += 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
