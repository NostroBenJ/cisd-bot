"""Is buying short-dated SPY premium ever +EV, and can we tell in advance?

    python -m research.vol_premium

**The question, stated as a trade.** Buy a one-day at-the-money straddle at
today's close; it expires tomorrow. You pay the implied move and receive the
realised move. Profit is `|actual move| - straddle cost`. That is the one
options position a level-2, single-leg, long-only account can actually put on,
and it is a direct bet against the variance risk premium.

**Why this is the cleanest test in the project.** A one-day forward horizon has
ZERO overlap: today's next-day return shares nothing with tomorrow's. Every
observation is independent, so the `sqrt(n_indep)` correction that the VRP
study needs for 21-day windows does not apply here. 1,051 independent
observations against 90 sessions of order flow and ~250 effective observations
in any 30-day-horizon VRP study.

**The prior is that this LOSES.** The variance risk premium is positive on
average -- implied exceeds realised -- which is exactly why selling premium is
the documented edge and buying it is the documented cost. So the unconditional
result should be negative, and if it is not, something is wrong with the
maths rather than right with the trade.

The real question is conditional: **are there identifiable days when implied is
cheap relative to what follows?** Those are the days a long-premium account is
on the right side of the premium instead of paying it.

Straddle price uses the zero-rate Black-Scholes at-the-money approximation,
`0.7979 * S * sigma * sqrt(T)` (that is `2 * phi(0)`, phi the standard normal
density). At the money and one day out this is accurate to well under a
percent, and it avoids needing a rate curve. Payoff assumes the straddle is
held to expiry and settles at intrinsic, which for a 0DTE position held to the
close is what happens.

Stdlib only.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from research.cboe import close_series

SPY_CSV = Path("C:/Users/jontr/Downloads/spy_vix_daily.csv")
TRADING_DAYS = 252.0

# Round-trip cost as a fraction of the straddle's price. A SPY 0DTE ATM
# straddle runs a couple of cents wide on each leg against a price of a few
# dollars; 2% of premium is a fair, slightly generous, estimate. It is a
# parameter because the conclusion should be checked against it, not hidden
# behind it.
COST_FRAC = 0.02


@dataclass
class Day:
    date: str
    spy: float
    ret: float          # next-day simple return, the straddle's payoff driver
    implied: float      # VIX1D as a decimal, e.g. 0.12
    straddle: float     # cost as a fraction of spot
    payoff: float       # |ret|
    pnl: float          # payoff - straddle - cost, as a fraction of spot


def load_spy() -> dict[str, float]:
    if not SPY_CSV.exists():
        raise FileNotFoundError(f"{SPY_CSV} not found -- run vrp_data.py to build it")
    out: dict[str, float] = {}
    with SPY_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["date"]] = float(row["spy"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def build(vol_series: str = "VIX1D", horizon_days: float = 1.0) -> list[Day]:
    """One row per day where both an implied vol and a NEXT close exist."""
    spy = load_spy()
    iv = close_series(vol_series)
    days = sorted(set(spy) & set(iv))
    out: list[Day] = []
    for i in range(len(days) - 1):
        d, nxt = days[i], days[i + 1]
        s0, s1 = spy[d], spy[nxt]
        if s0 <= 0 or s1 <= 0:
            continue
        sigma = iv[d] / 100.0
        if sigma <= 0:
            continue
        t = horizon_days / TRADING_DAYS
        straddle = 0.7979 * sigma * math.sqrt(t)      # fraction of spot
        ret = (s1 - s0) / s0
        payoff = abs(ret)
        pnl = payoff - straddle * (1.0 + COST_FRAC)
        out.append(Day(d, s0, ret, sigma, straddle, payoff, pnl))
    return out


def stats(xs: list[float]) -> tuple[float, float, float]:
    if len(xs) < 2:
        return math.nan, math.nan, math.nan
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs))
    return m, se, (m / se if se else math.nan)


def line(label: str, xs: list[float], scale: float = 10000.0) -> str:
    """Report in basis points of spot -- readable, and scale-free across a
    dataset where SPY ran from $390 to $770."""
    if len(xs) < 2:
        return f"  {label:<34} n={len(xs):<5} --"
    m, se, t = stats(xs)
    win = 100.0 * sum(1 for x in xs if x > 0) / len(xs)
    return (f"  {label:<34} n={len(xs):<5} {m*scale:+7.2f}bp  se {se*scale:5.2f}"
            f"  t = {t:+5.2f}  win {win:4.1f}%")


def deciles(rows: list[Day], key, name: str) -> None:
    """Sort by a predictor known at the close of day t, then report P&L by decile.

    The predictor must use ONLY information available before the return it is
    being asked to forecast. Every key below reads levels at day t and the
    outcome is day t+1."""
    vals = [(key(r), r.pnl) for r in rows if not math.isnan(key(r))]
    vals.sort(key=lambda kv: kv[0])
    n = len(vals)
    if n < 200:
        print(f"\n  {name}: too few rows ({n})")
        return
    print(f"\n  {name} -- deciles, low to high:")
    for q in range(10):
        lo, hi = q * n // 10, (q + 1) * n // 10
        chunk = [p for _, p in vals[lo:hi]]
        edge = f"{vals[lo][0]:.3f}-{vals[hi-1][0]:.3f}"
        m, se, t = stats(chunk)
        flag = "  <<<" if not math.isnan(t) and t > 2.0 else ""
        print(f"    d{q+1} [{edge:>13}]  n={len(chunk):<4} {m*10000:+7.2f}bp"
              f"  t = {t:+5.2f}{flag}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    rows = build("VIX1D")
    vix = close_series("VIX")
    v9 = close_series("VIX9D")
    v3 = close_series("VIX3M")
    vvix = close_series("VVIX")
    skew = close_series("SKEW")

    print("=" * 74)
    print("BUYING A ONE-DAY SPY STRADDLE")
    print("=" * 74)
    print(f"  {len(rows)} independent observations, {rows[0].date} -> {rows[-1].date}")
    print(f"  cost model: 0.7979*sigma*sqrt(1/252) of spot, plus {COST_FRAC:.0%} round trip")
    print(f"  a 1-day horizon has NO overlapping windows, so n_indep = n")

    pnl = [r.pnl for r in rows]
    print("\n  UNCONDITIONAL (buy every day):")
    print(line("every day", pnl))
    print(f"    median straddle cost {statistics.median([r.straddle for r in rows])*10000:.1f}bp"
          f"   median move {statistics.median([r.payoff for r in rows])*10000:.1f}bp")
    m, se, t = stats(pnl)
    verdict = ("NEGATIVE as expected -- the variance risk premium is real and "
               "buying premium pays it") if m < 0 else "POSITIVE -- suspect the maths before the market"
    print(f"    {verdict}")

    # Robustness on the headline number, same tests that killed first_bar.
    srt = sorted(pnl)
    k = max(1, len(srt) // 100)
    print(line("  excluding best 1%", srt[:-k]))
    print(line("  excluding worst 1%", srt[k:]))
    half = len(rows) // 2
    print(line("  first half", [r.pnl for r in rows[:half]]))
    print(line("  second half", [r.pnl for r in rows[half:]]))

    print("\n" + "-" * 74)
    print("  CONDITIONAL -- is there a subset where buying premium wins?")
    print("  Every predictor is known at the close of day t; the outcome is t+1.")
    print("-" * 74)

    deciles(rows, lambda r: r.implied * 100.0, "VIX1D level")
    deciles(rows, lambda r: r.implied * 100.0 / vix.get(r.date, math.nan),
            "VIX1D / VIX  (short vol vs 30-day)")
    deciles(rows, lambda r: v9.get(r.date, math.nan) / vix.get(r.date, math.nan),
            "VIX9D / VIX")
    deciles(rows, lambda r: vix.get(r.date, math.nan) / v3.get(r.date, math.nan),
            "VIX / VIX3M  (term structure slope)")
    deciles(rows, lambda r: vvix.get(r.date, math.nan), "VVIX  (vol of vol)")
    deciles(rows, lambda r: skew.get(r.date, math.nan), "SKEW  (tail pricing)")

    print("\n" + "=" * 74)
    print("  Six predictors x ten deciles is sixty comparisons. At p<0.05 about")
    print("  three will clear t=2 by chance alone, so a single flagged decile is")
    print("  NOT a finding. What would count: a monotone gradient across deciles,")
    print("  surviving a holdout and the leave-out-the-best-1% test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
