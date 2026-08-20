"""The momentum strategy, run through the PRODUCTION path.

    python run_momentum.py                 # today's target portfolio, dry run
    python run_momentum.py --backtest      # replay history through the broker
    python run_momentum.py --equity 1000

`research/cross_momentum.py` measures the strategy. THIS runs it -- ranking,
target weights, order diff, dollar accounting, fills. Two independent
implementations of one idea.

They must agree. `--backtest` replays the same months through `SimBroker` and
prints both numbers side by side, because a research result that the trading
path cannot reproduce is a description of a strategy nobody is going to run.
This is the same discipline as `verify_bot.py [8]`, which found five silent
divergences between the engine and the harness.

Nothing here places a real order. `execute()` defaults to dry_run, and the
live Alpaca path additionally needs a typed acknowledgement.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys

from bot.broker import SimBroker, execute
from bot.portfolio import equal_weight, rebalance
from research.cross_momentum import (DECILE, ann, load_prices, month_ends, ret,
                                     sharpe)

FORM, SKIP = 12, 2          # the pre-registered winner from experiment 11
INVESTED = 0.98             # 2% cash buffer against fills worse than quoted


def rank(px, mes, k: int) -> list[str]:
    """Top `DECILE` symbols by formation return, as of month-end index `k`.

    Reads only prices at or before mes[k - SKIP]; the skip is what keeps
    short-term reversal out of the ranking."""
    rank_start, rank_end = mes[k - FORM], mes[k - SKIP]
    elig = []
    for s, series in px.items():
        if mes[k] not in series:
            continue
        r = ret(series, rank_start, rank_end)
        if not math.isnan(r):
            elig.append((r, s))
    elig.sort(reverse=True)
    return [s for _, s in elig[:DECILE]]


def backtest(px, mes, equity: float, cost_bp: float) -> tuple[list[float], float]:
    """Replay through the real portfolio + broker path. Returns (monthly, final)."""
    broker = SimBroker(cash=equity, prices={}, slippage_bp=cost_bp)
    monthly, prev_eq = [], equity

    for k in range(FORM, len(mes) - 1):
        d = mes[k]
        marks = {s: series[d] for s, series in px.items() if d in series}
        broker.mark(marks)
        acct = broker.account()

        picks = [s for s in rank(px, mes, k) if s in marks]
        if len(picks) < DECILE:
            continue

        # no_trade_band=0: the research code rebalances fully every month, so
        # the comparison has to as well. The band is a live-trading feature and
        # turning it on here would compare two different strategies.
        plan = rebalance(equal_weight(picks, INVESTED), acct.positions,
                         acct.equity, no_trade_band=0.0, min_trade_notional=0.01)
        rep = execute(plan, broker, dry_run=False)
        if rep.failed:
            print(f"  {d}: {len(rep.failed)} orders failed -- {rep.failed[0][1][:70]}")

        nxt = mes[k + 1]
        broker.mark({s: series[nxt] for s, series in px.items() if nxt in series})
        eq = broker.account().equity
        monthly.append(eq / prev_eq - 1.0)
        prev_eq = eq

    return monthly, prev_eq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--cost-bp", type=float, default=5.0)
    args = ap.parse_args()

    dates, px, spy = load_prices()
    mes = month_ends(dates)

    if not args.backtest:
        k = len(mes) - 1
        picks = rank(px, mes, k)
        targets = equal_weight(picks, INVESTED)
        plan = rebalance(targets, {}, args.equity, no_trade_band=0.0)
        print(f"\n12-{SKIP} momentum, formation {mes[k - FORM]} -> {mes[k - SKIP]}")
        print(f"as of {mes[k]},  equity ${args.equity:,.2f}\n")
        print(f"  {'sym':<7}{'weight':>8}{'notional':>12}{'last':>11}")
        for t, o in zip(targets, plan.orders):
            last = px[t.symbol][mes[k]]
            print(f"  {t.symbol:<7}{100*t.weight:>7.1f}%{o.notional:>12.2f}{last:>11.2f}")
        print(f"\n  invested ${plan.gross_traded:,.2f}   cash ${plan.cash_after:,.2f}")
        whole = sum(px[s][mes[k]] for s in picks)
        print(f"  whole shares would need ${whole:,.2f} "
              f"({whole / args.equity:.1f}x this account)")
        rep = execute(plan, SimBroker(args.equity), dry_run=True)
        print(f"  {rep.reconciled}")
        return 0

    print(f"\nBACKTEST -- production path vs research path")
    print(f"{len(px)} symbols, {len(mes)} month-ends, "
          f"{FORM}-{SKIP} formation, {args.cost_bp:.1f}bp/side\n")

    monthly, final = backtest(px, mes, args.equity, args.cost_bp)
    print(f"  production path : {ann(monthly):+7.2f}%/yr   SR {sharpe(monthly):+.2f}   "
          f"n {len(monthly)}   ${args.equity:,.0f} -> ${final:,.2f}")

    from research.cross_momentum import run
    research, _ = run(mes, px, FORM, SKIP, args.cost_bp, None)
    print(f"  research path   : {ann(research):+7.2f}%/yr   SR {sharpe(research):+.2f}   "
          f"n {len(research)}")

    n = min(len(monthly), len(research))
    if n:
        diffs = [a - b for a, b in zip(monthly[-n:], research[-n:])]
        worst = max(abs(d) for d in diffs)
        mean = statistics.mean(diffs)
        print(f"\n  monthly difference: mean {1e4*mean:+.2f}bp, "
              f"worst {1e4*worst:.2f}bp over {n} months")
        gap = abs(ann(monthly) - ann(research))
        if gap < 1.0:
            print(f"  AGREE -- annualised gap {gap:.2f}pp is within the cash-buffer")
            print(f"  and slippage-model difference. The bot runs what was measured.")
        else:
            print(f"  DISAGREE -- annualised gap {gap:.2f}pp. One of the two is")
            print(f"  wrong, and the research number is not evidence about the bot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
