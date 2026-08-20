"""Verification for `bot/portfolio.py` and `bot/broker.py`.

    python verify_portfolio.py

The load-bearing checks are [3] (sells precede buys, or a cash account rejects
the plan) and [6] (a full round trip through the simulator conserves value to
the cent). Portfolio arithmetic that does not close is how an account quietly
ends up leveraged.
"""

from __future__ import annotations

import math
import sys

from bot.broker import LIVE_ACK, AlpacaBroker, ExecutionReport, SimBroker, execute
from bot.portfolio import Order, Target, drift, equal_weight, rebalance

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


def main() -> int:
    TOP10 = ["INTC", "CAT", "VLO", "CSCO", "GOOGL", "FDX", "LLY", "MRK", "DAL", "HAL"]
    PX = {"INTC": 92.80, "CAT": 816.15, "VLO": 346.26, "CSCO": 110.55,
          "GOOGL": 344.72, "FDX": 328.38, "LLY": 1280.34, "MRK": 152.20,
          "DAL": 83.29, "HAL": 35.02}

    print("\n[1] targets validate their own inputs")
    for bad in (-0.1, 1.5, float("nan")):
        try:
            Target("X", bad)
            check(f"weight {bad} rejected", False)
        except ValueError:
            check(f"weight {bad} rejected", True)
    try:
        Order("X", "hold", 10.0, "")
        check("bad side rejected", False)
    except ValueError:
        check("bad side rejected", True)
    try:
        Order("X", "buy", 0.0, "")
        check("zero notional rejected", False)
    except ValueError:
        check("zero notional rejected", True)

    print("\n[2] leverage is refused unless asked for")
    over = [Target("A", 0.6), Target("B", 0.6)]
    try:
        rebalance(over, {}, 1000.0)
        check("weights summing to 1.2 rejected", False)
    except ValueError:
        check("weights summing to 1.2 rejected", True)
    p = rebalance(over, {}, 1000.0, allow_leverage=True)
    check("allowed explicitly", len(p.orders) == 2)
    try:
        rebalance([Target("A", 0.5), Target("A", 0.5)], {}, 1000.0)
        check("duplicate symbol rejected", False)
    except ValueError:
        check("duplicate symbol rejected", True)

    print("\n[3] sells are emitted before buys")
    tg = equal_weight(["B", "C"], invested=1.0)
    plan = rebalance(tg, {"A": 500.0, "B": 100.0}, 1000.0)
    sides = [o.side for o in plan.orders]
    first_buy = sides.index("buy") if "buy" in sides else len(sides)
    last_sell = len(sides) - 1 - sides[::-1].index("sell") if "sell" in sides else -1
    check("every sell precedes every buy", last_sell < first_buy, str(sides))
    check("A is exited entirely",
          any(o.symbol == "A" and o.side == "sell" and abs(o.notional - 500) < 1e-6
              for o in plan.orders))

    print("\n[4] the no-trade band suppresses small drifts")
    tg = equal_weight(["A", "B"], invested=1.0)          # 500 / 500
    plan = rebalance(tg, {"A": 502.0, "B": 498.0}, 1000.0, no_trade_band=0.005)
    check("2-dollar drift on 1000 is skipped", not plan.orders, str(plan.orders))
    check("both skips are reported", len(plan.skipped) == 2, str(plan.skipped))
    plan = rebalance(tg, {"A": 560.0, "B": 440.0}, 1000.0, no_trade_band=0.005)
    check("60-dollar drift trades", len(plan.orders) == 2)

    print("\n[5] the real decile is expressible at $1,000")
    tg = equal_weight(TOP10, invested=0.98)
    plan = rebalance(tg, {}, 1000.0)
    check("ten orders produced", len(plan.orders) == 10, str(len(plan.orders)))
    check("all are buys", all(o.side == "buy" for o in plan.orders))
    notional = sum(o.notional for o in plan.orders)
    check("invests 98% of equity", abs(notional - 980.0) < 1e-6, f"{notional:.2f}")
    check("each position is ~$98", all(abs(o.notional - 98.0) < 1e-6 for o in plan.orders))
    whole = sum(PX.values())
    check("whole shares would have cost 3.5x the account", whole > 3500,
          f"${whole:,.2f}")
    print(f"       notional sizing: 10 names for $980.00")
    print(f"       whole shares:    10 names for ${whole:,.2f}  (impossible)")

    print("\n[6] a round trip through the simulator conserves value")
    b = SimBroker(cash=1000.0, prices=PX)
    a0 = b.account()
    check("starts flat at 1000", abs(a0.equity - 1000.0) < 1e-9 and not a0.positions)
    rep = execute(rebalance(equal_weight(TOP10, 0.98), {}, a0.equity), b, dry_run=False)
    check("all ten submitted", len(rep.submitted) == 10 and rep.ok, str(rep.failed[:2]))
    a1 = b.account()
    check("equity unchanged by a zero-cost fill", abs(a1.equity - 1000.0) < 1e-6,
          f"{a1.equity:.6f}")
    check("ten positions held", len(a1.positions) == 10, str(len(a1.positions)))
    check("cash is the 2% buffer", abs(a1.cash - 20.0) < 1e-6, f"{a1.cash:.4f}")
    check("drift from target is ~zero",
          drift(equal_weight(TOP10, 0.98), a1.positions, a1.equity) < 1e-9)

    print("\n[7] a rebalance to a new decile closes the old names")
    NEW = TOP10[:6] + ["NVDA", "AAPL", "MSFT", "KO"]
    b.mark({s: PX.get(s, 100.0) for s in NEW})
    a = b.account()
    plan = rebalance(equal_weight(NEW, 0.98), a.positions, a.equity)
    exits = [o for o in plan.orders if o.reason.startswith("exit")]
    check("the four dropped names are exited", len(exits) == 4,
          str(sorted(o.symbol for o in exits)))
    # TWO-WAY turnover: swapping 4 of 10 names sells 40% and buys 40%.
    check("two-way turnover is ~80%", 0.70 < plan.turnover(a.equity) < 0.85,
          f"{plan.turnover(a.equity):.3f}")
    rep = execute(plan, b, dry_run=False)
    check("rebalance executed cleanly", rep.ok, str(rep.failed[:2]))
    a2 = b.account()
    check("still ten positions", len(a2.positions) == 10, str(sorted(a2.positions)))
    check("equity still conserved", abs(a2.equity - 1000.0) < 1e-6, f"{a2.equity:.6f}")

    print("\n[8] execution refuses to send unless asked, every time")
    b2 = SimBroker(cash=1000.0, prices=PX)
    rep = execute(rebalance(equal_weight(TOP10, 0.98), {}, 1000.0), b2)
    check("dry_run is the DEFAULT", not rep.submitted and "DRY RUN" in rep.reconciled)
    check("nothing was filled", not b2.fills and not b2.account().positions)

    print("\n[9] cash accounting is real, not notional")
    b3 = SimBroker(cash=100.0, prices=PX)
    try:
        b3.submit(Order("CAT", "buy", 500.0, "oversized"))
        check("buying beyond cash raises", False)
    except ValueError:
        check("buying beyond cash raises", True)
    check("nothing was recorded on the failed order", not b3.fills)

    print("\n[10] live trading cannot be reached without the phrase")
    try:
        AlpacaBroker(live=True, acknowledgement="yes")
        check("wrong phrase refused", False)
    except PermissionError:
        check("wrong phrase refused", True)
    except RuntimeError as e:
        check("wrong phrase refused", False, f"credential check ran first: {e}")
    import os
    had = os.environ.get("ALPACA_API_KEY")
    if not had:
        try:
            AlpacaBroker(live=False)
            check("paper without credentials refuses", False)
        except RuntimeError:
            check("paper without credentials refuses clearly", True)
    else:
        print("       (ALPACA_API_KEY is set; skipping the no-credentials check)")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
