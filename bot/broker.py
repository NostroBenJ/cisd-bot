"""Broker adapters. One interface, three implementations, no special cases.

    SimBroker    -- pure local simulation, deterministic, used by the tests
    AlpacaBroker -- real API, PAPER by default; live requires a typed phrase
    (Robinhood)  -- deliberately absent, see below

Every adapter answers the same three questions -- what is the account worth,
what does it hold, and please place this order -- so the strategy above cannot
tell which one is attached. That symmetry is what makes a paper track record
mean anything about live behaviour.

**Reconciliation is not optional.** After any order, positions are re-read from
the broker and the broker's answer is treated as truth. A bot whose internal
idea of its holdings drifts from reality is the failure that turns one bad fill
into an unbounded one -- it keeps "correcting" toward a position it does not
have.

**Why there is no RobinhoodBroker.** Robinhood publishes no official REST API
for equities; the only supported programmatic path is the agentic MCP, which
runs an AI agent in the order path. That is the wrong shape for a mechanical
monthly rebalance, which must be deterministic and repeatable. The unofficial
reverse-engineered libraries violate Robinhood's terms and require live
credentials in a script. Neither belongs here. Robinhood stays a manual
destination: print the plan, place the orders by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from bot.portfolio import Order

LIVE_ACK = "I accept that this places real orders with real money"


@dataclass
class Account:
    equity: float
    cash: float
    positions: dict[str, float]      # symbol -> market value in dollars


class SimBroker:
    """Deterministic local simulation. Touches nothing real.

    Fills instantly at the supplied price with an optional slippage in basis
    points. Not a market model -- a way to exercise the plumbing and prove the
    portfolio arithmetic closes."""

    mode = "sim"

    def __init__(self, cash: float, prices: dict[str, float] | None = None,
                 slippage_bp: float = 0.0):
        self._cash = float(cash)
        self._pos: dict[str, float] = {}      # symbol -> shares
        self.prices = dict(prices or {})
        self.slippage_bp = slippage_bp
        self.fills: list[tuple[str, str, float, float]] = []

    def mark(self, prices: dict[str, float]) -> None:
        self.prices.update(prices)

    def account(self) -> Account:
        pos = {s: q * self.prices[s] for s, q in self._pos.items()
               if q > 0 and s in self.prices}
        return Account(self._cash + sum(pos.values()), self._cash, pos)

    def submit(self, order: Order) -> None:
        px = self.prices.get(order.symbol)
        if not px or px <= 0:
            raise ValueError(f"no price for {order.symbol}")
        slip = px * self.slippage_bp / 1e4
        fill = px + slip if order.side == "buy" else px - slip
        qty = order.notional / fill
        if order.side == "buy":
            if order.notional > self._cash + 1e-9:
                raise ValueError(
                    f"insufficient cash: need {order.notional:.2f}, have {self._cash:.2f}")
            self._cash -= order.notional
            self._pos[order.symbol] = self._pos.get(order.symbol, 0.0) + qty
        else:
            held = self._pos.get(order.symbol, 0.0)
            qty = min(qty, held)
            self._cash += qty * fill
            self._pos[order.symbol] = held - qty
            if self._pos[order.symbol] <= 1e-12:
                self._pos.pop(order.symbol, None)
        self.fills.append((order.symbol, order.side, qty, fill))

    def reconcile(self) -> tuple[bool, str]:
        return True, "sim: internal state is the truth"


class AlpacaBroker:
    """Alpaca REST. PAPER by default.

    Live trading requires `live=True` AND the typed acknowledgement -- not a
    config flag someone can set without reading it. Credentials come from the
    environment (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`), never from a file in
    this repo and never from an argument that could end up in a shell history.

    The `alpaca` SDK is imported inside the methods so the rest of the package
    stays importable without it.
    """

    def __init__(self, live: bool = False, acknowledgement: str = ""):
        if live and acknowledgement != LIVE_ACK:
            raise PermissionError(
                "Live trading requires the explicit typed acknowledgement. "
                "If that felt like an obstacle, that was the intent."
            )
        self.live = live
        self.mode = "alpaca-live" if live else "alpaca-paper"
        self.key = os.environ.get("ALPACA_API_KEY", "")
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not self.key or not self.secret:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY are not set in the "
                "environment. Create a free paper account at alpaca.markets, "
                "then set both before running."
            )

    def _client(self):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise RuntimeError("pip install alpaca-py") from e
        return TradingClient(self.key, self.secret, paper=not self.live)

    def account(self) -> Account:
        c = self._client()
        a = c.get_account()
        pos = {p.symbol: float(p.market_value) for p in c.get_all_positions()}
        return Account(float(a.equity), float(a.cash), pos)

    def submit(self, order: Order) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        # A FULL EXIT closes by position, never by notional. A notional sell is
        # converted to shares at the CURRENT price, so a position that has
        # fallen since the plan was built implies more shares than are held and
        # the order is rejected outright -- which is exactly what happened on
        # the first live rebalance (FDX, "insufficient qty available"). The
        # broker already knows the exact quantity; ask it to close.
        if order.reason.startswith("exit"):
            self._client().close_position(order.symbol)
            return

        req = MarketOrderRequest(
            symbol=order.symbol,
            notional=round(order.notional, 2),
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            # DAY, not GTC. An order that outlives the decision that produced
            # it will eventually fill on a thesis nobody still holds.
            time_in_force=TimeInForce.DAY,
        )
        self._client().submit_order(req)

    def reconcile(self) -> tuple[bool, str]:
        """Re-read from the broker. The broker is always right."""
        try:
            a = self.account()
        except Exception as e:                     # noqa: BLE001 - report, never guess
            return False, f"could not read account: {type(e).__name__}: {e}"
        return True, f"{len(a.positions)} positions, equity {a.equity:,.2f}"


@dataclass
class ExecutionReport:
    submitted: list[Order] = field(default_factory=list)
    failed: list[tuple[Order, str]] = field(default_factory=list)
    reconciled: str = ""

    @property
    def ok(self) -> bool:
        return not self.failed


def execute(plan, broker, *, dry_run: bool = True) -> ExecutionReport:
    """Send a RebalancePlan to a broker.

    `dry_run` defaults to TRUE. Placing orders is the one thing in this package
    that cannot be undone, so it is the one thing that must be asked for
    explicitly every single time."""
    rep = ExecutionReport()
    if dry_run:
        rep.reconciled = f"DRY RUN -- {len(plan.orders)} orders not sent"
        return rep

    for o in plan.orders:                      # sells already ordered first
        try:
            broker.submit(o)
            rep.submitted.append(o)
        except Exception as e:                 # noqa: BLE001
            rep.failed.append((o, f"{type(e).__name__}: {e}"))

    ok, msg = broker.reconcile()
    rep.reconciled = msg if ok else f"RECONCILE FAILED: {msg}"
    return rep
