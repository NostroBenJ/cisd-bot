"""Target weights in, orders out. Broker-agnostic.

A strategy says what it WANTS to hold, as fractions of equity. This module
works out the orders that get there from what is currently held. Nothing in
here knows about any particular broker, which is the point -- the same diff
runs against a paper account, a live account, or a pure simulation, so the
thing that is tested is the thing that trades.

Three decisions worth stating, because each has a reason:

**Orders are in NOTIONAL DOLLARS, not shares.** At $1,000 equity, one share of
each of the current momentum decile costs $3,589 -- the portfolio is
unexpressible in whole shares. Fractional/notional orders are not a nicety
here, they are the difference between holding ten names and holding three.

**Sells are emitted before buys.** Proceeds fund purchases, and in a cash
account they are not even available until settlement. A buy-first ordering
either rejects or quietly leverages, depending on the broker.

**Small drifts are not traded.** A no-trade band means a position that has
wandered 0.4% off target is left alone. Rebalancing to the dollar every period
converts an edge into commission and spread -- and the smaller the account, the
more of the edge that is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Target:
    """A desired holding, as a fraction of total equity."""

    symbol: str
    weight: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0) or math.isnan(self.weight):
            raise ValueError(f"weight for {self.symbol} must be in [0,1], got {self.weight}")


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str            # "buy" | "sell"
    notional: float      # dollars, always positive
    reason: str

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be buy or sell, got {self.side!r}")
        if not (self.notional > 0):
            raise ValueError(f"notional must be positive, got {self.notional}")


@dataclass
class RebalancePlan:
    orders: list[Order] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    cash_after: float = 0.0

    @property
    def gross_traded(self) -> float:
        return sum(o.notional for o in self.orders)

    def turnover(self, equity: float) -> float:
        """TWO-WAY turnover: both the sells and the buys count.

        Replacing four names in a ten-name portfolio reads as 0.80, not 0.40.
        The momentum literature usually quotes the one-way figure, so halve
        this before comparing to a paper -- and note that `turnover_cost` in
        research/cross_momentum.py charges both sides too, consistently."""
        return self.gross_traded / equity if equity > 0 else math.nan


def equal_weight(symbols: list[str], invested: float = 1.0) -> list[Target]:
    """Equal weights over `symbols`, summing to `invested`.

    `invested` below 1.0 leaves a deliberate cash buffer. Fully invested is
    the wrong default for a cash account, where a fill at a worse price than
    quoted can otherwise overdraw the balance."""
    if not symbols:
        return []
    w = invested / len(symbols)
    return [Target(s, w) for s in symbols]


def rebalance(
    targets: list[Target],
    positions: dict[str, float],      # symbol -> current market value, dollars
    equity: float,
    *,
    min_trade_notional: float = 1.0,
    no_trade_band: float = 0.005,     # fraction of equity
    allow_leverage: bool = False,
) -> RebalancePlan:
    """Orders that move `positions` toward `targets`.

    `positions` is market VALUE per symbol, not share count -- the diff is
    computed in dollars throughout so no price lookup is needed and no rounding
    to share sizes creeps in. The broker adapter converts to whatever the API
    wants.
    """
    if not (equity > 0) or math.isnan(equity):
        raise ValueError(f"equity must be positive, got {equity}")

    total_w = sum(t.weight for t in targets)
    if total_w > 1.0 + 1e-9 and not allow_leverage:
        raise ValueError(
            f"target weights sum to {total_w:.4f}, which is leverage. "
            f"Pass allow_leverage=True only for a margin account that intends it."
        )
    seen = [t.symbol for t in targets]
    if len(seen) != len(set(seen)):
        raise ValueError("duplicate symbol in targets")

    want = {t.symbol: t.weight * equity for t in targets}
    have = dict(positions)
    plan = RebalancePlan()
    band = no_trade_band * equity

    # Sells first: exits, then trims. Proceeds fund the buys.
    for sym in sorted(set(have) | set(want)):
        cur, tgt = have.get(sym, 0.0), want.get(sym, 0.0)
        d = tgt - cur
        if d >= 0:
            continue
        size = -d
        if sym not in want:
            plan.orders.append(Order(sym, "sell", size, "exit -- not in target"))
        elif size < max(band, min_trade_notional):
            plan.skipped.append(f"{sym} trim {size:.2f} inside band")
        else:
            plan.orders.append(Order(sym, "sell", size, "trim to target"))

    for sym in sorted(set(have) | set(want)):
        cur, tgt = have.get(sym, 0.0), want.get(sym, 0.0)
        d = tgt - cur
        if d <= 0:
            continue
        if d < max(band, min_trade_notional):
            plan.skipped.append(f"{sym} add {d:.2f} inside band")
            continue
        plan.orders.append(
            Order(sym, "buy", d, "open" if cur == 0 else "add to target"))

    plan.cash_after = equity - sum(want.values())
    return plan


def drift(targets: list[Target], positions: dict[str, float], equity: float) -> float:
    """Total absolute deviation from target, as a fraction of equity.

    The number to schedule on: rebalance when drift exceeds a threshold rather
    than on a fixed calendar, and turnover falls without the portfolio wandering."""
    if not (equity > 0):
        return math.nan
    want = {t.symbol: t.weight * equity for t in targets}
    syms = set(want) | set(positions)
    return sum(abs(want.get(s, 0.0) - positions.get(s, 0.0)) for s in syms) / equity
