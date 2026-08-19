"""Executors. Shadow and live share one interface so the logic above them
cannot tell which is running.

That symmetry is the whole design. If shadow and live took different code
paths, a shadow track record would prove nothing about live behaviour — which
is the usual reason paper trading and real trading diverge.

`LiveExecutor` refuses to construct unless handed an explicit acknowledgement.
Not a boolean buried in a config file: a value someone had to type on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    symbol: str
    direction: int          # +1 long, -1 short, 0 flat
    shares: int
    entry: float
    stop: float
    target: float
    opened_at: int

    @property
    def is_open(self) -> bool:
        return self.direction != 0 and self.shares > 0


@dataclass
class Fill:
    ts: int
    symbol: str
    side: str               # "buy" | "sell"
    shares: int
    price: float
    mode: str


class ShadowExecutor:
    """Records what would have happened. Touches nothing real."""

    mode = "shadow"

    def __init__(self):
        self.fills: list[Fill] = []
        self.position: Position | None = None

    def open(self, ts: int, symbol: str, direction: int, shares: int,
             price: float, stop: float, target: float) -> Position:
        side = "buy" if direction > 0 else "sell"
        self.fills.append(Fill(ts, symbol, side, shares, price, self.mode))
        self.position = Position(symbol, direction, shares, price, stop, target, ts)
        return self.position

    def close(self, ts: int, price: float) -> float:
        """Close the open position. Returns realised P&L in dollars."""
        p = self.position
        if p is None or not p.is_open:
            return 0.0
        side = "sell" if p.direction > 0 else "buy"
        self.fills.append(Fill(ts, p.symbol, side, p.shares, price, self.mode))
        pnl = (price - p.entry) * p.direction * p.shares
        self.position = None
        return pnl

    def reconcile(self) -> tuple[bool, str]:
        """Shadow has no external truth to reconcile against."""
        return True, "shadow: no broker state"


LIVE_ACK = "I have a validated signal and accept real losses"


class LiveExecutor:
    """Places real orders. Not implemented, and refuses to pretend.

    When this is written it must, in this order: place the order, then READ THE
    POSITION BACK from the broker, and treat the broker's answer as truth. A
    bot whose internal position drifts from the broker's is the failure that
    turns one bad fill into an unbounded one.
    """

    mode = "live"

    def __init__(self, acknowledgement: str):
        if acknowledgement != LIVE_ACK:
            raise PermissionError(
                "LiveExecutor requires an explicit typed acknowledgement. "
                "If that felt like an obstacle, that was the intent."
            )
        raise NotImplementedError(
            "Live execution is not built. Nothing has earned it: eight "
            "experiments in FINDINGS.md, no signal with a passing "
            "ValidationRecord."
        )
