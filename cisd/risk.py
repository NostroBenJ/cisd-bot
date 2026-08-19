"""Hard limits for an unattended bot.

The bot places and manages orders with no confirmation dialog, which is what
makes these non-optional. Every limit is a percentage of session-starting
equity, so none of them wait on the account being funded -- the only thing that
does is the equity figure itself, supplied at `start_session`.

**Every gate fails closed.** An unknown equity, an unparseable state, a missing
input: the answer is "no trade". The asymmetry is deliberate. A bot that
declines a good signal costs an opportunity; a bot that trades on a
half-initialised state costs money, and it does so at machine speed.

Sizing on long options defaults to treating the **full premium as the risk**,
not the delta-estimated loss at the underlying stop. Premium is the amount that
can actually be lost -- a gap through the stop, a halt, or a fast repricing all
resolve toward total loss on a short-dated long option, and the delta estimate
quietly assumes an orderly exit that a bad morning will not provide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .sessions import parse_session
from .signal import Signal


@dataclass(frozen=True)
class RiskLimits:
    """All limits as percentages of session-starting equity.

    Defaults are deliberately tight. They are a starting point for a bot with
    no measured edge, not a recommendation -- and they should be revisited only
    against a real distribution from stage 3, never against a good week."""

    max_daily_loss_pct: float = 3.0
    max_risk_per_trade_pct: float = 1.0
    max_trades_per_day: int = 4
    max_concurrent_positions: int = 2
    max_contracts_per_trade: int = 10

    # Everything is closed at this NY time regardless of state. The traded
    # window governs ENTRIES; this governs EXITS, and it must be earlier than
    # the close so a stuck order still has time to fill.
    hard_flat_time: str = "1155"

    # Presence of this file blocks new entries and forces a flatten. A file is
    # the right mechanism because it works when the process is wedged, needs no
    # IPC, and the user can create it from anywhere.
    kill_file: str = "data/KILL"

    # A signal whose stop is further away than this (as a fraction of price) is
    # refused rather than sized down. A very wide stop usually means the block
    # formed on a disorderly bar, and sizing to one contract does not make that
    # a good trade.
    max_stop_distance_pct: float = 1.5

    sizing_mode: str = "premium"  # "premium" | "delta"


@dataclass
class Decision:
    allowed: bool
    reason: str
    contracts: int = 0

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class RiskGate:
    """Session-scoped risk state. One per trading day.

    `start_session` must be called before anything else; until it is, every
    gate refuses."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    equity_at_open: float = math.nan
    realised_pnl: float = 0.0
    trades_today: int = 0
    open_positions: int = 0
    _session: str | None = None
    _halted: str | None = None

    def start_session(self, equity: float, session_date: str) -> None:
        if not (equity > 0) or math.isnan(equity):
            raise ValueError(f"session equity must be positive, got {equity!r}")
        self.equity_at_open = equity
        self.realised_pnl = 0.0
        self.trades_today = 0
        self.open_positions = 0
        self._session = session_date
        self._halted = None

    @property
    def ready(self) -> bool:
        return self._session is not None and self.equity_at_open > 0

    # ── budgets ─────────────────────────────────────────────────────────────

    @property
    def daily_loss_budget(self) -> float:
        return self.equity_at_open * self.limits.max_daily_loss_pct / 100.0

    @property
    def per_trade_budget(self) -> float:
        return self.equity_at_open * self.limits.max_risk_per_trade_pct / 100.0

    @property
    def loss_remaining(self) -> float:
        """How much more can be lost today before the bot stops."""
        return self.daily_loss_budget + min(self.realised_pnl, 0.0)

    # ── state transitions ───────────────────────────────────────────────────

    def record_open(self, contracts: int, cost: float) -> None:
        self.open_positions += 1
        self.trades_today += 1

    def record_close(self, pnl: float) -> None:
        self.realised_pnl += pnl
        self.open_positions = max(0, self.open_positions - 1)
        if self.realised_pnl <= -self.daily_loss_budget:
            self._halted = (
                f"daily loss limit hit: {self.realised_pnl:.2f} "
                f"against a {self.daily_loss_budget:.2f} budget"
            )

    def halt(self, reason: str) -> None:
        self._halted = reason

    @property
    def halted_reason(self) -> str | None:
        return self._halted

    # ── gates ───────────────────────────────────────────────────────────────

    def kill_file_present(self) -> bool:
        return Path(self.limits.kill_file).exists()

    def must_flatten(self, minute_of_day_ny: int) -> str | None:
        """Reason to close everything now, or None."""
        if self.kill_file_present():
            return f"kill file present at {self.limits.kill_file}"
        flat_at, _ = parse_session(f"{self.limits.hard_flat_time}-{self.limits.hard_flat_time}")
        if minute_of_day_ny >= flat_at:
            return f"hard flat time {self.limits.hard_flat_time} reached"
        if self._halted:
            return self._halted
        return None

    def _pre_trade_gates(self, minute_of_day_ny: int) -> Decision | None:
        """The gates that do not depend on WHAT is being bought.

        Shared by the options path and the shares path so the two can never
        drift. Returns the refusal, or None to mean carry on."""
        if not self.ready:
            return Decision(False, "risk gate not initialised for a session")

        flat = self.must_flatten(minute_of_day_ny)
        if flat:
            return Decision(False, f"flattening: {flat}")

        if self.trades_today >= self.limits.max_trades_per_day:
            return Decision(False, f"daily trade cap ({self.limits.max_trades_per_day}) reached")

        if self.open_positions >= self.limits.max_concurrent_positions:
            return Decision(False, f"already holding {self.open_positions} positions")

        if self.loss_remaining <= 0:
            return Decision(False, "daily loss budget exhausted")

        return None

    def _stop_too_wide(self, sig: Signal) -> Decision | None:
        stop_pct = 100.0 * sig.risk_per_share / max(sig.entry, 1e-9)
        if stop_pct > self.limits.max_stop_distance_pct:
            return Decision(
                False,
                f"stop is {stop_pct:.2f}% away, over the "
                f"{self.limits.max_stop_distance_pct}% limit",
            )
        return None

    def can_open_shares(self, sig: Signal, price: float,
                        minute_of_day_ny: int) -> Decision:
        """Shares of the underlying, not option contracts.

        This exists because the research harness measures the UNDERLYING, and
        a shadow run has to trade the same instrument the statistics were
        computed on. Sizing here is exact rather than conservative: the loss at
        the stop is known, so risk budget / stop distance is the share count,
        with no premium-versus-delta question to answer.

        The options path stays the default for live trading -- see `can_open`
        and the note on `sizing_mode`."""
        blocked = self._pre_trade_gates(minute_of_day_ny)
        if blocked is not None:
            return blocked

        if not (price > 0) or math.isnan(price):
            return Decision(False, f"no usable price ({price!r})")

        wide = self._stop_too_wide(sig)
        if wide is not None:
            return wide

        risk = sig.risk_per_share
        if not (risk > 0) or math.isnan(risk):
            return Decision(False, f"stop distance is not usable ({risk!r})")

        budget = self.budget_for_trade()
        shares = int(budget // risk)

        # Cash account, no margin: the position cannot cost more than equity.
        # A tight stop otherwise sizes to a notional the account cannot pay for.
        affordable = int(self.equity_at_open // price)
        shares = min(shares, affordable)

        if shares < 1:
            return Decision(
                False,
                f"budget {budget:.2f} at a {risk:.2f} stop, and "
                f"{self.equity_at_open:.2f} equity at {price:.2f}, "
                f"will not cover one share",
            )
        return Decision(True, "ok", shares)

    def can_open(self, sig: Signal, contract_price: float,
                 minute_of_day_ny: int, delta: float = 0.5) -> Decision:
        """May the bot open this position, and at what size?

        `contract_price` is the option's ask in dollars per share (so a $1.50
        quote costs $150 for one contract)."""
        blocked = self._pre_trade_gates(minute_of_day_ny)
        if blocked is not None:
            return blocked

        if not (contract_price > 0) or math.isnan(contract_price):
            return Decision(False, f"no usable contract price ({contract_price!r})")

        wide = self._stop_too_wide(sig)
        if wide is not None:
            return wide

        contracts = self.size(sig, contract_price, delta)
        if contracts < 1:
            return Decision(
                False,
                f"budget {self.budget_for_trade():.2f} will not cover one "
                f"contract at {contract_price * 100:.2f}",
            )
        return Decision(True, "ok", contracts)

    def budget_for_trade(self) -> float:
        """Risk budget for the next trade: the per-trade limit, further capped
        by what is left of the daily budget. Without the second term a bot one
        dollar from its daily limit would still risk a full position."""
        return max(0.0, min(self.per_trade_budget, self.loss_remaining))

    def size(self, sig: Signal, contract_price: float, delta: float = 0.5) -> int:
        """Contracts to buy, given the risk budget.

        `premium` mode treats the whole premium as at risk, which is the only
        bound that a gap cannot exceed. `delta` mode estimates the loss at the
        underlying stop and will size larger; it assumes an orderly exit."""
        budget = self.budget_for_trade()
        if budget <= 0 or contract_price <= 0:
            return 0

        if self.limits.sizing_mode == "delta":
            if not (0 < delta <= 1):
                return 0
            risk_per_contract = delta * sig.risk_per_share * 100.0
            # Never claim less risk than a total loss of premium would be
            # cheaper -- a deep-ITM contract's delta-loss can exceed premium.
            risk_per_contract = min(risk_per_contract, contract_price * 100.0)
        else:
            risk_per_contract = contract_price * 100.0

        if risk_per_contract <= 0:
            return 0
        n = int(budget // risk_per_contract)
        return max(0, min(n, self.limits.max_contracts_per_trade))

    def status(self) -> str:
        if not self.ready:
            return "risk gate: not initialised"
        return (
            f"risk gate: equity {self.equity_at_open:,.2f} | "
            f"pnl {self.realised_pnl:+,.2f} | "
            f"loss budget {self.loss_remaining:,.2f} of {self.daily_loss_budget:,.2f} | "
            f"trades {self.trades_today}/{self.limits.max_trades_per_day} | "
            f"open {self.open_positions}/{self.limits.max_concurrent_positions}"
            + (f" | HALTED: {self._halted}" if self._halted else "")
        )
