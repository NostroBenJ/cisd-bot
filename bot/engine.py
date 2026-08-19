"""The loop: snapshot -> exits -> signal -> gates -> size -> execute -> journal.

Signals are the SAME callables the research harness validates --
`(bars, i) -> +1 | -1 | 0`. There is no bot-specific reimplementation, because
a reimplementation is a place for the tested thing and the traded thing to
quietly diverge.

The order of gates is deliberate: cheapest and most fatal first.

    1. staleness      -- never act on a frozen feed
    2. open position  -- exits are evaluated before entries, always
    3. session window -- outside the traded hours, do nothing
    4. signal         -- is there anything to do
    5. validation     -- has this signal earned the right to trade money
    6. risk           -- loss budget, trade cap, sizing, kill file
    7. execute

Every refusal is journalled with its reason, so "why was that one skipped" is
always answerable after the fact. Logging only the fills is how a bot becomes
impossible to debug.

The instrument here is SHARES of the underlying, not option contracts. That is
not a simplification -- it is the only honest choice, because every statistic
in FINDINGS.md was computed on underlying moves. A shadow run on options would
be measuring something the research never tested.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from bot.execution import ShadowExecutor
from bot.feed import ArchiveFeed, Snapshot
from bot.journal import Decision, Journal
from bot.validation import Registry
from cisd.indicators import atr
from cisd.risk import RiskGate, RiskLimits
from cisd.sessions import in_window, parse_session
from cisd.signal import Signal as Sig, Target

OPEN_MIN = 9 * 60 + 30       # 09:30 NY, as minutes from midnight


@dataclass
class BotConfig:
    symbol: str = "SPY"
    signal_name: str = "unnamed"
    # Defaults are the HARNESS defaults, verbatim. Any of these that differs
    # from `research.harness.measure` makes the shadow run a measurement of
    # something that was never validated.
    stop_atr: float = 1.0
    target_atr: float = 2.0
    atr_length: int = 14
    max_hold_bars: int = 60      # harness time exit
    cost_atr: float = 0.02       # round-trip cost, in ATRs, charged on exit
    one_per_session: bool = True # overlapping entries are not independent
    # The SAME string the research harness takes, because it is the same
    # window. Expressing it as a minute offset here and a HHMM string there is
    # how the tested thing and the traded thing quietly stop being the same
    # thing -- the first draft of this file did exactly that, and started
    # entries 30 minutes after the harness did.
    window: str = "0930-1200"
    flat_by_min: int = 385       # 15:55 -- nothing is carried overnight
    max_data_age_s: float = 90.0
    equity: float = 1000.0

    @property
    def flat_hhmm(self) -> str:
        m = OPEN_MIN + self.flat_by_min
        return f"{m // 60:02d}{m % 60:02d}"


class Bot:
    """Shadow by default.

    Live needs BOTH a signal with a passing ValidationRecord and an explicitly
    acknowledged LiveExecutor. Neither is granted implicitly, and no argument
    to this constructor can produce both by accident.
    """

    def __init__(self, cfg: BotConfig, signal, registry: Registry,
                 journal: Journal, executor=None, limits: RiskLimits | None = None):
        self.cfg = cfg
        self.signal = signal
        self.registry = registry
        self.journal = journal
        self.exec = executor or ShadowExecutor()

        # The risk gate carries its own hard flat time. If it disagrees with
        # the config, entries silently stop being possible partway through the
        # window -- so make the two agree, loudly.
        if limits is None:
            limits = RiskLimits(hard_flat_time=cfg.flat_hhmm)
        elif limits.hard_flat_time != cfg.flat_hhmm:
            raise ValueError(
                f"hard_flat_time {limits.hard_flat_time} disagrees with "
                f"flat_by_min {cfg.flat_by_min} ({cfg.flat_hhmm}). One of them "
                f"would silently win."
            )
        self.risk = RiskGate(limits)
        self.win_start, self.win_end = parse_session(cfg.window)
        self.allowed, self.gate_reason = registry.may_trade(
            cfg.signal_name, window=cfg.window,
            stop_atr=cfg.stop_atr, target_atr=cfg.target_atr)

    # -- journalling ---------------------------------------------------------

    def _log(self, snap: Snapshot, action: str, reason: str, direction: int = 0,
             price: float = math.nan, shares: int = 0, stop: float = math.nan,
             target: float = math.nan, **ctx) -> None:
        self.journal.write(Decision(
            ts=snap.as_of, wall_ts=time.time(), symbol=self.cfg.symbol,
            signal=self.cfg.signal_name, direction=direction, action=action,
            reason=reason, mode=self.exec.mode, price=price, shares=shares,
            stop=stop, target=target, data_age_s=snap.data_age_s, context=ctx))

    # -- the loop ------------------------------------------------------------

    def run_session(self, feed: ArchiveFeed, day: str) -> float:
        """Replay one session minute by minute. Returns realised P&L."""
        bars = feed.session_bars(day)
        if len(bars) < 100:
            return 0.0

        # ATR on the CONTINUOUS series, exactly as the harness does it. Per
        # session it is stone cold for the first 14 minutes, which silently
        # disqualified every early-session signal in the first draft.
        hist = feed.history_before(day, self.cfg.atr_length)
        a = atr(hist + bars, self.cfg.atr_length)
        off = len(hist)

        self.risk.start_session(self.cfg.equity, day)
        day_pnl = 0.0
        traded_today = False

        for minute, bar in enumerate(bars):
            snap = feed.snapshot(day, minute)
            now_ny = OPEN_MIN + minute

            # 1 -- staleness. First, because everything after it is nonsense
            # if the feed has frozen.
            if snap.is_stale(self.cfg.max_data_age_s):
                self._log(snap, "blocked", f"stale feed ({snap.data_age_s:.0f}s)")
                continue

            # 2 -- exits, before entries, always.
            pos = self.exec.position
            if pos is not None and pos.is_open:
                day_pnl += self._manage(snap, pos, bar, minute, now_ny)
                continue

            # 3 -- the traded window
            if not in_window(bar.minute_of_day_ny, self.win_start, self.win_end):
                continue
            if self.cfg.one_per_session and traded_today:
                continue

            # 4 -- signal. The harness callable, unmodified.
            d = self.signal(snap.bars, len(snap.bars) - 1)
            if d == 0:
                continue

            av = a[off + minute] if off + minute < len(a) else math.nan
            if math.isnan(av) or av <= 0:
                self._log(snap, "blocked", "ATR not warmed up", direction=d)
                continue

            entry = bar.close
            stop = entry - self.cfg.stop_atr * av * d
            target = entry + self.cfg.target_atr * av * d

            # 5 -- validation. A signal without evidence never trades, whatever
            # executor happens to be attached. This is the point of the whole
            # package: as of today it blocks everything, and that is correct.
            if not self.allowed:
                self._log(snap, "blocked", f"not validated: {self.gate_reason}",
                          direction=d, price=entry, stop=stop, target=target)
                continue

            # 6 -- risk and sizing
            sig = Sig(ts=minute, symbol=self.cfg.symbol,
                      direction="long" if d > 0 else "short", entry=entry,
                      stop=stop, targets=[Target(self.cfg.target_atr, target, 1.0)],
                      grade="-", score=0.0, stack=0, timeframe=1,
                      setup_type="none", tf_agreement=1, bias="neutral")
            ok = self.risk.can_open_shares(sig, price=entry, minute_of_day_ny=now_ny)
            if not ok:
                self._log(snap, "blocked", f"risk: {ok.reason}", direction=d,
                          price=entry, stop=stop, target=target)
                continue

            # 7 -- execute
            self.exec.open(minute, self.cfg.symbol, d, ok.contracts, entry, stop, target)
            self.risk.record_open(ok.contracts, entry * ok.contracts)
            traded_today = True
            self._cost_per_share = self.cfg.cost_atr * av
            self._log(snap, "open", "signal fired", direction=d, price=entry,
                      shares=ok.contracts, stop=stop, target=target,
                      atr=round(av, 4))

        return day_pnl

    def _manage(self, snap: Snapshot, pos, bar, minute: int, now_ny: int) -> float:
        """Exit logic for an open position. Returns realised P&L, or 0.0."""
        long = pos.direction > 0
        hit_stop = bar.low <= pos.stop if long else bar.high >= pos.stop
        hit_target = bar.high >= pos.target if long else bar.low <= pos.target

        # A bar that spans both counts as the STOP. Intrabar order is
        # unknowable from OHLC, and the optimistic reading invents edge that
        # will not be there in the fills.
        if hit_stop:
            return self._close(snap, pos.stop, minute, "stop")
        if hit_target:
            return self._close(snap, pos.target, minute, "target")

        if minute - pos.opened_at >= self.cfg.max_hold_bars:
            return self._close(snap, bar.close, minute, "time exit")

        flat = self.risk.must_flatten(now_ny)
        if flat:
            return self._close(snap, bar.close, minute, f"flatten: {flat}")

        return 0.0

    def _close(self, snap: Snapshot, price: float, minute: int, why: str) -> float:
        shares = self.exec.position.shares if self.exec.position else 0
        pnl = self.exec.close(minute, price)
        # Round-trip cost, charged once on exit. A shadow run without it is a
        # backtest of a frictionless market, which is the single most common
        # way paper results fail to survive contact with fills.
        cost = getattr(self, "_cost_per_share", 0.0) * shares
        pnl -= cost
        self.risk.record_close(pnl)
        # Four decimals, not two. At a 0.3 ATR a round trip costs 0.006 per
        # share, which rounds to 0.00 and reads as a frictionless fill.
        self._log(snap, "close", why, price=price, shares=shares,
                  pnl=round(pnl, 4), cost=round(cost, 4),
                  cost_per_share=round(getattr(self, "_cost_per_share", 0.0), 4))
        return pnl
