"""Scoring harness for intraday signals.

A signal is any callable `(bars, i) -> +1 | -1 | 0`. That is the whole
interface. It must read only bars at or before `i`; the harness cannot enforce
that, so `verify_research.py` checks it by truncation instead.

**Exits are ATR-scaled, not fixed in dollars.** SPY ran from $70 to $770 over
the period we care about; a $0.30 stop is a different trade at each end. Every
distance here is a multiple of ATR at entry, which makes results comparable
across two decades and across instruments.

**The same three conservatisms as before, restated because they matter:**

* A bar whose range spans both the stop and the target counts as a **stop**.
  Intrabar order is unknowable from OHLC.
* Entry is the close of the signal bar. A real fill is a second later and
  slightly worse; `cost_atr` charges for that.
* Outcomes are on the underlying. Options add a vol premium, a spread and
  theta on top -- this is the necessary condition, never the sufficient one.

**Why the null matters.** An asymmetric stop/target structure produces a
non-zero expectancy on random entries. A 0.5 ATR stop against a 2.0 ATR target
loses far more often than it wins regardless of any signal. Scoring a candidate
against zero instead of against that structure's own base rate is how a neutral
signal gets mistaken for an edge -- which is precisely the check missing from
the CISD work.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable

from cisd.bars import Bar
from cisd.indicators import atr
from cisd.sessions import in_window, parse_session

Signal = Callable[[list[Bar], int], int]


@dataclass
class Trade:
    index: int
    ts: int
    direction: int
    entry: float
    stop: float
    target: float
    exit_price: float
    outcome: str          # "target" | "stop" | "time"
    r_multiple: float
    bars_held: int


@dataclass
class Result:
    name: str
    trades: list[Trade] = field(default_factory=list)
    sessions: int = 0

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def rs(self) -> list[float]:
        return [t.r_multiple for t in self.trades]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return math.nan
        return sum(1 for t in self.trades if t.r_multiple > 0) / self.n

    @property
    def mean_r(self) -> float:
        return statistics.mean(self.rs) if self.trades else math.nan

    @property
    def se(self) -> float:
        if self.n < 2:
            return math.nan
        return statistics.stdev(self.rs) / math.sqrt(self.n)

    @property
    def t_stat(self) -> float:
        se = self.se
        if math.isnan(se) or se == 0:
            return math.nan
        return self.mean_r / se

    def outcome_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for t in self.trades:
            c[t.outcome] = c.get(t.outcome, 0) + 1
        return c

    def versus(self, null: "Result") -> float:
        """t-statistic of this result's mean R against a null's mean R.

        Welch's form, because the two samples have different sizes and
        different variances. This is the number that decides whether a
        candidate beat the structure it was traded with, rather than beating
        zero."""
        if self.n < 2 or null.n < 2:
            return math.nan
        va, vb = statistics.variance(self.rs), statistics.variance(null.rs)
        denom = math.sqrt(va / self.n + vb / null.n)
        return (self.mean_r - null.mean_r) / denom if denom else math.nan


def evaluate_entry(
    bars: list[Bar],
    i: int,
    direction: int,
    atr_value: float,
    stop_atr: float,
    target_atr: float,
    max_hold_bars: int,
    cost_atr: float = 0.02,
) -> Trade | None:
    """Resolve one entry to first touch of stop or target, same session."""
    if direction == 0 or math.isnan(atr_value) or atr_value <= 0:
        return None
    entry = bars[i].close
    risk = stop_atr * atr_value
    reward = target_atr * atr_value
    if risk <= 0:
        return None

    long = direction > 0
    stop = entry - risk if long else entry + risk
    target = entry + reward if long else entry - reward
    day = bars[i].session_date_ny

    j = i + 1
    while j < len(bars) and bars[j].session_date_ny == day and (j - i) <= max_hold_bars:
        b = bars[j]
        hit_stop = b.low <= stop if long else b.high >= stop
        hit_target = b.high >= target if long else b.low <= target
        if hit_stop:
            r = -1.0 - cost_atr * atr_value / risk
            return Trade(i, bars[i].ts, direction, entry, stop, target, stop, "stop", r, j - i)
        if hit_target:
            r = target_atr / stop_atr - cost_atr * atr_value / risk
            return Trade(i, bars[i].ts, direction, entry, stop, target, target, "target", r, j - i)
        j += 1

    if j - 1 <= i:
        return None
    close = bars[j - 1].close
    move = (close - entry) if long else (entry - close)
    r = move / risk - cost_atr * atr_value / risk
    return Trade(i, bars[i].ts, direction, entry, stop, target, close, "time", r, j - 1 - i)


def measure(
    bars: list[Bar],
    signal: Signal,
    name: str = "signal",
    stop_atr: float = 1.0,
    target_atr: float = 2.0,
    max_hold_bars: int = 60,
    window: str = "0930-1200",
    atr_length: int = 14,
    cost_atr: float = 0.02,
    one_per_session: bool = True,
) -> Result:
    """Score a signal over `bars`.

    `one_per_session` is on by default. Overlapping entries from the same
    impulse are not independent observations, and counting them as such
    inflates n, shrinks the standard error, and manufactures significance --
    the same overlap problem the VRP study handles with sqrt(n_indep)."""
    a = atr(bars, atr_length)
    start, end = parse_session(window)
    res = Result(name)
    seen_sessions: set[str] = set()
    used: set[str] = set()

    for i, bar in enumerate(bars):
        seen_sessions.add(bar.session_date_ny)
        if not in_window(bar.minute_of_day_ny, start, end):
            continue
        if one_per_session and bar.session_date_ny in used:
            continue
        d = signal(bars, i)
        if d == 0:
            continue
        t = evaluate_entry(bars, i, d, a[i], stop_atr, target_atr, max_hold_bars, cost_atr)
        if t is not None:
            res.trades.append(t)
            used.add(bar.session_date_ny)

    res.sessions = len(seen_sessions)
    return res


def format_row(r: Result, null: Result | None = None) -> str:
    if r.n == 0:
        return f"  {r.name:<30} {'--':>6}"
    vs = f"{r.versus(null):+6.2f}" if null is not None and null is not r else "     -"
    return (f"  {r.name:<30} {r.n:>5} {100 * r.win_rate:>5.0f}% "
            f"{r.mean_r:>+7.3f} {r.se:>6.3f} {r.t_stat:>+7.2f} {vs}")


HEADER = (f"  {'signal':<30} {'n':>5} {'win':>6} {'meanR':>7} {'se':>6} "
          f"{'t vs 0':>7} {'t vs null':>6}")
