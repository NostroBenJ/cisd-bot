"""Session windows, key opens, and event blackouts.

Pine expresses sessions as `"0930-1600"` strings interpreted in an exchange
timezone. This module reproduces that, including the wrap-past-midnight case
(`"1800-0000"`, `"2000-0000"`) which is the one every naive `start <= t < end`
implementation gets wrong -- it silently matches nothing, and the Asia session
features quietly evaluate to False forever without raising anything.

The event blackout is an addition. Reasoning is on `EventCalendar`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .bars import NY, Bar

# ── session strings ─────────────────────────────────────────────────────────


def parse_session(spec: str) -> tuple[int, int]:
    """`"0930-1600"` -> (570, 960), both in minutes since NY midnight.

    An end of `"0000"` becomes 1440 rather than 0, so `"1800-0000"` reads as
    18:00 to end-of-day instead of an empty window."""
    try:
        start_s, end_s = spec.split("-")
        start = int(start_s[:2]) * 60 + int(start_s[2:])
        end = int(end_s[:2]) * 60 + int(end_s[2:])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"bad session spec {spec!r}, expected 'HHMM-HHMM'") from exc
    if end == 0:
        end = 24 * 60
    return start, end


def in_window(minute_of_day: int, start: int, end: int) -> bool:
    """Half-open [start, end). Wraps past midnight when start > end."""
    if start <= end:
        return start <= minute_of_day < end
    return minute_of_day >= start or minute_of_day < end


def in_session(bar: Bar, spec: str) -> bool:
    start, end = parse_session(spec)
    return in_window(bar.minute_of_day_ny, start, end)


def trading_date(bar: Bar, day_boundary_min: int = 18 * 60) -> str:
    """The trading date a bar belongs to, rolling at 18:00 NY.

    A bar at 20:00 Monday belongs to Tuesday's session. Regular-hours-only
    backtests never hit this, but the Asia/London features and the overnight
    gap logic depend on it, and using the calendar date there would split every
    overnight range across two days."""
    d = bar.dt_ny
    if d.hour * 60 + d.minute >= day_boundary_min:
        d = d + timedelta(days=1)
    return d.strftime("%Y-%m-%d")


# ── key opens ───────────────────────────────────────────────────────────────

# Minute-of-day for each key open the Pine model tracks. The values are the
# opening print of the first bar at or after that minute, held for the session.
KEY_OPENS: dict[str, int] = {
    "midnight": 0,
    "premarket": 8 * 60 + 30,
    "ny_open": 9 * 60 + 30,
    "macro": 10 * 60,
    "pm": 13 * 60 + 30,
    "futures": 18 * 60,
}

DEFAULT_KEY_OPENS = ("ny_open", "macro", "pm")


@dataclass
class KeyOpenTracker:
    """Captures the opening price at each enabled key-open minute.

    Resets on the NY calendar date, matching the Pine, which recomputes these
    per day. `enabled` names must be keys of `KEY_OPENS`."""

    enabled: tuple[str, ...] = DEFAULT_KEY_OPENS
    levels: dict[str, float] = field(default_factory=dict)
    _date: str | None = None

    def __post_init__(self) -> None:
        unknown = set(self.enabled) - set(KEY_OPENS)
        if unknown:
            raise ValueError(f"unknown key opens: {sorted(unknown)}")

    def update(self, bar: Bar) -> None:
        d = bar.session_date_ny
        if d != self._date:
            self._date = d
            self.levels = {}
        m = bar.minute_of_day_ny
        for name in self.enabled:
            if name in self.levels:
                continue
            # First bar at or past the key minute takes the open. On a 5-minute
            # chart the 10:00 macro open is the 10:00 bar's open; if the feed is
            # missing that bar, the next one stands in rather than the level
            # going missing for the day.
            if m >= KEY_OPENS[name]:
                self.levels[name] = bar.open

    def nearest_distance(self, price: float) -> float | None:
        """Absolute distance to the closest captured key open, or None."""
        if not self.levels:
            return None
        return min(abs(price - lvl) for lvl in self.levels.values())

    def near(self, price: float, tolerance: float) -> bool:
        d = self.nearest_distance(price)
        return d is not None and d <= tolerance


# ── event blackout (addition, not a port) ───────────────────────────────────


@dataclass
class EventCalendar:
    """Scheduled macro events to stand aside for.

    Why this is here: SPY's intraday structure around an 08:30 CPI print or a
    14:00 FOMC statement is not the structure the rest of the model was built
    on. A sweep into a release is not engineered liquidity being taken -- it is
    a repricing, and the rejection-block logic reads it as a setup precisely
    when it is most likely to run over the stop. The Pine has no notion of this
    and will happily grade an A+ signal thirty seconds before a Fed statement.

    Events are supplied as data rather than derived. Non-farm payrolls is
    mechanically the first Friday, but CPI drifts and FOMC is a published
    schedule, so a rules-based guess would be wrong often enough to be worse
    than nothing. `load` reads a JSON list of
    `{"ts": <epoch seconds UTC>, "name": "CPI"}` objects.

    An empty calendar blocks nothing, which is the honest default -- it means
    "we have no event data", and the backtest reports it as such rather than
    pretending events do not exist."""

    events: list[tuple[int, str]] = field(default_factory=list)
    before_min: int = 15
    after_min: int = 15

    @classmethod
    def load(cls, path: str | Path, **kw) -> EventCalendar:
        p = Path(path)
        if not p.exists():
            return cls(**kw)
        raw = json.loads(p.read_text(encoding="utf-8"))
        events = [(int(e["ts"]), str(e.get("name", "event"))) for e in raw]
        events.sort()
        return cls(events=events, **kw)

    @classmethod
    def from_local(cls, entries: list[tuple[str, str]], **kw) -> EventCalendar:
        """Build from `[("2026-08-13 08:30", "CPI"), ...]` in NY local time."""
        events = []
        for stamp, name in entries:
            dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=NY)
            events.append((int(dt.timestamp()), name))
        events.sort()
        return cls(events=events, **kw)

    def blocked(self, ts: int) -> str | None:
        """Name of the event blacking out `ts`, or None if clear."""
        for ev_ts, name in self.events:
            if ev_ts - self.before_min * 60 <= ts <= ev_ts + self.after_min * 60:
                return name
        return None

    def is_empty(self) -> bool:
        return not self.events
