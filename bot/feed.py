"""Where bars come from, and how stale they are.

**Staleness is a first-class gate, not a log line.** The single most dangerous
failure in an unattended bot is a feed that quietly stops updating: every
computation still succeeds, every number looks plausible, and the bot trades a
market that no longer exists. So a Snapshot always carries its data age and the
engine refuses to act when that age exceeds a threshold.

Two implementations behind one interface. `ArchiveFeed` replays the recorded
1-minute history for backtests and shadow runs; a live feed would poll the API.
The engine cannot tell them apart, which is the point -- shadow and live differ
only in the executor, never in the logic being tested.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from cisd.bars import Bar

STORE = Path(__file__).resolve().parent.parent / "data" / "uw" / "SPY_ohlc_1m"


def _f(v, d=math.nan) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


@dataclass
class Snapshot:
    """Everything the engine may look at, at one moment."""

    symbol: str
    as_of: int                  # bar time in epoch seconds
    bars: list[Bar]             # chronological, ending at as_of
    data_age_s: float           # seconds between the newest bar and 'now'
    source: str

    @property
    def last(self) -> Bar | None:
        return self.bars[-1] if self.bars else None

    def is_stale(self, max_age_s: float) -> bool:
        return math.isnan(self.data_age_s) or self.data_age_s > max_age_s


class ArchiveFeed:
    """Replays recorded 1-minute bars. Used for shadow runs and backtests.

    Loads whole sessions but only ever exposes bars up to `as_of`, so a caller
    cannot accidentally read forward. That guarantee is what makes a shadow run
    comparable to a live one."""

    def __init__(self, symbol: str = "SPY", store: Path = STORE):
        self.symbol = symbol
        self.store = store
        self._cache: dict[str, list[Bar]] = {}
        self._sessions: list[str] | None = None
        self._hist: dict[tuple[str, int], list[Bar]] = {}

    def sessions(self) -> list[str]:
        # Cached. This is called from `history_before`, which the engine calls
        # once per minute -- globbing a thousand files 390 times a session took
        # the verification suite from seconds to minutes.
        if self._sessions is None:
            self._sessions = sorted(p.stem for p in self.store.glob("*.json"))
        return self._sessions

    def session_bars(self, day: str) -> list[Bar]:
        if day in self._cache:
            return self._cache[day]
        p = self.store / f"{day}.json"
        if not p.exists():
            return []
        rows = json.loads(p.read_text(encoding="utf-8"))
        reg = [r for r in rows if r.get("market_time") == "r"]
        # The feed is newest-first. Sorting here rather than at every call site
        # is the difference between a signal and its mirror image.
        reg.sort(key=lambda r: r["start_time"])
        bars: list[Bar] = []
        for r in reg:
            c = _f(r["close"])
            if math.isnan(c):
                continue
            # Real epoch seconds, left edge, NOT the minute index. Every
            # research signal keys off `bar.session_date_ny` and
            # `minute_of_day_ny`, both derived from `ts` -- give it an index
            # and session detection collapses to one endless day, silently.
            ts = int(datetime.strptime(r["start_time"], "%Y-%m-%dT%H:%M:%SZ")
                     .replace(tzinfo=timezone.utc).timestamp())
            bars.append(Bar(ts=ts, open=_f(r["open"]), high=_f(r["high"]),
                            low=_f(r["low"]), close=c, volume=_f(r.get("volume"), 0.0)))
        self._cache[day] = bars
        return bars

    def history_before(self, day: str, days: int) -> list[Bar]:
        """Full sessions immediately before `day`, oldest first.

        Signals that reference the previous session -- gap fades, sweeps of
        yesterday extremes -- return 0 forever without this, which reads as
        "no setups today" rather than as the bug it is."""
        if days <= 0:
            return []
        key = (day, days)
        if key in self._hist:
            return self._hist[key]
        all_days = self.sessions()
        if day not in all_days:
            return []
        k = all_days.index(day)
        out: list[Bar] = []
        for d in all_days[max(0, k - days):k]:
            out.extend(self.session_bars(d))
        self._hist[key] = out
        return out

    def snapshot(self, day: str, minute_index: int,
                 history_days: int = 1) -> Snapshot:
        """Bars through `minute_index` of `day`, plus `history_days` of prior
        sessions, and nothing after.

        The forward cut is the guarantee that makes a shadow run comparable to
        a live one. The backward reach is what makes the signals mean the same
        thing they meant in the harness."""
        bars = self.session_bars(day)
        today = bars[: minute_index + 1]
        if not today:
            return Snapshot(self.symbol, 0, [], data_age_s=float("nan"),
                            source=f"archive:{day}")
        visible = self.history_before(day, history_days) + today
        return Snapshot(self.symbol, today[-1].ts, visible,
                        data_age_s=0.0, source=f"archive:{day}")


class LiveFeed:
    """Placeholder for the real thing. Deliberately refuses rather than faking.

    A live feed that silently returns nothing is how a bot ends up trading on
    an empty book. When this is implemented it must set `data_age_s` from the
    server's own timestamp, not from local fetch time -- otherwise the UI shows
    '2s ago' over a 15-minute-old price."""

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol

    def snapshot(self, *args, **kwargs) -> Snapshot:
        raise NotImplementedError(
            "LiveFeed is not built. Shadow runs use ArchiveFeed; going live "
            "requires implementing this against a server-stamped timestamp."
        )
