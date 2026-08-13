"""Bar containers and timeframe aggregation.

Everything downstream reads bars through here, so this module owns two things
that are easy to get quietly wrong:

1. **Bar timestamps are the LEFT edge**, in epoch seconds, UTC. A 5-minute bar
   stamped 13:30:00Z covers [13:30, 13:35). TradingView, Robinhood and
   FirstRate all label bars this way; if a vendor labels the right edge, shift
   it at the loader, not here.

2. **Aggregation is session-anchored**, because TradingView is. A 60-minute SPY
   bar on a regular-hours chart runs 09:30-10:30, not 10:00-11:00. Anchoring to
   the hour instead would silently shift every higher-timeframe rejection block
   by half an hour and no test would notice.

Stdlib only. `zoneinfo` handles the DST arithmetic that the whole session layer
depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

# Minutes after NY midnight. 09:30 = 570.
RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar. `ts` is the left edge in epoch seconds, UTC."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def dt_utc(self) -> datetime:
        return datetime.fromtimestamp(self.ts, UTC)

    @property
    def dt_ny(self) -> datetime:
        return datetime.fromtimestamp(self.ts, NY)

    @property
    def minute_of_day_ny(self) -> int:
        """Minutes since NY midnight. Session logic keys off this."""
        d = self.dt_ny
        return d.hour * 60 + d.minute

    @property
    def session_date_ny(self) -> str:
        """NY calendar date as YYYY-MM-DD. Not the trading date for overnight
        bars -- see `sessions.trading_date` when that distinction matters."""
        return self.dt_ny.strftime("%Y-%m-%d")

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_top(self) -> float:
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)

    @property
    def body_fraction(self) -> float:
        """Body as a share of full range. 0.0 for a doji, 1.0 for a marubozu.

        A zero-range bar returns 0.0 rather than dividing by zero. That choice
        matters: a zero-range bar then fails every `min_body` test, which is the
        conservative direction (it forms no rejection block)."""
        r = self.range
        return (self.body_top - self.body_bottom) / r if r > 0 else 0.0

    @property
    def is_up(self) -> bool:
        return self.close > self.open

    @property
    def is_down(self) -> bool:
        return self.close < self.open

    def close_position(self) -> float:
        """Where the close sits in the bar's range, 0.0 at the low, 1.0 at the
        high. Zero-range bars return 0.5 -- neither strong-up nor strong-down."""
        r = self.range
        return (self.close - self.low) / r if r > 0 else 0.5


def is_rth(bar: Bar) -> bool:
    """True for bars inside regular trading hours (09:30 <= t < 16:00 NY).

    Holiday and half-day handling is deliberately absent: on a holiday there
    are no bars, and on a half day the extra bars simply do not exist in the
    feed. Nothing here needs a calendar."""
    m = bar.minute_of_day_ny
    return RTH_OPEN_MIN <= m < RTH_CLOSE_MIN


def bucket_index(bar: Bar, minutes: int, anchor_min: int = RTH_OPEN_MIN) -> tuple[str, int]:
    """Return the aggregation key for `bar` at a `minutes`-wide timeframe.

    The key is (NY date, bucket number measured from `anchor_min`). Bars before
    the anchor on a given day get negative bucket numbers, which is correct and
    keeps premarket bars from folding into the 09:30 bar.

    Floor division is deliberate -- Python floors toward negative infinity, so
    a bar at 09:28 with a 5-minute timeframe lands in bucket -1 (09:25-09:30),
    not bucket 0."""
    offset = bar.minute_of_day_ny - anchor_min
    return (bar.session_date_ny, offset // minutes)


def resample(bars: list[Bar], minutes: int, anchor_min: int = RTH_OPEN_MIN) -> list[Bar]:
    """Aggregate 1-minute bars up to a `minutes`-wide timeframe.

    Input must be sorted ascending and should be a single, consistent session
    type (all-RTH or all-ETH); mixing them changes what the anchor means.

    The output bar's `ts` is the left edge of the *bucket*, not of the first bar
    that happened to fall in it. A 5-minute bucket whose first trade arrived at
    09:31 is still stamped 09:30 -- otherwise a thin session would produce bars
    whose timestamps drift away from the grid the higher timeframes assume.
    """
    if minutes <= 0:
        raise ValueError(f"timeframe must be positive minutes, got {minutes}")
    if not bars:
        return []

    out: list[Bar] = []
    cur_key: tuple[str, int] | None = None
    o = h = l = c = 0.0
    v = 0.0
    cur_ts = 0

    for b in bars:
        key = bucket_index(b, minutes, anchor_min)
        if key != cur_key:
            if cur_key is not None:
                out.append(Bar(cur_ts, o, h, l, c, v))
            cur_key = key
            cur_ts = _bucket_start_ts(b, minutes, anchor_min)
            o, h, l, c, v = b.open, b.high, b.low, b.close, b.volume
        else:
            h = max(h, b.high)
            l = min(l, b.low)
            c = b.close
            v += b.volume

    if cur_key is not None:
        out.append(Bar(cur_ts, o, h, l, c, v))
    return out


def _bucket_start_ts(bar: Bar, minutes: int, anchor_min: int) -> int:
    """Epoch seconds of the left edge of the bucket containing `bar`.

    Built by rounding in NY wall-clock minutes and converting back, so a bucket
    that straddles a DST transition lands on the wall-clock grid a chart would
    draw rather than on a fixed number of seconds from midnight."""
    ny = bar.dt_ny
    offset = (ny.hour * 60 + ny.minute) - anchor_min
    start_offset = (offset // minutes) * minutes
    midnight = ny.replace(hour=0, minute=0, second=0, microsecond=0)
    start = midnight + timedelta(minutes=anchor_min + start_offset)
    return int(start.timestamp())


def filter_rth(bars: list[Bar]) -> list[Bar]:
    return [b for b in bars if is_rth(b)]


def slice_window(bars: list[Bar], start_min: int, end_min: int) -> list[Bar]:
    """Bars whose NY minute-of-day falls in [start_min, end_min).

    Used for the traded window (09:30-12:00 by default). Windows that wrap
    midnight -- Asia, say -- are handled by `sessions.in_session`, not here."""
    return [b for b in bars if start_min <= b.minute_of_day_ny < end_min]
