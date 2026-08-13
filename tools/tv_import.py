"""Load a TradingView chart-data export.

TradingView's CSV is not quite stable across versions, so this parser is
deliberately tolerant about the things that vary and strict about the things
that would corrupt a comparison:

* **Time column.** Either ISO 8601 with an offset (`2026-08-12T09:30:00-04:00`)
  or a bare unix timestamp. Both are accepted; a naive ISO string without an
  offset is REJECTED rather than assumed to be UTC or local, because guessing
  wrong shifts every bar by hours and the diff would then fail for a reason
  that has nothing to do with the port.
* **Column names.** Case and surrounding whitespace vary. Matching is
  normalised.
* **Bar alignment.** TradingView labels intraday bars by their left edge, same
  as `cisd.bars`. Verified in `verify_tv_import`, not assumed.

Stdlib only.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from cisd.bars import Bar


@dataclass
class TvExport:
    """Bars plus whatever indicator series the export carried."""

    bars: list[Bar]
    series: dict[str, list[float]] = field(default_factory=dict)

    def signal_rows(self) -> list[tuple[int, int, float, float, float]]:
        """(ts, direction, score, stop, displacement) for every signal bar.

        Empty when the export has no `x_sig_dir` column, which means the export
        patch was not applied -- reported loudly by the caller rather than
        silently producing zero signals to compare against."""
        dirs = self.series.get("x_sig_dir")
        if not dirs:
            return []
        scores = self.series.get("x_sig_score", [])
        stops = self.series.get("x_sig_stop", [])
        disps = self.series.get("x_sig_disp", [])

        out = []
        for i, d in enumerate(dirs):
            if math.isnan(d) or int(d) == 0:
                continue
            out.append((
                self.bars[i].ts,
                int(d),
                scores[i] if i < len(scores) else math.nan,
                stops[i] if i < len(stops) else math.nan,
                disps[i] if i < len(disps) else math.nan,
            ))
        return out

    def has_export_patch(self) -> bool:
        return "x_sig_dir" in self.series


def _parse_time(raw: str) -> int:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty time cell")
    # Bare unix timestamp.
    try:
        v = float(raw)
        return int(v)
    except ValueError:
        pass
    iso = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        raise ValueError(
            f"time {raw!r} has no UTC offset. Re-export with ISO time, or the "
            f"comparison would be silently shifted by the chart's timezone."
        )
    return int(dt.timestamp())


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load(path: str | Path) -> TvExport:
    """Read a TradingView export into bars and named series."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Export from TradingView first "
            f"(see tools/pine_export_patch.pine for the exact steps)."
        )

    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{p} has no header row")
        cols = {_norm(c): c for c in reader.fieldnames}

        required = ("time", "open", "high", "low", "close")
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(
                f"{p} is missing column(s) {missing}. Found: {reader.fieldnames}"
            )

        vol_col = cols.get("volume")
        extra = [c for c in cols if c.startswith("x_")]

        bars: list[Bar] = []
        series: dict[str, list[float]] = {c: [] for c in extra}

        for row in reader:
            ts = _parse_time(row[cols["time"]])
            try:
                o = float(row[cols["open"]])
                h = float(row[cols["high"]])
                l = float(row[cols["low"]])
                c = float(row[cols["close"]])
            except (ValueError, TypeError):
                # TradingView writes blank rows at session gaps. Skipping is
                # right: a synthesised bar is exactly the phantom-data problem
                # this project already hit with Robinhood's history.
                continue
            v = 0.0
            if vol_col:
                try:
                    v = float(row[vol_col])
                except (ValueError, TypeError):
                    v = 0.0
            bars.append(Bar(ts, o, h, l, c, v))
            for c_norm in extra:
                raw = row[cols[c_norm]]
                try:
                    series[c_norm].append(float(raw))
                except (ValueError, TypeError):
                    series[c_norm].append(math.nan)

    if not bars:
        raise ValueError(f"{p} contained no usable bars")

    # Ascending order is assumed everywhere downstream.
    if any(bars[i].ts >= bars[i + 1].ts for i in range(len(bars) - 1)):
        order = sorted(range(len(bars)), key=lambda i: bars[i].ts)
        bars = [bars[i] for i in order]
        series = {k: [v[i] for i in order] for k, v in series.items()}

    return TvExport(bars=bars, series=series)


def infer_timeframe_minutes(bars: list[Bar]) -> int:
    """Modal spacing between consecutive bars, in minutes.

    Uses the mode rather than the mean because session gaps and weekends would
    drag an average far away from the real bar size."""
    if len(bars) < 3:
        raise ValueError("need at least 3 bars to infer a timeframe")
    counts: dict[int, int] = {}
    for a, b in zip(bars, bars[1:]):
        gap = (b.ts - a.ts) // 60
        if gap > 0:
            counts[gap] = counts.get(gap, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]
