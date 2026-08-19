"""Append-only record of every decision, including the ones to do nothing.

Logging only the trades is how you end up unable to answer "why didn't it take
that one?" -- which is the question you will actually have. Every evaluation
gets a line, whether it produced an order, was blocked by a gate, or found no
signal at all.

JSONL, one object per line, opened in append mode and flushed per write. A
crash mid-session loses the current line and nothing else.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Decision:
    """One evaluation at one moment."""

    ts: int                       # bar time the decision was made on
    wall_ts: float                # when the bot actually ran
    symbol: str
    signal: str
    direction: int                # +1 / -1 / 0
    action: str                   # "open" | "close" | "hold" | "blocked" | "none"
    reason: str
    mode: str                     # "shadow" | "live"
    price: float = float("nan")
    shares: int = 0
    stop: float = float("nan")
    target: float = float("nan")
    data_age_s: float = float("nan")
    context: dict = field(default_factory=dict)


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, d: Decision) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(d), sort_keys=True) + "\n")
            fh.flush()

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line from a crash. Skip it; do not discard the
                # rest of the file for one bad record.
                continue
        return out

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.read():
            k = f"{row.get('mode','?')}:{row.get('action','?')}"
            counts[k] = counts.get(k, 0) + 1
        return counts
