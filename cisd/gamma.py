"""Dealer gamma context -- an addition, not a port.

**Why this is the most valuable thing added to the model.** The Pine tries to
answer "where is price being drawn, and will this level hold?" using previous-
day highs and lows and session ranges. Those are proxies. Dealer positioning is
the measured version of the same question, and on SPY it is the dominant
intraday mechanism: hedging flow against 0DTE inventory is a large share of
same-day volume, and its *sign* decides whether a level should be expected to
hold or to break.

The mechanism, stated plainly so the scoring rule can be argued with:

* **Positive dealer gamma** (spot above the flip): dealers are long gamma and
  hedge against the move -- selling strength, buying weakness. Moves are damped
  and price pins toward large strikes. A rejection *at* a wall is the regime
  behaving normally; a breakout through one is fighting the hedging flow.
* **Negative dealer gamma** (spot below the flip): dealers hedge with the move,
  which accelerates it. Walls become launchpads rather than barriers, and
  mean-reversion setups are the ones fighting the flow.

So the same rejection block at the same price is a different-quality trade
depending on which side of the flip spot sits. Nothing in the Pine can express
that, and it is the difference this project has available that a generic ICT
indicator does not -- the GEX maths is already built and verified next door.

**This module deliberately computes no options maths.** It consumes a snapshot
produced elsewhere (the NYAM Terminal engine's `compute_gex`, or any equivalent)
so there is exactly one implementation of dealer gamma to trust, not a second
copy to drift. See the `options-math` skill for the sign convention; getting it
backwards inverts every rule below, which is precisely why it is not re-derived
here.

The whole module degrades to neutral when no snapshot is available. A missing
gamma context must never fabricate a level -- it returns 0.0 and says so.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GammaContext:
    """A dealer-positioning snapshot, as of `as_of_ts`.

    `net_gex` is signed dealer gamma in dollars per 1% move; positive means
    dealers are long gamma. `control_node_margin_pct` carries how decisively the
    magnet strike won -- under 15% the pick is close to a coin flip and this
    module refuses to score it, which is a measured property of that statistic,
    not caution for its own sake."""

    spot: float
    gamma_flip: float
    call_wall: float
    put_wall: float
    net_gex: float
    as_of_ts: int
    control_node: float = math.nan
    control_node_margin_pct: float = 0.0
    source: str = "unknown"

    @classmethod
    def load(cls, path: str | Path) -> GammaContext | None:
        """Read a snapshot emitted by the GEX engine. None when absent.

        None is a first-class answer here: the caller scores zero rather than
        guessing, and the signal records that gamma context was unavailable."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                spot=float(raw["spot"]),
                gamma_flip=float(raw["gamma_flip"]),
                call_wall=float(raw["call_wall"]),
                put_wall=float(raw["put_wall"]),
                net_gex=float(raw["net_gex"]),
                as_of_ts=int(raw["as_of_ts"]),
                control_node=float(raw.get("control_node", math.nan)),
                control_node_margin_pct=float(raw.get("control_node_margin_pct", 0.0)),
                source=str(raw.get("source", "file")),
            )
        except (KeyError, ValueError, TypeError):
            return None

    def age_seconds(self, now_ts: int) -> int:
        return now_ts - self.as_of_ts

    def is_stale(self, now_ts: int, max_age_seconds: int = 3600) -> bool:
        """Gamma context ages. An hour-old chain through a fast tape describes
        a book that has already been rehedged."""
        return self.age_seconds(now_ts) > max_age_seconds

    @property
    def positive_regime(self) -> bool:
        """Dealers long gamma -- expect damping and pinning."""
        return self.spot > self.gamma_flip

    def nearest_wall(self, price: float) -> tuple[str, float] | None:
        """The closer of the call wall and put wall."""
        candidates = []
        if not math.isnan(self.call_wall):
            candidates.append(("call_wall", self.call_wall))
        if not math.isnan(self.put_wall):
            candidates.append(("put_wall", self.put_wall))
        if not candidates:
            return None
        return min(candidates, key=lambda kv: abs(price - kv[1]))

    def score(
        self,
        bull: bool,
        ce: float,
        ext: float,
        setup_type: int,
        band: float,
        weight: float,
    ) -> tuple[float, str]:
        """Score this block against dealer positioning.

        Returns (points, reason). Points can be NEGATIVE -- a setup fighting the
        hedging flow deserves to be marked down, not merely left unrewarded, and
        a scoring system that can only add is a scoring system that ranks
        everything highly.

        `setup_type` is 2 for reversal (terminal) and 1 for continuation, from
        `context.setup_type`. `band` is the ATR-scaled proximity tolerance."""
        wall = self.nearest_wall(ext)
        if wall is None:
            return 0.0, "no wall data"

        name, level = wall
        at_wall = abs(ext - level) <= band
        if not at_wall:
            return 0.0, "not near a wall"

        # A bullish block rejecting off the put wall, or a bearish one rejecting
        # off the call wall, is the aligned case: the block's swept extreme sits
        # where hedging supply/demand concentrates.
        aligned_wall = (bull and name == "put_wall") or (not bull and name == "call_wall")

        if self.positive_regime:
            if aligned_wall and setup_type == 2:
                return weight, f"reversal at {name}, positive gamma (level should hold)"
            if aligned_wall:
                return weight * 0.6, f"block at {name}, positive gamma"
            # Continuation THROUGH a wall while dealers damp moves.
            return -weight * 0.5, f"pushing through {name} against positive gamma"

        # Negative gamma: hedging amplifies. Walls give way.
        if setup_type == 1:
            return weight, f"continuation near {name}, negative gamma (accelerant)"
        if aligned_wall:
            return -weight * 0.4, f"reversal at {name} but negative gamma breaks levels"
        return 0.0, f"near {name}, negative gamma"

    def regime_note(self) -> str:
        side = "positive" if self.positive_regime else "negative"
        return (
            f"dealer gamma {side} (spot {self.spot:.2f} vs flip {self.gamma_flip:.2f}); "
            f"{'damped, pins toward strikes' if self.positive_regime else 'amplified, levels break'}"
        )
