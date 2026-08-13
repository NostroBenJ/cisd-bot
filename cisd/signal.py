"""The signal contract -- entry, stop, targets, and the reasoning behind them.

The Pine emits a human-readable string: direction, grade and price, with no
stop and no target. Nothing downstream can trade that. This module defines what
the bot actually consumes.

**On the name "sigma".** The Pine calls its projection levels 2.0σ / 2.5σ / 4.0σ,
but they are not standard deviations of anything -- they are multiples of the
displacement leg (the distance from the block's swept extreme to the furthest
point of the move out of it). A reader seeing "σ" reasonably assumes a
distribution and a probability, and there is none. The field is named
`leg_multiple` here and the σ label is kept only for continuity with the chart.
Mislabelling a deterministic projection as a statistical one is exactly the kind
of plausible-wrong-number this codebase is meant to refuse.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Target:
    """One profit target."""

    leg_multiple: float
    price: float
    r_multiple: float

    @property
    def label(self) -> str:
        return f"{self.leg_multiple}σ"


@dataclass
class Signal:
    """A tradeable signal. Everything the bot and the journal need.

    `stop` is the block's swept extreme plus the invalidation buffer -- the
    price at which the premise is wrong, not a percentage someone chose."""

    ts: int
    symbol: str
    direction: str            # "long" | "short"
    entry: float
    stop: float
    targets: list[Target]
    grade: str
    score: float
    stack: int
    timeframe: int
    setup_type: str           # "reversal" | "continuation" | "none"
    tf_agreement: int
    bias: str                 # "bull" | "bear" | "neutral"
    reasons: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    gamma_regime: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    def r_multiple_at(self, price: float) -> float:
        """How many R a given exit price represents. Negative below the stop."""
        risk = self.risk_per_share
        if risk <= 0:
            return 0.0
        move = (price - self.entry) if self.is_long else (self.entry - price)
        return move / risk

    def to_json(self) -> str:
        d = asdict(self)
        d["risk_per_share"] = round(self.risk_per_share, 4)
        return json.dumps(d, indent=2, sort_keys=True)

    def summary(self) -> str:
        """One line for a log or an alert."""
        t = self.targets[0].price if self.targets else math.nan
        return (
            f"{self.direction.upper()} {self.symbol} {self.grade} ({self.score:.0f}) "
            f"{self.timeframe}m {self.setup_type} | entry {self.entry:.2f} "
            f"stop {self.stop:.2f} t1 {t:.2f} | stack {self.stack} "
            f"| {len(self.reasons)} factors"
        )

    def explain(self) -> str:
        """Full reasoning, ordered by contribution. This is what makes it into
        the journal, and it is why the grader records components rather than
        just a total."""
        lines = [self.summary(), ""]
        ranked = sorted(self.reasons.items(), key=lambda kv: -abs(kv[1]))
        for name, pts in ranked:
            sign = "+" if pts >= 0 else ""
            lines.append(f"  {sign}{pts:>6.1f}  {name}")
        if self.notes:
            lines.append("")
            lines.extend(f"  - {n}" for n in self.notes)
        if self.warnings:
            lines.append("")
            lines.extend(f"  ! {w}" for w in self.warnings)
        return "\n".join(lines)


def build_targets(
    entry: float,
    extreme: float,
    displacement_extreme: float,
    stop: float,
    multiples: tuple[float, ...],
    bull: bool,
) -> list[Target]:
    """Project targets from the displacement leg.

    Port of the Pine's projection maths:

        leg = |displacement_extreme - extreme|
        target = extreme + direction * multiple * leg

    Note that targets are measured from the block's EXTREME, not from the entry.
    That is the Pine's convention and it is kept, but it has a consequence worth
    stating: because entry sits some distance above the extreme already, the
    achievable R is lower than the multiple suggests. A 2.0 leg multiple is not
    a 2R trade. `r_multiple` on each target reports the truth.

    Returns an empty list when the leg is degenerate -- a zero or inverted leg
    means displacement never happened, and projecting from it would invent
    levels out of noise."""
    leg = (displacement_extreme - extreme) if bull else (extreme - displacement_extreme)
    if leg <= 0 or math.isnan(leg):
        return []

    risk = abs(entry - stop)
    direction = 1.0 if bull else -1.0
    out: list[Target] = []
    for m in multiples:
        price = extreme + direction * m * leg
        move = (price - entry) if bull else (entry - price)
        r = move / risk if risk > 0 else 0.0
        # A target behind the entry is not a target. This happens when the
        # entry ran well past the projection before the trigger fired, and
        # silently keeping it would book an instant "win" in the backtest.
        if move > 0:
            out.append(Target(leg_multiple=m, price=price, r_multiple=r))
    return out
