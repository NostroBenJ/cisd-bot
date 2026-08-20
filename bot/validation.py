"""The gate: what a signal must prove before it is allowed to trade money.

Every negative result in FINDINGS.md is here as a required field. A signal
without a record runs in shadow forever; a signal whose record fails any test
runs in shadow forever. There is no override flag, deliberately -- an override
is the thing you reach for at 3pm on a red day.

The thresholds are not arbitrary:

* `t_vs_null >= 2.5` -- not 2.0. We ran ~36 tests in one batch; at p<0.05 two
  would clear by luck. 2.5 is the crude multiple-comparison tax.
* `holdout_t > 0` -- the sign must survive on data not used to build it. Not
  significance, just the sign, because holdouts are small.
* `survives_concentration` -- mean R stays positive after dropping the best 1%
  of trades. `first_bar_continuation` passed everything else and INVERTED
  here: 65 trades out of 6,521 carried its entire edge.
* `n >= 200` -- below that the standard error is too wide to distinguish a
  0.1R edge from zero, which is the range real edges live in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MIN_TRADES = 200
MIN_T_VS_NULL = 2.5


@dataclass(frozen=True)
class ValidationRecord:
    """Evidence that a signal beat a null on out-of-sample data."""

    signal: str
    n_trades: int
    mean_r: float
    t_vs_null: float
    holdout_t: float
    survives_concentration: bool
    tested_on: str            # data range, e.g. "2022-07..2026-08"
    tested_at: str            # ISO date the test was run

    # The conditions the evidence was gathered under. A record earned on
    # 0930-1200 at a 2.0/2.0 ATR stop and target says NOTHING about the same
    # signal traded 1000-1200 at 1.0/3.0 -- different window, different exit
    # geometry, different distribution. These are matched against the live
    # config, and a mismatch refuses the trade.
    window: str = "0930-1200"
    stop_atr: float = 2.0
    target_atr: float = 2.0

    # "signal"    -- per-trade, scored in R multiples against a random-entry null
    # "portfolio" -- periodic rebalance, scored per period against a random-basket
    #                null. The thresholds mean different things, so the kind is
    #                recorded rather than inferred.
    kind: str = "signal"

    notes: str = ""

    def failures(self) -> list[str]:
        """Every reason this record does not qualify. Empty means it passes."""
        out: list[str] = []
        if self.n_trades < MIN_TRADES:
            out.append(f"only {self.n_trades} trades (need {MIN_TRADES})")
        if self.mean_r <= 0:
            out.append(f"mean R is {self.mean_r:+.4f}, not positive")
        if self.t_vs_null < MIN_T_VS_NULL:
            out.append(f"t vs null {self.t_vs_null:+.2f} (need >= {MIN_T_VS_NULL})")
        if self.holdout_t <= 0:
            out.append(f"holdout t {self.holdout_t:+.2f} does not keep its sign")
        if not self.survives_concentration:
            out.append("edge disappears when the best 1% of trades are removed")
        return out

    @property
    def passes(self) -> bool:
        return not self.failures()


@dataclass
class Registry:
    """Which signals exist, and which of them have earned the right to trade."""

    records: dict[str, ValidationRecord] = field(default_factory=dict)

    def add(self, rec: ValidationRecord) -> None:
        self.records[rec.signal] = rec

    def may_trade(self, signal: str, window: str | None = None,
                  stop_atr: float | None = None,
                  target_atr: float | None = None) -> tuple[bool, str]:
        """(allowed, reason). Unknown signals are refused, not assumed fine.

        Pass the live config so the evidence is checked against the conditions
        it will actually be traded under. Omitting them checks the statistics
        only, which is weaker -- callers that can supply them should."""
        rec = self.records.get(signal)
        if rec is None:
            return False, "no validation record -- shadow only"

        fails = rec.failures()

        if window is not None and window != rec.window:
            fails.append(f"validated on window {rec.window}, configured {window}")
        if stop_atr is not None and abs(stop_atr - rec.stop_atr) > 1e-9:
            fails.append(f"validated at {rec.stop_atr} ATR stop, configured {stop_atr}")
        if target_atr is not None and abs(target_atr - rec.target_atr) > 1e-9:
            fails.append(f"validated at {rec.target_atr} ATR target, configured {target_atr}")

        if fails:
            return False, "; ".join(fails)
        return True, (f"validated: n={rec.n_trades}, mean {rec.mean_r:+.3f}R, "
                      f"t vs null {rec.t_vs_null:+.2f}, tested {rec.tested_on}, "
                      f"{rec.window} @ {rec.stop_atr}/{rec.target_atr} ATR")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([asdict(r) for r in self.records.values()],
                                indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        p = Path(path)
        if not p.exists():
            return cls()
        reg = cls()
        for row in json.loads(p.read_text(encoding="utf-8")):
            reg.add(ValidationRecord(**row))
        return reg
