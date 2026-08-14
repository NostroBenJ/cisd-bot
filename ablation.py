"""Gate ablation: what does each filter cost in signals, and is it worth it?

    python ablation.py data/tv_export.csv

The indicator produces roughly eight signals a year at its live settings. That
is too few to trade and far too few to measure. This asks, one gate at a time:
turn it off and how many signals come back -- and, crucially, **are the
recovered signals any good**.

Counting alone would be actively misleading. The gate that costs the most
signals might be the one doing the most work; removing it would look like
progress right up until the money moved. So every scenario is scored on the
underlying leg as well.

**What the outcome measure is, exactly.** From the bar after entry, walk
forward within the same session until the stop or the first target is touched.
Whichever is touched first decides the trade. Anything unresolved by the
session close is marked out at the close as a time stop. Three deliberate
conservatisms:

  * A bar whose range spans BOTH the stop and the target counts as a **stop**.
    Intrabar order is unknowable from OHLC, and the optimistic assumption is
    how backtests manufacture edge that does not exist.
  * Entry is the close of the signal bar, with no slippage and no spread. Real
    fills are worse, so every number here is an upper bound.
  * This is the **underlying** leg, not options P&L. A 2R move in SPY is not a
    2R move in a call -- theta, spread and the delta path all intervene. That
    is stage 4; this is a necessary condition, not a sufficient one.

**On the statistics.** With single-digit signal counts nothing here is
significant, and the report says so rather than printing a tidy win rate over
six trades. The point of the exercise is to find a configuration that produces
enough signals to be measurable at all.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, replace

from cisd.bars import NY, Bar
from cisd.config import Config, pine_bug_compat
from cisd.engine import CisdEngine
from cisd.signal import Signal
from tools.tv_import import load


@dataclass
class Outcome:
    signal: Signal
    result: str        # "target" | "stop" | "time"
    exit_price: float
    r_multiple: float
    bars_held: int


def evaluate(sig: Signal, bars: list[Bar], index_of: dict[int, int]) -> Outcome | None:
    """Resolve one signal against the underlying, same session only."""
    i = index_of.get(sig.ts)
    if i is None or not sig.targets:
        return None
    target = sig.targets[0].price
    day = bars[i].session_date_ny
    risk = sig.risk_per_share
    if risk <= 0:
        return None

    j = i + 1
    while j < len(bars) and bars[j].session_date_ny == day:
        b = bars[j]
        if sig.is_long:
            hit_stop = b.low <= sig.stop
            hit_target = b.high >= target
        else:
            hit_stop = b.high >= sig.stop
            hit_target = b.low <= target
        # Both in one bar: assume the stop. See the module docstring.
        if hit_stop:
            return Outcome(sig, "stop", sig.stop, -1.0, j - i)
        if hit_target:
            r = sig.targets[0].r_multiple
            return Outcome(sig, "target", target, r, j - i)
        j += 1

    if j - 1 <= i:
        return None
    close = bars[j - 1].close
    return Outcome(sig, "time", close, sig.r_multiple_at(close), j - 1 - i)


@dataclass
class Result:
    name: str
    signals: int
    sessions: int
    outcomes: list[Outcome]

    @property
    def resolved(self) -> int:
        return len(self.outcomes)

    @property
    def wins(self) -> int:
        return sum(1 for o in self.outcomes if o.result == "target")

    @property
    def stops(self) -> int:
        return sum(1 for o in self.outcomes if o.result == "stop")

    @property
    def times(self) -> int:
        return sum(1 for o in self.outcomes if o.result == "time")

    @property
    def hit_rate(self) -> float:
        return self.wins / self.resolved if self.resolved else math.nan

    @property
    def total_r(self) -> float:
        return sum(o.r_multiple for o in self.outcomes)

    @property
    def mean_r(self) -> float:
        return self.total_r / self.resolved if self.resolved else math.nan

    @property
    def se_r(self) -> float:
        """Standard error of mean R. Signals are near-independent here (rarely
        more than one per session), so no overlap correction is applied -- but
        with counts this small the SE is wide enough that it is the headline,
        not a footnote."""
        if self.resolved < 2:
            return math.nan
        return statistics.stdev([o.r_multiple for o in self.outcomes]) / math.sqrt(self.resolved)


def run(name: str, cfg: Config, bars: list[Bar], index_of: dict[int, int]) -> Result:
    engine = CisdEngine(cfg, symbol="SPY")
    sigs = engine.run(bars)
    outs = [o for o in (evaluate(s, bars, index_of) for s in sigs) if o is not None]
    sessions = len({bars[index_of[s.ts]].session_date_ny for s in sigs if s.ts in index_of})
    return Result(name, len(sigs), sessions, outs)


# Each entry: label, and the single change applied to the baseline.
ABLATIONS: list[tuple[str, dict]] = [
    ("anchor gate made real (defect 1)", {"gate_intrinsic_any_sweep": False}),
    ("track blocks down to B", {"min_grade_to_track": "B"}),
    ("track blocks down to C", {"min_grade_to_track": "C"}),
    ("signal on B grade", {"min_grade_to_signal": "B"}),
    ("no bias requirement", {"require_bias": False}),
    ("no confluence-stack gate", {"use_stack_gate": False}),
    ("stack >= 2 instead of 3", {"min_stack": 2}),
    ("draw-on-liquidity not gating", {"dol_mode": "Tag only (no gate)"}),
    ("no displacement requirement", {"require_displacement": False}),
    ("no strong-close requirement", {"require_strong_close": False}),
    ("no minimum sweep distance", {"require_min_sweep": False}),
    ("body >= 30% instead of 45%", {"min_body_fraction": 0.30}),
    ("no CISD trigger", {"use_cisd_trigger": False}),
    ("no anti-stack", {"anti_stack": False}),
    ("quota 12/session", {"max_signals_per_session": 12}),
    ("swing lookback 5 instead of 10", {"swing_lookback": 5}),
]


def fmt(r: Result, base: int) -> str:
    delta = r.signals - base
    d = f"{delta:+d}" if delta else "  0"
    if r.resolved == 0:
        return f"  {r.name:<34} {r.signals:>5} {d:>6}      --"
    hr = f"{100 * r.hit_rate:.0f}%"
    mr = f"{r.mean_r:+.2f}"
    se = f"+-{r.se_r:.2f}" if not math.isnan(r.se_r) else ""
    return (f"  {r.name:<34} {r.signals:>5} {d:>6}   {hr:>4} "
            f"{mr:>7} {se:>7}  {r.total_r:+7.1f}")


def evaluate_at_r(
    sig: Signal, bars: list[Bar], index_of: dict[int, int], r_target: float
) -> Outcome | None:
    """Resolve a signal against a target placed at `r_target` x risk.

    Separate from `evaluate` because the Pine's targets are multiples of the
    displacement leg, which is a different quantity per signal -- a 2.0 leg
    multiple came out around 3R on the examples checked. Placing the target in
    R lets the exit be swept independently of how big the displacement
    happened to be, which is the only way to ask whether the signals are
    directionally right but badly harvested."""
    i = index_of.get(sig.ts)
    if i is None:
        return None
    risk = sig.risk_per_share
    if risk <= 0:
        return None
    target = sig.entry + r_target * risk if sig.is_long else sig.entry - r_target * risk
    day = bars[i].session_date_ny

    j = i + 1
    while j < len(bars) and bars[j].session_date_ny == day:
        b = bars[j]
        if sig.is_long:
            hit_stop, hit_target = b.low <= sig.stop, b.high >= target
        else:
            hit_stop, hit_target = b.high >= sig.stop, b.low <= target
        if hit_stop:
            return Outcome(sig, "stop", sig.stop, -1.0, j - i)
        if hit_target:
            return Outcome(sig, "target", target, r_target, j - i)
        j += 1

    if j - 1 <= i:
        return None
    close = bars[j - 1].close
    return Outcome(sig, "time", close, sig.r_multiple_at(close), j - 1 - i)


def exit_sweep(cfg: Config, label: str, bars: list[Bar], index_of: dict[int, int]) -> None:
    """Is there ANY target distance that makes this signal set profitable?

    The signals are fixed; only the harvest changes. If the model is
    directionally right and merely over-reaching for targets, a nearer target
    will show it. If every target distance is negative, the entries themselves
    carry no edge and no exit rule will rescue them."""
    engine = CisdEngine(cfg, symbol="SPY")
    sigs = engine.run(bars)
    print(f"\n  exit sweep -- {label}  ({len(sigs)} signals)")
    print(f"    {'target':>8} {'win':>5} {'meanR':>8} {'se':>8} {'totR':>8}  "
          f"{'stops':>6} {'time':>5}")
    for rt in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        outs = [o for o in (evaluate_at_r(s, bars, index_of, rt) for s in sigs)
                if o is not None]
        if not outs:
            continue
        rs = [o.r_multiple for o in outs]
        wins = sum(1 for o in outs if o.result == "target")
        stops = sum(1 for o in outs if o.result == "stop")
        times = sum(1 for o in outs if o.result == "time")
        mean = sum(rs) / len(rs)
        se = statistics.stdev(rs) / math.sqrt(len(rs)) if len(rs) > 1 else math.nan
        se_s = f"+-{se:.2f}" if not math.isnan(se) else "     --"
        print(f"    {rt:>7.2f}R {100*wins/len(outs):>4.0f}% {mean:>+8.2f} {se_s:>8} "
              f"{sum(rs):>+8.1f}  {stops:>6} {times:>5}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--rb", default="5,15")
    ap.add_argument("--bias", type=int, default=60)
    ap.add_argument("--corrected", action="store_true",
                    help="ablate from the corrected build instead of the "
                         "as-you-run-it build")
    args = ap.parse_args()

    export = load(args.csv)
    bars = export.bars
    index_of = {b.ts: i for i, b in enumerate(bars)}
    sessions = len({b.session_date_ny for b in bars})

    rb = tuple(int(x) for x in args.rb.split(",") if x.strip())
    plain = Config(rb_timeframes=rb, bias_timeframe=args.bias, htf_fvg_timeframe=60)
    baseline_cfg = plain if args.corrected else pine_bug_compat(plain)

    print("=" * 84)
    print("CISD gate ablation")
    print("=" * 84)
    print(f"  {len(bars)} bars over {sessions} sessions, "
          f"{bars[0].dt_ny:%Y-%m-%d} to {bars[-1].dt_ny:%Y-%m-%d}")
    print(f"  baseline: {'corrected build' if args.corrected else 'as you run it (pine_bug_compat)'}")
    print("  outcome: first touch of stop or target-1, same session, "
          "ambiguous bars count as STOP")
    print("  underlying leg only -- NOT options P&L, and no slippage or spread")

    base = run("baseline", baseline_cfg, bars, index_of)
    print(f"\n  baseline: {base.signals} signals over {sessions} sessions "
          f"(one every {sessions / base.signals:.0f} sessions)"
          if base.signals else "\n  baseline produced NO signals")
    if base.resolved:
        print(f"            {base.wins} target / {base.stops} stop / {base.times} time-stop"
              f"   mean {base.mean_r:+.2f}R   total {base.total_r:+.1f}R")

    print("\n" + "-" * 84)
    print(f"  {'single change from baseline':<34} {'sigs':>5} {'delta':>6}   "
          f"{'win':>4} {'meanR':>7} {'se':>7}  {'totR':>7}")
    print("-" * 84)

    results = []
    for label, change in ABLATIONS:
        try:
            cfg = replace(baseline_cfg, **change)
        except TypeError as exc:
            print(f"  {label:<34}  SKIPPED ({exc})")
            continue
        r = run(label, cfg, bars, index_of)
        results.append(r)
        print(fmt(r, base.signals))

    print("-" * 84)

    ranked = sorted(results, key=lambda r: -(r.signals - base.signals))
    top = [r for r in ranked if r.signals > base.signals][:3]
    if top:
        print("\n  Biggest signal recovery:")
        for r in top:
            n = r.signals - base.signals
            rate = sessions / r.signals if r.signals else float("inf")
            print(f"    {r.name}: +{n} signals (one every {rate:.0f} sessions)")

    # Sweep the exit on the baseline and on the largest signal set available.
    # A small sample cannot show much; the widest one is where an edge would
    # first become visible if there were one.
    exit_sweep(baseline_cfg, "baseline", bars, index_of)
    widest = max(results, key=lambda r: r.signals) if results else None
    if widest is not None and widest.signals > base.signals:
        change = dict(ABLATIONS[[r.name for r in results].index(widest.name)][1])
        exit_sweep(replace(baseline_cfg, **change), widest.name, bars, index_of)

    print("\n  Read this as cost, not as improvement. A gate that costs many")
    print("  signals may be the one earning its keep -- compare mean R, and")
    print("  treat any mean R whose standard error straddles zero as unproven.")
    print("  Options P&L is stage 4; a positive underlying leg is necessary,")
    print("  not sufficient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
