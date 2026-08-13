"""Diff the Python port against a TradingView export, signal for signal.

    python compare_tv.py data/tv_export.csv

Until this passes, the port is unverified against the thing it copies and its
signal set should not be treated as the Pine's.

**It runs in `pine_bug_compat` mode.** The point is to reproduce the original
INCLUDING its six defects. A diff against the corrected build would disagree by
design and prove nothing. Once parity holds, the fixes can be turned on one at
a time and the change in the signal set is then a measurement rather than a
hope.

The comparison is layered, because "the signals differ" is not a diagnosis:

  layer 1  bars        did we even load the same series
  layer 2  ATR         the scale every threshold is measured in
  layer 3  PDH/PDL     the dealing range the grader leans on
  layer 4  bias        the 60m directional gate
  layer 5  signals     the actual output

A mismatch at layer 2 makes every later layer meaningless, so they are reported
in order and the first failing layer is what to fix.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import replace

from cisd.bars import NY
from cisd.config import Config, pine_bug_compat
from cisd.engine import CisdEngine, build_period_ranges
from cisd.indicators import atr
from tools.tv_import import infer_timeframe_minutes, load

BAR_TOL = 1e-6


def _fmt_ts(ts: int) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(ts, NY).strftime("%Y-%m-%d %H:%M")


def _rel_err(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return math.nan
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def layer_bars(export) -> int:
    print("\nlayer 1 · bars")
    bars = export.bars
    tf = infer_timeframe_minutes(bars)
    print(f"  {len(bars)} bars, inferred timeframe {tf}m")
    print(f"  {_fmt_ts(bars[0].ts)}  ->  {_fmt_ts(bars[-1].ts)}")

    bad = [b for b in bars if not (b.low <= min(b.open, b.close)
                                   and b.high >= max(b.open, b.close))]
    if bad:
        print(f"  FAIL {len(bad)} bars violate OHLC invariants")
        return tf

    zero_vol = sum(1 for b in bars if b.volume <= 0)
    if zero_vol == len(bars):
        # Every bar zero almost always means the export carried no volume
        # column at all, which is fine for a parity diff (compat mode reads no
        # volume) but fatal for stage 3, where the volume terms are live.
        print("  warn no volume in this export -- fine for the parity diff, "
              "but stage 3 needs a feed that carries it")
    elif zero_vol:
        pct = 100.0 * zero_vol / len(bars)
        # A PARTIALLY zero volume column is the phantom-bar signature that
        # Robinhood's history had, and that is worth shouting about.
        print(f"  FAIL {zero_vol} bars ({pct:.1f}%) have zero volume while "
              f"others do not -- suspect synthesised gap-fill bars")
    print("  ok")
    return tf


def layer_atr(export, cfg: Config) -> None:
    print("\nlayer 2 · ATR")
    tv = export.series.get("x_atr")
    if not tv:
        print("  skipped (no x_atr column -- export patch not applied?)")
        return
    ours = atr(export.bars, cfg.atr_length)

    # TradingView computes ATR over more history than the export contains, so
    # the first values disagree until Wilder smoothing has washed the seed out.
    warm = min(len(ours), max(cfg.atr_length * 10, 200))
    errs = [
        _rel_err(a, b) for a, b in zip(ours[warm:], tv[warm:])
        if not math.isnan(a) and not math.isnan(b)
    ]
    if not errs:
        print("  FAIL no overlapping ATR values to compare")
        return
    med = statistics.median(errs)
    worst = max(errs)
    print(f"  compared {len(errs)} values after {warm}-bar warmup")
    print(f"  median relative error {med:.2e}   worst {worst:.2e}")
    if med < 1e-4:
        print("  ok")
    else:
        print("  FAIL ATR disagrees -- every ATR-scaled threshold is off, "
              "fix this before reading any later layer")


def layer_dealing_range(export, cfg: Config) -> None:
    print("\nlayer 3 · previous-day range")
    tv_h = export.series.get("x_pdh")
    tv_l = export.series.get("x_pdl")
    if not tv_h or not tv_l:
        print("  skipped (no x_pdh / x_pdl columns)")
        return

    prev_day, _ = build_period_ranges(export.bars, cfg.prev_period_lag)
    mismatches = 0
    compared = 0
    first: tuple[str, float, float] | None = None
    for i, b in enumerate(export.bars):
        ph, pl = prev_day.get(b.session_date_ny, (math.nan, math.nan))
        if math.isnan(ph) or math.isnan(tv_h[i]):
            continue
        compared += 1
        if _rel_err(ph, tv_h[i]) > 1e-6 or _rel_err(pl, tv_l[i]) > 1e-6:
            mismatches += 1
            if first is None:
                first = (_fmt_ts(b.ts), ph, tv_h[i])

    print(f"  compared {compared} bars, {mismatches} mismatched")
    if compared == 0:
        print("  warn nothing to compare -- export may not span two sessions")
    elif mismatches == 0:
        print("  ok")
    else:
        print(f"  FAIL first at {first[0]}: ours {first[1]:.2f} vs TV {first[2]:.2f}")
        print("  likely cause: the export covers regular hours only while the "
              "Pine's daily security call includes extended hours, or vice versa")


def _compare_series(
    label: str, ours: list[float], tv: list[float] | None, tol: float
) -> bool:
    """Compare one port-computed series against its TradingView column.

    Bars where either side is nan are skipped rather than counted as matches:
    a nan on both sides usually means "not warmed up yet", and agreeing
    trivially there would paper over a real gap elsewhere."""
    if tv is None:
        print(f"    {label:<15} skipped (column absent -- export patch v1?)")
        return True
    n = min(len(ours), len(tv))
    compared = mismatched = 0
    first: tuple[int, float, float] | None = None
    for i in range(n):
        a, b = ours[i], tv[i]
        if math.isnan(a) or math.isnan(b):
            continue
        compared += 1
        if abs(a - b) > tol:
            mismatched += 1
            if first is None:
                first = (i, a, b)
    if compared == 0:
        print(f"    {label:<15} nothing to compare")
        return True
    pct = 100.0 * mismatched / compared
    status = "ok  " if mismatched == 0 else "FAIL"
    line = f"    {label:<15} {status} {compared - mismatched}/{compared} match"
    if mismatched:
        line += f"  ({pct:.1f}% differ)"
    print(line)
    if first is not None:
        print(f"                    first at bar {first[0]}: "
              f"ours {first[1]:.4f} vs TV {first[2]:.4f}")
    return mismatched == 0


def layer_context(export, cfg: Config) -> None:
    """Per-bar diff of every context input the grader reads.

    This is where a six-point score gap becomes a named cause. Each row feeds
    a specific grade term and is annotated with what that term is worth, so a
    failure here maps directly onto an observed score difference."""
    print("\nlayer 3b · context series (per bar)")
    engine = CisdEngine(cfg, symbol="SPY", collect_diagnostics=True)
    engine.run(export.bars)
    d = engine.diagnostics
    g = export.series.get

    if g("x_pwh") is None:
        print("    export patch v1 detected -- re-export with v2 to check the")
        print("    weekly range, depth function, draw targets and sweeps.")

    print("  dealing range (feeds premium/discount, worth up to 22 points)")
    _compare_series("prev day high", d["pdh"], g("x_pdh"), 1e-4)
    _compare_series("prev day low", d["pdl"], g("x_pdl"), 1e-4)
    _compare_series("prev week high", d["pwh"], g("x_pwh"), 1e-4)
    _compare_series("prev week low", d["pwl"], g("x_pwl"), 1e-4)

    print("  the depth function itself (0.65/0.35 blend, 1.5 exponent)")
    _compare_series("depth bull", d["pd_depth_bull"], g("x_pd_bull"), 1e-6)
    _compare_series("depth bear", d["pd_depth_bear"], g("x_pd_bear"), 1e-6)

    print("  draw on liquidity (reversal vs continuation = 6 points)")
    _compare_series("draw above", d["dol_up"], g("x_dol_up"), 1e-4)
    _compare_series("draw below", d["dol_dn"], g("x_dol_dn"), 1e-4)

    print("  engineered liquidity sweeps (16 points)")
    _compare_series("sweep bull", d["sweep_bull"], g("x_swp_bull"), 0.5)
    _compare_series("sweep bear", d["sweep_bear"], g("x_swp_bear"), 0.5)


def layer_bias(export, cfg: Config) -> None:
    print("\nlayer 4 · top-down bias")
    tv = export.series.get("x_bias")
    if not tv:
        print("  skipped (no x_bias column)")
        return
    vals = [v for v in tv if not math.isnan(v)]
    if not vals:
        print("  FAIL bias column is entirely empty")
        return
    ups = sum(1 for v in vals if v > 0)
    downs = sum(1 for v in vals if v < 0)
    zeros = sum(1 for v in vals if v == 0)
    print(f"  TV bias distribution: +1={ups}  -1={downs}  0={zeros}")
    if zeros == len(vals):
        print("  warn bias never left zero -- with the Pine's `biasDir >= 0` "
              "test that reads as permanently BULLISH, which is defect 3")
    else:
        print("  ok (distribution recorded; per-bar diff needs the port to "
              "expose its bias series -- see README stage 2)")


def layer_signals(export, cfg: Config, score_tol: float) -> bool:
    print("\nlayer 5 · signals")
    if not export.has_export_patch():
        print("  FAIL no x_sig_dir column.")
        print("  Append tools/pine_export_patch.pine to the indicator and "
              "re-export -- the Pine has no plot() calls of its own, so a")
        print("  plain export carries OHLCV and nothing else.")
        return False

    tv_rows = export.signal_rows()
    engine = CisdEngine(cfg, symbol="SPY")
    ours = engine.run(export.bars)

    tv_by_ts = {ts: (d, sc, st, dp, ty) for ts, d, sc, st, dp, ty in tv_rows}
    our_by_ts = {s.ts: s for s in ours}

    matched, wrong_dir, wrong_score = [], [], []
    for ts, (d, sc, _st, _dp, _ty) in tv_by_ts.items():
        s = our_by_ts.get(ts)
        if s is None:
            continue
        our_dir = 1 if s.is_long else -1
        if our_dir != d:
            wrong_dir.append(ts)
        elif not math.isnan(sc) and abs(s.score - sc) > score_tol:
            wrong_score.append((ts, s.score, sc))
        else:
            matched.append(ts)

    tv_only = sorted(set(tv_by_ts) - set(our_by_ts))
    port_only = sorted(set(our_by_ts) - set(tv_by_ts))

    print(f"  TradingView signals : {len(tv_rows)}")
    print(f"  port signals        : {len(ours)}")
    print(f"  matched             : {len(matched)}")
    print(f"  direction mismatch  : {len(wrong_dir)}")
    print(f"  score mismatch      : {len(wrong_score)} (tolerance {score_tol})")
    print(f"  TradingView only    : {len(tv_only)}")
    print(f"  port only           : {len(port_only)}")

    for ts in tv_only[:5]:
        d, sc, _, _, _ = tv_by_ts[ts]
        print(f"    missed  {_fmt_ts(ts)}  dir={d:+d} score={sc:.1f}")
    for ts in port_only[:5]:
        s = our_by_ts[ts]
        print(f"    extra   {_fmt_ts(ts)}  {s.direction} score={s.score:.1f} "
              f"({s.timeframe}m, {s.setup_type})")
    for ts, a, b in wrong_score[:5]:
        print(f"    score   {_fmt_ts(ts)}  ours {a:.1f} vs TV {b:.1f}")

    exact = (
        len(tv_rows) > 0
        and not tv_only and not port_only
        and not wrong_dir and not wrong_score
    )
    if exact:
        print("\n  PARITY -- the port reproduces the Pine exactly.")
    elif tv_rows:
        agree = len(matched) / max(len(tv_by_ts), 1) * 100
        print(f"\n  {agree:.0f}% of TradingView signals reproduced. Not parity.")
        print("  Work the earliest divergence first; later ones are usually "
              "downstream of it (block state carries forward).")
    else:
        print("\n  The export contains no signals. Check the indicator "
              "settings produce signals over this range before diffing.")
    return exact


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="TradingView chart-data export")
    ap.add_argument("--rb", default="5,15",
                    help="RB timeframes in minutes, as set in the indicator")
    ap.add_argument("--bias", type=int, default=60, help="bias timeframe")
    ap.add_argument("--fvg", type=int, default=60, help="HTF FVG timeframe")
    ap.add_argument("--score-tol", type=float, default=0.5,
                    help="allowed absolute score difference")
    ap.add_argument("--corrected", action="store_true",
                    help="run the CORRECTED build instead of pine_bug_compat. "
                         "Expect disagreement -- this measures the fixes, it "
                         "does not test parity.")
    args = ap.parse_args()

    export = load(args.csv)
    rb = tuple(int(x) for x in args.rb.split(",") if x.strip())

    base = Config(rb_timeframes=rb, bias_timeframe=args.bias,
                  htf_fvg_timeframe=args.fvg)
    cfg = base if args.corrected else pine_bug_compat(base)

    print("=" * 70)
    print("CISD port vs TradingView")
    print("=" * 70)
    print(f"  mode: {'CORRECTED (not a parity test)' if args.corrected else 'pine_bug_compat'}")
    print(f"  rb timeframes {rb}, bias {args.bias}m, fvg {args.fvg}m")

    tf = layer_bars(export)
    if not args.corrected and tf not in rb:
        print(f"\n  note: chart timeframe is {tf}m and is not among the RB "
              f"timeframes {rb}. That is normal -- the Pine reads the CISD "
              f"trigger on the chart timeframe beneath the RB arrays.")

    layer_atr(export, cfg)
    layer_dealing_range(export, cfg)
    layer_context(export, cfg)
    layer_bias(export, cfg)
    ok = layer_signals(export, cfg, args.score_tol)

    print("\n" + "=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
