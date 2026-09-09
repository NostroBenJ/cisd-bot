"""v30 backtest -- does fixing the Pine's defects change the answer?

    python backtest_v30.py --config v30  --out results/v30.json
    python backtest_v30.py --config null --out results/null.json
    python backtest_v30.py --summarize results/v29.json results/aug6.json results/v30.json results/null.json

Three builds of the same model, run over every archived 1-minute SPY session
(data/uw/SPY_ohlc_1m, July 2022 onward) and scored the same way:

    v29    the Pine as the user ran it: every known defect present
           (`pine_bug_compat`), the port's own additions switched off.
    aug6   v29 plus the six defects fixed in August (anchor gate, free grade
           credit, two-day-stale dealing range, latching bias, mixed ATR
           scales, midnight quota, stale SMT).
    v30    aug6 plus the two fixes found in September: the trigger fires on
           the FIRST close through the initiating open, and a tap stays armed
           for the trigger's own lookback instead of dying the bar price steps
           out of the zone. This is `CISD_Powell_RB_v30.pine`.

The port's additions (volume, VWAP, gamma, timeframe agreement, lull penalty,
opening-noise skip, traded window) stay OFF in all three, because the question
is what the Pine does, not what the port could do.

Scoring is `ablation.evaluate`: entry at the signal bar's close, first touch of
the stop or target-1 within the same session, a bar that spans both counts as
a STOP, unresolved trades marked at the session close. A friction of 0.02 ATR
per trade is charged on top (`r_net`), the same convention as the research
harness, so the signal and the random-entry null are on one footing.

The null (`--config null`) takes one random entry per session, random
direction, at a grid of ATR-scaled stop/target structures averaged over many
seeds. The summary picks the structure closest to what the v30 signals
actually used and reports the t-statistic against it -- against the
structure's own base rate, not against zero.

Limits, stated once: SPY only, so SMT divergence is unavailable in every build
(it needs ES/YM or QQQ/DIA bars); the underlying leg, not options P&L; no
spread. A negative result here is decisive, a positive one is necessary and
not sufficient.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ablation import evaluate  # noqa: E402
from bot.feed import ArchiveFeed  # noqa: E402
from cisd.bars import NY, Bar  # noqa: E402
from cisd.config import Config, pine_bug_compat  # noqa: E402
from cisd.engine import CisdEngine  # noqa: E402
from cisd.indicators import atr  # noqa: E402
from cisd.sessions import in_window, parse_session  # noqa: E402
from research.harness import evaluate_entry  # noqa: E402

COST_ATR = 0.02

SIX_FIXES = dict(
    gate_intrinsic_any_sweep=False,
    free_sweep_credit=False,
    prev_period_lag=1,
    bias_expires=True,
    unify_atr_timeframe=True,
    quota_resets_at_session=True,
    smt_max_age_bars=12,
)
NEW_FIXES = dict(cisd_first_close=True, armed_mode="touch_window")

NULL_STOPS = (0.5, 1.0, 1.5, 2.0)
NULL_TARGETS_R = (1.0, 2.0, 3.0)
NULL_SEEDS = 20
NULL_WINDOW = "0930-1600"


def build_config(name: str) -> Config:
    plain = Config(rb_timeframes=(5, 15), bias_timeframe=60, htf_fvg_timeframe=60)
    compat = pine_bug_compat(plain)
    if name == "v29":
        return compat
    if name == "aug6":
        return replace(compat, **SIX_FIXES)
    if name == "v30":
        return replace(compat, **SIX_FIXES, **NEW_FIXES)
    raise SystemExit(f"unknown config {name!r}")


def load_bars(years: set[int] | None) -> tuple[list[Bar], int]:
    feed = ArchiveFeed()
    days = feed.sessions()
    if years:
        days = [d for d in days if int(d[:4]) in years]
    bars: list[Bar] = []
    for d in days:
        bars.extend(feed.session_bars(d))
    return bars, len(days)


def _ny(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, NY)


# ── model runs ──────────────────────────────────────────────────────────────


def run_model(name: str, bars: list[Bar], sessions: int) -> dict:
    return run_config(build_config(name), bars, sessions, name)


def run_config(cfg: Config, bars: list[Bar], sessions: int, name: str) -> dict:
    index_of = {b.ts: i for i, b in enumerate(bars)}
    a = atr(bars, cfg.atr_length)

    engine = CisdEngine(cfg, symbol="SPY")
    sigs = engine.run(bars)

    trades = []
    for s in sigs:
        o = evaluate(s, bars, index_of)
        if o is None:
            continue
        i = index_of[s.ts]
        risk = s.risk_per_share
        cost_r = COST_ATR * a[i] / risk if risk > 0 and not math.isnan(a[i]) else 0.0
        trades.append({
            "ts": s.ts,
            "date": _ny(s.ts).strftime("%Y-%m-%d %H:%M"),
            "direction": s.direction,
            "entry": s.entry,
            "stop": s.stop,
            "target": s.targets[0].price if s.targets else None,
            "target_r": s.targets[0].r_multiple if s.targets else None,
            "stop_atr": risk / a[i] if a[i] > 0 else None,
            "outcome": o.result,
            "r": o.r_multiple,
            "r_net": o.r_multiple - cost_r,
            "bars_held": o.bars_held,
            "grade": s.grade,
            "score": s.score,
            "timeframe": s.timeframe,
            "setup_type": s.setup_type,
            "bias": s.bias,
        })

    st = engine.stats
    return {
        "config": name,
        "bars": len(bars),
        "sessions": sessions,
        "first": _ny(bars[0].ts).strftime("%Y-%m-%d"),
        "last": _ny(bars[-1].ts).strftime("%Y-%m-%d"),
        "signals": len(sigs),
        "funnel": {
            "formed": st.blocks_formed,
            "confirmed": st.blocks_confirmed,
            "armed": st.blocks_armed,
            "triggers": st.triggers,
            "signals": st.signals,
        },
        "rejections": dict(sorted(st.rejected.items(), key=lambda kv: -kv[1])),
        "trades": trades,
    }


# ── the random-entry null ───────────────────────────────────────────────────


def run_null(bars: list[Bar], sessions: int) -> dict:
    a = atr(bars, 14)
    start, end = parse_session(NULL_WINDOW)
    by_session: dict[str, list[int]] = {}
    for i, b in enumerate(bars):
        if in_window(b.minute_of_day_ny, start, end):
            by_session.setdefault(b.session_date_ny, []).append(i)

    grid = {}
    for stop_atr in NULL_STOPS:
        for tgt_r in NULL_TARGETS_R:
            target_atr = stop_atr * tgt_r
            rs_all: list[float] = []
            per_seed_means: list[float] = []
            for seed in range(NULL_SEEDS):
                rng = random.Random(1000 + seed)
                rs: list[float] = []
                for day, idxs in by_session.items():
                    i = rng.choice(idxs)
                    d = 1 if rng.random() < 0.5 else -1
                    t = evaluate_entry(bars, i, d, a[i], stop_atr, target_atr, 10_000, COST_ATR)
                    if t is not None:
                        rs.append(t.r_multiple)
                if rs:
                    per_seed_means.append(statistics.mean(rs))
                    rs_all.extend(rs)
            key = f"{stop_atr}/{tgt_r}"
            grid[key] = {
                "stop_atr": stop_atr,
                "target_r": tgt_r,
                "n_per_seed": len(rs_all) / NULL_SEEDS,
                "mean_r_net": statistics.mean(rs_all) if rs_all else math.nan,
                "sd_r": statistics.pstdev(rs_all) if rs_all else math.nan,
                "seed_spread": statistics.pstdev(per_seed_means) if per_seed_means else math.nan,
                "win": sum(1 for r in rs_all if r > 0) / len(rs_all) if rs_all else math.nan,
            }
    return {
        "config": "null",
        "bars": len(bars),
        "sessions": sessions,
        "first": _ny(bars[0].ts).strftime("%Y-%m-%d"),
        "last": _ny(bars[-1].ts).strftime("%Y-%m-%d"),
        "window": NULL_WINDOW,
        "seeds": NULL_SEEDS,
        "grid": grid,
    }


# ── summary ─────────────────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else math.nan


def _se(xs: list[float]) -> float:
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else math.nan


def _fmt(x: float, w: int = 7, p: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'--':>{w}}"
    return f"{x:{'+' if sign else ''}{w}.{p}f}"


def summarize(paths: list[str]) -> None:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    null = next((r for r in runs if r["config"] == "null"), None)
    models = [r for r in runs if r["config"] != "null"]
    if not models:
        print("no model runs to summarise")
        return

    span = models[0]
    years = (datetime.strptime(span["last"], "%Y-%m-%d") - datetime.strptime(span["first"], "%Y-%m-%d")).days / 365.25
    print("=" * 92)
    print(f"CISD v29 -> v30 backtest  ·  1-minute SPY  ·  {span['first']} to {span['last']}  "
          f"({span['sessions']} sessions, {years:.1f} years)")
    print("=" * 92)
    print("  entry = signal-bar close · first touch of stop / target-1, same session · both-in-one-bar = STOP")
    print(f"  r_net charges {COST_ATR} ATR per trade · underlying leg only · SMT unavailable (SPY-only archive)")

    # pick the null structure nearest the v30 (or last) model's medians
    ref = next((r for r in models if r["config"] == "v30"), models[-1])
    stops = [t["stop_atr"] for t in ref["trades"] if t.get("stop_atr")]
    tgts = [t["target_r"] for t in ref["trades"] if t.get("target_r")]
    med_stop = statistics.median(stops) if stops else math.nan
    med_tgt = statistics.median(tgts) if tgts else math.nan
    null_row = None
    if null and stops and tgts:
        null_row = min(
            null["grid"].values(),
            key=lambda g: abs(math.log(g["stop_atr"] / med_stop)) + abs(math.log(g["target_r"] / med_tgt)),
        )

    print(f"\n  {ref['config']} signals: median stop {_fmt(med_stop, 5)} ATR, median target-1 {_fmt(med_tgt, 5)} R")
    if null_row:
        print(f"  null structure used: {null_row['stop_atr']} ATR stop / {null_row['target_r']} R target, "
              f"{null['seeds']} seeds, ~{null_row['n_per_seed']:.0f} trades each: "
              f"mean r_net {_fmt(null_row['mean_r_net'], 6, 3, True)}  win {100 * null_row['win']:.0f}%  "
              f"(seed-to-seed spread {_fmt(null_row['seed_spread'], 5, 3)})")

    print("\n" + "-" * 92)
    print(f"  {'build':<6} {'sigs':>5} {'/yr':>5} {'win':>5} {'meanR':>8} {'r_net':>8} {'se':>7} "
          f"{'t vs 0':>7} {'t vs null':>9}  {'tgt':>4} {'stop':>4} {'time':>4}  {'drop best':>9}")
    print("-" * 92)
    for r in models:
        tr = r["trades"]
        rs = [t["r"] for t in tr]
        rn = [t["r_net"] for t in tr]
        n = len(rn)
        win = sum(1 for x in rn if x > 0) / n if n else math.nan
        se = _se(rn)
        t0 = _mean(rn) / se if n > 1 and se > 0 else math.nan
        tnull = math.nan
        if null_row and n > 1:
            # Welch against the null's pooled draws (its per-trade SD and its
            # effective n of one seed's worth of sessions).
            va = statistics.variance(rn)
            vb = null_row["sd_r"] ** 2
            nb = null_row["n_per_seed"]
            denom = math.sqrt(va / n + vb / nb)
            tnull = (_mean(rn) - null_row["mean_r_net"]) / denom if denom else math.nan
        outs = {"target": 0, "stop": 0, "time": 0}
        for t in tr:
            outs[t["outcome"]] = outs.get(t["outcome"], 0) + 1
        drop = _mean(sorted(rn)[:-1]) if n > 2 else math.nan
        print(f"  {r['config']:<6} {r['signals']:>5} {r['signals'] / years:>5.1f} "
              f"{100 * win if n else float('nan'):>4.0f}% {_fmt(_mean(rs), 8, 3, True)} {_fmt(_mean(rn), 8, 3, True)} "
              f"{_fmt(se, 7, 3)} {_fmt(t0, 7, 2, True)} {_fmt(tnull, 9, 2, True)}  "
              f"{outs['target']:>4} {outs['stop']:>4} {outs['time']:>4}  {_fmt(drop, 9, 3, True)}")
    print("-" * 92)

    # funnel and per-year
    print(f"\n  {'build':<6} {'formed':>8} {'confirmed':>10} {'armed':>7} {'triggers':>9} {'signals':>8}   per-year signals")
    for r in models:
        f = r["funnel"]
        by_year: dict[str, int] = {}
        for t in r["trades"]:
            y = t["date"][:4]
            by_year[y] = by_year.get(y, 0) + 1
        yrs = "  ".join(f"{y}:{c}" for y, c in sorted(by_year.items()))
        print(f"  {r['config']:<6} {f['formed']:>8} {f['confirmed']:>10} {f['armed']:>7} {f['triggers']:>9} {f['signals']:>8}   {yrs}")

    for r in models:
        if r.get("rejections"):
            top = ", ".join(f"{k} {v}" for k, v in list(r["rejections"].items())[:5])
            print(f"  {r['config']} rejections: {top}")

    # direction split
    print(f"\n  {'build':<6} {'longs':>6} {'meanR':>8}   {'shorts':>6} {'meanR':>8}")
    for r in models:
        L = [t["r_net"] for t in r["trades"] if t["direction"] == "long"]
        S = [t["r_net"] for t in r["trades"] if t["direction"] == "short"]
        print(f"  {r['config']:<6} {len(L):>6} {_fmt(_mean(L), 8, 3, True)}   {len(S):>6} {_fmt(_mean(S), 8, 3, True)}")

    # overlap between builds: which v29 trades survive into v30?
    if len(models) >= 2:
        sets = {r["config"]: {t["ts"] for t in r["trades"]} for r in models}
        names = [r["config"] for r in models]
        print("\n  shared signal bars between builds:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_, b_ = names[i], names[j]
                print(f"    {a_} ∩ {b_}: {len(sets[a_] & sets[b_])}  (of {len(sets[a_])} / {len(sets[b_])})")

    print("\n  Read: t vs null is the number that matters. |t| < 2 is indistinguishable from")
    print("  random entries with the same exit structure. 'drop best' is the mean after")
    print("  removing the single best trade -- if the sign flips, the result is one trade.")


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=["v29", "aug6", "v30", "null"])
    ap.add_argument("--out")
    ap.add_argument("--years", help="comma-separated years to include, default all")
    ap.add_argument("--summarize", nargs="*")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return 0
    if not args.config or not args.out:
        ap.error("--config and --out are required unless --summarize")

    years = {int(y) for y in args.years.split(",")} if args.years else None
    bars, sessions = load_bars(years)
    if len(bars) < 1000:
        raise SystemExit(f"only {len(bars)} bars loaded")
    print(f"{args.config}: {len(bars)} bars, {sessions} sessions, "
          f"{_ny(bars[0].ts):%Y-%m-%d} to {_ny(bars[-1].ts):%Y-%m-%d}", flush=True)

    res = run_null(bars, sessions) if args.config == "null" else run_model(args.config, bars, sessions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if args.config == "null":
        print("null grid written")
    else:
        rn = [t["r_net"] for t in res["trades"]]
        print(f"{args.config}: {res['signals']} signals, {len(rn)} resolved, mean r_net {_mean(rn):+.3f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
