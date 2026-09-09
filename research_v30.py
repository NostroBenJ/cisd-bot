"""Split-sample research on the v30 build -- can the model be made to work?

    python research_v30.py --period is  --run all          # ablations, in-sample
    python research_v30.py --period is  --report           # table + exit sweep
    python research_v30.py --period oos --run base,<name>  # ONE out-of-sample test
    python research_v30.py --period oos --report

The archive is split once and never re-cut:

    is    2022-07-11 .. 2024-12-31   selection: ablate gates, choose exits
    oos   2025-01-01 .. 2026-08-12   confirmation: the chosen configuration,
                                     evaluated ONCE

Everything selected on `is` is a hypothesis until `oos` agrees. The report
prints how many configurations have been looked at on each period, because
that number is part of the result: the best of twenty noise draws has an
expected t of about +2 on its own.

Each ablation is a single change from the v30 build (`backtest_v30.build_
config("v30")`), scored exactly as `backtest_v30.run_config` scores it, and
compared with the random-entry null at the nearest exit structure. Results are
cached in results/ablate/<period>_<name>.json and never overwritten, so runs
are resumable and a number, once printed, does not move.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ablation import evaluate_at_r  # noqa: E402
from backtest_v30 import build_config, run_config, run_null, _fmt, _mean, _se  # noqa: E402
from bot.feed import ArchiveFeed  # noqa: E402
from cisd.bars import NY, Bar  # noqa: E402
from cisd.config import Config  # noqa: E402
from cisd.engine import CisdEngine  # noqa: E402

OUT = ROOT / "results" / "ablate"
SPLIT = "2025-01-01"

# Single changes from the v30 build. Order is by how much structure each one
# releases (from the rejection counts of the full-sample run), so the early
# entries are the ones that decide whether a measurable sample exists at all.
ABLATIONS: dict[str, dict] = {
    "base": {},
    "track_B": {"min_grade_to_track": "B"},
    "track_C": {"min_grade_to_track": "C"},
    "signal_B": {"min_grade_to_signal": "B"},
    "anchor_any_sweep": {"gate_intrinsic_any_sweep": True},
    "no_bias_req": {"require_bias": False},
    "no_stack_gate": {"use_stack_gate": False},
    "stack_2": {"min_stack": 2},
    "dol_tag_only": {"dol_mode": "Tag only (no gate)"},
    "no_strong_close": {"require_strong_close": False},
    "no_min_sweep": {"require_min_sweep": False},
    "body_030": {"min_body_fraction": 0.30},
    "no_cisd_trigger": {"use_cisd_trigger": False},
    "no_anti_stack": {"anti_stack": False},
    "swing_5": {"swing_lookback": 5},
    "add_1m_blocks": {"base_timeframe": 1},
    "quota_12": {"max_signals_per_session": 12},
    "cisd_swing_anchored": {"cisd_mode": "swing_anchored"},
}

# Combinations are added here AFTER the single-change table has been read,
# each with the reason it was chosen. Anything listed here counts as a trial.
COMBOS: dict[str, dict] = {
    # Chosen from the in-sample single-change table on 2026-09-08. The gates
    # barely bind; the signal count is set by the formation rules and the
    # timeframe set. These are the three changes that release the most
    # signals without collapsing the win rate (no_cisd_trigger did collapse
    # it), combined to get a sample large enough to measure at all.
    "wide": {"base_timeframe": 1, "require_min_sweep": False, "min_body_fraction": 0.30},
    # wide with every discretionary gate off: the loosest defensible build.
    "wide_open": {"base_timeframe": 1, "require_min_sweep": False, "min_body_fraction": 0.30,
                  "dol_mode": "Tag only (no gate)", "require_bias": False, "use_stack_gate": False},
    # the two strongest single changes only
    "sweep_1m": {"base_timeframe": 1, "require_min_sweep": False},
}


def period_bars(period: str) -> tuple[list[Bar], int]:
    feed = ArchiveFeed()
    days = feed.sessions()
    if period == "is":
        days = [d for d in days if d < SPLIT]
    elif period == "oos":
        days = [d for d in days if d >= SPLIT]
    elif period != "all":
        raise SystemExit(f"unknown period {period!r}")
    bars: list[Bar] = []
    for d in days:
        bars.extend(feed.session_bars(d))
    return bars, len(days)


def config_for(name: str) -> Config:
    base = build_config("v30")
    if name in ABLATIONS:
        return replace(base, **ABLATIONS[name])
    if name in COMBOS:
        return replace(base, **COMBOS[name])
    raise SystemExit(f"unknown configuration {name!r}")


def result_path(period: str, name: str) -> Path:
    return OUT / f"{period}_{name}.json"


def ensure_null(period: str, bars: list[Bar], sessions: int) -> dict:
    p = result_path(period, "null")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    res = run_null(bars, sessions)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def run_names(period: str, names: list[str]) -> None:
    bars, sessions = period_bars(period)
    print(f"{period}: {len(bars)} bars, {sessions} sessions", flush=True)
    ensure_null(period, bars, sessions)
    for name in names:
        p = result_path(period, name)
        if p.exists():
            print(f"  {name}: cached", flush=True)
            continue
        t0 = datetime.now()
        res = run_config(config_for(name), bars, sessions, name)
        res["change"] = ABLATIONS.get(name, COMBOS.get(name, {}))
        res["period"] = period
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1), encoding="utf-8")
        rn = [t["r_net"] for t in res["trades"]]
        print(f"  {name}: {res['signals']} signals, mean r_net {_mean(rn):+.3f}  "
              f"({(datetime.now() - t0).seconds}s)", flush=True)


# ── report ──────────────────────────────────────────────────────────────────


def _null_row(null: dict, trades: list[dict]) -> dict | None:
    stops = [t["stop_atr"] for t in trades if t.get("stop_atr")]
    tgts = [t["target_r"] for t in trades if t.get("target_r")]
    if not stops or not tgts:
        return None
    ms, mt = statistics.median(stops), statistics.median(tgts)
    return min(null["grid"].values(),
               key=lambda g: abs(math.log(g["stop_atr"] / ms)) + abs(math.log(g["target_r"] / mt)))


def _t_vs_null(rn: list[float], nr: dict | None) -> float:
    if nr is None or len(rn) < 2:
        return math.nan
    va, vb, nb = statistics.variance(rn), nr["sd_r"] ** 2, nr["n_per_seed"]
    denom = math.sqrt(va / len(rn) + vb / nb)
    return (_mean(rn) - nr["mean_r_net"]) / denom if denom else math.nan


def report(period: str) -> None:
    files = sorted(OUT.glob(f"{period}_*.json"))
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    null = next((r for r in runs if r["config"] == "null"), None)
    runs = [r for r in runs if r["config"] != "null"]
    if not runs:
        print(f"no {period} runs yet")
        return
    span = runs[0]
    years = (datetime.strptime(span["last"], "%Y-%m-%d") - datetime.strptime(span["first"], "%Y-%m-%d")).days / 365.25
    print("=" * 100)
    print(f"v30 research · {period.upper()} · {span['first']} to {span['last']} · "
          f"{span['sessions']} sessions · {len(runs)} configurations looked at on this period")
    print("=" * 100)
    print(f"  {'configuration':<20} {'sigs':>5} {'/yr':>5} {'win':>5} {'r_net':>7} {'se':>6} "
          f"{'t null':>7} {'drop1':>7} | {'longs':>5} {'meanR':>7} {'shorts':>6} {'meanR':>7} | {'tgt':>4} {'stop':>4} {'time':>4}")
    print("-" * 100)
    order = list(ABLATIONS) + list(COMBOS)
    runs.sort(key=lambda r: order.index(r["config"]) if r["config"] in order else 99)
    for r in runs:
        tr = r["trades"]
        rn = [t["r_net"] for t in tr]
        n = len(rn)
        nr = _null_row(null, tr) if null else None
        win = sum(1 for x in rn if x > 0) / n if n else math.nan
        drop = _mean(sorted(rn)[:-1]) if n > 2 else math.nan
        L = [t["r_net"] for t in tr if t["direction"] == "long"]
        S = [t["r_net"] for t in tr if t["direction"] == "short"]
        outs = {"target": 0, "stop": 0, "time": 0}
        for t in tr:
            outs[t["outcome"]] = outs.get(t["outcome"], 0) + 1
        print(f"  {r['config']:<20} {r['signals']:>5} {r['signals'] / years:>5.1f} "
              f"{100 * win if n else float('nan'):>4.0f}% {_fmt(_mean(rn), 7, 3, True)} {_fmt(_se(rn), 6, 3)} "
              f"{_fmt(_t_vs_null(rn, nr), 7, 2, True)} {_fmt(drop, 7, 3, True)} | "
              f"{len(L):>5} {_fmt(_mean(L), 7, 3, True)} {len(S):>6} {_fmt(_mean(S), 7, 3, True)} | "
              f"{outs['target']:>4} {outs['stop']:>4} {outs['time']:>4}")
    print("-" * 100)
    if null:
        print("  null (random entry, same session, 20 seeds), by structure stop-ATR/target-R: "
              + "  ".join(f"{k} {g['mean_r_net']:+.3f}" for k, g in null["grid"].items()))
    print("  t null = Welch t against the random-entry null at the run's own median stop/target structure.")
    print("  drop1 = mean r_net after removing the single best trade.")
    print(f"  {len(runs)} configurations on this period: the best of that many noise draws has an "
          f"expected max t of about {_expected_max_t(len(runs)):+.1f}.")


def _expected_max_t(k: int) -> float:
    # Expected maximum of k standard normals, closed-form approximation.
    if k <= 1:
        return 0.0
    return math.sqrt(2 * math.log(k)) - (math.log(math.log(k)) + math.log(4 * math.pi)) / (2 * math.sqrt(2 * math.log(k)))


def exit_sweep(period: str, name: str) -> None:
    """Fixed signal set, targets re-placed in R. Is the harvest the problem?"""
    bars, sessions = period_bars(period)
    index_of = {b.ts: i for i, b in enumerate(bars)}
    sigs = CisdEngine(config_for(name), symbol="SPY").run(bars)
    print(f"\n  exit sweep · {period} · {name} · {len(sigs)} signals")
    print(f"    {'target':>7} {'win':>5} {'meanR':>8} {'se':>7} {'stops':>6} {'time':>5}")
    for rt in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        outs = [o for o in (evaluate_at_r(s, bars, index_of, rt) for s in sigs) if o is not None]
        if not outs:
            continue
        rs = [o.r_multiple for o in outs]
        wins = sum(1 for o in outs if o.result == "target")
        print(f"    {rt:>6.2f}R {100 * wins / len(outs):>4.0f}% {_mean(rs):>+8.3f} {_fmt(_se(rs), 7, 3)} "
              f"{sum(1 for o in outs if o.result == 'stop'):>6} {sum(1 for o in outs if o.result == 'time'):>5}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=["is", "oos", "all"], required=True)
    ap.add_argument("--run", help="comma-separated configuration names, or 'all' for every single-change ablation")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--exit-sweep", metavar="NAME")
    args = ap.parse_args()

    if args.run:
        names = list(ABLATIONS) if args.run == "all" else [n.strip() for n in args.run.split(",")]
        run_names(args.period, names)
    if args.report:
        report(args.period)
    if args.exit_sweep:
        exit_sweep(args.period, args.exit_sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
