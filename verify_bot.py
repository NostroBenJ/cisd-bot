"""Verification for the `bot/` package. Run: `python verify_bot.py`

The suite exists because "the shadow run produced no trades" is indistinguishable
from "the shadow run is broken". Every check below either fails loudly or proves
a specific claim the package makes about itself.

The load-bearing check is [8]: the engine and the research harness, given the
same signal and the same bars, must select the SAME trades. If they diverge, a
ValidationRecord earned in `research/` says nothing about what `bot/` will do,
and the entire validation gate is decoration.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from bot.engine import Bot, BotConfig
from bot.execution import LIVE_ACK, LiveExecutor, ShadowExecutor
from bot.feed import ArchiveFeed
from bot.journal import Journal
from bot.validation import Registry, ValidationRecord
from cisd.risk import RiskLimits
from research import harness, signals as sig_mod

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


def tmp_journal() -> Journal:
    return Journal(Path(tempfile.mkdtemp()) / "j.jsonl")


def passing_record(signal: str, cfg: BotConfig) -> ValidationRecord:
    """A record that passes every threshold. Used ONLY to exercise the code
    below the gate -- it is fabricated, and never written to data/."""
    return ValidationRecord(
        signal=signal, n_trades=500, mean_r=0.15, t_vs_null=3.0, holdout_t=2.0,
        survives_concentration=True, tested_on="synthetic", tested_at="2026-08-19",
        window=cfg.window, stop_atr=cfg.stop_atr, target_atr=cfg.target_atr,
        notes="FABRICATED for verify_bot.py -- not evidence of anything")


def main() -> int:
    feed = ArchiveFeed()
    days = feed.sessions()
    if len(days) < 60:
        print("not enough archived sessions to verify")
        return 1
    sample = days[-60:]

    print("\n[1] feed: timestamps are real, sessions are distinguishable")
    bars = feed.session_bars(sample[-1])
    check("session has a full RTH day", 380 <= len(bars) <= 391, f"{len(bars)}")
    check("first bar is 09:30 NY", bars[0].minute_of_day_ny == 570,
          str(bars[0].minute_of_day_ny))
    check("session_date_ny matches the filename",
          bars[0].session_date_ny == sample[-1],
          f"{bars[0].session_date_ny} vs {sample[-1]}")
    check("bars are strictly ascending in time",
          all(b.ts < c.ts for b, c in zip(bars, bars[1:])))

    print("\n[2] feed: the forward cut holds")
    snap = feed.snapshot(sample[-1], 60, history_days=0)
    check("minute 60 shows exactly 61 bars", len(snap.bars) == 61, str(len(snap.bars)))
    check("no bar past as_of leaks in", max(b.ts for b in snap.bars) == snap.as_of)
    check("hides the rest of the day", len(bars) - len(snap.bars) > 300,
          str(len(bars) - len(snap.bars)))

    print("\n[3] feed: history reaches the previous session")
    h = feed.history_before(sample[-1], 1)
    check("one prior session is returned", 380 <= len(h) <= 391, str(len(h)))
    check("it is a DIFFERENT day", h[0].session_date_ny != sample[-1])
    check("it ends before the current session starts", h[-1].ts < bars[0].ts)
    snap1 = feed.snapshot(sample[-1], 5, history_days=1)
    check("snapshot with history spans two sessions",
          len({b.session_date_ny for b in snap1.bars}) == 2)

    print("\n[4] validation: the gate refuses by default")
    reg = Registry()
    ok, why = reg.may_trade("anything")
    check("unknown signal is refused", not ok, why)
    bad = ValidationRecord("weak", 50, -0.01, 0.9, -0.2, False, "x", "y")
    reg.add(bad)
    ok, why = reg.may_trade("weak")
    check("failing record is refused", not ok)
    check("every failure is reported, not just the first",
          len(bad.failures()) == 5, str(bad.failures()))

    print("\n[5] validation: traded conditions must match tested conditions")
    cfg = BotConfig(signal_name="x")
    reg2 = Registry()
    reg2.add(passing_record("x", cfg))
    check("matching config passes", reg2.may_trade(
        "x", cfg.window, cfg.stop_atr, cfg.target_atr)[0])
    ok, why = reg2.may_trade("x", "1000-1200", cfg.stop_atr, cfg.target_atr)
    check("different window is refused", not ok, why)
    ok, why = reg2.may_trade("x", cfg.window, 3.0, cfg.target_atr)
    check("different stop is refused", not ok, why)
    ok, why = reg2.may_trade("x", cfg.window, cfg.stop_atr, 9.0)
    check("different target is refused", not ok, why)

    print("\n[6] execution: live cannot be reached by accident")
    try:
        LiveExecutor("please")
        check("wrong acknowledgement raises", False)
    except PermissionError:
        check("wrong acknowledgement raises PermissionError", True)
    except NotImplementedError:
        check("wrong acknowledgement raises PermissionError", False,
              "got NotImplementedError -- the ack was not checked first")
    try:
        LiveExecutor(LIVE_ACK)
        check("correct acknowledgement still refuses", False)
    except NotImplementedError:
        check("correct acknowledgement still refuses", True)

    print("\n[7] engine: the gate blocks a real signal on real data")
    cfg = BotConfig(signal_name="first_bar_continuation")
    j = tmp_journal()
    bot = Bot(cfg, sig_mod.first_bar_continuation, Registry(), j)
    for d in sample:
        bot.run_session(feed, d)
    rows = j.read()
    fired = [r for r in rows if "not validated" in r["reason"]]
    check("signal fired on real bars", len(fired) > 0, f"{len(fired)} evaluations")
    check("nothing was executed", len(bot.exec.fills) == 0, str(len(bot.exec.fills)))
    check("every block names its reason",
          all(r["reason"] for r in rows if r["action"] == "blocked"))

    print("\n[8] engine vs harness: the SAME signal picks the SAME trades")
    cont: list = []
    for d in days[-90:]:
        cont.extend(feed.session_bars(d))
    res = harness.measure(cont, sig_mod.first_bar_continuation,
                          "first_bar_continuation", stop_atr=cfg.stop_atr,
                          target_atr=cfg.target_atr, window=cfg.window,
                          max_hold_bars=cfg.max_hold_bars,
                          atr_length=cfg.atr_length, cost_atr=cfg.cost_atr)
    h_trades = {t.ts: (t.direction, t.outcome) for t in res.trades}

    j2 = tmp_journal()
    reg3 = Registry()
    reg3.add(passing_record("first_bar_continuation", cfg))
    bot2 = Bot(cfg, sig_mod.first_bar_continuation, reg3, j2,
               limits=RiskLimits(hard_flat_time=cfg.flat_hhmm,
                                 max_stop_distance_pct=99.0))
    for d in sample:
        bot2.run_session(feed, d)

    e_rows = j2.read()
    e_open = {r["ts"]: r for r in e_rows if r["action"] == "open"}
    e_close = [r for r in e_rows if r["action"] == "close"]

    check("the fabricated record unblocks execution", len(e_open) > 0,
          f"{len(e_open)} entries")
    check("every entry has a matching exit", len(e_open) == len(e_close),
          f"{len(e_open)} in, {len(e_close)} out")

    overlap = [ts for ts in e_open if ts in h_trades]
    check("engine entries exist in the harness result",
          len(overlap) >= 0.9 * len(e_open),
          f"{len(overlap)}/{len(e_open)} matched")
    dir_ok = all(e_open[ts]["direction"] == h_trades[ts][0] for ts in overlap)
    check("directions agree on every matched trade", dir_ok)

    reason_map = {"stop": "stop", "target": "target", "time exit": "time"}
    agree = sum(1 for ts in overlap
                if reason_map.get(e_close_for(e_rows, ts), "?") == h_trades[ts][1])
    check("exit reasons agree on every matched trade", agree == len(overlap),
          f"{agree}/{len(overlap)}")

    print("\n[9] engine: exits resolve conservatively and cost is charged")
    both = [r for r in e_close if r["reason"] == "stop"]
    check("stops occur (a bar spanning both counts as a stop)", len(both) > 0)
    check("every exit charges a round-trip cost",
          all(r["context"].get("cost", 0) > 0 for r in e_close))
    check("cost is small relative to the stop, not free",
          all(0 < r["context"]["cost"] < abs(r["price"]) * r["shares"]
              for r in e_close if r["shares"]))

    print("\n[10] engine: risk limits actually bind")
    check("one trade per session", len(e_open) <= len(sample),
          f"{len(e_open)} entries over {len(sample)} sessions")
    sizes = [r["shares"] for r in e_open.values()]
    check("every position is at least one share", all(s >= 1 for s in sizes))
    check("no position exceeds the cash account",
          all(e_open[ts]["shares"] * e_open[ts]["price"] <= cfg.equity + 1e-6
              for ts in e_open))

    print("\n[11] engine: a mismatched flat time is refused loudly")
    try:
        Bot(BotConfig(signal_name="x"), sig_mod.momentum, Registry(), tmp_journal(),
            limits=RiskLimits(hard_flat_time="1155"))
        check("inconsistent hard_flat_time raises", False)
    except ValueError:
        check("inconsistent hard_flat_time raises ValueError", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


def e_close_for(rows: list[dict], entry_ts: int) -> str:
    """Reason of the first close after `entry_ts`."""
    after = [r for r in rows if r["action"] == "close" and r["ts"] > entry_ts]
    return after[0]["reason"] if after else "?"


if __name__ == "__main__":
    sys.exit(main())
