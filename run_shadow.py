"""Shadow run: replay archived sessions through the real bot, decide nothing.

    python run_shadow.py                      # 60 most recent sessions
    python run_shadow.py --days 250
    python run_shadow.py --signal gap_fade

This is the entry point, and it is deliberately the ONLY one. There is no
`--live` flag to fat-finger. Going live means writing `LiveExecutor`, which
refuses to construct without a typed acknowledgement, and registering a
`ValidationRecord` that passes -- two separate acts by two separate pieces of
code, neither reachable from a command line argument.

What this proves today: the whole path runs, and every signal is blocked. The
blocking is the feature. Nothing in FINDINGS.md has earned a passing record, so
a bot that traded anything right now would be a bot whose gate does not work.
"""

from __future__ import annotations

import argparse
import inspect
from collections import Counter

from bot.engine import Bot, BotConfig
from bot.feed import ArchiveFeed
from bot.journal import Journal
from bot.validation import Registry
from research import signals as sig_mod

JOURNAL = "data/shadow_journal.jsonl"
REGISTRY = "data/validation.json"


def load_signal(name: str):
    fn = getattr(sig_mod, name, None)
    if fn is None or not inspect.isfunction(fn) or name.startswith("_"):
        avail = [n for n, f in vars(sig_mod).items()
                 if inspect.isfunction(f) and not n.startswith("_")]
        raise SystemExit(f"unknown signal {name!r}. available: {', '.join(sorted(avail))}")
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="first_bar_continuation")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--equity", type=float, default=1000.0)
    args = ap.parse_args()

    feed = ArchiveFeed()
    days = feed.sessions()[-args.days:]
    if not days:
        raise SystemExit("no archived sessions in data/uw/SPY_ohlc_1m/")

    try:
        registry = Registry.load(REGISTRY)
    except FileNotFoundError:
        registry = Registry()

    journal = Journal(JOURNAL)
    cfg = BotConfig(signal_name=args.signal, equity=args.equity)
    bot = Bot(cfg, load_signal(args.signal), registry, journal)

    print(f"signal   {args.signal}")
    print(f"sessions {len(days)}  ({days[0]} -> {days[-1]})")
    print(f"gate     {'ALLOWED' if bot.allowed else 'BLOCKED'} -- {bot.gate_reason}")
    print()

    total = 0.0
    for day in days:
        total += bot.run_session(feed, day)

    counts = Counter(r["reason"] for r in journal.read()
                     if r.get("signal") == args.signal)
    print("decisions")
    for reason, n in counts.most_common(10):
        print(f"  {n:>6}  {reason}")
    print()
    print(f"fills    {len(bot.exec.fills)}")
    print(f"P&L      {total:+.2f} ({bot.exec.mode})")


if __name__ == "__main__":
    main()
