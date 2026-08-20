"""Refresh daily equity history from Alpaca. Self-contained; no MCP, no agent.

    python -m research.fetch_equities            # incremental
    python -m research.fetch_equities --check    # verify only, write nothing

The scheduled rebalance cannot call an AI agent to fetch data, so the pipeline
needs its own feed. This is it.

**Uses the SIP feed, deliberately.** SIP is the consolidated tape and matches
the Robinhood history the backtest was built on to a **0.0bp median**. The free
IEX feed is partial-volume and differs by 1.6-3.0bp per day -- small, but a
ranking is a sort, and a few basis points reorders names near the decile
boundary. Silently swapping the data source under a tested strategy is the kind
of change that never throws an error and quietly invalidates the backtest.

**It verifies before it writes.** Every refresh cross-checks the overlapping
dates against what is already on disk and REFUSES to write if they disagree by
more than a whisker. A feed that changes character -- adjustment policy, tape,
vendor -- gets caught here rather than in six months of drifting returns.

History note: bars before ~2020-07 come from the original Robinhood pull and
are preserved. Alpaca supplies everything after. The overlap is what proves the
join is clean.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta

STORE = pathlib.Path("data/equities")
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
FEED = "sip"
MAX_MEDIAN_BP = 1.0      # refuse the write above this
MAX_P95_BP = 10.0
BATCH = 20


def load_env(path: str = ".env") -> None:
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def headers() -> dict[str, str]:
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not k or not s:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (and no .env)")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def fetch(symbols: list[str], start: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    h = headers()
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        token = None
        while True:
            u = (f"{DATA_URL}?symbols={','.join(chunk)}&timeframe=1Day"
                 f"&start={start}&limit=10000&adjustment=split&feed={FEED}")
            if token:
                u += f"&page_token={token}"
            with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=90) as r:
                d = json.loads(r.read())
            for sym, bars in d.get("bars", {}).items():
                out.setdefault(sym, []).extend(bars)
            token = d.get("next_page_token")
            if not token:
                break
    return out


def read_csv(p: pathlib.Path) -> dict[str, tuple]:
    if not p.exists():
        return {}
    return {r["date"]: (r["open"], r["high"], r["low"], r["close"], r["volume"])
            for r in csv.DictReader(p.open(encoding="utf-8"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--days", type=int, default=500,
                    help="how far back to refresh (12-2 needs ~400 calendar days)")
    args = ap.parse_args()

    load_env()
    from research.cross_momentum import EXCLUDE
    universe = sorted(p.stem for p in STORE.glob("*.csv")
                      if not p.stem.startswith("_") and p.stem not in EXCLUDE)
    if not universe:
        print(f"no CSVs in {STORE}")
        return 1
    start = (date.today() - timedelta(days=args.days)).isoformat()
    print(f"refreshing {len(universe)} symbols from {start}  (feed={FEED})")

    try:
        got = fetch(universe + ["SPY"], start)
    except urllib.error.HTTPError as e:
        print(f"fetch failed {e.code}: {e.read()[:200].decode('utf-8','replace')}")
        return 1

    print(f"received {len(got)} symbols\n")

    all_diffs, written, refused, missing = [], 0, [], []
    for sym in universe + ["SPY"]:
        bars = got.get(sym)
        if not bars:
            missing.append(sym)
            continue
        p = STORE / (f"_{sym}.csv" if sym == "SPY" else f"{sym}.csv")
        old = read_csv(p)
        new = {b["t"][:10]: (b["o"], b["h"], b["l"], b["c"], b.get("v", 0))
               for b in bars}

        common = set(old) & set(new)
        diffs = []
        for d in common:
            a, b = float(old[d][3]), float(new[d][3])
            if b > 0:
                diffs.append(1e4 * abs(a - b) / b)
        if diffs:
            med = statistics.median(diffs)
            p95 = sorted(diffs)[min(len(diffs) - 1, int(0.95 * len(diffs)))]
            all_diffs.extend(diffs)
            if med > MAX_MEDIAN_BP or p95 > MAX_P95_BP:
                refused.append((sym, med, p95, len(common)))
                continue

        if args.check:
            continue
        merged = dict(old)
        merged.update({d: tuple(str(x) for x in v) for d, v in new.items()})
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for d in sorted(merged):
                w.writerow((d,) + tuple(merged[d]))
        written += 1

    if all_diffs:
        med = statistics.median(all_diffs)
        p95 = sorted(all_diffs)[int(0.95 * len(all_diffs))]
        print(f"agreement with existing data over {len(all_diffs):,} overlapping days:")
        print(f"  median {med:.2f}bp   p95 {p95:.2f}bp   max {max(all_diffs):.1f}bp")

    if missing:
        print(f"\nNO DATA for {len(missing)}: {missing[:10]}")
    if refused:
        print(f"\nREFUSED TO WRITE {len(refused)} symbols -- the feed disagrees "
              f"with what is on disk:")
        for sym, med, p95, n in refused[:10]:
            print(f"  {sym:<6} median {med:.1f}bp  p95 {p95:.1f}bp  over {n} days")
        print("  Investigate before overriding. A data source that changed "
              "character\n  invalidates the backtest silently.")

    print(f"\n{'checked' if args.check else 'wrote'} "
          f"{len(all_diffs) and len(universe) - len(refused) - len(missing) or written}"
          f" symbols" + ("" if args.check else f", {written} written"))
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
