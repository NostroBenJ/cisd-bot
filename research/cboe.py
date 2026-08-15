"""Free CBOE volatility history — the deep dataset.

    python -m research.cboe fetch     # download / refresh everything
    python -m research.cboe status

Verified live on 2026-08-14. Eight series, all free, all updated daily, no
key and no subscription:

    VIX     9,251 days from 1990   30-day implied vol
    SKEW    9,206 days from 1990   tail pricing
    VVIX    5,083 days from 2006   vol of vol
    VIX1Y   4,929 days from 2007
    VIX6M   4,684 days from 2008
    VIX3M   4,252 days from 2009
    VIX9D   3,926 days from 2011
    VIX1D   1,067 days from 2022   one-day vol -- the 0DTE gauge

**Why this matters more than it looks.** Every study in this project has failed
for want of statistical power: 90 sessions of order flow, 259 sessions of price,
one year of candidates. This is **9,251 observations**, which is roughly a
hundred times the order-flow sample and improves the standard error by about
ten times. The trade is resolution for depth -- daily instead of minute -- and
for the one question that has real evidence behind it, daily is the right
resolution anyway.

**And it is the right KIND of data.** The variance risk premium persists because
it is compensation for bearing tail risk, not a pattern waiting to be
arbitraged out. That is why it survived when twelve price patterns did not.
These series measure exactly that premium and its term structure.

Stdlib only. Data is cached to `data/cboe/`; re-running refreshes in place.
"""

from __future__ import annotations

import csv
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "data" / "cboe"
BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research/1.0"

SERIES = {
    "VIX":   "VIX_History.csv",     # 30-day implied vol, 1990-
    "VIX1D": "VIX1D_History.csv",   # 1-day, 2022-  (the 0DTE gauge)
    "VIX9D": "VIX9D_History.csv",   # 9-day,  2011-
    "VIX3M": "VIX3M_History.csv",   # 3-month, 2009-
    "VIX6M": "VIX6M_History.csv",   # 6-month, 2008-
    "VIX1Y": "VIX1Y_History.csv",   # 1-year, 2007-
    "SKEW":  "SKEW_History.csv",    # tail pricing, 1990-
    "VVIX":  "VVIX_History.csv",    # vol of vol, 2006-
}


def fetch_one(name: str, filename: str) -> tuple[int, str, str]:
    """Download one series to `data/cboe/<name>.csv`. Returns (rows, first, last)."""
    req = urllib.request.Request(BASE + filename, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        body = r.read()
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / f"{name}.csv").write_bytes(body)
    lines = body.decode("utf-8", "replace").splitlines()
    if len(lines) < 2:
        return 0, "", ""
    return len(lines) - 1, lines[1].split(",")[0], lines[-1].split(",")[0]


def load(name: str) -> dict[str, dict[str, float]]:
    """Read a cached series as {iso_date: {field: value}}.

    CBOE dates are M/D/YYYY; they are normalised to ISO here so every series
    joins on the same key. Rows that fail to parse are skipped rather than
    coerced to zero -- a zero in a volatility series is not a missing value,
    it is a wrong one, and it would quietly poison any average built on it."""
    p = STORE / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} -- run `python -m research.cboe fetch` first")
    out: dict[str, dict[str, float]] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("DATE") or "").strip()
            if not raw:
                continue
            try:
                m, d, y = raw.split("/")
                key = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            except ValueError:
                continue
            vals: dict[str, float] = {}
            for k, v in row.items():
                if k == "DATE" or v in (None, ""):
                    continue
                try:
                    vals[k.strip().lower()] = float(v)
                except (TypeError, ValueError):
                    continue
            if vals:
                out[key] = vals
    return out


def close_series(name: str) -> dict[str, float]:
    """{iso_date: closing level}. Handles both the OHLC and single-column forms."""
    raw = load(name)
    out: dict[str, float] = {}
    for d, vals in raw.items():
        if "close" in vals:
            out[d] = vals["close"]
        elif len(vals) == 1:
            out[d] = next(iter(vals.values()))
    return out


def cmd_fetch() -> int:
    print(f"downloading to {STORE}\n")
    print(f"  {'series':<8} {'rows':>7}  {'first':<12} {'last':<12}")
    print("  " + "-" * 44)
    for name, filename in SERIES.items():
        try:
            n, first, last = fetch_one(name, filename)
            print(f"  {name:<8} {n:>7}  {first:<12} {last:<12}")
        except urllib.error.HTTPError as e:
            print(f"  {name:<8} HTTP {e.code}")
        except Exception as e:
            print(f"  {name:<8} {type(e).__name__}: {e}")
    return 0


def cmd_status() -> int:
    if not STORE.exists():
        print("nothing fetched yet -- run `python -m research.cboe fetch`")
        return 0
    print(f"  {'series':<8} {'days':>7}  {'first':<12} {'last':<12}")
    print("  " + "-" * 44)
    for name in SERIES:
        p = STORE / f"{name}.csv"
        if not p.exists():
            print(f"  {name:<8}  (missing)")
            continue
        s = close_series(name)
        ks = sorted(s)
        print(f"  {name:<8} {len(ks):>7}  {ks[0]:<12} {ks[-1]:<12}")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    return {"fetch": cmd_fetch, "status": cmd_status}.get(cmd, cmd_status)()


if __name__ == "__main__":
    sys.exit(main())
