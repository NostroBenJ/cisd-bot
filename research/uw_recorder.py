"""Record Unusual Whales flow history before it ages out.

    python -m research.uw_recorder backfill      # everything still reachable
    python -m research.uw_recorder today         # one day, for a daily job
    python -m research.uw_recorder status        # what we hold

**This is time-critical in a way nothing else in this project is.** The key's
history window is a ROLLING ~4 months: on 2026-08-13 the oldest reachable date
was 2026-04-06, and anything before that returns 403 permanently. Roughly one
trading day falls off the back edge every day. Data not captured today is not
recoverable at any price on this tier.

So the rule here is **record everything, decide later**. Storage is free and
re-fetching is impossible. Endpoints are captured whether or not a current
hypothesis needs them, because the cost of a wrong guess is asymmetric: an
unused file wastes bytes, a missing day cannot be reconstructed.

Handles the three UW traps from the skill: numbers arrive as JSON strings (we
store raw and coerce at read time, so a coercion bug never corrupts the
archive), everything is wrapped in `{"data": ...}` which is sometimes an object
rather than a list, and a 403 means a tier or history limit rather than a bad
key. Requests are sequential -- the plan's concurrency cap is 3 and there is no
deadline here worth risking 429s for.

Stdlib only.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "uw"
ENV_CANDIDATES = [
    ROOT / ".env",
    Path((str(Path.home()) + "/dev/nyam-terminal/engine/.env")),
]

# Endpoint -> URL template. {t} is the ticker. Every one of these accepts a
# `date` parameter, which is what makes them backtestable at all -- the other
# nineteen flow endpoints are live-only and can never be recovered for a past
# session.
ENDPOINTS = {
    "net_prem_ticks":   "/api/stock/{t}/net-prem-ticks",
    "nope":             "/api/stock/{t}/nope",
    "greek_exposure":   "/api/stock/{t}/greek-exposure",
    "gex_strike":       "/api/stock/{t}/greek-exposure/strike",
    "spot_exposures":   "/api/stock/{t}/spot-exposures/strike",
    "gex_levels":       "/api/stock/{t}/gex-levels",
    "oi_change":        "/api/stock/{t}/oi-change",
    "interpolated_iv":  "/api/stock/{t}/interpolated-iv",
    "price_levels":     "/api/stock/{t}/option/stock-price-levels",
    "flow_per_strike":  "/api/stock/{t}/flow-per-strike-intraday",
    # Price from the SAME source as the flow, so timestamps align without a
    # cross-vendor join -- and unlike the TradingView export it carries volume.
    "ohlc_1m":          "/api/stock/{t}/ohlc/1m",
    "ohlc_5m":          "/api/stock/{t}/ohlc/5m",
}

# Endpoints needing more than `date`. The OHLC default page is small; a full
# session of 1-minute bars including extended hours needs room for ~1,440.
EXTRA_PARAMS = {
    "ohlc_1m": {"limit": 2500},
    "ohlc_5m": {"limit": 2500},
}
MARKET_ENDPOINTS = {
    "market_tide":      "/api/market/market-tide",
    "market_oi_change": "/api/market/oi-change",
}

TICKERS = ("SPY",)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) uw-recorder/1.0"


def api_key() -> str:
    """Resolve the key at call time, not import time.

    The adapter next door once read this at import and silently reported "not
    set" whenever it ran outside the server process."""
    for p in ENV_CANDIDATES:
        if p.exists():
            m = re.search(r"UW_API_KEY\s*=\s*(\S+)", p.read_text(encoding="utf-8"))
            if m:
                return m.group(1).strip().strip('"').strip("'")
    raise RuntimeError(f"UW_API_KEY not found in any of: {ENV_CANDIDATES}")


def fetch(path: str, key: str, **params) -> tuple[str, object]:
    """(status, payload). Status is 'ok', 'http403', 'http429' or an error name."""
    url = "https://api.unusualwhales.com" + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "User-Agent": UA,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        # `data` is sometimes an object rather than a list.
        payload = body.get("data", body) if isinstance(body, dict) else body
        return "ok", payload
    except urllib.error.HTTPError as e:
        # 403 here means a tier or history-depth limit, not a bad key.
        return f"http{e.code}", None
    except Exception as e:
        return type(e).__name__, None


def _rows(payload) -> int:
    if payload is None:
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 1 if payload else 0


def capture_day(day: str, key: str, force: bool = False) -> dict[str, int]:
    """Pull every endpoint for one date. Idempotent: existing files are kept."""
    out: dict[str, int] = {}
    jobs = [(f"{tk}_{name}", tmpl.format(t=tk))
            for tk in TICKERS for name, tmpl in ENDPOINTS.items()]
    jobs += [(name, tmpl) for name, tmpl in MARKET_ENDPOINTS.items()]

    for name, path in jobs:
        dest = STORE / name / f"{day}.json"
        if dest.exists() and not force:
            out[name] = -1          # already held
            continue
        extra = EXTRA_PARAMS.get(name.split("_", 1)[-1], {})
        status, payload = fetch(path, key, date=day, **extra)
        if status == "http429":
            time.sleep(5)
            status, payload = fetch(path, key, date=day, **extra)
        if status == "ok" and _rows(payload) > 0:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Store RAW. Coercion happens at read time so a parsing bug can
            # never corrupt an archive that cannot be re-fetched.
            dest.write_text(json.dumps(payload), encoding="utf-8")
            out[name] = _rows(payload)
        else:
            out[name] = 0
        time.sleep(0.12)
    return out


def trading_days(start: date, end: date) -> list[str]:
    """Weekdays in range. Holidays simply return empty and are skipped."""
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def find_earliest(key: str, probe: str = "/api/stock/SPY/net-prem-ticks") -> date:
    """Binary-search the back edge of the rolling window."""
    today = date.today()
    lo, hi = today - timedelta(days=400), today - timedelta(days=1)
    while hi.weekday() >= 5:
        hi -= timedelta(days=1)
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) // 2
        while mid.weekday() >= 5:
            mid += timedelta(days=1)
        if mid >= hi:
            break
        status, payload = fetch(probe, key, date=mid.isoformat())
        if status == "ok" and _rows(payload) > 0:
            hi = mid
        else:
            lo = mid
        time.sleep(0.15)
    return hi


def cmd_backfill() -> int:
    key = api_key()
    print("finding the back edge of the rolling window...")
    earliest = find_earliest(key)
    today = date.today()
    days = trading_days(earliest, today - timedelta(days=1))
    print(f"reachable from {earliest} -> {today}: {len(days)} weekdays")
    print(f"storing to {STORE}\n")

    got = held = empty = 0
    for i, day in enumerate(days, 1):
        res = capture_day(day, key)
        new = sum(1 for v in res.values() if v > 0)
        old = sum(1 for v in res.values() if v == -1)
        non = sum(1 for v in res.values() if v == 0)
        got += new
        held += old
        empty += non
        flag = "" if (new or old) else "   (holiday or no data)"
        print(f"  [{i:>3}/{len(days)}] {day}  new {new:>2}  held {old:>2}  empty {non:>2}{flag}")
    print(f"\nfetched {got} files, {held} already held, {empty} empty")
    return 0


def cmd_today() -> int:
    key = api_key()
    day = (date.today() - timedelta(days=1)).isoformat()
    d = date.fromisoformat(day)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    day = d.isoformat()
    res = capture_day(day, key)
    print(f"{day}: " + "  ".join(f"{k}={v}" for k, v in res.items()))
    return 0


def cmd_status() -> int:
    if not STORE.exists():
        print(f"nothing recorded yet ({STORE} does not exist)")
        return 0
    print(f"{'series':<26} {'days':>5}  {'first':<12} {'last':<12}")
    print("-" * 60)
    total = 0
    for sub in sorted(STORE.iterdir()):
        if not sub.is_dir():
            continue
        files = sorted(f.stem for f in sub.glob("*.json"))
        total += len(files)
        if files:
            print(f"{sub.name:<26} {len(files):>5}  {files[0]:<12} {files[-1]:<12}")
    print("-" * 60)
    print(f"{total} files total")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    return {"backfill": cmd_backfill, "today": cmd_today, "status": cmd_status}.get(
        cmd, cmd_status)()


if __name__ == "__main__":
    sys.exit(main())
