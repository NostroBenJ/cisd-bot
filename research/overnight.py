"""Overnight vs intraday decomposition of SPY returns.

    python -m research.overnight

The question: SPY's total return splits exactly into the part earned while the
market is CLOSED (previous close -> today's open) and the part earned while it
is OPEN (today's open -> today's close). In logs that split is an identity, not
an approximation, which makes it checkable -- see `[0]`.

The published claim, measured over decades on US equity indices, is that
essentially all of the return accrues overnight and the intraday session
contributes ~nothing. If that holds here it is directly tradeable by a cash
account with one share: buy the close, sell the open, one round trip per day.

Reasons to expect this to disappoint, stated before looking:

- **Four years is not decades.** 1,032 sessions is a short window for an effect
  usually measured since 1993.
- **Dividends bias it DOWN.** SPY drops by the dividend at the open on four
  ex-dates a year. Price-only data books that as an overnight loss, so the
  overnight figure here understates the true total return by roughly the
  1.1-1.3%/yr yield, concentrated in four days.
- **Costs eat small edges.** Two auction crossings per day. At 1bp round trip a
  strategy needs >2.5%/yr gross just to break even.

Every number is reported gross AND net, split-half, and with the best 1% of
days removed -- the test that inverted `first_bar_continuation`.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

STORE = Path("data/uw/SPY_ohlc_1m")
TRADING_DAYS = 252
DEFAULT_COST_BP = 1.0     # round trip, in basis points of notional


@dataclass
class Day:
    d: str
    rth_open: float
    rth_close: float
    pre_first: float        # first premarket print of the day
    post_last: float        # last postmarket print of the day

    @property
    def dt(self) -> date:
        return date.fromisoformat(self.d)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load_days(store: Path = STORE) -> list[Day]:
    """One record per session, from the raw archive.

    Uses the RTH open and close specifically -- the first `r` bar's open and
    the last `r` bar's close -- because those are the prices an auction fill
    actually gets. Extended-hours prints are captured separately and are NOT
    used for the primary test; they are thin and often stale."""
    out: list[Day] = []
    for p in sorted(store.glob("*.json")):
        rows = json.loads(p.read_text(encoding="utf-8"))
        rows.sort(key=lambda r: r["start_time"])
        rth = [r for r in rows if r.get("market_time") == "r"]
        pre = [r for r in rows if r.get("market_time") == "pr"]
        post = [r for r in rows if r.get("market_time") == "po"]
        if len(rth) < 300:
            continue
        o, c = _f(rth[0]["open"]), _f(rth[-1]["close"])
        if math.isnan(o) or math.isnan(c) or o <= 0 or c <= 0:
            continue
        out.append(Day(
            d=p.stem, rth_open=o, rth_close=c,
            pre_first=_f(pre[0]["open"]) if pre else math.nan,
            post_last=_f(post[-1]["close"]) if post else math.nan))
    return out


# ── statistics (stdlib; cross-checked in research/gs_crosscheck.py) ──────────


@dataclass
class Stat:
    name: str
    xs: list[float]

    @property
    def n(self) -> int:
        return len(self.xs)

    @property
    def mean(self) -> float:
        return statistics.mean(self.xs) if self.n else math.nan

    @property
    def se(self) -> float:
        if self.n < 2:
            return math.nan
        return statistics.stdev(self.xs) / math.sqrt(self.n)

    @property
    def t(self) -> float:
        return self.mean / self.se if self.se else math.nan

    @property
    def ann_pct(self) -> float:
        """Annualised, compounded from the mean LOG return."""
        return 100.0 * (math.exp(self.mean * TRADING_DAYS) - 1.0)

    @property
    def sharpe(self) -> float:
        if self.n < 2:
            return math.nan
        sd = statistics.stdev(self.xs)
        return (self.mean / sd) * math.sqrt(TRADING_DAYS) if sd else math.nan

    def row(self) -> str:
        return (f"  {self.name:<22} {self.n:>5} {1e4 * self.mean:>+8.2f}bp "
                f"{1e4 * self.se:>7.2f} {self.t:>+7.2f} {self.ann_pct:>+8.2f}% "
                f"{self.sharpe:>+6.2f}")


HEADER = (f"  {'leg':<22} {'n':>5} {'mean/day':>10} {'se':>7} {'t':>7} "
          f"{'ann':>9} {'SR':>6}")


def welch(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return math.nan
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    return (statistics.mean(a) - statistics.mean(b)) / se if se else math.nan


def drop_best(xs: list[float], frac: float = 0.01) -> list[float]:
    """Remove the best `frac` of observations. An edge that lives in a handful
    of days is not an edge you can size into."""
    k = max(1, int(round(frac * len(xs))))
    return sorted(xs)[:-k]


# ── the test ────────────────────────────────────────────────────────────────


def main() -> int:
    cost_bp = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_COST_BP
    days = load_days()
    if len(days) < 200:
        print(f"only {len(days)} usable sessions in {STORE}")
        return 1

    overnight: list[float] = []
    intraday: list[float] = []
    total: list[float] = []
    keep: list[Day] = []
    skipped_gap = 0

    for prev, cur in zip(days, days[1:]):
        gap = (cur.dt - prev.dt).days
        if gap > 5:
            # A hole in the archive, not a weekend. A "close to open" spanning
            # a two-week gap is not an overnight return.
            skipped_gap += 1
            continue
        on = math.log(cur.rth_open / prev.rth_close)
        idy = math.log(cur.rth_close / cur.rth_open)
        overnight.append(on)
        intraday.append(idy)
        total.append(math.log(cur.rth_close / prev.rth_close))
        keep.append(cur)

    print(f"\nSPY overnight vs intraday   {keep[0].d} -> {keep[-1].d}")
    print(f"{len(keep)} paired sessions"
          + (f", {skipped_gap} skipped for archive gaps > 5 days" if skipped_gap else ""))

    print("\n[0] structural identity: overnight + intraday == total (logs, exact)")
    worst = max(abs(o + i - t) for o, i, t in zip(overnight, intraday, total))
    ok = worst < 1e-12
    print(f"  {'ok  ' if ok else 'FAIL'} max deviation {worst:.3e} over {len(total)} days")
    if not ok:
        return 1

    print("\n[1] the decomposition, gross of costs")
    print(HEADER)
    s_on, s_id, s_tot = Stat("overnight", overnight), Stat("intraday", intraday), Stat("total", total)
    for s in (s_on, s_id, s_tot):
        print(s.row())
    print(f"\n  overnight - intraday: Welch t = {welch(overnight, intraday):+.2f}")
    print(f"  the two legs sum to the total by construction, so this is a")
    print(f"  decomposition of buy-and-hold, not two independent strategies.")

    print(f"\n[2] net of costs ({cost_bp:.2f}bp round trip, both legs traded daily)")
    c = cost_bp / 1e4
    n_on = Stat("overnight net", [x - c for x in overnight])
    n_id = Stat("intraday net", [x - c for x in intraday])
    n_bh = Stat("buy & hold", total)   # one round trip for the whole period
    print(HEADER)
    for s in (n_on, n_id, n_bh):
        print(s.row())
    edge = n_on.ann_pct - n_bh.ann_pct
    print(f"\n  overnight-only beats buy & hold by {edge:+.2f}%/yr after costs")
    print(f"  break-even cost for the overnight leg: "
          f"{1e4 * s_on.mean:.2f}bp per round trip")

    print("\n[3] concentration: drop the best 1% of days")
    for label, xs in (("overnight", overnight), ("intraday", intraday)):
        full, cut = Stat(label, xs), Stat(label, drop_best(xs))
        flipped = "  <-- FLIPS SIGN" if (full.mean > 0) != (cut.mean > 0) else ""
        print(f"  {label:<10} t {full.t:+6.2f} -> {cut.t:+6.2f}   "
              f"ann {full.ann_pct:+7.2f}% -> {cut.ann_pct:+7.2f}%"
              f"   ({len(xs) - len(cut.xs)} days removed){flipped}")

    print("\n[4] split-half stability")
    h = len(overnight) // 2
    for label, xs in (("overnight", overnight), ("intraday", intraday)):
        a, b = Stat(label, xs[:h]), Stat(label, xs[h:])
        same = "same sign" if (a.mean > 0) == (b.mean > 0) else "SIGN FLIP"
        print(f"  {label:<10} first {a.ann_pct:+7.2f}% (t {a.t:+5.2f})   "
              f"second {b.ann_pct:+7.2f}% (t {b.t:+5.2f})   {same}")
    print(f"  boundary: {keep[h].d}")

    print("\n[5] the overnight window, split by extended-hours prints")
    print("  Thin and often stale -- directional only, do not size on it.")
    post_leg, gap_leg, pre_leg = [], [], []
    for prev, cur in zip(keep, keep[1:]):
        if (cur.dt - prev.dt).days > 5:
            continue
        if any(math.isnan(v) or v <= 0 for v in
               (prev.post_last, cur.pre_first, prev.rth_close, cur.rth_open)):
            continue
        post_leg.append(math.log(prev.post_last / prev.rth_close))   # 16:00 -> ~20:00
        gap_leg.append(math.log(cur.pre_first / prev.post_last))     # ~20:00 -> ~04:00
        pre_leg.append(math.log(cur.rth_open / cur.pre_first))       # ~04:00 -> 09:30
    print(HEADER)
    for s in (Stat("post-close 16-20", post_leg), Stat("true gap 20-04", gap_leg),
              Stat("pre-open 04-0930", pre_leg)):
        print(s.row())

    print("\n[6] caveats that bound the conclusion")
    yrs = (keep[-1].dt - keep[0].dt).days / 365.25
    print(f"  sample is {yrs:.1f} years; the published effect is measured over decades")
    print(f"  dividends are NOT in this data -- SPY's ~1.2%/yr yield is booked")
    print(f"  as four overnight LOSSES a year, so overnight is understated by")
    print(f"  roughly that much and intraday is unaffected")
    print(f"  costs assumed {cost_bp:.2f}bp round trip; re-run with an argument to vary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
