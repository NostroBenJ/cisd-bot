"""Zarattini, Aziz & Barbon (2025) intraday momentum on SPY — replication.

    python -m research.intraday_momentum              # full sample
    python -m research.intraday_momentum --to 2024-03-01   # their window only

Paper: "Beat the Market: An Effective Intraday Momentum Strategy for S&P500
ETF (SPY)". Claims 9.7%/yr at Sharpe 1.24 unlevered (2007 - early 2024), or
19.6%/yr at Sharpe 1.33 with dynamic sizing up to 4x.

**THE RULES, as specified in the paper.**

Noise Area, recomputed for every time-of-day HH:MM:

    sigma(HH:MM) = mean over the last 14 sessions of
                   |price_at(HH:MM) / open(9:30) - 1|
    Upper = max(open_today, close_yesterday) * (1 + sigma)
    Lower = min(open_today, close_yesterday) * (1 - sigma)

The gap adjustment is the max/min: after a gap down, the upper boundary is
pushed up by the gap, because the gap is itself evidence of imbalance.

Decisions ONLY at :00 and :30, entries and stops alike -- the paper's stated
defence against reacting to transient spikes. Long above Upper, short below
Lower. Trailing stop is `max(Upper, VWAP)` for longs and `min(Lower, VWAP)`
for shorts; a cross to the opposite boundary closes and reverses. Everything
is flat at the close.

**Timing convention, and why it matters.** Bars are labelled by left edge, so
the bar labelled 09:59 covers 09:59:00-10:00:00 and its close IS the price at
10:00. This module therefore reads `price_at(T) = close of the bar labelled
T - 1 minute`. Using the close of the bar labelled T would be reading one
minute into the future at every single decision -- small, systematic, and
exactly the kind of thing that turns a flat strategy into a profitable-looking
one.

**Costs** are the paper's: $0.0035/share commission plus $0.001/share slippage,
charged per side.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "data" / "uw" / "SPY_ohlc_1m"

OPEN_MIN = 9 * 60 + 30                       # 09:30
DECISIONS = [t for t in range(600, 931, 30)]  # 10:00 .. 15:30, half-hourly
LOOKBACK = 14
COST_PER_SHARE_PER_SIDE = 0.0035 + 0.001


def _f(v, d=math.nan) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


@dataclass
class Day:
    date: str
    open_px: float
    close_px: float
    price_at: dict[int, float]      # minute-of-day -> price AT that minute
    vwap_at: dict[int, float]       # minute-of-day -> session VWAP at that minute
    last_min: int


def load_days() -> list[Day]:
    """One Day per session, regular hours only, chronological."""
    out: list[Day] = []
    for p in sorted(STORE.glob("*.json")):
        rows = json.loads(p.read_text(encoding="utf-8"))
        reg = [r for r in rows if r.get("market_time") == "r"]
        if len(reg) < 300:
            continue
        # UW returns bars newest-first. Unsorted, everything below is garbage.
        reg.sort(key=lambda r: r["start_time"])

        price_at: dict[int, float] = {}
        vwap_at: dict[int, float] = {}
        pv = vol = 0.0
        first_open = _f(reg[0]["open"])
        for r in reg:
            hh, mm = r["start_time"][11:13], r["start_time"][14:16]
            # start_time is UTC; 13:30Z == 09:30 ET. Convert by offset from the
            # session open rather than by timezone maths -- the feed is already
            # filtered to regular hours, so the first bar IS 09:30 ET.
            c, h, l = _f(r["close"]), _f(r["high"]), _f(r["low"])
            v = _f(r.get("volume"), 0.0)
            if math.isnan(c):
                continue
            pv += ((h + l + c) / 3.0) * v
            vol += v
            idx = len(price_at)
            minute = OPEN_MIN + idx
            # The bar labelled `minute` closes at `minute + 1`, so its close is
            # the price AT minute+1.
            price_at[minute + 1] = c
            vwap_at[minute + 1] = (pv / vol) if vol > 0 else c
        if not price_at:
            continue
        last = max(price_at)
        out.append(Day(p.stem, first_open, price_at[last], price_at, vwap_at, last))
    return out


@dataclass
class Trade:
    date: str
    entered: int
    exited: int
    side: int
    entry: float
    exit: float
    shares: int
    pnl: float


@dataclass
class Result:
    trades: list[Trade] = field(default_factory=list)
    daily: list[tuple[str, float]] = field(default_factory=list)   # (date, return)
    equity: list[float] = field(default_factory=list)


def sigma_table(days: list[Day], i: int) -> dict[int, float]:
    """sigma for each decision time, from the previous LOOKBACK sessions."""
    out: dict[int, float] = {}
    window = days[max(0, i - LOOKBACK) : i]
    if len(window) < LOOKBACK:
        return out
    for t in DECISIONS + [None]:
        vals = []
        for d in window:
            key = t if t is not None else d.last_min
            px = d.price_at.get(key)
            if px is not None and d.open_px > 0:
                vals.append(abs(px / d.open_px - 1.0))
        if vals:
            out[t if t is not None else -1] = statistics.mean(vals)
    return out


def run(days: list[Day], dynamic: bool = False, capital: float = 100_000.0,
        target_vol: float = 0.02, max_lev: float = 4.0) -> Result:
    """Simulate the strategy day by day."""
    res = Result()
    aum = capital
    res.equity.append(aum)
    prev_close = None
    daily_rets: list[float] = []

    for i, d in enumerate(days):
        sig = sigma_table(days, i)
        if not sig or prev_close is None:
            prev_close = d.close_px
            daily_rets.append(0.0)
            continue

        anchor_hi = max(d.open_px, prev_close)
        anchor_lo = min(d.open_px, prev_close)

        # Position size fixed at the open, as the paper specifies.
        lev = 1.0
        if dynamic:
            if len(daily_rets) >= LOOKBACK:
                w = daily_rets[-LOOKBACK:]
                sd = statistics.stdev(w) if len(w) > 1 else 0.0
                lev = min(max_lev, target_vol / sd) if sd > 0 else max_lev
            else:
                lev = 1.0
        shares = int((aum * lev) // d.open_px) if d.open_px > 0 else 0

        side = 0
        entry_px = 0.0
        entry_t = 0
        day_pnl = 0.0

        def close_pos(px: float, t: int) -> None:
            nonlocal side, day_pnl
            if side == 0:
                return
            gross = (px - entry_px) * side * shares
            day_pnl += gross - COST_PER_SHARE_PER_SIDE * shares * 2
            res.trades.append(Trade(d.date, entry_t, t, side, entry_px, px, shares, gross))
            side = 0

        for t in DECISIONS:
            px = d.price_at.get(t)
            if px is None or t not in sig:
                continue
            up = anchor_hi * (1.0 + sig[t])
            lo = anchor_lo * (1.0 - sig[t])
            vw = d.vwap_at.get(t, px)

            if side == 1:
                stop = max(up, vw)
                if px < stop:
                    close_pos(px, t)
            elif side == -1:
                stop = min(lo, vw)
                if px > stop:
                    close_pos(px, t)

            if side == 0 and shares > 0:
                if px > up:
                    side, entry_px, entry_t = 1, px, t
                elif px < lo:
                    side, entry_px, entry_t = -1, px, t

        if side != 0:
            close_pos(d.close_px, d.last_min)

        ret = day_pnl / aum if aum > 0 else 0.0
        aum += day_pnl
        res.daily.append((d.date, ret))
        res.equity.append(aum)
        daily_rets.append(ret)
        prev_close = d.close_px

    return res


def stats(res: Result) -> dict:
    rets = [r for _, r in res.daily]
    if len(rets) < 20:
        return {}
    n = len(rets)
    total = res.equity[-1] / res.equity[0] - 1.0
    years = n / 252.0
    irr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else float("nan")
    vol = statistics.stdev(rets) * math.sqrt(252)
    sharpe = (statistics.mean(rets) * 252) / vol if vol > 0 else float("nan")
    peak, mdd = res.equity[0], 0.0
    for e in res.equity:
        peak = max(peak, e)
        mdd = max(mdd, 1.0 - e / peak)
    wins = sum(1 for r in rets if r > 0)
    return {"sessions": n, "total": total, "irr": irr, "vol": vol,
            "sharpe": sharpe, "mdd": mdd, "hit": wins / n,
            "trades": len(res.trades)}


def monthly(res: Result) -> dict[str, float]:
    """Compounded return per calendar month, in percent."""
    acc: dict[str, float] = {}
    for date, r in res.daily:
        k = date[:7]
        acc[k] = (1.0 + acc.get(k, 0.0)) * (1.0 + r) - 1.0
    return {k: v * 100 for k, v in acc.items()}


# Paper FAQ table, monthly % for the dynamic (levered) version.
PAPER_MONTHLY = {
    "2022-07": 0.2, "2022-08": 6.3, "2022-09": -1.0, "2022-10": 5.8,
    "2022-11": 1.6, "2022-12": 0.7,
    "2023-01": 2.9, "2023-02": -1.3, "2023-03": 7.8, "2023-04": 1.8,
    "2023-05": 2.9, "2023-06": 4.1, "2023-07": 2.2, "2023-08": 6.0,
    "2023-09": 2.9, "2023-10": -1.1, "2023-11": 0.5, "2023-12": 3.8,
    "2024-01": 8.8, "2024-02": -1.5, "2024-03": -0.4, "2024-04": 5.8,
    "2024-05": -4.3, "2024-06": 1.6, "2024-07": 8.2, "2024-08": -2.8,
    "2024-09": 4.1, "2024-10": 6.7, "2024-11": -2.6, "2024-12": 5.7,
    "2025-01": -1.2, "2025-02": 7.6, "2025-03": -0.2, "2025-04": 4.3,
    "2025-05": -2.1, "2025-06": -2.0, "2025-07": -2.0, "2025-08": -2.9,
    "2025-09": 1.0,
}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default=None, help="stop before this date (YYYY-MM-DD)")
    ap.add_argument("--from", dest="frm", default=None)
    args = ap.parse_args()

    days = load_days()
    if args.frm:
        days = [d for d in days if d.date >= args.frm]
    if args.to:
        days = [d for d in days if d.date < args.to]

    print("=" * 76)
    print("INTRADAY MOMENTUM (Zarattini/Aziz/Barbon) — replication")
    print("=" * 76)
    print(f"  {len(days)} sessions, {days[0].date} -> {days[-1].date}")
    print(f"  decisions at :00 and :30 from 10:00 to 15:30, flat at the close")
    print(f"  costs ${COST_PER_SHARE_PER_SIDE:.4f}/share/side (paper's figures)")

    for label, dyn in (("1x notional", False), ("dynamic sizing (<=4x)", True)):
        res = run(days, dynamic=dyn)
        s = stats(res)
        if not s:
            print(f"\n  {label}: too few sessions")
            continue
        print(f"\n  {label}")
        print(f"    total {s['total']*100:+8.1f}%   IRR {s['irr']*100:+6.2f}%   "
              f"vol {s['vol']*100:5.2f}%   Sharpe {s['sharpe']:+5.2f}")
        print(f"    max drawdown {s['mdd']*100:5.1f}%   hit {s['hit']*100:4.1f}%   "
              f"trades {s['trades']}")

        if dyn:
            ours = monthly(res)
            common = sorted(set(ours) & set(PAPER_MONTHLY))
            if common:
                print(f"\n    monthly vs the paper's published table ({len(common)} months)")
                print(f"      {'month':<9} {'ours':>7} {'paper':>7} {'diff':>7}")
                diffs = []
                for k in common:
                    d_ = ours[k] - PAPER_MONTHLY[k]
                    diffs.append(d_)
                    print(f"      {k:<9} {ours[k]:+7.1f} {PAPER_MONTHLY[k]:+7.1f} {d_:+7.1f}")
                corr_n = len(common)
                mo = statistics.mean([ours[k] for k in common])
                mp = statistics.mean([PAPER_MONTHLY[k] for k in common])
                num = sum((ours[k]-mo)*(PAPER_MONTHLY[k]-mp) for k in common)
                den = math.sqrt(sum((ours[k]-mo)**2 for k in common)
                                * sum((PAPER_MONTHLY[k]-mp)**2 for k in common))
                print(f"      mean ours {mo:+.2f}%  paper {mp:+.2f}%  "
                      f"correlation {num/den if den else float('nan'):+.3f}  n={corr_n}")
                print("      A faithful implementation should track the SIGN and rough")
                print("      SIZE month to month. Correlation well below ~0.7 means the")
                print("      implementation differs, and nothing downstream is meaningful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
