"""Does dealer gamma regime actually decide whether fading works?

    python -m research.gamma_regime

**The claim being tested** is Ben's own rule, written in the vault as
`lesson_gamma_regime_dictates_fade_vs_chase` and never checked against data:

    positive gamma -> dealers hedge AGAINST the move, moves are damped,
        breakouts fail, FADING is correct
    negative gamma -> dealers hedge WITH the move, moves accelerate,
        FADING is a loaded gun

**PREDICTION, stated before the numbers:**

    higher net gamma -> variance ratio LOWER  (more mean-reverting)
    lower  net gamma -> variance ratio HIGHER (more trending)

The test is the **difference across regimes**, not the level in any one of
them. SPY intraday may mean-revert in all conditions; VR < 1 on high-gamma days
would then say nothing.

**A caveat on sign conventions.** UW's aggregation may or may not use the same
dealer sign convention as `analysis/gex.py`. So the prediction above is stated
in terms of the *observable* -- higher versus lower `call_gamma + put_gamma` --
and if the effect appears with the opposite sign that is reported as an
INVERTED result, not quietly reinterpreted as confirmation. Deciding which
direction counts as a win after seeing the data is how any dataset confirms any
hypothesis.

**Why a variance ratio and not a trading rule.** A rule yields one observation
per session. The variance ratio is a *within-day* statistic estimated from ~390
returns, so each day gives one reasonably precise number and the daily numbers
are close to independent. More power, and it measures the mechanism directly
rather than through the noise of an exit rule.

    VR(q) = Var(q-bar return) / (q * Var(1-bar return))
    VR > 1 trending · VR = 1 random walk · VR < 1 mean-reverting

**Three traps in this data, all handled:**

1. UW returns OHLC **reverse-chronologically**. Unsorted, every return flips
   sign and mean reversion reads as momentum.
2. Today's gamma is not known at today's open, so the regime uses the **prior
   session's** value. `gex_levels` is also unusable for this: UW's gamma flip
   sits systematically above spot (median -2.7%, above on only 5 of 87 days),
   which is the known definitional gap in the notes. Net gamma from
   `greek-exposure` splits 33/67 instead and is the real measure.
3. `greek-exposure` returns a trailing YEAR whenever called inside the rolling
   window, so the sample is ~248 sessions rather than the 90 the flow archive
   holds.

Stdlib only.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "data" / "uw"
GREEK_SERIES = STORE / "SPY_greek_series"


@dataclass
class Session:
    date: str
    open_px: float
    close_px: float
    prev_gamma: float       # prior session's net gamma -- known at today's open
    rets: list[float]       # 1-minute log returns, regular hours, chronological

    @property
    def day_return(self) -> float:
        return (self.close_px - self.open_px) / self.open_px


def _f(v, default=math.nan) -> float:
    """UW sends every number as a string. Coerce, never guess."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_net_gamma() -> dict[str, float]:
    """{date: call_gamma + put_gamma} from the trailing-year greek series."""
    out: dict[str, float] = {}
    for p in GREEK_SERIES.glob("*.json"):
        for r in json.loads(p.read_text(encoding="utf-8")):
            g = _f(r.get("call_gamma")) + _f(r.get("put_gamma"))
            if not math.isnan(g):
                out[r["date"]] = g
    return out


def load_sessions() -> list[Session]:
    """One Session per day holding both price and a PRIOR net-gamma reading."""
    gamma = load_net_gamma()
    gdays = sorted(gamma)
    out: list[Session] = []

    for p in sorted((STORE / "SPY_ohlc_1m").glob("*.json")):
        day = p.stem
        # Strictly-earlier reading -- not "yesterday's date", the previous day
        # actually held, so a holiday cannot let the value reach forward.
        prior = [d for d in gdays if d < day]
        if not prior:
            continue

        rows = json.loads(p.read_text(encoding="utf-8"))
        reg = [r for r in rows if r.get("market_time") == "r"]
        # SORT. The feed is reverse-chronological.
        reg.sort(key=lambda r: r["start_time"])
        closes = [c for c in (_f(r["close"]) for r in reg) if not math.isnan(c) and c > 0]
        if len(closes) < 100:
            continue

        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        out.append(Session(day, _f(reg[0]["open"]), closes[-1], gamma[prior[-1]], rets))
    return out


def variance_ratio(rets: list[float], q: int) -> float:
    """VR(q) from NON-overlapping q-bar sums.

    Non-overlapping keeps it honest: overlapping windows share data, shrinking
    the apparent variance and biasing VR toward 1 by an amount that depends
    on q."""
    n = len(rets) // q
    if n < 8:
        return math.nan
    ones = rets[: n * q]
    var1 = statistics.variance(ones)
    if var1 <= 0:
        return math.nan
    blocks = [sum(ones[i * q : (i + 1) * q]) for i in range(n)]
    return statistics.variance(blocks) / (q * var1)


def autocorr1(rets: list[float]) -> float:
    n = len(rets)
    if n < 30:
        return math.nan
    m = statistics.mean(rets)
    num = sum((rets[i] - m) * (rets[i - 1] - m) for i in range(1, n))
    den = sum((r - m) ** 2 for r in rets)
    return num / den if den > 0 else math.nan


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    a = [x for x in a if not math.isnan(x)]
    b = [x for x in b if not math.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return math.nan, math.nan
    ma, mb = statistics.mean(a), statistics.mean(b)
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    return ma - mb, ((ma - mb) / se if se else math.nan)


def summarise(xs: list[float], null: float | None = None) -> str:
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return f"n={len(xs)}  --"
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs))
    s = f"n={len(xs):<4} {m:+.4f}  se {se:.4f}"
    if null is not None and se:
        s += f"   t vs {null:g} = {(m - null) / se:+5.2f}"
    return s


def report(name: str, S: list[Session], stat, null: float | None) -> None:
    """Binary split by sign, then a tercile gradient.

    The gradient matters more than the split. A monotone progression across
    terciles is hard to produce by chance; a single binary difference is not."""
    hi = [stat(s) for s in S if s.prev_gamma > 0]
    lo = [stat(s) for s in S if s.prev_gamma <= 0]
    d, t = welch(hi, lo)
    print(f"\n  {name}")
    print(f"    net gamma > 0    {summarise(hi, null)}")
    print(f"    net gamma <= 0   {summarise(lo, null)}")
    print(f"    difference       {d:+.4f}   Welch t = {t:+5.2f}   <-- predicted NEGATIVE")

    ranked = sorted(S, key=lambda s: s.prev_gamma)
    k = len(ranked) // 3
    print(f"    tercile gradient (low gamma -> high gamma):")
    for i, (lab, chunk) in enumerate(
        (("low ", ranked[:k]), ("mid ", ranked[k : 2 * k]), ("high", ranked[2 * k :]))
    ):
        vals = [x for x in (stat(s) for s in chunk) if not math.isnan(x)]
        if len(vals) < 2:
            continue
        m = statistics.mean(vals)
        se = statistics.stdev(vals) / math.sqrt(len(vals))
        print(f"      {lab}  n={len(vals):<4} {m:+.4f}  se {se:.4f}")


def _corr(a: list[float], b: list[float]) -> float:
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else math.nan


def control_for_vol_clustering(S: list[Session]) -> None:
    """Does gamma predict volatility BEYOND what yesterday's volatility says?

    This is the control that decides whether the volatility result is a finding
    or a restatement of a known effect. Volatility clusters -- a quiet day
    follows a quiet day -- and net gamma is heavily correlated with recent
    volatility, so the raw difference could be entirely second-hand.

    The test holds prior volatility roughly fixed by banding on it, then asks
    whether gamma still separates inside each band. Bands use disjoint days, so
    the three results are independent and can be combined."""
    vol = {s.date: statistics.stdev(s.rets) * math.sqrt(390) * 100
           for s in S if len(s.rets) > 30}
    days = sorted(vol)
    prev_vol = {days[i]: vol[days[i - 1]] for i in range(1, len(days))}
    rows = [(s.prev_gamma, prev_vol[s.date], vol[s.date]) for s in S if s.date in prev_vol]
    if len(rows) < 60:
        print("\n  control: too few sessions")
        return

    g, pv, tv = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    print(f"\n  CONTROL -- is this just volatility clustering?  ({len(rows)} sessions)")
    print(f"    corr(prior gamma, today vol) = {_corr(g, tv):+.3f}")
    print(f"    corr(prior VOL,   today vol) = {_corr(pv, tv):+.3f}   <- clustering")
    print(f"    corr(prior gamma, prior vol) = {_corr(g, pv):+.3f}   <- confounding")

    order = sorted(rows, key=lambda r: r[1])
    k = len(order) // 3
    zs = []
    print("    within bands of similar PRIOR vol, does gamma still separate?")
    for lab, chunk in (("low prior vol ", order[:k]),
                       ("mid prior vol ", order[k : 2 * k]),
                       ("high prior vol", order[2 * k :])):
        med = statistics.median([c[0] for c in chunk])
        hi = [c[2] for c in chunk if c[0] > med]
        lo = [c[2] for c in chunk if c[0] <= med]
        d, t = welch(hi, lo)
        zs.append(t)
        print(f"      {lab}  n={len(chunk):<4} diff {d:+.3f}  t = {t:+5.2f}")
    good = [z for z in zs if not math.isnan(z)]
    if good:
        # Stouffer: the bands are disjoint sets of days, so combining is fair.
        print(f"      combined (Stouffer) z = {sum(good) / math.sqrt(len(good)):+.2f}")


def test_tradeability(S: list[Session]) -> None:
    """If gamma predicts volatility, does buying a straddle on high-vol days pay?

    This is the only expression a level-2, long-only account has for a
    volatility view. The prior is that it does NOT work: dealers know their own
    gamma, so implied volatility should already reflect it, and you would pay
    more for the straddle on exactly the days it pays more."""
    try:
        from research.vol_premium import build, stats
    except Exception as e:
        print(f"\n  tradeability: skipped ({e})")
        return
    straddle = {d.date: d.pnl for d in build("VIX1D")}
    gam = {s.date: s.prev_gamma for s in S}
    rows = [(gam[d], straddle[d]) for d in sorted(set(gam) & set(straddle))]
    if len(rows) < 60:
        print("\n  tradeability: too few overlapping days")
        return

    m, se, t = stats([p for _, p in rows])
    print(f"\n  TRADEABILITY -- buy a 1-day straddle; does gamma pick the days?")
    print(f"    all days       n={len(rows):<4} {m * 10000:+7.2f}bp  t = {t:+5.2f}   (the VRP)")
    rows.sort(key=lambda r: r[0])
    k = len(rows) // 3
    for lab, chunk in (("low  gamma", rows[:k]),
                       ("mid  gamma", rows[k : 2 * k]),
                       ("high gamma", rows[2 * k :])):
        xs = [p for _, p in chunk]
        m, se, t = stats(xs)
        print(f"    {lab}     n={len(xs):<4} {m * 10000:+7.2f}bp  se {se * 10000:5.2f}  t = {t:+5.2f}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    S = load_sessions()
    if not S:
        print("no sessions -- run the OHLC backfill first")
        return 1

    pos = sum(1 for s in S if s.prev_gamma > 0)
    print("=" * 74)
    print("DOES GAMMA REGIME DECIDE WHETHER FADING WORKS?")
    print("=" * 74)
    print(f"  {len(S)} sessions, {S[0].date} -> {S[-1].date}")
    print(f"  regime = PRIOR session's net gamma (call_gamma + put_gamma)")
    print(f"  positive {pos}  ·  negative {len(S) - pos}")
    print("\n  PREDICTED, before the numbers:")
    print("    higher net gamma -> LOWER variance ratio (more mean-reverting)")
    print("    so every 'difference' row below should be NEGATIVE")

    report("variance ratio VR(5)   (1.0 = random walk)", S,
           lambda s: variance_ratio(s.rets, 5), 1.0)
    report("variance ratio VR(15)", S,
           lambda s: variance_ratio(s.rets, 15), 1.0)
    report("lag-1 autocorrelation of 1-minute returns", S,
           lambda s: autocorr1(s.rets), 0.0)
    report("realised intraday vol (% of spot)", S,
           lambda s: statistics.stdev(s.rets) * math.sqrt(390) * 100 if len(s.rets) > 30 else math.nan,
           None)

    control_for_vol_clustering(S)
    test_tradeability(S)

    print("\n" + "=" * 74)
    print("  The hypothesis lives on the DIFFERENCE rows and the tercile")
    print("  gradient. A difference with no gradient behind it is a coin flip")
    print("  that landed once. An inverted sign is an inverted result, not a")
    print("  confirmation under a different convention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
