# Findings

A record of what was tested and what came back, so none of it gets re-tested
by accident. Every number here is reproducible from the scripts in this repo
plus a TradingView export.

## Summary

**No edge was found.** Not in the CISD model, not in twelve simple price
patterns, not in the one candidate that initially looked promising. The one
thing with real supporting evidence — the variance risk premium — is the one
thing the current account cannot trade.

---

## 1 · CISD° Powell rejection-block model — REJECTED

TradingView Deep Backtesting, 1-minute SPY, 185 trades, 2006–2026, bar
magnifier on, 1 tick slippage + $2 commission.

| | |
|---|---|
| mean | **−$21.94/trade**, se $6.20, **t = −3.54** |
| win rate | 21.1% |
| total | −$4,058 on $100k |

Removing the worst trade (2020-03-17 COVID long, −$819) makes it **more**
significant: t = −3.95. Decays monotonically by era, worst in the most recent.
5-minute agrees independently: 149 trades, PF 0.397.

**Seven defects found in the Pine along the way**, all documented in
`config.Config`. The largest: the Institutional Anchor gate is a no-op
(`rSwept` is unconditionally 1.0), and "previous day high/low" is actually
**two** days back (`lookahead_off` with a `[1]` offset — verified at 98.1%
across 257 sessions).

The model fired 8–11 times/year regardless of timeframe. Even a working
version could not have supported meaningful returns at that frequency.

## 2 · Base rates — the control that was missing

`python -m research.baserate`. Signal-free entries, one per session, 40 draws
averaged, ten stop/target structures.

- **Every structure is negative** after a 0.02 ATR round-trip cost.
- **Tight stops are ~4× worse**: a 0.5 ATR stop runs −0.10 to −0.13 R against
  −0.025 to −0.034 R for a 2.0 ATR stop. Pure exit geometry, no signal.
- **There is no directional drift in 09:30–12:00.** Long minus short is ±0.01 R
  at every structure. SPY's drift lives overnight, not in the morning session.

Measured again in Pine over 25 years, 5,128 trades: **−$2.03/trade, PF 0.944,
win rate 50.33%** — a coin flip, with the loss fully explained by the ~$4
round-trip friction. The control behaves like a control.

## 3 · Twelve simple candidates — nothing beat the null

`python -m research.batch`. Opening-range break/fade, gap fade/continue,
previous-day sweep-reclaim and break, momentum, mean reversion, range
expansion/fade, first-bar continuation, inside-bar break. Three exit
structures each.

**Nothing reached |t| ≥ 2.5 against the null.** Best was mean reversion at
+1.74. One year, ~250 trades each, SE ≈ 0.06 — enough to rule out edges above
~0.2R and nothing smaller.

## 4 · First-bar continuation — REJECTED on concentration

The only candidate worth a full look. 6,521 trades over 26.6 years.

| | first bar | null |
|---|---|---|
| per trade | +$0.96 | −$2.03 |
| profit factor | 1.026 | 0.944 |
| win rate | 52.91% | 50.33% |
| max drawdown | 5.09% | 18.26% |

Against zero: **t = +0.96**. Long only +1.55, short only −0.18.

**The concentration test is what kills it.** Drop the best 1% — 65 trades of
6,521 — and the mean flips from +0.0051% to −0.0102%, **t = −2.06**. The whole
result lives in sixty-five trades across twenty-six years. A real edge survives
losing its best percentile; this becomes significantly negative.

No era consistency: positive 2000–2009, negative 2010–2019, positive
2020–2026, no era reaching |t| = 1.2. Holdout is consistent only in being
insignificant on both sides (in +0.52, out +0.94).

Economically moot anyway: **+0.23%/year** against ~8% for buy-and-hold, and a
per-trade edge of 0.004% — a fraction of the spread, and nowhere near enough to
pay the 20%+ vol premium that buying options costs.

---

## What this rules out, and what it doesn't

**Ruled out:** simple price-pattern prediction of SPY direction in the morning
session. This is the most heavily arbitraged market in the world and the
efficient-market baseline is what we measured, rather than assumed.

**Not ruled out**, because never tested:

1. **Order flow and positioning.** Dark pool prints, net premium ticks, GEX,
   flow alerts — information that is not in the price series. The Unusual
   Whales subscription carries 199 endpoints and 8 are wired.
2. **Cross-asset.** SPY against the VIX term structure, bonds, sector
   dispersion.
3. **Scheduled events.** Macro releases with known timing.
4. **Variance risk premium.** See below.

## The one thing with real evidence

`vrp_study.py` already measured it: implied vol systematically exceeds
subsequent realized vol on SPY. That is not a pattern that gets arbitraged
away — it is **compensation for bearing tail risk**, which is why it persists.

It is also the opposite of what this account is set up to do. Level 2,
single-leg, long-only means *paying* that premium on every trade rather than
collecting it. Harvesting it needs spreads (level 3) or the capital for
cash-secured puts, plus a real answer to the February 2020 problem that the
VRP study's own tail analysis identified.

**The edge with the best evidence is the one currently untradeable. That is the
most useful thing this project established.**
