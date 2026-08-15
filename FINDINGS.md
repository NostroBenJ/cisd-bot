# Findings

A record of what was tested and what came back, so none of it gets re-tested
by accident. Every number is reproducible from the scripts in this repo.

## Summary

Six experiments. **Five negative, one strongly positive — and the positive one
runs in the direction this account cannot trade.**

| # | test | n | result |
|---|---|---:|---|
| 1 | CISD° Powell model | 185 trades | **t = −3.54** · rejected |
| 2 | base rates (the control) | 5,128 | no drift in 09:30–12:00; every exit structure negative |
| 3 | twelve price candidates | ~250 ea | nothing beat the null |
| 4 | first-bar continuation | 6,521 | **t = +0.96**, dies on concentration |
| 5 | order-flow archive | 90 days | built, not yet tested — underpowered |
| 6 | **variance risk premium** | **1,050** | **t = −4.26** · real, stable, unexploitable here |

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

**Seven defects found in the Pine**, all documented in `config.Config`. The two
largest: the Institutional Anchor gate is a no-op (`rSwept` is unconditionally
1.0, so three of its four filters never fire), and "previous day high/low" is
actually **two** days back (`lookahead_off` with a `[1]` offset — verified at
98.1% across 257 sessions).

The model fired 8–11 times/year regardless of timeframe. Even a working version
could not have mattered at that frequency.

## 2 · Base rates — the control that was missing

`python -m research.baserate`, and `random (NULL)` in `RESEARCH_SIGNALS.pine`.

- **Every exit structure is negative** before any signal exists.
- **Tight stops are ~4× worse**: a 0.5 ATR stop runs −0.10 to −0.13 R against
  −0.025 to −0.034 R for 2.0 ATR. Pure geometry.
- **There is no directional drift in 09:30–12:00.** Long minus short is ±0.01 R
  at every structure. SPY's drift lives overnight.

Pine, 25 years, 5,128 trades: **−$2.03/trade, PF 0.944, win 50.33%** — a coin
flip, with the loss fully explained by ~$4 round-trip friction. The control
behaves like a control.

## 3 · Twelve simple candidates — nothing beat the null

`python -m research.batch`. Opening-range break/fade, gap fade/continue,
previous-day sweep-reclaim and break, momentum, mean reversion, range
expansion/fade, first-bar continuation, inside-bar break, × three structures.

**Nothing reached |t| ≥ 2.5 against the null.** Best was mean reversion at
+1.74. One year, ~250 trades each, SE ≈ 0.06 — rules out edges above ~0.2R and
nothing smaller.

## 4 · First-bar continuation — REJECTED on concentration

6,521 trades over 26.6 years. PF 1.026 against the null's 0.944, drawdown
5.09% against 18.26%. It looked like the first win.

| | mean | t |
|---|---:|---:|
| all trades | +0.0051% | **+0.96** |
| **excluding best 1% (65 trades)** | −0.0102% | **−2.06** |
| excluding worst 1% | +0.0203% | +4.12 |

**Dropping 65 of 6,521 trades flips it from positive to significantly
negative.** Twenty-six years of expectancy lived in sixty-five trades. A real
edge survives losing its best percentile; this inverts.

No era consistency (positive 2000–09, **negative 2010–19**, positive 2020–26,
no era reaching |t| = 1.2). Economically moot anyway: **+0.23%/year** against
~8% for buy-and-hold.

## 5 · Order-flow archive — built, not yet tested

`python -m research.uw_recorder`. See `data/UW_SCHEMA.md`.

**34 of 53 UW flow endpoints accept a `date` parameter**, so they are
backtestable. But history is a **rolling ~4-month window** — binary-searched to
between 2026-04-03 and 2026-04-06; earlier returns 403. Roughly one trading day
falls off the back edge per day, permanently.

Archived: **1,086 files, 90 sessions, 195MB**, plus 1-minute OHLC with volume.
Verified real — `market_time == "r"` gives exactly 35,100 bars = 390 × 90.

`net_prem_ticks` is richer than expected: per minute it carries premium and
volume split by **ask side vs bid side** (aggressive buyers vs sellers) and
**`net_delta`**, the net delta transacted.

**Not yet tested, because 90 sessions is underpowered** — the same position that
made experiments 1–4 inconclusive. The archive compounds: 250 sessions in a
year, 500 in two. The $375/mo tier would have given 2 years immediately; it was
declined on cost.

## 6 · Variance risk premium — REAL, and unexploitable at level 2

`python -m research.vol_premium`. **The first decisive result in the project.**

Buy a 1-day ATM SPY straddle at the close, hold to expiry. Cost is
`0.7979·σ·√(1/252)` of spot plus a 2% round trip; payoff is the realised move.
**1,050 observations — and a one-day horizon has ZERO overlapping windows, so
n_indep = n.** No `sqrt(n_indep)` correction needed. Cleanest sample here.

| | mean | t |
|---|---:|---:|
| every day | **−8.79 bp** | **−4.26** |
| excluding best 1% | −11.96 bp | −6.72 |
| excluding worst 1% | −6.95 bp | −3.49 |
| first half | −8.37 bp | −2.98 |
| second half | −9.21 bp | −3.04 |

Win rate 36.2%. **Stable across both halves and both tails** — it gets *more*
negative without its best trades, the opposite of experiment 4. This is what a
real effect looks like.

**No conditional escape.** Six predictors (VIX1D level, VIX1D/VIX, VIX9D/VIX,
VIX/VIX3M, VVIX, SKEW) × ten deciles = sixty cells. **Not one significantly
positive.** At p<0.05 about three should have cleared t = 2 by chance; zero did.

**A stated hypothesis, rejected.** I predicted the ~17% of days where VIX1D
exceeds VIX would be when buying premium is cheap. Tested: **−3.13 bp,
t = −0.36** — still negative, indistinguishable from the other 83%.

**The other side:** selling that straddle earns **+8.79 bp/day, ≈ +24.8%/yr of
spot gross** — and the seller's **worst 1% of days cost 36% of the edge**. That
concentration is not a flaw in the trade, it *is* the trade. The premium exists
as compensation for exactly that tail.

---

## What is settled

**Simple price-pattern prediction of SPY intraday direction does not work.**
Measured, not assumed, on the most competitively traded instrument in the world.

**A level-2, single-leg, long-only options account cannot make money on SPY
volatility.** Not by buying always, not by buying selectively. Closed with
evidence.

**The one durable edge measured here runs the wrong way for this account.** The
VRP persists because it is compensation for bearing tail risk, not a pattern to
be arbitraged. Harvesting it needs level 3 (defined-risk credit spreads) or
capital for cash-secured puts — plus a real answer to the tail.

## Method notes worth keeping

1. **Always measure the null first.** An asymmetric stop/target has a non-zero
   base rate with no signal at all. Scoring against zero attributes exit
   geometry to the signal.
2. **Always drop the best 1%.** Experiment 4 passed every other check and died
   here. A real edge survives losing its best percentile.
3. **Count the comparisons.** Twelve candidates × three structures is 36 tests;
   ~2 will look significant by chance. Sixty decile cells, ~3.
4. **A one-day forward horizon has no overlap.** Where the question allows it,
   this beats a longer horizon on statistical cleanliness even with fewer rows.
5. **Verify the data is real.** Robinhood's history beyond ~6 months is constant
   `777.77`, volume 0, `interpolated: true`. UW's OHLC is reverse-chronological
   while its flow is chronological — joining without sorting reads the future.

## Infrastructure

`cisd/` — the Pine port, 167 verification checks (`verify_cisd.py`)
`research/` — signal harness, base rates, batch runner, UW recorder, CBOE
loader, VRP study; 34 checks (`verify_research.py`)
`tools/` — TradingView import, Pine strategy builds, the research harness in Pine
`analyze_trades.py` — Strategy Tester export → expectancy with standard errors

**Robinhood has no official public REST API** for equities or options. Only the
Crypto Trading API and the **agentic MCP** (already connected). The unofficial
reverse-engineered endpoints are unsupported and break without notice.
