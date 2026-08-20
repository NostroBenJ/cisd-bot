# Findings

A record of what was tested and what came back, so none of it gets re-tested
by accident. Every number is reproducible from the scripts in this repo.

## Summary

Eight experiments. **Six negative, two with real measured effects — and both of
those run in directions this account cannot trade.**

| # | test | n | result |
|---|---|---:|---|
| 1 | CISD° Powell model | 185 trades | **t = −3.54** · rejected |
| 2 | base rates (the control) | 5,128 | no drift in 09:30–12:00; every exit structure negative |
| 3 | twelve price candidates | ~250 ea | nothing beat the null |
| 4 | first-bar continuation | 6,521 | **t = +0.96**, dies on concentration |
| 5 | order-flow archive | 90 days | built, not yet tested — underpowered |
| 6 | **variance risk premium** | **1,050** | **t = −4.26** · real, stable, unexploitable here |
| 7 | gamma regime (the vault rule) | 247 | direction NO (t = −1.22) · size YES (t = −6.57) |
| 8 | published intraday momentum | 594 OOS | Sharpe **1.83 → 0.08** · dead after publication |

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

## 8 · Published intraday momentum strategy — DEAD OUT OF SAMPLE

`python -m research.intraday_momentum`. Zarattini, Aziz & Barbon (2025),
*"Beat the Market: An Effective Intraday Momentum Strategy for SPY."* Claims
9.7%/yr at Sharpe 1.24 unlevered over 2007–early 2024, 19.6% with up to 4×
sizing.

**Replication first.** Their FAQ publishes monthly returns, so the check is
month by month rather than against a 17-year Sharpe:

| | in-sample overlap | out-of-sample |
|---|---|---|
| monthly correlation with their table | **+0.830** (20 mo) | **+0.899** (19 mo) |
| mean monthly, ours vs theirs | +2.44% vs +2.67% | +0.60% vs +1.29% |

Hit ratio reconciles too: 31.3% over all days looks wrong against their 43%,
but **37% of sessions never leave the Noise Area**; over traded days only it is
49.4%.

**Then the out-of-sample run, once, with no tweaking afterward:**

| period | n | IRR | Sharpe |
|---|---:|---:|---:|
| in-sample 2022-07→2024-02 | 396 | **+11.79%** | **+1.83** |
| out-of-sample 2024-03→2026-08 | 594 | **+0.32%** | **+0.08** |
| beyond even their published updates (2025-10+) | 201 | **−5.23%** | **−1.04** |

Year by year at 1×, the decay is monotone: Sharpe **2.11 → 1.78 → 0.42 → 0.41
→ −0.81** across 2022–2026.

**This is not an implementation failure.** Correlation with their own table is
*higher* out-of-sample than in. **Their published numbers show the same
collapse** — mean monthly +2.67% before, +1.29% after.

Concentration test on the out-of-sample days: all 594 average +0.20bp
(t = +0.13); **remove the best five days and it goes negative** at −1.68bp.
Whatever remains is a handful of sessions, not an edge.

The paper's first version is May 2024 and the strategy stops working in
early-to-mid 2024. Crowding after publication, regime change, or a
flattering in-sample window are all consistent and cannot be separated here.

**Prediction recorded before running: "meaningfully worse but not dead."**
Wrong, in the optimistic direction.

---

## 9 · Overnight vs intraday — the famous effect does not reproduce, and the tradeable version loses

`python -m research.overnight [cost_bp]` · 1,017 paired sessions, 2022-07-12 →
2026-08-12, built from the **extended-hours half of the archive we had been
discarding** (327,145 `pr`/`po` bars across 1,032 files).

The published claim on US equity indices is that essentially all of the return
accrues overnight while the intraday session contributes ~nothing. Directly
tradeable by a cash account at one share: buy the close, sell the open.

**Structural check first.** In logs the split is an identity, not an
approximation: `overnight + intraday == total`, max deviation **3.09e-16** over
1,017 days. And the total reconciles to reality — the model reports +18.89%/yr
against SPY's actual +18.63% CAGR over the same 4.09 years.

| leg | n | mean/day | t | annualised | Sharpe |
|---|---:|---:|---:|---:|---:|
| overnight | 1,017 | +4.32bp | +2.16 | +11.49% | +1.07 |
| intraday | 1,017 | +2.55bp | +0.97 | +6.64% | +0.48 |
| total | 1,017 | +6.87bp | +2.18 | +18.89% | +1.09 |

**The effect does not reproduce.** Overnight minus intraday is **Welch
t = +0.53**. Intraday is *positive* here, merely noisier — not the ~zero the
literature describes. The two legs cannot be distinguished in this sample.

**And the tradeable version loses to doing nothing.** Trading only the
overnight leg means forfeiting the intraday return and paying two auction
crossings a day:

| cost assumption | overnight net | vs buy & hold |
|---|---:|---:|
| 1.0bp round trip | +8.71%/yr | **−10.17%/yr** |
| 0.3bp (SPY 1¢ spread) | +10.65%/yr | **−8.24%/yr** |

Break-even cost for the overnight leg alone is 4.32bp; above that it is
negative outright. But break-even against *zero* is the wrong bar — the bar is
buy-and-hold, and it loses to that at any cost. Risk-adjusted the two are a
wash (SR 1.07 vs 1.09), so there is nothing to lever into either.

**Concentration kills what significance there is.** Dropping the best 1%:
overnight t **+2.16 → +0.96**; intraday **flips sign**, +6.64% → −1.80%. Ten
days out of 1,017 carry it — the same failure that inverted experiment 4.

Split-half is the one thing it passes: both legs keep their sign across the
2024-07-25 boundary, overnight strengthening (t +0.99 → +2.00).

**The one genuinely interesting residue.** Splitting the overnight window using
the extended-hours prints locates where it lives:

| sub-window | mean/day | t | Sharpe |
|---|---:|---:|---:|
| post-close 16:00–20:00 | +0.89bp | +1.06 | +0.53 |
| **true gap 20:00–04:00** | **+2.97bp** | **+2.41** | **+1.20** |
| pre-open 04:00–09:30 | +0.46bp | +0.33 | +0.16 |

Two thirds of the overnight return accrues in the eight hours when US retail
cannot trade at all. That is consistent with it being compensation for
overnight gap risk rather than an inefficiency — and it is emphatically **not**
tradeable: those prints are thin, often stale, and unreachable from a
Robinhood cash account.

**Bounds on the conclusion.** 4.1 years against a literature measured over
decades. Dividends are absent from this price-only data, so SPY's ~1.2%/yr
yield is booked as four overnight *losses* a year — the overnight figure is
understated by roughly that much, which would lift it to ~12.7%/yr and still
not separate it from intraday.

**Verdict: negative.** Not reproduced, not separable, not tradeable, and what
significance exists rests on ten days.

## 10 · Moving averages — nothing, across 36 pre-registered tests

`python -m research.ema_test` · 399,114 bars, 1,027 sessions, 2022-07 → 2026-08.

None of the original twelve candidates used a moving average. That was a gap in
coverage, not a judgement, so it got closed. Parameters were **pre-registered
as the conventional ones** -- 9/21, 8/21, 12/26, 20/50 crossovers and pullbacks,
plus 21/50 trend and slope filters -- and deliberately **not searched**. A search
over fast/slow pairs manufactures a winner: at 36 tests, ~2 clear p<0.05 by
chance alone.

**Nothing cleared |t| >= 2.5 against the null.** Best of 36 was
`ema_pullback_9_21` at **+1.95** (2.0/2.0 ATR). The Bonferroni threshold for 36
tests is **|t| >= 3.20**. Win rates sat between 47% and 53%; every mean R was
within a standard error of the null.

**One row earns its place as a teaching case:**

```
ema_cross_12_26   n 1012   meanR -0.085   t vs 0 -2.72   t vs null -1.35
```

Against *zero* that is a significant losing signal, and the obvious move is to
invert and trade it. Against the **null** it is noise -- the −2.72 is the exit
geometry losing money by itself, not the signal. Inverting it would have bought
nothing. This is precisely the error that measuring against zero produces, and
the reason the null exists.

**Verdict: closed.** Not to be re-tested with different lengths. Re-testing with
new parameters after seeing this table is the search this experiment was
designed to avoid.

## 11 · Cross-sectional momentum — the best result yet, and still short of the bar

`python -m research.cross_momentum [cost_bp]` · 100 US large caps, 92 month-ends,
2019-01 → 2026-08. Daily history pulled from Robinhood `get_equity_historicals`
(1,918 bars each, split-adjusted, no gaps, no zero-volume days).

The Jegadeesh-Titman strategy: rank on past return, hold the winners, rebalance
monthly. Long-only decile, because the account cannot short.

**The null is the equal-weighted universe, not zero** — holding any ten of these
hundred earns the market. 200 random 10-name draws per month, compared PAIRED,
because both portfolios carry the same beta and an unpaired test drowns the
difference in market noise.

| formation | n | annualised | SR | vs null | paired t vs null | paired t vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| 6-1 | 85 | +18.39% | 0.83 | +5.48% | +0.98 | +0.79 |
| 12-1 | 79 | +22.00% | 0.96 | +9.01% | +1.45 | +1.40 |
| **12-2** | **79** | **+26.75%** | **1.11** | **+13.77%** | **+2.13** | **+2.12** |

Benchmarks: equal-weight universe +14.27%/yr (SR 0.82), SPY +14.15% (SR 0.86).

**This is the strongest signal the project has found.** It also **does not pass**:

- t = **+2.13** against a **2.5** bar, and Bonferroni for 3 tests needs **2.39**.
- n = 79 months. `MIN_TRADES` is 200.
- It *does* survive concentration (2.13 → 1.86 dropping the best month) and
  split-half (first 1.32, second 1.66, same sign) — the two tests that killed
  earlier candidates.
- The ordering 12-2 > 12-1 > 6-1 is monotone and matches the literature, which
  is weak corroboration but not evidence.

The validation gate correctly refuses it. That is the architecture working.

**Known upward bias, unremovable here.** The universe is 100 end-2018 large caps
chosen from memory, and **all 100 still trade** — ~100% survivorship. A
point-in-time universe would contain acquired and collapsed names. Levels are
overstated; only the paired difference is worth reading, and only relatively.

**The binding practical constraint.** One share of each of today's top ten costs
**$3,589.71** — over 3× the intended account. Expressing a ten-name decile at
$1,000 requires fractional shares, or the portfolio collapses to three or four
names and loses the diversification that makes decile momentum work at all.

**Verdict: suggestive, not validated.** The first candidate worth re-testing
rather than abandoning — on a real point-in-time universe, with more history.

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
