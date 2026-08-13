# cisd-bot

CISD° Powell rejection-block model, ported from Pine to Python so it can be
**measured**. Target: SPY single-leg options, executed through Robinhood's
agentic account.

```bash
python verify_cisd.py
```

128 checks, all passing.

## Status

| Stage | State |
|---|---|
| 1 · Python port of the indicator | **done** — this repo |
| 2 · Bar-for-bar validation against TradingView | **not started** — needs TV data export |
| 3 · Signal quality on SPY underlying (free data) | **not started** |
| 4 · Options P&L (OptionsDX intraday chains) | **not started** |
| 5 · Execution bot + dashboard | **not started** |

Nothing here has been checked against real market data yet. Every number below
comes from synthetic bars and proves only that the code does what it says.

## Why the port exists

`CISD_Powell_RB_v29_SPY.pine` is an `indicator()`, not a `strategy()`. It has
never produced a win rate, an expectancy, or any number that could be checked.
It has ~120 tunable inputs and no measurement, which is the shape of a tool for
encoding discretion, not a tested edge.

## Defects found in the Pine, and what was done

Each is a switch in `config.Config`, defaulting to the corrected behaviour.
`pine_bug_compat()` restores all of them for signal-for-signal diffing.

**1 · The Institutional Anchor gate is a no-op.** `f_rb` sets `rSwept := 1.0`
unconditionally in both branches, and a block cannot form without a sweep — so
`sweptV` is always 1.0, so `intrinsicPath` is always true, so `anchored` is
always true. The FVG, key-open and premium/discount gates never filter
anything. Check `[12]` demonstrates it: with every anchor off, the corrected
build admits **0** blocks and the Pine build still admits 23.
→ `gate_intrinsic_any_sweep` (default `False`)

**2 · Every block collects 6.4 free grade points.** Same root cause: the sweep
term degrades to `w_sweep * 0.4` whenever `sweptV > 0.5`, which is always. The
real base score is 24.4, not the 18 the code states.
→ `free_sweep_credit` (default `False`)

**3 · The top-down bias latches forever and defaults bullish.** `f_cisdState`
declares `var int dir = 0` and only ever assigns it — never resets — so the
bias is "the last 60m CISD that ever fired", possibly days stale. And
`biasBull = biasDir >= 0` makes zero read as bullish, so on a cold start every
long is permitted and every short blocked by a bias that was never established.
→ `bias_expires` (default `True`); unset bias is now genuinely neutral

**4 · Two different ATRs, silently.** `f_rb` computes ATR on the *block's*
timeframe inside `request.security`, while displacement and proximity use the
*chart* timeframe's ATR. On a 1-minute chart with a 15-minute block those
differ by roughly the timeframe ratio, so `sweepAtrM = 0.08` and
`dispAtrM = 0.9` are not on a comparable scale and cannot be tuned together.
→ `unify_atr_timeframe` (default `True`)

**5 · The signal quota resets at midnight, not at the open.** `dayofmonth`
rolls at 00:00 NY, so any overnight signal spends the day's budget before 09:30.
→ `quota_resets_at_session` (default `True`)

**6 · SMT confirmation is stale by construction.** A pivot cannot confirm until
`smtPiv` bars later, and the Pine then accepts it for another `sweepLook` bars
— a ~17-bar-wide window on a 1-minute chart.
→ `smt_max_age_bars` caps total age including pivot lag

**7 · No exit exists, and the alert is a human string.** The invalidation level
and the σ projections are computed internally and never emitted. `signal.py`
now emits entry, stop, targets, R multiples and the full reasoning as JSON.

**Grade is not selective.** Score ceiling is 169 in the Pine (201 with the
additions) against an "A" threshold of 66. A block reaches A from many
independent directions, and on synthetic textbook data *every* signal grades
A+. The thresholds need recalibrating against a real score distribution —
that is a Stage 3 output, not something to guess now.

## What was added, and why

| Addition | Reason |
|---|---|
| **Dealer gamma context** (`gamma.py`) | The dominant intraday mechanism on SPY, and its *sign* decides whether a level holds or breaks. The Pine approximates "where is price drawn" with PDH/PDL; this is the measured version. Consumes a snapshot from the existing verified GEX engine — it computes no options maths itself, so there is one implementation to trust, not a second to drift. |
| **Relative volume** | The Pine reads no volume at all. A block on a thin lunch bar and one on a 4x sweep grade identically. Normalised by minute-of-day, because SPY's 09:35 volume is several times its 12:15 volume every day. |
| **Session VWAP** | The reference intraday execution benchmark on SPY and the level most algorithmic flow is measured against. The model scores proximity to key opens and previous-day levels but not to this. |
| **Multi-timeframe agreement** | Scored, not gated, so the backtest can price what agreement is worth instead of it being an invisible precondition. |
| **Time-of-day handling** | Opening minutes are noise-dominated; the midday lull produces structure that does not follow through. Neither is represented in the Pine. |
| **Event blackout** | A sweep into a CPI print is a repricing, not engineered liquidity being taken. The Pine will grade an A+ thirty seconds before an FOMC statement. Data-driven, empty by default — an empty calendar blocks nothing and says so. |
| **Negative scoring** | Pine terms only ever add, so the ceiling towers over the threshold. A model that cannot mark a setup down cannot rank. |
| **Canonical CISD mode** | ICT anchors the change in state to the run that drove price *into the swing extreme*; the Pine accepts any recent opposing run. Both implemented, `cisd_mode` selects. |

## The three checks that matter

- **`[9]` prefix stability** — running on the first N bars and the first N+K bars
  must give identical signals over the shared prefix. This is the no-lookahead
  proof. Without it the backtest is fiction.
- **`[10]` mirror symmetry** — reflect every price and swap highs with lows;
  every long must become a short at the same bar with the same score. Catches
  the one-branch bug that hides longest.
- **`[12]` anchor-gate vacuity** — demonstrates defect 1 rather than asserting it.

Two bugs in **this port** were caught by the suite during development: the bias
tracker compared a 60-minute index against a 1-minute one (so bias read
permanently neutral, which looks exactly like a working gate), and `setup_type`
measured draw targets from the block midpoint where the Pine uses the current
close (which silently disqualified the textbook sweep-of-PDL reversal).

## Layout

```
cisd/
  bars.py         Bar container, session-anchored timeframe aggregation
  indicators.py   ATR/RMA, pivots, VWAP, relative volume — Pine-exact
  sessions.py     Session windows, key opens, event blackout
  detect.py       Rejection-block formation, CISD trigger
  context.py      Dealing range, liquidity pools, FVGs, gaps, draw on liquidity
  gamma.py        Dealer positioning context (addition)
  grade.py        Scoring and confluence counting
  signal.py       Entry/stop/target contract and reasoning
  engine.py       Lifecycle state machine, multi-timeframe orchestration
  config.py       Every parameter, with port/fix/new labelled
verify_cisd.py    128 checks
```

Stdlib only. No numpy, no pandas.

## Next

1. Export TradingView signals and diff against `pine_bug_compat()` — until that
   matches, the port is unverified against the thing it copies.
2. Real 1-minute SPY bars. **Robinhood's history is phantom beyond ~6 months**
   (constant `777.77`, volume 0, `interpolated: true`) — verified, and it would
   have produced a flawless meaningless equity curve. FirstRate gives 1 year
   free, 27 years for $99.95.
3. Measure: signal count, grade distribution, stop-vs-target hit rate, MFE/MAE,
   with the `sqrt(n_indep)` correction for overlapping signals.
