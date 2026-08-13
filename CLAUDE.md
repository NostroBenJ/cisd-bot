# CLAUDE.md

CISD° rejection-block model → SPY options bot. Real money will be sized off
this, so a plausible-looking wrong number is worse than a missing one.

## Rules

**Stdlib-only in the model.** No numpy, no pandas, no scipy in `cisd/`. If a
closed form exists, write it out. Presentation and data-fetching may use third-
party libraries, but they import *inside the function that needs them* so the
model stays importable with zero dependencies.

**Nothing is trusted until it is measured.** The Pine original was an
`indicator()` with ~120 inputs and no backtest — that is the failure mode this
repo exists to correct. Do not add a parameter without a way to measure what it
does. Do not tune a parameter against a chart.

**Structural identities over assertions.** A property that must hold for any
correct implementation is worth more than a hundred hand-picked cases:

- prefix stability (no lookahead) — see `verify_cisd.py [9]`
- mirror symmetry (long/short parity) — `[10]`
- gate monotonicity (tightening cannot add signals) — `[11]`
- funnel ordering: formed ≥ confirmed ≥ armed ≥ triggered ≥ signalled

Add the equivalent for anything new. Both branches, always — half-tested code
where one direction is exercised twice and the other never is how a
long-only-correct model reaches production.

**Never fabricate a level.** A missing gamma snapshot, absent peer bars, or an
unloaded event calendar must degrade to "unavailable" and be recorded in the
signal's `warnings`, never to a default that looks like data. `DealingRange()`
with no data is not favourable and has zero depth; `GammaContext.load` returns
`None`, not a guess.

**Label probabilistic output honestly.** The Pine's "σ" levels are multiples of
the displacement leg, not standard deviations. The field is `leg_multiple`. If
a number is not what its name implies, rename it.

**Anti-repaint is a correctness property, not an optimisation.** Higher-
timeframe work happens only when an HTF bar has closed. No function reads an
index past the one it was given. Any change here must keep `[9]` passing.

## Conventions

- Bar timestamps are the **left edge**, epoch seconds, UTC. Shift at the loader
  if a vendor labels the right edge.
- Timeframe aggregation is **session-anchored** (09:30), because TradingView is.
  A 60-minute SPY bar runs 09:30–10:30.
- `T` in years where it appears; `r`, `q`, `sigma` are decimals, continuously
  compounded. Greeks broker-scaled at the boundary — see the `options-math`
  skill, and do not re-derive the dealer sign convention here.
- Config fields are labelled `port` / `fix` / `new` in comments. Defaults are
  the corrected behaviour; `pine_bug_compat()` restores every defect for
  diffing against TradingView.

## Data

- **Robinhood historicals are phantom beyond ~6 months** — constant `777.77`,
  volume 0, `interpolated: true`. Live/recent only. Never feed them to a
  backtest.
- DoltHub `post-no-preference/options` is EOD only, 2019–Jun 2024. It cannot
  price an intraday exit.
- Stage 3 needs 1-minute *bars*, not options data. Stage 4 needs OptionsDX
  intraday chains.

## Execution

The bot runs as the user's own process against the Robinhood **Agentic** cash
account (`option_level_2`, single-leg long calls/puts only — no spreads, and
the MCP does not support multi-leg regardless). Cash settlement is T+1, so
intraday recycling is capped without a limited-margin upgrade.

Hard limits are not optional: max daily loss, max concurrent positions, max
trades/day, hard flat time, and a kill file checked every loop.
