# UW archive — data dictionary

What `research/uw_recorder.py` stores, and the traps in reading it back.
Derived from the live key on 2026-08-13, not from the spec.

**Coverage:** 90 trading days, 2026-04-06 → 2026-08-12. Rolling ~4-month
window; anything earlier is permanently 403 on this tier.

## Integrity check (2026-08-13)

- 72,665 one-minute bars across 90 days.
- `market_time` splits to `r` 35,100 / `pr` 21,867 / `po` 15,698. The regular
  count is **exactly 390 × 90** — every session complete, none padded.
- Zero-volume bars: 0.03%. Zero-range bars: 14.0%, concentrated in extended
  hours. This is a real tape, not the constant-price gap-fill that Robinhood's
  history turned out to be.
- Median 1-minute bar range $0.145.

## Traps

1. **Every number is a JSON string.** `'-114920.00'`, `'43874'`. Coerce on
   read. The archive stores raw so a coercion bug can never corrupt it.
2. **OHLC is REVERSE-chronological; net-prem-ticks is chronological.** On
   2026-08-12 `ohlc_1m[0].start_time` is 19:59Z and `[-1]` is 13:30Z, while
   `net_prem_ticks[0].tape_time` is 13:30 and `[-1]` is 20:15. Joining them
   without sorting produces a series that reads the future. **Sort both by
   time before any join.**
3. **Timestamps differ in form.** OHLC uses `start_time`/`end_time` as UTC with
   a `Z`; net-prem-ticks uses `tape_time`, naive, also UTC. 13:30Z = 09:30 ET.
4. **Windows differ.** Flow runs 13:30–20:15Z (406 rows, through 16:15 ET);
   regular-hours OHLC runs 13:30–19:59Z (390 rows). Intersect on 390.
5. **Net premium is PER-TICK, not cumulative.** Verified: the last tick is
   +$392 against a median absolute tick of $300,300. Cumulate it yourself.

## Series

### `SPY_net_prem_ticks` — 406 rows/day, 1-minute
The main event. Richer than the name suggests:

| field | meaning |
|---|---|
| `tape_time` | minute, UTC, naive |
| `net_call_premium` / `net_put_premium` | premium that minute, signed |
| `net_call_volume` / `net_put_volume` | contracts, signed |
| `call_volume` / `put_volume` | gross contracts |
| `call_volume_ask_side` / `call_volume_bid_side` | **aggressive buyers vs sellers** |
| `put_volume_ask_side` / `put_volume_bid_side` | same, puts |
| `net_delta` | **net delta transacted** — the most direct directional-flow measure |

The ask/bid split is the classic order-flow construction: ask-side volume is
someone lifting the offer (aggressive), bid-side is hitting the bid. `net_delta`
is the single most promising column and was not on my list before reading it.

### `SPY_ohlc_1m` / `SPY_ohlc_5m`
`open,high,low,close,volume,total_volume,start_time,end_time,market_time`.
Filter `market_time == "r"` for regular hours. Carries volume, which the
TradingView export did not.

### Others
- `SPY_gex_levels` — one row: `call_wall`, `put_wall`, `gamma_flip`, `gamma_magnet`
- `SPY_greek_exposure`, `SPY_gex_strike`, `SPY_spot_exposures` — dealer positioning
- `SPY_oi_change` — 50 rows/day, per-contract OI with `curr_oi` / `prev_*`
- `SPY_interpolated_iv` — 9 rows: IV and `implied_move_perc` at 1…365 days
- `SPY_price_levels`, `SPY_flow_per_strike`
- `market_tide`, `market_oi_change` — market-wide

## Sample-size reality

90 sessions. The intraday resolution does **not** buy independent observations:
adjacent minutes are heavily autocorrelated, so the effective count is nearer 90
than 36,000. Apply the `sqrt(n_indep)` correction from `vrp_study.py` to any
statistic computed here.
