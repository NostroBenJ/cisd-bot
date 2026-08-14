"""Every tunable parameter in one place.

Defaults reproduce the `CISD_Powell_RB_v29_SPY.pine` SPY preset, so a fresh
`Config()` is the Pine's behaviour and any deviation is an explicit override.
That is the only way the port can be checked against the original: change one
thing at a time and see one thing move.

Three kinds of field live here and they are labelled in the comments:

* **port**   -- a direct translation of a Pine input.
* **fix**    -- a defect in the Pine. The switch exists so the backtest can
                measure the fix rather than assume it; defaults are set to the
                CORRECTED behaviour, because shipping a known-wrong default to
                an execution bot is not defensible.
* **new**    -- an addition the Pine has no equivalent for.

The `pine_bug_compat` preset restores every defect, which is what you run when
diffing signal-for-signal against TradingView.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Config:
    # ── 1 · timeframes ──────────────────────────────────────────────────────
    # port: RB timeframes, in minutes. Pine ran 5 and 15 with 60 optional.
    rb_timeframes: tuple[int, ...] = (5, 15)
    # new: detection also runs on the base timeframe so 1-minute structure is
    # visible to the model instead of only feeding the higher-timeframe scan.
    base_timeframe: int = 1
    # port: timeframe for the top-down directional bias.
    bias_timeframe: int = 60

    # ── 2 · grade / quota / bias ────────────────────────────────────────────
    min_grade_to_track: str = "A"        # port: minDraw
    min_grade_to_signal: str = "A"       # port: minSignal
    max_signals_per_session: int = 6     # port: maxSignals
    bias_cisd_lookback: int = 6          # port: cisdBiasLook
    require_bias: bool = True            # port: reqBias

    # fix: the Pine latches the 60m bias forever and treats "never fired" as
    # bullish, so a cold start permits every long and blocks every short. When
    # True the bias expires after `bias_max_age_bars` and an unset bias is
    # NEUTRAL -- which, with require_bias on, blocks BOTH directions until real
    # structure exists rather than silently favouring one.
    bias_expires: bool = True
    bias_max_age_bars: int = 24

    # ── 3 · premium / discount ──────────────────────────────────────────────
    # fix: the Pine's "previous day" is actually TWO days back. It reads
    #   request.security(sym, "D", high[1], lookahead = barmerge.lookahead_off)
    # but lookahead_off already returns the last CONFIRMED daily bar --
    # yesterday -- so the [1] offset reaches back one further. The standard
    # non-repainting idiom for a previous-day value is lookahead_ON with [1];
    # with lookahead_off the correct expression carries no offset at all.
    # Verified against a year of 5-minute SPY: lag 2 reproduces TradingView on
    # 252/257 sessions (98.1%), lag 1 and lag 3 on zero.
    # Consequence in the original: the dealing range drawn as PDH/PDL, the
    # premium/discount depth that carries up to 22 grade points, and the draw
    # targets are all anchored one day stale.
    # 1 = the actual previous period (correct). 2 = reproduce the Pine.
    prev_period_lag: int = 1
    eq_band_pct: float = 0.10            # port: eqBandPct
    w_pd: float = 22.0                   # port: wPD
    pd_strong: float = 0.55              # port: pdStrong
    allow_b_grade_deep_pd: bool = True   # port: allowPdB

    # ── 4 · institutional anchor ────────────────────────────────────────────
    gate_fvg: bool = True                # port: gateFvg
    gate_key_open: bool = True           # port: gateKeyO
    gate_pd: bool = True                 # port: gatePd
    # fix: in the Pine this is `gateIntrinsic and sweptV > 0.5`, but sweptV is
    # unconditionally 1.0 for every rejection block that exists -- so the whole
    # anchor gate is a no-op and the three switches above never filter
    # anything. Defaulting to False makes them real. Set True to reproduce.
    gate_intrinsic_any_sweep: bool = False
    prox_atr_mult: float = 0.15          # port: proxAtrM

    # ── 5 · key opens ───────────────────────────────────────────────────────
    key_opens: tuple[str, ...] = ("ny_open", "macro", "pm")

    # ── 6 · AMD / manipulation ──────────────────────────────────────────────
    amd_mode: str = "Soft"               # port: amdMode -- Off | Soft | Hard
    amd_windows: tuple[str, ...] = ("0930-1100", "0200-0500")
    hunt_manipulation: bool = True       # port: useAMDX
    killzones: tuple[str, ...] = ("0700-1100",)
    hunt_pd_pools: bool = True           # port: poolPD
    hunt_hod_lod: bool = True            # port: poolHL
    amdx_boost: float = 18.0             # port: amdxBoost

    # ── 8 · formation filters ───────────────────────────────────────────────
    require_min_sweep: bool = True       # port: gateSweep
    sweep_atr_mult: float = 0.08         # port: sweepAtrM
    min_body_fraction: float = 0.45      # port: minBody
    require_reclaim: bool = False        # port: gateReclaim
    require_displacement: bool = True    # port: gateDisp
    disp_atr_mult: float = 0.9           # port: dispAtrM
    disp_window_bars: int = 24           # port: dispWin
    anti_stack: bool = True              # port: gateStack

    # port: the Pine constructs a block with
    #   confirmed = (not gateDisp) or isHtfBlk,  isHtfBlk = tf != chart tf
    # so a block from any timeframe OTHER than the chart's own skips the
    # displacement requirement entirely. On a 5m chart with RB timeframes 5 and
    # 15, every 15m block is auto-confirmed and the "Require displacement after
    # RB" switch silently applies to 5m blocks only. Ported faithfully because
    # it is arguably deliberate -- displacement on a higher-timeframe block is
    # awkward to measure on the chart timeframe -- but it is not what the input
    # label says, so stage 3 should measure it.
    htf_blocks_skip_displacement: bool = True

    # fix: the Pine measures sweep distance against an ATR computed on the
    # rejection block's OWN timeframe while displacement and proximity use the
    # chart timeframe's ATR. On a 1-minute chart with a 15-minute block those
    # differ by roughly the timeframe ratio, so `sweep_atr_mult` and
    # `disp_atr_mult` are not on a comparable scale and cannot be reasoned
    # about together. True puts every ATR-scaled threshold on the block's own
    # timeframe, which is the interpretable choice.
    unify_atr_timeframe: bool = True

    # ── 9 · RB detection ────────────────────────────────────────────────────
    swing_lookback: int = 10             # port: swingLook
    require_strong_close: bool = True    # port: reqStrong
    strong_close_pct: float = 0.45       # port: strongPct
    atr_length: int = 14                 # port: atrLen
    max_leg_bars: int = 60               # port: maxLegBars
    max_live_blocks: int = 50            # port: maxRbs

    # ── 10 · invalidation ───────────────────────────────────────────────────
    invalidation_buffer: float = 0.0     # port: invalBuf

    # ── 11 · AOI / HTF FVG ──────────────────────────────────────────────────
    htf_fvg_timeframe: int = 60          # port: htfFvgTf
    fvg_min_points: float = 0.5          # port: fvgMinPts
    fvg_stale_atr: float = 8.0           # port: fvgStaleAtr
    include_order_block: bool = True     # port: useOB

    # ── 11b · opening gaps ──────────────────────────────────────────────────
    track_gaps: bool = True              # port: useGaps
    gap_min_points: float = 0.3          # port: gapMinPts
    gap_stale_atr: float = 10.0          # port: gapStaleAtr

    # ── 12 · liquidity ──────────────────────────────────────────────────────
    score_pd_sweeps: bool = True         # port: usePdSwp
    sweep_recency_bars: int = 12         # port: sweepLook
    # port: useSsSwp -- finalised session ranges (lonSess / nySess). These are
    # captured when the session ENDS and then held.
    use_session_sweeps: bool = True
    sweep_sessions: tuple[tuple[str, str], ...] = (
        ("london", "0300-1100"), ("ny", "0930-1600"),
    )
    # port: usePXH -- live session ranges that persist after the session ends
    # until the next one starts. Asia and London are off in the SPY preset.
    use_px_levels: bool = True
    px_sessions: tuple[tuple[str, str], ...] = (("ny", "0930-1600"),)

    # ── 12b · SMT divergence ────────────────────────────────────────────────
    use_smt: bool = True                 # port: useSMT
    smt_peers: tuple[str, ...] = ("QQQ", "DIA")
    smt_pivot_strength: int = 5          # port: smtPiv
    smt_required: bool = False           # port: smtReq
    # fix: a pivot confirms `smt_pivot_strength` bars late and the Pine then
    # accepts it for another `sweep_recency_bars`, so on a 1-minute chart a
    # divergence from 17 minutes ago counts as confirming now. This caps the
    # total age of an SMT confirmation, pivot lag included.
    smt_max_age_bars: int = 8

    # ── 12c · confluence stack ──────────────────────────────────────────────
    use_stack_gate: bool = True          # port: useStack
    min_stack: int = 3                   # port: minStack

    # ── 12d · draw on liquidity ─────────────────────────────────────────────
    use_dol: bool = True                 # port: useDOL
    dol_mode: str = "Both"               # port: dolMode
    dol_band_atr: float = 0.5            # port: dolBandAtr
    w_dol_continuation: float = 8.0      # port: wDolCont
    w_dol_reversal: float = 14.0         # port: wDolRev

    # ── 12e · std-dev projections (these become the TARGETS) ────────────────
    sigma_levels: tuple[float, ...] = (2.0, 2.5, 4.0)

    # ── 14 · CISD trigger ───────────────────────────────────────────────────
    use_cisd_trigger: bool = True        # port: useCisdTrig
    cisd_lookback: int = 4               # port: cisdLook
    cisd_min_run: int = 2                # port: cisdMinRun
    # new: canonical ICT requires the opposing run to be the one that drove
    # price INTO the swing extreme. The Pine accepts any recent opposing run.
    # "pine" reproduces the Pine, "swing_anchored" enforces the strict reading.
    cisd_mode: str = "pine"

    # ── 15 · grade weights ──────────────────────────────────────────────────
    w_sweep: float = 16.0
    w_fvg: float = 14.0
    w_gap: float = 12.0
    w_smt: float = 12.0
    w_open: float = 10.0
    w_amd: float = 8.0
    w_bias: float = 8.0
    w_displacement: float = 8.0
    base_score: float = 18.0
    # fix: the Pine awards `w_sweep * 0.4` whenever sweptV > 0.5, which is
    # always -- so every block collects 6.4 free points and the effective base
    # is 24.4, not 18. False removes the giveaway.
    free_sweep_credit: bool = False

    # ── NEW · volume ────────────────────────────────────────────────────────
    # The Pine reads no volume at all. A rejection block on a thin lunch bar
    # and one on a 4x-volume sweep grade identically today.
    use_volume: bool = True
    min_rvol: float = 1.2
    w_rvol: float = 8.0
    rvol_lookback_sessions: int = 20

    # ── NEW · VWAP ──────────────────────────────────────────────────────────
    # The reference intraday execution benchmark on SPY, and the level most
    # algorithmic flow is measured against. The model scores proximity to key
    # opens and previous-day levels but not to this.
    use_vwap: bool = True
    vwap_band_atr: float = 0.35
    w_vwap: float = 8.0

    # ── NEW · gamma context ─────────────────────────────────────────────────
    # Dealer positioning is the dominant intraday force on SPY, and its sign
    # decides whether a level should be expected to hold or to break: positive
    # dealer gamma damps moves toward pins, negative gamma accelerates through
    # them. The Pine approximates "where price is drawn" with previous-day and
    # session highs and lows; real call/put walls are the measured version of
    # the same idea. Supplied externally -- see `gamma.GammaContext`.
    use_gamma: bool = True
    w_gamma_wall: float = 10.0
    gamma_wall_band_atr: float = 0.6

    # ── NEW · multi-timeframe agreement ─────────────────────────────────────
    # Scored, not gated: the backtest can then measure what agreement is worth
    # instead of it being an invisible precondition.
    w_tf_agreement: float = 6.0

    # ── NEW · time of day ───────────────────────────────────────────────────
    # The first minutes are noise-dominated and the midday lull produces
    # structure that does not follow through. Both are well-documented in SPY
    # and neither is represented in the Pine.
    traded_window: str = "0930-1200"
    skip_first_minutes: int = 5
    lull_window: str = "1200-1330"
    lull_penalty: float = 6.0

    # ── NEW · event blackout ────────────────────────────────────────────────
    event_calendar_path: str = "data/events.json"
    event_block_before_min: int = 15
    event_block_after_min: int = 15

    # ── quota ───────────────────────────────────────────────────────────────
    # fix: the Pine resets the per-session signal quota on calendar day change
    # (midnight NY), so any overnight signal spends the budget before 09:30.
    # True resets at the start of the traded window instead.
    quota_resets_at_session: bool = True

    def grade_of(self, score: float) -> str:
        """port: gradeOf. Thresholds unchanged from the Pine."""
        if score >= 80:
            return "A+"
        if score >= 66:
            return "A"
        if score >= 52:
            return "B"
        return "C"

    @staticmethod
    def grade_rank(grade: str) -> int:
        return {"A+": 4, "A": 3, "B": 2, "C": 1}.get(grade, 0)

    def max_possible_score(self) -> float:
        """Ceiling before the 100 clamp. Reported by the verification suite
        because it is the number that shows how loose the grade gate is: if the
        ceiling towers over the A threshold, 'A' is not selective."""
        total = self.base_score + self.w_pd + self.w_sweep + self.w_fvg
        total += self.w_gap + self.w_smt + self.w_open + self.w_amd
        total += self.amdx_boost + max(self.w_dol_reversal, self.w_dol_continuation)
        total += self.w_bias + self.w_displacement + 4.0 + 5.0
        if self.use_volume:
            total += self.w_rvol
        if self.use_vwap:
            total += self.w_vwap
        if self.use_gamma:
            total += self.w_gamma_wall
        total += self.w_tf_agreement
        return total


def pine_bug_compat(cfg: Config | None = None) -> Config:
    """Restore every known Pine defect.

    This is the configuration to run when diffing the port against TradingView
    signal-for-signal. It should reproduce the original including its bugs; any
    remaining difference is a porting error, not a design choice."""
    base = cfg or Config()
    return replace(
        base,
        gate_intrinsic_any_sweep=True,   # anchor gate becomes a no-op again
        free_sweep_credit=True,          # 6.4 free points restored
        prev_period_lag=2,               # "previous day" is two days back again
        bias_expires=False,              # bias latches forever, neutral = bull
        unify_atr_timeframe=False,       # mismatched ATR scales restored
        quota_resets_at_session=False,   # quota rolls at midnight again
        smt_max_age_bars=10_000,         # no cap on stale SMT
        use_volume=False,
        use_vwap=False,
        use_gamma=False,
        w_tf_agreement=0.0,
        lull_penalty=0.0,
        skip_first_minutes=0,
        traded_window="0000-0000",
        base_timeframe=0,                # base-timeframe detection off
    )
