"""Scoring and confluence counting.

Port of Pine `f_grade` and `f_stackCount`, plus the new terms. Every component
records *why* it fired into a `reasons` dict, because the bot has to be able to
explain a trade after the fact and "score 74" explains nothing. That dict is
what ends up in the journal and on the dashboard.

Two deliberate departures from the Pine, both flagged in `config.Config`:

* The free 6.4-point sweep credit is gone by default. In the Pine every block
  collects it unconditionally, because the variable it is gated on is always
  true, so the real base score is 24.4 rather than the 18 the code appears to
  say.
* Terms can now be NEGATIVE. The Pine's score only ever accumulates, so its
  ceiling towers over the "A" threshold and grade stops discriminating. A model
  that cannot mark a setup down cannot rank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Config
from .context import DealingRange, FvgStore, GapStore, LiquidityPools
from .gamma import GammaContext


@dataclass
class GradeInputs:
    """Everything the grader reads, assembled by the engine."""

    bull: bool
    ce: float
    ext: float
    timeframe: int
    strong_displacement: bool
    large_range: bool
    rvol: float
    near_key_open: bool
    in_amd_window: bool
    amdx_setup: bool
    setup_type: int
    bias_bull: bool | None
    smt_confirmed: bool
    tf_agreement: int
    in_lull: bool
    atr_value: float
    # False when the feed carries no volume at all. Distinct from "volume was
    # low": an absent column must not be scored as thin participation, which
    # would silently reject every signal and read as "the model found nothing".
    volume_available: bool = True
    vwap: float = math.nan
    gamma: GammaContext | None = None


@dataclass
class GradeResult:
    score: float
    reasons: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, name: str, points: float, note: str | None = None) -> None:
        if points != 0.0:
            self.reasons[name] = round(points, 2)
            if note:
                self.notes.append(note)


def grade(
    inp: GradeInputs,
    cfg: Config,
    dr: DealingRange,
    pools: LiquidityPools,
    fvgs: FvgStore,
    gaps: GapStore,
) -> GradeResult:
    """Score a rejection block, 0-100.

    The base is a floor, not a reward -- it exists so a block with no confluence
    at all still ranks above one that failed formation entirely."""
    res = GradeResult(score=cfg.base_score)
    res.reasons["base"] = cfg.base_score

    # ── premium / discount ──────────────────────────────────────────────────
    depth = dr.depth(inp.ce, inp.bull)
    if depth > 0:
        pts = depth * cfg.w_pd
        res.score += pts
        res.add("premium_discount", pts, f"{'discount' if inp.bull else 'premium'} depth {depth:.2f}")

    # ── engineered liquidity sweep ──────────────────────────────────────────
    real_sweep = pools.recently_swept(inp.bull)
    if real_sweep:
        res.score += cfg.w_sweep
        res.add("liquidity_sweep", cfg.w_sweep, "swept engineered liquidity")
    elif cfg.free_sweep_credit:
        # Pine-compatible giveaway: awarded to every block that exists.
        pts = cfg.w_sweep * 0.4
        res.score += pts
        res.add("sweep_credit_pine", pts, "pine-compat unconditional sweep credit")

    # ── higher-timeframe fair value gap ─────────────────────────────────────
    if fvgs.contains(inp.ce, inp.bull):
        res.score += cfg.w_fvg
        res.add("htf_fvg", cfg.w_fvg, "midpoint inside a higher-timeframe fair value gap")

    # ── opening gap ─────────────────────────────────────────────────────────
    if gaps.contains(inp.ce):
        res.score += cfg.w_gap
        res.add("opening_gap", cfg.w_gap, "midpoint inside an opening gap")

    # ── SMT divergence ──────────────────────────────────────────────────────
    if inp.smt_confirmed:
        res.score += cfg.w_smt
        res.add("smt", cfg.w_smt, "correlated-asset divergence confirms")

    # ── key open proximity ──────────────────────────────────────────────────
    if inp.near_key_open:
        res.score += cfg.w_open
        res.add("key_open", cfg.w_open, "formed at a key session open")

    # ── AMD window / manipulation ───────────────────────────────────────────
    if inp.in_amd_window:
        res.score += cfg.w_amd
        res.add("amd_window", cfg.w_amd, "inside an accumulation-manipulation window")
    if inp.amdx_setup:
        res.score += cfg.amdx_boost
        res.add("manipulation", cfg.amdx_boost, "session manipulation detected")

    # ── draw on liquidity ───────────────────────────────────────────────────
    if inp.setup_type == 2:
        res.score += cfg.w_dol_reversal
        res.add("dol_reversal", cfg.w_dol_reversal, "terminal: reached its draw")
    elif inp.setup_type == 1:
        res.score += cfg.w_dol_continuation
        res.add("dol_continuation", cfg.w_dol_continuation, "continuation toward its draw")

    # ── top-down bias ───────────────────────────────────────────────────────
    # None means genuinely unknown, which earns nothing. The Pine treats unknown
    # as bullish and pays it the full weight.
    if inp.bias_bull is not None and inp.bias_bull == inp.bull:
        res.score += cfg.w_bias
        res.add("bias_aligned", cfg.w_bias, "aligned with higher-timeframe bias")

    # ── displacement and size ───────────────────────────────────────────────
    if inp.strong_displacement:
        res.score += cfg.w_displacement
        res.add("displacement", cfg.w_displacement, "displaced strongly out of the zone")
    if inp.large_range:
        res.score += 4.0
        res.add("large_range", 4.0, "wide-range formation bar")

    # ── timeframe weighting (higher timeframe blocks rank above lower) ──────
    tf_bonus = 5.0 if inp.timeframe >= 60 else 3.0 if inp.timeframe >= 15 else 0.0
    if tf_bonus:
        res.score += tf_bonus
        res.add("timeframe", tf_bonus, f"{inp.timeframe}m block")

    # ── NEW: relative volume ────────────────────────────────────────────────
    if cfg.use_volume and inp.volume_available:
        if inp.rvol >= cfg.min_rvol:
            # Scaled so 1.2x earns a little and 3x earns the full weight,
            # capped -- a 20x volume spike is a news event, not 20x the edge.
            scaled = min((inp.rvol - cfg.min_rvol) / (3.0 - cfg.min_rvol), 1.0)
            pts = cfg.w_rvol * max(scaled, 0.25)
            res.score += pts
            res.add("relative_volume", pts, f"{inp.rvol:.1f}x normal volume for this minute")
        else:
            pts = -cfg.w_rvol * 0.5
            res.score += pts
            res.add("thin_volume", pts, f"only {inp.rvol:.1f}x normal volume")

    # ── NEW: VWAP proximity ─────────────────────────────────────────────────
    if cfg.use_vwap and not math.isnan(inp.vwap) and not math.isnan(inp.atr_value):
        band = inp.atr_value * cfg.vwap_band_atr
        if abs(inp.ce - inp.vwap) <= band:
            res.score += cfg.w_vwap
            res.add("vwap", cfg.w_vwap, "formed at session VWAP")

    # ── NEW: dealer gamma ───────────────────────────────────────────────────
    if cfg.use_gamma and inp.gamma is not None and not math.isnan(inp.atr_value):
        band = inp.atr_value * cfg.gamma_wall_band_atr
        pts, why = inp.gamma.score(
            inp.bull, inp.ce, inp.ext, inp.setup_type, band, cfg.w_gamma_wall
        )
        if pts:
            res.score += pts
            res.add("dealer_gamma", pts, why)

    # ── NEW: multi-timeframe agreement ──────────────────────────────────────
    if cfg.w_tf_agreement and inp.tf_agreement > 1:
        pts = cfg.w_tf_agreement * min((inp.tf_agreement - 1) / 2.0, 1.0)
        res.score += pts
        res.add("tf_agreement", pts, f"{inp.tf_agreement} timeframes agree")

    # ── NEW: midday lull penalty ────────────────────────────────────────────
    if inp.in_lull and cfg.lull_penalty:
        res.score -= cfg.lull_penalty
        res.add("midday_lull", -cfg.lull_penalty, "formed in the midday lull")

    res.score = max(0.0, min(res.score, 100.0))
    return res


def stack_count(
    bull: bool,
    ce: float,
    near_key_open: bool,
    in_amd: bool,
    setup_type: int,
    bias_bull: bool | None,
    smt_confirmed: bool,
    dr: DealingRange,
    pools: LiquidityPools,
    fvgs: FvgStore,
    gaps: GapStore,
    cfg: Config,
    rvol: float = 1.0,
    at_vwap: bool = False,
    volume_available: bool = True,
) -> int:
    """Count independent confluences. Port of `f_stackCount` plus the new ones.

    This is a count, not a score -- it answers "how many different reasons",
    which is a different question from "how good", and gating on it stops a
    single heavily-weighted term from carrying a signal on its own."""
    n = 0
    n += 1 if pools.recently_swept(bull) else 0
    n += 1 if dr.favourable(ce, bull, cfg.eq_band_pct) else 0
    n += 1 if fvgs.contains(ce, bull) else 0
    n += 1 if gaps.contains(ce) else 0
    n += 1 if near_key_open else 0
    n += 1 if in_amd else 0
    n += 1 if (bias_bull is not None and bias_bull == bull) else 0
    n += 1 if smt_confirmed else 0
    n += 1 if setup_type != 0 else 0
    if cfg.use_volume and volume_available and rvol >= cfg.min_rvol:
        n += 1
    if cfg.use_vwap and at_vwap:
        n += 1
    return n
