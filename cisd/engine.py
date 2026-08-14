"""The lifecycle state machine and multi-timeframe orchestration.

A rejection block passes through five states, and the order matters:

    formed -> displaced -> price left the zone -> price returned (armed)
           -> closed through the initiating open (CISD) -> SIGNAL

Skipping "price left the zone" would let a block trigger on the same push that
created it, which is the difference between a retest and a chase. The Pine gets
this right via `everLeft`; it is reproduced here.

**Anti-repaint by construction.** Higher-timeframe detection runs only when an
HTF bar has *closed*, determined by comparing bar end times against the current
base bar's timestamp. No function reads an index greater than the one it was
given. A backtest and a live run therefore see identical inputs at identical
times, which is the property the whole exercise depends on -- an ICT model that
repaints backtests beautifully and trades badly is the default outcome here.

The CISD trigger is evaluated on the BASE timeframe regardless of which
timeframe produced the block, matching the Pine's design note: the change in
state is read on the chart timeframe beneath the higher-timeframe arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .bars import Bar, resample
from .config import Config
from .context import (
    DealingRange,
    FvgStore,
    GapStore,
    LiquidityPools,
    draw_targets,
    find_fvg,
    setup_type,
)
from .detect import RejectionBlock, cisd_close, cisd_direction, detect_rejection_block
from .gamma import GammaContext
from .grade import GradeInputs, grade, stack_count
from .indicators import atr, pivot_high, pivot_low, relative_volume, session_vwap
from .sessions import EventCalendar, KeyOpenTracker, in_session, in_window, parse_session
from .signal import Signal, build_targets


def _modal_spacing_minutes(bars: list[Bar]) -> int:
    """Most common gap between consecutive bars, in minutes.

    The mode, not the mean: session gaps and weekends would drag an average
    far away from the actual bar size."""
    counts: dict[int, int] = {}
    for a, b in zip(bars, bars[1:]):
        gap = (b.ts - a.ts) // 60
        if gap > 0:
            counts[gap] = counts.get(gap, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else 1


# ── bias ────────────────────────────────────────────────────────────────────


@dataclass
class BiasTracker:
    """Higher-timeframe directional bias, with an expiry.

    The Pine latches this forever and initialises it to a value that reads as
    bullish, so on a cold start every long is permitted and every short blocked
    by a bias that has never actually been established. Here an unset or expired
    bias is None -- genuinely neutral -- and with `require_bias` on that blocks
    both directions rather than silently favouring one.

    Ages are counted in BIAS-timeframe bars, tracked internally. Taking the
    index from the caller would silently compare a 60-minute index against a
    1-minute one, which expires the bias on every bar and reads as permanently
    neutral -- a failure that looks exactly like "the gate is working"."""

    cfg: Config
    direction: int = 0
    set_at: int = -1
    _now: int = -1

    def update(self, bars: list[Bar], index: int, cfg: Config) -> None:
        self._now = index
        d = cisd_direction(bars, index, cfg)
        if d != 0:
            self.direction = d
            self.set_at = index

    def current(self) -> bool | None:
        if self.direction == 0 or self.set_at < 0:
            return None
        if self.cfg.bias_expires and (self._now - self.set_at) > self.cfg.bias_max_age_bars:
            return None
        return self.direction > 0

    def label(self) -> str:
        c = self.current()
        return "neutral" if c is None else ("bull" if c else "bear")


# ── SMT divergence ──────────────────────────────────────────────────────────


@dataclass
class SmtTracker:
    """Divergence against correlated peers (QQQ, DIA).

    A bearish divergence is: we made a higher high, the peer did not. It says
    the move lacks broad participation.

    The Pine's staleness problem is fixed here. A pivot cannot confirm until
    `smt_pivot_strength` bars after the fact, and the Pine then accepts that
    confirmation for another `sweep_recency_bars` -- on a 1-minute chart, a
    divergence from seventeen minutes ago counts as confirming now.
    `smt_max_age_bars` caps the TOTAL age, pivot lag included."""

    cfg: Config
    available: bool = False
    _bull_at: int | None = None
    _bear_at: int | None = None
    _my_highs: list[float] = field(default_factory=list)
    _my_lows: list[float] = field(default_factory=list)
    _peer_highs: list[float] = field(default_factory=list)
    _peer_lows: list[float] = field(default_factory=list)

    def build(self, bars: list[Bar], peer: list[Bar] | None) -> None:
        """Precompute pivot-aligned divergence flags."""
        if not self.cfg.use_smt or peer is None or len(peer) != len(bars):
            self.available = False
            return
        self.available = True
        k = self.cfg.smt_pivot_strength
        ph = pivot_high(bars, k, k)
        pl = pivot_low(bars, k, k)
        self._ph, self._pl, self._peer = ph, pl, peer

    def update(self, bars: list[Bar], index: int) -> None:
        if not self.available:
            return
        k = self.cfg.smt_pivot_strength
        p = index - k
        if p < 0:
            return
        if self._ph[index] is not None:
            self._my_highs.append(bars[p].high)
            self._peer_highs.append(self._peer[p].high)
            if len(self._my_highs) >= 2:
                if self._my_highs[-1] > self._my_highs[-2] and self._peer_highs[-1] < self._peer_highs[-2]:
                    self._bear_at = index
        if self._pl[index] is not None:
            self._my_lows.append(bars[p].low)
            self._peer_lows.append(self._peer[p].low)
            if len(self._my_lows) >= 2:
                if self._my_lows[-1] < self._my_lows[-2] and self._peer_lows[-1] > self._peer_lows[-2]:
                    self._bull_at = index

    def confirmed(self, bull: bool, index: int) -> bool:
        if not self.available:
            return False
        at = self._bull_at if bull else self._bear_at
        if at is None:
            return False
        return (index - at) <= self.cfg.smt_max_age_bars


# ── daily / weekly ranges ───────────────────────────────────────────────────


def build_period_ranges(
    bars: list[Bar], lag: int = 1
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Previous-day and previous-week high/low, keyed by session date.

    Returned maps answer "for a bar on date D, what were the previous period's
    extremes" -- so the lookup is already lagged and cannot leak the current
    day's range into the current day's scoring.

    `lag` is how many periods back to reach. 1 is the actual previous period.
    2 reproduces the Pine, whose `request.security(..., high[1],
    lookahead_off)` lands two periods back rather than one -- see
    `Config.prev_period_lag`."""
    if lag < 1:
        raise ValueError(f"lag must be at least 1, got {lag}")
    daily: dict[str, tuple[float, float]] = {}
    weekly_by_key: dict[str, tuple[float, float]] = {}
    order: list[str] = []
    week_of: dict[str, str] = {}

    for b in bars:
        d = b.session_date_ny
        if d not in daily:
            daily[d] = (b.high, b.low)
            order.append(d)
            iso = b.dt_ny.isocalendar()
            week_of[d] = f"{iso.year}-W{iso.week:02d}"
        else:
            h, l = daily[d]
            daily[d] = (max(h, b.high), min(l, b.low))
        wk = week_of[d]
        if wk not in weekly_by_key:
            weekly_by_key[wk] = (b.high, b.low)
        else:
            h, l = weekly_by_key[wk]
            weekly_by_key[wk] = (max(h, b.high), min(l, b.low))

    prev_day: dict[str, tuple[float, float]] = {}
    for i, d in enumerate(order):
        prev_day[d] = daily[order[i - lag]] if i >= lag else (math.nan, math.nan)

    week_order: list[str] = []
    for d in order:
        wk = week_of[d]
        if wk not in week_order:
            week_order.append(wk)
    prev_week: dict[str, tuple[float, float]] = {}
    for d in order:
        wk = week_of[d]
        idx = week_order.index(wk)
        prev_week[d] = (
            weekly_by_key[week_order[idx - lag]] if idx >= lag else (math.nan, math.nan)
        )

    return prev_day, prev_week


# ── the engine ──────────────────────────────────────────────────────────────


@dataclass
class EngineStats:
    """Counters the verification suite and the backtest report read.

    `blocks_formed` versus `signals` is the number that says whether the gates
    are doing anything at all -- if they are nearly equal, the gate chain is
    decorative, which is exactly what the Pine's anchor gate turned out to be."""

    bars: int = 0
    blocks_formed: int = 0
    blocks_confirmed: int = 0
    blocks_armed: int = 0
    blocks_invalidated: int = 0
    triggers: int = 0
    signals: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    grade_counts: dict[str, int] = field(default_factory=dict)
    # Every rejection with the bar it happened on. This is what turns "the
    # port missed a signal" into "the port rejected it here, for this reason",
    # which is the difference between a diagnosis and a guess.
    rejected_at: list[tuple[int, str]] = field(default_factory=list)
    _ts: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        self.rejected_at.append((self._ts, reason))

    def around(self, ts: int, window_s: int = 3600) -> list[tuple[int, str]]:
        """Rejections within `window_s` seconds of `ts`, oldest first."""
        return [(t, r) for t, r in self.rejected_at if abs(t - ts) <= window_s]


class CisdEngine:
    """Runs the model over a series of base-timeframe bars.

    Usage:

        engine = CisdEngine(Config(), symbol="SPY")
        signals = engine.run(one_minute_bars)

    Peer bars for SMT and a gamma snapshot are optional; both degrade to
    "unavailable" rather than to a fabricated value, and the resulting signals
    record which inputs were missing."""

    def __init__(
        self,
        cfg: Config | None = None,
        symbol: str = "SPY",
        gamma: GammaContext | None = None,
        events: EventCalendar | None = None,
        peer_bars: list[Bar] | None = None,
        collect_diagnostics: bool = False,
    ) -> None:
        self.cfg = cfg or Config()
        self.symbol = symbol
        self.gamma = gamma
        self.events = events or EventCalendar()
        self.peer_bars = peer_bars
        self.stats = EngineStats()
        self.blocks: list[RejectionBlock] = []
        self.signals: list[Signal] = []
        # Per-bar context series, recorded only when asked. These exist so the
        # TradingView diff can compare the INPUT layers bar by bar instead of
        # inferring them from four signal bars a year apart -- a score that is
        # six points off tells you nothing about which of eight terms moved.
        self.collect_diagnostics = collect_diagnostics
        self.diagnostics: dict[str, list[float]] = {
            k: [] for k in (
                "pdh", "pdl", "pwh", "pwl", "pd_depth_bull", "pd_depth_bear",
                "dol_up", "dol_dn", "sweep_bull", "sweep_bear",
            )
        }

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self, base_bars: list[Bar]) -> list[Signal]:
        cfg = self.cfg
        if len(base_bars) < 50:
            raise ValueError(f"need at least 50 bars to warm up, got {len(base_bars)}")

        # The Pine compares a block's timeframe against `timeframe.period`,
        # the CHART timeframe. Infer it from the bars rather than assuming the
        # base is 1-minute: a 5-minute export makes 5 the chart timeframe.
        self._chart_tf = _modal_spacing_minutes(base_bars)
        # A feed with no volume column at all must not be read as thin
        # participation -- an absent column is not low participation, and
        # scoring it as such silently rejects every signal.
        self._has_volume = any(b.volume > 0 for b in base_bars)
        timeframes = self._timeframes()
        series = {tf: (base_bars if tf == 1 else resample(base_bars, tf)) for tf in timeframes}
        atrs = {tf: atr(series[tf], cfg.atr_length) for tf in timeframes}

        bias_series = resample(base_bars, cfg.bias_timeframe)
        bias = BiasTracker(cfg)
        bias_ptr = 0

        fvg_series = resample(base_bars, cfg.htf_fvg_timeframe)
        fvg_ptr = 0

        base_atr = atrs.get(1) or atr(base_bars, cfg.atr_length)
        vwap, _vwap_sd = session_vwap(base_bars)
        rvol = relative_volume(base_bars, cfg.rvol_lookback_sessions)

        prev_day, prev_week = build_period_ranges(base_bars, cfg.prev_period_lag)

        smt = SmtTracker(cfg)
        smt.build(base_bars, self.peer_bars)

        dr = DealingRange()
        pools = LiquidityPools(cfg)
        fvgs = FvgStore(cfg)
        gaps = GapStore(cfg)
        keyopens = KeyOpenTracker(cfg.key_opens)

        tf_ptr = {tf: 0 for tf in timeframes}
        traded_start, traded_end = parse_session(cfg.traded_window)
        session_signals = 0
        cur_session: str | None = None

        for i, bar in enumerate(base_bars):
            self.stats.bars += 1
            self.stats._ts = bar.ts

            # session bookkeeping
            d = bar.session_date_ny
            if d != cur_session:
                cur_session = d
                # The quota ALWAYS resets on a day boundary. `quota_resets_at_
                # session` chooses WHICH boundary -- the Pine rolls on the NY
                # calendar date (midnight), so an overnight signal spends the
                # budget before 09:30, while the corrected build rolls at the
                # traded-window open. On regular-hours data the two coincide.
                # Treating the flag as "never reset" made the counter
                # accumulate across the whole year and silently blocked every
                # signal after the sixth.
                session_signals = 0
                pd_h, pd_l = prev_day.get(d, (math.nan, math.nan))
                pw_h, pw_l = prev_week.get(d, (math.nan, math.nan))
                dr = DealingRange(pd_h, pd_l, pw_h, pw_l)
                if cfg.track_gaps and i > 0:
                    gaps.add_open_gap(base_bars[i - 1].close, bar.open, bar.ts)

            keyopens.update(bar)
            pools.update(bar, i, dr)
            gaps.update(bar, base_atr[i])
            smt.update(base_bars, i)

            if self.collect_diagnostics:
                d = self.diagnostics
                d["pdh"].append(dr.pdh)
                d["pdl"].append(dr.pdl)
                d["pwh"].append(dr.pwh)
                d["pwl"].append(dr.pwl)
                d["pd_depth_bull"].append(dr.depth(bar.close, True))
                d["pd_depth_bear"].append(dr.depth(bar.close, False))
                up, dn = draw_targets(bar.close, dr, pools)
                d["dol_up"].append(up if up is not None else math.nan)
                d["dol_dn"].append(dn if dn is not None else math.nan)
                d["sweep_bull"].append(1.0 if pools.recently_swept(True) else 0.0)
                d["sweep_bear"].append(1.0 if pools.recently_swept(False) else 0.0)

            # advance the bias timeframe as its bars close
            while (
                bias_ptr < len(bias_series)
                and bias_series[bias_ptr].ts + cfg.bias_timeframe * 60 <= bar.ts
            ):
                bias.update(bias_series, bias_ptr, cfg)
                bias_ptr += 1

            # advance the fair-value-gap timeframe
            while (
                fvg_ptr < len(fvg_series)
                and fvg_series[fvg_ptr].ts + cfg.htf_fvg_timeframe * 60 <= bar.ts
            ):
                g = find_fvg(fvg_series, fvg_ptr, cfg.fvg_min_points)
                if g is not None:
                    fvgs.add(g)
                fvg_ptr += 1
            fvgs.prune(bar.close, base_atr[i])

            # detect new blocks on each timeframe as its bars close
            for tf in timeframes:
                s = series[tf]
                while tf_ptr[tf] < len(s) and s[tf_ptr[tf]].ts + tf * 60 <= bar.ts:
                    j = tf_ptr[tf]
                    a = atrs[tf][j] if j < len(atrs[tf]) else math.nan
                    blk = detect_rejection_block(s, j, cfg, a, tf, rvol[i])
                    if blk is not None:
                        self._intake(blk, bar, i, dr, pools, fvgs, gaps, keyopens, bias, smt, base_atr[i], vwap[i])
                    tf_ptr[tf] += 1

            # lifecycle + trigger
            in_traded = in_window(bar.minute_of_day_ny, traded_start, traded_end)
            past_open_noise = bar.minute_of_day_ny >= (9 * 60 + 30 + cfg.skip_first_minutes)
            sig = self._advance(
                base_bars, i, bar, dr, pools, fvgs, gaps, keyopens, bias, smt,
                base_atr[i], vwap[i], rvol[i], atrs, series,
            )
            if sig is not None:
                blocked = self.events.blocked(bar.ts)
                if blocked:
                    self.stats.reject(f"event_blackout:{blocked}")
                elif not in_traded:
                    self.stats.reject("outside_traded_window")
                elif not past_open_noise:
                    self.stats.reject("opening_noise")
                elif session_signals >= cfg.max_signals_per_session:
                    self.stats.reject("session_quota")
                else:
                    session_signals += 1
                    self.stats.signals += 1
                    self.stats.grade_counts[sig.grade] = self.stats.grade_counts.get(sig.grade, 0) + 1
                    self.signals.append(sig)

        return self.signals

    # ── helpers ─────────────────────────────────────────────────────────────

    def _timeframes(self) -> list[int]:
        """Timeframes to detect on, deduplicated.

        `base_timeframe` is clamped up to the chart timeframe. Asking for
        1-minute detection on 5-minute bars would resample to an identity copy
        and then detect the same blocks twice under two different timeframe
        labels -- which also defeats anti-stack, since that only compares
        blocks sharing a timeframe."""
        tfs = list(self.cfg.rb_timeframes)
        base = self.cfg.base_timeframe
        if base:
            base = max(base, getattr(self, "_chart_tf", base))
            if base not in tfs:
                tfs.insert(0, base)
        return sorted(set(t for t in tfs if t > 0))

    def _intake(
        self, blk, bar, i, dr, pools, fvgs, gaps, keyopens, bias, smt, base_atr_v, vwap_v
    ) -> None:
        """Grade a new block and admit it, applying the anchor and stack gates."""
        cfg = self.cfg
        atr_v = base_atr_v
        if math.isnan(atr_v):
            return

        prox = atr_v * cfg.prox_atr_mult
        near_open = keyopens.near(blk.ce, prox)
        in_fvg = fvgs.contains(blk.ce, blk.bull)
        fav_pd = dr.favourable(blk.ce, blk.bull, cfg.eq_band_pct)
        in_gap = gaps.contains(blk.ce)
        in_amd = cfg.amd_mode != "Off" and any(in_session(bar, w) for w in cfg.amd_windows)
        amdx = pools.manipulation(blk.bull)

        # The anchor gate. In the Pine `gate_intrinsic_any_sweep` is effectively
        # always true, which makes this whole test vacuous; with the fix on, the
        # three real anchors have to carry it.
        anchored = (
            (cfg.gate_fvg and in_fvg)
            or (cfg.gate_key_open and near_open)
            or (cfg.gate_pd and fav_pd)
            or in_gap
            or amdx
            or cfg.gate_intrinsic_any_sweep
        )
        if not anchored:
            self.stats.reject("not_anchored")
            return
        if cfg.amd_mode == "Hard" and not in_amd:
            self.stats.reject("amd_hard_block")
            return

        bias_bull = bias.current()
        st = setup_type(blk.bull, blk.ce, blk.ext, bar.close, bias_bull, dr, pools, cfg, atr_v)
        agreement = self._agreement(blk, atr_v)

        gi = GradeInputs(
            bull=blk.bull, ce=blk.ce, ext=blk.ext, timeframe=blk.timeframe,
            strong_displacement=blk.strong_close, large_range=blk.large_range,
            rvol=blk.rvol, near_key_open=near_open, in_amd_window=in_amd,
            amdx_setup=amdx, setup_type=st, bias_bull=bias_bull,
            smt_confirmed=smt.confirmed(blk.bull, i), tf_agreement=agreement,
            in_lull=in_session(bar, cfg.lull_window), atr_value=atr_v,
            vwap=vwap_v, gamma=self.gamma, volume_available=self._has_volume,
        )
        result = grade(gi, cfg, dr, pools, fvgs, gaps)

        if cfg.grade_rank(cfg.grade_of(result.score)) < cfg.grade_rank(cfg.min_grade_to_track):
            self.stats.reject("below_track_grade")
            return

        blk.score = result.score
        blk.reasons = result.reasons
        blk.dol_type = st
        blk.smt_ok = gi.smt_confirmed
        blk.amdx = amdx
        is_htf = blk.timeframe != self._chart_tf
        blk.confirmed = (not cfg.require_displacement) or (
            cfg.htf_blocks_skip_displacement and is_htf
        )

        if cfg.anti_stack and self._superseded(blk):
            self.stats.reject("anti_stack")
            return

        self.blocks.append(blk)
        self.stats.blocks_formed += 1
        if len(self.blocks) > cfg.max_live_blocks:
            self.blocks.pop(0)

    def _superseded(self, new: RejectionBlock) -> bool:
        """One block per level. A new block overlapping an existing same-side
        block on the same timeframe is dropped if the incumbent scores higher,
        and replaces it otherwise."""
        keep: list[RejectionBlock] = []
        superseded = False
        for z in self.blocks:
            if (
                z.active
                and z.timeframe == new.timeframe
                and z.bull == new.bull
                and min(new.top, z.top) - max(new.bottom, z.bottom) > 0
            ):
                if z.score >= new.score:
                    superseded = True
                    keep.append(z)
                else:
                    z.active = False
            else:
                keep.append(z)
        self.blocks = keep
        return superseded

    def _agreement(self, blk: RejectionBlock, atr_v: float) -> int:
        """How many timeframes hold an active same-direction block near this
        one. This is the "check itself across timeframes" requirement, scored
        rather than gated so the backtest can price it."""
        band = atr_v * 1.0
        tfs = {blk.timeframe}
        for z in self.blocks:
            if z.active and z.bull == blk.bull and abs(z.ce - blk.ce) <= band:
                tfs.add(z.timeframe)
        return len(tfs)

    def _advance(
        self, base_bars, i, bar, dr, pools, fvgs, gaps, keyopens, bias, smt,
        base_atr_v, vwap_v, rvol_v, atrs, series,
    ) -> Signal | None:
        """Advance every live block one bar and return the best signal, if any."""
        cfg = self.cfg
        if math.isnan(base_atr_v):
            return None

        best: tuple[float, Signal] | None = None

        for z in self.blocks:
            if not z.active:
                continue

            block_atr = self._block_atr(z, atrs, series, bar.ts, base_atr_v)
            bars_alive = self._bars_alive(z, bar, cfg)

            # 1 · displacement confirmation
            if not z.confirmed:
                z.displacement_peak = max(z.displacement_peak, bar.high) if z.bull else min(z.displacement_peak, bar.low)
                moved = (z.displacement_peak - z.ce) if z.bull else (z.ce - z.displacement_peak)
                if moved >= block_atr * cfg.disp_atr_mult:
                    z.confirmed = True
                    z.displacement_extreme = z.displacement_peak
                    self.stats.blocks_confirmed += 1
                elif bars_alive > cfg.disp_window_bars * z.timeframe:
                    z.active = False
                continue

            # 2 · invalidation and expiry
            if z.invalidated_by(bar, cfg.invalidation_buffer):
                z.active = False
                self.stats.blocks_invalidated += 1
                continue
            if not z.tapped and bars_alive > cfg.max_leg_bars * z.timeframe:
                z.active = False
                continue

            if not z.tapped:
                z.displacement_extreme = (
                    max(z.displacement_extreme, bar.high) if z.bull else min(z.displacement_extreme, bar.low)
                )

            # 3 · price must LEAVE the zone before a retest counts
            if not z.ever_left:
                z.ever_left = bar.low > z.top if z.bull else bar.high < z.bottom
                continue

            # 4 · price returned into the zone -> armed
            if z.contains(bar):
                if not z.armed:
                    z.armed = True
                    if not z.ever_armed:
                        z.ever_armed = True
                        self.stats.blocks_armed += 1

            # 5 · the CISD trigger, read on the base timeframe.
            #
            # THE ORDER OF 4, 5 AND 6 IS LOAD-BEARING. The Pine arms, then
            # tests the trigger, and only then disarms. Disarming before the
            # trigger -- which reads as the obvious cleanup -- discards every
            # signal where price retested the zone and pushed back out of it on
            # the SAME bar that the change in state fired. That is the normal
            # shape of the setup, not an edge case: the bar that closes through
            # the initiating open is usually the bar that leaves the zone.
            if z.armed and not z.tapped:
                fired = True
                if cfg.use_cisd_trigger:
                    fired, _lvl = cisd_close(base_bars, i, z.bull, cfg)
                if fired:
                    z.tapped = True
                    self.stats.triggers += 1
                    sig = self._emit(
                        z, base_bars, i, bar, dr, pools, fvgs, gaps, keyopens,
                        bias, smt, base_atr_v, vwap_v, rvol_v,
                    )
                    if sig is not None and (best is None or sig.score > best[0]):
                        best = (sig.score, sig)

            # 6 · disarm only after the trigger has had its chance
            if z.armed and not z.tapped:
                left = bar.low > z.top if z.bull else bar.high < z.bottom
                if left:
                    z.armed = False

        return best[1] if best else None

    def _block_atr(self, z, atrs, series, ts, fallback) -> float:
        """ATR on the block's own timeframe when unified, else the base ATR.

        The Pine mixes these -- sweep distance uses the block's timeframe while
        displacement uses the chart's -- which makes the two multipliers
        incomparable and untunable together."""
        if not self.cfg.unify_atr_timeframe:
            return fallback
        s = series.get(z.timeframe)
        a = atrs.get(z.timeframe)
        if not s or not a:
            return fallback
        idx = 0
        for j in range(len(s) - 1, -1, -1):
            if s[j].ts + z.timeframe * 60 <= ts:
                idx = j
                break
        v = a[idx] if idx < len(a) else math.nan
        return fallback if math.isnan(v) else v

    @staticmethod
    def _bars_alive(z: RejectionBlock, bar: Bar, cfg: Config) -> int:
        """Age in BASE-timeframe bars. The Pine scales its caps by the
        timeframe ratio; the same scaling is applied at the comparison site."""
        return max(0, (bar.ts - z.born_ts) // 60)

    def _emit(
        self, z, base_bars, i, bar, dr, pools, fvgs, gaps, keyopens, bias, smt, atr_v, vwap_v, rvol_v
    ) -> Signal | None:
        """Re-check every gate at trigger time and build the signal.

        Re-grading here rather than reusing the formation score is deliberate:
        the block may have formed twenty minutes ago and the context around it
        has moved. The Pine does the same."""
        cfg = self.cfg
        bias_bull = bias.current()
        near_open = keyopens.near(z.ce, atr_v * cfg.prox_atr_mult)
        in_amd = cfg.amd_mode != "Off" and any(in_session(bar, w) for w in cfg.amd_windows)
        st = setup_type(z.bull, z.ce, z.ext, bar.close, bias_bull, dr, pools, cfg, atr_v)
        smt_ok = smt.confirmed(z.bull, i)
        agreement = self._agreement(z, atr_v)
        at_vwap = not math.isnan(vwap_v) and abs(z.ce - vwap_v) <= atr_v * cfg.vwap_band_atr

        gi = GradeInputs(
            bull=z.bull, ce=z.ce, ext=z.ext, timeframe=z.timeframe,
            strong_displacement=z.strong_close, large_range=z.large_range,
            rvol=rvol_v, near_key_open=near_open, in_amd_window=in_amd,
            amdx_setup=z.amdx, setup_type=st, bias_bull=bias_bull,
            smt_confirmed=smt_ok, tf_agreement=agreement,
            in_lull=in_session(bar, cfg.lull_window), atr_value=atr_v,
            vwap=vwap_v, gamma=self.gamma, volume_available=self._has_volume,
        )
        result = grade(gi, cfg, dr, pools, fvgs, gaps)
        g = cfg.grade_of(result.score)

        if cfg.grade_rank(g) < cfg.grade_rank(cfg.min_grade_to_signal):
            deep = dr.deep_on_both(z.ce, z.bull) and dr.depth(z.ce, z.bull) >= cfg.pd_strong
            if not (cfg.allow_b_grade_deep_pd and result.score >= 60 and deep):
                self.stats.reject("below_signal_grade")
                return None

        if cfg.require_bias:
            strong_pd = dr.favourable(z.ce, z.bull, cfg.eq_band_pct) and dr.depth(z.ce, z.bull) >= cfg.pd_strong
            if bias_bull is None and not strong_pd:
                self.stats.reject("bias_neutral")
                return None
            if bias_bull is not None and bias_bull != z.bull and not strong_pd:
                self.stats.reject("bias_opposed")
                return None

        stk = stack_count(
            z.bull, z.ce, near_open, in_amd, st, bias_bull, smt_ok,
            dr, pools, fvgs, gaps, cfg, rvol_v, at_vwap, self._has_volume,
        )
        if cfg.use_stack_gate and stk < cfg.min_stack:
            self.stats.reject("stack_too_thin")
            return None

        if cfg.smt_required and not smt_ok:
            self.stats.reject("smt_required")
            return None

        if cfg.use_dol:
            if cfg.dol_mode == "Both" and st == 0:
                self.stats.reject("no_draw")
                return None
            if cfg.dol_mode == "Continuation only" and st != 1:
                self.stats.reject("not_continuation")
                return None
            if cfg.dol_mode == "Reversal only" and st != 2:
                self.stats.reject("not_reversal")
                return None

        if cfg.use_volume and self._has_volume and rvol_v < cfg.min_rvol:
            self.stats.reject("thin_volume")
            return None

        entry = bar.close
        stop = z.ext - cfg.invalidation_buffer if z.bull else z.ext + cfg.invalidation_buffer
        targets = build_targets(
            entry, z.ext, z.displacement_extreme, stop, cfg.sigma_levels, z.bull
        )
        if not targets:
            self.stats.reject("no_valid_target")
            return None

        warnings: list[str] = []
        if not smt.available:
            warnings.append("SMT unavailable (no peer bars supplied)")
        if self.gamma is None:
            warnings.append("no dealer gamma snapshot")
        elif self.gamma.is_stale(bar.ts):
            warnings.append(f"gamma snapshot {self.gamma.age_seconds(bar.ts) // 60}m old")
        if not self._has_volume:
            warnings.append("feed carries no volume; volume terms disabled")
        if self.events.is_empty():
            warnings.append("no event calendar loaded; macro blackouts not enforced")

        return Signal(
            ts=bar.ts, symbol=self.symbol,
            direction="long" if z.bull else "short",
            entry=entry, stop=stop, targets=targets,
            grade=g, score=result.score, stack=stk,
            timeframe=z.timeframe,
            setup_type={2: "reversal", 1: "continuation"}.get(st, "none"),
            tf_agreement=agreement, bias=bias.label(),
            reasons=result.reasons, notes=result.notes,
            gamma_regime=self.gamma.regime_note() if self.gamma else "unknown",
            warnings=warnings,
        )
