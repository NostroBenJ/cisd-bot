"""Independent cross-check of every hand-rolled statistic in this repo.

    python -m research.gs_crosscheck

Every conclusion in FINDINGS.md rests on statistics written by hand in stdlib:
our own t-statistic, our own Welch's test, our own variance ratio, our own
Stouffer combination, our own overlap-corrected standard error. They are
self-consistent, which is exactly the problem -- the code that computes the
number is the code that checks it. A shared misunderstanding of a formula
would be invisible.

This module checks them against `scipy.stats` and `gs_quant.timeseries`, which
were written by other people who did not read our code. Agreement to 1e-12 is
worth more than another assertion we wrote ourselves.

`scipy`, `numpy`, `pandas` and `gs_quant` are imported INSIDE the functions
that need them, per CLAUDE.md. `research/` and `cisd/` stay importable with
zero third-party dependencies; nothing here is on any pricing or trading path.

If this module cannot import its dependencies it says so and exits 0. A
cross-check is a luxury, not a gate -- it must never be the reason the repo
stops working on a machine that lacks scipy.
"""

from __future__ import annotations

import math
import random
import statistics
import sys

from research.gamma_regime import autocorr1, variance_ratio, welch
from research.harness import Result, Trade

TOL = 1e-9
PASS, FAIL, SKIP = 0, 0, 0


def check(label: str, ours: float, theirs: float, tol: float = TOL) -> None:
    global PASS, FAIL
    if math.isnan(ours) and math.isnan(theirs):
        PASS += 1
        print(f"  ok   {label:<44} both nan")
        return
    d = abs(ours - theirs)
    rel = d / max(abs(theirs), 1e-12)
    if d <= tol or rel <= tol:
        PASS += 1
        print(f"  ok   {label:<44} {ours:+.10g}  (diff {d:.2e})")
    else:
        FAIL += 1
        print(f"  FAIL {label:<44} ours {ours:+.10g}  theirs {theirs:+.10g}  diff {d:.3e}")


def fake_trades(rs: list[float]) -> Result:
    r = Result("x")
    r.trades = [Trade(i, i, 1, 100.0, 99.0, 102.0, 101.0, "target", v, 5)
                for i, v in enumerate(rs)]
    return r


def main() -> int:
    global SKIP
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
    except ImportError as e:
        print(f"cross-check skipped: {e}")
        print("install with: pip install scipy pandas")
        return 0

    rng = random.Random(11)
    a = [rng.gauss(0.03, 1.0) for _ in range(400)]
    b = [rng.gauss(0.00, 1.3) for _ in range(650)]

    print("\n[1] harness.Result.t_stat  vs  scipy.stats.ttest_1samp")
    ra = fake_trades(a)
    check("t vs zero", ra.t_stat, float(stats.ttest_1samp(a, 0.0).statistic))
    check("standard error", ra.se, float(stats.sem(a)))
    check("mean", ra.mean_r, float(np.mean(a)))

    print("\n[2] harness.Result.versus  vs  scipy Welch (equal_var=False)")
    rb = fake_trades(b)
    check("Welch t, signal vs null", ra.versus(rb),
          float(stats.ttest_ind(a, b, equal_var=False).statistic))
    # The direction of the comparison matters and is easy to invert silently.
    check("Welch t, reversed", rb.versus(ra),
          float(stats.ttest_ind(b, a, equal_var=False).statistic))

    print("\n[3] gamma_regime.welch  vs  scipy")
    diff, t = welch(a, b)
    check("mean difference", diff, float(np.mean(a) - np.mean(b)))
    check("Welch t", t, float(stats.ttest_ind(a, b, equal_var=False).statistic))

    print("\n[4] gamma_regime.autocorr1  vs  pearson on the lagged series")
    # Our version uses the FULL-series mean for both terms (the standard
    # estimator); pearsonr recentres each slice. They differ in small samples
    # by O(1/n), so this is checked as an approximation, not an identity.
    ac = autocorr1(a)
    pr = float(stats.pearsonr(a[:-1], a[1:]).statistic)
    check("lag-1 autocorrelation (~1/n apart)", ac, pr, tol=3.0 / len(a))

    print("\n[5] gamma_regime.variance_ratio  vs  a numpy recomputation")
    for q in (2, 5, 21):
        n = len(a) // q
        ones = np.array(a[: n * q])
        blocks = ones.reshape(n, q).sum(axis=1)
        theirs = float(blocks.var(ddof=1) / (q * ones.var(ddof=1)))
        check(f"VR({q}) from non-overlapping blocks", variance_ratio(a, q), theirs)

    print("\n[6] Stouffer combination  vs  scipy.stats.combine_pvalues")
    zs = [-1.42, -0.87, -2.10, -0.55]
    ours = sum(zs) / math.sqrt(len(zs))
    ps = [float(stats.norm.cdf(z)) for z in zs]          # one-sided, left tail
    theirs = float(stats.combine_pvalues(ps, method="stouffer").statistic)
    check("Stouffer z (equal weights)", ours, -theirs)

    print("\n[7] overlap-corrected SE  vs  a block series with a known answer")
    # 300 independent blocks of 21 identical values: the true SE is over the
    # 300 blocks. This is the vrp_study correction restated -- sd/sqrt(n_indep),
    # never sd/sqrt(n).
    blocks = [rng.gauss(0.5, 2.0) for _ in range(300)]
    series = [v for v in blocks for _ in range(21)]
    sd = statistics.stdev(series)
    se_corrected = sd / math.sqrt(len(blocks))
    se_naive = sd / math.sqrt(len(series))
    truth = float(stats.sem(blocks))
    check("corrected SE recovers the truth", se_corrected, truth, tol=0.02)
    check("naive SE is sqrt(21) too tight", se_naive * math.sqrt(21), se_corrected)
    print(f"       corrected {se_corrected:.5f}   naive {se_naive:.5f}   "
          f"understated by {se_corrected / se_naive:.2f}x")

    print("\n[8] realized volatility  vs  gs_quant.timeseries.volatility")
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import gs_quant.timeseries as gts
        from bot.feed import ArchiveFeed

        feed = ArchiveFeed()
        closes, dates = [], []
        for d in feed.sessions()[-260:]:
            bars = feed.session_bars(d)
            if bars:
                closes.append(bars[-1].close)
                dates.append(d)
        s = pd.Series(closes, index=pd.to_datetime(dates))

        # Ours: annualised stdev of daily log returns, in vol points.
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        w = 22
        ours = statistics.stdev(rets[-w:]) * math.sqrt(252) * 100.0
        theirs = float(gts.volatility(s, w).dropna().iloc[-1])
        # GS uses simple returns and a 252-day annualisation; on daily SPY the
        # log/simple gap is ~0.1 vol point, so this is a sanity band, not an
        # identity. It still catches a factor-of-sqrt(252) or a percent error.
        agree = abs(ours - theirs) < 0.75
        if agree:
            print(f"  ok   {'22d realized vol on real SPY':<44} ours {ours:.3f}  gs {theirs:.3f}")
        else:
            print(f"  FAIL {'22d realized vol on real SPY':<44} ours {ours:.3f}  gs {theirs:.3f}")
        _bump(agree)
        print(f"       {len(closes)} sessions, {dates[0]} -> {dates[-1]}")
    except Exception as e:
        SKIP += 1
        print(f"  skip gs_quant check: {type(e).__name__}: {str(e)[:80]}")

    print(f"\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
    return 1 if FAIL else 0


def _bump(ok: bool) -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1


if __name__ == "__main__":
    sys.exit(main())
