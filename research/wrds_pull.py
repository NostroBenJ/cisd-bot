"""Pull CRSP from WRDS. The dataset that fixes everything wrong with our data.

    python -m research.wrds_pull --probe          # what is this account entitled to?
    python -m research.wrds_pull --pull           # extract CRSP monthly
    python -m research.wrds_pull --pull --start 1990-01-01

**One-time setup, which you must run yourself in a terminal:**

    python -c "import wrds; wrds.Connection()"

It prompts for your WRDS username and password, offers to create a `.pgpass`
file, and after that every connection is passwordless. Say yes to the pgpass
prompt. Claude never sees the password and never should.

## Why this dataset and not another

Every known defect in `research/cross_momentum.py` is a data defect, and CRSP
fixes all four at once:

1. **Survivorship.** Our universe is 100 firms that all survived. CRSP contains
   every firm that ever listed, and `msedelist` carries the DELISTING RETURN --
   what you actually got when a company was acquired or went to zero. Omitting
   it is the single largest bias in retail backtests.
2. **Total return.** `RET` includes dividends. Our price-only series
   systematically penalises dividend payers in the ranking.
3. **Point-in-time universe.** `SHRCD` and `EXCHCD` are dated, so membership is
   reconstructed as it was, not as it is.
4. **Corporate actions.** CRSP returns are adjusted for splits, spinoffs and
   distributions properly. This is what defeated us on FDX and HON, where a
   spinoff is not a split and a split-adjusted series carried a phantom 18-50%
   one-day move.

And it starts in 1925 rather than 2019, which addresses the other complaint:
79 monthly observations against a 200 minimum.

## The universe filter, and why each clause is there

`SHRCD IN (10, 11)` -- ordinary common shares only. Excludes ADRs, REITs,
closed-end funds, units, and everything else whose returns are not comparable.

`EXCHCD IN (1, 2, 3)` -- NYSE, AMEX, NASDAQ.

Both are the standard filters in the momentum literature, applied by date so a
firm enters and leaves as it actually did.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

OUT = pathlib.Path("data/crsp")

MONTHLY_SQL = """
SELECT a.permno, a.date, a.ret, a.prc, a.shrout,
       b.shrcd, b.exchcd, b.ticker, b.comnam,
       c.dlret, c.dlstcd
FROM {msf} AS a
LEFT JOIN {names} AS b
       ON a.permno = b.permno
      AND b.namedt <= a.date
      AND a.date <= b.nameendt
LEFT JOIN {delist} AS c
       ON a.permno = c.permno
      AND date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
WHERE a.date BETWEEN '{start}' AND '{end}'
  AND b.shrcd IN (10, 11)
  AND b.exchcd IN (1, 2, 3)
"""


def connect():
    import wrds
    try:
        return wrds.Connection()
    except Exception as e:                       # noqa: BLE001
        print(f"could not connect: {type(e).__name__}: {e}\n")
        print("Run this ONCE in your own terminal, then re-run:")
        print('    python -c "import wrds; wrds.Connection()"')
        print("Say yes when it offers to create a .pgpass file.")
        raise SystemExit(1)


def probe() -> int:
    db = connect()
    print("connected.\n")
    libs = db.list_libraries()
    crsp = sorted(l for l in libs if "crsp" in l.lower())
    print(f"{len(libs)} libraries entitled; CRSP-related: {crsp}\n")
    for lib in ("crsp", "crspq", "crsp_a_stock"):
        if lib not in libs:
            continue
        try:
            tables = db.list_tables(library=lib)
        except Exception as e:                   # noqa: BLE001
            print(f"  {lib}: {type(e).__name__}")
            continue
        want = [t for t in tables
                if t in ("msf", "msf_v2", "msenames", "msedelist",
                         "stksecurityinfohist", "dsf", "dsf_v2")]
        print(f"  {lib}: {len(tables)} tables; relevant here -> {sorted(want)}")
    print("\nIf you see msf + msenames + msedelist, the classic path works.")
    print("If you only see msf_v2 / stksecurityinfohist, the subscription is on")
    print("the newer CIZ format and the query needs adjusting -- tell me which.")
    return 0


def pull(start: str, end: str) -> int:
    import pandas as pd
    db = connect()
    libs = db.list_libraries()
    lib = "crsp" if "crsp" in libs else ("crspq" if "crspq" in libs else None)
    if lib is None:
        print(f"no CRSP library found. Entitled: {sorted(libs)[:25]}")
        return 1

    sql = MONTHLY_SQL.format(msf=f"{lib}.msf", names=f"{lib}.msenames",
                             delist=f"{lib}.msedelist", start=start, end=end)
    print(f"querying {lib}.msf {start} -> {end} ... (this takes a few minutes)")
    df = db.raw_sql(sql, date_cols=["date"])
    print(f"{len(df):,} rows, {df.permno.nunique():,} unique PERMNOs")

    # THE delisting adjustment. A firm that is acquired or fails has a final
    # partial return that lives in dlret, not ret. Dropping it is the bias this
    # entire exercise exists to remove -- and it is silent, because the row
    # simply stops appearing.
    d = df["dlret"].fillna(0.0)
    r = df["ret"].fillna(0.0)
    both = df["ret"].notna() & df["dlret"].notna()
    only_dl = df["ret"].isna() & df["dlret"].notna()
    df["ret_adj"] = r
    df.loc[both, "ret_adj"] = (1 + df.loc[both, "ret"]) * (1 + df.loc[both, "dlret"]) - 1
    df.loc[only_dl, "ret_adj"] = df.loc[only_dl, "dlret"]
    df.loc[df["ret"].isna() & df["dlret"].isna(), "ret_adj"] = float("nan")

    df["mktcap"] = df["prc"].abs() * df["shrout"]      # prc<0 means bid/ask midpoint

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "crsp_monthly.parquet"
    try:
        df.to_parquet(p, index=False)
    except Exception:                            # noqa: BLE001 - pyarrow missing
        p = OUT / "crsp_monthly.csv"
        df.to_csv(p, index=False)
    print(f"wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    print(f"\ndelisting events with a return: {df['dlret'].notna().sum():,}")
    print(f"rows where dlret is the ONLY return: {int(only_dl.sum()):,}")
    print(f"date range: {df.date.min().date()} -> {df.date.max().date()}")
    print(f"months: {df.date.dt.to_period('M').nunique():,}")
    per = df.groupby(df.date.dt.to_period("M")).permno.nunique()
    print(f"stocks per month: min {per.min():,}  median {int(per.median()):,}  "
          f"max {per.max():,}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default="2026-12-31")
    a = ap.parse_args()
    if a.probe:
        return probe()
    if a.pull:
        return pull(a.start, a.end)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
