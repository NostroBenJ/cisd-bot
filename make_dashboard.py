"""Render the bot dashboard from data/dashboard.json.

    python make_dashboard.py

The page is a DIVERGENCE MONITOR, not a portfolio viewer. Alpaca already shows
positions and P&L; what nothing else shows is whether the live paper run is
tracking the backtest, and whether the strategy has earned the right to trade.
Both of those are controls. A list of holdings is decoration.

So the gate verdict is the loudest element on the page and the failing criteria
are drawn as bars against their thresholds -- because "t = 2.13" means nothing
without "the bar is 2.5" sitting next to it.

Self-contained: no external JS, no chart library. The SVG paths are computed
here and written into the file.
"""

from __future__ import annotations

import json
import pathlib

SRC = pathlib.Path("data/dashboard.json")
OUT = pathlib.Path("dashboard.html")

W, H = 920, 300
PAD_L, PAD_R, PAD_T, PAD_B = 52, 14, 16, 28


def path_for(vals: list[float], vmin: float, vmax: float) -> str:
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = PAD_L + (i / max(n - 1, 1)) * (W - PAD_L - PAD_R)
        y = H - PAD_B - ((v - vmin) / (vmax - vmin)) * (H - PAD_T - PAD_B)
        pts.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(pts)


def area_for(vals: list[float], vmin: float, vmax: float) -> str:
    p = path_for(vals, vmin, vmax)
    x0 = PAD_L
    x1 = PAD_L + (W - PAD_L - PAD_R)
    return f"{p} L{x1:.1f},{H - PAD_B} L{x0:.1f},{H - PAD_B} Z"


def nice_ticks(vmin: float, vmax: float, n: int = 4) -> list[float]:
    span = vmax - vmin
    raw = span / n
    mag = 10 ** (len(str(int(raw))) - 1)
    step = round(raw / mag) * mag or mag
    out, v = [], (int(vmin / step) + 1) * step
    while v < vmax:
        out.append(v)
        v += step
    return out


def threshold_bar(label: str, value: float, bar: float, fmt: str,
                  passing: bool, note: str) -> str:
    """A criterion drawn against its threshold. The tick is the bar."""
    scale = max(value, bar) * 1.25
    pct = 100 * value / scale
    tick = 100 * bar / scale
    cls = "pass" if passing else "fail"
    return f"""
      <div class="crit {cls}">
        <div class="crit-head">
          <span class="crit-label">{label}</span>
          <span class="crit-val">{fmt}</span>
        </div>
        <div class="track">
          <div class="fill" style="width:{pct:.1f}%"></div>
          <div class="tick" style="left:{tick:.1f}%"><span>{bar:g}</span></div>
        </div>
        <div class="crit-note">{note}</div>
      </div>"""


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    bt, gate = d["backtest"], d["gate"]
    strat = [c["equity"] for c in d["curve"]]
    spy = [c["equity"] for c in d["spy_curve"]]
    months = [c["month"] for c in d["curve"]]

    vmin, vmax = 0.0, max(max(strat), max(spy)) * 1.06
    ticks = "".join(
        f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" '
        f'y1="{H - PAD_B - ((t - vmin)/(vmax - vmin))*(H - PAD_T - PAD_B):.1f}" '
        f'y2="{H - PAD_B - ((t - vmin)/(vmax - vmin))*(H - PAD_T - PAD_B):.1f}"/>'
        f'<text class="ytick" x="{PAD_L - 8}" '
        f'y="{H - PAD_B - ((t - vmin)/(vmax - vmin))*(H - PAD_T - PAD_B) + 3.5:.1f}">'
        f'${t:,.0f}</text>'
        for t in nice_ticks(vmin, vmax))
    xlab = "".join(
        f'<text class="xtick" x="{PAD_L + (i/max(len(months)-1,1))*(W-PAD_L-PAD_R):.1f}" '
        f'y="{H - 8}">{months[i][:4]}</text>'
        for i in range(0, len(months), 12))

    last_s, last_p = strat[-1], spy[-1]
    ex = "".join(f"<li><code>{k}</code> {v}</li>" for k, v in d["excluded"].items())
    rows = "".join(
        f"""<tr>
          <td class="sym">{p['sym']}</td>
          <td class="num">{p['qty']:.4f}</td>
          <td class="num">${p['mv']:.2f}</td>
          <td class="wcell"><div class="wbar"><i style="width:{100*p['mv']/sum(q['mv'] for q in d['positions']):.1f}%"></i></div></td>
          <td class="num {'up' if p['upl'] >= 0 else 'down'}">{p['upl']:+.2f}</td>
        </tr>""" for p in d["positions"])

    reb = "".join(
        f"""<tr><td class="sym">{r['context']['month']}</td>
            <td class="num">{r['context']['submitted']}</td>
            <td class="num">{r['context']['failed']}</td>
            <td class="num">${r['context']['sleeve']:,.2f}</td>
            <td><span class="chip ok">{r['reason']}</span></td></tr>"""
        for r in d["rebalances"])

    crits = (
        threshold_bar("Independent observations", bt["n_months"], 200,
                      f"{bt['n_months']} months", False,
                      "79 monthly rebalances. The minimum is 200.")
        + threshold_bar("t-statistic vs random-decile null", 2.13, 2.5,
                        "+2.13", False,
                        "Bonferroni across 3 pre-registered formations wants 2.39.")
    )

    html = f"""<title>Momentum Bot Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  --ground:#F4F6F9; --surface:#FFFFFF; --raise:#FBFCFD;
  --ink:#141821; --muted:#5C6577; --faint:#8B94A8;
  --line:#DDE2EA; --line-soft:#EAEEF3;
  --accent:#4257D4; --accent-soft:#E8EBFA;
  --pass:#17795E; --warn:#A9670A; --warn-bg:#FDF4E3; --warn-line:#E8C98A;
  --fail:#B03A34; --up:#17795E; --down:#B03A34;
  --shadow:0 1px 2px rgba(20,24,33,.05), 0 8px 24px -12px rgba(20,24,33,.14);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#10131A; --surface:#191D27; --raise:#1F2431;
    --ink:#E6E9F0; --muted:#98A1B4; --faint:#6E7789;
    --line:#262C39; --line-soft:#212734;
    --accent:#8194F0; --accent-soft:#222941;
    --pass:#4FBF98; --warn:#E0A542; --warn-bg:#2A2213; --warn-line:#5C4A22;
    --fail:#E0736B; --up:#4FBF98; --down:#E0736B;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#10131A; --surface:#191D27; --raise:#1F2431;
  --ink:#E6E9F0; --muted:#98A1B4; --faint:#6E7789;
  --line:#262C39; --line-soft:#212734;
  --accent:#8194F0; --accent-soft:#222941;
  --pass:#4FBF98; --warn:#E0A542; --warn-bg:#2A2213; --warn-line:#5C4A22;
  --fail:#E0736B; --up:#4FBF98; --down:#E0736B;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
}}
*,*::before,*::after {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1000px; margin:0 auto; padding:32px 20px 64px;
        display:flex; flex-direction:column; gap:22px; }}
h1,h2,h3 {{ font-family:Archivo,system-ui,sans-serif; margin:0; text-wrap:balance; }}
h1 {{ font-size:29px; font-weight:700; letter-spacing:-.02em; }}
h2 {{ font-size:13px; font-weight:600; letter-spacing:.10em; text-transform:uppercase;
      color:var(--faint); }}
code, .num, .sym, .mono {{ font-family:"IBM Plex Mono",ui-monospace,monospace;
                           font-variant-numeric:tabular-nums; }}

header {{ display:flex; justify-content:space-between; align-items:flex-end;
          gap:16px; flex-wrap:wrap; border-bottom:1px solid var(--line);
          padding-bottom:16px; }}
.sub {{ color:var(--muted); font-size:13.5px; margin-top:5px; }}
.chip {{ display:inline-block; font-family:"IBM Plex Mono",monospace; font-size:11px;
         font-weight:600; letter-spacing:.06em; text-transform:uppercase;
         padding:4px 9px; border-radius:3px; border:1px solid var(--line);
         color:var(--muted); background:var(--raise); }}
.chip.paper {{ color:var(--accent); border-color:var(--accent); background:var(--accent-soft); }}
.chip.ok {{ color:var(--pass); border-color:color-mix(in srgb,var(--pass) 40%,transparent); }}

.banner {{ background:var(--warn-bg); border:1px solid var(--warn-line);
           border-left:4px solid var(--warn); border-radius:5px; padding:16px 18px; }}
.banner .row {{ display:flex; align-items:center; gap:11px; flex-wrap:wrap; }}
.banner strong {{ font-family:Archivo,sans-serif; font-size:16px; color:var(--warn); }}
.banner p {{ margin:7px 0 0; font-size:14px; color:var(--ink); }}
.banner .why {{ font-family:"IBM Plex Mono",monospace; font-size:13px; color:var(--warn); }}

.card {{ background:var(--surface); border:1px solid var(--line); border-radius:6px;
         padding:18px 20px; box-shadow:var(--shadow); }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
          background:var(--line); border:1px solid var(--line); border-radius:6px;
          overflow:hidden; }}
.stat {{ background:var(--surface); padding:15px 17px; }}
.stat .k {{ font-size:11px; letter-spacing:.07em; text-transform:uppercase;
            color:var(--faint); font-weight:600; }}
.stat .v {{ font-family:Archivo,sans-serif; font-size:25px; font-weight:600;
            letter-spacing:-.02em; margin-top:3px; font-variant-numeric:tabular-nums; }}
.stat .d {{ font-size:12.5px; color:var(--muted); margin-top:1px; }}

.crit {{ padding:13px 0; border-bottom:1px solid var(--line-soft); }}
.crit:last-child {{ border-bottom:0; padding-bottom:2px; }}
.crit-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; }}
.crit-label {{ font-size:14px; font-weight:500; }}
.crit-val {{ font-family:"IBM Plex Mono",monospace; font-weight:600; font-size:14px;
             font-variant-numeric:tabular-nums; }}
.crit.fail .crit-val {{ color:var(--fail); }}
.crit.pass .crit-val {{ color:var(--pass); }}
.track {{ position:relative; height:9px; background:var(--line-soft);
          border-radius:2px; margin:9px 0 7px; }}
.fill {{ height:100%; border-radius:2px; background:var(--fail); opacity:.75; }}
.crit.pass .fill {{ background:var(--pass); }}
.tick {{ position:absolute; top:-5px; bottom:-5px; width:2px; background:var(--ink); }}
.tick span {{ position:absolute; top:-17px; left:50%; transform:translateX(-50%);
              font-family:"IBM Plex Mono",monospace; font-size:10.5px;
              font-weight:600; color:var(--ink); }}
.crit-note {{ font-size:12.5px; color:var(--muted); }}

svg {{ display:block; width:100%; height:auto; }}
.grid {{ stroke:var(--line-soft); stroke-width:1; }}
.ytick,.xtick {{ fill:var(--faint); font-family:"IBM Plex Mono",monospace; font-size:10.5px; }}
.ytick {{ text-anchor:end; }} .xtick {{ text-anchor:middle; }}
.s-line {{ fill:none; stroke:var(--accent); stroke-width:2; stroke-linejoin:round; }}
.s-area {{ fill:var(--accent); opacity:.09; }}
.b-line {{ fill:none; stroke:var(--faint); stroke-width:1.5; stroke-dasharray:4 3; }}
.dot {{ fill:var(--accent); stroke:var(--surface); stroke-width:2.5; }}
.legend {{ display:flex; gap:18px; font-size:13px; color:var(--muted); margin-top:10px;
           flex-wrap:wrap; }}
.legend i {{ display:inline-block; width:16px; height:2.5px; vertical-align:middle;
             margin-right:7px; border-radius:2px; }}

.tw {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ text-align:left; font-size:11px; letter-spacing:.07em; text-transform:uppercase;
      color:var(--faint); font-weight:600; padding:0 10px 9px 0;
      border-bottom:1px solid var(--line); white-space:nowrap; }}
td {{ padding:9px 10px 9px 0; border-bottom:1px solid var(--line-soft); }}
tr:last-child td {{ border-bottom:0; }}
th.num,td.num {{ text-align:right; }}
.sym {{ font-weight:600; }}
.up {{ color:var(--up); }} .down {{ color:var(--down); }}
.wcell {{ width:34%; min-width:110px; }}
.wbar {{ height:7px; background:var(--line-soft); border-radius:2px; overflow:hidden; }}
.wbar i {{ display:block; height:100%; background:var(--accent); opacity:.55; }}
ul.notes {{ margin:0; padding-left:18px; color:var(--muted); font-size:13.5px; }}
ul.notes li {{ margin:5px 0; }}
ul.notes code {{ color:var(--ink); font-weight:600; font-size:13px; }}
footer {{ color:var(--faint); font-size:12.5px; border-top:1px solid var(--line);
          padding-top:14px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
@media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} h1 {{ font-size:24px; }} }}
</style>

<div class="wrap">
  <header>
    <div>
      <h1>Momentum Bot Monitor</h1>
      <div class="sub">12&#8211;2 cross-sectional momentum &middot; {d['universe']} US large caps
        &middot; formation {d['formation']['from']} &rarr; {d['formation']['to']}</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <span class="chip paper">Alpaca Paper</span>
      <span class="chip">{d['generated']}</span>
    </div>
  </header>

  <div class="banner">
    <div class="row">
      <strong>Not validated</strong>
      <span class="why">{gate['name']}</span>
    </div>
    <p>This strategy has <em>not</em> earned the right to trade real money. The bot
      checks this before every rebalance and would refuse a live broker outright.
      It runs in paper because paper is where unvalidated strategies belong.</p>
  </div>

  <section class="card">
    <h2>Why it is refused</h2>
    <div style="margin-top:12px">{crits}</div>
    <div class="crit pass" style="border-top:1px solid var(--line-soft);padding-top:13px">
      <div class="crit-head"><span class="crit-label">Passing already</span>
        <span class="crit-val">3 of 5</span></div>
      <div class="crit-note" style="margin-top:5px">Mean excess is positive
        (+106bp/month) &middot; holdout keeps its sign (+0.88) &middot; survives the
        concentration test (t 2.13 &rarr; 1.86 dropping the best month).</div>
    </div>
  </section>

  <div class="stats">
    <div class="stat"><div class="k">Backtest</div><div class="v">+{bt['ann']}%</div>
      <div class="d">per year, {bt['n_months']} months</div></div>
    <div class="stat"><div class="k">SPY, same window</div><div class="v">+{bt['spy_ann']}%</div>
      <div class="d">per year</div></div>
    <div class="stat"><div class="k">Sharpe</div><div class="v">{bt['sharpe']}</div>
      <div class="d">monthly, annualised</div></div>
    <div class="stat"><div class="k">Deployed</div>
      <div class="v">${sum(p['mv'] for p in d['positions']):,.0f}</div>
      <div class="d">{len(d['positions'])} positions, 98% of sleeve</div></div>
  </div>

  <section class="card">
    <h2>Backtest equity &middot; $1,000 start</h2>
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Backtest equity curve against SPY">
      {ticks}
      <path class="s-area" d="{area_for(strat, vmin, vmax)}"/>
      <path class="b-line" d="{path_for(spy, vmin, vmax)}"/>
      <path class="s-line" d="{path_for(strat, vmin, vmax)}"/>
      <circle class="dot" r="4"
        cx="{PAD_L + (W - PAD_L - PAD_R):.1f}"
        cy="{H - PAD_B - ((strat[-1]-vmin)/(vmax-vmin))*(H-PAD_T-PAD_B):.1f}"/>
      {xlab}
    </svg>
    <div class="legend">
      <span><i style="background:var(--accent)"></i>Momentum decile &mdash;
        <strong class="mono">${last_s:,.0f}</strong></span>
      <span><i style="background:var(--faint)"></i>SPY &mdash;
        <strong class="mono">${last_p:,.0f}</strong></span>
    </div>
    <ul class="notes" style="margin-top:12px">
      <li>Levels are <strong>overstated</strong>: all 100 names in the universe survived
        to today, so the backtest carries roughly 100&#37; survivorship bias. CRSP fixes this.</li>
    </ul>
  </section>

  <div class="grid2">
    <section class="card">
      <h2>Holdings</h2>
      <div class="tw"><table>
        <tr><th>Symbol</th><th class="num">Qty</th><th class="num">Value</th>
            <th>Weight</th><th class="num">P&amp;L</th></tr>
        {rows}
      </table></div>
    </section>

    <section class="card">
      <h2>Data integrity</h2>
      <ul class="notes" style="margin-top:12px">
        <li>Price refresh runs on the <strong>SIP</strong> consolidated tape and matches
          the original history to <strong>0.00bp</strong> median over 34,056 overlapping days.</li>
        <li>The refresh verifies before it writes and <strong>refuses</strong> on
          disagreement. Two names are excluded:</li>
      </ul>
      <ul class="notes">{ex}</ul>
      <ul class="notes">
        <li>A spinoff is not a split, so split-adjusted history carries a phantom
          one-day drop. Refreshing blindly would have injected a fake &minus;17.8&#37;
          into FDX and dropped it for a reason that never happened.</li>
      </ul>
    </section>
  </div>

  <section class="card">
    <h2>Rebalance log</h2>
    <div class="tw"><table>
      <tr><th>Month</th><th class="num">Orders</th><th class="num">Failed</th>
          <th class="num">Sleeve</th><th>Status</th></tr>
      {reb}
    </table></div>
    <ul class="notes" style="margin-top:12px">
      <li>Scheduled weekdays 15:40 ET. The script decides whether a rebalance is due,
        so holidays, weekends and a sleeping laptop all resolve themselves.</li>
      <li>A month counts as done only on a <strong>clean</strong> run &mdash; a partial
        failure would otherwise strand a position the strategy had already dropped.</li>
    </ul>
  </section>

  <footer>
    Generated from <code>data/dashboard.json</code> &middot;
    <code>python make_dashboard.py</code> &middot; paper trading only, live execution
    is refused in code rather than by configuration.
  </footer>
</div>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
