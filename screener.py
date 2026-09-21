#!/usr/bin/env python3
"""PV/MA buy/sell daily screener.

Scans WATCHLIST, fits a Student-t tail parameter (nu) to recent returns to
gauge how reliable the price/volume signal is for each name, and renders the
result as a static index.html report (no notebook / display required).
"""
import base64
import html
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t as tdist

warnings.filterwarnings("ignore")

# ============================================================
# WATCHLIST -- edit this list freely
# ============================================================
WATCHLIST = [
    # SGX
    "AWX.SI",    # AEM Holdings
    "H02.SI",    # Haw Par
    "BS6.SI",    # YZJ Shipbuilding
    "O39.SI",    # OCBC Bank
    "MZH.SI",    # Nanofilm
    "BEC.SI",    # BEC World
    "1082.KL",   # Heng Huat

    # US
    "MSTR",      # Strategy (MicroStrategy)
    "KO",        # Coca-Cola

    # Add more below:
    # "AAPL",
    # "TSLA",
    # "D05.SI",  # DBS
]

# ============================================================
# SIGNAL CONFIG -- tune these
# ============================================================
NU_WINDOW       = 20     # days to fit nu
MA_WINDOW       = 60     # price MA window
VOL_MA_WINDOW   = 60     # volume MA window
MA_THRESH_BUY   = -15    # % below MA -> buy zone
MA_THRESH_SELL  = 30     # % above MA -> sell zone
VOL_THRESH      = 100    # % above vol MA -> spike
FROTH_LOOKBACK  = 20     # days to count sell signals
FROTH_THRESH    = 3      # sell signals to qualify as frothy
LOOKBACK_DAYS   = 250    # trading days of history to download

# NU gate thresholds
NU_HIGH         = 7      # nu < this -> HIGH reliability
NU_MED          = 15     # nu < this -> MEDIUM
NU_LOW          = 25     # nu < this -> LOW, else SKIP

OUTPUT_FILE     = "index.html"
SGT             = ZoneInfo("Asia/Singapore")

# ============================================================
# Palette (validated categorical/status set -- see dataviz skill)
# ============================================================
COLORS = {
    "surface":     "#fcfcfb",
    "surface_dark": "#1a1a19",
    "page":        "#f9f9f7",
    "page_dark":   "#0d0d0d",
    "ink":         "#0b0b0b",
    "ink_dark":    "#ffffff",
    "ink2":        "#52514e",
    "ink2_dark":   "#c3c2b7",
    "muted":       "#898781",
    "grid":        "#e1e0d9",
    "grid_dark":   "#2c2c2a",
    "baseline":    "#c3c2b7",
    "baseline_dark": "#383835",
    "border":      "rgba(11,11,11,0.10)",
    "border_dark": "rgba(255,255,255,0.10)",
    "good":        "#0ca30c",
    "warning":     "#fab219",
    "serious":     "#ec835a",
    "critical":    "#d03b3b",
    "blue":        "#2a78d6",
}


def fit_nu(returns):
    try:
        nu, loc, scale = tdist.fit(returns)
        return min(nu, 100), loc, scale
    except Exception:
        return 100.0, 0.0, 1.0


def download_watchlist(watchlist, start_date):
    data, failed = {}, []
    for ticker in watchlist:
        try:
            df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < max(NU_WINDOW, MA_WINDOW, VOL_MA_WINDOW) + 10:
                failed.append((ticker, f"only {len(df)} days"))
                continue
            df["ret"] = df["Close"].pct_change()
            df.dropna(subset=["ret"], inplace=True)
            data[ticker] = df
        except Exception as e:
            failed.append((ticker, str(e)[:60]))
    return data, failed


def scan_ticker(ticker, df):
    close = df["Close"]
    volume = df["Volume"]

    recent_rets = df["ret"].iloc[-NU_WINDOW:].values
    nu_now, _, _ = fit_nu(recent_rets)

    nu_vals = []
    for i in range(NU_WINDOW, min(len(df), NU_WINDOW + LOOKBACK_DAYS)):
        nu_i, _, _ = fit_nu(df["ret"].iloc[i - NU_WINDOW:i].values)
        nu_vals.append(nu_i)
    nu_median = float(np.median(nu_vals)) if nu_vals else 100.0

    if nu_now < NU_HIGH:
        reliability, rel_label = "★★★", "HIGH"
    elif nu_now < NU_MED:
        reliability, rel_label = "★★☆", "MED"
    elif nu_now < NU_LOW:
        reliability, rel_label = "★☆☆", "LOW"
    else:
        reliability, rel_label = "☆☆☆", "SKIP"

    price_ma = close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    ma_dist_now = ((close.iloc[-1] / price_ma.iloc[-1]) - 1) * 100 if not np.isnan(price_ma.iloc[-1]) else 0.0

    vol_ma = volume.rolling(VOL_MA_WINDOW, min_periods=VOL_MA_WINDOW).mean()
    vol_dist_now = ((volume.iloc[-1] / vol_ma.iloc[-1]) - 1) * 100 if not np.isnan(vol_ma.iloc[-1]) else 0.0

    sell_count = 0
    for i in range(max(0, len(df) - FROTH_LOOKBACK), len(df)):
        window_rets = df["ret"].iloc[max(0, i - NU_WINDOW):i].values
        if len(window_rets) < NU_WINDOW:
            continue
        nu_i, loc_i, scale_i = fit_nu(window_rets)
        try:
            t_hi = tdist.ppf(0.95, nu_i, loc=loc_i, scale=scale_i)
            if df["ret"].iloc[i] > t_hi:
                sell_count += 1
        except Exception:
            pass

    frothy = sell_count >= FROTH_THRESH

    signals = []
    if ma_dist_now <= MA_THRESH_BUY:
        if vol_dist_now >= VOL_THRESH:
            signals.append("PV BUY")
        signals.append("MA BUY")

    if frothy and ma_dist_now >= MA_THRESH_SELL and vol_dist_now >= VOL_THRESH:
        signals.append("V3 SELL")
    elif ma_dist_now >= MA_THRESH_SELL:
        signals.append("MA EXTENDED")
    elif frothy:
        signals.append("FROTHY")

    return {
        "ticker": ticker,
        "close": float(close.iloc[-1]),
        "nu_now": float(nu_now),
        "nu_median": nu_median,
        "reliability": reliability,
        "rel_label": rel_label,
        "ma_dist": float(ma_dist_now),
        "vol_dist": float(vol_dist_now),
        "sell_count": int(sell_count),
        "frothy": bool(frothy),
        "signals": signals,
    }


def action_for(signals):
    if "PV BUY" in signals:
        return "HIGH CONVICTION BUY (price + volume)", "good"
    if "MA BUY" in signals:
        return "BUY (price below MA)", "good"
    if "V3 SELL" in signals:
        return "TRIM 20-30% (frothy + extended + volume)", "critical"
    if "MA EXTENDED" in signals:
        return "WATCH (extended but not frothy)", "warning"
    if "FROTHY" in signals:
        return "WATCH (frothy but not extended)", "warning"
    return "", "muted"


def rel_color(rel_label):
    return {"HIGH": "good", "MED": "warning", "LOW": "serious", "SKIP": "muted"}[rel_label]


# ============================================================
# SVG charts (hand-rolled, native <title> tooltips, no JS)
# ============================================================
def svg_scatter(rows, width=640, height=400):
    pad_l, pad_r, pad_t, pad_b = 56, 24, 24, 40
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    xs = [r["nu_now"] for r in rows]
    ys = [r["ma_dist"] for r in rows]
    x_min, x_max = 0, max([NU_LOW + 10] + xs) * 1.05
    y_min = min([MA_THRESH_BUY - 10] + ys)
    y_max = max([MA_THRESH_SELL + 10] + ys)

    def sx(v):
        return pad_l + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v):
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart-svg" role="img" aria-label="Nu vs moving-average distance signal map">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="var(--surface-1)" />')

    # gridlines (y)
    for gv in np.linspace(y_min, y_max, 6):
        y = sy(gv)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1" />')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 3:.1f}" text-anchor="end" class="axis-label">{gv:.0f}%</text>')

    for gv in np.linspace(x_min, x_max, 6):
        x = sx(gv)
        parts.append(f'<text x="{x:.1f}" y="{height - pad_b + 16}" text-anchor="middle" class="axis-label">{gv:.0f}</text>')

    # threshold reference lines
    parts.append(f'<line x1="{pad_l}" y1="{sy(MA_THRESH_BUY):.1f}" x2="{width - pad_r}" y2="{sy(MA_THRESH_BUY):.1f}" stroke="var(--good)" stroke-width="1.5" stroke-dasharray="5,4" />')
    parts.append(f'<line x1="{pad_l}" y1="{sy(MA_THRESH_SELL):.1f}" x2="{width - pad_r}" y2="{sy(MA_THRESH_SELL):.1f}" stroke="var(--critical)" stroke-width="1.5" stroke-dasharray="5,4" />')
    parts.append(f'<line x1="{sx(NU_MED):.1f}" y1="{pad_t}" x2="{sx(NU_MED):.1f}" y2="{height - pad_b}" stroke="var(--warning)" stroke-width="1.5" stroke-dasharray="5,4" />')
    if y_min < 0 < y_max:
        parts.append(f'<line x1="{pad_l}" y1="{sy(0):.1f}" x2="{width - pad_r}" y2="{sy(0):.1f}" stroke="var(--baseline)" stroke-width="1" />')

    # axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="var(--baseline)" stroke-width="1" />')
    parts.append(f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="var(--baseline)" stroke-width="1" />')
    parts.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 4}" text-anchor="middle" class="axis-title">Current nu (lower = fatter tails = more reliable)</text>')
    parts.append(f'<text x="14" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" class="axis-title" transform="rotate(-90 14 {pad_t + plot_h / 2:.1f})">% from {MA_WINDOW}d MA</text>')

    for r in rows:
        cx, cy = sx(r["nu_now"]), sy(r["ma_dist"])
        if r["ma_dist"] <= MA_THRESH_BUY:
            zone = "good"
        elif r["ma_dist"] >= MA_THRESH_SELL:
            zone = "critical"
        else:
            zone = "muted"
        title = (f'{r["ticker"]}: nu={r["nu_now"]:.1f}, MA={r["ma_dist"]:+.0f}%, '
                  f'Vol={r["vol_dist"]:+.0f}%, signals={", ".join(r["signals"]) or "none"}')
        near_right = cx > width - pad_r - 44
        label_x = cx - 10 if near_right else cx + 10
        anchor = "end" if near_right else "start"
        parts.append(f'<g class="point">'
                      f'<title>{html.escape(title)}</title>'
                      f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="var(--{zone})" stroke="var(--surface-1)" stroke-width="1.5" />'
                      f'<text x="{label_x:.1f}" y="{cy - 8:.1f}" text-anchor="{anchor}" class="point-label">{html.escape(r["ticker"])}</text>'
                      f'</g>')

    parts.append("</svg>")
    return "".join(parts)


def svg_nu_bars(rows, width=640):
    sorted_rows = sorted(rows, key=lambda r: r["nu_now"])
    row_h = 30
    pad_l, pad_r, pad_t, pad_b = 90, 60, 30, 30
    height = pad_t + pad_b + row_h * len(sorted_rows)
    plot_w = width - pad_l - pad_r
    x_max = max([NU_LOW + 10] + [r["nu_now"] for r in sorted_rows])

    def sx(v):
        return pad_l + v / x_max * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart-svg" role="img" aria-label="Nu environment by ticker">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="var(--surface-1)" />')

    thresholds = [(NU_HIGH, "good", "HIGH"), (NU_MED, "warning", "MED")]
    close_labels = abs(sx(NU_HIGH) - sx(NU_MED)) < 60
    for idx, (gv, color, label) in enumerate(thresholds):
        x = sx(gv)
        label_y = pad_t - 10 - (12 if close_labels and idx == 1 else 0)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t - 6}" x2="{x:.1f}" y2="{height - pad_b}" stroke="var(--{color})" stroke-width="1.5" stroke-dasharray="4,3" />')
        parts.append(f'<text x="{x:.1f}" y="{label_y}" text-anchor="middle" class="axis-label">{label} &lt;{gv}</text>')

    for i, r in enumerate(sorted_rows):
        y = pad_t + i * row_h
        bar_w = sx(r["nu_now"]) - pad_l
        color = rel_color(r["rel_label"])
        title = f'{r["ticker"]}: nu={r["nu_now"]:.1f} ({r["rel_label"]})'
        parts.append(f'<text x="{pad_l - 10}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end" class="axis-label">{html.escape(r["ticker"])}</text>')
        parts.append(f'<g><title>{html.escape(title)}</title>'
                      f'<rect x="{pad_l}" y="{y + 5}" width="{max(bar_w, 1):.1f}" height="{row_h - 10}" fill="var(--{color})" rx="3" />'
                      f'</g>')
        parts.append(f'<text x="{sx(r["nu_now"]) + 8:.1f}" y="{y + row_h / 2 + 4:.1f}" class="axis-label">{r["nu_now"]:.1f}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ============================================================
# HTML report
# ============================================================
CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --good: #0ca30c;
  --warning: #d68500;
  --serious: #ec835a;
  --critical: #d03b3b;
  --blue: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --good: #0ca30c;
    --warning: #fab219;
    --serious: #ec835a;
    --critical: #e66767;
    --blue: #3987e5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 24px 16px 64px;
}
.wrap { max-width: 920px; margin: 0 auto; }
header h1 { font-size: 1.4rem; margin: 0 0 4px; }
header p { color: var(--text-secondary); margin: 0 0 4px; font-size: 0.9rem; }
section { margin-top: 32px; }
h2 { font-size: 1.05rem; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  margin-bottom: 10px;
}
.card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.ticker { font-weight: 700; font-size: 1.05rem; }
.badge {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  padding: 2px 8px;
  border-radius: 999px;
  color: #fff;
}
.badge.good { background: var(--good); }
.badge.warning { background: var(--warning); }
.badge.critical { background: var(--critical); }
.badge.muted { background: var(--muted); }
.metrics { color: var(--text-secondary); font-size: 0.85rem; margin-top: 6px; font-variant-numeric: tabular-nums; }
.action { margin-top: 6px; font-weight: 600; }
.action.good { color: var(--good); }
.action.warning { color: var(--serious); }
.action.critical { color: var(--critical); }
.note { color: var(--text-secondary); font-size: 0.85rem; }
.chart-svg { width: 100%; height: auto; display: block; }
.axis-label { font-size: 10px; fill: var(--text-secondary); }
.axis-title { font-size: 11px; fill: var(--text-secondary); }
.point-label { font-size: 10px; fill: var(--text-primary); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.8rem; color: var(--text-secondary); margin-top: 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.swatch.good { background: var(--good); }
.swatch.warning { background: var(--warning); }
.swatch.serious { background: var(--serious); }
.swatch.critical { background: var(--critical); }
.swatch.muted { background: var(--muted); }
footer { margin-top: 40px; color: var(--muted); font-size: 0.78rem; border-top: 1px solid var(--border); padding-top: 12px; }
"""


def build_html(scan_rows, failed, config):
    now_sgt = datetime.now(SGT)
    rows_by_ticker = {r["ticker"]: r for r in scan_rows}
    active = [r for r in scan_rows if r["signals"]]

    active_html = []
    if not active:
        active_html.append('<p class="note">No signals today.</p>')
    else:
        order = {"PV BUY": 0, "MA BUY": 1, "V3 SELL": 2, "MA EXTENDED": 3, "FROTHY": 4}
        active_sorted = sorted(active, key=lambda r: min(order.get(s, 9) for s in r["signals"]))
        for r in active_sorted:
            action_text, action_tone = action_for(r["signals"])
            gate_warn = ""
            if r["nu_now"] >= NU_MED:
                gate_warn = f'<p class="note">nu={r["nu_now"]:.0f} &gt; {NU_MED} -- signal unreliable for this stock type</p>'
            active_html.append(f"""
<div class="card">
  <div class="card-head">
    <span class="ticker">{html.escape(r['ticker'])}</span>
    <span class="badge {rel_color(r['rel_label'])}">{r['reliability']} {r['rel_label']}</span>
  </div>
  <div class="metrics">
    ${r['close']:.2f} &middot; nu={r['nu_now']:.1f} &middot; MA={r['ma_dist']:+.0f}% &middot;
    Vol={r['vol_dist']:+.0f}% &middot; Sells(20d)={r['sell_count']}
  </div>
  <div class="metrics">{html.escape(" + ".join(r['signals']))}</div>
  <div class="action {action_tone}">&rarr; {html.escape(action_text)}</div>
  {gate_warn}
</div>""")

    full_rows = []
    for r in sorted(scan_rows, key=lambda r: r["ticker"]):
        sig = ", ".join(r["signals"]) if r["signals"] else "&mdash;"
        full_rows.append(f"""<tr>
<td>{html.escape(r['ticker'])}</td><td>{r['close']:.2f}</td><td>{r['nu_now']:.1f}</td>
<td>{r['nu_median']:.1f}</td><td>{r['reliability']} {r['rel_label']}</td>
<td>{r['ma_dist']:+.0f}%</td><td>{r['vol_dist']:+.0f}%</td><td>{r['sell_count']}</td>
<td>{sig}</td></tr>""")

    alerts = []
    for r in scan_rows:
        if MA_THRESH_BUY < r["ma_dist"] <= MA_THRESH_BUY + 5:
            alerts.append(f"{r['ticker']}: MA={r['ma_dist']:+.0f}% &rarr; {abs(r['ma_dist'] - MA_THRESH_BUY):.0f}pp from buy zone")
        if MA_THRESH_SELL - 10 <= r["ma_dist"] < MA_THRESH_SELL:
            alerts.append(f"{r['ticker']}: MA={r['ma_dist']:+.0f}% &rarr; {abs(MA_THRESH_SELL - r['ma_dist']):.0f}pp from sell zone")
    alerts_html = "".join(f"<li>{a}</li>" for a in alerts) if alerts else '<li class="note">No stocks near thresholds.</li>'

    failed_html = ""
    if failed:
        items = ", ".join(f"{html.escape(t)} ({html.escape(reason)})" for t, reason in failed)
        failed_html = f'<p class="note">Skipped: {items}</p>'

    scatter_svg = svg_scatter(scan_rows)
    bars_svg = svg_nu_bars(scan_rows)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PV/MA Daily Screener</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<h1>PV/MA Buy/Sell Daily Screener</h1>
<p>Generated {now_sgt.strftime('%Y-%m-%d %H:%M')} SGT &middot; {len(scan_rows)} of {len(WATCHLIST)} tickers scanned</p>
<p>nu window {NU_WINDOW}d &middot; MA {MA_WINDOW}d &middot; Vol MA {VOL_MA_WINDOW}d &middot; Buy zone &lt;{MA_THRESH_BUY}% below MA &middot; Vol spike &gt;{VOL_THRESH}%</p>
{failed_html}
</header>

<section>
<h2>Active Signals</h2>
{''.join(active_html)}
</section>

<section>
<h2>Full Watchlist Status</h2>
<table>
<thead><tr><th>Ticker</th><th>Close</th><th>nu now</th><th>nu med</th><th>Reliability</th><th>MA%</th><th>Vol%</th><th>Sells</th><th>Signal</th></tr></thead>
<tbody>{''.join(full_rows)}</tbody>
</table>
</section>

<section>
<h2>Nu Environment &mdash; which stocks to trust signals on</h2>
{bars_svg}
<div class="legend">
<span><span class="swatch good"></span>HIGH (nu&lt;{NU_HIGH})</span>
<span><span class="swatch warning"></span>MED (nu&lt;{NU_MED})</span>
<span><span class="swatch serious"></span>LOW (nu&lt;{NU_LOW})</span>
<span><span class="swatch muted"></span>SKIP</span>
</div>
</section>

<section>
<h2>Signal Map &mdash; nu vs price position</h2>
{scatter_svg}
<div class="legend">
<span><span class="swatch good"></span>Buy zone (&lt;{MA_THRESH_BUY}% of MA)</span>
<span><span class="swatch critical"></span>Sell zone (&gt;{MA_THRESH_SELL}% of MA)</span>
<span><span class="swatch muted"></span>Neutral</span>
</div>
</section>

<section>
<h2>Proximity Alerts &mdash; approaching signal thresholds</h2>
<ul>{alerts_html}</ul>
</section>

<footer>
Not investment advice. Edit WATCHLIST in screener.py to add/remove tickers.
</footer>
</div>
</body>
</html>
"""


def main():
    start_date = (datetime.now() - timedelta(days=int(LOOKBACK_DAYS * 1.6))).strftime("%Y-%m-%d")
    data, failed = download_watchlist(WATCHLIST, start_date)

    scan_rows = [scan_ticker(ticker, df) for ticker, df in data.items() if ticker in WATCHLIST]
    scan_rows = [r for t in WATCHLIST for r in scan_rows if r["ticker"] == t]

    out = build_html(scan_rows, failed, config={})
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {OUTPUT_FILE} with {len(scan_rows)} tickers scanned, {len(failed)} skipped.")


if __name__ == "__main__":
    main()
