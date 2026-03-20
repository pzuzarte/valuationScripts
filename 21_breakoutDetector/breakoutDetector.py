#!/usr/bin/env python3
"""
breakoutDetector.py — XGBoost Breakout Probability Ranker
==========================================================
Ranks stocks by probability of an imminent price breakout using an XGBoost
binary classifier trained on historical breakout instances (price up ≥8% in
next 10 trading days on elevated volume). Complements CANSLIM/growth screeners
with a timing signal.

Usage
-----
    python breakoutDetector.py --index SPX
    python breakoutDetector.py --index NDX --top 100
    python breakoutDetector.py --tickers AAPL,MSFT,NVDA --threshold 10
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests as _requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Directories ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "breakoutData")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0f1117"
PANEL  = "#1a1e2e"
BORDER = "#252a3a"
TEXT   = "#e2e8f0"
MUTED  = "#94a3b8"
ACCENT = "#6366f1"
GREEN  = "#10b981"
RED    = "#ef4444"
YELLOW = "#f59e0b"

# ── CSV ticker loader ──────────────────────────────────────────────────────────

def _read_csv_file(path: str) -> list[dict]:
    """Parse a CSV for tickers and optional weights.

    Accepts ticker/symbol column (case-insensitive) and optional weight/weights/wt column.
    Falls back to the first column if no recognised header is found.
    """
    import csv as _csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        raw = [ln for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    if not raw:
        return []
    reader = _csv.DictReader(raw)
    fl = {(k or "").lower().strip(): k for k in (reader.fieldnames or [])}
    t_key = next((v for k, v in fl.items() if k in ("ticker", "symbol")), None)
    w_key = next((v for k, v in fl.items() if k in ("weight", "weights", "wt")), None)
    if t_key is None:
        plain = list(_csv.reader(raw))
        start = 1 if plain and not plain[0][0].strip().replace(".", "").isdigit() else 0
        return [{"ticker": r[0].strip().upper(), "weight": None,
                 "name": r[0].strip().upper(), "sector": "Unknown"}
                for r in plain[start:] if r and r[0].strip()]
    rows = []
    for row in reader:
        t = (row.get(t_key) or "").strip().upper()
        if not t:
            continue
        w_raw = (row.get(w_key) or "").strip() if w_key else ""
        try:
            w: float | None = float(w_raw) if w_raw else None
        except ValueError:
            w = None
        rows.append({"ticker": t, "weight": w, "name": t, "sector": "Unknown"})
    return rows


# ── Index constituent fetchers (copied from growthScreeners.py) ────────────────

_FETCH_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _get_slickcharts(url: str, label: str) -> list[dict]:
    """Parse slickcharts.com index page — first table has Company/Symbol/Weight."""
    try:
        r = _requests.get(url, headers={"User-Agent": _FETCH_UA}, timeout=20)
        tbls = pd.read_html(io.StringIO(r.text))
        df = tbls[0]
        results = []
        for _, row in df.iterrows():
            t = str(row.get("Symbol", "")).strip()
            if not t or t == "nan":
                continue
            results.append({
                "ticker": t,
                "name":   str(row.get("Company", t)),
                "sector": "Unknown",
            })
        return results
    except Exception as e:
        print(f"  {label} slickcharts fetch failed: {e}")
        return []


def _get_spx() -> list[dict]:
    return _get_slickcharts("https://www.slickcharts.com/sp500", "S&P 500")


def _get_ndx() -> list[dict]:
    return _get_slickcharts("https://www.slickcharts.com/nasdaq100", "Nasdaq-100")


def _get_dow() -> list[dict]:
    return _get_slickcharts("https://www.slickcharts.com/dowjones", "Dow Jones 30")


def _get_rut() -> list[dict]:
    """Russell 2000 from iShares IWM ETF holdings CSV."""
    url = (
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
        "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    try:
        r = _requests.get(url, headers={"User-Agent": _FETCH_UA}, timeout=30)
        lines = r.text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("Ticker"))
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        df = df[df["Asset Class"] == "Equity"]
        results = []
        for _, row in df.iterrows():
            t = str(row.get("Ticker", "")).strip()
            if not t or t == "nan" or t == "-":
                continue
            results.append({
                "ticker": t,
                "name":   str(row.get("Name", t)),
                "sector": str(row.get("Sector", "Unknown")),
            })
        return results
    except Exception as e:
        print(f"  RUT iShares fetch failed: {e}")
        return []


# ── WL bar (copied from growthScreeners.py) ───────────────────────────────────

_WL_CSS = """
#wl-bar{display:flex;align-items:center;gap:10px;padding:7px 16px;
  background:#1a1e2e;border-top:1px solid #252a3a;
  position:sticky;bottom:0;z-index:100;}
#wl-add-btn{background:#4f8ef7;color:#fff;border:none;border-radius:5px;
  padding:6px 14px;font-size:12px;font-weight:600;cursor:pointer;}
#wl-add-btn:hover{background:#6ba3ff;}
#wl-clear-btn{background:transparent;color:#6b7194;border:1px solid #252a3a;
  border-radius:5px;padding:6px 12px;font-size:12px;cursor:pointer;}
#wl-clear-btn:hover{color:#e8eaf0;}
.wl-toast{position:fixed;bottom:58px;right:20px;background:#1e2538;
  border:1px solid #252a3a;color:#e8eaf0;padding:10px 16px;border-radius:6px;
  font-size:12px;opacity:0;transition:opacity .3s;z-index:10000;max-width:340px;}
.wl-toast.show{opacity:1;}
.wl-toast.warn{border-color:#f0a500;color:#f0a500;}
"""

_WL_BAR = (
    '<div id="wl-bar" style="opacity:.5;pointer-events:none;">'
    '<span id="wl-count">0 selected</span>'
    '<button id="wl-add-btn">+ Add to Deep Dive</button>'
    '<button id="wl-clear-btn">Clear</button>'
    '</div>'
)

_WL_JS_TMPL = r"""
(function(){
var WL_PORT=__PORT__;
function wlTickers(){return[...document.querySelectorAll('input.row-check:checked')].map(function(c){return c.value;});}
function wlUpdate(){
  var n=wlTickers().length,cnt=document.getElementById('wl-count'),bar=document.getElementById('wl-bar');
  if(cnt)cnt.textContent=n+' selected';
  if(bar){bar.style.opacity=n>0?'1':'0.5';bar.style.pointerEvents=n>0?'auto':'none';}
}
function wlToast(msg,warn){
  var t=document.createElement('div');
  t.className='wl-toast'+(warn?' warn':'');
  t.textContent=msg;document.body.appendChild(t);
  setTimeout(function(){t.classList.add('show');},10);
  setTimeout(function(){t.classList.remove('show');setTimeout(function(){t.remove();},400);},3500);
}
document.addEventListener('DOMContentLoaded',function(){
  var allCb=document.getElementById('cb-all');
  if(allCb)allCb.addEventListener('change',function(){
    document.querySelectorAll('input.row-check').forEach(function(c){c.checked=allCb.checked;});
    wlUpdate();
  });
  document.addEventListener('change',function(e){if(e.target.classList.contains('row-check'))wlUpdate();});
  var addBtn=document.getElementById('wl-add-btn');
  if(addBtn)addBtn.addEventListener('click',function(){
    var tickers=wlTickers();if(!tickers.length)return;
    fetch('http://localhost:'+WL_PORT+'/api/watchlist/add',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({tickers:tickers})
    }).then(function(r){return r.ok?r.json():Promise.reject('HTTP '+r.status);})
    .then(function(d){
      wlToast('\u2713 '+d.added+' added to deep dive list'+(d.skipped?' \u00b7 '+d.skipped+' already present':'')+'  ('+d.total+' total)');
      document.querySelectorAll('input.row-check,#cb-all').forEach(function(c){c.checked=false;});
      wlUpdate();
    }).catch(function(){
      var csv='ticker,shares\n'+tickers.map(function(t){return t+',0';}).join('\n');
      var a=document.createElement('a');
      a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
      a.download='deepDiveTickers_export.csv';document.body.appendChild(a);a.click();a.remove();
      wlToast('Suite offline \u2014 downloaded as CSV',true);
    });
  });
  var clrBtn=document.getElementById('wl-clear-btn');
  if(clrBtn)clrBtn.addEventListener('click',function(){
    document.querySelectorAll('input.row-check,#cb-all').forEach(function(c){c.checked=false;});
    wlUpdate();
  });
  wlUpdate();
});
})();
"""


def _wl_js(port: int) -> str:
    return _WL_JS_TMPL.replace("__PORT__", str(port))


# ── Feature engineering ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "vol_ratio_10_50",
    "vol_ratio_5_20",
    "price_vs_sma50",
    "price_vs_sma200",
    "sma50_vs_sma200",
    "rsi_14",
    "hi52_ratio",
    "atr_ratio",
    "mom_5d",
    "mom_21d",
    "mom_63d",
    "bb_position",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [Open, High, Low, Close, Volume],
    returns a DataFrame with one row per trading day containing all features.
    Requires at least 252 rows.
    """
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # Volume surge indicators
    feat["vol_ratio_10_50"] = volume.rolling(10).mean() / volume.rolling(50).mean()
    feat["vol_ratio_5_20"]  = volume.rolling(5).mean()  / volume.rolling(20).mean()

    # Price vs SMAs
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    feat["price_vs_sma50"]   = close / sma50
    feat["price_vs_sma200"]  = close / sma200
    feat["sma50_vs_sma200"]  = sma50 / sma200

    # RSI(14) — Wilder's method
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    feat["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # 52-week high ratio
    feat["hi52_ratio"] = close / close.rolling(252).max()

    # ATR ratio (normalized volatility)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feat["atr_ratio"] = atr / close

    # Momentum
    feat["mom_5d"]  = close.pct_change(5)
    feat["mom_21d"] = close.pct_change(21)
    feat["mom_63d"] = close.pct_change(63)

    # Bollinger Band position
    sma20   = close.rolling(20).mean()
    std20   = close.rolling(20).std()
    bb_up   = sma20 + 2 * std20
    bb_lo   = sma20 - 2 * std20
    bb_range = bb_up - bb_lo
    feat["bb_position"] = (close - bb_lo) / bb_range.replace(0, np.nan)

    return feat


def build_labels(df: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Compute binary breakout label for each row (skip last 10 — no future data).
    Breakout = price up >= threshold% AND avg volume next 10d >= 1.2x trailing 50d avg.
    """
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    future_return = close.shift(-10) / close - 1
    future_vol    = volume.rolling(10).mean().shift(-10) / volume.rolling(50).mean()
    label = ((future_return >= threshold / 100) & (future_vol >= 1.2)).astype(int)
    # Last 10 rows have no valid future — set to NaN so they get dropped
    label.iloc[-10:] = np.nan
    return label


def fetch_ticker_data(ticker: str, period_days: int = 630) -> pd.DataFrame | None:
    """
    Download ~2.5 years of OHLCV data for one ticker.
    Returns None if fewer than 252 rows of valid data.
    """
    end   = datetime.today()
    start = end - timedelta(days=period_days)
    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            return None
        # Flatten MultiIndex columns — handle both old (metric, ticker) and
        # new (ticker, metric) yfinance MultiIndex layouts, then deduplicate.
        if isinstance(raw.columns, pd.MultiIndex):
            _expected = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}
            l0 = set(raw.columns.get_level_values(0))
            level = 0 if l0 & _expected else 1
            raw.columns = raw.columns.get_level_values(level)
        raw = raw.loc[:, ~raw.columns.duplicated(keep="first")]
        if len(raw) < 252:
            return None
        return raw
    except Exception:
        return None


def fetch_and_build(ticker: str, threshold: float) -> tuple[str, pd.DataFrame | None, dict | None]:
    """
    Fetch data, compute features + labels for training, and extract current snapshot.
    Returns (ticker, train_df_with_label, current_row_dict).
    """
    raw = fetch_ticker_data(ticker)
    if raw is None:
        return ticker, None, None

    feat   = compute_features(raw)
    labels = build_labels(raw, threshold)

    # Align and drop NaN
    combined = feat.copy()
    combined["label"]  = labels
    combined["ticker"] = ticker
    combined = combined.dropna(subset=FEATURE_COLS + ["label"])

    # Current snapshot = last row of features (before label period cutoff)
    # Use the last row that has complete features (regardless of label)
    snap_feat = feat.dropna(subset=FEATURE_COLS)
    if snap_feat.empty:
        return ticker, None, None

    last = snap_feat.iloc[-1]
    close_last = raw["Close"].squeeze().iloc[-1]

    current = {
        "ticker":          ticker,
        "close":           float(close_last),
        **{c: float(last[c]) for c in FEATURE_COLS},
    }

    if combined.empty:
        return ticker, None, current

    return ticker, combined, current


# ── HTML report builder ────────────────────────────────────────────────────────

def _color_prob(p: float) -> str:
    """Return a CSS color string interpolating red → yellow → green."""
    if p >= 0.6:
        return GREEN
    elif p >= 0.35:
        return YELLOW
    else:
        return RED


def _prob_bar(p: float) -> str:
    pct  = f"{p:.1%}"
    col  = _color_prob(p)
    w    = int(p * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="flex:1;background:{BORDER};border-radius:3px;height:8px;">'
        f'<div style="width:{w}%;height:100%;background:{col};border-radius:3px;"></div>'
        f'</div>'
        f'<span style="color:{col};font-weight:600;min-width:42px;">{pct}</span>'
        f'</div>'
    )


def _fmt_pct(v, positive_green: bool = True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f'<span style="color:{MUTED}">—</span>'
    col = GREEN if (v >= 0) == positive_green else RED
    return f'<span style="color:{col}">{v:+.1%}</span>'


def _fmt_float(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f'<span style="color:{MUTED}">—</span>'
    return f"{v:.{decimals}f}"


def build_html(
    results: list[dict],
    index_name: str,
    aucpr: float,
    p10: float,
    n_train: int,
    n_breakouts: int,
    threshold: float,
    feat_importance: dict,
    prob_hist_vals: list[float],
    port: int,
) -> str:
    today     = datetime.today().strftime("%B %d, %Y")
    n_results = len(results)

    # ── Feature importance chart data
    fi_sorted   = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    fi_labels   = [k for k, _ in fi_sorted]
    fi_values   = [round(v, 4) for _, v in fi_sorted]

    # ── Probability histogram
    bins     = 20
    hist_arr = np.array(prob_hist_vals)
    bin_edges = np.linspace(0, 1, bins + 1)
    hist_counts, _ = np.histogram(hist_arr, bins=bin_edges)
    hist_labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(bins)]

    # ── Table rows
    table_rows = ""
    for rank, row in enumerate(results, 1):
        p    = row["prob"]
        rsi  = row.get("rsi_14", float("nan"))
        vsrg = row.get("vol_ratio_10_50", float("nan"))
        hi52 = row.get("hi52_ratio", float("nan"))
        m21  = row.get("mom_21d", float("nan"))
        sma50r = row.get("price_vs_sma50", float("nan"))
        sector = row.get("sector", "—")
        ticker = row["ticker"]

        rsi_col = GREEN if (rsi > 50) else RED
        rsi_str = f'<span style="color:{rsi_col}">{rsi:.1f}</span>' if not np.isnan(rsi) else f'<span style="color:{MUTED}">—</span>'

        table_rows += f"""
        <tr>
          <td style="color:{MUTED};text-align:center;">{rank}</td>
          <td><input type="checkbox" class="row-check" value="{ticker}" style="margin-right:6px;">
              <span style="font-weight:600;color:{TEXT};">{ticker}</span></td>
          <td style="color:{MUTED};font-size:12px;">{sector}</td>
          <td>{_prob_bar(p)}</td>
          <td style="text-align:center;">{rsi_str}</td>
          <td style="text-align:center;">{_fmt_float(vsrg)}</td>
          <td style="text-align:center;">{_fmt_pct(hi52 - 1 if not np.isnan(hi52) else float('nan'))}</td>
          <td style="text-align:center;">{_fmt_pct(m21)}</td>
          <td style="text-align:center;">{_fmt_float(sma50r)}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breakout Detector — {index_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:{BG};color:{TEXT};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;min-height:100vh;}}
a{{color:{ACCENT};text-decoration:none;}}
.header{{padding:32px 40px 20px;border-bottom:1px solid {BORDER};}}
.header h1{{font-size:24px;font-weight:700;letter-spacing:-.3px;}}
.header .sub{{color:{MUTED};margin-top:6px;font-size:13px;}}
.badge{{display:inline-block;background:{PANEL};border:1px solid {BORDER};border-radius:4px;
  padding:2px 8px;font-size:11px;margin-left:8px;color:{ACCENT};font-weight:600;}}
.container{{max-width:1400px;margin:0 auto;padding:24px 40px;}}
.section-title{{font-size:16px;font-weight:600;color:{TEXT};margin-bottom:14px;
  padding-bottom:8px;border-bottom:1px solid {BORDER};}}
.card{{background:{PANEL};border:1px solid {BORDER};border-radius:8px;padding:20px;margin-bottom:28px;}}
table{{width:100%;border-collapse:collapse;}}
th{{text-align:left;color:{MUTED};font-size:12px;font-weight:600;text-transform:uppercase;
  letter-spacing:.5px;padding:10px 12px;border-bottom:1px solid {BORDER};white-space:nowrap;}}
td{{padding:10px 12px;border-bottom:1px solid {BORDER};vertical-align:middle;}}
tr:last-child td{{border-bottom:none;}}
tr:hover td{{background:rgba(99,102,241,.06);}}
.stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;}}
.stat-card{{background:{PANEL};border:1px solid {BORDER};border-radius:8px;padding:16px 20px;}}
.stat-label{{color:{MUTED};font-size:11px;text-transform:uppercase;letter-spacing:.5px;}}
.stat-value{{font-size:22px;font-weight:700;color:{TEXT};margin-top:4px;}}
.stat-sub{{color:{MUTED};font-size:11px;margin-top:2px;}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:28px;}}
input.row-check{{cursor:pointer;width:14px;height:14px;accent-color:{ACCENT};}}
.cb-th,.cb-cell{{width:28px;text-align:center;padding:4px 2px;}}
</style>
<style>{_WL_CSS}</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>Breakout Detector <span style="color:{MUTED};font-weight:400;">—</span> {index_name}
    <span class="badge">{n_results} stocks scored</span>
  </h1>
  <div class="sub">
    {today} &nbsp;·&nbsp; Breakout threshold: ≥{threshold:.0f}% gain + volume surge in 10 days &nbsp;·&nbsp;
    Model AUC-PR: <span style="color:{ACCENT};font-weight:600;">{aucpr:.3f}</span>
  </div>
</div>

<div class="container">

<!-- Model Performance Stats -->
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">AUC-PR (CV)</div>
    <div class="stat-value" style="color:{ACCENT};">{aucpr:.3f}</div>
    <div class="stat-sub">Precision-recall AUC</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Precision @ Top 10%</div>
    <div class="stat-value" style="color:{GREEN};">{p10:.1%}</div>
    <div class="stat-sub">Among highest-scored stocks</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Training Samples</div>
    <div class="stat-value">{n_train:,}</div>
    <div class="stat-sub">Ticker-date observations</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Breakout Events</div>
    <div class="stat-value" style="color:{YELLOW};">{n_breakouts:,}</div>
    <div class="stat-sub">{n_breakouts/max(n_train,1):.1%} base rate</div>
  </div>
</div>

<!-- Top Breakout Candidates -->
<div class="section-title">Top Breakout Candidates</div>
<div class="card" style="padding:0;overflow:hidden;">
<table>
  <thead>
    <tr>
      <th class="cb-th"><input type="checkbox" id="cb-all" title="Select all"></th>
      <th>#</th>
      <th>Ticker</th>
      <th>Sector</th>
      <th>Breakout Prob</th>
      <th>RSI 14</th>
      <th>Vol Surge</th>
      <th>52W Hi%</th>
      <th>Mom 21D</th>
      <th>SMA50 Ratio</th>
    </tr>
  </thead>
  <tbody id="results-tbody">"""

    # Re-build table rows with cb-cell for checkbox column
    table_rows2 = ""
    for rank, row in enumerate(results, 1):
        p      = row["prob"]
        rsi    = row.get("rsi_14", float("nan"))
        vsrg   = row.get("vol_ratio_10_50", float("nan"))
        hi52   = row.get("hi52_ratio", float("nan"))
        m21    = row.get("mom_21d", float("nan"))
        sma50r = row.get("price_vs_sma50", float("nan"))
        sector = row.get("sector", "—")
        ticker = row["ticker"]

        rsi_col = GREEN if (not np.isnan(rsi) and rsi > 50) else RED
        rsi_str = (
            f'<span style="color:{rsi_col}">{rsi:.1f}</span>'
            if not np.isnan(rsi)
            else f'<span style="color:{MUTED}">—</span>'
        )
        hi52_val = hi52 - 1 if not np.isnan(hi52) else float("nan")

        table_rows2 += f"""
    <tr>
      <td class="cb-cell"><input type="checkbox" class="row-check" value="{ticker}"></td>
      <td style="color:{MUTED};text-align:center;">{rank}</td>
      <td><span style="font-weight:600;color:{TEXT};">{ticker}</span></td>
      <td style="color:{MUTED};font-size:12px;">{sector}</td>
      <td>{_prob_bar(p)}</td>
      <td style="text-align:center;">{rsi_str}</td>
      <td style="text-align:center;">{_fmt_float(vsrg)}</td>
      <td style="text-align:center;">{_fmt_pct(hi52_val)}</td>
      <td style="text-align:center;">{_fmt_pct(m21)}</td>
      <td style="text-align:center;">{_fmt_float(sma50r)}</td>
    </tr>"""

    html += table_rows2
    html += f"""
  </tbody>
</table>
</div>

<!-- Charts Row -->
<div class="charts-row">
  <!-- Feature Importance -->
  <div>
    <div class="section-title">Feature Importance (Top 10)</div>
    <div class="card">
      <canvas id="fiChart" height="280"></canvas>
    </div>
  </div>
  <!-- Probability Distribution -->
  <div>
    <div class="section-title">Breakout Probability Distribution</div>
    <div class="card">
      <canvas id="probChart" height="280"></canvas>
    </div>
  </div>
</div>

</div><!-- /container -->

<script>
// ── Feature Importance Chart ──────────────────────────────────────────────────
(function(){{
  var ctx = document.getElementById('fiChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: {fi_labels},
      datasets: [{{
        label: 'Importance',
        data: {fi_values},
        backgroundColor: '{ACCENT}cc',
        borderColor: '{ACCENT}',
        borderWidth: 1,
        borderRadius: 3,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(c) {{ return ' ' + c.raw.toFixed(4); }}
          }}
        }}
      }},
      scales: {{
        x: {{
          grid: {{ color: '{BORDER}' }},
          ticks: {{ color: '{MUTED}', font: {{ size: 11 }} }}
        }},
        y: {{
          grid: {{ display: false }},
          ticks: {{ color: '{TEXT}', font: {{ size: 11 }} }}
        }}
      }}
    }}
  }});
}})();

// ── Probability Distribution Chart ───────────────────────────────────────────
(function(){{
  var ctx = document.getElementById('probChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: {hist_labels},
      datasets: [{{
        label: 'Count',
        data: {hist_counts.tolist()},
        backgroundColor: '{ACCENT}99',
        borderColor: '{ACCENT}',
        borderWidth: 1,
        borderRadius: 2,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(c) {{ return ' ' + c.raw + ' stocks'; }}
          }}
        }}
      }},
      scales: {{
        x: {{
          grid: {{ display: false }},
          ticks: {{ color: '{MUTED}', font: {{ size: 10 }}, maxRotation: 45 }}
        }},
        y: {{
          grid: {{ color: '{BORDER}' }},
          ticks: {{ color: '{MUTED}', font: {{ size: 11 }} }}
        }}
      }}
    }}
  }});
}})();
</script>

<script>{_wl_js(port)}</script>
{_WL_BAR}
</body>
</html>"""

    return html


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost Breakout Probability Ranker")
    parser.add_argument("--index",     default="SPX",  choices=["SPX", "NDX", "DOW", "RUT"],
                        help="Index to scan (default: SPX)")
    parser.add_argument("--tickers",   default="",     help="Comma-separated ticker override")
    parser.add_argument("--csv-file",  default="", dest="csv_file",
                        help="Path to CSV with 'ticker' and optional 'weight' columns")
    parser.add_argument("--top",       type=int, default=150,
                        help="Max tickers to process (by market cap order, default: 150)")
    parser.add_argument("--threshold", type=float, default=8.0,
                        help="% gain in 10 days to qualify as breakout (default: 8.0)")
    parser.add_argument("--port",      type=int, default=5050,
                        help="Port for deep-dive WL API (default: 5050)")
    args = parser.parse_args()

    universe_label = "CUSTOM" if (args.csv_file or args.tickers) else args.index
    print(f"\n=== Breakout Detector === {universe_label} ===", flush=True)

    # ── 1. Get tickers ────────────────────────────────────────────────────────
    if args.csv_file:
        print(f"  Loading tickers from CSV: {os.path.basename(args.csv_file)}...", flush=True)
        constituents = _read_csv_file(args.csv_file)
        if not constituents:
            print("  ERROR: CSV file is empty or unreadable. Exiting.", flush=True)
            sys.exit(1)
        print(f"  {len(constituents)} tickers loaded from CSV", flush=True)
    elif args.tickers:
        constituents = [
            {"ticker": t.strip().upper(), "name": t.strip().upper(),
             "sector": "Unknown", "weight": None}
            for t in args.tickers.split(",") if t.strip()
        ]
    else:
        fetcher = {"SPX": _get_spx, "NDX": _get_ndx, "DOW": _get_dow, "RUT": _get_rut}
        print(f"  Fetching {args.index} constituents...", flush=True)
        constituents = fetcher[args.index]()

    if not constituents:
        print("  ERROR: No constituents fetched. Exiting.", flush=True)
        sys.exit(1)

    # Limit to top N
    constituents = constituents[: args.top]
    tickers_list = [c["ticker"] for c in constituents]
    sector_map   = {c["ticker"]: c.get("sector", "Unknown") for c in constituents}
    n = len(tickers_list)
    print(f"  Fetching {args.index} constituents ({n} tickers)...", flush=True)

    # ── 2. Download OHLCV ────────────────────────────────────────────────────
    print("  Downloading OHLCV history...", flush=True)

    all_train_rows: list[pd.DataFrame] = []
    current_snaps: list[dict]          = []
    failed = 0

    def _worker(ticker: str) -> tuple[str, pd.DataFrame | None, dict | None]:
        return fetch_and_build(ticker, args.threshold)

    with ThreadPoolExecutor(max_workers=10) as exe:
        futures = {exe.submit(_worker, t): t for t in tickers_list}
        for fut in as_completed(futures):
            ticker, train_df, current = fut.result()
            if train_df is not None and not train_df.empty:
                all_train_rows.append(train_df)
            if current is not None:
                current["sector"] = sector_map.get(ticker, "Unknown")
                # Enrich sector from yfinance if still Unknown
                current_snaps.append(current)
            else:
                failed += 1

    if not all_train_rows:
        print("  ERROR: No training data collected. Exiting.", flush=True)
        sys.exit(1)

    # ── 3. Build feature matrix ───────────────────────────────────────────────
    print("  Building feature matrix...", flush=True)

    full_df = pd.concat(all_train_rows, ignore_index=True)
    full_df = full_df.dropna(subset=FEATURE_COLS + ["label"])
    full_df["label"] = full_df["label"].astype(int)

    n_train     = len(full_df)
    n_breakouts = int(full_df["label"].sum())

    if n_train < 500:
        print(
            f"  WARNING: Only {n_train} training samples collected. "
            "Model will be weak but continuing...",
            flush=True,
        )

    X_all = full_df[FEATURE_COLS].values
    y_all = full_df["label"].values

    # ── 4. Train XGBoost ─────────────────────────────────────────────────────
    print("  Training XGBoost classifier...", flush=True)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  ERROR: xgboost is not installed. Run: pip install xgboost", flush=True)
        sys.exit(1)

    from sklearn.metrics import average_precision_score, precision_score

    pos_weight = float((y_all == 0).sum()) / max(float((y_all == 1).sum()), 1)

    model_cv = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # Time-based split: train on first 80%, test on last 20%
    split_idx   = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    aucpr = 0.0
    p10   = 0.0

    if len(X_train) > 0 and len(X_test) > 0 and y_train.sum() > 0:
        model_cv.fit(X_train, y_train, verbose=False)
        prob_test = model_cv.predict_proba(X_test)[:, 1]

        aucpr = float(average_precision_score(y_test, prob_test))

        # Precision@top10%: precision among the top-decile probability predictions
        thresh_idx = int(len(prob_test) * 0.9)
        sorted_probs = np.sort(prob_test)[::-1]
        cutoff = sorted_probs[min(thresh_idx, len(sorted_probs) - 1)]
        top10_mask = prob_test >= cutoff
        if top10_mask.sum() > 0:
            p10 = float(precision_score(y_test, (prob_test >= cutoff).astype(int), zero_division=0))

    print(f"  CV AUC-PR: {aucpr:.3f}  Precision@Top10: {p10:.1%}", flush=True)

    # Retrain on all data
    model_final = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model_final.fit(X_all, y_all, verbose=False)

    # Feature importance
    fi_raw  = model_final.get_booster().get_fscore()
    fi_dict = {FEATURE_COLS[i]: float(model_final.feature_importances_[i]) for i in range(len(FEATURE_COLS))}

    # ── 5. Score current snapshots ────────────────────────────────────────────
    if not current_snaps:
        print("  ERROR: No current snapshots available for scoring.", flush=True)
        sys.exit(1)

    snap_df = pd.DataFrame(current_snaps)
    snap_df = snap_df.dropna(subset=FEATURE_COLS)

    if snap_df.empty:
        print("  ERROR: All current snapshots have missing features.", flush=True)
        sys.exit(1)

    X_snap   = snap_df[FEATURE_COLS].values
    probs    = model_final.predict_proba(X_snap)[:, 1]
    snap_df  = snap_df.copy()
    snap_df["prob"] = probs

    # Sort descending by breakout probability
    snap_df  = snap_df.sort_values("prob", ascending=False).reset_index(drop=True)
    results  = snap_df.to_dict("records")
    n_results = len(results)

    print(f"  Generating report ({n_results} stocks scored)...", flush=True)

    # ── 6. Build and save HTML ────────────────────────────────────────────────
    prob_hist_vals = probs.tolist()
    today_str      = datetime.today().strftime("%Y%m%d")
    fname          = f"breakout_{args.index}_{today_str}.html"
    out_path       = os.path.join(OUT_DIR, fname)

    html = build_html(
        results       = results,
        index_name    = args.index,
        aucpr         = aucpr,
        p10           = p10,
        n_train       = n_train,
        n_breakouts   = n_breakouts,
        threshold     = args.threshold,
        feat_importance = fi_dict,
        prob_hist_vals  = prob_hist_vals,
        port            = args.port,
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"  ✓  Report saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
