#!/usr/bin/env python3
"""
earningsPredictor.py — Earnings Surprise Probability Predictor
==============================================================
Trains an XGBoost binary classifier on historical earnings beat/miss data,
then predicts beat probability for stocks reporting in the next N days.

Usage
-----
    python earningsPredictor.py --index SPX
    python earningsPredictor.py --index NDX --days 30 --top 50
    python earningsPredictor.py --tickers AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

# ── sys.path so sibling packages are importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests as _requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Directories ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "earningsPredData")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
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
    """Parse a CSV for tickers. Accepts ticker/symbol column (case-insensitive).
    Falls back to the first column if no recognised header is found."""
    import csv as _csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        raw = [ln for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    if not raw:
        return []
    reader = _csv.DictReader(raw)
    fl = {(k or "").lower().strip(): k for k in (reader.fieldnames or [])}
    t_key = next((v for k, v in fl.items() if k in ("ticker", "symbol")), None)
    if t_key is None:
        plain = list(_csv.reader(raw))
        start = 1 if plain and not plain[0][0].strip().replace(".", "").isdigit() else 0
        return [{"ticker": r[0].strip().upper(), "name": r[0].strip().upper(), "sector": "Unknown"}
                for r in plain[start:] if r and r[0].strip()]
    return [{"ticker": (row.get(t_key) or "").strip().upper(),
             "name": (row.get(t_key) or "").strip().upper(), "sector": "Unknown"}
            for row in reader if (row.get(t_key) or "").strip()]


# ── User-agent for web requests ────────────────────────────────────────────────
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/122.0.0.0 Safari/537.36")


# ══════════════════════════════════════════════════════════════════════════════
# Index constituent fetchers  (ported from growthScreeners.py)
# ══════════════════════════════════════════════════════════════════════════════

def _get_slickcharts(url: str, label: str) -> list[dict]:
    try:
        r = _requests.get(url, headers={"User-Agent": _UA}, timeout=20)
        tbls = pd.read_html(io.StringIO(r.text))
        df = tbls[0]
        results = []
        for _, row in df.iterrows():
            t = str(row.get("Symbol", "")).strip()
            if not t or t == "nan":
                continue
            results.append({"ticker": t, "name": str(row.get("Company", t)), "sector": "Unknown"})
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
    url = ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
           "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")
    try:
        r = _requests.get(url, headers={"User-Agent": _UA}, timeout=30)
        lines = r.text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("Ticker"))
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        df = df[df["Asset Class"] == "Equity"]
        results = []
        for _, row in df.iterrows():
            t = str(row.get("Ticker", "")).strip()
            if not t or t == "nan" or t == "-":
                continue
            results.append({"ticker": t, "name": str(row.get("Name", t)),
                            "sector": str(row.get("Sector", "Unknown"))})
        return results
    except Exception as e:
        print(f"  RUT iShares fetch failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Watchlist bar  (copied verbatim from growthScreeners.py)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Feature helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe(val, lo=None, hi=None, default=np.nan):
    """Return val clamped to [lo, hi], or default if val is None/NaN."""
    try:
        v = float(val)
        if not np.isfinite(v):
            return default
        if lo is not None and v < lo:
            return lo
        if hi is not None and v > hi:
            return hi
        return v
    except Exception:
        return default


def _price_return(prices: pd.Series, n_days: int) -> float:
    """n-day return ending at the last row."""
    if prices is None or len(prices) < n_days + 1:
        return np.nan
    end   = float(prices.iloc[-1])
    start = float(prices.iloc[-(n_days + 1)])
    if start == 0:
        return np.nan
    return (end - start) / start


def _realized_vol(prices: pd.Series, n_days: int) -> float:
    if prices is None or len(prices) < n_days + 1:
        return np.nan
    rets = prices.iloc[-(n_days + 1):].pct_change().dropna()
    if len(rets) < 2:
        return np.nan
    return float(rets.std())


def _beat_stats(eh: pd.DataFrame) -> tuple[float, float]:
    """Return (beat_rate, avg_surprise_pct) from earnings_history DataFrame."""
    try:
        if eh is None or eh.empty:
            return np.nan, np.nan
        cols = {c.lower(): c for c in eh.columns}
        actual_col   = cols.get("epsactual",   cols.get("reported eps",  None))
        estimate_col = cols.get("epsestimate", cols.get("eps estimate", None))
        if actual_col is None or estimate_col is None:
            return np.nan, np.nan
        df = eh[[actual_col, estimate_col]].dropna()
        if df.empty:
            return np.nan, np.nan
        actual   = df[actual_col].astype(float)
        estimate = df[estimate_col].astype(float)
        beats    = (actual >= estimate).astype(float)
        surp_pct = np.where(
            estimate != 0,
            (actual - estimate) / estimate.abs() * 100,
            np.nan
        )
        n = min(len(beats), 8)
        beat_rate   = float(beats.iloc[:n].mean())
        avg_surp    = float(np.nanmean(surp_pct[:n]))
        return beat_rate, avg_surp
    except Exception:
        return np.nan, np.nan


def _fetch_ticker_data(ticker: str) -> dict:
    """Fetch all features + earnings history for a single ticker."""
    result: dict = {"ticker": ticker}
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}

        # ── Earnings history beat stats ────────────────────────────────────
        try:
            eh = t.earnings_history
            if eh is None or (hasattr(eh, "empty") and eh.empty):
                eh = t.quarterly_earnings
        except Exception:
            eh = None
        result["_earnings_history"] = eh

        beat_rate, avg_surp = _beat_stats(eh)
        result["beat_rate"]       = beat_rate
        result["avg_surprise_pct"] = avg_surp

        # ── Fundamentals ───────────────────────────────────────────────────
        result["rev_growth"]   = _safe(info.get("revenueGrowth"))
        result["eps_growth"]   = _safe(info.get("earningsGrowth"))
        result["gross_margin"] = _safe(info.get("grossMargins"))
        result["roe"]          = _safe(info.get("returnOnEquity"))

        pe_raw  = _safe(info.get("trailingPE"),  5, 200)
        fpe_raw = _safe(info.get("forwardPE"),   5, 200)
        result["pe"]              = pe_raw
        result["fwd_pe"]          = fpe_raw
        result["pe_vs_sector_pct"] = pe_raw / 30.0 if np.isfinite(pe_raw) else np.nan

        # ── Sector / market cap ────────────────────────────────────────────
        result["sector"]     = info.get("sector", "Unknown") or "Unknown"
        result["market_cap"] = _safe(info.get("marketCap"), 0)
        result["name"]       = info.get("longName", ticker) or ticker

        # ── Price momentum ─────────────────────────────────────────────────
        try:
            hist = t.history(period="6mo", auto_adjust=True)
            closes = hist["Close"] if "Close" in hist.columns else None
        except Exception:
            closes = None

        result["mom_1m"]    = _price_return(closes, 21)
        result["mom_3m"]    = _price_return(closes, 63)
        vol10 = _realized_vol(closes, 10)
        vol30 = _realized_vol(closes, 30)
        result["vol_ratio"] = (vol10 / vol30) if (np.isfinite(vol10) and np.isfinite(vol30) and vol30 > 0) else np.nan

        # ── Upcoming earnings date ─────────────────────────────────────────
        try:
            cal = t.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                # Some yfinance versions return a DataFrame
                if "Earnings Date" in cal.index:
                    ed_raw = cal.loc["Earnings Date"].iloc[0]
                elif "Earnings Date" in cal.columns:
                    ed_raw = cal["Earnings Date"].iloc[0]
                else:
                    ed_raw = None
            elif isinstance(cal, dict):
                ed_raw = cal.get("Earnings Date", [None])[0] if isinstance(cal.get("Earnings Date"), list) else cal.get("Earnings Date")
            else:
                ed_raw = None

            if ed_raw is not None:
                if isinstance(ed_raw, (int, float)):
                    result["earnings_date"] = datetime.utcfromtimestamp(ed_raw).date()
                elif hasattr(ed_raw, "date"):
                    result["earnings_date"] = ed_raw.date()
                else:
                    result["earnings_date"] = pd.Timestamp(ed_raw).date()
            else:
                result["earnings_date"] = None
        except Exception:
            result["earnings_date"] = None

    except Exception as e:
        result["_error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Training-set builder
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "beat_rate", "avg_surprise_pct",
    "rev_growth", "eps_growth", "gross_margin", "roe",
    "mom_1m", "mom_3m", "vol_ratio",
    "pe", "fwd_pe", "pe_vs_sector_pct",
]


def _build_training_rows(ticker_data_list: list[dict]) -> pd.DataFrame:
    """
    One row per ticker.  Label = beat in the most-recent quarter (holdout).
    beat_rate feature = computed on the remaining quarters (leave-one-out)
    so there is no label leakage from the feature.

    This avoids the fatal flaw of the previous approach where every ticker
    contributed N identical feature rows (current-day snapshot × N quarters),
    which caused the model to predict the unconditional base-rate for everyone.
    """
    rows = []
    for td in ticker_data_list:
        eh = td.get("_earnings_history")
        if eh is None or (hasattr(eh, "empty") and eh.empty):
            continue
        try:
            cols_lower   = {c.lower(): c for c in eh.columns}
            actual_col   = cols_lower.get("epsactual",   cols_lower.get("reported eps", None))
            estimate_col = cols_lower.get("epsestimate", cols_lower.get("eps estimate", None))
            if actual_col is None or estimate_col is None:
                continue
            df_eh = eh[[actual_col, estimate_col]].dropna().head(8)
            if df_eh.empty:
                continue

            actual_arr   = df_eh[actual_col].astype(float).values
            estimate_arr = df_eh[estimate_col].astype(float).values

            # Label: did the company beat in its most-recent reported quarter?
            label = int(actual_arr[0] >= estimate_arr[0])

            # Leave-one-out beat_rate: computed on quarters 1-N (exclude most-recent)
            if len(actual_arr) >= 2:
                rest_beats  = (actual_arr[1:] >= estimate_arr[1:]).astype(float)
                loo_beat_rate = float(rest_beats.mean())
                loo_avg_surp  = float(
                    np.nanmean((actual_arr[1:] - estimate_arr[1:]) /
                               np.abs(estimate_arr[1:]).clip(min=1e-6) * 100)
                )
            else:
                loo_beat_rate = np.nan
                loo_avg_surp  = np.nan

            row = {}
            for f in FEATURE_COLS:
                row[f] = td.get(f, np.nan)
            # Override with leave-one-out versions (no leakage)
            row["beat_rate"]      = loo_beat_rate
            row["avg_surprise_pct"] = loo_avg_surp
            row["label"] = label
            rows.append(row)

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# HTML generation
# ══════════════════════════════════════════════════════════════════════════════

def _pct_bar(value: float, color: str, width_pct: float) -> str:
    """Render a progress-bar cell."""
    pct = round(value * 100, 1) if np.isfinite(value) else 0.0
    bar_w = round(min(max(width_pct, 0), 100))
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="background:{BORDER};border-radius:3px;width:80px;height:8px;overflow:hidden;">'
        f'<div style="background:{color};width:{bar_w}%;height:100%;border-radius:3px;"></div></div>'
        f'<span style="color:{color};font-weight:600;">{pct:.1f}%</span></div>'
    )


def _color_pct(val: float, good_positive: bool = True) -> str:
    if not np.isfinite(val):
        return f'<span style="color:{MUTED}">—</span>'
    color = GREEN if (val >= 0) == good_positive else RED
    return f'<span style="color:{color};">{val:+.1f}%</span>'


def _fmt(val: float, decimals: int = 2, suffix: str = "") -> str:
    if not np.isfinite(val):
        return f'<span style="color:{MUTED}">—</span>'
    return f"{val:.{decimals}f}{suffix}"


def build_html(
    upcoming: list[dict],
    all_preds: list[dict],
    index_name: str,
    cv_auc: float,
    cv_acc: float,
    n_train: int,
    feature_importances: dict,
    days: int,
    low_data_warning: bool,
    suite_port: int,
) -> str:

    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_up      = len(upcoming)
    auc_str   = f"{cv_auc:.3f}" if np.isfinite(cv_auc) else "N/A"
    acc_str   = f"{cv_acc:.1%}" if np.isfinite(cv_acc) else "N/A"

    # ── Histogram data ─────────────────────────────────────────────────────
    all_beat_probs = [r["beat_prob"] for r in all_preds if np.isfinite(r.get("beat_prob", np.nan))]
    hist_bins  = list(range(0, 105, 5))
    hist_counts = [0] * (len(hist_bins) - 1)
    for p in all_beat_probs:
        idx = min(int(p * 100 // 5), len(hist_counts) - 1)
        hist_counts[idx] += 1
    hist_labels_js = str([f"{b}–{b+5}%" for b in hist_bins[:-1]])
    hist_counts_js = str(hist_counts)

    # ── Feature importance data ────────────────────────────────────────────
    fi_sorted = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    fi_labels_js = str([k for k, _ in fi_sorted])
    fi_vals_js   = str([round(v, 4) for _, v in fi_sorted])

    # ── Upcoming earnings rows ─────────────────────────────────────────────
    today = datetime.now(timezone.utc).date()

    if n_up == 0:
        table_html = (
            f'<div style="text-align:center;color:{MUTED};padding:40px;">'
            f'No upcoming earnings found in the next {days} days.</div>'
        )
    else:
        rows_html = ""
        for r in upcoming:
            ed = r.get("earnings_date")
            days_away = (ed - today).days if ed else "?"
            beat_prob  = r.get("beat_prob", np.nan)
            beat_rate  = r.get("beat_rate", np.nan)
            avg_surp   = r.get("avg_surprise_pct", np.nan)
            gm         = r.get("gross_margin", np.nan)
            rev_g      = r.get("rev_growth", np.nan)
            mom1       = r.get("mom_1m", np.nan)

            # Beat prob bar
            if np.isfinite(beat_prob):
                bar_color = GREEN if beat_prob >= 0.6 else (YELLOW if beat_prob >= 0.45 else RED)
                prob_cell = _pct_bar(beat_prob, bar_color, beat_prob * 100)
            else:
                prob_cell = f'<span style="color:{MUTED}">N/A</span>'

            # Beat rate colored
            if np.isfinite(beat_rate):
                br_color = GREEN if beat_rate >= 0.625 else (YELLOW if beat_rate >= 0.5 else RED)
                br_cell  = f'<span style="color:{br_color};font-weight:600;">{beat_rate:.0%}</span>'
            else:
                br_cell  = f'<span style="color:{MUTED}">—</span>'

            ed_str = ed.strftime("%b %d") if ed else "?"
            da_str = f"+{days_away}d" if isinstance(days_away, int) else "?"

            ticker = r["ticker"]
            rows_html += f"""
            <tr>
              <td style="padding:4px 6px;"><input type="checkbox" class="row-check" value="{ticker}"></td>
              <td style="font-weight:700;color:{TEXT};">{ticker}</td>
              <td style="color:{MUTED};font-size:12px;">{r.get('sector','')}</td>
              <td style="color:{TEXT};">{ed_str} <span style="color:{MUTED};font-size:11px;">({da_str})</span></td>
              <td>{prob_cell}</td>
              <td>{br_cell}</td>
              <td>{_color_pct(avg_surp)}</td>
              <td>{_fmt(gm * 100 if np.isfinite(gm) else np.nan, 1, '%') if np.isfinite(gm) else '<span style="color:' + MUTED + '">—</span>'}</td>
              <td>{_color_pct(rev_g * 100 if np.isfinite(rev_g) else np.nan)}</td>
              <td>{_color_pct(mom1 * 100 if np.isfinite(mom1) else np.nan)}</td>
            </tr>"""

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
          <thead>
            <tr style="background:{BORDER};color:{MUTED};text-align:left;">
              <th style="padding:8px 6px;"><input type="checkbox" id="cb-all"></th>
              <th style="padding:8px 6px;">Ticker</th>
              <th style="padding:8px 6px;">Sector</th>
              <th style="padding:8px 6px;">Earnings Date</th>
              <th style="padding:8px 6px;">Beat Prob</th>
              <th style="padding:8px 6px;">Beat Rate</th>
              <th style="padding:8px 6px;">Avg Surp %</th>
              <th style="padding:8px 6px;">Gross Margin</th>
              <th style="padding:8px 6px;">Rev Growth</th>
              <th style="padding:8px 6px;">Mom 1M</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>"""

    low_warn_banner = ""
    if low_data_warning:
        low_warn_banner = (
            f'<div style="background:#7c3b00;color:{YELLOW};padding:10px 16px;'
            f'border-radius:6px;margin-bottom:16px;font-size:13px;">'
            f'Warning: fewer than 50 training samples — predictions may be unreliable.</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Earnings Predictor — {index_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:{BG};color:{TEXT};font-family:'Inter',system-ui,sans-serif;font-size:14px;line-height:1.5;}}
a{{color:{ACCENT};text-decoration:none;}}
.card{{background:{PANEL};border:1px solid {BORDER};border-radius:10px;padding:20px;margin-bottom:18px;}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}}
h1{{font-size:22px;font-weight:700;color:{TEXT};}}
h2{{font-size:15px;font-weight:600;color:{MUTED};margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em;}}
.stat{{display:flex;flex-direction:column;gap:2px;}}
.stat .val{{font-size:26px;font-weight:700;color:{ACCENT};}}
.stat .lbl{{font-size:12px;color:{MUTED};}}
.stats-row{{display:flex;gap:24px;flex-wrap:wrap;}}
table tr:hover{{background:rgba(255,255,255,.03);}}
</style>
<style>{_WL_CSS}</style>
</head>
<body>
<div style="max-width:1200px;margin:0 auto;padding:24px 16px;">

<!-- Header -->
<div class="card" style="margin-bottom:18px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <h1>Earnings Surprise Predictor &mdash; {index_name}</h1>
      <p style="color:{MUTED};margin-top:4px;font-size:13px;">
        {n_up} stocks reporting in the next {days} days &nbsp;&bull;&nbsp;
        CV AUC: {auc_str} &nbsp;&bull;&nbsp; Generated {ts}
      </p>
    </div>
    <div class="stats-row">
      <div class="stat"><span class="val">{n_up}</span><span class="lbl">Upcoming</span></div>
      <div class="stat"><span class="val">{auc_str}</span><span class="lbl">CV AUC</span></div>
      <div class="stat"><span class="val">{acc_str}</span><span class="lbl">CV Accuracy</span></div>
      <div class="stat"><span class="val">{n_train:,}</span><span class="lbl">Train Samples</span></div>
    </div>
  </div>
</div>

<!-- Warning banner -->
{low_warn_banner}

<!-- Upcoming Earnings Table -->
<div class="card">
  <h2>Upcoming Earnings — Sorted by Beat Probability</h2>
  {table_html}
</div>

<!-- Model Performance + Feature Importance -->
<div class="grid2">
  <div class="card">
    <h2>Model Performance</h2>
    <div class="stats-row" style="margin-bottom:14px;">
      <div class="stat"><span class="val">{auc_str}</span><span class="lbl">Cross-Val AUC</span></div>
      <div class="stat"><span class="val">{acc_str}</span><span class="lbl">Accuracy</span></div>
      <div class="stat"><span class="val">{n_train:,}</span><span class="lbl">Training Rows</span></div>
    </div>
    <p style="color:{MUTED};font-size:12px;line-height:1.6;">
      XGBoost classifier trained with 5-fold stratified cross-validation.
      Beat label: actual EPS &ge; estimated EPS. Features derived from current
      yfinance snapshots. Model is re-trained on all data before prediction.
      Probabilities are uncalibrated &mdash; use as relative ranking signal,
      not absolute probability.
    </p>
  </div>

  <div class="card">
    <h2>Feature Importance (Top 10)</h2>
    <canvas id="fiChart" height="200"></canvas>
  </div>
</div>

<!-- Beat Probability Distribution -->
<div class="card">
  <h2>Beat Probability Distribution &mdash; Full Universe</h2>
  <canvas id="histChart" height="120"></canvas>
</div>

</div><!-- /container -->

<script>
// Feature Importance Chart
(function(){{
  var ctx = document.getElementById('fiChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: {fi_labels_js},
      datasets: [{{
        label: 'Importance',
        data: {fi_vals_js},
        backgroundColor: '{ACCENT}',
        borderRadius: 4,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '{BORDER}' }}, ticks: {{ color: '{MUTED}' }} }},
        y: {{ grid: {{ color: '{BORDER}' }}, ticks: {{ color: '{TEXT}' }} }}
      }}
    }}
  }});
}})();

// Histogram Chart
(function(){{
  var ctx2 = document.getElementById('histChart').getContext('2d');
  new Chart(ctx2, {{
    type: 'bar',
    data: {{
      labels: {hist_labels_js},
      datasets: [{{
        label: '# Stocks',
        data: {hist_counts_js},
        backgroundColor: '{ACCENT}88',
        borderColor: '{ACCENT}',
        borderWidth: 1,
        borderRadius: 3,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '{BORDER}' }}, ticks: {{ color: '{MUTED}', maxRotation: 45, minRotation: 45 }} }},
        y: {{ grid: {{ color: '{BORDER}' }}, ticks: {{ color: '{MUTED}' }}, beginAtZero: true }}
      }}
    }}
  }});
}})();
</script>

<script>{_wl_js(suite_port)}</script>
{_WL_BAR}
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Earnings Surprise Predictor")
    parser.add_argument("--index",    default="SPX", choices=["SPX", "NDX", "DOW", "RUT"],
                        help="Index universe (default: SPX)")
    parser.add_argument("--tickers",  default="",
                        help="Comma-separated ticker override")
    parser.add_argument("--csv-file", default="", dest="csv_file",
                        help="Path to CSV with a 'ticker' column")
    parser.add_argument("--top",      type=int, default=100,
                        help="Limit universe to top N by market cap (default: 100)")
    parser.add_argument("--days",     type=int, default=60,
                        help="Look-ahead window for upcoming earnings (default: 60)")
    parser.add_argument("--port",     type=int, default=5050,
                        help="Suite port for WL endpoint (default: 5050)")
    args = parser.parse_args()

    import logging as _logging
    _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)

    index_name     = "CUSTOM" if (args.csv_file or args.tickers) else args.index
    today          = datetime.now(timezone.utc).date()
    cutoff         = today + timedelta(days=args.days)

    print(f"\n=== Earnings Surprise Predictor === {index_name} ===", flush=True)

    # ── 1. Get universe ────────────────────────────────────────────────────
    if args.csv_file:
        print(f"  Loading tickers from CSV: {os.path.basename(args.csv_file)}...", flush=True)
        constituents = _read_csv_file(args.csv_file)
        if not constituents:
            print("  ERROR: CSV file is empty or unreadable. Exiting.", flush=True)
            sys.exit(1)
        print(f"  {len(constituents)} tickers loaded from CSV", flush=True)
    elif args.tickers:
        tickers_raw  = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        constituents = [{"ticker": t, "name": t, "sector": "Unknown"} for t in tickers_raw]
    else:
        print(f"  Fetching {args.index} constituents...", flush=True)
        fetcher      = {"SPX": _get_spx, "NDX": _get_ndx, "DOW": _get_dow, "RUT": _get_rut}[args.index]
        constituents = fetcher()

    if not constituents:
        print("  ERROR: No constituents found. Exiting.", flush=True)
        sys.exit(1)

    tickers = [c["ticker"] for c in constituents]
    n = len(tickers)

    # ── 2. Download ticker data ────────────────────────────────────────────
    print(f"  Downloading ticker data ({n} tickers)...", flush=True)

    ticker_data: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_ticker_data, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                ticker_data.append(fut.result())
            except Exception:
                pass

    # Sort by market cap descending, keep top N
    ticker_data.sort(key=lambda x: x.get("market_cap", 0) or 0, reverse=True)
    ticker_data = ticker_data[: args.top]

    # ── 3. Build training set ──────────────────────────────────────────────
    print("  Building training set...", flush=True)
    train_df = _build_training_rows(ticker_data)
    n_train  = len(train_df)
    low_data_warning = n_train < 50

    if n_train == 0:
        print("  WARNING: 0 training samples — cannot train model.", flush=True)
        cv_auc      = np.nan
        cv_acc      = np.nan
        model       = None
        fi_dict: dict = {}
    else:
        if low_data_warning:
            print(f"  WARNING: only {n_train} training samples — predictions may be unreliable.", flush=True)

        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score

        X = train_df[FEATURE_COLS].copy()
        y = train_df["label"].astype(int)

        # Impute missing features with column median
        for col in FEATURE_COLS:
            med = X[col].median()
            X[col] = X[col].fillna(med if np.isfinite(med) else 0.0)

        # With small training sets XGBoost collapses to the base rate.
        # Use logistic regression (L2) for <150 samples — it generalises
        # better and produces calibrated, differentiated probabilities.
        USE_XGB = n_train >= 150
        if USE_XGB:
            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                scale_pos_weight=1.0,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )
            model_name = "XGBoost"
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(
                    C=0.5, max_iter=1000, random_state=42,
                    class_weight="balanced",
                )),
            ])
            model_name = "Logistic Regression"

        # ── 4. Cross-validate ──────────────────────────────────────────────
        print(f"  Training {model_name} (CV, {n_train} samples)...", flush=True)
        skf = StratifiedKFold(n_splits=min(5, int(y.value_counts().min())), shuffle=True, random_state=42)
        try:
            cv_res = cross_validate(
                model, X, y, cv=skf,
                scoring={
                    "auc": make_scorer(roc_auc_score, needs_proba=True, response_method="predict_proba"),
                    "acc": make_scorer(accuracy_score),
                },
                return_train_score=False,
            )
            cv_auc = float(np.mean(cv_res["test_auc"]))
            cv_acc = float(np.mean(cv_res["test_acc"]))
        except Exception as e:
            print(f"  CV failed: {e}", flush=True)
            cv_auc = np.nan
            cv_acc = np.nan

        print(f"  CV AUC: {cv_auc:.3f}  Accuracy: {cv_acc:.1%}", flush=True)

        # Retrain on full data
        model.fit(X, y)
        # Extract feature importances — method differs by model type
        try:
            if USE_XGB:
                importances = model.feature_importances_.tolist()
            else:
                # Logistic regression: use absolute coefficient magnitudes
                coefs = model.named_steps["clf"].coef_[0]
                importances = np.abs(coefs).tolist()
            fi_dict = dict(zip(FEATURE_COLS, importances))
        except Exception:
            fi_dict = {}

    # ── 5. Identify upcoming earnings ─────────────────────────────────────
    print(f"  Identifying upcoming earnings ({args.days}-day window)...", flush=True)

    upcoming: list[dict] = []
    all_preds: list[dict] = []

    for td in ticker_data:
        ed = td.get("earnings_date")

        # Predict beat probability
        beat_prob = np.nan
        if model is not None and n_train > 0:
            try:
                from xgboost import XGBClassifier  # already imported above
                row_feats = []
                for f in FEATURE_COLS:
                    val = td.get(f, np.nan)
                    if not np.isfinite(float(val) if val is not None else np.nan):
                        val = 0.0
                    row_feats.append(float(val))
                row_arr = np.array(row_feats).reshape(1, -1)
                beat_prob = float(model.predict_proba(row_arr)[0][1])
            except Exception:
                beat_prob = np.nan

        td["beat_prob"] = beat_prob
        all_preds.append(td)

        if ed is not None and today <= ed <= cutoff:
            upcoming.append(td)

    # Sort upcoming by beat_prob descending
    upcoming.sort(key=lambda x: x.get("beat_prob", -1) if np.isfinite(x.get("beat_prob", np.nan)) else -1, reverse=True)

    n_upcoming = len(upcoming)
    print(f"  Identifying upcoming earnings ({n_upcoming} found)...", flush=True)

    # ── 6. Generate report ─────────────────────────────────────────────────
    print("  Generating report...", flush=True)

    html = build_html(
        upcoming=upcoming,
        all_preds=all_preds,
        index_name=index_name,
        cv_auc=cv_auc,
        cv_acc=cv_acc,
        n_train=n_train,
        feature_importances=fi_dict,
        days=args.days,
        low_data_warning=low_data_warning,
        suite_port=args.port,
    )

    date_str = datetime.now().strftime("%Y%m%d")
    fname    = f"earnings_pred_{index_name}_{date_str}.html"
    out_path = os.path.join(OUT_DIR, fname)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✓  Report saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
