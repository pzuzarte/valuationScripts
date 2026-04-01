#!/usr/bin/env python3
"""
growthScreeners.py — 7-Factor Growth Screener
==============================================
Scans an entire index (SPX/NDX/DOW/RUT/TSX), fetches financial data for
every constituent, computes 7 growth screener metrics in a single pass,
and outputs a rich dark-themed interactive HTML report.

Screeners:
  1. RS Rating          — O'Neil-style relative strength (percentile rank 1-99)
  2. EPS Acceleration   — Annual EPS growth accelerating ≥5pp YoY
  3. Estimate Revision  — Forward EPS revised up vs trailing growth
  4. Beat Rate          — Beats in ≥3 of last 4 reported quarters
  5. GARP               — PEG ≤1.5 with EPS growth ≥15%
  6. FCF Compounder     — FCF CAGR ≥20% + expanding margins
  7. Gross Margin Exp   — Revenue growth ≥15% + expanding gross margin

Usage
-----
    python growthScreeners.py --index SPX
    python growthScreeners.py --index NDX
    python growthScreeners.py --tickers AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import subprocess
import sys
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests as _requests

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Directories ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette / theme ────────────────────────────────────────────────────────────
BG      = "#0d0d1a"
PANEL   = "#13132a"
ACCENT  = "#7c6af7"
BORDER  = "#2a2a4a"
TEXT    = "#e8e8f0"
SUBTEXT = "#8888aa"
GREEN   = "#4ade80"
RED     = "#f87171"
GOLD    = "#fbbf24"
BLUE    = "#60a5fa"

# ── Index constituent fetchers ─────────────────────────────────────────────────

_FETCH_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
             "AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/122.0.0.0 Safari/537.36")


def _get_slickcharts(url: str, label: str) -> list[dict]:
    """Parse slickcharts.com index page — first table has Company/Symbol/Weight."""
    print(f"  Fetching {label} constituents from slickcharts.com...")
    try:
        r = _requests.get(url, headers={"User-Agent": _FETCH_UA}, timeout=20)
        tbls = pd.read_html(io.StringIO(r.text))
        df = tbls[0]  # first table is always the constituent list
        results = []
        for _, row in df.iterrows():
            t = str(row.get("Symbol", "")).strip()
            if not t or t == "nan":
                continue
            results.append({
                "ticker": t,
                "name":   str(row.get("Company", t)),
                "sector": "Unknown",   # slickcharts has no sector; filled by yfinance later
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
    """Russell 2000 from iShares IWM ETF holdings CSV — authoritative and current."""
    print("  Fetching Russell 2000 constituents from iShares IWM holdings...")
    url = ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
           "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")
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


def _get_tsx() -> list[dict]:
    """S&P/TSX Composite via TradingView screener (Canada market)."""
    print("  Fetching S&P/TSX constituents via TradingView screener...")
    try:
        from tradingview_screener import Query, Column
        q = (Query()
             .set_markets("canada")
             .where(Column("is_primary") == True)
             .select("name", "full_name", "sector")
             .limit(600))
        _, df = q.get_scanner_data()
        if df is None or df.empty:
            return []
        results = []
        for _, row in df.iterrows():
            ticker = str(row.get("name", "")).replace("TSX:", "")
            results.append({
                "ticker": ticker,
                "name":   str(row.get("full_name", ticker)),
                "sector": str(row.get("sector", "Unknown")),
            })
        return results
    except Exception as e:
        print(f"  TSX TradingView fetch failed: {e}")
        return []


def get_index_tickers(index: str) -> list[dict]:
    """Return list of {ticker, name, sector} dicts for the chosen index."""
    if index == "SPX":
        return _get_spx()
    if index == "NDX":
        return _get_ndx()
    if index == "DOW":
        return _get_dow()
    if index == "RUT":
        return _get_rut()
    if index == "TSX":
        return _get_tsx()
    return []


# ── Data helpers ───────────────────────────────────────────────────────────────

def _f(df, row_keys, col=0):
    """Get float value from a DataFrame by trying row key names, returning col index."""
    if df is None or df.empty:
        return math.nan
    for k in row_keys:
        if k in df.index:
            row = df.loc[k].dropna()
            if len(row) > col:
                try:
                    return float(row.iloc[col])
                except Exception:
                    pass
    return math.nan


# ── Per-ticker fetch ───────────────────────────────────────────────────────────

def _fetch_ticker(ticker: str) -> dict:
    """Fetch all raw data for one ticker. Returns dict with error key if failed."""
    try:
        import yfinance as yf
        tk   = yf.Ticker(ticker)
        info = tk.info or {}
        hist = tk.history(period="14mo")
        inc  = tk.income_stmt
        cf   = tk.cash_flow
        try:
            ed = tk.earnings_dates
        except Exception:
            ed = None
        return {
            "ticker": ticker,
            "info":   info,
            "hist":   hist,
            "inc":    inc,
            "cf":     cf,
            "ed":     ed,
            "error":  None,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ── RS raw computation (before cross-sectional ranking) ───────────────────────

def _compute_rs_raw(hist) -> float:
    """IBD-style weighted 12m return excluding last 21 trading days."""
    if hist is None or len(hist) < 30:
        return math.nan
    try:
        closes = hist["Close"].dropna()
        if len(closes) < 30:
            return math.nan

        # Exclude last 21 trading days (approx 1 month)
        n = len(closes)
        excl = min(21, n - 20)
        main = closes.iloc[:n - excl]

        if len(main) < 20:
            return math.nan

        # Split remaining history into 4 quarters (oldest first)
        q_len = len(main) // 4
        if q_len < 5:
            # Fall back to simple 12m return
            start_price = float(closes.iloc[0])
            end_price   = float(closes.iloc[n - excl - 1])
            if start_price <= 0:
                return math.nan
            return end_price / start_price - 1.0

        q1_start = float(main.iloc[0])
        q1_end   = float(main.iloc[q_len - 1])
        q2_start = float(main.iloc[q_len])
        q2_end   = float(main.iloc[2 * q_len - 1])
        q3_start = float(main.iloc[2 * q_len])
        q3_end   = float(main.iloc[3 * q_len - 1])
        q4_start = float(main.iloc[3 * q_len])
        q4_end   = float(main.iloc[-1])

        def safe_ret(s, e):
            if s <= 0:
                return math.nan
            return e / s - 1.0

        q1 = safe_ret(q1_start, q1_end)
        q2 = safe_ret(q2_start, q2_end)
        q3 = safe_ret(q3_start, q3_end)
        q4 = safe_ret(q4_start, q4_end)

        if any(math.isnan(x) for x in [q1, q2, q3, q4]):
            # Fall back to simple return
            sp = float(main.iloc[0])
            ep = float(main.iloc[-1])
            if sp <= 0:
                return math.nan
            return ep / sp - 1.0

        return 0.4 * q4 + 0.2 * q3 + 0.2 * q2 + 0.2 * q1
    except Exception:
        return math.nan


def _near_52w_high(hist) -> float:
    """Current price / 52-week high * 100."""
    if hist is None or len(hist) < 5:
        return math.nan
    try:
        closes = hist["Close"].dropna()
        if len(closes) < 5:
            return math.nan
        current  = float(closes.iloc[-1])
        high_52w = float(closes.tail(min(252, len(closes))).max())
        if high_52w <= 0:
            return math.nan
        return current / high_52w * 100.0
    except Exception:
        return math.nan


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(raw: dict) -> dict:
    """Compute all screener metrics for one ticker. Returns flat dict."""
    ticker = raw["ticker"]
    base   = {"ticker": ticker}

    if raw.get("error"):
        base["fetch_error"] = raw["error"]
        return base

    info = raw.get("info") or {}
    hist = raw.get("hist")
    inc  = raw.get("inc")
    cf   = raw.get("cf")
    ed   = raw.get("ed")

    # ── Basic info ──────────────────────────────────────────────────────
    base["name"]   = info.get("longName") or info.get("shortName") or ticker
    base["sector"] = info.get("sector") or "Unknown"
    base["industry"] = info.get("industry") or ""

    # ── RS raw (cross-sectional ranking done later) ──────────────────────
    base["rs_raw"]          = _compute_rs_raw(hist)
    base["near_52w_high"]   = _near_52w_high(hist)

    # ── EPS Acceleration ────────────────────────────────────────────────
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    ni0 = _f(inc, ["Net Income", "Net Income Common Stockholders"], col=0)
    ni1 = _f(inc, ["Net Income", "Net Income Common Stockholders"], col=1)
    ni2 = _f(inc, ["Net Income", "Net Income Common Stockholders"], col=2)

    trailing_eps_info = info.get("trailingEps")

    def safe_eps(ni):
        if math.isnan(ni) or shares is None or shares <= 0:
            return math.nan
        return ni / shares

    eps0 = safe_eps(ni0)
    eps1 = safe_eps(ni1)
    eps2 = safe_eps(ni2)

    # Override eps0 with trailingEps if more reliable
    if trailing_eps_info is not None:
        try:
            eps0 = float(trailing_eps_info)
        except Exception:
            pass

    def eps_growth(e_new, e_old):
        # Require a meaningful prior-year base (≥$0.10) to prevent tiny-denominator distortion
        if math.isnan(e_new) or math.isnan(e_old) or e_old < 0.10 or e_new <= 0:
            return math.nan
        return min(e_new / e_old - 1.0, 3.0)   # cap at 300% to suppress base-effect outliers

    g_y0_y1 = eps_growth(eps0, eps1)   # recent: y1 -> y0
    g_y1_y2 = eps_growth(eps1, eps2)   # prior:  y2 -> y1

    eps_accel = False
    if not math.isnan(g_y0_y1) and not math.isnan(g_y1_y2):
        if g_y0_y1 > g_y1_y2 + 0.05 and g_y0_y1 > 0.15:
            eps_accel = True

    def pct_str(v):
        if math.isnan(v):
            return math.nan
        return v * 100.0

    base["eps_g_y1_y2_pct"] = pct_str(g_y1_y2)
    base["eps_g_y0_y1_pct"] = pct_str(g_y0_y1)
    base["eps_accel"]       = eps_accel

    # ── Estimate Revision ───────────────────────────────────────────────
    fwd_eps     = info.get("forwardEps")
    trail_eps   = info.get("trailingEps")
    hist_eg     = info.get("earningsGrowth")

    implied_fwd_growth   = math.nan
    revision_delta       = math.nan
    est_revision_pass    = False

    if fwd_eps is not None and trail_eps is not None:
        try:
            fwd_eps   = float(fwd_eps)
            trail_eps = float(trail_eps)
            if abs(trail_eps) > 1e-9:
                implied_fwd_growth = (fwd_eps - trail_eps) / abs(trail_eps)
        except Exception:
            pass

    if not math.isnan(implied_fwd_growth) and hist_eg is not None:
        try:
            hist_eg_f = float(hist_eg)
            revision_delta = implied_fwd_growth - hist_eg_f
            if revision_delta > 0.05 and implied_fwd_growth > 0.10:
                est_revision_pass = True
        except Exception:
            pass

    base["fwd_eps"]               = fwd_eps if fwd_eps is not None else math.nan
    base["trailing_eps"]          = trail_eps if trail_eps is not None else math.nan
    base["implied_fwd_growth_pct"] = pct_str(implied_fwd_growth)
    base["hist_eps_growth_pct"]   = pct_str(hist_eg) if hist_eg is not None else math.nan
    base["revision_delta_pct"]    = pct_str(revision_delta)
    base["est_revision_pass"]     = est_revision_pass

    # ── EPS Beat Rate ────────────────────────────────────────────────────
    eps_beats_n   = 0
    eps_quarters_n = 0
    beat_rate     = math.nan
    beat_rate_pass = False

    if ed is not None and not (hasattr(ed, 'empty') and ed.empty):
        try:
            ed_df = ed.copy()
            # Normalise column names
            col_map = {}
            for c in ed_df.columns:
                cl = str(c).lower()
                if "estimate" in cl:
                    col_map[c] = "estimate"
                elif "reported" in cl:
                    col_map[c] = "reported"
                elif "surprise" in cl:
                    col_map[c] = "surprise"
            ed_df = ed_df.rename(columns=col_map)

            if "reported" in ed_df.columns and "estimate" in ed_df.columns:
                ed_df = ed_df.dropna(subset=["reported"])
                ed_df = ed_df.head(4)
                eps_quarters_n = len(ed_df)
                if eps_quarters_n > 0:
                    beats = (ed_df["reported"].astype(float) > ed_df["estimate"].astype(float)).sum()
                    eps_beats_n = int(beats)
                    beat_rate   = eps_beats_n / eps_quarters_n * 100.0
                    if eps_beats_n >= 3:
                        beat_rate_pass = True
        except Exception:
            pass

    base["eps_beats_n"]    = eps_beats_n
    base["eps_quarters_n"] = eps_quarters_n
    base["beat_rate_pct"]  = beat_rate
    base["beat_rate_pass"] = beat_rate_pass

    # ── GARP ─────────────────────────────────────────────────────────────
    peg = info.get("trailingPegRatio")
    eg  = info.get("earningsGrowth")

    peg_f          = math.nan
    eps_growth_pct = math.nan
    garp_pass      = False

    if peg is not None:
        try:
            peg_f = float(peg)
        except Exception:
            pass

    if (math.isnan(peg_f) or peg_f <= 0) and eg is not None:
        try:
            pe = info.get("trailingPE")
            if pe is not None and eg is not None:
                eg_f = float(eg)
                pe_f = float(pe)
                if eg_f > 0 and pe_f > 0:
                    peg_f = pe_f / (eg_f * 100.0)
        except Exception:
            pass

    if eg is not None:
        try:
            eps_growth_pct = float(eg) * 100.0
        except Exception:
            pass

    if not math.isnan(peg_f) and peg_f > 0 and peg_f <= 1.5:
        if not math.isnan(eps_growth_pct) and eps_growth_pct >= 15.0:
            garp_pass = True

    base["peg_ratio"]      = peg_f
    base["eps_growth_pct"] = eps_growth_pct
    base["garp_pass"]      = garp_pass

    # ── FCF Growth (Compounder) ───────────────────────────────────────────
    def get_fcf(col_idx):
        op = _f(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], col=col_idx)
        cx = _f(cf, ["Capital Expenditure", "Capital Expenditures"], col=col_idx)
        if math.isnan(op) or math.isnan(cx):
            return math.nan
        return op + cx  # capex is negative in yfinance

    fcf0 = get_fcf(0)
    fcf1 = get_fcf(1)
    fcf2 = get_fcf(2)

    rev0 = _f(inc, ["Total Revenue"], col=0)
    rev1 = _f(inc, ["Total Revenue"], col=1)
    rev2 = _f(inc, ["Total Revenue"], col=2)

    def fcf_margin(fcf, rev):
        if math.isnan(fcf) or math.isnan(rev) or rev <= 0:
            return math.nan
        return fcf / rev * 100.0

    fm0 = fcf_margin(fcf0, rev0)
    fm1 = fcf_margin(fcf1, rev1)

    fcf_cagr = math.nan
    if not math.isnan(fcf2) and not math.isnan(fcf0) and fcf2 > 0 and fcf0 > 0:
        try:
            fcf_cagr = (fcf0 / fcf2) ** (1.0 / 2.0) - 1.0
        except Exception:
            pass

    fcf_margin_expanding = False
    if not math.isnan(fm0) and not math.isnan(fm1):
        fcf_margin_expanding = fm0 > fm1

    fcf_pass = False
    if not math.isnan(fcf_cagr) and fcf_cagr >= 0.20 and fcf_margin_expanding:
        fcf_pass = True

    base["fcf_y0"]              = fcf0
    base["fcf_y1"]              = fcf1
    base["fcf_y2"]              = fcf2
    base["fcf_margin_y0_pct"]   = fm0
    base["fcf_margin_y1_pct"]   = fm1
    base["fcf_cagr_3y_pct"]     = pct_str(fcf_cagr)
    base["fcf_pass"]            = fcf_pass

    # ── Gross Margin Expansion ────────────────────────────────────────────
    gp0 = _f(inc, ["Gross Profit"], col=0)
    gp1 = _f(inc, ["Gross Profit"], col=1)

    gm0 = math.nan if (math.isnan(gp0) or math.isnan(rev0) or rev0 <= 0) else gp0 / rev0 * 100.0
    gm1 = math.nan if (math.isnan(gp1) or math.isnan(rev1) or rev1 <= 0) else gp1 / rev1 * 100.0

    rev_growth = math.nan
    if not math.isnan(rev0) and not math.isnan(rev1) and rev1 > 0:
        rev_growth = (rev0 / rev1 - 1.0) * 100.0

    gm_delta = math.nan
    if not math.isnan(gm0) and not math.isnan(gm1):
        gm_delta = gm0 - gm1

    gm_expansion_pass = False
    if not math.isnan(rev_growth) and rev_growth >= 15.0:
        if not math.isnan(gm_delta) and gm_delta > 0:
            gm_expansion_pass = True

    base["rev_growth_pct"]     = rev_growth
    base["gm_y0_pct"]          = gm0
    base["gm_y1_pct"]          = gm1
    base["gm_delta_pct"]       = gm_delta
    base["gm_expansion_pass"]  = gm_expansion_pass

    # ── Screener pass count ───────────────────────────────────────────────
    # RS Rating and rs_pass are added after cross-sectional ranking
    base["rs_rating"]    = math.nan   # filled in after ranking
    base["rs_pass"]      = False      # filled in after ranking

    flags = [
        base["rs_pass"],
        eps_accel,
        est_revision_pass,
        beat_rate_pass,
        garp_pass,
        fcf_pass,
        gm_expansion_pass,
    ]
    base["screeners_passed"] = sum(1 for f in flags if f is True)

    return base


# ── RS cross-sectional ranking ─────────────────────────────────────────────────

def rank_rs(metrics_list: list[dict]) -> None:
    """In-place: add rs_rating (1–99 percentile) and rs_pass, update screeners_passed."""
    rs_vals = [(i, m["rs_raw"]) for i, m in enumerate(metrics_list)
               if not math.isnan(m.get("rs_raw", math.nan))]

    if not rs_vals:
        return

    idxs, vals = zip(*rs_vals)
    arr    = np.array(vals, dtype=float)
    n      = len(arr)
    ranks  = arr.argsort().argsort()  # 0-based rank
    # Map to 1–99
    pct    = np.clip(((ranks / (n - 1)) * 98 + 1).astype(int), 1, 99) if n > 1 else np.array([50] * n)

    for list_pos, (orig_idx, _) in enumerate(rs_vals):
        m = metrics_list[orig_idx]
        rs_r   = int(pct[list_pos])
        h52    = m.get("near_52w_high", math.nan)
        rs_p   = rs_r >= 80 and (not math.isnan(h52)) and h52 >= 75.0

        m["rs_rating"] = rs_r
        m["rs_pass"]   = rs_p

        # Recount screeners_passed now that rs_pass is known
        flags = [
            rs_p,
            m.get("eps_accel", False),
            m.get("est_revision_pass", False),
            m.get("beat_rate_pass", False),
            m.get("garp_pass", False),
            m.get("fcf_pass", False),
            m.get("gm_expansion_pass", False),
        ]
        m["screeners_passed"] = sum(1 for f in flags if f is True)

        badges = []
        if rs_p:                              badges.append("RS")
        if m.get("eps_accel"):                badges.append("ACCEL")
        if m.get("est_revision_pass"):        badges.append("REV")
        if m.get("beat_rate_pass"):           badges.append("BEAT")
        if m.get("garp_pass"):                badges.append("GARP")
        if m.get("fcf_pass"):                 badges.append("FCF")
        if m.get("gm_expansion_pass"):        badges.append("GM")
        m["screener_badges"] = badges


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_f(v, decimals=1, suffix=""):
    """Format float or return em-dash."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{float(v):.{decimals}f}{suffix}"
    except Exception:
        return "—"


def _fmt_pct(v, decimals=1):
    return _fmt_f(v, decimals, "%")


def _bool_badge(v):
    return '<span class="pass">&#10003;</span>' if v else '<span class="fail">&#8212;</span>'


# ── Deep-dive watchlist bar (mirrors magicFormula.py / growthScreener.py) ─────
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


# ── Help modal ────────────────────────────────────────────────────────────────
_HELP_CSS = """
.help-overlay{display:none;position:fixed;inset:0;z-index:9990;background:rgba(0,0,0,.65);backdrop-filter:blur(4px)}
.help-overlay.open{display:flex;align-items:center;justify-content:center}
.help-modal{background:#1a1e2e;border:1px solid #2d3348;border-radius:12px;width:660px;max-width:95vw;max-height:85vh;overflow-y:auto;padding:28px 32px;position:relative;box-shadow:0 24px 64px rgba(0,0,0,.6)}
.help-modal h2{font-size:16px;font-weight:700;color:#e2e8f0;margin:0 0 8px}
.help-desc{font-size:12px;color:#94a3b8;line-height:1.7;margin-bottom:16px}
.help-sec{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6366f1;margin:18px 0 8px;border-bottom:1px solid #252a3a;padding-bottom:6px;font-weight:700}
.help-tbl{width:100%;border-collapse:collapse;font-size:12px}
.help-tbl td{padding:6px 8px;border-bottom:1px solid #1e2234;vertical-align:top}
.help-tbl td:first-child{color:#e2e8f0;font-weight:600;white-space:nowrap;min-width:130px;padding-right:14px}
.help-tbl td:last-child{color:#94a3b8;line-height:1.6}
.help-close{position:absolute;top:14px;right:14px;background:none;border:1px solid #2d3348;border-radius:6px;color:#94a3b8;font-size:14px;cursor:pointer;padding:3px 10px}
.help-close:hover{color:#e2e8f0;border-color:#6366f1}
.help-btn{display:inline-flex;align-items:center;gap:5px;padding:5px 13px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.35);color:#a5b4fc;cursor:pointer;transition:opacity .15s;white-space:nowrap}
.help-btn:hover{opacity:.8}
"""

_HELP_JS = """
function openHelp(){document.getElementById('helpOverlay').classList.add('open')}
function closeHelp(){document.getElementById('helpOverlay').classList.remove('open')}
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeHelp()});
"""

_HELP_MODAL_MFG = """
<div id="helpOverlay" class="help-overlay" onclick="if(event.target===this)closeHelp()"><div class="help-modal">
<button class="help-close" onclick="closeHelp()">&#x2715;</button>
<h2>&#x24D8; Multi-Factor Growth &mdash; How It Works</h2>
<p class="help-desc">Runs 7 independent growth screeners simultaneously. Each is a separate pass/fail test; the <b>Passes</b> column shows how many a stock satisfies. Higher passes = more convergent growth signals across momentum, fundamentals, and valuation.</p>
<div class="help-sec">The 7 Screeners</div>
<table class="help-tbl">
<tr><td>1. RS Rating</td><td>Relative Strength &ge;70 vs. the index &mdash; stock is outperforming at least 70% of peers</td></tr>
<tr><td>2. EPS Acceleration</td><td>Quarter-over-quarter EPS growth trending upward &mdash; earnings momentum is building</td></tr>
<tr><td>3. Estimate Revision</td><td>Analyst consensus estimates revised upward recently &mdash; Wall St is raising its outlook</td></tr>
<tr><td>4. Beat Rate</td><td>Stock historically beats EPS estimates &ge;60% of the time &mdash; management guidance is conservative</td></tr>
<tr><td>5. GARP</td><td>Growth at a Reasonable Price: PEG &lt;2 and EPS growth &gt;15% &mdash; growth is not overpriced</td></tr>
<tr><td>6. FCF Compounder</td><td>Free cash flow growing and FCF margin &ge;10% &mdash; growth is backed by real cash generation</td></tr>
<tr><td>7. Gross Margin Expansion</td><td>Gross margin improving year-over-year &mdash; pricing power or improving unit economics</td></tr>
</table>
<div class="help-sec">Column Guide</div>
<table class="help-tbl">
<tr><td>Passes</td><td>Number of the 7 screeners passed. 6&ndash;7 = strong multi-factor convergence.</td></tr>
<tr><td>RS Rating</td><td>0&ndash;99 relative strength score vs. index peers</td></tr>
<tr><td>EPS Accel</td><td>EPS growth acceleration trend indicator</td></tr>
<tr><td>Rev Growth</td><td>Year-over-year revenue growth (%)</td></tr>
<tr><td>PEG</td><td>Forward P/E &divide; growth rate. &lt;1 = attractive, &lt;2 = acceptable</td></tr>
<tr><td>FCF Margin</td><td>Free cash flow &divide; revenue (%)</td></tr>
<tr><td>GM Delta</td><td>Year-over-year gross margin change &mdash; positive = expanding margins</td></tr>
</table>
<div class="help-sec">Color Coding</div>
<table class="help-tbl">
<tr><td style="color:#00d68f">6&ndash;7 passes (green)</td><td>Strong multi-factor convergence &mdash; growth signals are broad-based</td></tr>
<tr><td style="color:#4895ef">4&ndash;5 passes (blue)</td><td>Solid &mdash; most growth factors aligned</td></tr>
<tr><td style="color:#ffd166">2&ndash;3 passes (yellow)</td><td>Mixed signals &mdash; some growth present but not confirmed across factors</td></tr>
<tr><td style="color:#ff4d6d">0&ndash;1 passes (red)</td><td>Failing most screeners &mdash; weak growth profile</td></tr>
</table>
</div></div>
"""

# ── HTML report builder ────────────────────────────────────────────────────────

def build_html(metrics_list: list[dict], index_name: str) -> str:
    import os as _os
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    total   = len(metrics_list)
    suite_port = int(_os.environ.get("VALUATION_SUITE_PORT", 5050))

    # Screener pass counts
    sc = {
        "rs":    sum(1 for m in metrics_list if m.get("rs_pass")),
        "accel": sum(1 for m in metrics_list if m.get("eps_accel")),
        "rev":   sum(1 for m in metrics_list if m.get("est_revision_pass")),
        "beat":  sum(1 for m in metrics_list if m.get("beat_rate_pass")),
        "garp":  sum(1 for m in metrics_list if m.get("garp_pass")),
        "fcf":   sum(1 for m in metrics_list if m.get("fcf_pass")),
        "gm":    sum(1 for m in metrics_list if m.get("gm_expansion_pass")),
        "multi": sum(1 for m in metrics_list if m.get("screeners_passed", 0) >= 3),
    }

    # Sort by screeners_passed desc by default
    sorted_metrics = sorted(metrics_list, key=lambda m: m.get("screeners_passed", 0), reverse=True)

    # ── Summary cards ─────────────────────────────────────────────────
    def card(attr, label, count, threshold):
        return f"""
        <div class="summary-card" onclick="filterTable('{attr}')" id="card-{attr}">
          <div class="card-label">{label}</div>
          <div class="card-count">{count}</div>
          <div class="card-threshold">{threshold}</div>
        </div>"""

    cards_html = (
        card("multi",  "All Screens",         sc["multi"], "passing ≥3 screeners") +
        card("rs",     "RS Leaders",          sc["rs"],    "RS ≥80 &amp; &gt;75% of 52w High") +
        card("accel",  "EPS Acceleration",    sc["accel"], "Accel ≥5pp &amp; g &gt;15%") +
        card("rev",    "Estimate Revision",   sc["rev"],   "Fwd EPS Δ &gt;5pp vs trailing") +
        card("beat",   "EPS Beat Rate",       sc["beat"],  "≥3 beats in last 4 qtrs") +
        card("garp",   "GARP",                sc["garp"],  "PEG ≤1.5 &amp; EPS g ≥15%") +
        card("fcf",    "FCF Compounder",      sc["fcf"],   "FCF CAGR ≥20% + expanding margin") +
        card("gm",     "Gross Margin Exp",    sc["gm"],    "Rev g ≥15% &amp; GM expanding")
    )

    # ── Table rows ────────────────────────────────────────────────────
    def ticker_color(n):
        if n >= 3: return GOLD
        if n >= 2: return BLUE
        return SUBTEXT

    rows_html = []
    for m in sorted_metrics:
        n     = m.get("screeners_passed", 0)
        tc    = ticker_color(n)
        tick  = m.get("ticker", "")
        name  = m.get("name", tick)
        sect  = m.get("sector", "")

        rs_r  = m.get("rs_rating", math.nan)
        h52   = m.get("near_52w_high", math.nan)

        rs_color  = f'style="color:{GREEN}"' if isinstance(rs_r, (int, float)) and not (isinstance(rs_r, float) and math.isnan(rs_r)) and rs_r >= 80 else ""
        h52_color = f'style="color:{GREEN}"' if not (isinstance(h52, float) and math.isnan(h52)) and h52 >= 75 else ""

        eg_y0y1 = m.get("eps_g_y0_y1_pct", math.nan)
        eg_y1y2 = m.get("eps_g_y1_y2_pct", math.nan)
        accel   = m.get("eps_accel", False)
        eg_bold = "font-weight:bold;" if accel else ""

        def _pct_color(v, pos_green=True):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            try:
                fv = float(v)
                if pos_green:
                    return f'style="color:{"" if fv == 0 else (GREEN if fv > 0 else RED)}"'
                else:
                    return f'style="color:{"" if fv == 0 else (RED if fv > 0 else GREEN)}"'
            except Exception:
                return ""

        peg_f = m.get("peg_ratio", math.nan)
        peg_c = ""
        if not (isinstance(peg_f, float) and math.isnan(peg_f)) and peg_f is not None:
            try:
                peg_c = f'style="color:{GREEN}"' if float(peg_f) <= 1.5 and float(peg_f) > 0 else ""
            except Exception:
                pass

        fcf_cagr = m.get("fcf_cagr_3y_pct", math.nan)
        fcf_c    = f'style="color:{GREEN}"' if not (isinstance(fcf_cagr, float) and math.isnan(fcf_cagr)) and isinstance(fcf_cagr, (int, float)) and fcf_cagr >= 20 else ""

        # data attributes for filtering (1/0)
        d_rs    = "1" if m.get("rs_pass")            else "0"
        d_accel = "1" if m.get("eps_accel")          else "0"
        d_rev   = "1" if m.get("est_revision_pass")  else "0"
        d_beat  = "1" if m.get("beat_rate_pass")     else "0"
        d_garp  = "1" if m.get("garp_pass")          else "0"
        d_fcf   = "1" if m.get("fcf_pass")           else "0"
        d_gm    = "1" if m.get("gm_expansion_pass")  else "0"
        d_multi = "1" if n >= 3                      else "0"

        badge_html = f'<span class="passes-badge n{min(n,3)}">{n}/7</span>'

        rows_html.append(f"""
        <tr data-rs="{d_rs}" data-accel="{d_accel}" data-rev="{d_rev}"
            data-beat="{d_beat}" data-garp="{d_garp}" data-fcf="{d_fcf}"
            data-gm="{d_gm}" data-multi="{d_multi}" data-passes="{n}">
          <td class="cb-cell"><input type="checkbox" class="row-check" value="{tick}"></td>
          <td><span class="ticker-cell" style="color:{tc}">{tick}</span></td>
          <td class="name-cell">{name}</td>
          <td class="sector-cell">{sect}</td>
          <td {rs_color}>{_fmt_f(rs_r, 0)}</td>
          <td {h52_color}>{_fmt_pct(h52)}</td>
          <td {_pct_color(eg_y1y2)}>{_fmt_pct(eg_y1y2)}</td>
          <td {_pct_color(eg_y0y1)} style="{eg_bold}color:{'inherit' if not accel else GREEN}">{_fmt_pct(eg_y0y1)}</td>
          <td>{_bool_badge(accel)}</td>
          <td {_pct_color(m.get("revision_delta_pct", math.nan))}>{_fmt_pct(m.get("revision_delta_pct", math.nan))}</td>
          <td>{_bool_badge(m.get("est_revision_pass", False))}</td>
          <td>{m.get("eps_beats_n", 0)}/{m.get("eps_quarters_n", 0)}</td>
          <td>{_bool_badge(m.get("beat_rate_pass", False))}</td>
          <td {peg_c}>{_fmt_f(peg_f)}</td>
          <td {_pct_color(m.get("eps_growth_pct", math.nan))}>{_fmt_pct(m.get("eps_growth_pct", math.nan))}</td>
          <td>{_bool_badge(m.get("garp_pass", False))}</td>
          <td {fcf_c}>{_fmt_pct(fcf_cagr)}</td>
          <td>{_fmt_pct(m.get("fcf_margin_y0_pct", math.nan))}</td>
          <td>{_bool_badge(m.get("fcf_pass", False))}</td>
          <td {_pct_color(m.get("rev_growth_pct", math.nan))}>{_fmt_pct(m.get("rev_growth_pct", math.nan))}</td>
          <td {_pct_color(m.get("gm_delta_pct", math.nan))}>{_fmt_pct(m.get("gm_delta_pct", math.nan), 2)}</td>
          <td>{_bool_badge(m.get("gm_expansion_pass", False))}</td>
          <td>{badge_html}</td>
        </tr>""")

    table_body = "\n".join(rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Factor Growth — {index_name}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:      {BG};
    --panel:   {PANEL};
    --accent:  {ACCENT};
    --border:  {BORDER};
    --text:    {TEXT};
    --subtext: {SUBTEXT};
    --green:   {GREEN};
    --red:     {RED};
    --gold:    {GOLD};
    --blue:    {BLUE};
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    font-size: 13px;
    min-height: 100vh;
  }}

  /* Header */
  .header {{
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 18px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .header-left h1 {{
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.3px;
  }}
  .header-left .subtitle {{
    font-size: 0.78rem;
    color: var(--subtext);
    margin-top: 3px;
  }}
  .header-right {{
    font-size: 0.78rem;
    color: var(--subtext);
    text-align: right;
  }}

  /* Summary cards */
  .cards-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding: 18px 28px 0;
  }}
  .summary-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    min-width: 130px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    user-select: none;
  }}
  .summary-card:hover {{ border-color: var(--accent); }}
  .summary-card.active {{
    border-color: var(--accent);
    background: rgba(124, 106, 247, 0.12);
  }}
  .card-label {{
    font-size: 0.72rem;
    color: var(--subtext);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .card-count {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.2;
    margin: 2px 0;
  }}
  .card-threshold {{
    font-size: 0.7rem;
    color: var(--subtext);
  }}

  /* Filter pills */
  .filter-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    padding: 14px 28px;
  }}
  .filter-pill {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.76rem;
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
    color: var(--subtext);
  }}
  .filter-pill:hover {{ border-color: var(--accent); color: var(--text); }}
  .filter-pill.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
  }}

  /* Controls row */
  .controls-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 28px 10px;
    flex-wrap: wrap;
  }}
  .row-count {{
    font-size: 0.78rem;
    color: var(--subtext);
    margin-left: auto;
  }}
  .btn {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 14px;
    color: var(--text);
    font-size: 0.78rem;
    cursor: pointer;
    transition: border-color 0.15s;
  }}
  .btn:hover {{ border-color: var(--accent); }}

  /* Table wrapper */
  .table-wrap {{
    padding: 0 28px 40px;
    overflow-x: auto;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    min-width: 1600px;
  }}
  col.c-ticker  {{ width: 72px; }}
  col.c-name    {{ width: 160px; }}
  col.c-sector  {{ width: 130px; }}
  col.c-num     {{ width: 66px; }}
  col.c-check   {{ width: 46px; }}
  col.c-passes  {{ width: 56px; }}

  thead th {{
    position: sticky;
    top: 0;
    background: var(--panel);
    border-bottom: 2px solid var(--border);
    padding: 9px 6px;
    text-align: right;
    font-size: 0.7rem;
    color: var(--subtext);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  thead th:first-child,
  thead th:nth-child(2),
  thead th:nth-child(3) {{
    text-align: left;
  }}
  thead th:hover {{ color: var(--accent); }}
  thead th.sort-asc::after  {{ content: " ↑"; color: var(--accent); }}
  thead th.sort-desc::after {{ content: " ↓"; color: var(--accent); }}

  tbody tr {{
    border-bottom: 1px solid rgba(42, 42, 74, 0.5);
    transition: background 0.1s;
  }}
  tbody tr:hover {{ background: rgba(124, 106, 247, 0.06); }}
  tbody tr.hidden {{ display: none; }}

  td {{
    padding: 7px 6px;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    vertical-align: middle;
  }}
  td:first-child,
  td:nth-child(2),
  td:nth-child(3) {{ text-align: left; }}

  .ticker-cell {{ font-weight: 700; font-size: 0.82rem; letter-spacing: 0.3px; }}
  .name-cell   {{ font-size: 0.72rem; color: var(--subtext); }}
  .sector-cell {{ font-size: 0.72rem; color: var(--subtext); }}

  .pass {{ color: var(--green); font-size: 0.9rem; }}
  .fail {{ color: var(--subtext); }}

  .passes-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 700;
    background: rgba(136, 136, 170, 0.15);
    color: var(--subtext);
  }}
  .passes-badge.n1 {{ background: rgba(96, 165, 250, 0.15); color: var(--blue); }}
  .passes-badge.n2 {{ background: rgba(96, 165, 250, 0.20); color: var(--blue); }}
  .passes-badge.n3 {{ background: rgba(251, 191, 36, 0.18); color: var(--gold); }}

  /* Footer */
  .footer {{
    padding: 20px 28px;
    font-size: 0.7rem;
    color: var(--subtext);
    border-top: 1px solid var(--border);
    line-height: 1.7;
  }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

  /* ── Deep-dive checkboxes ── */
  .cb-th  {{ width: 28px; text-align: center; padding: 4px 2px; }}
  .cb-cell {{ width: 28px; text-align: center; padding: 4px 2px; }}
  input.row-check {{ cursor: pointer; width: 14px; height: 14px; accent-color: #7c6af7; }}
</style>
<style>{_WL_CSS}{_HELP_CSS}</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <h1>Multi-Factor Growth &mdash; {index_name}</h1>
    <div class="subtitle">7-factor growth analysis &bull; {total} constituents &bull; {ts}</div>
  </div>
  <div class="header-right">
    Generated {ts}<br>
    yfinance &bull; cross-sectional RS ranking
  </div>
</div>

<!-- Summary cards -->
<div class="cards-row" id="cards-row">
  {cards_html}
</div>

<!-- Filter pills -->
<div class="filter-row" id="filter-row">
  <span class="filter-pill active" onclick="filterByPill(this, null)">All ({total})</span>
  <span class="filter-pill" onclick="filterByPill(this, 'multi')">≥3 Screens ({sc["multi"]})</span>
  <span class="filter-pill" onclick="filterByPill(this, 'rs')">RS Leaders ({sc["rs"]})</span>
  <span class="filter-pill" onclick="filterByPill(this, 'accel')">EPS Accel ({sc["accel"]})</span>
  <span class="filter-pill" onclick="filterByPill(this, 'rev')">Est Revision ({sc["rev"]})</span>
  <span class="filter-pill" onclick="filterByPill(this, 'beat')">Beat Rate ({sc["beat"]})</span>
  <span class="filter-pill" onclick="filterByPill(this, 'garp')">GARP ({sc["garp"]})</span>
  <span class="filter-pill" onclick="filterByPill(this, 'fcf')">FCF Compounder ({sc["fcf"]})</span>
  <span class="filter-pill" onclick="filterByPill(this, 'gm')">Gross Margin Exp ({sc["gm"]})</span>
</div>

<!-- Controls row -->
<div class="controls-row">
  <button class="btn" onclick="exportCSV()">Export CSV</button>
  <button class="help-btn" onclick="openHelp()">&#x24D8; How it works</button>
  <span class="row-count" id="row-count">Showing {total} of {total}</span>
</div>

<!-- Table -->
<div class="table-wrap">
<table id="main-table">
  <colgroup>
    <col class="c-ticker">
    <col class="c-name">
    <col class="c-sector">
    <col class="c-num">  <!-- RS -->
    <col class="c-num">  <!-- 52w -->
    <col class="c-num">  <!-- EPS g Y-2→Y-1 -->
    <col class="c-num">  <!-- EPS g Y-1→Y0 -->
    <col class="c-check"><!-- Accel -->
    <col class="c-num">  <!-- Fwd EPS Δ -->
    <col class="c-check"><!-- Est Rev -->
    <col class="c-num">  <!-- EPS Beats -->
    <col class="c-check"><!-- Beat Rate -->
    <col class="c-num">  <!-- PEG -->
    <col class="c-num">  <!-- EPS g% -->
    <col class="c-check"><!-- GARP -->
    <col class="c-num">  <!-- FCF CAGR -->
    <col class="c-num">  <!-- FCF Margin -->
    <col class="c-check"><!-- FCF -->
    <col class="c-num">  <!-- Rev g% -->
    <col class="c-num">  <!-- GM Δ -->
    <col class="c-check"><!-- GM Exp -->
    <col class="c-passes">
  </colgroup>
  <thead>
    <tr>
      <th class="cb-th"><input type="checkbox" id="cb-all" title="Select all visible"></th>
      <th onclick="sortTable(0,'str')">Ticker</th>
      <th onclick="sortTable(1,'str')">Company</th>
      <th onclick="sortTable(2,'str')">Sector</th>
      <th onclick="sortTable(3,'num')">RS</th>
      <th onclick="sortTable(4,'num')">52w High%</th>
      <th onclick="sortTable(5,'num')">EPS g Y-2→Y-1</th>
      <th onclick="sortTable(6,'num')">EPS g Y-1→Y0</th>
      <th onclick="sortTable(7,'str')">Accel</th>
      <th onclick="sortTable(8,'num')">Fwd EPS Δ</th>
      <th onclick="sortTable(9,'str')">Est Rev</th>
      <th onclick="sortTable(10,'str')">EPS Beats</th>
      <th onclick="sortTable(11,'str')">Beat Rate</th>
      <th onclick="sortTable(12,'num')">PEG</th>
      <th onclick="sortTable(13,'num')">EPS g%</th>
      <th onclick="sortTable(14,'str')">GARP</th>
      <th onclick="sortTable(15,'num')">FCF CAGR 3Y</th>
      <th onclick="sortTable(16,'num')">FCF Margin</th>
      <th onclick="sortTable(17,'str')">FCF</th>
      <th onclick="sortTable(18,'num')">Rev g%</th>
      <th onclick="sortTable(19,'num')">GM Δ pp</th>
      <th onclick="sortTable(20,'str')">GM Exp</th>
      <th onclick="sortTable(21,'num')">Passes</th>
    </tr>
  </thead>
  <tbody id="table-body">
    {table_body}
  </tbody>
</table>
</div>

<!-- Footer -->
<div class="footer">
  <strong>Notes:</strong><br>
  EPS acceleration uses annual income statement data (yfinance). Acceleration = recent EPS growth exceeds prior year growth by ≥5pp AND recent growth &gt;15%.<br>
  Estimate revision is approximated from forward vs trailing EPS — for precise revision history, use a premium data provider.<br>
  Revenue beat rate requires earnings surprise data not available in yfinance; EPS beat rate shown only (last 4 reported quarters).<br>
  RS Rating computed as cross-sectional percentile rank (1–99) of IBD-style quarterly-weighted 12-month return, excluding the most recent month.<br>
  GARP: PEG ratio from yfinance trailingPegRatio; falls back to trailingPE / (earningsGrowth × 100) when unavailable.<br>
  FCF CAGR = 2-year CAGR of operating cash flow minus capex. FCF margin = FCF / total revenue.
</div>

<script>
// ── Current filter state ────────────────────────────────────────────────────
let _currentAttr = null;

function filterTable(attr) {{
  _currentAttr = attr;
  const tbody = document.getElementById('table-body');
  const rows  = tbody.querySelectorAll('tr');
  let shown = 0;
  rows.forEach(r => {{
    let vis = false;
    if (attr === null) {{
      vis = true;
    }} else if (attr === 'multi') {{
      vis = parseInt(r.dataset.passes || '0') >= 3;
    }} else {{
      vis = r.dataset[attr] === '1';
    }}
    r.classList.toggle('hidden', !vis);
    if (vis) shown++;
  }});
  document.getElementById('row-count').textContent = 'Showing ' + shown + ' of {total}';

  // Highlight active card
  document.querySelectorAll('.summary-card').forEach(c => c.classList.remove('active'));
  if (attr !== null) {{
    const card = document.getElementById('card-' + attr);
    if (card) card.classList.add('active');
  }}
}}

function filterByPill(pillEl, attr) {{
  document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
  pillEl.classList.add('active');
  filterTable(attr);
  _applyTabSort(attr);   // re-sort by the most relevant metric for this tab
}}

// ── Per-tab default sort ─────────────────────────────────────────────────────
// col indices map to cells[] in the tbody rows (0 = checkbox, 1 = Ticker, ...)
const _TAB_CFG = {{
  'null':  {{ by:'passes' }},
  'multi': {{ by:'passes' }},
  'rs':    {{ col:4,  asc:false }},           // RS desc
  'accel': {{ col:7,  asc:false }},           // EPS g Y-1→Y0 desc
  'rev':   {{ col:9,  asc:false }},           // Fwd EPS Δ desc
  'beat':  {{ col:11, asc:false, tp:'beat' }}, // EPS beats numerator desc
  'garp':  {{ col:13, asc:true  }},           // PEG asc (lower = better)
  'fcf':   {{ col:16, asc:false }},           // FCF CAGR desc
  'gm':    {{ col:20, asc:false }},           // GM Δ pp desc
}};

function _cellVal(row, col, tp) {{
  if (!row.cells[col]) return tp === 'beat' ? 0 : NaN;
  const t = row.cells[col].textContent.trim();
  if (tp === 'beat') return parseInt(t.split('/')[0]) || 0;
  return parseFloat(t.replace(/[^0-9.\-]/g, ''));
}}

function _applyTabSort(attr) {{
  const cfg = _TAB_CFG[String(attr)] || {{ by:'passes' }};
  const tbody = document.getElementById('table-body');
  const rows  = Array.from(tbody.querySelectorAll('tr'));

  rows.sort((a, b) => {{
    if (cfg.by === 'passes') {{
      return (parseInt(b.dataset.passes) || 0) - (parseInt(a.dataset.passes) || 0);
    }}
    let av = _cellVal(a, cfg.col, cfg.tp);
    let bv = _cellVal(b, cfg.col, cfg.tp);
    const nanVal = cfg.asc ? Infinity : -Infinity;
    if (isNaN(av)) av = nanVal;
    if (isNaN(bv)) bv = nanVal;
    return cfg.asc ? av - bv : bv - av;
  }});

  rows.forEach(r => tbody.appendChild(r));

  // Update header sort indicators
  _sortCol = -1; _sortAsc = !cfg.asc;
  document.querySelectorAll('thead th').forEach((th, i) => {{
    th.classList.remove('sort-asc', 'sort-desc');
    if (cfg.col != null && i === cfg.col) {{
      th.classList.add(cfg.asc ? 'sort-asc' : 'sort-desc');
    }}
  }});
}}

// ── Sorting ─────────────────────────────────────────────────────────────────
let _sortCol = -1;
let _sortAsc = true;

function sortTable(colIdx, type) {{
  const tbody = document.getElementById('table-body');
  const rows  = Array.from(tbody.querySelectorAll('tr'));

  if (_sortCol === colIdx) {{
    _sortAsc = !_sortAsc;
  }} else {{
    _sortCol = colIdx;
    // Numeric columns default to descending (highest first) on first click
    _sortAsc = (type !== 'num');
  }}

  rows.sort((a, b) => {{
    // +1 offset: cells[0] is the checkbox, data starts at cells[1]
    const cellIdx = colIdx + 1;
    const ac = a.cells[cellIdx] ? a.cells[cellIdx].textContent.trim() : '';
    const bc = b.cells[cellIdx] ? b.cells[cellIdx].textContent.trim() : '';
    if (type === 'num') {{
      // For "X/7" badges, strip the "/7" so we sort on X not "X7"
      const cleanA = ac.replace(/\/\d+/, '').replace(/[^0-9.\-]/g, '');
      const cleanB = bc.replace(/\/\d+/, '').replace(/[^0-9.\-]/g, '');
      const an = cleanA !== '' ? parseFloat(cleanA) : (_sortAsc ? Infinity : -Infinity);
      const bn = cleanB !== '' ? parseFloat(cleanB) : (_sortAsc ? Infinity : -Infinity);
      return _sortAsc ? an - bn : bn - an;
    }} else {{
      return _sortAsc ? ac.localeCompare(bc) : bc.localeCompare(ac);
    }}
  }});

  rows.forEach(r => tbody.appendChild(r));

  // Update header indicators
  document.querySelectorAll('thead th').forEach((th, i) => {{
    th.classList.remove('sort-asc', 'sort-desc');
    if (i === colIdx) th.classList.add(_sortAsc ? 'sort-asc' : 'sort-desc');
  }});

  // Re-apply filter
  filterTable(_currentAttr);
}}

// ── Export CSV ───────────────────────────────────────────────────────────────
function exportCSV() {{
  const table  = document.getElementById('main-table');
  const rows   = table.querySelectorAll('tr:not(.hidden)');
  const lines  = [];
  rows.forEach(r => {{
    const cells = Array.from(r.querySelectorAll('th, td')).map(c =>
      '"' + c.textContent.trim().replace(/"/g, '""') + '"'
    );
    lines.push(cells.join(','));
  }});
  const blob = new Blob([lines.join('\\n')], {{type: 'text/csv'}});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'growth_screeners_{index_name}_{ts.replace(":", "-").replace(" ", "_")}.csv';
  a.click();
  URL.revokeObjectURL(url);
}}

// ── Init row count ───────────────────────────────────────────────────────────
(function() {{
  document.getElementById('row-count').textContent = 'Showing {total} of {total}';
}})();
</script>

<script>{_wl_js(suite_port)}</script>
<script>{_HELP_JS}</script>
{_WL_BAR}
{_HELP_MODAL_MFG}
</body>
</html>"""

    return html


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Multi-Factor Growth — 7-factor growth analysis")
    p.add_argument("--index",   default="SPX",
                   choices=["SPX", "NDX", "DOW", "RUT", "TSX"],
                   help="Index to screen (default: SPX)")
    p.add_argument("--tickers", default="",
                   help="Comma-separated ticker override (skips index fetch)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Suppress yfinance's "No earnings dates found, symbol may be delisted"
    # logger — fires for every ticker without a scheduled upcoming earnings date,
    # which is the majority of the index at any given time.
    import logging as _logging
    _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)

    print(f"\n=== Multi-Factor Growth === {args.index} ===")

    # 1. Get tickers
    if args.tickers.strip():
        ticker_dicts = [
            {"ticker": t.strip().upper(), "name": t.strip().upper(), "sector": "Unknown"}
            for t in args.tickers.split(",")
            if t.strip()
        ]
        print(f"  Using {len(ticker_dicts)} custom tickers.")
    else:
        ticker_dicts = get_index_tickers(args.index)
        if not ticker_dicts:
            print("  ERROR: Could not fetch index constituents. Exiting.")
            sys.exit(1)
        print(f"  Found {len(ticker_dicts)} constituents.")

    tickers = [d["ticker"] for d in ticker_dicts]
    ticker_meta = {d["ticker"]: d for d in ticker_dicts}

    total  = len(tickers)
    done   = 0
    raws   = []

    # 2. Parallel fetch
    print(f"  Fetching data for {total} tickers (8 threads)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        future_map = {ex.submit(_fetch_ticker, t): t for t in tickers}
        for fut in as_completed(future_map):
            raw = fut.result()
            # Merge meta (name/sector from index source if info is missing)
            tick = raw.get("ticker", "")
            meta = ticker_meta.get(tick, {})
            if raw.get("info") is not None:
                if not raw["info"].get("longName") and meta.get("name"):
                    raw["info"]["longName"] = meta["name"]
                if not raw["info"].get("sector") and meta.get("sector"):
                    raw["info"]["sector"] = meta["sector"]
            raws.append(raw)
            done += 1
            print(f"  [{done}/{total}] {tick}          ", end="\r", flush=True)

    print(f"\n  Fetch complete. {sum(1 for r in raws if not r.get('error'))} succeeded, "
          f"{sum(1 for r in raws if r.get('error'))} failed.")

    # 3. Compute metrics
    print("  Computing metrics...")
    metrics_list = []
    for raw in raws:
        try:
            m = compute_metrics(raw)
        except Exception as e:
            m = {"ticker": raw.get("ticker", "?"), "fetch_error": str(e), "screeners_passed": 0}
        metrics_list.append(m)

    # RS cross-sectional ranking
    rank_rs(metrics_list)

    # 4. Summary
    sc_names = ["RS Leaders", "EPS Accel", "Est Revision", "Beat Rate", "GARP", "FCF Compounder", "Gross Margin Exp"]
    sc_keys  = ["rs_pass", "eps_accel", "est_revision_pass", "beat_rate_pass", "garp_pass", "fcf_pass", "gm_expansion_pass"]
    print("\n  Screener summary:")
    for name, key in zip(sc_names, sc_keys):
        n = sum(1 for m in metrics_list if m.get(key))
        print(f"    {name:<24} {n:>4} / {total}")
    multi = sum(1 for m in metrics_list if m.get("screeners_passed", 0) >= 3)
    print(f"    {'Passing ≥3 screeners':<24} {multi:>4} / {total}")

    # 5. Build HTML
    print("\n  Building HTML report...")
    html = build_html(metrics_list, args.index)

    # 6. Save
    ts_file = datetime.now().strftime("%Y_%m_%d_%H%M")
    out_name = f"growth_screeners_{args.index}_{ts_file}.html"
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    kb = os.path.getsize(out_path) // 1024
    print(f"\n  Saved: {out_path}  ({kb} KB)")

    # 7. Open browser
    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        try:
            import webbrowser
            webbrowser.open(f"file://{out_path}")
        except Exception:
            try:
                subprocess.Popen(["open", out_path])
            except Exception:
                pass


if __name__ == "__main__":
    main()
