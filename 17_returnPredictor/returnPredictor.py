"""
Return Predictor
================
XGBoost cross-sectional factor model. Trains on quarterly factor snapshots
(momentum, valuation, quality) and ranks the current universe by predicted
forward return. Uses walk-forward validation to estimate out-of-sample IC.

  --index   SPX | NDX | DOW | RUT    (default: SPX)
  --tickers comma-separated override
  --horizon trading days ahead        (default: 63)
  --top     N  limit by market cap    (default: 150)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests as _requests
import yfinance as yf
from scipy.stats import rankdata, spearmanr
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Directories ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "predictorData")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
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

    Accepts any of: ticker, symbol (column name, case-insensitive).
    Optional weight/weights/wt column is stored and shown in the output.
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
        # no recognised header — treat first column as ticker
        plain = list(_csv.reader(raw))
        start = 1 if plain and not plain[0][0].strip().replace(".", "").isdigit() else 0
        rows = []
        for row in plain[start:]:
            t = row[0].strip().upper() if row else ""
            if t:
                rows.append({"ticker": t, "weight": None, "name": t, "sector": "Unknown"})
        return rows
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
    print(f"  Fetching {label} constituents from slickcharts.com...", flush=True)
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
        print(f"  {label} slickcharts fetch failed: {e}", flush=True)
        return []


def _get_spx() -> list[dict]:
    return _get_slickcharts("https://www.slickcharts.com/sp500", "S&P 500")


def _get_ndx() -> list[dict]:
    return _get_slickcharts("https://www.slickcharts.com/nasdaq100", "Nasdaq-100")


def _get_dow() -> list[dict]:
    return _get_slickcharts("https://www.slickcharts.com/dowjones", "Dow Jones 30")


def _get_rut() -> list[dict]:
    """Russell 2000 from iShares IWM ETF holdings CSV — authoritative and current."""
    print("  Fetching Russell 2000 constituents from iShares IWM holdings...", flush=True)
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
        print(f"  RUT iShares fetch failed: {e}", flush=True)
        return []


# ── Feature computation ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "mom_21", "mom_63", "mom_126", "mom_252",
    "vol_21", "ma50_ratio", "ma200_ratio", "hi52_ratio",
    "pe", "pb", "ps", "roe", "gross_margin",
    "rev_growth", "eps_growth", "debt_eq",
]


def _safe_close(hist: pd.DataFrame) -> pd.Series:
    """Return the Close column as a 1-D Series regardless of yfinance version."""
    col = hist["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col.squeeze()


def _price_features(hist: pd.DataFrame, as_of_idx: int) -> dict:
    """Compute price-based features from history ending at as_of_idx."""
    close = _safe_close(hist).iloc[: as_of_idx + 1]
    if len(close) < 10:
        return {}

    def _mom(n: int) -> float:
        if len(close) < n + 1:
            return np.nan
        return float(close.iloc[-1] / close.iloc[-n - 1] - 1) * 100

    def _sma(n: int) -> float:
        if len(close) < n:
            return np.nan
        return float(close.iloc[-n:].mean())

    last = float(close.iloc[-1])

    sma50  = _sma(50)
    sma200 = _sma(200)
    hi252  = float(close.iloc[-min(252, len(close)):].max())

    vol_series = close.pct_change().dropna()
    if len(vol_series) >= 21:
        vol21 = float(vol_series.iloc[-21:].std() * (252 ** 0.5) * 100)
    else:
        vol21 = np.nan

    return {
        "mom_21":     _mom(21),
        "mom_63":     _mom(63),
        "mom_126":    _mom(126),
        "mom_252":    _mom(252),
        "vol_21":     vol21,
        "ma50_ratio":  last / sma50  if sma50  and sma50  > 0 else np.nan,
        "ma200_ratio": last / sma200 if sma200 and sma200 > 0 else np.nan,
        "hi52_ratio":  last / hi252  if hi252  and hi252  > 0 else np.nan,
    }


def _fundamental_features(info: dict) -> dict:
    """Extract fundamental features from yfinance info dict."""
    def _cap(v, lo, hi):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return max(lo, min(hi, float(v)))

    pe          = _cap(info.get("trailingPE"),                       -50, 200)
    pb          = info.get("priceToBook")
    ps          = info.get("priceToSalesTrailingTwelveMonths")
    roe_raw     = info.get("returnOnEquity")
    gm_raw      = info.get("grossMargins")
    rev_raw     = info.get("revenueGrowth")
    eps_raw     = info.get("earningsGrowth")
    deq_raw     = info.get("debtToEquity")

    def _pct(v):
        if v is None:
            return np.nan
        try:
            return float(v) * 100
        except Exception:
            return np.nan

    def _f(v):
        if v is None:
            return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan

    return {
        "pe":          pe,
        "pb":          _f(pb),
        "ps":          _f(ps),
        "roe":         _pct(roe_raw),
        "gross_margin": _pct(gm_raw),
        "rev_growth":  _pct(rev_raw),
        "eps_growth":  _pct(eps_raw),
        "debt_eq":     _f(deq_raw),
    }


# ── Per-ticker data fetch ──────────────────────────────────────────────────────

def _fetch_info(ticker: str) -> tuple[str, dict]:
    """Fetch yfinance info dict for one ticker (used with low-concurrency pool)."""
    try:
        return ticker, yf.Ticker(ticker).info or {}
    except Exception:
        return ticker, {}


def fetch_all_data(tickers: list[str]) -> dict[str, dict]:
    """
    Batch-download 2.5 years of price history for all tickers in one call,
    then fetch .info dicts with a small thread pool to avoid crumb conflicts.
    Returns dict keyed by ticker.
    """
    import time as _time
    end   = datetime.today()
    start = end - timedelta(days=int(2.5 * 365))

    # ── Price history: one batch call avoids per-ticker session/crumb issues ──
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
        group_by="ticker",
    )

    hists: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                h = raw[t].copy().dropna(how="all")
                if len(h) >= 30:
                    hists[t] = h
            except KeyError:
                pass
    else:
        # Single ticker — raw is already flat
        if len(tickers) == 1 and not raw.empty and len(raw) >= 30:
            hists[tickers[0]] = raw.copy()

    # ── Info: low concurrency to keep crumb state consistent ──────────────────
    infos: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for t, info in pool.map(_fetch_info, list(hists.keys())):
            infos[t] = info
            _time.sleep(0.05)   # gentle pacing

    result: dict[str, dict] = {}
    for t, hist in hists.items():
        info   = infos.get(t, {})
        mcap   = info.get("marketCap", 0) or 0
        sector = info.get("sector", "Unknown") or "Unknown"
        result[t] = {"ticker": t, "hist": hist, "info": info,
                     "mcap": mcap, "sector": sector}
    return result


# ── Training data construction ─────────────────────────────────────────────────

def _trading_day_index(hist: pd.DataFrame, offset: int) -> int:
    """Return the integer index that is `offset` trading days from the end."""
    idx = len(hist) - 1 - offset
    return max(0, idx)


def build_training_rows(ticker_data: dict, horizon: int, n_snapshots: int = 7) -> list[dict]:
    """
    For each of n_snapshots quarter-end dates (spaced ~63 td apart going back),
    compute features + forward-return label.
    Returns list of row dicts with 'ticker', 'snapshot', all features, and 'label'.
    """
    hist   = ticker_data["hist"]
    info   = ticker_data["info"]
    ticker = ticker_data["ticker"]
    n      = len(hist)

    rows = []
    for snap_i in range(n_snapshots):
        # snapshot index: counting back from end in multiples of 63
        as_of_offset = snap_i * 63
        as_of_idx    = n - 1 - as_of_offset
        if as_of_idx < 60:
            continue

        # forward return label: price horizon days after snapshot
        fwd_idx = as_of_idx + horizon
        if fwd_idx >= n:
            continue  # not enough future data

        close_s = _safe_close(hist)
        p_now = float(close_s.iloc[as_of_idx])
        p_fwd = float(close_s.iloc[fwd_idx])
        if p_now <= 0:
            continue
        fwd_return = (p_fwd / p_now - 1) * 100

        pf = _price_features(hist, as_of_idx)
        ff = _fundamental_features(info)

        row = {"ticker": ticker, "snapshot": snap_i, "label": fwd_return}
        row.update(pf)
        row.update(ff)
        rows.append(row)

    return rows


def cross_sectional_rank(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalize column values to 0–1 percentile rank within df."""
    vals = df[col].values.copy().astype(float)
    mask = ~np.isnan(vals)
    out  = np.full(len(vals), np.nan)
    if mask.sum() > 1:
        ranks = rankdata(vals[mask], method="average")
        out[mask] = (ranks - 1) / (mask.sum() - 1)
    elif mask.sum() == 1:
        out[mask] = 0.5
    return pd.Series(out, index=df.index)


# ── Walk-forward validation ────────────────────────────────────────────────────

def walk_forward_ic(df: pd.DataFrame, feature_cols: list[str], model_params: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Sort snapshot indices chronologically (0 = most recent history, higher = older).
    Use first 4 unique snapshots for train, last 2 for test.
    Returns (Spearman IC, test_pred, test_actual).
    """
    snaps = sorted(df["snapshot"].unique(), reverse=True)  # oldest first (highest snap_i)
    if len(snaps) < 3:
        return np.nan, np.array([]), np.array([])

    train_snaps = snaps[: max(4, len(snaps) - 2)]
    test_snaps  = snaps[max(4, len(snaps) - 2):]

    if not test_snaps:
        test_snaps  = snaps[-2:]
        train_snaps = snaps[:-2]

    train_df = df[df["snapshot"].isin(train_snaps)].copy()
    test_df  = df[df["snapshot"].isin(test_snaps)].copy()

    train_df = train_df.dropna(subset=feature_cols + ["label_rank"])
    test_df  = test_df.dropna(subset=feature_cols + ["label_rank"])

    if len(train_df) < 20 or len(test_df) < 5:
        return np.nan, np.array([]), np.array([])

    X_train = train_df[feature_cols].values
    y_train = train_df["label_rank"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label_rank"].values

    m = XGBRegressor(**model_params)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)

    ic, _ = spearmanr(pred, y_test)
    return float(ic), pred, y_test


# ── HTML generation ────────────────────────────────────────────────────────────

def _fmt(v, decimals=1, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}{suffix}"


def _quartile_label(score: float) -> tuple[str, str]:
    """Return (label, color) for a 0–100 score."""
    if score >= 75:
        return "Top 25%", GREEN
    elif score >= 50:
        return "Upper Mid", ACCENT
    elif score >= 25:
        return "Lower Mid", YELLOW
    else:
        return "Bottom 25%", RED


def generate_html(
    ranked: pd.DataFrame,
    index_name: str,
    horizon: int,
    ic: float,
    feature_importances: dict,
    test_pred: np.ndarray,
    test_actual: np.ndarray,
    out_path: str,
) -> None:
    today_str = datetime.today().strftime("%B %d, %Y")
    ic_str    = f"{ic:.3f}" if not np.isnan(ic) else "N/A"
    n_tickers = len(ranked)

    # Build table rows
    table_rows = []
    for rank_i, row in enumerate(ranked.itertuples(), 1):
        score    = getattr(row, "score", 0.0)
        qlabel, qcolor = _quartile_label(score)
        ticker   = row.ticker
        sector   = getattr(row, "sector", "—")
        mom63    = _fmt(getattr(row, "mom_63", np.nan), 1, "%")
        pe_val   = _fmt(getattr(row, "pe", np.nan), 1)
        roe_val  = _fmt(getattr(row, "roe", np.nan), 1, "%")
        gm_val   = _fmt(getattr(row, "gross_margin", np.nan), 1, "%")

        score_color = GREEN if score >= 60 else (YELLOW if score >= 40 else RED)

        table_rows.append(f"""
        <tr>
          <td style="color:{MUTED};text-align:center">{rank_i}</td>
          <td style="font-weight:700;color:{TEXT}">{ticker}</td>
          <td style="color:{MUTED}">{sector}</td>
          <td style="text-align:right;color:{score_color};font-weight:700">{score:.1f}</td>
          <td style="text-align:center"><span style="background:{qcolor}22;color:{qcolor};padding:2px 8px;border-radius:9999px;font-size:0.75rem">{qlabel}</span></td>
          <td style="text-align:right;color:{TEXT}">{mom63}</td>
          <td style="text-align:right;color:{TEXT}">{pe_val}</td>
          <td style="text-align:right;color:{TEXT}">{roe_val}</td>
          <td style="text-align:right;color:{TEXT}">{gm_val}</td>
        </tr>""")

    table_html = "\n".join(table_rows)

    # Feature importance chart data
    fi_items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    fi_labels = [f'"{k}"' for k, _ in fi_items]
    fi_values = [f"{v:.4f}" for _, v in fi_items]
    fi_labels_js = "[" + ", ".join(fi_labels) + "]"
    fi_values_js = "[" + ", ".join(fi_values) + "]"

    # Backtest scatter data
    scatter_pts = ""
    if len(test_pred) > 0 and len(test_actual) > 0:
        pts = [f"{{x:{float(p):.4f},y:{float(a):.2f}}}" for p, a in zip(test_pred, test_actual)]
        scatter_pts = "[" + ", ".join(pts) + "]"
    else:
        scatter_pts = "[]"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Return Predictor — {index_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {BG}; color: {TEXT}; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.5; }}
  h1 {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -0.5px; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: {TEXT}; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 20px; }}
  .header {{ margin-bottom: 28px; }}
  .subtitle {{ color: {MUTED}; font-size: 0.9rem; margin-top: 4px; }}
  .panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
  .regime-banner {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: center; }}
  .regime-stat {{ display: flex; flex-direction: column; gap: 2px; }}
  .regime-stat .val {{ font-size: 1.4rem; font-weight: 700; }}
  .regime-stat .lbl {{ font-size: 0.75rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.05em; }}
  .divider {{ width: 1px; background: {BORDER}; height: 40px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ padding: 10px 12px; text-align: left; color: {MUTED}; font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid {BORDER}; cursor: pointer; user-select: none; white-space: nowrap; }}
  th:hover {{ color: {TEXT}; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid {BORDER}22; }}
  tr:hover td {{ background: {BORDER}33; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media(max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
  .chart-wrap {{ position: relative; height: 340px; }}
  .ic-badge {{ display: inline-block; background: {ACCENT}22; color: {ACCENT}; border: 1px solid {ACCENT}44; border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; font-weight: 600; margin-left: 10px; vertical-align: middle; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>Return Predictor &mdash; {index_name}
      <span class="ic-badge">IC: {ic_str}</span>
    </h1>
    <p class="subtitle">{n_tickers} tickers &bull; {today_str} &bull; Horizon: {horizon} trading days &bull; XGBoost cross-sectional factor model</p>
  </div>

  <!-- Regime Banner -->
  <div class="panel">
    <div class="regime-banner">
      <div class="regime-stat">
        <span class="val" style="color:{ACCENT}">{ic_str}</span>
        <span class="lbl">Walk-Forward IC</span>
      </div>
      <div class="divider"></div>
      <div class="regime-stat">
        <span class="val" style="color:{GREEN}">{int(n_tickers * 0.25)}</span>
        <span class="lbl">Top Predicted Tickers</span>
      </div>
      <div class="divider"></div>
      <div class="regime-stat">
        <span class="val" style="color:{TEXT}">{horizon}d</span>
        <span class="lbl">Forecast Horizon</span>
      </div>
      <div class="divider"></div>
      <div class="regime-stat">
        <span class="val" style="color:{TEXT}">{n_tickers}</span>
        <span class="lbl">Universe Size</span>
      </div>
      <div class="divider"></div>
      <div class="regime-stat">
        <span class="val" style="color:{YELLOW}">7</span>
        <span class="lbl">Training Snapshots</span>
      </div>
    </div>
  </div>

  <!-- Ranked Table -->
  <div class="panel">
    <h2>Ranked Universe</h2>
    <div style="overflow-x:auto">
      <table id="rankTable">
        <thead>
          <tr>
            <th onclick="sortTable(0)" style="text-align:center">Rank</th>
            <th onclick="sortTable(1)">Ticker</th>
            <th onclick="sortTable(2)">Sector</th>
            <th onclick="sortTable(3)" style="text-align:right">Score (0–100)</th>
            <th onclick="sortTable(4)" style="text-align:center">Expected Return Quartile</th>
            <th onclick="sortTable(5)" style="text-align:right">Mom 63d</th>
            <th onclick="sortTable(6)" style="text-align:right">P/E</th>
            <th onclick="sortTable(7)" style="text-align:right">ROE</th>
            <th onclick="sortTable(8)" style="text-align:right">Gross Margin</th>
          </tr>
        </thead>
        <tbody>
{table_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-grid">
    <div class="panel">
      <h2>Top-10 Feature Importances</h2>
      <div class="chart-wrap">
        <canvas id="fiChart"></canvas>
      </div>
    </div>
    <div class="panel">
      <h2>Walk-Forward: Predicted Rank vs Actual Return</h2>
      <div class="chart-wrap">
        <canvas id="scatterChart"></canvas>
      </div>
    </div>
  </div>

</div>

<script>
// ── Table sort ──────────────────────────────────────────────────────────────
let _sortDir = {{}};
function sortTable(col) {{
  const tbl = document.getElementById("rankTable");
  const tbody = tbl.tBodies[0];
  const rows = Array.from(tbody.rows);
  const asc = !_sortDir[col];
  _sortDir = {{}};
  _sortDir[col] = asc;
  rows.sort((a, b) => {{
    let va = a.cells[col].innerText.replace(/[%—]/g, "").trim();
    let vb = b.cells[col].innerText.replace(/[%—]/g, "").trim();
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

// ── Chart defaults ──────────────────────────────────────────────────────────
Chart.defaults.color = "{MUTED}";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

// Feature Importance Chart
new Chart(document.getElementById("fiChart"), {{
  type: "bar",
  data: {{
    labels: {fi_labels_js},
    datasets: [{{
      label: "Importance",
      data: {fi_values_js},
      backgroundColor: "{ACCENT}cc",
      borderColor: "{ACCENT}",
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => " " + ctx.parsed.x.toFixed(4) }} }} }},
    scales: {{
      x: {{ grid: {{ color: "{BORDER}" }}, ticks: {{ color: "{MUTED}" }} }},
      y: {{ grid: {{ display: false }}, ticks: {{ color: "{TEXT}", font: {{ size: 11 }} }} }}
    }}
  }}
}});

// Backtest Scatter Chart
new Chart(document.getElementById("scatterChart"), {{
  type: "scatter",
  data: {{
    datasets: [{{
      label: "Test samples",
      data: {scatter_pts},
      backgroundColor: "{ACCENT}99",
      pointRadius: 4,
      pointHoverRadius: 6,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{
        label: ctx => `Pred rank: ${{ctx.parsed.x.toFixed(2)}}  Actual ret: ${{ctx.parsed.y.toFixed(1)}}%`
      }} }}
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: "Predicted Rank (0–1)", color: "{MUTED}" }},
        grid: {{ color: "{BORDER}" }},
        ticks: {{ color: "{MUTED}" }}
      }},
      y: {{
        title: {{ display: true, text: "Actual Return (%)", color: "{MUTED}" }},
        grid: {{ color: "{BORDER}" }},
        ticks: {{ color: "{MUTED}" }}
      }}
    }}
  }}
}});
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost cross-sectional return predictor")
    parser.add_argument("--index",    default="SPX", choices=["SPX", "NDX", "DOW", "RUT"])
    parser.add_argument("--tickers",  default="", help="Comma-separated ticker override")
    parser.add_argument("--csv-file", default="", dest="csv_file",
                        help="Path to CSV with 'ticker' and optional 'weight' columns")
    parser.add_argument("--horizon",  type=int, default=63, choices=[30, 63, 126])
    parser.add_argument("--top",      type=int, default=150)
    args = parser.parse_args()

    today_label = datetime.today().strftime("%Y%m%d")
    universe_label = "CUSTOM" if (args.csv_file or args.tickers) else args.index
    out_path = os.path.join(OUT_DIR, f"return_predictor_{universe_label}_{today_label}.html")

    print(f"\n=== Return Predictor === {universe_label} ===", flush=True)

    # ── Step 1: Get tickers ──────────────────────────────────────────────────
    if args.csv_file:
        print(f"  Loading tickers from CSV: {os.path.basename(args.csv_file)}...", flush=True)
        constituents = _read_csv_file(args.csv_file)
        if not constituents:
            print("  ERROR: CSV file is empty or unreadable. Exiting.", flush=True)
            sys.exit(1)
        print(f"  {len(constituents)} tickers loaded from CSV", flush=True)
    elif args.tickers:
        constituents = [{"ticker": t.strip().upper(), "sector": "Unknown",
                         "name": t.strip().upper(), "weight": None}
                        for t in args.tickers.split(",") if t.strip()]
    else:
        print(f"  Fetching {args.index} constituents...", flush=True)
        fetchers = {"SPX": _get_spx, "NDX": _get_ndx, "DOW": _get_dow, "RUT": _get_rut}
        constituents = fetchers[args.index]()

    if not constituents:
        print("  ERROR: No constituents found. Exiting.", flush=True)
        sys.exit(1)

    tickers_all = [c["ticker"] for c in constituents]

    # ── Step 2: Download price history + info ────────────────────────────────
    print(f"  Downloading price history ({len(tickers_all)} tickers)...", flush=True)

    all_data = fetch_all_data(tickers_all)
    n_fetched = len(all_data)
    n_failed  = len(tickers_all) - n_fetched
    if n_failed:
        print(f"  Skipped {n_failed} tickers (insufficient history).", flush=True)

    # Trim to top-N by market cap
    if args.top and len(all_data) > args.top:
        sorted_by_cap = sorted(all_data.values(), key=lambda d: d["mcap"], reverse=True)
        all_data = {d["ticker"]: d for d in sorted_by_cap[: args.top]}

    print(f"  Building feature matrix...", flush=True)

    # ── Step 3: Build training rows ──────────────────────────────────────────
    all_rows: list[dict] = []
    N_SNAPSHOTS = 7

    for tkr, tdata in all_data.items():
        rows = build_training_rows(tdata, args.horizon, n_snapshots=N_SNAPSHOTS)
        all_rows.extend(rows)

    if not all_rows:
        print("  ERROR: No training rows built. Exiting.", flush=True)
        sys.exit(1)

    train_df = pd.DataFrame(all_rows)

    # Cross-sectional label rank within each snapshot
    train_df["label_rank"] = np.nan
    for snap_i, grp in train_df.groupby("snapshot"):
        if len(grp) < 5:
            continue
        ranks = rankdata(grp["label"].values, method="average")
        norm  = (ranks - 1) / max(len(ranks) - 1, 1)
        train_df.loc[grp.index, "label_rank"] = norm

    # Cross-sectional feature ranks within each snapshot
    for col in FEATURE_COLS:
        train_df[f"_rank_{col}"] = np.nan
        for snap_i, grp in train_df.groupby("snapshot"):
            mask  = grp[col].notna()
            if mask.sum() < 5:
                continue
            sub   = grp.loc[mask, col].values.astype(float)
            ranks = rankdata(sub, method="average")
            norm  = (ranks - 1) / max(len(sub) - 1, 1)
            train_df.loc[grp.index[mask], f"_rank_{col}"] = norm

    rank_features = [f"_rank_{c}" for c in FEATURE_COLS]

    # Drop rows with >40% missing ranked features
    max_missing = int(0.4 * len(rank_features))
    train_df = train_df.dropna(subset=["label_rank"])
    train_df = train_df[train_df[rank_features].isna().sum(axis=1) <= max_missing].copy()

    # Fill remaining NaNs with 0.5 (neutral rank)
    train_df[rank_features] = train_df[rank_features].fillna(0.5)

    n_samples  = len(train_df)
    n_features = len(rank_features)

    print(f"  Training XGBoost ({n_samples} samples, {n_features} features)...", flush=True)

    model_params = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, n_jobs=-1,
        verbosity=0,
    )

    # Walk-forward validation
    ic, test_pred, test_actual = walk_forward_ic(train_df, rank_features, model_params)
    print(f"  Walk-forward IC: {ic:.3f}" if not np.isnan(ic) else "  Walk-forward IC: N/A", flush=True)

    # Retrain on all data
    final_model = XGBRegressor(**model_params)
    X_all = train_df[rank_features].values
    y_all = train_df["label_rank"].values
    final_model.fit(X_all, y_all)

    # Feature importances (mapped back to original feature names)
    raw_imp = final_model.feature_importances_
    feat_imp = {FEATURE_COLS[i]: float(raw_imp[i]) for i in range(len(FEATURE_COLS))}

    # ── Step 4: Score current universe ──────────────────────────────────────
    current_rows: list[dict] = []
    for tkr, tdata in all_data.items():
        hist = tdata["hist"]
        info = tdata["info"]
        if len(hist) < 30:
            continue
        pf = _price_features(hist, len(hist) - 1)
        ff = _fundamental_features(info)
        row = {"ticker": tkr, "sector": tdata["sector"]}
        row.update(pf)
        row.update(ff)
        current_rows.append(row)

    if not current_rows:
        print("  ERROR: No current feature rows. Exiting.", flush=True)
        sys.exit(1)

    curr_df = pd.DataFrame(current_rows)

    # Cross-sectional rank features for current snapshot
    for col in FEATURE_COLS:
        curr_df[f"_rank_{col}"] = np.nan
        mask = curr_df[col].notna()
        if mask.sum() < 5:
            continue
        sub   = curr_df.loc[mask, col].values.astype(float)
        ranks = rankdata(sub, method="average")
        norm  = (ranks - 1) / max(len(sub) - 1, 1)
        curr_df.loc[mask, f"_rank_{col}"] = norm

    curr_df[rank_features] = curr_df[rank_features].fillna(0.5)

    X_curr = curr_df[rank_features].values
    pred_scores = final_model.predict(X_curr)

    curr_df["pred_rank"] = pred_scores
    curr_df["score"]     = (pred_scores * 100).round(1)

    # Sort by predicted rank descending
    curr_df = curr_df.sort_values("pred_rank", ascending=False).reset_index(drop=True)

    # ── Step 5: Generate HTML ────────────────────────────────────────────────
    print(f"  Generating report...", flush=True)

    generate_html(
        ranked=curr_df,
        index_name=args.index,
        horizon=args.horizon,
        ic=ic if not np.isnan(ic) else float("nan"),
        feature_importances=feat_imp,
        test_pred=test_pred,
        test_actual=test_actual,
        out_path=out_path,
    )

    abs_path = os.path.abspath(out_path)
    print(f"  ✓  Report saved → {abs_path}", flush=True)


if __name__ == "__main__":
    main()
