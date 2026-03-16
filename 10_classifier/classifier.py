#!/usr/bin/env python3
"""
classifier.py — Behavioural Clustering Tool
============================================
Takes the constituents of a major index, downloads their price history,
engineers return/risk/momentum features, then clusters them with K-Means,
Hierarchical (Ward) or DBSCAN.  The result is an interactive Plotly HTML
report with:

  • 2-D scatter (t-SNE / UMAP / PCA) — one dot per stock, coloured by cluster
  • Cluster summary table — avg 1-M/3-M/1-Y return, vol, beta per cluster
  • Correlation heatmap reordered by cluster
  • PCA explained-variance bar chart

Usage
-----
    python classifier.py --index SPX --method kmeans --viz tsne
    python classifier.py --index SPX --method kmeans --n_clusters 8 --viz tsne
    python classifier.py --index RUT --method hierarchical --viz umap
    python classifier.py --index TSX --method dbscan --viz pca

    Omit --n_clusters (or pass 0) to auto-select k:
      K-Means      → silhouette scan (k=2..15, balance filter)
      Hierarchical → dendrogram gap method (Ward merge-height gaps)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timedelta

import numpy  as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Directories ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "classifierData")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette / theme ────────────────────────────────────────────────────────────
ACCENT   = "#7c6af7"
BG       = "#0d0d0f"
PANEL    = "#17171c"
TEXT     = "#e8e8f0"
SUBTEXT  = "#8888a8"
BORDER   = "#2a2a3a"

# 20 distinct cluster colours
CLUSTER_COLOURS = [
    "#7c6af7", "#f7706a", "#6af7a4", "#f7c86a", "#6ab8f7",
    "#f76ab0", "#a4f76a", "#f76a6a", "#6af7f7", "#c86af7",
    "#f7a46a", "#6a6af7", "#6af788", "#f76af7", "#f7e46a",
    "#6af7c8", "#886af7", "#f76a88", "#a4c86a", "#6ab0f7",
]

# ── Index constituent fetchers ─────────────────────────────────────────────────

def _wiki_table(url: str, ticker_col: str, name_col: str | None = None,
                sector_col: str | None = None) -> list[dict]:
    """Parse first HTML table from a Wikipedia URL."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ClassifierBot/1.0 (educational)"}
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  ✗  Wikipedia fetch failed: {e}")
        return []

    # Find <table class="wikitable">
    import re
    table_m = re.search(r'<table[^>]*wikitable[^>]*>(.*?)</table>',
                        html, re.DOTALL | re.IGNORECASE)
    if not table_m:
        return []

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_m.group(1), re.DOTALL | re.IGNORECASE)
    if not rows:
        return []

    # Parse header
    header_cells = re.findall(r'<th[^>]*>(.*?)</th>', rows[0], re.DOTALL | re.IGNORECASE)
    headers = [re.sub(r'<[^>]+>', '', c).strip().replace('\n', ' ') for c in header_cells]

    def col_idx(name: str | None) -> int | None:
        if name is None:
            return None
        for i, h in enumerate(headers):
            if name.lower() in h.lower():
                return i
        return None

    ti = col_idx(ticker_col)
    ni = col_idx(name_col)
    si = col_idx(sector_col)

    if ti is None:
        return []

    results = []
    for row in rows[1:]:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if len(cells) <= ti:
            continue
        ticker = cells[ti].split('\n')[0].strip()
        if not ticker or not ticker.replace('.', '').replace('-', '').isalnum():
            continue
        results.append({
            "ticker": ticker,
            "name":   cells[ni].split('\n')[0].strip() if (ni is not None and len(cells) > ni) else ticker,
            "sector": cells[si].split('\n')[0].strip() if (si is not None and len(cells) > si) else "Unknown",
        })
    return results


def _get_spx() -> list[dict]:
    print("  Fetching S&P 500 constituents from Wikipedia…")
    return _wiki_table(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        ticker_col="Symbol", name_col="Security", sector_col="GICS Sector"
    )


def _get_ndx() -> list[dict]:
    """Nasdaq-100 via Wikipedia (more reliable than hardcoding)."""
    print("  Fetching Nasdaq-100 constituents from Wikipedia…")
    rows = _wiki_table(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        ticker_col="Ticker", name_col="Company", sector_col="Sector"
    )
    if not rows:
        rows = _wiki_table(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            ticker_col="Symbol", name_col="Company", sector_col="Sector"
        )
    return rows


def _get_dow() -> list[dict]:
    """Dow Jones 30 via Wikipedia."""
    print("  Fetching Dow Jones 30 constituents from Wikipedia…")
    return _wiki_table(
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        ticker_col="Symbol", name_col="Company", sector_col="Industry"
    )


def _get_tradingview(index_id: str, label: str) -> list[dict]:
    """Fetch RUT / TSX constituents via tradingview-screener."""
    print(f"  Fetching {label} constituents via TradingView screener…")
    try:
        from tradingview_screener import Query, Column
        if index_id == "RUT":
            q = (Query()
                 .set_markets("america")
                 .where(Column("is_primary") == True)
                 .where(Column("index_membership").isin(["Russell 2000"]))
                 .select("name", "full_name", "sector")
                 .limit(2100))
        else:  # TSX
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
            ticker = str(row.get("name", "")).replace("AMEX:", "").replace("NYSE:", "").replace("NASDAQ:", "")
            results.append({
                "ticker": ticker,
                "name":   str(row.get("full_name", ticker)),
                "sector": str(row.get("sector", "Unknown")),
            })
        return results
    except Exception as e:
        print(f"  ✗  TradingView screener failed: {e}")
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
        return _get_tradingview("RUT", "Russell 2000")
    if index == "TSX":
        return _get_tradingview("TSX", "S&P/TSX")
    return []


# ── Price download ─────────────────────────────────────────────────────────────

def _clear_yf_db_cache() -> None:
    """Delete yfinance SQLite cache files.

    yfinance keeps per-ticker timezone + history data in SQLite DBs under
    ~/Library/Caches/py-yfinance (macOS) or ~/.cache/py-yfinance (Linux).
    When many threads open the same DB simultaneously the OS-level file-
    descriptor / lock limit is exceeded, producing:
        OperationalError('unable to open database file')
    Wiping the DB before each batch forces a clean connection pool for every
    batch and eliminates the compounding-failure pattern observed with large
    ticker lists.
    """
    import glob
    for cache_dir in [
        os.path.expanduser("~/Library/Caches/py-yfinance"),
        os.path.expanduser("~/.cache/py-yfinance"),
    ]:
        for f in glob.glob(os.path.join(cache_dir, "*.db*")):
            try:
                os.remove(f)
            except OSError:
                pass


def fetch_prices(tickers: list[str], days: int = 380,
                 chunk_size: int = 50) -> pd.DataFrame:
    """Batch-download adjusted close prices using yf.download().

    Strategy
    --------
    A single one-time cache wipe is performed at the start to clear any
    stale lock files left by a previous run.  The DB is then left alone for
    the entire download so yfinance can initialise its schema tables (notably
    ``_tz_kv``) once and reuse them across all batches.

    Phase 1 — chunked batch download (chunk_size tickers at a time).
        • threads=False forces sequential per-ticker HTTP inside each batch,
          eliminating concurrent DNS resolution failures
          ("getaddrinfo() thread failed to start") and concurrent SQLite
          writes that exhaust OS file-descriptor limits.
        • chunk_size=50 balances progress visibility vs. batch-level overhead;
          since downloads are sequential within a batch, size is unconstrained
          by concurrency.

    Phase 2 — individual retry for any ticker that still did not land
        (TypeError / OperationalError survivors from Phase 1).
        Uses yf.Ticker().history() as a single-threaded fallback.
    """
    import yfinance as yf

    # One-time cleanup of stale cache files from previous runs only.
    # NEVER call _clear_yf_db_cache() inside either download loop — doing so
    # deletes the DB between batches, causing yfinance to create a fresh empty
    # file without initialising the _tz_kv table, which breaks every
    # subsequent download with OperationalError('no such table: _tz_kv').
    _clear_yf_db_cache()

    end   = datetime.today()
    start = end - timedelta(days=days)
    start_s = start.strftime("%Y-%m-%d")
    end_s   = end.strftime("%Y-%m-%d")

    chunks = [tickers[i:i + chunk_size]
              for i in range(0, len(tickers), chunk_size)]

    print(f"  Downloading {len(tickers)} tickers "
          f"({days} days, {len(chunks)} batches of ≤{chunk_size})…")
    t0 = time.time()

    frames: list[pd.DataFrame] = []
    downloaded: set[str] = set()

    # ── Phase 1: chunked batch downloads ──────────────────────────────────
    for i, chunk in enumerate(chunks, 1):
        try:
            raw = yf.download(
                chunk,
                start=start_s,
                end=end_s,
                auto_adjust=True,
                progress=False,
                threads=False,   # sequential: no concurrent DNS/SQLite pressure
            )
            if raw.empty:
                continue
            # Extract 'Close' level (multi-level when >1 ticker in batch)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            elif "Close" in raw.columns:
                close = raw[["Close"]].rename(columns={"Close": chunk[0]})
            else:
                close = raw
            frames.append(close)
            downloaded.update(close.columns.tolist())
        except Exception as exc:
            print(f"  ⚠  Batch {i}/{len(chunks)} failed: {exc}")

    # ── Phase 2: individual retry for still-missing tickers ───────────────
    missing = [t for t in tickers if t not in downloaded]
    if missing:
        print(f"  Retrying {len(missing)} tickers individually…")
        retry_frames: list[pd.DataFrame] = []
        for ticker in missing:
            try:
                hist = yf.Ticker(ticker).history(
                    start=start_s, end=end_s, auto_adjust=True
                )
                if not hist.empty and "Close" in hist.columns:
                    s = hist["Close"].rename(ticker)
                    s.index = s.index.tz_localize(None) if s.index.tzinfo else s.index
                    retry_frames.append(s.to_frame())
            except Exception:
                pass
        if retry_frames:
            frames.extend(retry_frames)

    elapsed = time.time() - t0
    print(f"  Downloaded in {elapsed:.1f}s")

    if not frames:
        return pd.DataFrame()

    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]

    # Drop columns with too many NaN (< 60 % valid rows)
    min_rows = int(len(close) * 0.60)
    close = close.dropna(axis=1, thresh=min_rows)

    print(f"  {close.shape[1]} tickers survived coverage filter")
    return close


# ── Feature engineering ────────────────────────────────────────────────────────

MOMENTUM_WINDOWS = [5, 10, 21, 42, 63, 126, 189, 252]   # days


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix with one row per ticker:
      - Return momentum for 8 look-back windows
      - Realised volatility (21d, 63d)
      - Volatility ratio (vol_21d / vol_126d) — vol regime
      - Beta to SPY (252d rolling, last value)
      - Alpha vs SPY (annualised excess return, 252d)
      - Max drawdown over 252d
      - Return skewness over 126d
      - Return kurtosis over 126d — tail risk
      - Lag-1 autocorrelation of 21d returns — mean-reversion vs momentum
      - Trend R² over 63d — trend smoothness
      - Price vs 52-week high — cycle positioning
    """
    import yfinance as yf

    print("  Engineering features…")
    returns = prices.pct_change().dropna(how="all")

    # ── SPY beta reference ──────────────────────────────────────────────────
    spy_raw = yf.download("SPY", period="2y", auto_adjust=True, progress=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_close = spy_raw["Close"]["SPY"]
    else:
        spy_close = spy_raw["Close"]
    spy_ret = spy_close.pct_change().dropna()
    spy_ret = spy_ret.reindex(returns.index).dropna()

    features: dict[str, pd.Series] = {}

    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) < 50:
            continue

        feat: dict[str, float] = {}

        # Momentum
        for w in MOMENTUM_WINDOWS:
            if len(r) >= w:
                feat[f"mom_{w}d"] = float(r.tail(w).add(1).prod() - 1)
            else:
                feat[f"mom_{w}d"] = np.nan

        # Realised vol
        feat["vol_21d"] = float(r.tail(21).std() * np.sqrt(252)) if len(r) >= 21 else np.nan
        feat["vol_63d"] = float(r.tail(63).std() * np.sqrt(252)) if len(r) >= 63 else np.nan

        # Downside vol — semi-deviation using only negative-return days (21d)
        # More relevant for drawdown risk than symmetric volatility
        r21_neg = r.tail(21)
        r21_neg = r21_neg[r21_neg < 0]
        feat["downside_vol_21d"] = float(r21_neg.std() * np.sqrt(252)) if len(r21_neg) >= 3 else np.nan

        # Sortino ratio (252d) — annualised return / annualised downside vol
        # Penalises downside volatility only; higher = better risk-adjusted return
        r252 = r.tail(252)
        neg252 = r252[r252 < 0]
        if len(neg252) >= 10 and len(r252) >= 30:
            ann_ret = float(r252.add(1).prod() ** (252 / len(r252)) - 1)
            dsv     = float(neg252.std() * np.sqrt(252))
            feat["sortino_ratio"] = ann_ret / dsv if dsv > 0 else np.nan
        else:
            feat["sortino_ratio"] = np.nan

        # Beta to SPY (rolling 252d on available aligned data)
        aligned = pd.concat([r, spy_ret], axis=1).dropna()
        aligned.columns = ["stock", "spy"]
        if len(aligned) >= 60:
            cov  = aligned["stock"].cov(aligned["spy"])
            var  = aligned["spy"].var()
            feat["beta"] = float(cov / var) if var > 0 else np.nan
        else:
            feat["beta"] = np.nan

        # Max drawdown (252 days)
        tail = r.tail(252)
        cum  = (1 + tail).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak
        feat["max_dd_252d"] = float(dd.min()) if len(tail) > 0 else np.nan

        # Skewness (126 days)
        feat["skew_126d"] = float(r.tail(126).skew()) if len(r) >= 30 else np.nan

        # Kurtosis (126 days) — tail-risk / fat-tails
        feat["kurtosis_126d"] = float(r.tail(126).kurt()) if len(r) >= 30 else np.nan

        # Volatility ratio (vol_21d / vol_126d) — vol-regime: >1 = expanding, <1 = contracting
        vol_126 = float(r.tail(126).std() * np.sqrt(252)) if len(r) >= 126 else np.nan
        if vol_126 and vol_126 > 0 and not np.isnan(feat.get("vol_21d", np.nan)):
            feat["vol_ratio"] = feat["vol_21d"] / vol_126
        else:
            feat["vol_ratio"] = np.nan

        # Lag-1 autocorrelation of 21d returns — positive = momentum, negative = mean-reversion
        r21 = r.tail(42)
        if len(r21) >= 10:
            try:
                feat["autocorr_21d"] = float(r21.autocorr(lag=1))
            except Exception:
                feat["autocorr_21d"] = np.nan
        else:
            feat["autocorr_21d"] = np.nan

        # Trend R² over 63d — smoothness of log-price trend (1 = perfect trend, 0 = choppy)
        if len(r) >= 63:
            from scipy import stats as _stats
            log_p = np.log(prices[ticker].dropna().tail(63).values)
            if len(log_p) >= 10:
                x = np.arange(len(log_p))
                _, _, r_val, _, _ = _stats.linregress(x, log_p)
                feat["trend_r2_63d"] = float(r_val ** 2)
            else:
                feat["trend_r2_63d"] = np.nan
        else:
            feat["trend_r2_63d"] = np.nan

        # Alpha vs SPY (annualised) — excess return beyond market beta
        aligned = pd.concat([r, spy_ret], axis=1).dropna()
        aligned.columns = ["stock", "spy"]
        if len(aligned) >= 126:
            cov = aligned["stock"].cov(aligned["spy"])
            var = aligned["spy"].var()
            b   = float(cov / var) if var > 0 else 0.0
            alpha_daily = float(aligned["stock"].mean()) - b * float(aligned["spy"].mean())
            feat["alpha_252d"] = alpha_daily * 252
        else:
            feat["alpha_252d"] = np.nan

        # Price vs 52-week high — cycle positioning (0 = at high, -0.5 = 50% below, etc.)
        p_series = prices[ticker].dropna()
        if len(p_series) >= 252:
            high_52w = float(p_series.tail(252).max())
            last_p   = float(p_series.iloc[-1])
            feat["price_vs_52w_high"] = (last_p / high_52w) - 1.0 if high_52w > 0 else np.nan
        else:
            feat["price_vs_52w_high"] = np.nan

        features[ticker] = pd.Series(feat)

    feat_df = pd.DataFrame(features).T
    feat_df.index.name = "ticker"

    # Drop rows with any NaN in core features; fill others with median
    n_before = len(feat_df)
    feat_df = feat_df.dropna(subset=["mom_63d", "vol_21d", "beta", "max_dd_252d"])
    print(f"  Feature matrix: {len(feat_df)} / {n_before} tickers (after NaN drop)")

    # Fill remaining NaN with column median
    for col in feat_df.columns:
        feat_df[col] = feat_df[col].fillna(feat_df[col].median())

    return feat_df


# ── Fundamental feature fetch ──────────────────────────────────────────────────

# Set of feature column names that come from yf.Ticker().info (not price history).
# Used in main() to decide whether to call fetch_fundamentals().
FUNDAMENTAL_FEATURES: frozenset[str] = frozenset({
    # Value
    "pe_ratio", "pb_ratio", "ps_ratio", "div_yield", "fcf_yield",
    "ev_ebitda", "ev_sales",
    # Growth
    "eps_growth_1y", "rev_growth_1y", "peg_ratio", "gross_margin",
    # Quality
    "roe", "roa", "op_margin", "net_margin", "debt_equity", "current_ratio",
})

# Outlier clip percentiles applied to fundamental columns before clustering.
# Fundamentals (P/E, PEG, etc.) can have extreme outliers that would dominate
# StandardScaler even after z-scoring.  Winsorising at [1 %, 99 %] per column
# preserves the distribution shape while removing pathological extremes.
_FUND_CLIP_PCT = (1.0, 99.0)


def fetch_fundamentals(tickers: list[str], max_workers: int = 20) -> pd.DataFrame:
    """Fetch fundamental valuation/growth metrics via yf.Ticker().info.

    Runs concurrently with ThreadPoolExecutor.  Returns a DataFrame indexed by
    ticker with columns:
      pe_ratio      — trailing P/E
      pb_ratio      — price / book
      div_yield     — dividend yield (0 for non-payers; NOT NaN)
      fcf_yield     — free cash flow / market cap
      ev_ebitda     — enterprise value / EBITDA
      eps_growth_1y — most-recent-quarter YoY earnings growth
      rev_growth_1y — most-recent-quarter YoY revenue growth
      peg_ratio     — trailing PEG ratio

    Missing values are filled with the column median after fetching.
    Columns are winsorised at [1 %, 99 %] to remove extreme outliers before
    they reach StandardScaler.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import yfinance as yf

    # yfinance .info field → our column name
    _INFO_MAP = {
        # Value
        "trailingPE":                      "pe_ratio",
        "priceToBook":                     "pb_ratio",
        "priceToSalesTrailing12Months":    "ps_ratio",
        "dividendYield":                   "div_yield",
        "enterpriseToEbitda":              "ev_ebitda",
        "enterpriseToRevenue":             "ev_sales",
        # Growth
        "earningsGrowth":                  "eps_growth_1y",
        "revenueGrowth":                   "rev_growth_1y",
        "trailingPegRatio":                "peg_ratio",
        "grossMargins":                    "gross_margin",
        # Quality
        "returnOnEquity":                  "roe",
        "returnOnAssets":                  "roa",
        "operatingMargins":                "op_margin",
        "profitMargins":                   "net_margin",
        "debtToEquity":                    "debt_equity",
        "currentRatio":                    "current_ratio",
    }

    def _fetch_one(ticker: str) -> tuple[str, dict]:
        try:
            full = yf.Ticker(ticker).info
            row: dict[str, float] = {}

            for yf_key, col in _INFO_MAP.items():
                v = full.get(yf_key)
                row[col] = float(v) if (v is not None and v == v) else np.nan

            # Non-payers: dividendYield is None → treat as 0, not NaN
            if np.isnan(row.get("div_yield", np.nan)):
                row["div_yield"] = 0.0

            # Negative PEG is meaningless for valuation → set to NaN
            if row.get("peg_ratio", np.nan) is not np.nan and row["peg_ratio"] < 0:
                row["peg_ratio"] = np.nan

            # FCF yield = freeCashflow / marketCap
            fcf  = full.get("freeCashflow")
            mcap = full.get("marketCap")
            if fcf is not None and mcap and float(mcap) > 0:
                row["fcf_yield"] = float(fcf) / float(mcap)
            else:
                row["fcf_yield"] = np.nan

            return ticker, row
        except Exception:
            return ticker, {}

    print(f"  Fetching fundamentals for {len(tickers)} tickers "
          f"(max_workers={max_workers})…", flush=True)
    t0 = time.time()

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for fut in as_completed(futs):
            ticker, row = fut.result()
            if row:
                results[ticker] = row
            done += 1
            if done % 100 == 0:
                print(f"    {done} / {len(tickers)}…", flush=True)

    elapsed = time.time() - t0
    print(f"  Fundamentals fetched in {elapsed:.1f}s  ({len(results)} / {len(tickers)} tickers)")

    if not results:
        return pd.DataFrame()

    fund_df = pd.DataFrame(results).T
    fund_df.index.name = "ticker"

    # Ensure all expected columns exist
    for col in list(_INFO_MAP.values()) + ["fcf_yield"]:
        if col not in fund_df.columns:
            fund_df[col] = np.nan

    # Fill missing with column median (except div_yield which stays 0 for non-payers)
    for col in fund_df.columns:
        med = fund_df[col].median()
        if col == "div_yield":
            fund_df[col] = fund_df[col].fillna(0.0)
        else:
            fund_df[col] = fund_df[col].fillna(med if not np.isnan(med) else 0.0)

    # Winsorise at [1 %, 99 %] per column to remove pathological outliers
    lo_pct, hi_pct = _FUND_CLIP_PCT
    for col in fund_df.columns:
        lo = float(fund_df[col].quantile(lo_pct / 100))
        hi = float(fund_df[col].quantile(hi_pct / 100))
        fund_df[col] = fund_df[col].clip(lo, hi)

    return fund_df


# ── Clustering pipeline ────────────────────────────────────────────────────────

K_SCAN_MIN, K_SCAN_MAX = 4, 15   # k search range for auto-selection
# k=2/3 are skipped: for large indexes (~500 stocks) the silhouette criterion
# always peaks at k=2 because the dominant variance axis (beta/volatility)
# produces one clean binary split while suppressing all finer structure.
# Starting at k=4 forces discovery of the meaningful sub-groups present in
# sector/style data.

# Minimum cluster size for k-means composite scan.
# Prevents picking a degenerate split where 1-2 extreme outliers form their
# own cluster — which produces a spuriously high silhouette score.
K_MIN_CLUSTER_FRAC = 0.02        # 2 % of n  (≥10 stocks for SPX)


def _auto_k_silhouette(Xp: np.ndarray) -> tuple[int, dict[int, float]]:
    """K-Means k-selection via composite silhouette + Calinski-Harabasz scan.

    Pure silhouette score tends to peak at k=2 for financial continuum data
    (the single largest variance axis — beta/vol — dominates).  Calinski-
    Harabasz (CH) measures between-cluster / within-cluster variance ratio
    and selects higher, more granular k values.  The two metrics are each
    normalised to [0, 1] across the scanned range and averaged into a
    composite score.

    Returns (best_k, {k: composite_score}).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    n_samples = Xp.shape[0]
    k_max     = min(K_SCAN_MAX, n_samples - 1)
    min_size  = max(2, int(K_MIN_CLUSTER_FRAC * n_samples))
    sil_raw:  dict[int, float] = {}
    ch_raw:   dict[int, float] = {}
    skipped: list[int] = []

    print(f"  Auto-selecting k ({K_SCAN_MIN}–{k_max}) via silhouette + "
          f"Calinski-Harabasz composite (min cluster size = {min_size})…",
          flush=True)

    for k in range(K_SCAN_MIN, k_max + 1):
        lbl    = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xp)
        counts = np.bincount(lbl.astype(int))
        if counts.min() < min_size:
            skipped.append(k)
            continue
        sil_raw[k] = float(silhouette_score(Xp, lbl))
        ch_raw[k]  = float(calinski_harabasz_score(Xp, lbl))

    if skipped:
        print(f"  Skipped degenerate k values (cluster < {min_size} stocks): {skipped}")

    if not sil_raw:
        print("  ⚠  All k values were degenerate; falling back to unconstrained scan.")
        for k in range(K_SCAN_MIN, k_max + 1):
            lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xp)
            sil_raw[k] = float(silhouette_score(Xp, lbl))
            ch_raw[k]  = float(calinski_harabasz_score(Xp, lbl))

    # Normalise each metric to [0, 1] then average into a composite score
    def _norm(d: dict[int, float]) -> dict[int, float]:
        lo, hi = min(d.values()), max(d.values())
        if hi == lo:
            return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    sil_n = _norm(sil_raw)
    ch_n  = _norm(ch_raw)
    scores: dict[int, float] = {
        k: round((sil_n[k] + ch_n[k]) / 2, 4) for k in sil_raw
    }

    best_k = max(scores, key=scores.get)
    print(f"  → Best k = {best_k}  "
          f"(composite = {scores[best_k]:.3f}, "
          f"sil = {sil_raw[best_k]:.3f}, "
          f"CH = {ch_raw[best_k]:.0f})")
    return best_k, scores


def _auto_k_dendrogram_gap(Xp: np.ndarray) -> tuple[int, dict[int, float]]:
    """Hierarchical (Ward) k-selection via dendrogram gap method.

    Fits the full Ward dendrogram once, then looks at the gaps between
    consecutive merge heights in the top K_SCAN_MAX range.  The largest
    gap indicates the most expensive merge — we stop just before it,
    yielding a k in [K_SCAN_MIN, K_SCAN_MAX].

    A balance filter (same K_MIN_CLUSTER_FRAC threshold used by the K-Means
    silhouette scan) skips any k where fcluster() would produce a degenerate
    split — e.g. one cluster of 3 stocks vs 497.  Without this filter the
    gap method tends to select k=2 whenever a handful of extreme outliers
    form a tiny branch that merges last, creating a spuriously large gap.

    Returns (best_k, {k: gap_size}) — the scores dict holds the
    merge-height gap for each *accepted* k (not silhouette scores).
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    n_samples = Xp.shape[0]
    k_max     = min(K_SCAN_MAX, n_samples - 1)
    min_size  = max(2, int(K_MIN_CLUSTER_FRAC * n_samples))

    print(f"  Auto-selecting k ({K_SCAN_MIN}–{k_max}) via dendrogram gap (Ward) "
          f"(min cluster size = {min_size})…", flush=True)

    Z = linkage(Xp, method="ward")
    # Z has shape (n-1, 4); Z[:, 2] are the merge heights in ascending order.
    # The merge that forms k clusters from k+1 is at row (n - k - 1).
    # Gap for going from k+1 → k clusters = Z[n-k-1, 2] - Z[n-k-2, 2].
    n = n_samples
    gaps:    dict[int, float] = {}
    skipped: list[int]        = []

    for k in range(K_SCAN_MIN, k_max + 1):
        # Check balance: fcluster uses 1-based labels → subtract 1 for bincount
        lbl    = fcluster(Z, k, criterion="maxclust") - 1
        counts = np.bincount(lbl.astype(int))
        if counts.min() < min_size:          # degenerate split — skip
            skipped.append(k)
            continue
        h_merge_into_k_minus_1 = float(Z[n - k,     2])
        h_merge_into_k         = float(Z[n - k - 1, 2])
        gaps[k] = h_merge_into_k_minus_1 - h_merge_into_k

    if skipped:
        print(f"  Skipped degenerate k values (cluster < {min_size} stocks): {skipped}")

    if gaps:
        best_k = max(gaps, key=gaps.get)
        print(f"  → Best k = {best_k}  (dendrogram gap = {gaps[best_k]:.4f})")
        return best_k, gaps

    # ── Fallback: gap scan from k=3, relaxed balance floor ───────────────────
    # Triggered when ALL k values fail the 2 % balance filter.  This means a
    # small set of extreme outlier stocks form a persistent tiny branch that
    # is present for every k in [K_SCAN_MIN, k_max] (e.g. 3 NVDA-type stocks
    # in an SPX run that always merge last, keeping their branch at 3 stocks
    # for any cut from k=2 to k=15).
    #
    # We skip k=2 (always the outlier-branch vs main-body binary split that
    # created the large gap in the first place) and re-scan k=3–k_max with a
    # relaxed floor of ≥2 stocks (only singletons are rejected).  This finds
    # the next meaningful gap within the main body of the index.
    print("  ⚠  All k values failed the balance filter; "
          "falling back to gap scan from k=3 (relaxed floor = 2 stocks).",
          flush=True)

    fallback_min = 2   # just reject singletons
    for k in range(max(K_SCAN_MIN, 3), k_max + 1):
        lbl    = fcluster(Z, k, criterion="maxclust") - 1
        counts = np.bincount(lbl.astype(int))
        if counts.min() < fallback_min:
            continue
        gaps[k] = float(Z[n - k, 2]) - float(Z[n - k - 1, 2])

    if gaps:
        best_k = max(gaps, key=gaps.get)
        print(f"  → Best k = {best_k}  (fallback gap = {gaps[best_k]:.4f})")
        return best_k, gaps

    # Ultimate fallback: unconstrained gap scan from k=3 (no balance check)
    print("  ⚠  Relaxed scan also failed; using unconstrained gap from k=3.",
          flush=True)
    for k in range(max(K_SCAN_MIN, 3), k_max + 1):
        gaps[k] = float(Z[n - k, 2]) - float(Z[n - k - 1, 2])
    best_k = max(gaps, key=gaps.get)
    print(f"  → Best k = {best_k}  (unconstrained gap = {gaps[best_k]:.4f})")
    return best_k, gaps


def auto_select_k(Xp: np.ndarray, method: str) -> tuple[int, dict[int, float]]:
    """Dispatch to the appropriate k-selection strategy.

    K-Means  → silhouette scan with balance filter
    Hierarchical (Ward) → dendrogram gap method

    Returns (best_k, scores) where `scores` is either
    {k: silhouette} or {k: gap_size} depending on method.
    """
    if method == "kmeans":
        return _auto_k_silhouette(Xp)
    else:  # hierarchical
        return _auto_k_dendrogram_gap(Xp)


def run_pipeline(feat_df: pd.DataFrame, method: str, n_clusters: int,
                 viz: str) -> tuple:
    """
    StandardScaler → PCA (12 components) → clustering → 2-D embedding.

    n_clusters = 0  triggers automatic k selection via silhouette scan
                    (K-Means and Hierarchical only; DBSCAN is always auto).

    Returns
    -------
    result_df     : DataFrame with columns [x, y, cluster]  (index = ticker)
    expl_var      : explained variance ratio per PCA component
    labels        : cluster label per sample (int array)
    sil_scan      : {k: silhouette_score} dict from the auto-scan, or {} if
                    n_clusters was supplied manually or method == 'dbscan'
    pca           : fitted sklearn PCA object
    feature_names : list of column names from feat_df
    pca_scores    : DataFrame (index=tickers, columns=[PC1, PC2, ...]) —
                    PCA-transformed coordinates before embedding
    """
    from sklearn.preprocessing  import StandardScaler
    from sklearn.decomposition  import PCA
    from sklearn.cluster        import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.manifold       import TSNE

    print(f"  Running pipeline: scaler → PCA → {method} → {viz}…")

    X = feat_df.values

    # 1. Standardise
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    # 2. PCA for denoising / speed
    n_comp = min(12, Xs.shape[1], Xs.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    Xp     = pca.fit_transform(Xs)
    expl_var = list(pca.explained_variance_ratio_)

    feature_names = list(feat_df.columns)
    pca_scores = pd.DataFrame(
        Xp,
        index=feat_df.index,
        columns=[f"PC{i+1}" for i in range(Xp.shape[1])]
    )

    # 3. Clustering
    sil_scan: dict[int, float] = {}

    if method == "kmeans":
        if n_clusters == 0:
            nc, sil_scan = auto_select_k(Xp, "kmeans")
        else:
            nc = min(n_clusters, len(feat_df) - 1)
        clf    = KMeans(n_clusters=nc, random_state=42, n_init=10)
        labels = clf.fit_predict(Xp).astype(int)

    elif method == "hierarchical":
        if n_clusters == 0:
            nc, sil_scan = auto_select_k(Xp, "hierarchical")
        else:
            nc = min(n_clusters, len(feat_df) - 1)
        clf    = AgglomerativeClustering(n_clusters=nc, linkage="ward")
        labels = clf.fit_predict(Xp).astype(int)

    elif method == "dbscan":
        # Auto-tune eps via the kneedle method on the sorted k-NN distance curve.
        #
        # Why not a fixed percentile?
        #   The 90th-percentile heuristic is far too permissive on large, dense
        #   datasets (500 SPX tickers): almost every point falls within each
        #   other's neighbourhood → everything merges into 1 cluster.
        #
        # Kneedle algorithm (simplified):
        #   1. Sort the 5-NN distances ascending.
        #   2. Normalise both axes to [0, 1].
        #   3. The "knee" is the index that maximises (y_norm − x_norm), i.e.
        #      the point of maximum deviation above the straight diagonal line.
        #      It marks the natural inflection between the dense-core region
        #      (cluster members) and the sparse tail (noise/outliers).
        #   4. Floor at the 5th-percentile distance so eps never collapses
        #      to zero on pathologically compact data.
        from sklearn.neighbors import NearestNeighbors
        nbrs      = NearestNeighbors(n_neighbors=5).fit(Xp)
        dists, _  = nbrs.kneighbors(Xp)
        knn_dists = np.sort(dists[:, -1])
        n_pts     = len(knn_dists)
        x_norm    = np.linspace(0.0, 1.0, n_pts)
        y_norm    = (knn_dists - knn_dists[0]) / (knn_dists[-1] - knn_dists[0] + 1e-12)
        knee_idx  = int(np.argmax(y_norm - x_norm))
        eps_knee  = float(knn_dists[knee_idx])
        eps_floor = float(np.percentile(knn_dists, 5))
        eps       = max(eps_knee, eps_floor)
        print(f"  DBSCAN eps = {eps:.4f}  (knee idx {knee_idx}/{n_pts})")
        clf    = DBSCAN(eps=eps, min_samples=5)
        labels = clf.fit_predict(Xp).astype(int)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 4. 2-D embedding for visualisation
    if viz == "tsne":
        perp = min(30, max(5, len(feat_df) // 10))
        # scikit-learn ≥1.4 renamed n_iter → max_iter; support both
        import sklearn
        tsne_iter_kwarg = (
            {"max_iter": 1000}
            if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4)
            else {"n_iter": 1000}
        )
        emb  = TSNE(n_components=2, perplexity=perp, random_state=42,
                    init="pca", **tsne_iter_kwarg).fit_transform(Xp)
        x, y = emb[:, 0], emb[:, 1]
    elif viz == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                                min_dist=0.1)
            emb     = reducer.fit_transform(Xp)
            x, y   = emb[:, 0], emb[:, 1]
        except ImportError:
            print("  ⚠  umap-learn not installed, falling back to PCA viz")
            x, y = Xp[:, 0], Xp[:, 1]
    else:  # pca
        x, y = Xp[:, 0], Xp[:, 1]

    result_df = pd.DataFrame({"x": x, "y": y, "cluster": labels},
                             index=feat_df.index)
    return result_df, expl_var, labels, sil_scan, pca, feature_names, pca_scores


# ── DTW clustering pipeline ────────────────────────────────────────────────────

def run_dtw_pipeline(prices: pd.DataFrame, n_clusters: int, viz: str,
                     window: int = 63) -> tuple:
    """
    DTW-based hierarchical clustering on raw normalised return series.

    Bypasses the feature matrix — clusters stocks by *price pattern similarity*
    with lag-tolerance that Pearson correlation misses.

    Returns same structure as run_pipeline but pca=None, feature_names=[],
    pca_scores=empty DataFrame.
    """
    from sklearn.cluster       import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold      import TSNE
    import scipy.spatial.distance as ssd

    print(f"  Running DTW pipeline: {window}d return series → hierarchical → {viz}…")

    ret = prices.pct_change().dropna(how="all").tail(window)
    # Keep tickers that have a full window
    valid = ret.columns[ret.notna().all()]
    ret   = ret[valid]

    tickers = list(ret.columns)
    n       = len(tickers)
    if n < 10:
        raise ValueError("Too few tickers with complete return history for DTW.")

    # Normalise each series to zero-mean unit-variance
    seqs = ret.values.T.astype(np.float64)      # (n, window)
    seqs = (seqs - seqs.mean(axis=1, keepdims=True)) / (seqs.std(axis=1, keepdims=True) + 1e-12)

    # --- DTW distance matrix ---
    def _build_dtw_matrix(S: np.ndarray, radius: int = 5) -> np.ndarray:
        """Pairwise DTW with Sakoe-Chiba band. Tries fast libs first."""
        try:
            from tslearn.metrics import cdist_dtw
            print("    dtw: using tslearn …")
            return cdist_dtw(S[:, :, np.newaxis],
                             global_constraint="sakoe_chiba",
                             sakoe_chiba_radius=radius)
        except ImportError:
            pass
        try:
            import dtaidistance.dtw as _dtw_lib
            print("    dtw: using dtaidistance …")
            return _dtw_lib.distance_matrix_fast(S, window=radius)
        except ImportError:
            pass
        # Pure-numpy fallback
        print(f"    dtw: pure-numpy fallback (n={S.shape[0]}, T={S.shape[1]}) …")
        nn, T = S.shape
        D = np.zeros((nn, nn), dtype=np.float32)
        for i in range(nn - 1):
            for j in range(i + 1, nn):
                s1, s2 = S[i], S[j]
                prev = np.full(T + 1, np.inf)
                prev[0] = 0.0
                for r in range(1, T + 1):
                    curr = np.full(T + 1, np.inf)
                    lo = max(1, r - radius)
                    hi = min(T, r + radius)
                    for c in range(lo, hi + 1):
                        cost = abs(s1[r-1] - s2[c-1])
                        curr[c] = cost + min(prev[c], prev[c-1], curr[c-1])
                    prev = curr
                D[i, j] = D[j, i] = prev[T]
        return D

    D = _build_dtw_matrix(seqs)
    # Ensure symmetry and zero diagonal
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)

    # --- Hierarchical clustering on precomputed distances ---
    if n_clusters == 0:
        # Simple heuristic: sqrt(n/2), clamped to [4, 12]
        nc = int(np.clip(round(np.sqrt(n / 2)), 4, 12))
        print(f"    dtw: auto k = {nc} (heuristic)")
    else:
        nc = min(n_clusters, n - 1)

    clf    = AgglomerativeClustering(n_clusters=nc, metric="precomputed", linkage="average")
    labels = clf.fit_predict(D).astype(int)

    # --- 2-D embedding for visualisation ---
    # Use TSNE with precomputed distance matrix
    perp = min(30, max(5, n // 10))
    import sklearn
    tsne_iter_kwarg = (
        {"max_iter": 1000}
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4)
        else {"n_iter": 1000}
    )
    if viz in ("tsne", "umap"):
        # TSNE can accept a precomputed distance matrix
        emb = TSNE(n_components=2, perplexity=perp, random_state=42,
                   metric="precomputed", init="random", **tsne_iter_kwarg).fit_transform(D)
    else:  # pca → use MDS
        from sklearn.manifold import MDS
        emb = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, n_init=1, max_iter=300).fit_transform(D)
    x, y = emb[:, 0], emb[:, 1]

    result_df = pd.DataFrame({"x": x, "y": y, "cluster": labels}, index=tickers)
    expl_var  = []          # not applicable for DTW
    sil_scan  = {}
    return result_df, expl_var, labels, sil_scan, None, [], pd.DataFrame()


# ── HTML report ────────────────────────────────────────────────────────────────

def _colour_for(cluster_id: int) -> str:
    if cluster_id == -1:
        return "#555577"   # DBSCAN noise
    return CLUSTER_COLOURS[cluster_id % len(CLUSTER_COLOURS)]


def build_html(embed_df: pd.DataFrame,
               feat_df:  pd.DataFrame,
               prices:   pd.DataFrame,
               meta:     dict[str, dict],   # ticker → {name, sector}
               expl_var: list[float],
               labels:   np.ndarray,
               args:     argparse.Namespace,
               sil_scan: dict[int, float] | None = None,
               pca=None,
               feature_names=None,
               pca_scores=None) -> str:
    """Assemble the full HTML report."""
    import plotly.graph_objects as go
    import plotly.io            as pio

    # ── Merge everything into one master frame ──────────────────────────────
    df = embed_df.copy()
    df["cluster"]  = labels
    df["name"]     = df.index.map(lambda t: meta.get(t, {}).get("name",   t))
    df["sector"]   = df.index.map(lambda t: meta.get(t, {}).get("sector", "Unknown"))

    # Return metrics from price series
    ret = prices.pct_change().dropna(how="all")
    for window, col in [(21, "ret_1m"), (63, "ret_3m"), (252, "ret_1y")]:
        r_series = ret.tail(window).add(1).prod().sub(1).mul(100)
        df[col]  = df.index.map(lambda t, rs=r_series: rs.get(t, np.nan))

    # These columns are used in hover tooltips / summary table but may be absent
    # when the user has deselected them via --features.  Fall back to NaN series.
    _nan = pd.Series(np.nan, index=feat_df.index)
    df["vol_21d_pct"] = (feat_df["vol_21d"].mul(100) if "vol_21d" in feat_df.columns else _nan).reindex(df.index)
    df["beta"]        = (feat_df["beta"]              if "beta"    in feat_df.columns else _nan).reindex(df.index)
    df["max_dd"]      = (feat_df["max_dd_252d"].mul(100) if "max_dd_252d" in feat_df.columns else _nan).reindex(df.index)

    unique_clusters = sorted(df["cluster"].unique())
    n_noise         = int((labels == -1).sum())

    # ── 1. Scatter plot ─────────────────────────────────────────────────────
    scatter_traces = []
    for cid in unique_clusters:
        mask  = df["cluster"] == cid
        sub   = df[mask]
        clabel = "Noise" if cid == -1 else f"Cluster {cid + 1}"
        colour = _colour_for(cid)

        hover = (
            "<b>%{customdata[0]}</b> (%{text})<br>"
            "Cluster: " + clabel + "<br>"
            "Sector: %{customdata[1]}<br>"
            "1M: %{customdata[2]:.1f}%  3M: %{customdata[3]:.1f}%  1Y: %{customdata[4]:.1f}%<br>"
            "Vol(ann): %{customdata[5]:.1f}%  Beta: %{customdata[6]:.2f}  MaxDD: %{customdata[7]:.1f}%"
            "<extra></extra>"
        )

        scatter_traces.append(go.Scatter(
            x    = sub["x"],
            y    = sub["y"],
            mode = "markers",
            name = clabel,
            text = sub.index.tolist(),
            customdata = sub[["name", "sector", "ret_1m", "ret_3m", "ret_1y",
                               "vol_21d_pct", "beta", "max_dd"]].values,
            hovertemplate = hover,
            marker = dict(
                color   = colour,
                size    = 8,
                opacity = 0.82,
                line    = dict(color="rgba(255,255,255,0.13)", width=0.5),
            ),
        ))

    viz_labels = {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA"}
    viz_label  = viz_labels.get(args.viz, args.viz.upper())

    scatter_fig = go.Figure(scatter_traces)
    scatter_fig.update_layout(
        template   = "plotly_dark",
        paper_bgcolor = BG,
        plot_bgcolor  = PANEL,
        font       = dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
        title      = dict(text=f"{viz_label} Embedding — {args.index} Behavioural Clusters",
                          x=0.02, font=dict(size=16, color=TEXT)),
        legend     = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis      = dict(title=f"{viz_label} 1", gridcolor=BORDER, zeroline=False),
        yaxis      = dict(title=f"{viz_label} 2", gridcolor=BORDER, zeroline=False),
        margin     = dict(l=50, r=20, t=60, b=50),
        height     = 600,
    )

    # ── 2. PCA explained-variance bar ──────────────────────────────────────
    pca_x = [f"PC{i+1}" for i in range(len(expl_var))]
    pca_y = [round(v * 100, 2) for v in expl_var]
    cum_y = list(np.cumsum(pca_y))

    pca_fig = go.Figure()
    pca_fig.add_trace(go.Bar(
        x=pca_x, y=pca_y,
        name="Explained %",
        marker_color=ACCENT,
    ))
    pca_fig.add_trace(go.Scatter(
        x=pca_x, y=cum_y,
        name="Cumulative %",
        mode="lines+markers",
        line=dict(color="#f7706a", width=2),
        marker=dict(size=6),
        yaxis="y2",
    ))
    pca_fig.update_layout(
        template      = "plotly_dark",
        paper_bgcolor = BG,
        plot_bgcolor  = PANEL,
        font          = dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
        title         = dict(text="PCA Explained Variance", x=0.02,
                             font=dict(size=14, color=TEXT)),
        legend        = dict(bgcolor="rgba(0,0,0,0)"),
        xaxis         = dict(gridcolor=BORDER),
        yaxis         = dict(title="Component %",   gridcolor=BORDER, range=[0, None]),
        yaxis2        = dict(title="Cumulative %",  overlaying="y", side="right",
                             gridcolor="rgba(0,0,0,0)", range=[0, 105]),
        margin        = dict(l=50, r=60, t=50, b=40),
        height        = 280,
        barmode       = "group",
    )

    # ── 3. Cluster summary table ───────────────────────────────────────────
    summary_rows = []
    for cid in sorted(c for c in unique_clusters if c != -1):
        sub  = df[df["cluster"] == cid]
        row  = {
            "Cluster":         f"Cluster {cid + 1}",
            "N":               len(sub),
            "Avg 1M (%)":      f"{sub['ret_1m'].mean():.1f}",
            "Avg 3M (%)":      f"{sub['ret_3m'].mean():.1f}",
            "Avg 1Y (%)":      f"{sub['ret_1y'].mean():.1f}",
            "Avg Vol (%)":     f"{sub['vol_21d_pct'].mean():.1f}",
            "Avg Beta":        f"{sub['beta'].mean():.2f}",
            "Avg MaxDD (%)":   f"{sub['max_dd'].mean():.1f}",
            "Top Sectors":     ", ".join(
                sub["sector"].value_counts().head(3).index.tolist()
            ),
        }
        summary_rows.append(row)

    if n_noise:
        sub = df[df["cluster"] == -1]
        summary_rows.append({
            "Cluster":       "Noise",
            "N":             len(sub),
            "Avg 1M (%)":   f"{sub['ret_1m'].mean():.1f}",
            "Avg 3M (%)":   f"{sub['ret_3m'].mean():.1f}",
            "Avg 1Y (%)":   f"{sub['ret_1y'].mean():.1f}",
            "Avg Vol (%)":  f"{sub['vol_21d_pct'].mean():.1f}",
            "Avg Beta":     f"{sub['beta'].mean():.2f}",
            "Avg MaxDD (%)":f"{sub['max_dd'].mean():.1f}",
            "Top Sectors":  "—",
        })

    summary_df = pd.DataFrame(summary_rows)

    # ── 4. Correlation heatmap (sample up to 150 for performance) ──────────
    ret_for_corr = prices.pct_change().dropna(how="all").tail(126)
    tickers_ord  = df.sort_values("cluster").index.tolist()
    tickers_corr = [t for t in tickers_ord if t in ret_for_corr.columns]
    if len(tickers_corr) > 150:
        # Keep top 5 per cluster
        keep = []
        for cid in unique_clusters:
            cl_tickers = df[df["cluster"] == cid].index.tolist()
            keep.extend(cl_tickers[:5])
        tickers_corr = [t for t in tickers_ord if t in keep][:150]

    # JSON-encode for the in-browser heatmap search
    tickers_corr_json = json.dumps(tickers_corr)

    corr_m = ret_for_corr[tickers_corr].corr().values

    corr_fig = go.Figure(go.Heatmap(
        z         = corr_m,
        x         = tickers_corr,
        y         = tickers_corr,
        colorscale= [[0, "#1a1a2e"], [0.5, PANEL], [1, ACCENT]],
        zmin      = -1, zmax = 1,
        showscale = True,
        hovertemplate = "%{x} / %{y}: %{z:.2f}<extra></extra>",
    ))
    corr_fig.update_layout(
        template      = "plotly_dark",
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        font          = dict(family="Inter, system-ui, sans-serif", color=TEXT, size=9),
        title         = dict(text="Return Correlation (126d) — ordered by cluster",
                             x=0.02, font=dict(size=14, color=TEXT)),
        xaxis         = dict(showticklabels=len(tickers_corr) <= 60, tickangle=45),
        yaxis         = dict(showticklabels=len(tickers_corr) <= 60),
        margin        = dict(l=80, r=20, t=50, b=80),
        height        = max(350, min(600, len(tickers_corr) * 4)),
    )

    # ── 5. Ticker detail table (full list) ────────────────────────────────
    detail_df = df[["name", "sector", "cluster", "ret_1m", "ret_3m", "ret_1y",
                    "vol_21d_pct", "beta", "max_dd"]].copy()
    detail_df["cluster_label"] = detail_df["cluster"].apply(
        lambda c: "Noise" if c == -1 else f"Cluster {c + 1}"
    )
    detail_df = detail_df.sort_values(["cluster", "ret_1y"], ascending=[True, False])
    detail_df.index.name = "Ticker"
    detail_df = detail_df.reset_index()

    # ── 6. Auto-selection chart (silhouette for k-means, gap for hierarchical) ─
    sil_div       = ""
    sil_auto_note = ""
    if sil_scan:
        best_k      = max(sil_scan, key=sil_scan.get)
        sil_ks      = list(sil_scan.keys())
        sil_vs      = [sil_scan[k] for k in sil_ks]
        bar_colours = [ACCENT if k == best_k else BORDER for k in sil_ks]

        is_gap = (args.method == "hierarchical")
        if is_gap:
            chart_title  = "Dendrogram Gap by k — Auto-selection (Ward)"
            y_axis_label = "Merge-height gap"
            hover_tmpl   = "k = %{x}<br>Gap = %{y:.4f}<extra></extra>"
            score_label  = f"gap = {sil_scan[best_k]:.4f}"
        else:
            chart_title  = "Composite Score by k — Auto-selection (Silhouette + Calinski-Harabász)"
            y_axis_label = "Composite score (normalised)"
            hover_tmpl   = "k = %{x}<br>Score = %{y:.4f}<extra></extra>"
            score_label  = f"composite = {sil_scan[best_k]:.3f}"

        sil_fig = go.Figure(go.Bar(
            x             = sil_ks,
            y             = sil_vs,
            marker_color  = bar_colours,
            hovertemplate = hover_tmpl,
        ))
        sil_fig.add_annotation(
            x=best_k, y=sil_scan[best_k],
            text=f"  ← selected  k={best_k}",
            showarrow=False,
            font=dict(color=ACCENT, size=11),
            xanchor="left",
        )
        sil_fig.update_layout(
            template      = "plotly_dark",
            paper_bgcolor = BG,
            plot_bgcolor  = PANEL,
            font          = dict(family="Inter, system-ui, sans-serif",
                                 color=TEXT, size=12),
            title         = dict(text=chart_title,
                                 x=0.02, font=dict(size=14, color=TEXT)),
            xaxis         = dict(title="k (number of clusters)", gridcolor=BORDER,
                                 tickmode="linear", dtick=1),
            yaxis         = dict(title=y_axis_label, gridcolor=BORDER),
            margin        = dict(l=60, r=20, t=50, b=50),
            height        = 260,
            showlegend    = False,
        )
        if not is_gap:
            n_total   = len(embed_df)
            min_sz    = max(2, int(K_MIN_CLUSTER_FRAC * n_total))
            skipped_k = [k for k in range(K_SCAN_MIN, K_SCAN_MAX + 1)
                         if k not in sil_scan and k <= max(sil_scan)]
            skip_note = (f" &nbsp;·&nbsp; {len(skipped_k)} degenerate k value(s) excluded "
                         f"(min cluster size = {min_sz})" if skipped_k else "")
        else:
            skip_note = ""
        sil_auto_note = (
            f"k = <b>{best_k}</b> selected automatically "
            f"({score_label}){skip_note}"
        )

    # ── Convert figures to div HTML ──────────────────────────────────────
    def _div(fig, div_id):
        return pio.to_html(fig, full_html=False, div_id=div_id,
                           include_plotlyjs=False, config={"responsive": True})

    scatter_div = _div(scatter_fig, "scatter")
    pca_div     = _div(pca_fig,     "pca-var")
    corr_div    = _div(corr_fig,    "corr-heat")
    if sil_scan:
        sil_div = _div(sil_fig, "sil-scan")

    # ── 7. Minimum Spanning Tree ───────────────────────────────────────────────
    from scipy.sparse.csgraph import minimum_spanning_tree as _mst

    ret_mst  = prices.pct_change().dropna(how="all").tail(126)
    tk_mst   = [t for t in df.sort_values("cluster").index if t in ret_mst.columns]
    if len(tk_mst) > 200:
        # Keep top 5 per cluster to avoid unreadable hairball
        keep_mst = []
        for cid in unique_clusters:
            cl_t = df[df["cluster"] == cid].index.tolist()
            keep_mst.extend([t for t in cl_t if t in tk_mst][:5])
        tk_mst = [t for t in tk_mst if t in keep_mst][:200]

    corr_mst = ret_mst[tk_mst].corr().clip(-1, 1).values
    dist_mst = np.sqrt(np.clip(2 * (1 - corr_mst), 0, 4))
    np.fill_diagonal(dist_mst, 0)

    mst_sparse = _mst(dist_mst)
    mst_coo    = mst_sparse.tocoo()

    # Use embed_df x,y positions projected to this ticker subset
    pos_df = embed_df.reindex(tk_mst).dropna()
    tk_mst_valid = pos_df.index.tolist()

    # Build edge traces
    edge_x, edge_y, edge_hover = [], [], []
    for r_i, c_i, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        ta, tb = tk_mst[r_i], tk_mst[c_i]
        if ta not in pos_df.index or tb not in pos_df.index:
            continue
        xa, ya = pos_df.loc[ta, "x"], pos_df.loc[ta, "y"]
        xb, yb = pos_df.loc[tb, "x"], pos_df.loc[tb, "y"]
        edge_x += [xa, xb, None]
        edge_y += [ya, yb, None]
        corr_val = 1 - (w ** 2) / 2
        edge_hover.append(f"{ta}–{tb}: corr={corr_val:.2f}")

    mst_fig = go.Figure()
    mst_fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="rgba(100,100,150,0.45)", width=0.8),
        hoverinfo="skip",
        name="MST edge",
        showlegend=False,
    ))

    # Node traces — one per cluster (for legend / colour)
    for cid in unique_clusters:
        sub_t = [t for t in tk_mst_valid if df.loc[t, "cluster"] == cid]
        if not sub_t:
            continue
        label = "Noise" if cid == -1 else f"Cluster {cid + 1}"
        mst_fig.add_trace(go.Scatter(
            x=[pos_df.loc[t, "x"] for t in sub_t],
            y=[pos_df.loc[t, "y"] for t in sub_t],
            mode="markers+text",
            name=label,
            text=sub_t,
            textposition="top center",
            textfont=dict(size=7, color="rgba(200,200,200,0.7)"),
            marker=dict(
                color=_colour_for(cid),
                size=7, opacity=0.9,
                line=dict(color="rgba(0,0,0,0.3)", width=0.5),
            ),
            hovertemplate="%{text}<extra>" + label + "</extra>",
        ))

    mst_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=11),
        title=dict(text="Minimum Spanning Tree — Correlation Distance Network",
                   x=0.02, font=dict(size=14, color=TEXT)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=520,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    mst_div = _div(mst_fig, "mst-graph")

    # ── 8. PCA factor loadings ─────────────────────────────────────────────────
    pca_loadings_div   = ""
    pca_top_stocks_div = ""
    if pca is not None and feature_names:
        n_show_comp = min(8, len(pca.components_))
        comp_labels = [f"PC{i+1}" for i in range(n_show_comp)]
        feat_labels = list(feature_names)

        # Clamp loadings to n_show_comp rows
        loadings = pca.components_[:n_show_comp]   # (n_show_comp, n_features)

        load_fig = go.Figure(go.Heatmap(
            z=loadings,
            x=feat_labels,
            y=comp_labels,
            colorscale=[[0, "#1a3a5c"], [0.5, PANEL], [1, ACCENT]],
            zmid=0,
            showscale=True,
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
            text=[[f"{v:.2f}" for v in row] for row in loadings],
            texttemplate="%{text}",
            textfont=dict(size=8),
        ))
        load_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=10),
            title=dict(text="PCA Feature Loadings — top components", x=0.02,
                       font=dict(size=14, color=TEXT)),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=60, r=20, t=50, b=100),
            height=max(240, n_show_comp * 34 + 100),
        )
        pca_loadings_div = _div(load_fig, "pca-loadings")

        # Top/bottom 15 stocks on PC1 and PC2
        if pca_scores is not None and not pca_scores.empty:
            pc_figs = []
            for pc_col in ["PC1", "PC2"]:
                if pc_col not in pca_scores.columns:
                    continue
                scores = pca_scores[pc_col].reindex(df.index).dropna().sort_values()
                n_each = min(15, len(scores) // 2)
                show_t  = pd.concat([scores.head(n_each), scores.tail(n_each)])
                colours = [
                    _colour_for(int(df.loc[t, "cluster"])) if t in df.index else BORDER
                    for t in show_t.index
                ]
                pc_fig = go.Figure(go.Bar(
                    y=show_t.index.tolist(),
                    x=show_t.values,
                    orientation="h",
                    marker_color=colours,
                    hovertemplate="%{y}: %{x:.2f}<extra></extra>",
                ))
                pc_fig.add_vline(x=0, line_color=BORDER, line_width=1)
                pc_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=BG, plot_bgcolor=PANEL,
                    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=10),
                    title=dict(text=f"{pc_col} Extreme Stocks (top & bottom {n_each})",
                               x=0.02, font=dict(size=13, color=TEXT)),
                    xaxis=dict(title=f"{pc_col} score", gridcolor=BORDER),
                    yaxis=dict(tickfont=dict(size=9)),
                    margin=dict(l=80, r=20, t=40, b=40),
                    height=max(300, n_each * 2 * 18 + 80),
                )
                pc_figs.append(_div(pc_fig, f"pca-top-{pc_col.lower()}"))
            pca_top_stocks_div = "\n".join(pc_figs)

    # ── 9. Density-based clusters on embedding ────────────────────────────────
    density_div = ""
    try:
        from sklearn.cluster import HDBSCAN as _HDBSCAN
        _have_hdbscan = True
    except ImportError:
        try:
            import hdbscan as _hdbscan_lib
            _have_hdbscan = True
        except ImportError:
            _have_hdbscan = False

    if _have_hdbscan:
        emb_xy = embed_df[["x", "y"]].values
        try:
            from sklearn.cluster import HDBSCAN as _HDBSCAN
            _hdb = _HDBSCAN(min_cluster_size=max(3, len(emb_xy) // 20),
                             min_samples=3, cluster_selection_epsilon=0.0)
        except ImportError:
            import hdbscan as _hdbscan_lib
            _hdb = _hdbscan_lib.HDBSCAN(min_cluster_size=max(3, len(emb_xy) // 20),
                                         min_samples=3)
        density_labels = _hdb.fit_predict(emb_xy)

        # Map unique density labels to colours (reuse _colour_for palette)
        density_unique = sorted(set(density_labels))
        dens_fig = go.Figure()
        for did in density_unique:
            mask  = density_labels == did
            sub_t = embed_df.index[mask].tolist()
            dlabel = "Noise" if did == -1 else f"D-Cluster {did + 1}"
            dens_fig.add_trace(go.Scatter(
                x=embed_df.loc[sub_t, "x"].values,
                y=embed_df.loc[sub_t, "y"].values,
                mode="markers",
                name=dlabel,
                text=sub_t,
                marker=dict(
                    color=_colour_for(did),
                    size=7, opacity=0.85,
                    line=dict(color="rgba(0,0,0,0.2)", width=0.5),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + "<br>".join(
                        f"{c}: %{{customdata[{i}]}}"
                        for i, c in enumerate(["Sector", "Cluster"])
                    )
                    + f"<extra>{dlabel}</extra>"
                ),
                customdata=[
                    [df.loc[t, "sector"] if t in df.index else "",
                     f"Cluster {int(df.loc[t, 'cluster'])+1}" if t in df.index else ""]
                    for t in sub_t
                ],
            ))
        dens_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG, plot_bgcolor=PANEL,
            font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=11),
            title=dict(text="Density Clusters (HDBSCAN on Embedding Geometry)",
                       x=0.02, font=dict(size=14, color=TEXT)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
            height=480,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        )
        density_div = _div(dens_fig, "density-scatter")

    # ── Build new panel HTML strings ──────────────────────────────────────────
    mst_panel_html = (
        '  <!-- MST network -->\n'
        '  <div class="panel">\n'
        '    <div class="panel-title">Minimum Spanning Tree — Correlation Distance Network</div>\n'
        f'    {mst_div}\n'
        '    <p style="margin-top:8px;font-size:12px;color:var(--subtext)">\n'
        '      Edges connect stocks nearest in Mantegna distance (d=\u221a(2(1\u2212\u03c1))). Node positions match the embedding above.\n'
        + (f'      Showing sampled subset ({len(tk_mst)} tickers, 5 per cluster).\n' if len(tk_mst) < len(embed_df) else '') +
        '    </p>\n'
        '  </div>'
    )

    density_panel_html = (
        '  <!-- Density clustering -->\n'
        '  <div class="panel">\n'
        '    <div class="panel-title">Density Clusters (HDBSCAN on Embedding)</div>\n'
        f'    {density_div}\n'
        '    <p style="margin-top:8px;font-size:12px;color:var(--subtext)">\n'
        '      HDBSCAN applied to the 2-D embedding coordinates. Finds geometrically dense groups without convex-shape assumptions. Grey = noise. Compare against feature-based clusters above.\n'
        '    </p>\n'
        '  </div>'
    ) if density_div else ''

    pca_panel_html = (
        '  <!-- PCA factor loadings -->\n'
        '  <div class="panel">\n'
        '    <div class="panel-title">PCA Factor Loadings</div>\n'
        f'    {pca_loadings_div}\n'
        '    <p style="margin-top:8px;font-size:12px;color:var(--subtext)">\n'
        '      Feature contributions to each principal component. Blue=negative, violet=positive. Larger absolute value = stronger driver.\n'
        '    </p>\n'
        '  </div>\n'
        '  <div class="grid-2">\n'
        f'    {pca_top_stocks_div}\n'
        '  </div>'
    ) if pca_loadings_div else ''

    # ── Summary table HTML ────────────────────────────────────────────────
    def _th(col): return f'<th>{col}</th>'
    def _td(val): return f'<td>{val}</td>'

    summ_header = "<tr>" + "".join(_th(c) for c in summary_df.columns) + "</tr>"
    summ_rows   = ""
    for _, row in summary_df.iterrows():
        summ_rows += "<tr>" + "".join(_td(v) for v in row.values) + "</tr>"

    # Colour-coding helper for detail table
    def _signed_td(val, fmt=".1f"):
        try:
            v = float(val)
            colour = "#6af7a4" if v > 0 else "#f7706a" if v < 0 else TEXT
            return f'<td style="color:{colour}">{v:{fmt[1:]}}</td>'
        except Exception:
            return f'<td>{val}</td>'

    detail_header = "<tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Cluster</th>"
    detail_header += "<th>1M %</th><th>3M %</th><th>1Y %</th><th>Vol %</th><th>Beta</th><th>MaxDD %</th></tr>"

    detail_rows = ""
    for _, row in detail_df.iterrows():
        cid = int(row["cluster"])
        col = _colour_for(cid)
        cl  = row["cluster_label"]
        detail_rows += (
            f'<tr>'
            f'<td><b>{row["Ticker"]}</b></td>'
            f'<td>{row["name"]}</td>'
            f'<td>{row["sector"]}</td>'
            f'<td style="color:{col};font-weight:600">{cl}</td>'
            + _signed_td(row["ret_1m"])
            + _signed_td(row["ret_3m"])
            + _signed_td(row["ret_1y"])
            + f'<td>{row["vol_21d_pct"]:.1f}</td>'
            + f'<td>{row["beta"]:.2f}</td>'
            + _signed_td(row["max_dd"])
            + f'</tr>'
        )

    # ── Meta info ─────────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y-%m-%d %H:%M")
    method_map  = {"kmeans": "K-Means", "hierarchical": "Hierarchical (Ward)", "dbscan": "DBSCAN", "dtw": "DTW + Hierarchical"}
    method_str  = method_map.get(args.method, args.method)
    n_c_str     = f"{len([c for c in unique_clusters if c != -1])} clusters"
    if n_noise:
        n_c_str += f" + {n_noise} noise pts"

    # Cluster colour legend chips
    legend_chips = ""
    for cid in sorted(c for c in unique_clusters if c != -1):
        col  = _colour_for(cid)
        legend_chips += (
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin:4px 10px 4px 0;background:{PANEL};border-radius:6px;'
            f'padding:4px 10px;border:1px solid {BORDER}">'
            f'<span style="width:12px;height:12px;border-radius:50%;'
            f'background:{col};display:inline-block"></span>'
            f'<span style="color:{TEXT};font-size:13px">Cluster {cid + 1}</span></span>'
        )
    if n_noise:
        legend_chips += (
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin:4px 0;background:{PANEL};border-radius:6px;'
            f'padding:4px 10px;border:1px solid {BORDER}">'
            f'<span style="width:12px;height:12px;border-radius:50%;'
            f'background:#555577;display:inline-block"></span>'
            f'<span style="color:{TEXT};font-size:13px">Noise</span></span>'
        )

    # Silhouette score (exclude noise)
    sil_str = "N/A"
    try:
        from sklearn.metrics import silhouette_score
        mask_valid = labels != -1
        if mask_valid.sum() >= 10 and len(set(labels[mask_valid])) >= 2:
            from sklearn.preprocessing import StandardScaler
            Xs_for_sil = StandardScaler().fit_transform(feat_df.values)
            sil = silhouette_score(Xs_for_sil[mask_valid], labels[mask_valid])
            sil_str = f"{sil:.3f}"
    except Exception:
        pass

    # DBSCAN high-noise warning (threshold: > 40 % noise)
    NOISE_WARN_THRESHOLD = 0.40
    noise_ratio     = n_noise / max(len(labels), 1)
    dbscan_warn_html = ""
    if args.method == "dbscan" and noise_ratio > NOISE_WARN_THRESHOLD:
        noise_pct = f"{noise_ratio * 100:.0f}%"
        n_real    = len([c for c in unique_clusters if c != -1])
        dbscan_warn_html = f"""
  <div class="warn-banner">
    <span class="warn-icon">⚠</span>
    <div>
      <b>DBSCAN: {noise_pct} of tickers classified as noise ({n_noise} / {len(labels)})</b><br>
      <span>
        DBSCAN is a density-based algorithm — it only forms clusters where points are
        tightly packed, and marks everything else as noise. S&P&nbsp;500 behavioural
        data is a smooth continuum with no clear density voids, so most stocks fall
        outside the dense cores and get flagged rather than assigned.<br>
        <b>Recommendation:</b> use <b>K-Means</b> or <b>Hierarchical (Ward)</b> to
        ensure every ticker is assigned to a cluster. DBSCAN is better suited as an
        outlier-detection pass than as a primary segmentation tool on this data.
      </span>
    </div>
  </div>"""

    # ── Final HTML assembly ──────────────────────────────────────────────

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Behavioural Classifier — {args.index}</title>
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:      {BG};
    --panel:   {PANEL};
    --accent:  {ACCENT};
    --text:    {TEXT};
    --subtext: {SUBTEXT};
    --border:  {BORDER};
  }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: Inter, system-ui, -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }}
  a {{ color: var(--accent); }}

  /* ── Header ── */
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
  .header-title {{ font-size: 22px; font-weight: 700; letter-spacing: -.5px; }}
  .header-meta  {{ font-size: 12px; color: var(--subtext); }}

  /* ── Pills row ── */
  .pill-row {{
    display: flex; flex-wrap: wrap; gap: 8px;
    padding: 14px 28px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
  }}
  .pill {{
    background: #1e1e2e;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: var(--subtext);
  }}
  .pill b {{ color: var(--text); }}

  /* ── Main layout ── */
  .main {{ padding: 24px 28px; max-width: 1600px; margin: 0 auto; }}

  /* ── Panels ── */
  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 22px;
  }}
  .panel-title {{
    font-size: 14px; font-weight: 600;
    color: var(--subtext); text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: 14px;
  }}

  /* ── Legend chips ── */
  .legend-wrap {{ margin-bottom: 10px; }}

  /* ── Tables ── */
  .table-wrap {{ overflow-x: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th {{
    background: #1e1e2e;
    color: var(--subtext);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: .05em;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  td {{
    padding: 7px 12px;
    border-bottom: 1px solid #1e1e2e;
    white-space: nowrap;
    vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #1e1e2e55; }}

  /* ── Search box ── */
  .search-wrap {{ margin-bottom: 12px; display: flex; gap: 10px; align-items: center; }}
  #ticker-search {{
    background: #1e1e2e;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 14px;
    color: var(--text);
    font-size: 13px;
    width: 240px;
  }}
  #ticker-search:focus {{ outline: none; border-color: var(--accent); }}
  .row-count {{ font-size: 12px; color: var(--subtext); }}

  /* ── Grid ── */
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
  }}
  @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

  /* ── Scrollable ticker table ── */
  #detail-tbody tr {{ cursor: pointer; }}

  /* ── DBSCAN high-noise warning banner ── */
  .warn-banner {{
    display: flex; gap: 14px; align-items: flex-start;
    background: #2a1800; border: 1px solid #c87a00;
    border-left: 4px solid #f0a000;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;
    color: #f0c060; font-size: 13px; line-height: 1.6;
  }}
  .warn-banner b {{ color: #ffd080; }}
  .warn-icon {{ font-size: 22px; flex-shrink: 0; margin-top: 2px; }}

  /* ── Plot ticker search bar ── */
  .plot-search-wrap {{
    display: flex; align-items: center; gap: 8px;
    background: #1a1a2e; border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 12px; flex-shrink: 0;
  }}
  .plot-search-lbl {{
    font-size: 11px; color: var(--subtext); white-space: nowrap;
    text-transform: uppercase; letter-spacing: .05em;
  }}
  #plot-ticker-search {{
    background: transparent; border: none; outline: none;
    color: var(--text); font-family: inherit; font-size: 13px;
    width: 100px; letter-spacing: 0.04em;
  }}
  #plot-ticker-search::placeholder {{ color: #404060; }}
  #plot-search-status {{ font-size: 12px; min-width: 80px; text-align: right; }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div>
    <div class="header-title">📊 Behavioural Classifier — {args.index}</div>
    <div class="header-meta">Generated {ts} &nbsp;·&nbsp; Valuation Suite</div>
  </div>
  <div class="header-meta" style="text-align:right">
    Method: <b>{method_str}</b> &nbsp;·&nbsp;
    Viz: <b>{viz_label}</b> &nbsp;·&nbsp;
    Result: <b>{n_c_str}</b> &nbsp;·&nbsp;
    Silhouette: <b>{sil_str}</b>
  </div>
</div>

<!-- Pill row -->
<div class="pill-row">
  <div class="pill">Index <b>{args.index}</b></div>
  <div class="pill">Method <b>{method_str}</b></div>
  <div class="pill">Visualisation <b>{viz_label}</b></div>
  <div class="pill">Tickers <b>{len(embed_df)}</b></div>
  <div class="pill">Features <b>{feat_df.shape[1]}</b></div>
  <div class="pill">Silhouette <b>{sil_str}</b></div>
</div>

<div class="main">

  <!-- Legend -->
  <div class="legend-wrap">{legend_chips}</div>

  <!-- DBSCAN high-noise warning (only rendered when noise > 40 %) -->
  {dbscan_warn_html}

  <!-- 2-D Scatter -->
  <div class="panel">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
      <div class="panel-title" style="margin-bottom:0">{viz_label} Embedding</div>
      <div class="plot-search-wrap">
        <span class="plot-search-lbl">Find ticker</span>
        <input type="text" id="plot-ticker-search" placeholder="e.g. NVDA, AAPL, MSFT"
               oninput="onPlotSearch(event)" autocomplete="off" spellcheck="false">
        <span id="plot-search-status"></span>
      </div>
    </div>
    {scatter_div}
    <p style="margin-top:10px;font-size:12px;color:var(--subtext)">
      {'<b>Note:</b> t-SNE axes have no absolute meaning — proximity indicates behavioural similarity; between-cluster distances are not comparable.' if args.viz == 'tsne' else ''}
      {'<b>Note:</b> UMAP preserves both local and approximate global structure. Axes have no absolute scale.' if args.viz == 'umap' else ''}
      {'<b>Note:</b> PCA axes represent the first two principal components of the feature space. Explained variance shown below.' if args.viz == 'pca' else ''}
    </p>
  </div>

  <!-- Cluster summary + (silhouette scan if auto) + PCA variance -->
  <div class="grid-2">
    <div class="panel">
      <div class="panel-title">Cluster Summary</div>
      <div class="table-wrap">
        <table>
          <thead>{summ_header}</thead>
          <tbody>{summ_rows}</tbody>
        </table>
      </div>
    </div>
    <div>
      {'<div class="panel"><div class="panel-title">K Selection — Silhouette Scan</div>'
       + '<p style="margin-bottom:10px;font-size:12px;color:var(--subtext)">' + sil_auto_note + '</p>'
       + sil_div
       + '</div>' if sil_scan else ''}
      <div class="panel">
        <div class="panel-title">PCA Explained Variance</div>
        {pca_div}
      </div>
    </div>
  </div>

  <!-- Correlation heatmap -->
  <div class="panel">
    <div class="panel-title">Return Correlation Heatmap (126d, ordered by cluster)</div>
    {corr_div}
    <p style="margin-top:8px;font-size:12px;color:var(--subtext)">
      Showing up to 150 tickers (5 per cluster); colour scale: purple = −1, dark = 0, violet = +1.
    </p>
  </div>

  {mst_panel_html}

  {density_panel_html}

  {pca_panel_html}

  <!-- Ticker detail table -->
  <div class="panel">
    <div class="panel-title">All Tickers</div>
    <div class="search-wrap">
      <input id="ticker-search" type="text" placeholder="Filter by ticker, name or sector…" oninput="filterTable()">
      <span class="row-count" id="row-count"></span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>{detail_header}</thead>
        <tbody id="detail-tbody">{detail_rows}</tbody>
      </table>
    </div>
  </div>

</div><!-- /main -->

<script>
// ── Embedded data for search ───────────────────────────────────────────────────
// Tickers in heatmap order (cluster-sorted); used by _searchHeatmap()
var CORR_TICKERS = {tickers_corr_json};

// ── Plot ticker search ─────────────────────────────────────────────────────────
// _origScatter: [{{color, size, opacity}}] per trace — cached on first search
var _origScatter   = null;
var _lastPlotQuery = "";

function _cacheScatter() {{
  if (_origScatter) return;
  var el = document.getElementById('scatter');
  if (!el || !el.data) return;
  _origScatter = el.data.map(function(trace) {{
    return {{
      color:   trace.marker.color,
      size:    trace.marker.size   !== undefined ? trace.marker.size   : 8,
      opacity: trace.marker.opacity !== undefined ? trace.marker.opacity : 0.82,
    }};
  }});
}}

function _restoreScatter() {{
  if (!_origScatter) return;
  var el = document.getElementById('scatter');
  if (!el) {{ _origScatter = null; return; }}
  var idxs   = _origScatter.map(function(_, i) {{ return i; }});
  var colors = _origScatter.map(function(o) {{ return o.color;   }});
  var sizes  = _origScatter.map(function(o) {{ return o.size;    }});
  var opas   = _origScatter.map(function(o) {{ return o.opacity; }});
  Plotly.restyle(el, {{'marker.color': colors, 'marker.size': sizes,
                        'marker.opacity': opas}}, idxs);
  _origScatter = null;
}}

function _searchScatter(queries) {{
  // queries: array of uppercase strings (one or more tickers)
  var el = document.getElementById('scatter');
  if (!el || !el.data) return {{found: false, tickers: []}};
  _cacheScatter();

  // Find best match for each query; collect {{t, p, ticker}} per match
  var matches = [];
  queries.forEach(function(q) {{
    if (!q) return;
    var passes = [
      function(s) {{ return s === q; }},
      function(s) {{ return s.startsWith(q); }},
      function(s) {{ return s.includes(q); }},
    ];
    for (var pass = 0; pass < passes.length; pass++) {{
      var hit = false;
      for (var t = 0; t < el.data.length; t++) {{
        var texts = el.data[t].text;
        if (!texts) continue;
        for (var p = 0; p < texts.length; p++) {{
          if (passes[pass](texts[p].toUpperCase())) {{
            matches.push({{t: t, p: p, ticker: texts[p]}});
            hit = true; break;
          }}
        }}
        if (hit) break;
      }}
      if (hit) break;
    }}
  }});

  if (matches.length === 0) {{ _restoreScatter(); return {{found: false, tickers: []}}; }}

  // Dim all traces uniformly
  Plotly.restyle(el, {{'marker.opacity': 0.07}});

  // Group matched points by trace index, then apply highlights per trace
  var byTrace = {{}};
  matches.forEach(function(m) {{
    if (!byTrace[m.t]) byTrace[m.t] = [];
    byTrace[m.t].push(m.p);
  }});
  Object.keys(byTrace).forEach(function(tStr) {{
    var t    = parseInt(tStr);
    var pts  = byTrace[t];
    var o    = _origScatter[t];
    var n    = el.data[t].text.length;
    var oCol = o.color;
    var oCols = Array.isArray(oCol) ? oCol.slice() : Array(n).fill(oCol);
    var newColors  = oCols.slice();
    var newSizes   = Array(n).fill(o.size);
    var newOpacity = Array(n).fill(0.07);
    pts.forEach(function(p) {{
      newColors[p]  = '#FFD700';
      newSizes[p]   = o.size * 2.5;
      newOpacity[p] = 1.0;
    }});
    Plotly.restyle(el, {{
      'marker.color':   [newColors],
      'marker.size':    [newSizes],
      'marker.opacity': [newOpacity],
    }}, [t]);
  }});

  return {{found: true, tickers: matches.map(function(m) {{ return m.ticker; }})}};
}}

function _restoreHeatmap() {{
  var el = document.getElementById('corr-heat');
  if (el) Plotly.relayout(el, {{shapes: []}});
}}

function _searchHeatmap(queries) {{
  // queries: array of uppercase strings
  var el = document.getElementById('corr-heat');
  if (!el || !CORR_TICKERS || !CORR_TICKERS.length) return;

  var shapes = [];
  queries.forEach(function(q) {{
    if (!q) return;
    var idx = -1;
    // exact → prefix → substring
    for (var i = 0; i < CORR_TICKERS.length; i++) {{
      if (CORR_TICKERS[i].toUpperCase() === q) {{ idx = i; break; }}
    }}
    if (idx === -1) {{
      for (var i = 0; i < CORR_TICKERS.length; i++) {{
        if (CORR_TICKERS[i].toUpperCase().startsWith(q)) {{ idx = i; break; }}
      }}
    }}
    if (idx === -1) {{
      for (var i = 0; i < CORR_TICKERS.length; i++) {{
        if (CORR_TICKERS[i].toUpperCase().includes(q)) {{ idx = i; break; }}
      }}
    }}
    if (idx === -1) return;
    shapes.push(
      {{ type: 'rect', layer: 'above', xref: 'x', yref: 'paper',
        x0: idx - 0.5, x1: idx + 0.5, y0: 0, y1: 1,
        fillcolor: 'rgba(255,215,0,0.13)', line: {{color: '#FFD700', width: 1.5}} }},
      {{ type: 'rect', layer: 'above', xref: 'paper', yref: 'y',
        x0: 0, x1: 1, y0: idx - 0.5, y1: idx + 0.5,
        fillcolor: 'rgba(255,215,0,0.13)', line: {{color: '#FFD700', width: 1.5}} }}
    );
  }});

  if (shapes.length === 0) {{ _restoreHeatmap(); return; }}
  Plotly.relayout(el, {{shapes: shapes}});
}}

function onPlotSearch(e) {{
  var raw      = e.target.value;
  var queries  = raw.split(',').map(function(s) {{ return s.trim().toUpperCase(); }}).filter(Boolean);
  _lastPlotQuery = raw;
  var statusEl = document.getElementById('plot-search-status');
  if (!queries.length) {{
    _restoreScatter();
    _restoreHeatmap();
    statusEl.textContent = "";
    return;
  }}
  var res = _searchScatter(queries);
  _searchHeatmap(queries);
  if (res.found) {{
    statusEl.textContent = res.tickers.join(', ');
    statusEl.style.color = '#FFD700';
  }} else {{
    statusEl.textContent = "not found";
    statusEl.style.color = '#e05c5c';
  }}
}}

// ── Detail table filter ────────────────────────────────────────────────────────
function filterTable() {{
  const q   = document.getElementById('ticker-search').value.toLowerCase();
  const trs = document.querySelectorAll('#detail-tbody tr');
  let   cnt = 0;
  trs.forEach(tr => {{
    const txt = tr.textContent.toLowerCase();
    const vis = txt.includes(q);
    tr.style.display = vis ? '' : 'none';
    if (vis) cnt++;
  }});
  document.getElementById('row-count').textContent = cnt + ' rows';
}}
window.addEventListener('DOMContentLoaded', () => {{
  document.getElementById('row-count').textContent =
    document.querySelectorAll('#detail-tbody tr').length + ' rows';
}});
</script>
</body>
</html>
"""
    return html


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Behavioural Clustering Tool")
    p.add_argument("--index",      default="SPX",
                   choices=["SPX", "NDX", "DOW", "RUT", "TSX"])
    p.add_argument("--method",     default="kmeans",
                   choices=["kmeans", "hierarchical", "dbscan", "dtw"])
    p.add_argument("--n_clusters", default=0, type=int,
                   help="Number of clusters (0 = auto via silhouette scan; "
                        "ignored for DBSCAN)")
    p.add_argument("--viz",        default="tsne",
                   choices=["tsne", "umap", "pca"])
    p.add_argument("--features",   default="",
                   help="Comma-separated list of feature columns to use for clustering. "
                        "Omit or leave empty to use all computed features (default). "
                        "Example: --features mom_21d,mom_63d,vol_21d,beta")
    return p.parse_args()


def main():
    args = parse_args()

    n_str = "auto" if args.n_clusters == 0 else str(args.n_clusters)
    print(f"\n{'='*60}")
    print(f"  Behavioural Classifier")
    print(f"  Index: {args.index}  |  Method: {args.method}  |  "
          f"N={n_str}  |  Viz: {args.viz}")
    print(f"{'='*60}\n")

    # 1. Constituents
    tickers_meta = get_index_tickers(args.index)
    if not tickers_meta:
        print("✗  No tickers fetched — cannot continue.")
        sys.exit(1)

    # Fix TSX tickers (TradingView returns "TSX:RY", strip prefix)
    for tm in tickers_meta:
        t = tm["ticker"]
        if ":" in t:
            tm["ticker"] = t.split(":")[-1]
        # TSX constituents need .TO suffix for yfinance
        if args.index == "TSX" and not tm["ticker"].endswith(".TO"):
            tm["ticker"] = tm["ticker"] + ".TO"

    meta = {tm["ticker"]: {"name": tm["name"], "sector": tm["sector"]}
            for tm in tickers_meta}
    tickers = list(meta.keys())
    print(f"  {len(tickers)} constituents found\n")

    # 2. Prices
    prices = fetch_prices(tickers, days=380)
    if prices.empty:
        print("✗  Price download returned empty DataFrame.")
        sys.exit(1)

    # 3. Features
    feat_df = compute_features(prices)
    if len(feat_df) < 10:
        print("✗  Too few tickers with valid features — cannot cluster.")
        sys.exit(1)

    # 3b. Fundamental features (Value / Growth lenses) — fetched only when requested
    #     Determine whether any fundamental features are needed before fetching.
    requested_set: set[str] | None = None
    if args.features:
        requested_set = {f.strip() for f in args.features.split(",") if f.strip()}

    need_fundamentals = (
        requested_set is None                          # all features → include fundamentals
        or bool(requested_set & FUNDAMENTAL_FEATURES)  # at least one fundamental requested
    )

    if need_fundamentals:
        fund_df = fetch_fundamentals(list(feat_df.index))
        if not fund_df.empty:
            # Align index then join; only keep rows present in feat_df
            fund_df = fund_df.reindex(feat_df.index)
            feat_df = feat_df.join(fund_df, how="left")
            # Fill any remaining NaN introduced by the join
            for col in fund_df.columns:
                if col in feat_df.columns:
                    fill_val = 0.0 if col == "div_yield" else feat_df[col].median()
                    feat_df[col] = feat_df[col].fillna(
                        fill_val if not (isinstance(fill_val, float) and np.isnan(fill_val)) else 0.0
                    )
            print(f"  Feature matrix expanded to {feat_df.shape[1]} columns "
                  f"(+{len(fund_df.columns)} fundamental)")

    # Optional feature selection — filter columns to the requested subset
    if args.features:
        requested = [f.strip() for f in args.features.split(",") if f.strip()]
        available = [f for f in requested if f in feat_df.columns]
        missing   = [f for f in requested if f not in feat_df.columns]
        if missing:
            print(f"  ⚠  Unknown feature(s) ignored: {missing}")
        if not available:
            print("✗  No valid features remain after filtering — cannot cluster.")
            sys.exit(1)
        if len(available) < len(feat_df.columns):
            print(f"  Feature selection: {len(available)} / {len(feat_df.columns)} features "
                  f"→ {available}")
        feat_df = feat_df[available]

    # Re-align prices to feat_df tickers
    prices = prices[[t for t in feat_df.index if t in prices.columns]]

    # 4. Cluster + embed
    if args.method == "dtw":
        embed_df, expl_var, labels, sil_scan, pca, feature_names, pca_scores = run_dtw_pipeline(
            prices, args.n_clusters, args.viz
        )
        # DTW result_df may have fewer tickers (only those with full window)
        # Re-align feat_df and prices to the tickers present in embed_df
        valid_tickers = embed_df.index.tolist()
        feat_df = feat_df.reindex(valid_tickers).dropna(how="all")
        prices  = prices[[t for t in valid_tickers if t in prices.columns]]
    else:
        embed_df, expl_var, labels, sil_scan, pca, feature_names, pca_scores = run_pipeline(
            feat_df, args.method, args.n_clusters, args.viz
        )

    # Warn loudly when DBSCAN produces a high noise ratio
    if args.method == "dbscan":
        n_noise_main  = int((labels == -1).sum())
        noise_ratio   = n_noise_main / max(len(labels), 1)
        if noise_ratio > 0.40:
            print(f"\n  ⚠  WARNING: DBSCAN classified {noise_ratio*100:.0f}% of tickers "
                  f"as noise ({n_noise_main}/{len(labels)}).")
            print("     S&P 500 behavioural data is a smooth continuum — DBSCAN cannot")
            print("     find the density voids it needs to form meaningful clusters.")
            print("     Consider switching to --method kmeans or --method hierarchical.\n")

    # 5. Build HTML
    print("  Building HTML report…")
    html = build_html(embed_df, feat_df, prices, meta, expl_var, labels, args,
                      sil_scan=sil_scan, pca=pca, feature_names=feature_names,
                      pca_scores=pca_scores)

    # 6. Save
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"classifier_{args.index}_{args.method}_{args.viz}_{ts}.html"
    out_path = os.path.join(OUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  ✓  Report saved → {out_path}")
    print(f"       Tickers: {len(feat_df)}")
    print(f"       Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    if -1 in labels:
        print(f"       Noise pts: {int((labels == -1).sum())}")
    print()


if __name__ == "__main__":
    main()
