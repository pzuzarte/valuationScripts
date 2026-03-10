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
      - Beta to SPY (252d rolling, last value)
      - Max drawdown over 252d
      - Return skewness over 126d
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


# ── Clustering pipeline ────────────────────────────────────────────────────────

K_SCAN_MIN, K_SCAN_MAX = 2, 15   # k search range for auto-selection
# Minimum cluster size for k-means silhouette scan.
# Prevents picking a degenerate split where 1-2 extreme outliers form their
# own cluster — which produces a spuriously high silhouette score.
K_MIN_CLUSTER_FRAC = 0.02        # 2 % of n  (≥10 stocks for SPX)


def _auto_k_silhouette(Xp: np.ndarray) -> tuple[int, dict[int, float]]:
    """K-Means k-selection via silhouette scan with balance filter.

    Returns (best_k, {k: silhouette_score}).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = Xp.shape[0]
    k_max     = min(K_SCAN_MAX, n_samples - 1)
    min_size  = max(2, int(K_MIN_CLUSTER_FRAC * n_samples))
    scores: dict[int, float] = {}
    skipped: list[int] = []

    print(f"  Auto-selecting k ({K_SCAN_MIN}–{k_max}) via silhouette score "
          f"(min cluster size = {min_size})…", flush=True)

    for k in range(K_SCAN_MIN, k_max + 1):
        lbl    = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xp)
        counts = np.bincount(lbl.astype(int))
        if counts.min() < min_size:
            skipped.append(k)
            continue
        scores[k] = float(silhouette_score(Xp, lbl))

    if skipped:
        print(f"  Skipped degenerate k values (cluster < {min_size} stocks): {skipped}")

    if not scores:
        print("  ⚠  All k values were degenerate; falling back to unconstrained scan.")
        for k in range(K_SCAN_MIN, k_max + 1):
            lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xp)
            scores[k] = float(silhouette_score(Xp, lbl))

    best_k = max(scores, key=scores.get)
    print(f"  → Best k = {best_k}  (silhouette = {scores[best_k]:.3f})")
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
                 viz: str) -> tuple[pd.DataFrame, list[float], np.ndarray, dict[int, float]]:
    """
    StandardScaler → PCA (12 components) → clustering → 2-D embedding.

    n_clusters = 0  triggers automatic k selection via silhouette scan
                    (K-Means and Hierarchical only; DBSCAN is always auto).

    Returns
    -------
    result_df : DataFrame with columns [x, y, cluster]  (index = ticker)
    expl_var  : explained variance ratio per PCA component
    labels    : cluster label per sample (int array)
    sil_scan  : {k: silhouette_score} dict from the auto-scan, or {} if
                n_clusters was supplied manually or method == 'dbscan'
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
    return result_df, expl_var, labels, sil_scan


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
               sil_scan: dict[int, float] | None = None) -> str:
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

    df["vol_21d_pct"] = feat_df["vol_21d"].mul(100).reindex(df.index)
    df["beta"]        = feat_df["beta"].reindex(df.index)
    df["max_dd"]      = feat_df["max_dd_252d"].mul(100).reindex(df.index)

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
            chart_title  = "Silhouette Score by k — Auto-selection"
            y_axis_label = "Avg silhouette score"
            hover_tmpl   = "k = %{x}<br>Silhouette = %{y:.4f}<extra></extra>"
            score_label  = f"silhouette = {sil_scan[best_k]:.3f}"

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
    method_map  = {"kmeans": "K-Means", "hierarchical": "Hierarchical (Ward)", "dbscan": "DBSCAN"}
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
        <input type="text" id="plot-ticker-search" placeholder="e.g. NVDA"
               oninput="onPlotSearch(event)" autocomplete="off" spellcheck="false" maxlength="12">
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

function _searchScatter(query) {{
  var el = document.getElementById('scatter');
  if (!el || !el.data) return {{found: false}};
  _cacheScatter();
  var q = query.toUpperCase();
  var foundT = -1, foundP = -1, foundTicker = "";

  // Three passes: exact → prefix → substring, across all cluster traces
  var passes = [
    function(s) {{ return s === q; }},
    function(s) {{ return s.startsWith(q); }},
    function(s) {{ return s.includes(q); }},
  ];
  outer: for (var pass = 0; pass < passes.length; pass++) {{
    for (var t = 0; t < el.data.length; t++) {{
      var texts = el.data[t].text;
      if (!texts) continue;
      for (var p = 0; p < texts.length; p++) {{
        if (passes[pass](texts[p].toUpperCase())) {{
          foundT = t; foundP = p; foundTicker = texts[p];
          break outer;
        }}
      }}
    }}
  }}

  if (foundT === -1) {{ _restoreScatter(); return {{found: false}}; }}

  // Dim every trace uniformly (no trace arg = all traces)
  Plotly.restyle(el, {{'marker.opacity': 0.07}});

  // Highlight the single matched point in its trace
  var o      = _origScatter[foundT];
  var n      = el.data[foundT].text.length;
  var oCol   = o.color;
  var oSize  = o.size;
  var oCols  = Array.isArray(oCol) ? oCol : Array(n).fill(oCol);
  var newColors  = oCols.slice();   newColors[foundP]  = '#FFD700';
  var newSizes   = Array(n).fill(oSize);  newSizes[foundP]   = oSize * 2.5;
  var newOpacity = Array(n).fill(0.07);   newOpacity[foundP] = 1.0;

  Plotly.restyle(el, {{
    'marker.color':   [newColors],
    'marker.size':    [newSizes],
    'marker.opacity': [newOpacity],
  }}, [foundT]);

  return {{found: true, ticker: foundTicker}};
}}

function _restoreHeatmap() {{
  var el = document.getElementById('corr-heat');
  if (el) Plotly.relayout(el, {{shapes: []}});
}}

function _searchHeatmap(query) {{
  var el = document.getElementById('corr-heat');
  if (!el || !CORR_TICKERS || !CORR_TICKERS.length) return;
  var q = query.toUpperCase();
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
  if (idx === -1) {{ _restoreHeatmap(); return; }}

  // Gold column + row bands via layout shapes
  Plotly.relayout(el, {{
    shapes: [
      {{ // vertical column band
        type: 'rect', layer: 'above',
        xref: 'x', yref: 'paper',
        x0: idx - 0.5, x1: idx + 0.5, y0: 0, y1: 1,
        fillcolor: 'rgba(255,215,0,0.13)',
        line: {{color: '#FFD700', width: 1.5}},
      }},
      {{ // horizontal row band
        type: 'rect', layer: 'above',
        xref: 'paper', yref: 'y',
        x0: 0, x1: 1, y0: idx - 0.5, y1: idx + 0.5,
        fillcolor: 'rgba(255,215,0,0.13)',
        line: {{color: '#FFD700', width: 1.5}},
      }}
    ]
  }});
}}

function onPlotSearch(e) {{
  var query    = e.target.value.trim();
  _lastPlotQuery = query;
  var statusEl = document.getElementById('plot-search-status');
  if (!query) {{
    _restoreScatter();
    _restoreHeatmap();
    statusEl.textContent = "";
    return;
  }}
  var res = _searchScatter(query);
  _searchHeatmap(query);
  if (res.found) {{
    statusEl.textContent = res.ticker;
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
                   choices=["kmeans", "hierarchical", "dbscan"])
    p.add_argument("--n_clusters", default=0, type=int,
                   help="Number of clusters (0 = auto via silhouette scan; "
                        "ignored for DBSCAN)")
    p.add_argument("--viz",        default="tsne",
                   choices=["tsne", "umap", "pca"])
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

    # Re-align prices to feat_df tickers
    prices = prices[[t for t in feat_df.index if t in prices.columns]]

    # 4. Cluster + embed
    embed_df, expl_var, labels, sil_scan = run_pipeline(
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
                      sil_scan=sil_scan)

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
