"""
sentimentLead.py
Tests whether news sentiment leads or lags price returns for a given ticker.
Computes daily VADER sentiment from yfinance news headlines, correlates lagged
sentiment with forward returns, fits a predictive regression, and outputs a
signal quality dashboard as an HTML report.

Usage:
    python sentimentLead.py --ticker AAPL [--lookback 180] [--lags 10]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests as _req

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# VADER sentiment
# ---------------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _SIA = SentimentIntensityAnalyzer()
except ImportError:
    _SIA = None

# ---------------------------------------------------------------------------
# statsmodels
# ---------------------------------------------------------------------------
try:
    import statsmodels.api as sm
    _SM_AVAILABLE = True
except ImportError:
    _SM_AVAILABLE = False

# ---------------------------------------------------------------------------
# CSS palette
# ---------------------------------------------------------------------------
BG      = "#0f1117"
PANEL   = "#1a1e2e"
BORDER  = "#252a3a"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
ACCENT  = "#6366f1"
GREEN   = "#10b981"
RED     = "#ef4444"
YELLOW  = "#f59e0b"

# ===========================================================================
# News fetching
# ===========================================================================

def _fetch_yfinance_news(ticker: str) -> list[dict]:
    """Fetch news from yfinance .news attribute.
    Handles both the legacy flat schema and the newer nested 'content' schema.
    """
    items = []
    try:
        t = yf.Ticker(ticker)
        raw = t.news or []
        for item in raw:
            # ── New schema (yfinance >=0.2.50): item = {id, content: {...}} ──
            if "content" in item and isinstance(item["content"], dict):
                c = item["content"]
                title   = c.get("title", "")
                summary = c.get("summary", "") or c.get("description", "")
                ts_str  = c.get("pubDate") or c.get("displayTime", "")
                src     = (c.get("provider") or {}).get("displayName", "yfinance")
                try:
                    dt = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    continue
            else:
                # ── Legacy schema: flat dict with providerPublishTime ──────
                title   = item.get("title", "")
                summary = item.get("summary", "") or item.get("description", "")
                src     = item.get("publisher", "yfinance")
                ts = item.get("providerPublishTime") or item.get("published") or 0
                if isinstance(ts, int) and ts > 0:
                    dt = datetime.utcfromtimestamp(ts)
                elif isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(
                            ts.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        continue
                else:
                    continue

            text = f"{title} {summary}".strip()
            if text:
                items.append({"title": title, "text": text, "dt": dt,
                              "source": src})
    except Exception:
        pass
    return items


def _fetch_gnews_rss(ticker: str) -> list[dict]:
    """Fetch news via Google News RSS using feedparser — no API key needed."""
    try:
        import feedparser as _fp
    except ImportError:
        return []

    items = []
    # Two queries: ticker symbol alone, and with 'stock' for better relevance
    queries = [f"{ticker} stock", ticker]
    seen: set[str] = set()
    for q in queries:
        url = (f"https://news.google.com/rss/search"
               f"?q={_req.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en")
        try:
            feed = _fp.parse(url)
            for entry in (feed.entries or []):
                title = entry.get("title", "")
                # Google News appends ' - Publisher' to each title — strip it
                title = re.sub(r"\s+-\s+[^-]+$", "", title).strip()
                key = title.lower()
                if not title or key in seen:
                    continue
                seen.add(key)
                pt = entry.get("published_parsed")
                if not pt:
                    continue
                import time as _time
                dt = datetime(*pt[:6])
                items.append({"title": title, "text": title,
                              "dt": dt, "source": "gnews"})
        except Exception:
            continue
    return items


def collect_news(ticker: str, lookback: int) -> pd.DataFrame:
    """Collect, deduplicate, and return news as a DataFrame."""
    cutoff = datetime.utcnow() - timedelta(days=lookback + 5)

    all_items: list[dict] = []
    all_items.extend(_fetch_yfinance_news(ticker))
    all_items.extend(_fetch_gnews_rss(ticker))

    if not all_items:
        return pd.DataFrame(columns=["dt", "title", "text", "source"])

    df = pd.DataFrame(all_items)
    df = df.dropna(subset=["title", "dt"])
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 5]
    # Deduplicate by title (case-insensitive)
    df["_title_lower"] = df["title"].str.lower()
    df = df.drop_duplicates(subset=["_title_lower"]).drop(columns=["_title_lower"])
    # Filter by lookback window
    df = df[df["dt"] >= cutoff]
    df = df.sort_values("dt", ascending=True).reset_index(drop=True)
    return df


# ===========================================================================
# VADER scoring
# ===========================================================================

def score_news(df: pd.DataFrame) -> pd.DataFrame:
    """Add compound VADER score column to news DataFrame."""
    if _SIA is None:
        raise RuntimeError(
            "vaderSentiment not installed. Run: pip install vaderSentiment"
        )
    texts = df["text"].fillna(df["title"]).tolist()
    scores = [_SIA.polarity_scores(str(t))["compound"] for t in texts]
    df = df.copy()
    df["compound"] = scores
    return df


def daily_sentiment(news_df: pd.DataFrame, lookback: int) -> pd.Series:
    """Aggregate compound scores to daily average. Fill missing days with 0."""
    if news_df.empty:
        return pd.Series(dtype=float)

    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["dt"]).dt.normalize()
    daily = news_df.groupby("date")["compound"].mean()

    # Build full date range
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.Timedelta(days=lookback + 5)
    idx = pd.date_range(start, end, freq="D")
    daily = daily.reindex(idx, fill_value=0.0)
    return daily


# ===========================================================================
# Price returns
# ===========================================================================

def fetch_returns(ticker: str, lookback: int) -> pd.Series:
    """Download price history and compute daily log returns."""
    period_days = lookback + 45
    raw = yf.download(ticker, period=f"{period_days}d", interval="1d",
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No price data found for {ticker}.")

    # Flatten MultiIndex — handle both (metric, ticker) and (ticker, metric) layouts
    if isinstance(raw.columns, pd.MultiIndex):
        _expected = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}
        l0 = set(raw.columns.get_level_values(0))
        level = 0 if l0 & _expected else 1
        raw.columns = raw.columns.get_level_values(level)
    raw = raw.loc[:, ~raw.columns.duplicated(keep="first")]

    close = raw["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    log_ret = np.log(close / close.shift(1)).dropna()
    return log_ret


# ===========================================================================
# Lead-lag analysis
# ===========================================================================

def lead_lag_analysis(sentiment: pd.Series, returns: pd.Series,
                      max_lag: int) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations for lag k in [-max_lag, max_lag].
    Positive k  → sentiment leads returns (sentiment at t predicts return at t+k).
    Negative k  → sentiment lags returns (return at t+k was driven by return at t).
    """
    # Align on common dates (business days in returns)
    common = sentiment.index.intersection(returns.index)
    s = sentiment.loc[common]
    r = returns.loc[common]

    records = []
    for k in range(-max_lag, max_lag + 1):
        if k == 0:
            # k=0: contemporaneous
            paired = pd.DataFrame({"s": s, "r": r}).dropna()
        elif k > 0:
            # sentiment at t predicts return at t+k
            s_shifted = s.shift(-k)
            paired = pd.DataFrame({"s": s_shifted, "r": r}).dropna()
        else:
            # k<0: return at t+k was followed by sentiment at t (sentiment lags)
            s_shifted = s.shift(-k)
            paired = pd.DataFrame({"s": s_shifted, "r": r}).dropna()

        if len(paired) < 10:
            records.append({"lag": k, "pearson": np.nan, "spearman": np.nan,
                            "n": len(paired)})
            continue

        pearson = paired["s"].corr(paired["r"], method="pearson")
        spearman = paired["s"].corr(paired["r"], method="spearman")
        records.append({"lag": k, "pearson": pearson, "spearman": spearman,
                        "n": len(paired)})

    return pd.DataFrame(records).set_index("lag")


def find_best_lead(corr_df: pd.DataFrame) -> tuple[int, float]:
    """Return (lag, pearson_r) for the best positive-lag correlation."""
    pos = corr_df[corr_df.index > 0]["pearson"].dropna()
    if pos.empty:
        return 1, 0.0
    best_lag = int(pos.idxmax())
    best_r = float(pos.max())
    return best_lag, best_r


# ===========================================================================
# Walk-forward regression
# ===========================================================================

def walk_forward_regression(sentiment: pd.Series, returns: pd.Series,
                             lag: int) -> dict:
    """
    Train OLS on first 70%, evaluate on last 30%.
    Feature: sentiment(t - lag), Target: return(t).
    """
    result = {
        "beta": np.nan, "tstat": np.nan, "pvalue": np.nan,
        "r2_is": np.nan, "r2_oos": np.nan,
        "n_train": 0, "n_test": 0,
        "is_predictive": False,
        "note": ""
    }

    if not _SM_AVAILABLE:
        result["note"] = "statsmodels not available — regression skipped."
        return result

    common = sentiment.index.intersection(returns.index)
    s = sentiment.loc[common]
    r = returns.loc[common]

    # Lag the sentiment: feature at time t is sentiment from t-lag days ago
    s_lagged = s.shift(lag)
    df_reg = pd.DataFrame({"feat": s_lagged, "target": r}).dropna()

    if len(df_reg) < 20:
        result["note"] = "Insufficient overlapping data for regression."
        return result

    n = len(df_reg)
    n_train = int(n * 0.70)
    n_test = n - n_train

    train = df_reg.iloc[:n_train]
    test = df_reg.iloc[n_train:]

    result["n_train"] = n_train
    result["n_test"] = n_test

    X_train = sm.add_constant(train["feat"])
    y_train = train["target"]

    try:
        model = sm.OLS(y_train, X_train).fit()
        result["beta"]   = float(model.params.get("feat", np.nan))
        result["tstat"]  = float(model.tvalues.get("feat", np.nan))
        result["pvalue"] = float(model.pvalues.get("feat", np.nan))
        result["r2_is"]  = float(model.rsquared)

        # OOS evaluation
        X_test = sm.add_constant(test["feat"])
        y_test = test["target"]
        y_pred = model.predict(X_test)

        ss_res = float(((y_test - y_pred) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        result["r2_oos"] = float(oos_r2)

        if not np.isnan(oos_r2) and oos_r2 >= 0:
            result["is_predictive"] = True
        else:
            result["note"] = "Sentiment signal not predictive in out-of-sample period"

    except Exception as exc:
        result["note"] = f"Regression error: {exc}"

    return result


# ===========================================================================
# Signal quality badge
# ===========================================================================

def signal_badge(oos_r2: float, beta: float) -> tuple[str, str]:
    """Return (label, color) for signal quality badge."""
    if np.isnan(oos_r2):
        return "No Signal", MUTED
    if oos_r2 >= 0.02 and beta > 0:
        return "Strong Signal", GREEN
    if 0.0 <= oos_r2 < 0.02 and beta > 0:
        return "Weak Signal", YELLOW
    if oos_r2 >= 0 and beta < 0:
        return "Contrarian Signal", ACCENT
    return "No Signal", MUTED


# ===========================================================================
# HTML generation
# ===========================================================================

def _fmt(val, decimals=4, fallback="N/A"):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return fallback
    return f"{val:.{decimals}f}"


def _color_score(score: float) -> str:
    if score > 0.05:
        return GREEN
    if score < -0.05:
        return RED
    return MUTED


def build_html(
    ticker: str,
    lookback: int,
    news_df: pd.DataFrame,
    daily_sent: pd.Series,
    returns: pd.Series,
    corr_df: pd.DataFrame,
    best_lag: int,
    best_r: float,
    reg: dict,
    today_sentiment: float,
    has_few_news: bool,
) -> str:

    date_range_start = (datetime.utcnow() - timedelta(days=lookback)).strftime("%Y-%m-%d")
    date_range_end   = datetime.utcnow().strftime("%Y-%m-%d")
    n_news = len(news_df)
    badge_label, badge_color = signal_badge(reg["r2_oos"], reg.get("beta", np.nan))

    # ---- Chart data ----
    # Lead-lag bar chart
    lag_labels   = corr_df.index.tolist()
    lag_pearson  = [round(v, 6) if not math.isnan(v) else 0 for v in corr_df["pearson"].tolist()]
    lag_colors   = [f'"{GREEN}"' if k > 0 else f'"{MUTED}"' for k in lag_labels]

    # Sentiment vs price chart: last lookback trading days
    common_idx = daily_sent.index.intersection(returns.index)
    # Use last lookback calendar days
    start_plot = pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=lookback)
    plot_idx = common_idx[common_idx >= start_plot]

    sent_plot = daily_sent.loc[plot_idx]
    ret_plot  = returns.loc[plot_idx]

    # Reconstruct price from returns (index to 100)
    price_recon = (ret_plot.cumsum().apply(math.exp) * 100).round(4)

    # 7-day rolling avg sentiment
    sent_roll = sent_plot.rolling(7, min_periods=1).mean().round(6)

    chart_dates = [str(d.date()) for d in plot_idx]
    chart_price = price_recon.tolist()
    chart_sent  = sent_roll.tolist()

    # News timeline (last 30 items)
    news_tail = news_df.tail(30).iloc[::-1].reset_index(drop=True)

    def news_rows():
        rows = []
        for _, row in news_tail.iterrows():
            score = row.get("compound", 0.0)
            color = _color_score(score)
            dt_str = row["dt"].strftime("%Y-%m-%d %H:%M") if pd.notnull(row["dt"]) else ""
            src = str(row.get("source", ""))
            title = str(row.get("title", ""))
            rows.append(
                f'<tr>'
                f'<td style="color:{TEXT};padding:6px 10px;border-bottom:1px solid {BORDER};'
                f'max-width:520px;word-break:break-word;">{title}</td>'
                f'<td style="color:{MUTED};padding:6px 10px;border-bottom:1px solid {BORDER};'
                f'white-space:nowrap;">{dt_str}</td>'
                f'<td style="color:{color};padding:6px 10px;border-bottom:1px solid {BORDER};'
                f'text-align:right;font-weight:600;">{score:+.3f}</td>'
                f'<td style="color:{MUTED};padding:6px 10px;border-bottom:1px solid {BORDER};">{src}</td>'
                f'</tr>'
            )
        return "\n".join(rows)

    few_news_banner = ""
    if has_few_news:
        few_news_banner = (
            f'<div style="background:{YELLOW}22;border:1px solid {YELLOW};border-radius:8px;'
            f'padding:12px 18px;margin-bottom:20px;color:{YELLOW};font-size:13px;">'
            f'Warning: fewer than 30 news items found ({n_news} items). '
            f'Sentiment estimates may be less reliable.</div>'
        )

    oos_note = ""
    if reg.get("note") and not reg.get("is_predictive", False):
        oos_note = (
            f'<div style="color:{YELLOW};font-size:13px;margin-top:8px;">'
            f'{reg["note"]}</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment Lead — {ticker}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {BG}; color: {TEXT}; font-family: -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, sans-serif; font-size: 14px; line-height: 1.6;
    padding: 24px 32px;
  }}
  h1 {{ font-size: 26px; font-weight: 700; color: {TEXT}; }}
  h2 {{ font-size: 16px; font-weight: 600; color: {TEXT}; margin-bottom: 14px; }}
  .subtitle {{ color: {MUTED}; font-size: 13px; margin-top: 4px; margin-bottom: 28px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .card {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 20px 24px;
  }}
  .full {{ grid-column: 1 / -1; }}
  .metric-label {{ color: {MUTED}; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 4px; }}
  .metric-val {{ font-size: 28px; font-weight: 700; color: {TEXT}; }}
  .metric-sub {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}
  .badge {{
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 13px; font-weight: 600; margin-top: 6px;
  }}
  .stat-table {{ width: 100%; border-collapse: collapse; }}
  .stat-table th {{
    text-align: left; color: {MUTED}; font-size: 12px; font-weight: 500;
    padding: 6px 10px; border-bottom: 1px solid {BORDER};
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .stat-table td {{ padding: 8px 10px; border-bottom: 1px solid {BORDER}; color: {TEXT}; }}
  .stat-table tr:last-child td {{ border-bottom: none; }}
  .chart-wrap {{ position: relative; height: 280px; }}
  .news-scroll {{ max-height: 400px; overflow-y: auto; }}
  .news-scroll table {{ width: 100%; border-collapse: collapse; }}
  .section {{ margin-bottom: 20px; }}
  @media (max-width: 800px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Sentiment Lead Indicator &mdash; {ticker}</h1>
<div class="subtitle">
  Analysis period: {date_range_start} &rarr; {date_range_end} &nbsp;|&nbsp;
  {n_news} news items collected
</div>

{few_news_banner}

<!-- ===== Signal Quality Card ===== -->
<div class="grid2 section">
  <div class="card">
    <h2>Signal Quality</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
      <div>
        <div class="metric-label">Today&rsquo;s Sentiment</div>
        <div class="metric-val" style="color:{_color_score(today_sentiment)};">
          {today_sentiment:+.3f}
        </div>
        <div class="metric-sub">VADER compound (−1 to +1)</div>
      </div>
      <div>
        <div class="metric-label">Optimal Lead Lag</div>
        <div class="metric-val">{best_lag}d</div>
        <div class="metric-sub">Sentiment predicts returns N days ahead</div>
      </div>
      <div>
        <div class="metric-label">Pearson r @ lag</div>
        <div class="metric-val">{_fmt(best_r, 3)}</div>
        <div class="metric-sub">Cross-correlation at optimal lag</div>
      </div>
    </div>
    <div style="margin-top:16px;">
      <div class="metric-label">OOS R²</div>
      <div class="metric-val">{_fmt(reg["r2_oos"], 4)}</div>
    </div>
    <div>
      <span class="badge" style="background:{badge_color}22;color:{badge_color};
        border:1px solid {badge_color};">{badge_label}</span>
    </div>
    {oos_note}
  </div>

  <!-- ===== Regression Stats Card ===== -->
  <div class="card">
    <h2>Regression Statistics (OLS)</h2>
    <table class="stat-table">
      <thead>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
        <tr><td>&beta; coefficient</td>
            <td style="color:{TEXT};font-weight:600;">{_fmt(reg["beta"], 6)}</td></tr>
        <tr><td>t-statistic</td>
            <td style="color:{TEXT};">{_fmt(reg["tstat"], 3)}</td></tr>
        <tr><td>p-value</td>
            <td style="color:{'#10b981' if (not math.isnan(reg['pvalue'])) and reg['pvalue'] < 0.05 else MUTED};">
              {_fmt(reg["pvalue"], 4)}</td></tr>
        <tr><td>In-sample R²</td>
            <td>{_fmt(reg["r2_is"], 4)}</td></tr>
        <tr><td>OOS R²</td>
            <td style="color:{'#10b981' if (not math.isnan(reg['r2_oos'])) and reg['r2_oos'] > 0 else RED};">
              {_fmt(reg["r2_oos"], 4)}</td></tr>
        <tr><td>Train / Test obs</td>
            <td>{reg["n_train"]} / {reg["n_test"]}</td></tr>
        <tr><td>Feature lag used</td>
            <td>{best_lag} day(s)</td></tr>
      </tbody>
    </table>
    {'<div style="color:' + MUTED + ';font-size:12px;margin-top:10px;">' + reg["note"] + '</div>'
      if reg.get("note") else ""}
  </div>
</div>

<!-- ===== Lead-Lag Correlation Chart ===== -->
<div class="card section">
  <h2>Lead-Lag Correlation (Pearson)</h2>
  <div style="color:{MUTED};font-size:12px;margin-bottom:12px;">
    Green bars (positive lag): sentiment predicts future returns.
    Gray bars (negative lag): sentiment follows past returns.
  </div>
  <div class="chart-wrap">
    <canvas id="lagChart"></canvas>
  </div>
</div>

<!-- ===== Sentiment vs Price Chart ===== -->
<div class="card section">
  <h2>Sentiment vs Price (last {lookback} days)</h2>
  <div style="color:{MUTED};font-size:12px;margin-bottom:12px;">
    Blue line: reconstructed price index (left axis).
    Purple line: 7-day rolling average VADER sentiment (right axis).
  </div>
  <div class="chart-wrap">
    <canvas id="sentPriceChart"></canvas>
  </div>
</div>

<!-- ===== News Timeline ===== -->
<div class="card section">
  <h2>News Sentiment Timeline (last 30 items)</h2>
  <div class="news-scroll">
    <table>
      <thead>
        <tr>
          <th style="color:{MUTED};font-size:12px;padding:6px 10px;border-bottom:1px solid {BORDER};
            text-align:left;">Headline</th>
          <th style="color:{MUTED};font-size:12px;padding:6px 10px;border-bottom:1px solid {BORDER};
            text-align:left;">Date</th>
          <th style="color:{MUTED};font-size:12px;padding:6px 10px;border-bottom:1px solid {BORDER};
            text-align:right;">Score</th>
          <th style="color:{MUTED};font-size:12px;padding:6px 10px;border-bottom:1px solid {BORDER};
            text-align:left;">Source</th>
        </tr>
      </thead>
      <tbody>
        {news_rows()}
      </tbody>
    </table>
  </div>
</div>

<div style="color:{MUTED};font-size:11px;margin-top:12px;">
  Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} &nbsp;|&nbsp;
  Sentiment: VADER &nbsp;|&nbsp; Price data: yfinance &nbsp;|&nbsp;
  Regression: OLS walk-forward (70/30 split)
</div>

<script>
// ---- Lead-Lag Chart ----
(function() {{
  const lagLabels = {lag_labels};
  const lagPearson = {lag_pearson};
  const lagColors = [{', '.join(lag_colors)}];

  const ctx = document.getElementById('lagChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: lagLabels,
      datasets: [{{
        label: 'Pearson r',
        data: lagPearson,
        backgroundColor: lagColors,
        borderColor: lagColors,
        borderWidth: 1,
        borderRadius: 3,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            title: (items) => `Lag: ${{items[0].label}}d`,
            label: (item) => `Pearson r = ${{item.raw.toFixed(4)}}`
          }}
        }},
        annotation: {{}}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'Lag (days)', color: '{MUTED}', font: {{ size: 12 }} }},
          grid: {{ color: '{BORDER}' }},
          ticks: {{ color: '{MUTED}' }}
        }},
        y: {{
          title: {{ display: true, text: 'Pearson Correlation', color: '{MUTED}', font: {{ size: 12 }} }},
          grid: {{ color: '{BORDER}' }},
          ticks: {{ color: '{MUTED}' }},
          suggestedMin: -0.3,
          suggestedMax: 0.3,
        }}
      }}
    }},
    plugins: [{{
      id: 'zeroline',
      afterDraw(chart) {{
        const {{ctx, scales: {{y}}}} = chart;
        const yZero = y.getPixelForValue(0);
        ctx.save();
        ctx.strokeStyle = '{TEXT}';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(chart.chartArea.left, yZero);
        ctx.lineTo(chart.chartArea.right, yZero);
        ctx.stroke();
        ctx.restore();
      }}
    }}]
  }});
}})();

// ---- Sentiment vs Price Chart ----
(function() {{
  const dates = {chart_dates};
  const prices = {chart_price};
  const sentiments = {chart_sent};

  const ctx = document.getElementById('sentPriceChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: dates,
      datasets: [
        {{
          label: 'Price Index',
          data: prices,
          borderColor: '#60a5fa',
          backgroundColor: 'transparent',
          yAxisID: 'yPrice',
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
        }},
        {{
          label: '7d Avg Sentiment',
          data: sentiments,
          borderColor: '{ACCENT}',
          backgroundColor: '{ACCENT}22',
          yAxisID: 'ySent',
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
          fill: true,
        }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          labels: {{ color: '{TEXT}', font: {{ size: 12 }} }}
        }},
        tooltip: {{ mode: 'index', intersect: false }}
      }},
      scales: {{
        x: {{
          grid: {{ color: '{BORDER}' }},
          ticks: {{
            color: '{MUTED}',
            maxTicksLimit: 12,
            maxRotation: 45,
          }}
        }},
        yPrice: {{
          type: 'linear',
          position: 'left',
          title: {{ display: true, text: 'Price Index', color: '#60a5fa', font: {{ size: 12 }} }},
          grid: {{ color: '{BORDER}' }},
          ticks: {{ color: '#60a5fa' }},
        }},
        ySent: {{
          type: 'linear',
          position: 'right',
          title: {{ display: true, text: 'Sentiment (7d avg)', color: '{ACCENT}', font: {{ size: 12 }} }},
          grid: {{ drawOnChartArea: false }},
          ticks: {{ color: '{ACCENT}' }},
          suggestedMin: -0.5,
          suggestedMax: 0.5,
        }}
      }}
    }}
  }});
}})();
</script>

</body>
</html>"""
    return html


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Lead Indicator — tests whether news sentiment leads price returns."
    )
    parser.add_argument("--ticker",   required=True, type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--lookback", default=180,   type=int, help="Days of history (default: 180)")
    parser.add_argument("--lags",     default=10,    type=int, help="Max lag to test in days (default: 10)")
    args = parser.parse_args()

    ticker   = args.ticker.upper().strip()
    lookback = max(30, args.lookback)
    lags     = max(1, args.lags)

    print(f"\n=== Sentiment Lead Indicator === {ticker} ===", flush=True)

    # ---- Output path ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "sentimentLeadData")
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    out_filename = f"sentiment_lead_{ticker}_{date_str}.html"
    out_path = os.path.join(out_dir, out_filename)

    # ---- News ----
    print(f"  Fetching news headlines ({lookback}d lookback)...", flush=True)
    news_df = collect_news(ticker, lookback)
    n_items = len(news_df)
    print(f"  {n_items} news items found", flush=True)
    has_few_news = n_items < 30

    if n_items == 0:
        print(f"  No news items found for {ticker}. Check ticker symbol.", flush=True)
        sys.exit(1)

    # ---- VADER ----
    print(f"  Computing daily VADER sentiment...", flush=True)
    if _SIA is None:
        print("  ERROR: vaderSentiment not installed. Run: pip install vaderSentiment", flush=True)
        sys.exit(1)

    news_df = score_news(news_df)
    daily_sent = daily_sentiment(news_df, lookback)
    n_days = int((daily_sent != 0).sum())
    print(f"  {n_days} days with sentiment data", flush=True)

    # ---- Price returns ----
    try:
        returns = fetch_returns(ticker, lookback)
    except ValueError as e:
        print(f"  ERROR: {e}", flush=True)
        sys.exit(1)

    # ---- Overlap check ----
    common = daily_sent.index.intersection(returns.index)
    if len(common) < 20:
        print(
            f"  ERROR: Only {len(common)} overlapping days between sentiment and returns. "
            f"Need at least 20. Try a larger --lookback value or check ticker.",
            flush=True,
        )
        sys.exit(1)

    # ---- Lead-lag ----
    print(f"  Running lead-lag analysis (±{lags} days)...", flush=True)
    corr_df = lead_lag_analysis(daily_sent, returns, lags)
    best_lag, best_r = find_best_lead(corr_df)
    print(f"  Optimal lead: {best_lag}d  Pearson r={best_r:.3f}", flush=True)

    # ---- Regression ----
    reg = walk_forward_regression(daily_sent, returns, best_lag)
    oos_r2 = reg["r2_oos"]
    print(f"  Walk-forward regression: OOS R²={_fmt(oos_r2, 3)}", flush=True)

    # ---- Today's sentiment ----
    today_key = daily_sent.index[-1] if not daily_sent.empty else None
    today_sentiment = float(daily_sent.iloc[-1]) if today_key is not None else 0.0

    # ---- HTML ----
    print(f"  Generating report...", flush=True)
    html = build_html(
        ticker=ticker,
        lookback=lookback,
        news_df=news_df,
        daily_sent=daily_sent,
        returns=returns,
        corr_df=corr_df,
        best_lag=best_lag,
        best_r=best_r,
        reg=reg,
        today_sentiment=today_sentiment,
        has_few_news=has_few_news,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✓  Report saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
