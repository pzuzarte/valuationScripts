"""
sentimentLead.py
Tests whether news sentiment leads or lags price returns for a given ticker.
Computes daily VADER sentiment from yfinance news headlines, correlates lagged
sentiment with forward returns, fits a predictive regression, and outputs a
signal quality dashboard as an HTML report.

Usage:
    python sentimentLead.py --ticker AAPL [--lookback 30] [--lags 10]
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
# Load .env from project root (if present) — never commit .env to git
# ---------------------------------------------------------------------------
def _load_env() -> None:
    """Parse KEY=value lines from .env in the project root directory."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val

_load_env()

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


def _fetch_finnhub_news(ticker: str, lookback: int, api_key: str) -> list[dict]:
    """
    Fetch historical company news from Finnhub.
    Free tier: 60 calls/min, ~1 year of history, true per-ticker filtering.
    Sign up free at https://finnhub.io — no credit card required.
    """
    items = []
    try:
        end_dt   = datetime.utcnow()
        start_dt = end_dt - timedelta(days=lookback + 5)
        url = (
            "https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}"
            f"&from={start_dt.strftime('%Y-%m-%d')}"
            f"&to={end_dt.strftime('%Y-%m-%d')}"
            f"&token={api_key}"
        )
        r = _req.get(url, timeout=10)
        if r.status_code != 200:
            return items
        for article in (r.json() or []):
            ts = article.get("datetime", 0)
            if not ts:
                continue
            dt      = datetime.utcfromtimestamp(int(ts))
            headline = article.get("headline", "")
            summary  = article.get("summary", "")
            text     = f"{headline} {summary}".strip()
            if not text:
                continue
            items.append({
                "title":  headline,
                "text":   text,
                "dt":     dt,
                "source": article.get("source", "finnhub"),
            })
    except Exception:
        pass
    return items


def collect_news(ticker: str, lookback: int,
                 finnhub_key: str = "") -> pd.DataFrame:
    """
    Collect, deduplicate, and return news as a DataFrame.
    If a Finnhub API key is provided, it is used as the primary source
    (genuine historical data up to 1 year back).  Falls back to yfinance
    + Google News RSS when no key is given (limited to ~2 weeks of recency).
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback + 5)

    all_items: list[dict] = []
    if finnhub_key:
        fh_items = _fetch_finnhub_news(ticker, lookback, finnhub_key)
        all_items.extend(fh_items)
        if fh_items:
            print(f"  Finnhub: {len(fh_items)} articles fetched", flush=True)
    if not all_items:
        # Free fallback — only covers ~2 weeks regardless of lookback setting
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
    """Aggregate compound scores to daily average.
    Only returns days that actually have news — do NOT fill gaps with 0.
    Filling with 0 adds fake 'neutral' signal on no-news days, diluting
    correlations and causing the regression to predict the base rate.
    """
    if news_df.empty:
        return pd.Series(dtype=float)

    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["dt"]).dt.normalize()
    daily = news_df.groupby("date")["compound"].mean()
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    # Keep only dates within the lookback window
    cutoff = pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=lookback + 5)
    daily = daily[daily.index >= cutoff]
    return daily.sort_index()


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
    Compute Spearman correlations for lag k in [-max_lag, max_lag].
    Operates only on news days (where sentiment is non-zero / actually observed).
    Positive k → sentiment at t predicts return at t+k (sentiment leads).
    Negative k → return moved first, sentiment followed (sentiment lags).
    Uses Spearman (rank) correlation — more robust with small N and non-normal data.
    """
    # Build a returns lookup for any date offset
    ret_idx = returns.index

    records = []
    for k in range(-max_lag, max_lag + 1):
        pairs_s, pairs_r = [], []
        for dt, s_val in sentiment.items():
            target_dt = dt + pd.Timedelta(days=k)
            # Find the nearest trading day within ±2 calendar days
            candidates = ret_idx[np.abs((ret_idx - target_dt).days) <= 2]
            if len(candidates) == 0:
                continue
            nearest = candidates[np.abs((candidates - target_dt).days).argmin()]
            pairs_s.append(s_val)
            pairs_r.append(float(returns.loc[nearest]))

        n = len(pairs_s)
        if n < 5:
            records.append({"lag": k, "pearson": np.nan, "spearman": np.nan, "n": n})
            continue

        arr_s = np.array(pairs_s)
        arr_r = np.array(pairs_r)
        pearson  = float(pd.Series(arr_s).corr(pd.Series(arr_r), method="pearson"))
        spearman = float(pd.Series(arr_s).corr(pd.Series(arr_r), method="spearman"))
        records.append({"lag": k, "pearson": pearson, "spearman": spearman, "n": n})

    return pd.DataFrame(records).set_index("lag")


def find_best_lead(corr_df: pd.DataFrame) -> tuple[int, float]:
    """Return (lag, spearman_r) for the best positive-lag correlation."""
    pos = corr_df[corr_df.index > 0]["spearman"].dropna()
    if pos.empty:
        return 1, 0.0
    best_lag = int(pos.idxmax())
    best_r   = float(pos.max())
    return best_lag, best_r


def event_study(sentiment: pd.Series, returns: pd.Series,
                horizons: list[int] | None = None) -> pd.DataFrame:
    """
    For each news day, bucket by sentiment sign and compute average
    cumulative return over forward horizons (1, 3, 5, 10 days).
    Returns DataFrame with columns: horizon, pos_ret, neg_ret, diff.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]
    ret_idx = returns.index
    pos_days = sentiment[sentiment >  0.05]
    neg_days = sentiment[sentiment < -0.05]

    def _cum_ret(days_series, horizon):
        vals = []
        for dt in days_series.index:
            future = dt + pd.Timedelta(days=horizon)
            cands  = ret_idx[(ret_idx > dt) & (ret_idx <= future + pd.Timedelta(days=3))]
            if len(cands) == 0:
                continue
            r_slice = returns.loc[cands[cands <= future + pd.Timedelta(days=3)]]
            cum = float(r_slice.sum()) * 100
            vals.append(cum)
        return np.nanmean(vals) if vals else np.nan

    rows = []
    for h in horizons:
        rows.append({
            "horizon":  h,
            "pos_ret":  _cum_ret(pos_days,  h),
            "neg_ret":  _cum_ret(neg_days,  h),
            "n_pos":    len(pos_days),
            "n_neg":    len(neg_days),
        })
    df = pd.DataFrame(rows)
    df["diff"] = df["pos_ret"] - df["neg_ret"]
    return df


# ===========================================================================
# Walk-forward regression (news-days only)
# ===========================================================================

def walk_forward_regression(sentiment: pd.Series, returns: pd.Series,
                             lag: int) -> dict:
    """
    OLS regression using ONLY news days (non-zero sentiment).
    Feature: sentiment(t), Target: cumulative return over next `lag` trading days.
    This avoids the sparse-zero-filling problem that causes the model to predict
    the unconditional mean for every ticker.
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

    ret_idx = returns.index

    # Build (sentiment, forward_return) pairs — only on news days
    pairs = []
    for dt, s_val in sentiment.items():
        future = dt + pd.Timedelta(days=lag)
        cands  = ret_idx[(ret_idx > dt) & (ret_idx <= future + pd.Timedelta(days=3))]
        if len(cands) == 0:
            continue
        r_slice = returns.loc[cands[cands <= future + pd.Timedelta(days=3)]]
        cum_ret = float(r_slice.sum())
        pairs.append({"feat": s_val, "target": cum_ret, "dt": dt})

    if len(pairs) < 8:
        result["note"] = f"Only {len(pairs)} news days — insufficient for regression."
        return result

    df_reg = pd.DataFrame(pairs).sort_values("dt").reset_index(drop=True)
    n       = len(df_reg)
    n_train = max(int(n * 0.70), n - 5)   # keep at least 5 OOS points
    n_test  = n - n_train

    if n_test < 3:
        result["note"] = "Too few out-of-sample points — showing in-sample only."
        n_train = n
        n_test  = 0

    train = df_reg.iloc[:n_train]
    test  = df_reg.iloc[n_train:]
    result["n_train"] = n_train
    result["n_test"]  = n_test

    try:
        X_train = sm.add_constant(train["feat"])
        model   = sm.OLS(train["target"], X_train).fit()
        result["beta"]   = float(model.params.get("feat", np.nan))
        result["tstat"]  = float(model.tvalues.get("feat", np.nan))
        result["pvalue"] = float(model.pvalues.get("feat", np.nan))
        result["r2_is"]  = float(model.rsquared)

        if n_test >= 3:
            X_test = sm.add_constant(test["feat"])
            y_pred = model.predict(X_test)
            y_test = test["target"]
            ss_res = float(((y_test - y_pred) ** 2).sum())
            ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
            oos_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            result["r2_oos"]       = float(oos_r2)
            result["is_predictive"] = bool(np.isfinite(oos_r2) and oos_r2 >= 0)
            if not result["is_predictive"]:
                result["note"] = "Sentiment not predictive in OOS period."
        else:
            result["r2_oos"]        = np.nan
            result["is_predictive"] = result["r2_is"] > 0.02 and result["pvalue"] < 0.10

    except Exception as exc:
        result["note"] = f"Regression error: {exc}"

    return result


# ===========================================================================
# Signal quality badge
# ===========================================================================

def signal_badge(best_r: float, beta: float, oos_r2: float) -> tuple[str, str]:
    """
    Return (label, color).  Primary signal: Spearman r at best lead lag.
    OOS R² is secondary — only available when N is large enough for a split.
    Thresholds:
      |r| ≥ 0.35 → meaningful correlation for sentiment data
      |r| ≥ 0.20 → weak but present
    """
    if np.isnan(best_r):
        return "No Signal", MUTED
    # OOS R² is the gold standard when available
    if np.isfinite(oos_r2):
        if oos_r2 >= 0.03 and beta > 0:
            return "Strong Signal ✓", GREEN
        if oos_r2 >= 0.01 and beta > 0:
            return "Weak Signal", YELLOW
        if oos_r2 >= 0 and beta < 0:
            return "Contrarian Signal", ACCENT
    # Fall back to Spearman r at best lead lag
    if best_r >= 0.35:
        return "Strong Signal ✓", GREEN
    if best_r >= 0.20:
        return "Weak Signal", YELLOW
    if best_r <= -0.20:
        return "Contrarian Signal", ACCENT
    return "No Signal", MUTED


# ===========================================================================
# Plain-language interpretation
# ===========================================================================

def build_interpretation(
    ticker: str,
    best_lag: int,
    best_r: float,
    reg: dict,
    badge_label: str,
    today_sentiment: float,
    n_days: int,
    lookback: int,
) -> str:
    """
    Generate a plain-English paragraph explaining what the numbers mean,
    including the apparent contradiction between a positive Spearman r
    and a negative OOS R².
    """
    lines: list[str] = []

    # ── Lead-lag finding ─────────────────────────────────────────────────────
    r_str    = f"{best_r:.2f}"
    lag_str  = f"{best_lag} trading day{'s' if best_lag != 1 else ''}"
    r_word   = "strong" if best_r >= 0.40 else ("moderate" if best_r >= 0.25 else "weak")
    lines.append(
        f"<b>Lead-lag relationship.</b> The analysis found a {r_word} cross-correlation "
        f"(Spearman r&nbsp;=&nbsp;{r_str}) between {ticker} news sentiment and price returns "
        f"at a <b>{lag_str} forward lag</b>. This means that on the news days captured in "
        f"this window, sentiment tended to point in the same direction as the stock's move "
        f"{lag_str} later."
    )

    # ── Explain Strong Signal vs negative OOS R² ─────────────────────────────
    oos_r2   = reg.get("r2_oos", float("nan"))
    n_train  = reg.get("n_train", 0)
    n_test   = reg.get("n_test",  0)
    is_r2    = reg.get("r2_is",   float("nan"))

    if "Strong Signal" in badge_label or "Weak Signal" in badge_label:
        if math.isfinite(oos_r2) and oos_r2 < 0:
            lines.append(
                f"<b>Why does the badge say '{badge_label}' but OOS R² is negative?</b> "
                f"These two metrics measure different things. The <i>Spearman correlation</i> "
                f"asks: &ldquo;across all {n_days} news days, does higher sentiment tend to precede "
                f"higher returns {lag_str} later?&rdquo; It answers yes (r&nbsp;=&nbsp;{r_str}). "
                f"The <i>OOS R&sup2;</i> asks something stricter: &ldquo;if we train a regression on the "
                f"first {n_train} news days and predict the last {n_test}, does the model beat "
                f"simply guessing the average?&rdquo; With only {n_test} test-set observations, a "
                f"single outlier day can swing OOS R² from positive to strongly negative — "
                f"it is statistically unreliable at this sample size. "
                f"The correlation ({r_str}) is the more trustworthy signal here; "
                f"OOS R² becomes meaningful with 50+ news days."
            )
        elif math.isfinite(oos_r2) and oos_r2 >= 0:
            lines.append(
                f"<b>Signal consistency.</b> The positive OOS R² ({oos_r2:.3f}) confirms "
                f"that the relationship held in the held-out test period (last {n_test} news days), "
                f"not just in the training window. This increases confidence that the "
                f"{lag_str} lead is genuine rather than a chance correlation."
            )

    # ── Today's sentiment implication ────────────────────────────────────────
    if math.isfinite(today_sentiment):
        sent_word = (
            "strongly positive" if today_sentiment >  0.30 else
            "mildly positive"   if today_sentiment >  0.05 else
            "neutral"           if today_sentiment >= -0.05 else
            "mildly negative"   if today_sentiment >= -0.30 else
            "strongly negative"
        )
        direction = "upward" if today_sentiment > 0.05 else ("downward" if today_sentiment < -0.05 else "flat")
        if "Signal" in badge_label and "No Signal" not in badge_label:
            lines.append(
                f"<b>Today's signal.</b> Today's sentiment for {ticker} is "
                f"<b>{sent_word} ({today_sentiment:+.3f})</b>. If the historical "
                f"{lag_str} lead holds, this is consistent with <b>{direction} price pressure "
                f"over the next {lag_str}</b>. Treat this as one probabilistic input — "
                f"not a forecast."
            )
        else:
            lines.append(
                f"<b>Today's sentiment</b> is {sent_word} ({today_sentiment:+.3f}), "
                f"but the overall signal quality is insufficient to draw directional conclusions."
            )

    # ── Data coverage caveat ─────────────────────────────────────────────────
    lines.append(
        f"<b>Data coverage.</b> This analysis used <b>{n_days} news days</b> from a "
        f"{lookback}-day lookback window. "
        + (
            "News data sourced from <b>Finnhub</b> — up to 1 year of genuine historical "
            "per-ticker coverage on the free tier."
            if n_days >= 20 and lookback >= 60
            else
            "Free news sources (Google News RSS, yfinance) carry roughly 2–3 weeks of "
            "recency regardless of the lookback setting. For stable signals, pass "
            "<code>--finnhub-key &lt;key&gt;</code> (free at finnhub.io) to unlock "
            "up to 1 year of historical data."
        )
    )

    # Wrap each paragraph
    paras = "".join(f'<p style="margin:0 0 12px 0">{l}</p>' for l in lines)
    return paras


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
    n_news  = len(news_df)
    n_days  = len(daily_sent)
    badge_label, badge_color = signal_badge(best_r, reg.get("beta", np.nan), reg.get("r2_oos", np.nan))

    interpretation_html = build_interpretation(
        ticker=ticker, best_lag=best_lag, best_r=best_r, reg=reg,
        badge_label=badge_label, today_sentiment=today_sentiment,
        n_days=n_days, lookback=lookback,
    )

    # ---- Chart data ----
    # Lead-lag bar chart
    lag_labels   = corr_df.index.tolist()
    lag_pearson  = [round(v, 6) if not math.isnan(v) else 0 for v in corr_df["pearson"].tolist()]
    lag_colors   = [f'"{GREEN}"' if k > 0 else f'"{MUTED}"' for k in lag_labels]

    # Sentiment vs price chart
    # Price: all trading days in lookback window (continuous line)
    # Sentiment: sparse — only actual news days; null on all other days
    start_plot  = pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=lookback + 5)
    all_trade_idx = returns.index[returns.index >= start_plot]

    # Reconstruct price index (base 100) from cumulative log returns
    ret_plot    = returns.loc[all_trade_idx]
    price_recon = (ret_plot.cumsum().apply(math.exp) * 100).round(4)

    # Build sentiment series aligned to all trading days: null where no news
    sent_aligned = pd.Series(
        [round(float(daily_sent.loc[d]), 6) if d in daily_sent.index else None
         for d in all_trade_idx],
        index=all_trade_idx,
    )

    chart_dates = [str(d.date()) for d in all_trade_idx]
    chart_price = price_recon.tolist()
    # JSON-safe: NaN/None → null in JS (pandas converts None to float NaN in Series)
    def _js_val(v):
        if v is None:
            return "null"
        try:
            if math.isnan(v) or math.isinf(v):
                return "null"
            return str(round(float(v), 6))
        except (TypeError, ValueError):
            return "null"
    chart_sent = [_js_val(v) for v in sent_aligned.tolist()]

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
    Blue line: price index rebased to 100 (left axis).
    Purple dots: daily VADER sentiment score on days with actual news coverage (right axis).
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

<!-- ===== Interpretation ===== -->
<div class="card section">
  <h2>Interpretation</h2>
  <div style="color:{TEXT};font-size:14px;line-height:1.75;">
    {interpretation_html}
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
  // sentiments: null on non-news days, value on news days
  const sentiments = [{",".join(chart_sent)}];

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
          borderWidth: 1.5,
          spanGaps: true,
        }},
        {{
          label: 'Daily Sentiment',
          data: sentiments,
          borderColor: '{ACCENT}',
          backgroundColor: '{ACCENT}99',
          yAxisID: 'ySent',
          tension: 0,
          pointRadius: 5,
          pointHoverRadius: 7,
          borderWidth: 0,
          showLine: false,
          spanGaps: false,
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
    parser.add_argument("--ticker",      required=True,  type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--lookback",    default=30,     type=int, help="Days of history (default: 30)")
    parser.add_argument("--lags",        default=10,     type=int, help="Max lag to test in days (default: 10)")
    parser.add_argument("--finnhub-key", default="",     type=str,
                        help="Finnhub API key for historical news (free at finnhub.io). "
                             "Without this, falls back to Google News RSS (~2 weeks only).")
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
    # Key priority: CLI arg → FINNHUB_API_KEY env var → .env file (loaded above)
    finnhub_key = args.finnhub_key.strip() or os.environ.get("FINNHUB_API_KEY", "").strip()
    if finnhub_key:
        print(f"  Fetching news from Finnhub ({lookback}d lookback)...", flush=True)
    else:
        print(f"  Fetching news headlines ({lookback}d lookback, free sources)...", flush=True)
        print("  Tip: add FINNHUB_API_KEY=<key> to .env for historical data (free at finnhub.io)", flush=True)
    news_df = collect_news(ticker, lookback, finnhub_key=finnhub_key)
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
    n_days = len(daily_sent)
    print(f"  {n_days} days with sentiment data", flush=True)

    # ---- Price returns ----
    try:
        returns = fetch_returns(ticker, lookback)
    except ValueError as e:
        print(f"  ERROR: {e}", flush=True)
        sys.exit(1)

    # ---- Overlap check ----
    # With sparse news data we may only have 5-15 actual news days — that's fine.
    common = daily_sent.index.intersection(returns.index)
    if len(common) < 5:
        print(
            f"  ERROR: Only {len(common)} overlapping days between sentiment and returns. "
            f"Need at least 5. Try a larger --lookback value or check ticker.",
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
