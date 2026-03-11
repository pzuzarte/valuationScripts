#!/usr/bin/env python3
"""
Sentiment Analyzer
==================
Deep-dive multi-source sentiment analysis for a portfolio of stocks.

Data sources (no API keys required):
  · News headlines     — yfinance + VADER / keyword scoring
  · Social media       — Reddit public JSON API (r/wallstreetbets, r/stocks, r/investing)
  · Short interest     — yfinance (shortPercentOfFloat, shortRatio, daysTocover)
  · Institutional flow — yfinance (heldPercentInstitutions, top holders)
  · Insider activity   — yfinance transactions + SEC Form 4 (EDGAR)
  · Options market     — yfinance put/call ratio + implied volatility + put/call skew
  · Analyst ratings    — yfinance recommendations summary + EPS estimate revisions
  · SEC 8-K filings    — EDGAR submissions API (material events)
  · SEC 13D/13G        — EDGAR activist investor disclosures
  · SEC 13F            — EDGAR institutional portfolio snapshots
  · Congressional      — House & Senate STOCK Act disclosures (public S3)
  · Google Trends      — pytrends (optional) search interest 3-month window
  · Wikipedia views    — Wikimedia REST API daily page view counts (no auth)

Bond/credit note: direct bond pricing requires paid APIs (Bloomberg, FINRA TRACE).
Proxy metrics (debtToEquity, currentRatio, creditRating) are shown from yfinance.

Usage:
  python sentimentAnalyzer.py portfolio.csv
  python sentimentAnalyzer.py portfolio.csv --days 60
  python sentimentAnalyzer.py NVDA,AAPL,MSFT      # comma-separated tickers
"""

import argparse
import csv
import json
import os
import sys
import time
import datetime
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.parse import quote_plus
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    import yfinance as yf
except ImportError:
    sys.exit("ERROR: yfinance required.  Run: pip install yfinance")

# ── Optional pytrends ─────────────────────────────────────────────────────────
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

# ── Optional VADER sentiment ───────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    def score_text(text: str) -> float:
        return round(_VADER.polarity_scores(str(text or ""))["compound"], 3)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    _POS = {
        "beat","beats","surpass","record","growth","grew","bullish","buy",
        "upgrade","upgraded","profit","gain","rise","soar","rally","strong",
        "outperform","exceed","exceeded","boost","optimistic","recovery",
        "expand","dividend","buyback","acquisition","launch","launches",
        "positive","upbeat","accelerate","partnership","approve","approved",
    }
    _NEG = {
        "miss","misses","missed","decline","loss","losses","bearish","sell",
        "downgrade","downgraded","lawsuit","debt","fall","slump","crash",
        "weak","warning","concern","uncertain","risk","probe","investigation",
        "fraud","layoff","layoffs","cut","reduce","penalty","fine","recall",
        "shortage","delay","negative","disappoints","disappoint",
    }
    def score_text(text: str) -> float:
        if not text:
            return 0.0
        words = set(str(text).lower().split())
        pos   = len(words & _POS)
        neg   = len(words & _NEG)
        total = pos + neg
        return round((pos - neg) / total, 3) if total else 0.0

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "sentimentData")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDGAR_UA = "ValuationSuite/1.0 research@example.com"   # EDGAR requires User-Agent

SIGNAL_WEIGHTS = {
    "news":          0.18,
    "social":        0.08,
    "trends":        0.07,
    "short":         0.14,
    "institutional": 0.09,
    "insider":       0.18,
    "options":       0.10,
    "analyst":       0.16,
}
SIGNAL_LABELS = {
    "news":          "News",
    "social":        "Social",
    "trends":        "Trends",
    "short":         "Short Int.",
    "institutional": "Institutional",
    "insider":       "Insider",
    "options":       "Options",
    "analyst":       "Analyst",
}

# Session-level caches (populated once, shared across ticker threads)
_CIK_MAP_CACHE:   dict = {}
_CONGRESS_CACHE:  dict = {}

# ── Formatting helpers ─────────────────────────────────────────────────────────
def days_ago(n: int) -> str:
    return (datetime.date.today() - datetime.timedelta(days=n)).isoformat()

def fmt_pct(v, signed=False, dec=1):
    if v is None:
        return "—"
    s = "+" if (signed and v >= 0) else ""
    return f"{s}{v:.{dec}f}%"

def fmt_money(v):
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3:  return f"${v:,.0f}"
    return f"${v:.2f}"

def sentiment_label(score) -> str:
    if score is None:  return "N/A"
    if score >= 0.30:  return "Bullish"
    if score >= 0.10:  return "Slightly Bullish"
    if score > -0.10:  return "Neutral"
    if score > -0.30:  return "Slightly Bearish"
    return "Bearish"

def sentiment_color(score) -> str:
    if score is None:  return "#6b7194"
    if score >= 0.30:  return "#00c896"
    if score >= 0.10:  return "#5cb85c"
    if score > -0.10:  return "#f0a500"
    if score > -0.30:  return "#e07b39"
    return "#e05c5c"

def badge(label: str, color: str) -> str:
    return (f'<span style="background:{color}22;color:{color};padding:2px 9px;'
            f'border-radius:10px;font-size:11px;font-weight:700;">{label}</span>')

def sentiment_badge(score) -> str:
    return badge(sentiment_label(score), sentiment_color(score))

def score_bar(score, width=88) -> str:
    """Center-anchored colored bar, -1..1."""
    if score is None:
        return f'<div style="width:{width}px;height:6px;background:#1c2030;border-radius:3px;display:inline-block;"></div>'
    color  = sentiment_color(score)
    half   = width // 2
    bar_w  = int(abs(score) * half)
    left   = half if score >= 0 else half - bar_w
    return (
        f'<div style="width:{width}px;height:6px;background:#1c2030;border-radius:3px;'
        f'position:relative;display:inline-block;vertical-align:middle;">'
        f'<div style="position:absolute;top:0;height:100%;background:{color};border-radius:3px;'
        f'left:{left}px;width:{bar_w}px;"></div>'
        f'<div style="position:absolute;top:0;left:{half}px;width:1px;height:100%;background:#6b7194;opacity:.4;"></div>'
        f'</div>'
    )

def score_to_radar(score) -> float:
    """Map -1..1 → 0..100 for radar charts."""
    if score is None:
        return 50.0
    return round(max(0.0, min(100.0, score * 50 + 50)), 1)

# ── Help popover content ───────────────────────────────────────────────────────
_HELP_DATA = {
    # Overview
    "composite-score": {
        "title": "Composite Sentiment Score",
        "body": (
            "<b>What it is:</b> A weighted average of all available signal scores, "
            "ranging from −1 (strongly bearish) to +1 (strongly bullish).<br><br>"
            "<b>Weights:</b> News 18% · Insider 18% · Analyst 16% · Short Interest 14% · "
            "Options 10% · Institutional 9% · Social/Reddit 8% · Trends 7%.<br><br>"
            "<b>How to read it:</b> ≥+0.30 = Bullish · +0.10 to +0.30 = Slightly Bullish · "
            "−0.10 to +0.10 = Neutral · −0.10 to −0.30 = Slightly Bearish · ≤−0.30 = Bearish.<br><br>"
            "<b>Note:</b> Only signals with available data contribute. "
            "A composite built on 3 signals is less reliable than one built on 7."
        ),
    },
    "signal-heatmap": {
        "title": "Signal Heatmap",
        "body": (
            "<b>What it is:</b> Per-ticker breakdown of each individual signal score alongside "
            "the composite. Green = bullish · Red = bearish · Yellow = neutral.<br><br>"
            "<b>Score bars:</b> The centre-anchored bar shows direction and magnitude — "
            "extending right = bullish, extending left = bearish.<br><br>"
            "<b>When to dig deeper:</b> If the composite is positive but Short Interest or "
            "Insider is strongly red, investigate before acting — conflicting signals reduce conviction."
        ),
    },
    "radar-charts": {
        "title": "Signal Breakdown — Radar Charts",
        "body": (
            "<b>What it is:</b> A radar chart for each ticker showing all signal dimensions "
            "simultaneously. Values are mapped to 0–100 (50 = neutral).<br><br>"
            "<b>How to read it:</b> Large round shape = broadly bullish. Small collapsed shape = "
            "broadly bearish. Irregular shape = mixed signals.<br><br>"
            "<b>Best use:</b> Quick visual comparison across tickers. Large asymmetries "
            "(e.g. strong analyst but weak insider) warrant further investigation."
        ),
    },
    "attention-polarization": {
        "title": "Attention & Polarization",
        "body": (
            "<b>Google Trends slope (↑ ↓ →):</b> Whether retail search interest for '[TICKER] stock' "
            "is accelerating (↑), declining (↓), or flat (→) over the last 3 months vs the prior 2 months. "
            "Rising attention often precedes volatility — in either direction.<br><br>"
            "<b>Wikipedia avg views:</b> Daily page views of the company's Wikipedia article. "
            "Academic research shows view spikes precede price moves by 1–3 days.<br><br>"
            "<b>Reddit polarization:</b> How divided Reddit opinion is. "
            "0.0 = consensus (nearly all bull or all bear). "
            "0.5 = perfectly split 50/50. "
            "High polarization near a catalyst = potential sharp move either way.<br><br>"
            "<b>Squeeze potential:</b> Combines short % of float + days-to-cover into a 0–100 score. "
            "High score = heavily shorted stock with limited liquidity to cover; "
            "a positive catalyst could trigger a short squeeze.<br><br>"
            "<b>Signal confirmation (N/M):</b> How many of the M available signals agree with "
            "the composite direction. 6/7 = strong conviction. 3/7 = conflicted market."
        ),
    },
    # News & Social
    "recent-news": {
        "title": "Recent News",
        "body": (
            "<b>Source:</b> yfinance news feed (Yahoo Finance), filtered to your lookback window.<br><br>"
            "<b>Sentiment score:</b> Each headline is scored −1 to +1 using VADER sentiment analysis "
            "(or keyword fallback). The coloured dot shows polarity.<br><br>"
            "<b>Limitations:</b> Scored on headline only — not the full article. "
            "Clickbait and sarcasm can mislead the scorer. "
            "Always read the article for material events (earnings, lawsuits, FDA approvals)."
        ),
    },
    "reddit-sentiment": {
        "title": "Reddit Sentiment",
        "body": (
            "<b>Source:</b> Reddit public JSON API — r/wallstreetbets, r/stocks, r/investing.<br><br>"
            "<b>Donut chart:</b> Posts classified Bullish / Bearish / Neutral via VADER on the post title.<br><br>"
            "<b>Post list:</b> Top 8 posts by upvote count. Click a title to open the original Reddit post.<br><br>"
            "<b>Limitations:</b> VADER misses memes, irony, and WSB slang ('to the moon' scores neutral). "
            "Volume matters: 5 posts = weak signal; 50+ is more meaningful."
        ),
    },
    "news-velocity": {
        "title": "News Velocity",
        "body": (
            "<b>What it is:</b> Article count in the last 7 days vs the prior 7-day window.<br><br>"
            "<b>Velocity ratio:</b> recent ÷ prior. "
            "↑ &gt;1.5 = coverage accelerating. → 0.8–1.5 = stable. ↓ &lt;0.8 = declining.<br><br>"
            "<b>Why it matters:</b> A surge in coverage often precedes major price moves "
            "before the sentiment of those articles is clear. "
            "High velocity + negative sentiment = particularly strong bearish signal."
        ),
    },
    "google-trends": {
        "title": "Google Trends",
        "body": (
            "<b>Source:</b> Google Trends via pytrends — '[TICKER] stock', Finance category, US, 3 months.<br><br>"
            "<b>Index:</b> 0–100 relative scale (100 = peak popularity for the period). Not absolute volume.<br><br>"
            "<b>Trend slope:</b> Last 4 weeks vs prior 8 weeks. Positive = rising retail interest.<br><br>"
            "<b>Limitations:</b> Google rate-limits pytrends — if blank, the request was blocked. "
            "Reflects US retail attention only."
        ),
    },
    "wikipedia-views": {
        "title": "Wikipedia Page Views",
        "body": (
            "<b>Source:</b> Wikimedia REST API — daily English Wikipedia page views for the company.<br><br>"
            "<b>Why it matters:</b> Moat &amp; Preis (2013) found Wikipedia view spikes precede "
            "stock price moves — particularly downward moves — by 1–3 days.<br><br>"
            "<b>Trend slope:</b> Last third of window vs first third. Positive = accelerating interest.<br><br>"
            "<b>Limitations:</b> Best for well-known companies. Small-caps with few views produce noisy signals."
        ),
    },
    # Smart Money
    "short-interest": {
        "title": "Short Interest",
        "body": (
            "<b>Short % of float:</b> Shares sold short as a % of tradeable shares. "
            "Above 15% = high; above 25% = very high.<br><br>"
            "<b>Short ratio (days to cover):</b> Short shares ÷ avg daily volume. "
            "How many trading days it would take all shorts to cover at current volume.<br><br>"
            "<b>Signal score:</b> 0% short → neutral (0.0). 15%+ → bearish (−1.0).<br><br>"
            "<b>Contrarian note:</b> Very high short interest can be a contrarian bullish signal "
            "if a squeeze is triggered. Cross-reference with the Squeeze Potential score."
        ),
    },
    "institutional-holdings": {
        "title": "Institutional Holdings",
        "body": (
            "<b>% Institutional:</b> Fraction of shares held by funds, ETFs, and pensions. "
            "High institutional ownership = professional validation.<br><br>"
            "<b>Score logic:</b> 65% = neutral · 90%+ = +1.0 · 40% or below = −1.0.<br><br>"
            "<b>Limitation:</b> Point-in-time snapshot — doesn't show direction of flow. "
            "Use the 13F filings in the SEC tab for directional changes."
        ),
    },
    "top-holders": {
        "title": "Top Institutional Holders",
        "body": (
            "<b>What it shows:</b> The largest institutional shareholders (from SEC 13F filings, ~45-day lag).<br><br>"
            "<b>% Held:</b> Each institution's ownership as a fraction of outstanding shares.<br><br>"
            "<b>Limitation:</b> 13F filings have up to 45-day delay. "
            "A fund that sold last week still appears here until the next quarterly filing."
        ),
    },
    "insider-transactions": {
        "title": "Insider Transactions",
        "body": (
            "<b>Who counts as an insider:</b> Officers, directors, and 10%+ shareholders "
            "required to disclose trades under SEC rules (Form 4, within 2 business days).<br><br>"
            "<b>Signal score:</b> Weighted by transaction value. Large open-market purchases "
            "are very bullish. Sales are bearish but given less weight — executives often sell "
            "for diversification, not just because they're bearish.<br><br>"
            "<b>Strongest signal:</b> Cluster buying — multiple insiders buying at the same time. "
            "Open-market purchases far outweigh option exercises as a bullish signal."
        ),
    },
    "squeeze-potential": {
        "title": "Short Squeeze Potential",
        "body": (
            "<b>Score:</b> 0–100 combining short % of float (60%) + days-to-cover (40%).<br><br>"
            "<b>Levels:</b> Low (0–25) · Moderate (25–50) · High (50–75) · Extreme (&gt;75)<br><br>"
            "<b>How a squeeze works:</b> A rising stock forces short sellers to buy to cover losses. "
            "That buying pushes the price higher, forcing more shorts to cover — a feedback loop. "
            "High short float + high DTC + positive catalyst = classic squeeze setup.<br><br>"
            "<b>Caveat:</b> High short interest can also mean smart money is right about the stock. "
            "Cross-reference with fundamentals and insider signals."
        ),
    },
    "congressional-trading": {
        "title": "Congressional Trading (STOCK Act)",
        "body": (
            "<b>Source:</b> House and Senate STOCK Act disclosures "
            "(housestockwatcher.com / senatestockwatcher.com public data).<br><br>"
            "<b>Why it matters:</b> Members of Congress must disclose trades within 45 days. "
            "Studies show congressional portfolios have historically outperformed — "
            "possibly reflecting policy knowledge.<br><br>"
            "<b>How to read it:</b> Coordinated buys across multiple members are more meaningful "
            "than a single trade. Window shows last 12 months regardless of date filter.<br><br>"
            "<b>Limitation:</b> Self-reported, sometimes filed late or with vague descriptions."
        ),
    },
    # Options & Analyst
    "pcr-iv": {
        "title": "Put/Call Ratio & Implied Volatility",
        "body": (
            "<b>Put/Call Ratio (PCR):</b> Total put volume ÷ total call volume for the nearest expiry ≥7 days. "
            "&lt;0.7 = bullish (call buying) · ~1.0 = neutral · &gt;1.4 = bearish (put buying/hedging).<br><br>"
            "<b>ATM Implied Volatility:</b> Market's expectation of future price movement, "
            "annualised %. IV &gt;50% is elevated for large caps; normal for small-caps and biotechs.<br><br>"
            "<b>Nearest expiry ≥7d:</b> All options metrics skip expiries &lt;7 days to avoid "
            "distortion from extreme theta decay on 1–2 day weeklies."
        ),
    },
    "implied-move": {
        "title": "Market-Implied Move (ATM Straddle)",
        "body": (
            "<b>Formula:</b> (ATM call mid + ATM put mid) ÷ spot price = ±implied move %<br><br>"
            "<b>What it means:</b> The option market's consensus expected price range by expiry. "
            "±5% = market expects the stock to stay within 5% of today's price.<br><br>"
            "<b>30d HV (Historical Volatility):</b> Actual annualised daily vol over the last 30 days.<br><br>"
            "<b>IV/HV ratio:</b><br>"
            "· &gt;1.3 = options expensive — market fears a move. Consider selling premium.<br>"
            "· ~1.0 = fairly priced.<br>"
            "· &lt;1.0 = options cheap — market calm. Consider buying premium.<br><br>"
            "<b>⚡ flag:</b> Earnings fall within the expiry window — straddle price reflects earnings risk."
        ),
    },
    "implied-move-chart": {
        "title": "Implied Move Chart — Normalized % Scale",
        "body": (
            "<b>How to read it:</b> Each bar spans −implied_move% to +implied_move%, centred at 0. "
            "All stocks share the same Y-axis scale for direct comparison.<br><br>"
            "<b>DTE labels:</b> Shown in parentheses — e.g. NVDA (13d). "
            "Compare tickers with similar DTEs; a 7-day implied move is always smaller than a 21-day one.<br><br>"
            "<b>Colours:</b> Green &lt;5% · Orange 5–10% · Red &gt;10%<br><br>"
            "<b>n/a:</b> Implied move &gt;40% is flagged as bad data — usually illiquid options with "
            "extremely wide bid/ask spreads."
        ),
    },
    "vol-skew": {
        "title": "Volatility Skew (IV Skew Ratio)",
        "body": (
            "<b>Formula:</b> OTM put IV (at ~88% of spot) ÷ ATM call IV.<br><br>"
            "<b>Why it matters:</b> OTM puts are almost always more expensive than equidistant calls — "
            "this is the 'volatility smirk'. The ratio measures how pronounced the fear premium is.<br><br>"
            "<b>How to read it:</b><br>"
            "· ~1.0–1.2 = normal/flat<br>"
            "· 1.2–1.5 = mild fear premium — market hedging downside<br>"
            "· &gt;1.5 = elevated fear — institutions paying up for protection (bearish signal)<br>"
            "· &lt;1.0 = calls bid up — speculative bullish demand (rare)<br><br>"
            "<b>Note:</b> High skew is a bearish signal but not a timing signal. "
            "Elevated skew can persist for months."
        ),
    },
    "eps-revision": {
        "title": "Earnings Estimate Revisions (60-Day)",
        "body": (
            "<b>What it shows:</b> Whether analysts have raised or cut their current-quarter EPS "
            "estimate over the past 60 days.<br><br>"
            "<b>↑ Raised:</b> Analysts more optimistic — bullish signal.<br>"
            "<b>↓ Cut:</b> Analysts revising down — bearish signal.<br><br>"
            "<b>Why it matters:</b> Systematic estimate revisions are one of the strongest "
            "fundamental momentum signals. Stocks with rising estimates tend to outperform.<br><br>"
            "<b>Source:</b> yfinance eps_trend — current vs 60daysAgo for the '0q' (current quarter)."
        ),
    },
    "earnings-calendar": {
        "title": "Upcoming Earnings & Consensus Estimates",
        "body": (
            "<b>Beat Rate:</b> How often the company beat consensus EPS over the last 8 quarters. "
            "100% = consistent beat history.<br><br>"
            "<b>Avg EPS Surprise %:</b> Average magnitude of beat/miss across reported quarters.<br><br>"
            "<b>Trend:</b> Whether recent beats are getting larger (Improving ↗), "
            "smaller (Fading ↘), or consistent (Stable →). Based on last 2 vs prior 2 quarters.<br><br>"
            "<b>⚡ on Days Away:</b> Earnings within 7 days — elevated event risk.<br><br>"
            "<b>Source:</b> yfinance calendar (estimates) + earnings_history (beat/miss)."
        ),
    },
    "eps-history": {
        "title": "EPS Surprise History — Last 8 Quarters",
        "body": (
            "<b>Green bar:</b> Beat — actual EPS ≥ consensus estimate.<br>"
            "<b>Red bar:</b> Miss — actual EPS &lt; estimate.<br><br>"
            "<b>Bar height:</b> EPS surprise % = (actual − estimate) ÷ |estimate| × 100.<br><br>"
            "<b>How to interpret:</b> Consistent beats of 5–15% = management under-promises "
            "and over-delivers. Shrinking beats over time may signal guidance pressure or "
            "analysts catching up to the real growth rate.<br><br>"
            "<b>Cap:</b> Values capped at ±50% to prevent near-zero-estimate outliers from "
            "distorting the chart scale."
        ),
    },
    "analyst-ratings": {
        "title": "Analyst Ratings",
        "body": (
            "<b>What it shows:</b> Distribution of analyst recommendations for the current "
            "and prior 3 quarters (Strong Buy / Buy / Hold / Sell / Strong Sell).<br><br>"
            "<b>Signal score:</b> Strong Buy=+1 · Buy=+0.5 · Hold=0 · Sell=−0.5 · Strong Sell=−1, "
            "divided by total analyst count.<br><br>"
            "<b>Important caveat:</b> ~55% of all sell-side ratings are Buy. "
            "A Hold from an analyst who previously had a Buy is effectively a downgrade. "
            "Analyst ratings are lagging indicators — price often moves before ratings change."
        ),
    },
    # SEC Filings
    "sec-8k": {
        "title": "SEC 8-K Filings (Material Events)",
        "body": (
            "<b>What is an 8-K?</b> A mandatory SEC disclosure for any material event shareholders "
            "need to know about.<br><br>"
            "<b>Common triggers:</b> Earnings results · major acquisitions · CEO changes · "
            "bankruptcy · legal settlements · guidance changes · product approvals.<br><br>"
            "<b>How to use it:</b> Click any link to read the full filing on SEC EDGAR. "
            "A cluster of 8-K filings in a short window = significant activity worth investigating.<br><br>"
            "<b>Source:</b> SEC EDGAR submissions API, filtered to your date window."
        ),
    },
    "sec-form4": {
        "title": "SEC Form 4 (Insider Filings)",
        "body": (
            "<b>What is Form 4?</b> Filed within 2 business days of any trade by an insider "
            "(officers, directors, 10%+ shareholders). The most timely official insider record.<br><br>"
            "<b>Transaction codes to watch:</b><br>"
            "· P = Open-market purchase — most bullish signal<br>"
            "· S = Sale — bearish, but context matters<br>"
            "· M = Option exercise — routine, usually followed by S<br>"
            "· A = Award/grant — not a market purchase, neutral<br><br>"
            "<b>Strongest signal:</b> Multiple P transactions clustered in time from different insiders."
        ),
    },
    "sec-activist": {
        "title": "Activist Disclosures (13D / 13G)",
        "body": (
            "<b>Trigger:</b> Any investor crossing 5% ownership must file within 10 days.<br><br>"
            "<b>13D = Activist:</b> Intends to influence management, push for changes, "
            "seek board seats, or force a sale. Historically triggers 7–10% average price jump on filing.<br><br>"
            "<b>13G = Passive:</b> 5%+ holder with no intent to influence. "
            "Less immediately actionable but confirms strong institutional interest.<br><br>"
            "<b>No filings = normal:</b> Most stocks have none. Absence is not a red flag."
        ),
    },
    "sec-13f": {
        "title": "Recent 13F Filings",
        "body": (
            "<b>What is a 13F?</b> Quarterly report by institutional managers with &gt;$100M AUM, "
            "disclosing all equity positions &gt;$10,000 or 10,000 shares.<br><br>"
            "<b>45-day lag:</b> Filed 45 days after quarter end. By the time you see a fund "
            "bought a stock, they may have already sold it.<br><br>"
            "<b>What the listing shows:</b> Filing date and a link to the XML on EDGAR. "
            "Click to see full holdings, or use WhaleWisdom / 13f.info for parsed views.<br><br>"
            "<b>Best use:</b> Confirm whether significant institutional tracking exists. "
            "Zero recent 13Fs = no major institutional tracker on this name."
        ),
    },
}


def help_btn(key: str) -> str:
    """Inline ⓘ button that opens the help popover for the given key."""
    safe = key.replace("'", "\\'")
    return (
        f'<button class="help-btn" onclick="showHelp(\'{safe}\',event)" '
        f'aria-label="Help" title="Click for help">ⓘ</button>'
    )

# ── HTTP helper ────────────────────────────────────────────────────────────────
def _get(url: str, ua: str = None, timeout: int = 14):
    try:
        headers = {"User-Agent": ua or "Mozilla/5.0", "Accept": "application/json"}
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None

# ── SEC EDGAR CIK lookup ───────────────────────────────────────────────────────
def _load_cik_map() -> dict:
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE:
        return _CIK_MAP_CACHE
    data = _get("https://www.sec.gov/files/company_tickers.json", ua=EDGAR_UA, timeout=20)
    if data:
        for entry in data.values():
            t = (entry.get("ticker") or "").upper()
            if t:
                _CIK_MAP_CACHE[t] = str(entry["cik_str"]).zfill(10)
    return _CIK_MAP_CACHE

def get_cik(ticker: str):
    return _load_cik_map().get(ticker.upper())

# ── Congressional trading data (STOCK Act disclosures) ────────────────────────
def _load_congress_data() -> dict:
    global _CONGRESS_CACHE
    if _CONGRESS_CACHE:
        return _CONGRESS_CACHE
    result = {"house": [], "senate": []}
    house = _get(
        "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
        timeout=25,
    )
    if isinstance(house, list):
        result["house"] = house
    senate = _get(
        "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
        timeout=25,
    )
    if isinstance(senate, list):
        result["senate"] = senate
    _CONGRESS_CACHE = result
    return result

def fetch_congressional(ticker: str, days: int = 365) -> list:
    data   = _load_congress_data()
    cutoff = days_ago(days)
    out    = []
    for txn in data.get("house", []):
        t    = (txn.get("ticker") or "").upper().strip("$")
        date = txn.get("transaction_date") or txn.get("disclosure_date") or ""
        if t == ticker.upper() and date >= cutoff:
            out.append({
                "chamber": "House",
                "member":  txn.get("representative", ""),
                "party":   txn.get("party", ""),
                "type":    txn.get("type", ""),
                "amount":  txn.get("amount", ""),
                "date":    date,
            })
    for txn in data.get("senate", []):
        t    = (txn.get("ticker") or "").upper().strip("$")
        date = txn.get("transaction_date") or txn.get("disclosure_date") or ""
        if t == ticker.upper() and date >= cutoff:
            fn = txn.get("first_name", "")
            ln = txn.get("last_name", "")
            out.append({
                "chamber": "Senate",
                "member":  txn.get("senator", f"{fn} {ln}").strip(),
                "party":   txn.get("party", ""),
                "type":    txn.get("type", ""),
                "amount":  txn.get("amount", ""),
                "date":    date,
            })
    return sorted(out, key=lambda x: x["date"], reverse=True)[:25]

# ── SEC EDGAR filings ──────────────────────────────────────────────────────────
def fetch_sec_filings(cik: str, form_type: str = "8-K",
                      days: int = 60, limit: int = 10) -> list:
    if not cik:
        return []
    data = _get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                ua=EDGAR_UA, timeout=15)
    if not data:
        return []
    recent = data.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    dates  = recent.get("filingDate", [])
    docs   = recent.get("primaryDocument", [])
    accns  = recent.get("accessionNumber", [])
    descs  = recent.get("primaryDocDescription", [])
    cutoff = days_ago(days)
    out    = []
    for form, date, doc, accn, desc in zip(forms, dates, docs, accns,
                                            descs or [""] * len(forms)):
        if form == form_type and date >= cutoff:
            cik_int    = int(cik)
            accn_clean = accn.replace("-", "")
            link = (f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
                    f"/{accn_clean}/{doc}")
            out.append({
                "form": form, "date": date, "doc": doc,
                "desc": desc or "", "link": link,
            })
        if len(out) >= limit:
            break
    return out

# ── Google Trends ──────────────────────────────────────────────────────────────
def fetch_google_trends(ticker: str, company_name: str = "") -> dict:
    """Fetch 3-month Google Trends interest for ticker stock search term."""
    empty = {"interest_now": None, "interest_avg": None,
             "trend_slope": None, "series": [], "kw": ""}
    if not PYTRENDS_AVAILABLE:
        return empty
    try:
        kw = ticker + " stock"
        pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pt.build_payload([kw], cat=7, timeframe="today 3-m", geo="US")
        df = pt.interest_over_time()
        if df is None or df.empty or kw not in df.columns:
            return empty
        vals = list(df[kw].values)
        if len(vals) < 12:
            return empty
        last12 = [float(v) for v in vals[-12:]]
        now_vals = last12[-4:]
        prior_vals = last12[-12:-4]
        interest_now = round(sum(now_vals) / max(len(now_vals), 1), 1)
        interest_avg = round(sum(prior_vals) / max(len(prior_vals), 1), 1)
        trend_slope = round((interest_now - interest_avg) / max(interest_avg, 1), 3)
        return {
            "interest_now": interest_now,
            "interest_avg": interest_avg,
            "trend_slope": trend_slope,
            "series": last12,
            "kw": kw,
        }
    except Exception:
        return empty


# ── Wikipedia page views ───────────────────────────────────────────────────────
def fetch_wikipedia_views(company_name: str, ticker: str, days: int = 60) -> dict:
    """Fetch daily Wikipedia page views via Wikimedia REST API (no auth required)."""
    empty = {"avg_daily": None, "peak_daily": None,
             "trend_slope": None, "series": [], "page": None}
    try:
        search_term = company_name if company_name else ticker
        search_url = (
            "https://en.wikipedia.org/w/api.php?action=opensearch"
            "&search=" + quote_plus(search_term) + "&limit=3&format=json"
        )
        opensearch = _get(search_url, timeout=10)
        if not opensearch or len(opensearch) < 2:
            if company_name:
                return fetch_wikipedia_views("", ticker, days)
            return empty
        titles = opensearch[1] if isinstance(opensearch[1], list) else []
        page_title = None
        for t in titles:
            if "disambiguation" not in t.lower():
                page_title = t.replace(" ", "_")
                break
        if not page_title:
            if company_name:
                return fetch_wikipedia_views("", ticker, days)
            return empty

        end_dt = datetime.date.today()
        start_dt = end_dt - datetime.timedelta(days=days)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")
        pv_url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
            "/en.wikipedia/all-access/all-agents/"
            + quote_plus(page_title) + "/daily/" + start_str + "/" + end_str
        )
        pv_data = _get(pv_url, ua="ValuationSuite/1.0", timeout=14)
        if not pv_data or "items" not in pv_data:
            return empty
        items = pv_data["items"]
        series = [int(item.get("views", 0)) for item in items]
        if not series:
            return empty
        avg_daily = round(sum(series) / len(series), 0)
        peak_daily = max(series)
        third = max(1, len(series) // 3)
        first_third_avg = sum(series[:third]) / third
        last_third_avg = sum(series[-third:]) / third
        trend_slope = round(
            (last_third_avg - first_third_avg) / max(first_third_avg, 1), 3
        )
        return {
            "avg_daily": int(avg_daily),
            "peak_daily": peak_daily,
            "trend_slope": trend_slope,
            "series": series,
            "page": page_title,
        }
    except Exception:
        return empty


# ── Short squeeze composite score ──────────────────────────────────────────────
def compute_squeeze(info: dict) -> dict:
    """Compute short squeeze potential composite score."""
    si_float = float(info.get("shortPercentOfFloat") or 0)
    dtc = float(info.get("shortRatio") or 0)
    si_score = min(100.0, si_float * 500)
    dtc_score = min(100.0, dtc * 10)
    score = round(si_score * 0.6 + dtc_score * 0.4, 1)
    if score > 75:
        level = "Extreme"
    elif score > 50:
        level = "High"
    elif score > 25:
        level = "Moderate"
    else:
        level = "Low"
    return {
        "score": score,
        "level": level,
        "si_float_pct": round(si_float * 100, 1),
        "dtc": round(dtc, 1),
    }


# ── Reddit sentiment ───────────────────────────────────────────────────────────
# StockTwits API is blocked by Cloudflare (requires JS challenge); replaced with
# Reddit's public JSON API which requires no authentication.
# Reddit enforces ~1 req/s on the public API — use a global lock so parallel
# ticker threads don't hammer it simultaneously and trigger 429s.
_REDDIT_UA   = "python:ValuationSuite:v1.0 (automated research tool)"
_REDDIT_SUBS = ["wallstreetbets", "stocks", "investing"]
_REDDIT_LOCK = threading.Lock()   # serialises all Reddit API calls across threads
_REDDIT_DELAY = 0.6               # seconds between requests (≤1 req/s limit)

# yfinance 1.2.0 uses a single shared global session + crumb across all Ticker
# instances.  Concurrent threads that call quoteSummary simultaneously race on
# that shared crumb and cause HTTP 500 "internal-error" responses.  Serialising
# all fetch_yf calls (Semaphore(1)) eliminates the race; Reddit / SEC / Google
# Trends / Wikipedia still run in parallel across threads after the lock is
# released, so wall-clock time is only modestly affected.
_YF_SEM = threading.Semaphore(1)

def fetch_reddit_sentiment(ticker: str, days: int = 30) -> dict:
    """Fetch recent Reddit posts mentioning ticker from WSB/stocks/investing."""
    empty = {"bullish": 0, "bearish": 0, "neutral": 0, "total": 0, "messages": []}
    cutoff_epoch = time.time() - days * 86400
    ticker_upper = ticker.upper()
    all_posts = []

    for sub in _REDDIT_SUBS:
        url = (f"https://www.reddit.com/r/{sub}/search.json"
               f"?q={quote_plus(ticker_upper)}&sort=new&limit=25&restrict_sr=1&t=month")
        data = None
        headers = {"User-Agent": _REDDIT_UA, "Accept": "application/json"}
        for attempt in range(3):          # up to 3 attempts with backoff
            with _REDDIT_LOCK:            # serialise all Reddit calls across threads
                try:
                    if _REQUESTS_AVAILABLE:
                        resp = _requests.get(url, headers=headers, timeout=12)
                        if resp.status_code == 429:
                            wait = 4 * (attempt + 1)  # 4s, 8s, 12s
                            time.sleep(wait)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                    else:
                        req = Request(url, headers=headers)
                        with urlopen(req, timeout=12) as r:
                            data = json.loads(r.read().decode("utf-8", errors="replace"))
                    time.sleep(_REDDIT_DELAY)
                    break
                except Exception as e:
                    code = getattr(e, "code", None) or getattr(getattr(e, "response", None), "status_code", None)
                    if code == 429:
                        wait = 4 * (attempt + 1)
                        time.sleep(wait)
                    else:
                        time.sleep(_REDDIT_DELAY)
                        break                       # non-429 error — skip this sub
        if not data:
            continue
        posts = (data.get("data") or {}).get("children") or []
        for p in posts:
            pd = p.get("data") or {}
            created = float(pd.get("created_utc", 0))
            if created < cutoff_epoch:
                continue
            title   = pd.get("title", "")
            selftext = (pd.get("selftext") or "")[:300]
            # Keep only posts that actually contain the ticker symbol
            combined = f"{title} {selftext}".upper()
            if ticker_upper not in combined:
                continue
            sc = score_text(f"{title} {selftext}")
            all_posts.append({
                "title":       title[:200],
                "body":        selftext,
                "score":       sc,
                "subreddit":   sub,
                "created_utc": created,
                "upvotes":     pd.get("score", 0),
                "url":         "https://reddit.com" + pd.get("permalink", ""),
            })

    if not all_posts:
        return empty

    # Sort by upvotes so highest-engagement posts surface first
    all_posts.sort(key=lambda x: x.get("upvotes", 0), reverse=True)

    bull = sum(1 for p in all_posts if p["score"] >  0.05)
    bear = sum(1 for p in all_posts if p["score"] < -0.05)
    neut = len(all_posts) - bull - bear

    messages = [
        {
            "body":       p["title"],
            "sentiment":  ("Bullish" if p["score"] > 0.05
                           else ("Bearish" if p["score"] < -0.05 else "")),
            "score":      p["score"],
            "subreddit":  p["subreddit"],
            "upvotes":    p["upvotes"],
            "created_at": (datetime.datetime.utcfromtimestamp(p["created_utc"])
                           .strftime("%Y-%m-%d") if p["created_utc"] else ""),
            "url":        p.get("url", ""),
        }
        for p in all_posts[:8]
    ]

    return {"bullish": bull, "bearish": bear, "neutral": neut,
            "total": len(all_posts), "messages": messages}

# ── yfinance (single Ticker object for all fields) ────────────────────────────
def fetch_yf(ticker: str, days: int = 30) -> dict:
    result = {
        "info": {}, "news": [], "institutional": [],
        "insider_txns": [], "rec_summary": [], "options": {}, "error": None,
    }
    try:
        tk   = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            pass
        result["info"] = info

        # ── News ──────────────────────────────────────────────────────────────
        try:
            raw_news = tk.news or []
            cutoff   = time.time() - days * 86400
            news_out = []
            for n in raw_news:
                content = n.get("content") if isinstance(n.get("content"), dict) else {}
                if content.get("title"):
                    title     = content.get("title", "")
                    publisher = (content.get("provider") or {}).get("displayName", "")
                    link      = (content.get("canonicalUrl") or {}).get("url", "")
                    pub_str   = content.get("pubDate", "")
                    try:
                        pub_ts = datetime.datetime.fromisoformat(
                            pub_str.replace("Z", "")).timestamp()
                    except Exception:
                        pub_ts = 0
                else:
                    title     = n.get("title", "")
                    publisher = n.get("publisher", "")
                    link      = n.get("link", "")
                    pub_ts    = n.get("providerPublishTime", 0)
                if not title or pub_ts < cutoff:
                    continue
                news_out.append({
                    "title":     title,
                    "publisher": publisher,
                    "link":      link,
                    "time":      pub_ts,
                    "score":     score_text(title),
                })
            result["news"] = sorted(news_out, key=lambda x: x["time"], reverse=True)

            # ── News velocity ───────────────────────────────────────────────
            now_ts = time.time()
            recent_7d = sum(1 for n in result["news"]
                            if n["time"] >= now_ts - 7 * 86400)
            prior_7d = sum(1 for n in result["news"]
                           if now_ts - 14 * 86400 <= n["time"] < now_ts - 7 * 86400)
            velocity_ratio = round(recent_7d / max(prior_7d, 1), 2)
            result["news_velocity"] = {
                "recent_7d": recent_7d,
                "prior_7d": prior_7d,
                "velocity_ratio": velocity_ratio,
            }
        except Exception:
            pass

        # ── Institutional holders ──────────────────────────────────────────────
        try:
            df = tk.institutional_holders
            if df is not None and not df.empty:
                rows = []
                for _, row in df.head(8).iterrows():
                    holder = str(row.get("Holder", row.get("holder", "")))
                    shares = row.get("Shares", row.get("shares", 0))
                    pct    = row.get("% Out", row.get("pctHeld",
                                    row.get("% Held", None)))
                    val    = row.get("Value", row.get("value", None))
                    rows.append({"holder": holder, "shares": shares,
                                 "pct": pct, "value": val})
                result["institutional"] = rows
        except Exception:
            pass

        # ── Insider transactions ───────────────────────────────────────────────
        try:
            df = tk.insider_transactions
            if df is not None and not df.empty:
                cutoff_dt = datetime.date.today() - datetime.timedelta(days=max(days, 90))
                rows = []
                for _, row in df.iterrows():
                    date_val = row.get("Start Date", row.get("startDate", ""))
                    try:
                        date_str = (date_val.date().isoformat()
                                    if hasattr(date_val, "date")
                                    else str(date_val)[:10])
                    except Exception:
                        date_str = ""
                    if date_str and date_str < str(cutoff_dt):
                        continue
                    shares = row.get("Shares", row.get("shares", 0))
                    try:
                        shares = int(str(shares).replace(",", "")) if shares else 0
                    except (ValueError, TypeError):
                        shares = 0
                    val = row.get("Value", row.get("value", None))
                    try:
                        val = float(str(val).replace(",", "").replace("$", "")) if val else None
                    except (ValueError, TypeError):
                        val = None
                    rows.append({
                        "insider":     str(row.get("Insider", row.get("insider", ""))),
                        "position":    str(row.get("Position", row.get("position", ""))),
                        "transaction": str(row.get("Transaction", row.get("transaction", ""))),
                        "shares":      shares,
                        "value":       val,
                        "date":        date_str,
                    })
                result["insider_txns"] = rows[:20]
        except Exception:
            pass

        # ── Analyst recommendations ────────────────────────────────────────────
        for getter in [
            lambda: tk.recommendations,
            lambda: (tk.get_recommendations_summary()
                     if hasattr(tk, "get_recommendations_summary") else None),
            lambda: (tk.recommendations_summary
                     if hasattr(tk, "recommendations_summary") else None),
        ]:
            try:
                df = getter()
                if df is None or df.empty:
                    continue
                cols = [c.lower().replace("_", "").replace(" ", "") for c in df.columns]
                if "strongbuy" in cols:
                    recs = []
                    for _, row in df.head(4).iterrows():
                        recs.append({
                            "period":     str(row.get("period", "")),
                            "strongBuy":  int(row.get("strongBuy", 0) or 0),
                            "buy":        int(row.get("buy", 0) or 0),
                            "hold":       int(row.get("hold", 0) or 0),
                            "sell":       int(row.get("sell", 0) or 0),
                            "strongSell": int(row.get("strongSell", 0) or 0),
                        })
                    if recs:
                        result["rec_summary"] = recs
                        break
            except Exception:
                continue

        # ── Options (put/call ratio + IV + implied move) ───────────────────────
        try:
            exp_dates = tk.options
            if exp_dates:
                today = datetime.date.today()
                # Prefer an expiry that is at least 7 days away so the straddle
                # reflects a meaningful time horizon and isn't distorted by
                # theta decay on 1–2 day weeklies.  Fall back to nearest if
                # nothing qualifies (e.g. index options with long gaps).
                chosen_exp = exp_dates[0]
                for ed in exp_dates:
                    try:
                        ed_days = (datetime.datetime.strptime(ed, "%Y-%m-%d").date()
                                   - today).days
                        if ed_days >= 7:
                            chosen_exp = ed
                            break
                    except Exception:
                        pass

                chain     = tk.option_chain(chosen_exp)
                calls_vol = chain.calls["volume"].fillna(0).sum() if "volume" in chain.calls.columns else 0
                puts_vol  = chain.puts["volume"].fillna(0).sum()  if "volume" in chain.puts.columns  else 0
                pc_ratio  = round(puts_vol / calls_vol, 3) if calls_vol > 0 else None
                spot      = (info.get("currentPrice") or info.get("regularMarketPrice"))
                iv = None
                if (spot and "strike" in chain.calls.columns
                        and "impliedVolatility" in chain.calls.columns):
                    atm = chain.calls.iloc[
                        (chain.calls["strike"] - spot).abs().argsort()[:1]]
                    if len(atm):
                        iv = round(float(atm["impliedVolatility"].values[0]), 3)

                result["options"] = {
                    "put_call_ratio":     pc_ratio,
                    "calls_volume":       int(calls_vol),
                    "puts_volume":        int(puts_vol),
                    "implied_volatility": iv,
                    "expiry":             chosen_exp,
                }

                # ── ATM straddle → implied move ─────────────────────────────
                try:
                    if spot and "strike" in chain.calls.columns:
                        exp_dt = datetime.datetime.strptime(chosen_exp, "%Y-%m-%d").date()
                        dte    = max(1, (exp_dt - datetime.date.today()).days)

                        atm_idx    = (chain.calls["strike"] - spot).abs().argsort().iloc[0]
                        atm_strike = float(chain.calls.iloc[atm_idx]["strike"])

                        c_rows = chain.calls[chain.calls["strike"] == atm_strike]
                        p_rows = chain.puts[chain.puts["strike"]  == atm_strike]

                        def _mid(row):
                            b = float(row.get("bid", 0) or 0)
                            a = float(row.get("ask", 0) or 0)
                            l = float(row.get("lastPrice", 0) or 0)
                            return (b + a) / 2 if (b > 0 and a > 0) else l

                        if not c_rows.empty and not p_rows.empty:
                            straddle = _mid(c_rows.iloc[0]) + _mid(p_rows.iloc[0])
                            if straddle > 0 and spot > 0:
                                impl_pct = straddle / spot
                                result["options"].update({
                                    "atm_strike":         atm_strike,
                                    "dte":                dte,
                                    "straddle_price":     round(straddle, 2),
                                    "implied_move_pct":   round(impl_pct * 100, 2),
                                    "implied_move_dollar": round(straddle, 2),
                                    "expected_upper":     round(spot + straddle, 2),
                                    "expected_lower":     round(spot - straddle, 2),
                                    "spot":               round(float(spot), 2),
                                })
                except Exception:
                    pass

                # ── Volatility skew: OTM put IV / ATM call IV ───────────────
                # Standard skew measure — IV ratio >1 means puts are priced
                # with a higher vol than equidistant calls (downside fear premium)
                try:
                    if (spot and "strike" in chain.calls.columns
                            and "impliedVolatility" in chain.calls.columns
                            and "impliedVolatility" in chain.puts.columns):
                        # ATM call IV
                        atm_c_idx = (chain.calls["strike"] - spot).abs().argsort().iloc[0]
                        atm_c_row = chain.calls.iloc[atm_c_idx]
                        atm_call_strike_val = float(atm_c_row["strike"])
                        atm_call_iv = float(atm_c_row.get("impliedVolatility", 0) or 0)

                        # OTM put at ~88% of spot (≈25-delta proxy)
                        otm_puts = chain.puts[chain.puts["strike"] <= spot * 0.90]
                        # OTM call at ~112% of spot for reference
                        otm_calls_df = chain.calls[chain.calls["strike"] >= spot * 1.10]

                        otm_put_strike_val = None
                        otm_put_iv = 0.0
                        if not otm_puts.empty:
                            op_idx = (otm_puts["strike"] - spot * 0.88).abs().argsort().iloc[0]
                            op_row = otm_puts.iloc[op_idx]
                            otm_put_strike_val = float(op_row["strike"])
                            otm_put_iv = float(op_row.get("impliedVolatility", 0) or 0)

                        otm_call_strike_val = None
                        otm_call_iv = 0.0
                        if not otm_calls_df.empty:
                            oc_idx = (otm_calls_df["strike"] - spot * 1.12).abs().argsort().iloc[0]
                            oc_row = otm_calls_df.iloc[oc_idx]
                            otm_call_strike_val = float(oc_row["strike"])
                            otm_call_iv = float(oc_row.get("impliedVolatility", 0) or 0)

                        # IV skew ratio: >1.5 = elevated downside fear (bearish)
                        if atm_call_iv > 0.01 and otm_put_iv > 0.01:
                            put_skew_iv = round(otm_put_iv / atm_call_iv, 3)
                            result["options"]["put_skew"]         = put_skew_iv
                            result["options"]["atm_call_iv"]      = round(atm_call_iv * 100, 1)
                            result["options"]["otm_put_iv"]       = round(otm_put_iv * 100, 1)
                            result["options"]["otm_call_iv"]      = (round(otm_call_iv * 100, 1)
                                                                      if otm_call_iv > 0.01 else None)
                            result["options"]["otm_put_strike"]   = otm_put_strike_val
                            result["options"]["otm_call_strike"]  = otm_call_strike_val
                            result["options"]["atm_call_strike"]  = atm_call_strike_val
                except Exception:
                    pass

                # ── 30-day realized volatility (for IV vs HV comparison) ────
                try:
                    hist = tk.history(period="35d")
                    if not hist.empty and len(hist) > 10:
                        rets  = hist["Close"].pct_change().dropna()
                        hv_30 = round(rets.std() * (252 ** 0.5) * 100, 1)
                        result["options"]["hv_30d"] = hv_30
                except Exception:
                    pass

        except Exception:
            pass

        # ── Earnings calendar + beat/miss history ──────────────────────────────
        result["earnings"] = {
            "next_date": None, "days_to_earnings": None,
            "eps_estimate": None, "rev_estimate": None,
            "history": [], "avg_surprise_pct": None,
            "beat_count": 0, "miss_count": 0,
        }
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date", [])
                if not hasattr(dates, "__iter__") or isinstance(dates, str):
                    dates = [dates] if dates else []
                dates = [d for d in dates if d]
                if dates:
                    nd = dates[0]
                    nd = nd.date() if hasattr(nd, "date") else nd
                    nd_str = str(nd)[:10]
                    result["earnings"]["next_date"] = nd_str
                    try:
                        dte_e = (datetime.date.fromisoformat(nd_str)
                                 - datetime.date.today()).days
                        result["earnings"]["days_to_earnings"] = dte_e
                    except Exception:
                        pass
                result["earnings"]["eps_estimate"] = (
                    cal.get("Earnings Average") or cal.get("EPS Trend Current"))
                result["earnings"]["rev_estimate"] = cal.get("Revenue Average")
            elif cal is not None and hasattr(cal, "loc"):
                # Older yfinance returns a DataFrame with metrics as columns
                for col in cal.columns:
                    if "Earnings Date" in str(col):
                        try:
                            nd = cal[col].iloc[0]
                            nd = nd.date() if hasattr(nd, "date") else nd
                            nd_str = str(nd)[:10]
                            result["earnings"]["next_date"] = nd_str
                            dte_e = (datetime.date.fromisoformat(nd_str)
                                     - datetime.date.today()).days
                            result["earnings"]["days_to_earnings"] = dte_e
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            eh = tk.earnings_history
            if eh is not None and not eh.empty:
                rows = []
                for _, row in eh.head(8).iterrows():
                    actual   = (row.get("epsActual")   or row.get("Reported EPS"))
                    estimate = (row.get("epsEstimate")  or row.get("Estimated EPS"))
                    surprise = (row.get("surprisePercent") or row.get("Surprise(%)"))
                    qtr      = str(getattr(row, "name", ""))[:10]
                    if actual is None or estimate is None:
                        continue
                    try:
                        actual   = float(actual)
                        estimate = float(estimate)
                        if surprise is not None:
                            # yfinance returns surprisePercent as a decimal
                            # (e.g. 0.145 = 14.5%). Multiply by 100 to get %.
                            surprise = float(surprise) * 100
                        elif estimate:
                            surprise = (actual - estimate) / abs(estimate) * 100
                        else:
                            surprise = None
                        # Cap at ±50% to prevent near-zero-estimate outliers
                        # from dominating the chart scale
                        if surprise is not None:
                            surprise = round(max(-50.0, min(50.0, surprise)), 1)
                        rows.append({
                            "quarter":      qtr,
                            "actual":       round(actual, 2),
                            "estimate":     round(estimate, 2),
                            "surprise_pct": surprise,
                            "beat":         actual >= estimate,
                        })
                    except (TypeError, ValueError):
                        continue
                result["earnings"]["history"]     = rows
                result["earnings"]["beat_count"]  = sum(1 for r in rows if r["beat"])
                result["earnings"]["miss_count"]  = len(rows) - result["earnings"]["beat_count"]
                surprises = [r["surprise_pct"] for r in rows if r.get("surprise_pct") is not None]
                if surprises:
                    result["earnings"]["avg_surprise_pct"] = round(
                        sum(surprises) / len(surprises), 1)
        except Exception:
            pass

        # ── EPS estimate revision direction ────────────────────────────────────
        try:
            eps_trend_df = tk.eps_trend
            if eps_trend_df is not None and not eps_trend_df.empty:
                if "0q" in eps_trend_df.index:
                    row_0q = eps_trend_df.loc["0q"]
                    cur_val = None
                    ago60_val = None
                    for col in row_0q.index:
                        col_lower = str(col).lower().replace(" ", "").replace("_", "")
                        if col_lower == "current":
                            try:
                                cur_val = float(row_0q[col])
                            except (TypeError, ValueError):
                                pass
                        elif col_lower == "60daysago":
                            try:
                                ago60_val = float(row_0q[col])
                            except (TypeError, ValueError):
                                pass
                    if cur_val is not None and ago60_val is not None and ago60_val != 0:
                        revision = (cur_val - ago60_val) / abs(ago60_val)
                        result["earnings"]["eps_revision_pct"] = round(revision * 100, 1)
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)
    return result

# ── Scoring ────────────────────────────────────────────────────────────────────
def score_news(news_items: list):
    if not news_items:
        return None
    scores = [n["score"] for n in news_items]
    return round(sum(scores) / len(scores), 3)

def score_social(st: dict):
    labeled = (st.get("bullish", 0) or 0) + (st.get("bearish", 0) or 0)
    if not labeled:
        return None
    return round((st["bullish"] - st["bearish"]) / labeled, 3)

def score_short(info: dict):
    pct = info.get("shortPercentOfFloat")
    if pct is None:
        return None
    # 0% → 0 (neutral), 15%+ → -1 (very bearish)
    return round(max(-1.0, -(min(float(pct), 0.30) / 0.15)), 3)

def score_institutional(info: dict):
    pct = info.get("heldPercentInstitutions")
    if pct is None:
        return None
    # Calibrated to large-cap reality: S&P 500 median inst. ownership ~65-75%.
    # 65% = neutral (0.0); 90%+ = +1.0 (unusually high); 40% or below = -1.0.
    # Formula: (pct - 0.65) / 0.25  → neutral at 65%, full bull at 90%, full bear at 40%.
    return round(max(-1.0, min(1.0, (float(pct) - 0.65) / 0.25)), 3)

def score_insider(txns: list):
    if not txns:
        return None
    buy_w = sell_w = 0
    for t in txns:
        tx     = str(t.get("transaction", "")).lower()
        shares = abs(t.get("shares", 0) or 0)
        val    = abs(t.get("value", 0) or 0)
        weight = max(val, shares * 1)
        if "sale" in tx or "sell" in tx:
            sell_w += weight
        elif "purchase" in tx or "buy" in tx or "acqui" in tx:
            buy_w += weight
    total = buy_w + sell_w
    if not total:
        return None
    return round((buy_w - sell_w) / total, 3)

def score_trends(gt: dict, wiki: dict) -> float:
    """Score Google Trends + Wikipedia attention. Returns -1..1 or None."""
    gt_score = None
    wiki_score = None
    slope = gt.get("trend_slope")
    if slope is not None:
        gt_score = max(-1.0, min(1.0, slope * 2))
    wiki_slope = wiki.get("trend_slope")
    if wiki_slope is not None:
        wiki_score = max(-1.0, min(1.0, wiki_slope * 2))
    available = [v for v in [gt_score, wiki_score] if v is not None]
    if not available:
        return None
    return round(sum(available) / len(available), 3)


def score_options(opts: dict):
    pc = opts.get("put_call_ratio")
    pc_score = None
    if pc is not None:
        # PC < 0.7 → bullish; 1.0 → neutral; > 1.4 → bearish
        pc_score = round(max(-1.0, min(1.0, -(float(pc) - 0.7) / 0.7)), 3)

    skew = opts.get("put_skew")
    skew_score = None
    if skew is not None:
        # IV skew ratio: 1.0=flat/neutral, 2.0=extreme fear → -1.0 bearish
        # <1.0 (calls bid up) → slight bullish, cap at +0.3
        skew_score = max(-1.0, min(0.3, -(float(skew) - 1.0) / 1.0))

    if pc_score is not None and skew_score is not None:
        return round(pc_score * 0.7 + skew_score * 0.3, 3)
    if pc_score is not None:
        return pc_score
    if skew_score is not None:
        return round(skew_score, 3)
    return None

def score_analyst(rec_summary: list):
    if not rec_summary:
        return None
    latest = rec_summary[0]
    sb  = latest.get("strongBuy", 0) or 0
    b   = latest.get("buy", 0) or 0
    h   = latest.get("hold", 0) or 0
    s   = latest.get("sell", 0) or 0
    ss  = latest.get("strongSell", 0) or 0
    total = sb + b + h + s + ss
    if not total:
        return None
    return round((sb * 1.0 + b * 0.5 + h * 0 + s * -0.5 + ss * -1.0) / total, 3)

def composite_score(scores: dict) -> float:
    wsum = wt = 0.0
    for k, w in SIGNAL_WEIGHTS.items():
        v = scores.get(k)
        if v is not None:
            wsum += v * w
            wt   += w
    return round(wsum / wt, 3) if wt else 0.0

# ── Per-ticker pipeline ────────────────────────────────────────────────────────
def analyze_ticker(ticker: str, days: int, cik: str) -> dict:
    with _YF_SEM:
        yf_data = fetch_yf(ticker, days)
    reddit_data = fetch_reddit_sentiment(ticker, days)
    congress    = fetch_congressional(ticker, days=max(days, 365))
    sec_8k   = fetch_sec_filings(cik, form_type="8-K",  days=days)
    sec_f4   = fetch_sec_filings(cik, form_type="4",    days=days, limit=12)

    # 13D/13G activist disclosures (up to 1 year)
    sec_13dg = fetch_sec_filings(cik, form_type="SC 13D", days=max(days, 365), limit=5)
    sec_13g  = fetch_sec_filings(cik, form_type="SC 13G", days=max(days, 365), limit=5)
    sec_activist = sorted(sec_13dg + sec_13g, key=lambda x: x["date"], reverse=True)[:8]

    # 13F institutional portfolio snapshots (6 months)
    sec_13f = fetch_sec_filings(cik, form_type="13F-HR", days=180, limit=3)

    info = yf_data["info"]

    # Google Trends and Wikipedia (require company name from info)
    long_name = info.get("longName", "")
    gt_data   = fetch_google_trends(ticker, long_name)
    wiki_data = fetch_wikipedia_views(long_name, ticker, days)

    scores = {
        "news":          score_news(yf_data["news"]),
        "social":        score_social(reddit_data),
        "trends":        score_trends(gt_data, wiki_data),
        "short":         score_short(info),
        "institutional": score_institutional(info),
        "insider":       score_insider(yf_data["insider_txns"]),
        "options":       score_options(yf_data["options"]),
        "analyst":       score_analyst(yf_data["rec_summary"]),
    }
    comp = composite_score(scores)

    # ── Structural metrics ─────────────────────────────────────────────────────
    rd = reddit_data
    labeled = rd.get("bullish", 0) + rd.get("bearish", 0)
    polarization = round(
        min(rd.get("bullish", 0), rd.get("bearish", 0)) / max(labeled, 1), 3
    )
    bullish_signals = sum(1 for v in scores.values() if v is not None and v >= 0.10)
    bearish_signals = sum(1 for v in scores.values() if v is not None and v <= -0.10)
    signal_agreement = bullish_signals if comp >= 0 else bearish_signals
    total_signals = sum(1 for v in scores.values() if v is not None)

    return {
        "ticker":          ticker,
        "name":            info.get("longName", ticker),
        "sector":          info.get("sector", ""),
        "price":           info.get("currentPrice") or info.get("regularMarketPrice"),
        "mktcap":          info.get("marketCap"),
        "scores":          scores,
        "composite":       comp,
        "news":            yf_data["news"],
        "reddit":          reddit_data,
        "institutional":   yf_data["institutional"],
        "insider_txns":    yf_data["insider_txns"],
        "rec_summary":     yf_data["rec_summary"],
        "options":         yf_data["options"],
        "earnings":        yf_data["earnings"],
        "info":            info,
        "congressional":   congress,
        "sec_8k":          sec_8k,
        "sec_f4":          sec_f4,
        "sec_activist":    sec_activist,
        "sec_13f":         sec_13f,
        "gt":              gt_data,
        "wiki":            wiki_data,
        "squeeze":         compute_squeeze(info),
        "news_velocity":   yf_data.get("news_velocity", {}),
        "polarization":    polarization,
        "signal_agreement": signal_agreement,
        "total_signals":   total_signals,
        "error":           yf_data.get("error"),
    }

# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0f14;--surface:#14171f;--sf2:#1c2030;--border:#252a3a;
  --text:#e8eaf2;--muted:#6b7194;--accent:#4f8ef7;
  --up:#00c896;--down:#e05c5c;--warn:#f0a500;--r:8px;
}
html,body{background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:14px;min-height:100%}
/* ── Header ── */
.hdr{background:var(--surface);border-bottom:1px solid var(--border);
  padding:18px 28px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.hdr-title{font-size:22px;font-weight:700}
.hdr-sub{font-size:13px;color:var(--muted)}
.hdr-badge{font-size:11px;padding:3px 10px;border-radius:10px;
  background:#4f8ef720;color:var(--accent);font-weight:600}
/* ── Tabs ── */
.tab-bar{display:flex;background:var(--surface);
  border-bottom:2px solid var(--border);position:sticky;top:0;
  z-index:20;overflow-x:auto}
.tab-btn{background:none;border:none;border-bottom:3px solid transparent;
  padding:14px 22px;color:var(--muted);cursor:pointer;font-size:13px;
  font-weight:600;white-space:nowrap;transition:color .15s,border-color .15s}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-btn:hover:not(.active){color:var(--text)}
.container{max-width:1600px;margin:0 auto;padding:24px 20px 60px}
.tab-panel{display:none}.tab-panel.active{display:block}
/* ── Cards ── */
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:20px;margin-bottom:20px}
.card-title{font-size:11px;font-weight:700;color:var(--muted);
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:16px}
/* ── Summary stat cards ── */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
  gap:12px;margin-bottom:20px}
.stat-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:16px;text-align:center}
.stat-val{font-size:28px;font-weight:700}
.stat-lbl{font-size:11px;color:var(--muted);margin-top:4px;
  text-transform:uppercase;letter-spacing:.04em}
/* ── Tables ── */
.tbl-wrap{overflow-x:auto;border-radius:var(--r);
  border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--sf2);color:var(--muted);font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:.06em;padding:9px 12px;
  text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
tbody tr{border-bottom:1px solid var(--border)}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--sf2)}
td{padding:9px 12px;vertical-align:middle}
td.num{text-align:right;font-family:monospace;font-size:12px}
td.ticker-cell{font-weight:700;font-size:13px;color:var(--accent)}
/* ── Chart layouts ── */
.chart-2col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.chart-3col{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.radar-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}
.radar-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:16px}
.radar-card h3{font-size:14px;font-weight:700;margin-bottom:2px}
.radar-card p{font-size:11px;color:var(--muted);margin-bottom:10px}
.radar-canvas{height:210px;position:relative}
.chart-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:20px}
.chart-card h3{font-size:13px;font-weight:700;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:16px}
.chart-canvas{position:relative}
/* ── Headlines ── */
.news-list{list-style:none}
.news-item{padding:10px 0;border-bottom:1px solid var(--border);
  display:flex;gap:10px;align-items:flex-start}
.news-item:last-child{border-bottom:none}
.news-body{flex:1}
.news-title a{color:var(--text);text-decoration:none;font-size:13px;font-weight:500}
.news-title a:hover{color:var(--accent)}
.news-meta{font-size:11px;color:var(--muted);margin-top:3px}
/* ── Reddit posts ── */
.twit{padding:9px 0;border-bottom:1px solid var(--border);font-size:12px}
.twit:last-child{border-bottom:none}
/* ── Donut legend ── */
.donut-wrap{display:flex;gap:20px;align-items:center}
.donut-legend{display:flex;flex-direction:column;gap:8px;font-size:12px}
.legend-row{display:flex;align-items:center;gap:8px}
.legend-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
/* ── Section header ── */
.sec-hdr{font-size:16px;font-weight:700;margin:28px 0 14px;
  padding-bottom:8px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:4px}
.ticker-tag{font-size:11px;font-weight:700;color:var(--accent);
  background:#4f8ef720;padding:3px 10px;border-radius:10px;
  display:inline-block;margin-bottom:10px}
/* ── Misc ── */
.up{color:var(--up)}.down{color:var(--down)}.warn{color:var(--warn)}
.muted{color:var(--muted)}.note{font-size:11px;color:var(--muted);font-style:italic}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}
@media(max-width:800px){
  .chart-2col,.chart-3col,.two-col,.three-col{grid-template-columns:1fr}
}
/* ── Help button ── */
.help-btn{background:none;border:none;color:var(--muted);font-size:13px;
  cursor:pointer;padding:0 0 0 6px;line-height:1;vertical-align:middle;
  opacity:.7;transition:opacity .15s,color .15s}
.help-btn:hover{opacity:1;color:var(--accent)}
/* ── Help popover ── */
#help-pop{display:none;position:fixed;z-index:9999;
  background:var(--surface);border:1px solid var(--accent);
  border-radius:var(--r);padding:18px 18px 14px;
  max-width:400px;width:min(400px,90vw);
  box-shadow:0 8px 40px rgba(0,0,0,.65);
  font-size:12.5px;line-height:1.6;color:var(--text)}
#help-pop-title{font-weight:700;font-size:14px;color:var(--accent);
  margin-bottom:10px;padding-bottom:8px;
  border-bottom:1px solid var(--border)}
#help-pop-body b{color:var(--text)}
#help-pop-close{position:absolute;top:10px;right:12px;
  background:none;border:none;color:var(--muted);
  font-size:20px;line-height:1;cursor:pointer;padding:0}
#help-pop-close:hover{color:var(--text)}
"""

# ── HTML tab renderers ─────────────────────────────────────────────────────────
def _tab_overview(all_data: dict) -> str:
    tickers = list(all_data.keys())
    bullish = sum(1 for d in all_data.values() if d["composite"] >= 0.10)
    neutral = sum(1 for d in all_data.values() if -0.10 <= d["composite"] < 0.10)
    bearish = sum(1 for d in all_data.values() if d["composite"] < -0.10)
    avg     = (sum(d["composite"] for d in all_data.values()) / len(all_data)
               if all_data else 0)

    h  = '<div class="stat-grid">'
    h += (f'<div class="stat-card"><div class="stat-val up">{bullish}</div>'
          f'<div class="stat-lbl">Bullish</div></div>')
    h += (f'<div class="stat-card"><div class="stat-val warn">{neutral}</div>'
          f'<div class="stat-lbl">Neutral</div></div>')
    h += (f'<div class="stat-card"><div class="stat-val down">{bearish}</div>'
          f'<div class="stat-lbl">Bearish</div></div>')
    avg_color = sentiment_color(avg)
    h += (f'<div class="stat-card"><div class="stat-val" style="color:{avg_color}">'
          f'{avg:+.2f}</div><div class="stat-lbl">Avg Score</div></div>')

    # Confirmed signals stat card
    total_sig_vals = [d.get("total_signals", 0) for d in all_data.values()]
    agree_vals     = [d.get("signal_agreement", 0) for d in all_data.values()]
    avg_conf = (round(
        sum(agree_vals[i] / max(total_sig_vals[i], 1) * 100
            for i in range(len(agree_vals))) / max(len(agree_vals), 1), 0
    ) if agree_vals else 0)
    avg_conf_int = int(avg_conf)
    conf_color = "#00c896" if avg_conf_int >= 60 else "#f0a500" if avg_conf_int >= 40 else "#e05c5c"
    h += (f'<div class="stat-card"><div class="stat-val" style="color:{conf_color}">'
          f'{avg_conf_int}%</div><div class="stat-lbl">Signal Confirmation</div></div>')
    h += '</div>'

    # Composite scores chart card
    h += '<div class="chart-card" style="margin-bottom:20px">'
    h += f'<h3>Composite Sentiment Score {help_btn("composite-score")}</h3>'
    bar_h = max(180, len(tickers) * 38)
    h += f'<div class="chart-canvas" style="height:{bar_h}px">'
    h += '<canvas id="chart-composite"></canvas></div></div>'

    # Signal heatmap table
    h += f'<div class="card"><div class="card-title">Signal Heatmap {help_btn("signal-heatmap")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Ticker</th><th>Composite</th>'
    for k in SIGNAL_WEIGHTS:
        h += f'<th>{SIGNAL_LABELS[k]}</th>'
    h += '</tr></thead><tbody>'
    for ticker, d in all_data.items():
        comp_color = sentiment_color(d["composite"])
        h += f'<tr><td class="ticker-cell">{ticker}</td>'
        h += (f'<td><span style="color:{comp_color};font-weight:700;">'
              f'{d["composite"]:+.2f}</span>&nbsp;'
              f'{score_bar(d["composite"])}</td>')
        for k in SIGNAL_WEIGHTS:
            s = d["scores"].get(k)
            c = sentiment_color(s)
            sv = f'{s:+.2f}' if s is not None else '—'
            h += (f'<td style="color:{c};font-weight:600;white-space:nowrap;">'
                  f'{sv}&nbsp;{score_bar(s, 60)}</td>')
        h += '</tr>'
    h += '</tbody></table></div></div>'

    # Per-ticker radar charts
    h += f'<div class="card"><div class="card-title">Signal Breakdown \u2014 Radar Charts {help_btn("radar-charts")}</div>'
    h += '<div class="radar-grid">'
    for ticker, d in all_data.items():
        comp_color = sentiment_color(d["composite"])
        label      = sentiment_label(d["composite"])
        h += f'<div class="radar-card">'
        h += (f'<h3>{ticker} &nbsp;<span style="color:{comp_color};font-size:12px">'
              f'{label}</span></h3>')
        h += f'<p style="color:var(--muted);font-size:11px">{d["name"]}</p>'
        h += f'<div class="radar-canvas"><canvas id="radar-{ticker}"></canvas></div>'
        h += '</div>'
    h += '</div></div>'

    # ── Attention & Polarization table ────────────────────────────────────────
    h += f'<div class="sec-hdr">Attention &amp; Polarization {help_btn("attention-polarization")}</div>'
    h += '<div class="card"><div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Google Trends Slope</th>'
          '<th>Wikipedia Avg Views</th><th>Reddit Polarization</th>'
          '<th>Squeeze Potential</th><th>Signal Confirmation</th></tr></thead><tbody>')
    for ticker, d in all_data.items():
        gt   = d.get("gt", {})
        wiki = d.get("wiki", {})
        sq   = d.get("squeeze", {})
        pol  = d.get("polarization", 0.0)
        sig_agree = d.get("signal_agreement", 0)
        tot_sig   = d.get("total_signals", 0)

        gt_slope  = gt.get("trend_slope")
        gt_now    = gt.get("interest_now")
        if gt_slope is not None:
            gt_arrow = "&#8593;" if gt_slope > 0.05 else ("&#8595;" if gt_slope < -0.05 else "&rarr;")
            gt_color = "#00c896" if gt_slope > 0.05 else ("#e05c5c" if gt_slope < -0.05 else "#f0a500")
            gt_str   = (f'<span style="color:{gt_color};font-weight:700">{gt_arrow} '
                        f'{gt_slope:+.2f}</span>')
            if gt_now is not None:
                gt_interest_str = str(int(gt_now))
                gt_str += f' <span class="muted" style="font-size:10px">({gt_interest_str}/100)</span>'
        else:
            gt_str = '<span class="muted">—</span>'

        wiki_avg = wiki.get("avg_daily")
        wiki_slope = wiki.get("trend_slope")
        if wiki_avg is not None:
            wiki_arrow = "&#8593;" if (wiki_slope and wiki_slope > 0.05) else ("&#8595;" if (wiki_slope and wiki_slope < -0.05) else "&rarr;")
            wiki_color = "#00c896" if (wiki_slope and wiki_slope > 0.05) else ("#e05c5c" if (wiki_slope and wiki_slope < -0.05) else "#f0a500")
            wiki_avg_str = f"{wiki_avg:,}"
            wiki_str = (f'{wiki_avg_str} <span style="color:{wiki_color}">{wiki_arrow}</span>')
        else:
            wiki_str = '<span class="muted">—</span>'

        reddit_total = d.get("reddit", {}).get("total", 0)
        if reddit_total == 0:
            pol_str = '<span class="muted">— <span style="font-size:10px">(no posts)</span></span>'
        else:
            pol_pct   = round(pol * 100)
            pol_color = "#f0a500" if pol > 0.35 else "#00c896" if pol < 0.15 else "#6b7194"
            pol_label = "Split" if pol > 0.35 else "Consensus" if pol < 0.15 else "Mixed"
            pol_str   = f'<span style="color:{pol_color}">{pol_pct}% ({pol_label})</span>'

        sq_level = sq.get("level", "—")
        sq_score = sq.get("score", 0)
        sq_color = "#e05c5c" if sq_level == "Extreme" else ("#f0a500" if sq_level == "High" else ("#cddc39" if sq_level == "Moderate" else "#00c896"))
        sq_str   = f'<span style="color:{sq_color};font-weight:700">{sq_level}</span> ({sq_score})'

        conf_pct = round(sig_agree / max(tot_sig, 1) * 100)
        conf_color2 = "#00c896" if conf_pct >= 60 else "#f0a500" if conf_pct >= 40 else "#e05c5c"
        sig_agree_str = str(sig_agree)
        tot_sig_str   = str(tot_sig)
        conf_str = (f'<span style="color:{conf_color2}">{conf_pct}%</span>'
                    f' <span class="muted" style="font-size:10px">({sig_agree_str}/{tot_sig_str})</span>')

        h += (f'<tr><td class="ticker-cell">{ticker}</td>'
              f'<td>{gt_str}</td>'
              f'<td>{wiki_str}</td>'
              f'<td>{pol_str}</td>'
              f'<td>{sq_str}</td>'
              f'<td>{conf_str}</td></tr>')
    h += '</tbody></table></div></div>'
    return h


def _tab_news_social(all_data: dict) -> str:
    h = ""
    for ticker, d in all_data.items():
        h += f'<div class="sec-hdr"><span class="ticker-tag">{ticker}</span> &nbsp;{d["name"]}</div>'
        h += '<div class="two-col">'

        # News column
        h += f'<div class="card"><div class="card-title">Recent News {help_btn("recent-news")}</div>'
        news = d.get("news", [])
        if news:
            h += '<ul class="news-list">'
            for n in news[:12]:
                sc    = n["score"]
                tc    = sentiment_color(sc)
                ts    = (datetime.datetime.fromtimestamp(n["time"]).strftime("%b %d, %Y")
                         if n["time"] else "")
                h += f'<li class="news-item">'
                h += (f'<div style="width:10px;flex-shrink:0;border-radius:2px;'
                      f'background:{tc};opacity:.8;margin-top:3px;height:14px;"></div>')
                h += f'<div class="news-body">'
                link = n.get("link", "")
                if link:
                    h += f'<div class="news-title"><a href="{link}" target="_blank">{n["title"]}</a></div>'
                else:
                    h += f'<div class="news-title">{n["title"]}</div>'
                h += f'<div class="news-meta">{n["publisher"]} · {ts} · Score: <span style="color:{tc}">{sc:+.2f}</span></div>'
                h += '</div></li>'
            h += '</ul>'
        else:
            h += '<p class="muted note">No news data available.</p>'
        h += '</div>'

        # Social / Reddit column
        rd    = d.get("reddit", {})
        total = rd.get("total", 0)
        h += f'<div class="card"><div class="card-title">Reddit Sentiment {help_btn("reddit-sentiment")}</div>'
        if total:
            bull = rd.get("bullish", 0)
            bear = rd.get("bearish", 0)
            neut = rd.get("neutral", 0)
            bull_pct = round(bull / total * 100)
            bear_pct = round(bear / total * 100)
            neut_pct = 100 - bull_pct - bear_pct
            h += '<div class="donut-wrap" style="margin-bottom:16px">'
            h += f'<div style="width:140px;height:140px;position:relative;flex-shrink:0"><canvas id="donut-{ticker}"></canvas></div>'
            h += '<div class="donut-legend">'
            h += f'<div class="legend-row"><div class="legend-dot" style="background:#00c896"></div>{bull} Bullish ({bull_pct}%)</div>'
            h += f'<div class="legend-row"><div class="legend-dot" style="background:#e05c5c"></div>{bear} Bearish ({bear_pct}%)</div>'
            h += f'<div class="legend-row"><div class="legend-dot" style="background:#6b7194"></div>{neut} Neutral ({neut_pct}%)</div>'
            h += f'<div style="font-size:11px;color:var(--muted);margin-top:4px">Based on {total} posts · r/wallstreetbets, r/stocks, r/investing</div>'
            h += '</div></div>'
            msgs = rd.get("messages", [])
            if msgs:
                h += '<div style="border-top:1px solid var(--border);padding-top:10px">'
                for m in msgs:
                    sent      = m.get("sentiment", "")
                    date_str  = m.get("created_at", "")[:10]
                    sub       = m.get("subreddit", "")
                    ups       = m.get("upvotes", 0)
                    url       = m.get("url", "")
                    badge_html = ""
                    if sent == "Bullish":
                        badge_html = '<span class="twit-sent twit-bull">▲ Bullish</span>'
                    elif sent == "Bearish":
                        badge_html = '<span class="twit-sent twit-bear">▼ Bearish</span>'
                    sub_tag = f'<span style="color:var(--accent);font-size:10px">r/{sub}</span>' if sub else ""
                    ups_tag = f'<span class="muted" style="font-size:10px">▲ {ups}</span>' if ups else ""
                    title_html = (f'<a href="{url}" target="_blank" style="color:var(--text);text-decoration:none">{m["body"]}</a>'
                                  if url else m["body"])
                    h += (f'<div class="twit">{title_html} {badge_html}'
                          f'<div style="margin-top:3px">{sub_tag} {ups_tag}'
                          f' <span class="muted" style="font-size:10px">{date_str}</span></div></div>')
                h += '</div>'
        else:
            h += '<p class="muted note">No Reddit posts found for this ticker in the selected date range.</p>'
        h += '</div>'
        h += '</div>'  # two-col

        # ── News Velocity ──────────────────────────────────────────────────────
        nv = d.get("news_velocity", {})
        r7   = nv.get("recent_7d", 0)
        p7   = nv.get("prior_7d", 0)
        vr   = nv.get("velocity_ratio", 1.0)
        if vr > 1.3:
            vel_arrow = "&#8593;"
            vel_color = "#00c896"
            vel_label = "Accelerating"
        elif vr < 0.8:
            vel_arrow = "&#8595;"
            vel_color = "#e05c5c"
            vel_label = "Slowing"
        else:
            vel_arrow = "&rarr;"
            vel_color = "#f0a500"
            vel_label = "Stable"
        vr_str  = f"{vr:.2f}"
        r7_str  = str(r7)
        p7_str  = str(p7)
        h += '<div class="card" style="margin-bottom:16px">'
        h += f'<div class="card-title">News Velocity {help_btn("news-velocity")}</div>'
        h += (f'<div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap">'
              f'<div><span class="muted" style="font-size:11px">Last 7d</span>'
              f'<div style="font-size:22px;font-weight:700">{r7_str}</div></div>'
              f'<div><span class="muted" style="font-size:11px">Prior 7–14d</span>'
              f'<div style="font-size:22px;font-weight:700">{p7_str}</div></div>'
              f'<div><span class="muted" style="font-size:11px">Velocity Ratio</span>'
              f'<div style="font-size:22px;font-weight:700;color:{vel_color}">'
              f'{vel_arrow} {vr_str}</div>'
              f'<div style="font-size:11px;color:{vel_color}">{vel_label}</div></div>'
              f'</div>')
        h += '</div>'

        # ── Search & Attention ─────────────────────────────────────────────────
        gt   = d.get("gt", {})
        wiki = d.get("wiki", {})
        h += '<div class="two-col" style="margin-bottom:16px">'

        # Google Trends card
        h += f'<div class="card"><div class="card-title">Google Trends {help_btn("google-trends")}</div>'
        if not PYTRENDS_AVAILABLE:
            h += '<p class="note">Install pytrends for Google Trends data: pip install pytrends</p>'
        elif gt.get("trend_slope") is None:
            h += '<p class="note muted">No Google Trends data available for this ticker.</p>'
        else:
            gt_slope_val  = gt["trend_slope"]
            gt_now_val    = gt.get("interest_now")
            gt_avg_val    = gt.get("interest_avg")
            gt_kw_val     = gt.get("kw", "")
            gt_sc_val     = score_trends(gt, {})
            gt_arrow2     = "&#8593;" if gt_slope_val > 0.05 else ("&#8595;" if gt_slope_val < -0.05 else "&rarr;")
            gt_color2     = "#00c896" if gt_slope_val > 0.05 else ("#e05c5c" if gt_slope_val < -0.05 else "#f0a500")
            gt_slope_str  = f"{gt_slope_val:+.2f}"
            gt_now_str    = str(int(gt_now_val)) if gt_now_val is not None else "—"
            gt_avg_str    = str(int(gt_avg_val)) if gt_avg_val is not None else "—"
            h += (f'<div style="margin-bottom:10px">'
                  f'{sentiment_badge(gt_sc_val)}'
                  f' <span style="color:{gt_color2};font-weight:700;margin-left:8px">'
                  f'{gt_arrow2} Slope: {gt_slope_str}</span></div>')
            h += (f'<div style="font-size:12px;color:var(--muted)">'
                  f'Interest Now: <strong style="color:var(--text)">{gt_now_str}/100</strong>'
                  f' &nbsp;|&nbsp; Baseline Avg: <strong style="color:var(--text)">{gt_avg_str}/100</strong></div>')
            h += f'<div style="font-size:11px;color:var(--muted);margin-top:4px">Query: {gt_kw_val} · Finance category · US · 3 months</div>'
        h += '</div>'

        # Wikipedia card
        h += f'<div class="card"><div class="card-title">Wikipedia Page Views {help_btn("wikipedia-views")}</div>'
        if wiki.get("avg_daily") is None:
            h += '<p class="note muted">No Wikipedia data available.</p>'
        else:
            wiki_avg_val   = wiki["avg_daily"]
            wiki_peak_val  = wiki.get("peak_daily", 0)
            wiki_slope_val = wiki.get("trend_slope", 0)
            wiki_page_val  = wiki.get("page", "")
            wiki_arrow3    = "&#8593;" if wiki_slope_val > 0.05 else ("&#8595;" if wiki_slope_val < -0.05 else "&rarr;")
            wiki_color3    = "#00c896" if wiki_slope_val > 0.05 else ("#e05c5c" if wiki_slope_val < -0.05 else "#f0a500")
            wiki_avg_str2  = f"{wiki_avg_val:,}"
            wiki_peak_str  = f"{wiki_peak_val:,}"
            wiki_slope_str = f"{wiki_slope_val:+.2f}"
            h += (f'<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:8px">'
                  f'<div><span class="muted" style="font-size:11px">Avg Daily</span>'
                  f'<div style="font-size:20px;font-weight:700">{wiki_avg_str2}</div></div>'
                  f'<div><span class="muted" style="font-size:11px">Peak</span>'
                  f'<div style="font-size:20px;font-weight:700">{wiki_peak_str}</div></div>'
                  f'<div><span class="muted" style="font-size:11px">Trend</span>'
                  f'<div style="font-size:20px;font-weight:700;color:{wiki_color3}">'
                  f'{wiki_arrow3} {wiki_slope_str}</div></div>'
                  f'</div>')
            if wiki_page_val:
                encoded_page = quote_plus(wiki_page_val)
                wiki_link = "https://en.wikipedia.org/wiki/" + encoded_page
                h += f'<div style="font-size:11px;color:var(--muted)">Page: <a href="{wiki_link}" target="_blank" style="color:var(--accent)">{wiki_page_val.replace("_"," ")}</a></div>'
        h += '</div>'
        h += '</div>'  # two-col for attention

    # Google Trends multi-line chart
    h += f'<div class="sec-hdr">Google Trends \u2014 12-Week Search Interest {help_btn("google-trends")}</div>'
    h += '<div class="chart-card" style="margin-bottom:20px">'
    h += '<h3>Search Interest Over Time (0–100 Index)</h3>'
    h += '<div class="chart-canvas" style="height:240px"><canvas id="chart-trends"></canvas></div>'
    h += '<p class="note" style="margin-top:8px">Finance category · US · weekly · pytrends required</p>'
    h += '</div>'

    # Wikipedia views multi-line chart
    h += f'<div class="sec-hdr">Wikipedia Page Views {help_btn("wikipedia-views")}</div>'
    h += '<div class="chart-card" style="margin-bottom:20px">'
    h += '<h3>Daily Wikipedia Page Views</h3>'
    h += '<div class="chart-canvas" style="height:240px"><canvas id="chart-wiki"></canvas></div>'
    h += '<p class="note" style="margin-top:8px">Source: Wikimedia REST API · all-access · all-agents</p>'
    h += '</div>'
    return h


def _tab_smart_money(all_data: dict) -> str:
    h = ""

    # ── Short Interest table ────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Short Interest {help_btn("short-interest")}</div>'
    h += '<div class="card"><div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Signal</th><th>% of Float</th>'
          '<th>Short Ratio (Days to Cover)</th><th>Shares Short</th>'
          '<th>Prev Month</th><th>Change</th></tr></thead><tbody>')
    for ticker, d in all_data.items():
        info = d.get("info", {})
        pct  = info.get("shortPercentOfFloat")
        rat  = info.get("shortRatio")
        sh   = info.get("sharesShort")
        shp  = info.get("sharesShortPriorMonth")
        sc   = d["scores"].get("short")
        chg  = None
        if sh and shp and shp > 0:
            chg = (sh - shp) / shp * 100
        h += f'<tr><td class="ticker-cell">{ticker}</td>'
        h += f'<td>{sentiment_badge(sc)}</td>'
        h += f'<td class="num">{fmt_pct(pct * 100 if pct else None)}</td>'
        h += f'<td class="num">{f"{rat:.1f}d" if rat else "—"}</td>'
        h += f'<td class="num">{fmt_money(sh)}</td>'
        h += f'<td class="num">{fmt_money(shp)}</td>'
        if chg is not None:
            cc = "#e05c5c" if chg > 0 else "#00c896"
            h += f'<td class="num" style="color:{cc}">{fmt_pct(chg, signed=True)}</td>'
        else:
            h += '<td class="num muted">—</td>'
        h += '</tr>'
    h += '</tbody></table></div></div>'

    # ── Institutional Holdings ──────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Institutional Holdings {help_btn("institutional-holdings")}</div>'
    h += '<div class="two-col">'

    # Summary table
    h += f'<div class="card"><div class="card-title">Ownership Summary {help_btn("institutional-holdings")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Ticker</th><th>Signal</th><th>% Institutional</th><th>% Insider</th><th>Debt/Equity</th><th>Credit Proxy</th></tr></thead><tbody>'
    for ticker, d in all_data.items():
        info  = d.get("info", {})
        ih    = info.get("heldPercentInstitutions")
        ii    = info.get("heldPercentInsiders")
        de    = info.get("debtToEquity")
        cr    = info.get("currentRatio")
        sc    = d["scores"].get("institutional")
        de_color = "#e05c5c" if de and de > 200 else ("#f0a500" if de and de > 100 else "#00c896")
        cr_color = "#e05c5c" if cr and cr < 1.0 else ("#f0a500" if cr and cr < 1.5 else "#00c896")
        h += f'<tr><td class="ticker-cell">{ticker}</td>'
        h += f'<td>{sentiment_badge(sc)}</td>'
        h += f'<td class="num">{fmt_pct(ih * 100 if ih else None)}</td>'
        h += f'<td class="num">{fmt_pct(ii * 100 if ii else None)}</td>'
        h += f'<td class="num" style="color:{de_color}">{f"{de:.0f}%" if de else "—"}</td>'
        h += f'<td class="num" style="color:{cr_color}">{f"{cr:.2f}x" if cr else "—"}</td>'
        h += '</tr>'
    h += '</tbody></table></div></div>'

    # Top holders table
    h += f'<div class="card"><div class="card-title">Top Institutional Holders {help_btn("top-holders")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Ticker</th><th>Institution</th><th>% Held</th><th>Value</th></tr></thead><tbody>'
    for ticker, d in all_data.items():
        for row in d.get("institutional", []):
            pct = row.get("pct")
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td>{row.get("holder", "")}</td>'
                  f'<td class="num">{fmt_pct(float(pct) * 100 if pct else None)}</td>'
                  f'<td class="num">{fmt_money(row.get("value"))}</td></tr>')
    h += '</tbody></table></div></div>'
    h += '</div>'  # two-col

    # ── Insider Transactions ────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Insider Transactions (yfinance) {help_btn("insider-transactions")}</div>'
    h += '<div class="card"><div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Signal</th><th>Insider</th>'
          '<th>Position</th><th>Transaction</th><th>Shares</th>'
          '<th>Value</th><th>Date</th></tr></thead><tbody>')
    any_insider = False
    for ticker, d in all_data.items():
        txns = d.get("insider_txns", [])
        sc   = d["scores"].get("insider")
        for t in txns:
            any_insider = True
            tx_lower = str(t.get("transaction", "")).lower()
            if "sale" in tx_lower or "sell" in tx_lower:
                tc = "#e05c5c"
            elif "purchase" in tx_lower or "buy" in tx_lower or "acqui" in tx_lower:
                tc = "#00c896"
            else:
                tc = "#6b7194"
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td>{sentiment_badge(sc)}</td>'
                  f'<td>{t.get("insider", "")}</td>'
                  f'<td class="muted">{t.get("position", "")}</td>'
                  f'<td style="color:{tc};font-weight:600">{t.get("transaction", "")}</td>'
                  f'<td class="num">{t.get("shares", "—"):,}</td>'
                  f'<td class="num">{fmt_money(t.get("value"))}</td>'
                  f'<td class="muted">{t.get("date", "")}</td></tr>')
    if not any_insider:
        h += '<tr><td colspan="8" class="muted note" style="padding:16px">No insider transactions found in the selected window.</td></tr>'
    h += '</tbody></table></div></div>'

    # ── Short Squeeze Potential ─────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Short Squeeze Potential {help_btn("squeeze-potential")}</div>'
    h += '<div class="card"><div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Short % Float</th><th>Days to Cover</th>'
          '<th>Squeeze Score</th><th>Level</th></tr></thead><tbody>')
    for ticker, d in all_data.items():
        sq = d.get("squeeze", {})
        si_pct  = sq.get("si_float_pct", 0)
        dtc     = sq.get("dtc", 0)
        sq_sc   = sq.get("score", 0)
        sq_lv   = sq.get("level", "Low")
        lv_color = ("#e05c5c" if sq_lv == "Extreme"
                    else "#f0a500" if sq_lv == "High"
                    else "#cddc39" if sq_lv == "Moderate"
                    else "#00c896")
        si_pct_str = f"{si_pct:.1f}%"
        dtc_str    = f"{dtc:.1f}d"
        sq_sc_str  = f"{sq_sc:.1f}"
        h += (f'<tr><td class="ticker-cell">{ticker}</td>'
              f'<td class="num">{si_pct_str}</td>'
              f'<td class="num">{dtc_str}</td>'
              f'<td class="num" style="font-weight:700">{sq_sc_str}</td>'
              f'<td style="color:{lv_color};font-weight:700">{sq_lv}</td></tr>')
    h += '</tbody></table></div></div>'

    # ── Congressional Trading ───────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Congressional Trading (STOCK Act Disclosures) {help_btn("congressional-trading")}</div>'
    h += '<div class="card">'
    h += '<p class="note" style="margin-bottom:12px">Legally required disclosures by members of Congress under the STOCK Act. House and Senate sources.</p>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Chamber</th><th>Member</th>'
          '<th>Party</th><th>Transaction</th><th>Amount</th><th>Date</th></tr></thead><tbody>')
    any_congress = False
    for ticker, d in all_data.items():
        for c in d.get("congressional", []):
            any_congress = True
            tx   = str(c.get("type", ""))
            tc   = "#00c896" if "purchase" in tx.lower() else "#e05c5c" if "sale" in tx.lower() else "#6b7194"
            party = c.get("party", "")
            pc    = "#4f8ef7" if party == "D" else "#e05c5c" if party == "R" else "#6b7194"
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td>{c["chamber"]}</td>'
                  f'<td>{c.get("member", "")}</td>'
                  f'<td style="color:{pc};font-weight:600">{party}</td>'
                  f'<td style="color:{tc};font-weight:600">{tx}</td>'
                  f'<td class="num">{c.get("amount", "")}</td>'
                  f'<td class="muted">{c.get("date", "")}</td></tr>')
    if not any_congress:
        h += ('<tr><td colspan="7" class="muted note" style="padding:16px">'
              'No congressional trades found for these tickers in the selected window.</td></tr>')
    h += '</tbody></table></div></div>'
    return h


def _tab_options_analyst(all_data: dict) -> str:
    h = ""

    # ── Options overview ────────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Options Market Sentiment {help_btn("pcr-iv")}</div>'
    h += '<div class="two-col">'

    h += f'<div class="card"><div class="card-title">Put/Call Ratio &amp; Implied Volatility {help_btn("pcr-iv")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Signal</th><th>Put/Call Ratio</th>'
          '<th>Calls Vol</th><th>Puts Vol</th><th>ATM IV</th>'
          '<th>Nearest Expiry</th></tr></thead><tbody>')
    for ticker, d in all_data.items():
        opts = d.get("options", {})
        sc   = d["scores"].get("options")
        pc   = opts.get("put_call_ratio")
        iv   = opts.get("implied_volatility")
        pc_color = "#00c896" if (pc and pc < 0.7) else "#e05c5c" if (pc and pc > 1.0) else "#f0a500"
        h += (f'<tr><td class="ticker-cell">{ticker}</td>'
              f'<td>{sentiment_badge(sc)}</td>'
              f'<td class="num" style="color:{pc_color};font-weight:700">'
              f'{f"{pc:.2f}" if pc else "—"}</td>'
              f'<td class="num">{fmt_money(opts.get("calls_volume"))}</td>'
              f'<td class="num">{fmt_money(opts.get("puts_volume"))}</td>'
              f'<td class="num">{fmt_pct(iv * 100 if iv else None)}</td>'
              f'<td class="muted">{opts.get("expiry", "—")}</td></tr>')
    h += '</tbody></table></div></div>'

    h += '<div class="chart-card"><h3>Put/Call Ratio by Ticker</h3>'
    h += '<div class="chart-canvas" style="height:220px"><canvas id="chart-pc"></canvas></div>'
    h += '<p class="note" style="margin-top:8px">Below 0.7 = bullish · 0.7–1.0 = neutral · Above 1.0 = bearish</p>'
    h += '</div>'
    h += '</div>'  # two-col

    # ── Market-Implied Move ─────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Market-Implied Move (ATM Straddle) {help_btn("implied-move")}</div>'
    h += '<div class="card">'
    h += ('<p class="note" style="margin-bottom:14px">'
          'The implied move is derived from the at-the-money (ATM) straddle price '
          '(call mid + put mid) for the nearest expiry. It represents the ± price '
          'move the options market is pricing in by expiration. '
          'IV/HV &gt; 1 means options are expensive relative to recent realized '
          'volatility; &lt; 1 means they are cheap.</p>')
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Price</th><th>Expiry</th><th>DTE</th>'
          '<th>Earnings In</th><th>ATM Strike</th><th>Straddle</th>'
          '<th>Implied Move ±%</th><th>Expected Range</th>'
          '<th>ATM IV</th><th>30d HV</th><th>IV / HV</th></tr></thead><tbody>')
    for ticker, d in all_data.items():
        opts       = d.get("options", {})
        spot       = opts.get("spot") or d.get("price")
        impl       = opts.get("implied_move_pct")
        hi         = opts.get("expected_upper")
        lo         = opts.get("expected_lower")
        iv         = opts.get("implied_volatility")
        hv         = opts.get("hv_30d")
        atm_s      = opts.get("atm_strike")
        strad      = opts.get("straddle_price")
        ivhv       = round(iv * 100 / hv, 2) if (iv and hv and hv > 0) else None
        ivhv_color = ("#e05c5c" if (ivhv and ivhv > 1.3)
                      else "#f0a500" if (ivhv and ivhv > 1.0)
                      else "#00c896" if ivhv else "#6b7194")
        spot_str   = f"${spot:,.2f}"   if spot  else "—"
        atm_str    = f"${atm_s:,.2f}"  if atm_s else "—"
        strad_str  = f"${strad:.2f}"   if strad else "—"
        impl_str   = (f"\u00b1{impl}%"
                      if impl and impl <= 40 else
                      f'<span style="color:#e05c5c" title="Likely bad data — illiquid options">'
                      f'\u00b1{impl}% \u26a0</span>'
                      if impl else "—")
        range_str  = f"${lo:,.2f} \u2013 ${hi:,.2f}" if (lo and hi) else "—"
        ivhv_str   = f"{ivhv:.2f}x"   if ivhv  else "—"
        expiry    = opts.get("expiry", "—")
        dte       = opts.get("dte", "—")
        earn      = d.get("earnings", {})
        dte_earn  = earn.get("days_to_earnings")
        earn_str  = f"{dte_earn}d" if dte_earn is not None else "—"
        # Flag if earnings fall within the option window (straddle is pricing in earnings)
        earn_flag = ""
        if dte_earn is not None and isinstance(dte, int) and dte_earn <= dte:
            earn_flag = ' <span style="color:#f0a500;font-size:11px" title="Earnings within expiry window">⚡</span>'
        elif dte_earn is not None and dte_earn <= 7:
            earn_flag = ' <span style="color:#f0a500;font-size:11px" title="Earnings this week">⚡</span>'
        h += (f'<tr><td class="ticker-cell">{ticker}</td>'
              f'<td class="num">{spot_str}</td>'
              f'<td class="muted">{expiry}</td>'
              f'<td class="num">{dte}</td>'
              f'<td class="num">{earn_str}{earn_flag}</td>'
              f'<td class="num">{atm_str}</td>'
              f'<td class="num">{strad_str}</td>'
              f'<td class="num" style="font-weight:700;color:#4f8ef7">{impl_str}</td>'
              f'<td class="num" style="font-size:11px">{range_str}</td>'
              f'<td class="num">{fmt_pct(iv * 100 if iv else None)}</td>'
              f'<td class="num">{fmt_pct(hv)}</td>'
              f'<td class="num" style="color:{ivhv_color};font-weight:700">{ivhv_str}</td></tr>')
    h += '</tbody></table></div></div>'

    # ── Volatility Skew ──────────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Volatility Skew {help_btn("vol-skew")}</div>'
    h += '<div class="card">'
    h += ('<p class="note" style="margin-bottom:14px">'
          'IV Skew = OTM put implied volatility (~88% of spot) / ATM call implied volatility. '
          'A ratio &gt;1.5 means the market is paying a significant premium for downside '
          'protection — a bearish signal. ~1.0 = flat vol surface. '
          '&lt;1.0 = calls bid up relative to puts (bullish).</p>')
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>ATM Call Strike</th><th>ATM Call IV</th>'
          '<th>OTM Put Strike (~88%)</th><th>OTM Put IV</th>'
          '<th>IV Skew Ratio</th><th>Signal</th></tr></thead><tbody>')
    for ticker, d in all_data.items():
        opts          = d.get("options", {})
        put_skew      = opts.get("put_skew")
        atm_c_iv      = opts.get("atm_call_iv")
        atm_c_strike  = opts.get("atm_call_strike")
        otm_p_strike  = opts.get("otm_put_strike")
        otm_p_iv      = opts.get("otm_put_iv")
        if put_skew is not None:
            if put_skew >= 1.5:
                sk_color = "#e05c5c"
                sk_label = "Elevated fear — bearish hedge demand"
            elif put_skew >= 1.2:
                sk_color = "#e07b39"
                sk_label = "Mild skew — slight caution"
            elif put_skew <= 0.9:
                sk_color = "#00c896"
                sk_label = "Flat/bullish — no fear premium"
            else:
                sk_color = "#f0a500"
                sk_label = "Neutral"
            atm_c_str  = f"${atm_c_strike:,.0f}" if atm_c_strike else "—"
            atm_iv_str = f"{atm_c_iv:.1f}%" if atm_c_iv else "—"
            otm_s_str  = f"${otm_p_strike:,.0f}" if otm_p_strike else "—"
            otm_iv_str = f"{otm_p_iv:.1f}%" if otm_p_iv else "—"
            sk_str     = f"{put_skew:.2f}x"
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td class="num">{atm_c_str}</td>'
                  f'<td class="num">{atm_iv_str}</td>'
                  f'<td class="num">{otm_s_str}</td>'
                  f'<td class="num">{otm_iv_str}</td>'
                  f'<td class="num" style="color:{sk_color};font-weight:700">{sk_str}</td>'
                  f'<td style="color:{sk_color}">{sk_label}</td></tr>')
        else:
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td colspan="6" class="muted note">No IV skew data available</td></tr>')
    h += '</tbody></table></div></div>'

    # Expected move % chart
    h += '<div class="chart-card" style="margin-bottom:20px">'
    h += f'<h3>Market-Implied Move by Ticker \u2014 Normalized % Scale {help_btn("implied-move-chart")}</h3>'
    h += '<div class="chart-canvas" style="height:280px"><canvas id="chart-implied-range"></canvas></div>'
    h += ('<p class="note" style="margin-top:8px">'
          'Bar height = ±implied move % derived from the ATM straddle on the first expiry ≥7 days out. '
          'Labels show DTE so you can compare like-for-like. '
          'Green &lt;5% · Orange 5–10% · Red &gt;10%. '
          'Y-axis is locked symmetric around 0. '
          'Stocks with implied move &gt;40% are flagged as n/a (likely bad options data). '
          'Hover for expiry date, price range, and DTE.</p>')
    h += '</div>'

    # ── Earnings ────────────────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Earnings Calendar &amp; Beat/Miss History {help_btn("earnings-calendar")}</div>'

    # Earnings calendar summary table
    h += '<div class="card" style="margin-bottom:16px">'
    h += f'<div class="card-title">Upcoming Earnings &amp; Consensus Estimates {help_btn("earnings-calendar")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Next Earnings</th><th>Days Away</th>'
          '<th>EPS Estimate</th><th>Revenue Estimate</th>'
          '<th>Beat Rate (last 8Q)</th><th>Avg EPS Surprise</th>'
          '<th>Trend</th></tr></thead><tbody>')
    for ticker, d in all_data.items():
        earn = d.get("earnings", {})
        nd   = earn.get("next_date", "—") or "—"
        dte_e = earn.get("days_to_earnings")
        eps_e = earn.get("eps_estimate")
        rev_e = earn.get("rev_estimate")
        beats = earn.get("beat_count", 0)
        total_q = beats + earn.get("miss_count", 0)
        avg_surp = earn.get("avg_surprise_pct")
        beat_rate = round(beats / total_q * 100) if total_q else None

        dte_str  = f"{dte_e}d" if dte_e is not None else "—"
        dte_color = ("#f0a500" if (dte_e is not None and dte_e <= 14)
                     else "#e05c5c" if (dte_e is not None and dte_e <= 7)
                     else "#6b7194")
        eps_str  = f"${eps_e:.2f}" if eps_e else "—"
        rev_str  = fmt_money(rev_e) if rev_e else "—"
        br_color = ("#00c896" if (beat_rate and beat_rate >= 70)
                    else "#f0a500" if (beat_rate and beat_rate >= 50)
                    else "#e05c5c" if beat_rate else "#6b7194")
        br_str   = f"{beat_rate}% ({beats}/{total_q})" if beat_rate is not None else "—"
        surp_color = ("#00c896" if (avg_surp and avg_surp > 2)
                      else "#e05c5c" if (avg_surp and avg_surp < 0)
                      else "#6b7194")
        surp_str = (f"+{avg_surp}%" if (avg_surp and avg_surp > 0)
                    else f"{avg_surp}%" if avg_surp else "—")

        # Trend: are last 2Q better or worse than prior 2Q?
        hist = earn.get("history", [])
        trend_str = "—"
        trend_color = "#6b7194"
        if len(hist) >= 4:
            recent_avg = (sum(r.get("surprise_pct", 0) or 0
                              for r in hist[:2]) / 2)
            prior_avg  = (sum(r.get("surprise_pct", 0) or 0
                              for r in hist[2:4]) / 2)
            diff = recent_avg - prior_avg
            if diff > 1:
                trend_str = "▲ Improving"
                trend_color = "#00c896"
            elif diff < -1:
                trend_str = "▼ Declining"
                trend_color = "#e05c5c"
            else:
                trend_str = "→ Stable"
                trend_color = "#f0a500"

        h += (f'<tr><td class="ticker-cell">{ticker}</td>'
              f'<td class="muted">{nd}</td>'
              f'<td class="num" style="color:{dte_color};font-weight:600">{dte_str}</td>'
              f'<td class="num">{eps_str}</td>'
              f'<td class="num">{rev_str}</td>'
              f'<td class="num" style="color:{br_color};font-weight:700">{br_str}</td>'
              f'<td class="num" style="color:{surp_color};font-weight:600">{surp_str}</td>'
              f'<td style="color:{trend_color};font-weight:600">{trend_str}</td></tr>')
    h += '</tbody></table></div></div>'

    # Beat/miss history chart
    h += '<div class="chart-card" style="margin-bottom:20px">'
    h += f'<h3>EPS Surprise History \u2014 Last 8 Quarters {help_btn("eps-history")}</h3>'
    h += '<div class="chart-canvas" style="height:260px"><canvas id="chart-earnings"></canvas></div>'
    h += ('<p class="note" style="margin-top:8px">'
          'Bars show EPS surprise % per quarter. Green = beat consensus · Red = miss. '
          'Source: yfinance earnings_history.</p>')
    h += '</div>'

    # ── Earnings Estimate Revisions ─────────────────────────────────────────────
    h += '<div class="card" style="margin-bottom:20px">'
    h += f'<div class="card-title">Earnings Estimate Revisions (60-Day) {help_btn("eps-revision")}</div>'
    h += ('<p class="note" style="margin-bottom:12px">'
          'Revision = (current EPS estimate &minus; 60-day-ago estimate) / |60-day-ago estimate|. '
          'Green = analysts raised, Red = analysts cut.</p>')
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Ticker</th><th>EPS Revision (60d)</th><th>Direction</th></tr></thead><tbody>'
    for ticker, d in all_data.items():
        earn = d.get("earnings", {})
        rev_pct = earn.get("eps_revision_pct")
        if rev_pct is not None:
            if rev_pct > 0:
                rev_color = "#00c896"
                rev_arrow = "&#8593;"
                rev_label = "Estimates Raised"
            elif rev_pct < 0:
                rev_color = "#e05c5c"
                rev_arrow = "&#8595;"
                rev_label = "Estimates Cut"
            else:
                rev_color = "#f0a500"
                rev_arrow = "&rarr;"
                rev_label = "Unchanged"
            rev_pct_str = f"{rev_pct:+.1f}%"
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td class="num" style="color:{rev_color};font-weight:700">'
                  f'{rev_arrow} {rev_pct_str}</td>'
                  f'<td style="color:{rev_color}">{rev_label}</td></tr>')
        else:
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td colspan="2" class="muted note">No EPS trend data available</td></tr>')
    h += '</tbody></table></div></div>'

    # Per-ticker beat/miss detail tables
    h += f'<div class="card"><div class="card-title">Quarterly EPS Detail {help_btn("eps-history")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Quarter</th><th>Actual EPS</th>'
          '<th>Estimated EPS</th><th>Surprise %</th><th>Result</th></tr></thead><tbody>')
    any_hist = False
    for ticker, d in all_data.items():
        hist = d.get("earnings", {}).get("history", [])
        for row in hist:
            any_hist = True
            surp = row.get("surprise_pct")
            beat = row.get("beat", False)
            sc   = "#00c896" if beat else "#e05c5c"
            lbl  = "✓ Beat" if beat else "✗ Miss"
            surp_str = (f"+{surp}%" if (surp and surp > 0)
                        else f"{surp}%" if surp is not None else "—")
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td class="muted">{row.get("quarter","")}</td>'
                  f'<td class="num">${row.get("actual",0):.2f}</td>'
                  f'<td class="num">${row.get("estimate",0):.2f}</td>'
                  f'<td class="num" style="color:{sc}">{surp_str}</td>'
                  f'<td style="color:{sc};font-weight:700">{lbl}</td></tr>')
    if not any_hist:
        h += ('<tr><td colspan="6" class="muted note" style="padding:16px">'
              'No earnings history found. yfinance may not carry history for these tickers.</td></tr>')
    h += '</tbody></table></div></div>'

    # ── Analyst Ratings ─────────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Analyst Ratings {help_btn("analyst-ratings")}</div>'
    h += '<div class="chart-card" style="margin-bottom:20px">'
    h += '<h3>Analyst Consensus Breakdown (Most Recent Period)</h3>'
    h += '<div class="chart-canvas" style="height:220px"><canvas id="chart-analyst"></canvas></div>'
    h += '</div>'

    h += f'<div class="card"><div class="card-title">Rating Detail {help_btn("analyst-ratings")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Ticker</th><th>Signal</th><th>Period</th>'
          '<th>Strong Buy</th><th>Buy</th><th>Hold</th>'
          '<th>Sell</th><th>Strong Sell</th><th>Consensus</th></tr></thead><tbody>')
    for ticker, d in all_data.items():
        sc  = d["scores"].get("analyst")
        rec = d.get("rec_summary", [])
        if rec:
            r = rec[0]
            sb = r.get("strongBuy", 0); b = r.get("buy", 0)
            h_ = r.get("hold", 0);     s = r.get("sell", 0)
            ss = r.get("strongSell", 0)
            total = sb + b + h_ + s + ss
            consensus = sentiment_label(sc) if sc is not None else "—"
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td>{sentiment_badge(sc)}</td>'
                  f'<td class="muted">{r.get("period", "")}</td>'
                  f'<td class="num up">{sb}</td>'
                  f'<td class="num" style="color:#5cb85c">{b}</td>'
                  f'<td class="num warn">{h_}</td>'
                  f'<td class="num" style="color:#e07b39">{s}</td>'
                  f'<td class="num down">{ss}</td>'
                  f'<td style="font-weight:600">{consensus}</td></tr>')
        else:
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td colspan="8" class="muted note">No analyst data</td></tr>')
    h += '</tbody></table></div></div>'
    return h


def _tab_sec_filings(all_data: dict) -> str:
    h = ""

    # ── 8-K Material Events ─────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Recent 8-K Filings (Material Events) {help_btn("sec-8k")}</div>'
    h += '<div class="card"><p class="note" style="margin-bottom:12px">'
    h += '8-K filings disclose material events: earnings releases, leadership changes, acquisitions, regulatory actions, etc.</p>'
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Ticker</th><th>Form</th><th>Date</th><th>Document</th></tr></thead><tbody>'
    any_8k = False
    for ticker, d in all_data.items():
        for f in d.get("sec_8k", []):
            any_8k = True
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td>{f["form"]}</td>'
                  f'<td class="muted">{f["date"]}</td>'
                  f'<td><a href="{f["link"]}" target="_blank" '
                  f'style="color:var(--accent);text-decoration:none">'
                  f'{f.get("desc") or f["doc"]}</a></td></tr>')
    if not any_8k:
        h += ('<tr><td colspan="4" class="muted note" style="padding:16px">'
              'No 8-K filings found in the selected window. Widen --days or check CIK lookup.</td></tr>')
    h += '</tbody></table></div></div>'

    # ── Form 4 Insider Filings ──────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Recent Form 4 Filings (Insider Transactions \u2014 SEC EDGAR) {help_btn("sec-form4")}</div>'
    h += '<div class="card"><p class="note" style="margin-bottom:12px">'
    h += 'Form 4 is filed within 2 business days of an insider transaction. Cross-references yfinance data.</p>'
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Ticker</th><th>Date</th><th>Document</th></tr></thead><tbody>'
    any_f4 = False
    for ticker, d in all_data.items():
        for f in d.get("sec_f4", []):
            any_f4 = True
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td class="muted">{f["date"]}</td>'
                  f'<td><a href="{f["link"]}" target="_blank" '
                  f'style="color:var(--accent);text-decoration:none">'
                  f'{f.get("desc") or f["doc"]}</a></td></tr>')
    if not any_f4:
        h += ('<tr><td colspan="3" class="muted note" style="padding:16px">'
              'No Form 4 filings found in the selected window.</td></tr>')
    h += '</tbody></table></div></div>'

    # ── Activist Disclosures (13D/13G) ──────────────────────────────────────────
    h += f'<div class="sec-hdr">Activist Disclosures (13D/13G) {help_btn("sec-activist")}</div>'
    h += '<div class="card"><p class="note" style="margin-bottom:12px">'
    h += ('13D: activist investor acquires &gt;5% stake. '
          '13G: passive investor (&gt;5% stake, no activist intent). '
          'Both require EDGAR disclosure within 10 days.')
    h += '</p>'
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Ticker</th><th>Form</th><th>Date</th><th>Document</th></tr></thead><tbody>'
    any_activist = False
    for ticker, d in all_data.items():
        for f in d.get("sec_activist", []):
            any_activist = True
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td style="font-weight:600;color:var(--warn)">{f["form"]}</td>'
                  f'<td class="muted">{f["date"]}</td>'
                  f'<td><a href="{f["link"]}" target="_blank" '
                  f'style="color:var(--accent);text-decoration:none">'
                  f'{f.get("desc") or f["doc"]}</a></td></tr>')
    if not any_activist:
        h += ('<tr><td colspan="4" class="muted note" style="padding:16px">'
              'No recent 13D/13G filings found.</td></tr>')
    h += '</tbody></table></div></div>'

    # ── Recent 13F Filings ───────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Recent 13F Filings (Institutional Portfolio Snapshots) {help_btn("sec-13f")}</div>'
    h += '<div class="card"><p class="note" style="margin-bottom:12px">'
    h += ('13F-HR filings show institutional portfolio snapshots. '
          'Parse the XML attachment for full holdings detail. '
          'Filed quarterly within 45 days of quarter-end.')
    h += '</p>'
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Ticker</th><th>Date</th><th>Document</th></tr></thead><tbody>'
    any_13f = False
    for ticker, d in all_data.items():
        for f in d.get("sec_13f", []):
            any_13f = True
            h += (f'<tr><td class="ticker-cell">{ticker}</td>'
                  f'<td class="muted">{f["date"]}</td>'
                  f'<td><a href="{f["link"]}" target="_blank" '
                  f'style="color:var(--accent);text-decoration:none">'
                  f'{f.get("desc") or f["doc"]}</a></td></tr>')
    if not any_13f:
        h += ('<tr><td colspan="3" class="muted note" style="padding:16px">'
              'No recent 13F filings found in the 180-day window.</td></tr>')
    h += '</tbody></table></div></div>'

    # ── Data Source Notes ───────────────────────────────────────────────────────
    h += '<div class="card" style="margin-top:8px">'
    h += '<div class="card-title">About the Data Sources</div>'
    rows = [
        ("News Sentiment",    "yfinance news feed + VADER lexicon (or keyword fallback)"),
        ("News Velocity",     "7d vs prior 7d article count ratio — computed from yfinance news timestamps"),
        ("Social Media",      "Reddit public JSON API — r/wallstreetbets, r/stocks, r/investing (no auth required)"),
        ("Google Trends",     "pytrends (optional) — Finance category, US, 3-month window, weekly data"),
        ("Wikipedia Views",   "Wikimedia REST API — daily page views, all-access, all-agents (no auth required)"),
        ("Short Interest",    "yfinance info: shortPercentOfFloat, shortRatio, sharesShort"),
        ("Institutional",     "yfinance institutional_holders — snapshot, not change-over-time"),
        ("Insider Activity",  "yfinance insider_transactions — last 90 days"),
        ("Options Market",    "yfinance option_chain — nearest expiry, ATM put/call ratio + OTM put skew"),
        ("Analyst Ratings",   "yfinance recommendations + eps_trend — period-based (0q = current quarter)"),
        ("SEC 8-K / Form 4",  "EDGAR submissions API (data.sec.gov) — no key required"),
        ("SEC 13D/13G",       "EDGAR submissions API — activist and passive large-holder disclosures (5%+ stake)"),
        ("SEC 13F",           "EDGAR submissions API — institutional portfolio snapshots (quarterly, 45-day lag)"),
        ("Congressional",     "House & Senate STOCK Act disclosures via house-stock-watcher.com"),
        ("Bond/Credit",       "Direct bond pricing requires paid APIs (Bloomberg, FINRA TRACE). "
                              "Proxy metrics shown: debtToEquity, currentRatio, creditRating (yfinance)."),
    ]
    h += '<table style="font-size:12px"><tbody>'
    for src, desc in rows:
        h += (f'<tr><td style="padding:6px 12px;font-weight:600;color:var(--accent);'
              f'white-space:nowrap">{src}</td>'
              f'<td style="padding:6px 12px;color:var(--muted)">{desc}</td></tr>')
    h += '</tbody></table></div>'
    return h


# ── Chart.js initialization JS ────────────────────────────────────────────────
def _build_charts_js(all_data: dict) -> str:
    tickers = list(all_data.keys())
    COLORS  = ["#4f8ef7","#00c896","#f0a500","#e05c5c","#9b59b6",
               "#1abc9c","#e67e22","#3498db","#e91e63","#cddc39"]

    js = "document.addEventListener('DOMContentLoaded',function(){\n"

    # ── Auto-align table headers to match their column's data alignment ─────────
    # Fixes all tables in one pass: right-aligned td.num columns get
    # right-aligned headers, center-aligned cells get centered headers.
    js += """
(function(){
  document.querySelectorAll('table').forEach(function(tbl){
    var firstRow = tbl.querySelector('tbody tr');
    if(!firstRow) return;
    var cells   = firstRow.querySelectorAll('td');
    var headers = tbl.querySelectorAll('thead th');
    cells.forEach(function(td, i){
      if(i >= headers.length) return;
      var align = window.getComputedStyle(td).textAlign;
      if(align === 'right' || align === 'center'){
        headers[i].style.textAlign = align;
      }
    });
  });
})();
"""

    # ── Composite horizontal bar ───────────────────────────────────────────────
    scores_0_100 = [round(all_data[t]["composite"] * 50 + 50, 1) for t in tickers]
    bar_colors   = [sentiment_color(all_data[t]["composite"]) for t in tickers]
    js += f"""
var ctx_comp = document.getElementById('chart-composite');
if(ctx_comp){{
  new Chart(ctx_comp,{{
    type:'bar',
    data:{{
      labels:{json.dumps(tickers)},
      datasets:[{{
        label:'Sentiment Score',
        data:{json.dumps(scores_0_100)},
        backgroundColor:{json.dumps([c + "cc" for c in bar_colors])},
        borderColor:{json.dumps(bar_colors)},
        borderWidth:1,
        borderRadius:4
      }}]
    }},
    options:{{
      indexAxis:'y',
      responsive:true,
      maintainAspectRatio:false,
      scales:{{
        x:{{min:0,max:100,
          grid:{{color:'#252a3a'}},
          ticks:{{color:'#6b7194',callback:function(v){{
            if(v===50)return'Neutral';
            if(v===0)return'Bearish';
            if(v===100)return'Bullish';
            return v;
          }}}}}},
        y:{{grid:{{display:false}},ticks:{{color:'#e8eaf2',font:{{weight:'bold',size:13}}}}}}
      }},
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{
          label:function(ctx){{
            var raw=ctx.raw;
            var score=((raw-50)/50).toFixed(2);
            return ' Score: '+raw.toFixed(0)+'/100  ('+score+')';
          }}
        }}}}
      }}
    }}
  }});
}}
"""

    # ── Per-ticker radar charts ────────────────────────────────────────────────
    signal_keys = list(SIGNAL_WEIGHTS.keys())
    signal_lbls = [SIGNAL_LABELS[k] for k in signal_keys]
    for i, ticker in enumerate(tickers):
        d      = all_data[ticker]
        values = [score_to_radar(d["scores"].get(k)) for k in signal_keys]
        color  = COLORS[i % len(COLORS)]
        js += f"""
var ctx_r_{ticker.replace('-','_').replace('.','_')} = document.getElementById('radar-{ticker}');
if(ctx_r_{ticker.replace('-','_').replace('.','_')}){{
  new Chart(ctx_r_{ticker.replace('-','_').replace('.','_')},{{
    type:'radar',
    data:{{
      labels:{json.dumps(signal_lbls)},
      datasets:[{{
        label:'{ticker}',
        data:{json.dumps(values)},
        backgroundColor:'{color}22',
        borderColor:'{color}',
        borderWidth:2,
        pointBackgroundColor:'{color}',
        pointRadius:3,
        pointHoverRadius:5
      }}]
    }},
    options:{{
      responsive:true,
      maintainAspectRatio:false,
      scales:{{r:{{
        min:0,max:100,
        ticks:{{color:'#6b7194',stepSize:25,backdropColor:'transparent',
               font:{{size:9}},display:false}},
        grid:{{color:'#252a3a'}},
        pointLabels:{{color:'#e8eaf2',font:{{size:10}}}},
        angleLines:{{color:'#252a3a'}}
      }}}},
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{
          label:function(ctx){{
            var v=ctx.raw;
            var score=((v-50)/50).toFixed(2);
            return ctx.dataset.label+': '+score;
          }}
        }}}}
      }}
    }}
  }});
}}
"""

    # ── Reddit sentiment donut charts ─────────────────────────────────────────
    for ticker, d in all_data.items():
        rd   = d.get("reddit", {})
        bull = rd.get("bullish", 0)
        bear = rd.get("bearish", 0)
        neut = rd.get("neutral", 0)
        if bull + bear + neut == 0:
            continue
        js += f"""
var ctx_d_{ticker.replace('-','_').replace('.','_')} = document.getElementById('donut-{ticker}');
if(ctx_d_{ticker.replace('-','_').replace('.','_')}){{
  new Chart(ctx_d_{ticker.replace('-','_').replace('.','_')},{{
    type:'doughnut',
    data:{{
      labels:['Bullish','Bearish','Neutral'],
      datasets:[{{
        data:[{bull},{bear},{neut}],
        backgroundColor:['#00c89699','#e05c5c99','#6b719499'],
        borderColor:['#00c896','#e05c5c','#6b7194'],
        borderWidth:1
      }}]
    }},
    options:{{
      responsive:true,
      maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
               tooltip:{{callbacks:{{label:function(ctx){{return ctx.label+': '+ctx.raw}}}}}}}}
    }}
  }});
}}
"""

    # ── Implied move % chart (normalized, centered at 0) ──────────────────────
    # Each bar shows [-impl_pct, +impl_pct] so all stocks are on the same % scale
    MAX_IMPL_PCT = 40.0   # cap: >40% from nearest expiry straddle = bad data
    impl_range_data = []
    impl_lbls       = []
    impl_colors     = []
    impl_tooltips   = []   # store extra info for tooltip
    valid_impls     = []   # for computing symmetric Y-axis bounds
    for t in tickers:
        opts = all_data[t].get("options", {})
        impl = opts.get("implied_move_pct")
        sp   = opts.get("spot") or all_data[t].get("price")
        lo   = opts.get("expected_lower")
        hi   = opts.get("expected_upper")
        dte  = opts.get("dte", "?")
        expiry = opts.get("expiry", "")
        if impl and impl <= MAX_IMPL_PCT:
            impl_range_data.append([-round(impl, 2), round(impl, 2)])
            # Label includes DTE so apples-to-oranges comparison is visible
            impl_lbls.append(f"{t} ({dte}d)" if isinstance(dte, int) else t)
            valid_impls.append(impl)
            # Colour by magnitude: <5% calm, 5–10% moderate, >10% elevated
            if impl < 5:
                impl_colors.append("rgba(0,200,150,0.45)")
            elif impl < 10:
                impl_colors.append("rgba(240,165,0,0.45)")
            else:
                impl_colors.append("rgba(224,92,92,0.45)")
            impl_tooltips.append({
                "impl":   impl,
                "spot":   round(sp, 2) if sp else None,
                "lo":     round(lo, 2) if lo else None,
                "hi":     round(hi, 2) if hi else None,
                "dte":    dte,
                "expiry": expiry,
            })
        else:
            # Bad data or missing — show as greyed-out zero bar
            impl_range_data.append(None)
            impl_lbls.append(f"{t} (n/a)")
            impl_colors.append("rgba(107,113,148,0.2)")
            impl_tooltips.append({"impl": None})

    # Compute a symmetric Y-axis bound: round the max implied move up to
    # nearest 5%, add 2% padding, then mirror it.  This guarantees 0 is always
    # dead-centre and the chart never looks asymmetric due to auto-scaling.
    if valid_impls:
        max_impl   = max(valid_impls)
        y_bound    = (int(max_impl / 5) + 1) * 5 + 2   # next 5% step + 2% pad
    else:
        y_bound = 10
    y_min_val = -y_bound
    y_max_val  = y_bound

    js += f"""
var ctx_ir = document.getElementById('chart-implied-range');
if(ctx_ir){{
  var implData     = {json.dumps(impl_range_data)};
  var implLbls     = {json.dumps(impl_lbls)};
  var implColors   = {json.dumps(impl_colors)};
  var implTooltips = {json.dumps(impl_tooltips)};
  new Chart(ctx_ir, {{
    type: 'bar',
    data: {{
      labels: implLbls,
      datasets: [{{
        label: 'Implied Move \u00b1%',
        data: implData,
        backgroundColor: implColors,
        borderColor: implColors.map(c => c.replace('0.45','0.9')),
        borderWidth: 1,
        borderRadius: 2,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{
          grid: {{display: false}},
          ticks: {{color: '#e8eaf2', font: {{weight: 'bold', size: 11}}}}
        }},
        y: {{
          min: {y_min_val},
          max: {y_max_val},
          grid: {{color: '#252a3a'}},
          ticks: {{
            color: '#6b7194',
            callback: function(v) {{ return (v > 0 ? '+' : '') + v + '%'; }}
          }},
          title: {{display: true, text: 'Implied Move (%)', color: '#6b7194'}}
        }}
      }},
      plugins: {{
        legend: {{display: false}},
        tooltip: {{
          callbacks: {{
            title: function(items) {{ return items[0].label; }},
            label: function(ctx) {{
              var tip = implTooltips[ctx.dataIndex];
              if (!tip || tip.impl === null) return ' No valid data';
              var lines = [
                ' Implied Move: \u00b1' + tip.impl + '%',
                ' Expiry: ' + (tip.expiry || '?') + '  (' + tip.dte + ' DTE)',
              ];
              if (tip.spot) {{
                lines.push(' Current Price: $' + tip.spot.toLocaleString());
              }}
              if (tip.lo && tip.hi) {{
                lines.push(' Expected Range: $' + tip.lo.toLocaleString()
                           + ' \u2013 $' + tip.hi.toLocaleString());
              }}
              return lines;
            }}
          }}
        }}
      }}
    }}
  }});
}}
"""

    # ── Put/Call ratio bar ─────────────────────────────────────────────────────
    pc_values = []
    pc_colors = []
    for t in tickers:
        pc = all_data[t].get("options", {}).get("put_call_ratio")
        pc_values.append(round(float(pc), 2) if pc else 0)
        c  = "#00c896" if (pc and pc < 0.7) else "#e05c5c" if (pc and pc > 1.0) else "#f0a500"
        pc_colors.append(c + "cc")
    js += f"""
var ctx_pc = document.getElementById('chart-pc');
if(ctx_pc){{
  new Chart(ctx_pc,{{
    type:'bar',
    data:{{
      labels:{json.dumps(tickers)},
      datasets:[{{
        label:'Put/Call Ratio',
        data:{json.dumps(pc_values)},
        backgroundColor:{json.dumps(pc_colors)},
        borderWidth:0,
        borderRadius:4
      }}]
    }},
    options:{{
      responsive:true,
      maintainAspectRatio:false,
      scales:{{
        x:{{grid:{{display:false}},ticks:{{color:'#e8eaf2',font:{{weight:'bold'}}}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194'}},
           title:{{display:true,text:'P/C Ratio',color:'#6b7194'}}}}
      }},
      plugins:{{legend:{{display:false}}}}
    }}
  }});
}}
"""

    # ── Earnings beat/miss bar chart ──────────────────────────────────────────
    # One dataset per ticker, x-axis = quarters, y-axis = surprise %
    # Build unified quarter labels (union of all tickers' quarters)
    earn_datasets = []
    earn_all_qtrs = []
    for t in tickers:
        hist = all_data[t].get("earnings", {}).get("history", [])
        for r in hist:
            q = r.get("quarter", "")
            if q and q not in earn_all_qtrs:
                earn_all_qtrs.append(q)
    # Sort oldest→newest (left→right on chart); keep last 8 quarters
    earn_all_qtrs = sorted(set(earn_all_qtrs))[-8:]

    def _qtr_label(date_str: str) -> str:
        """Convert raw quarter-end date '2025-07-31' → 'Q3 '25'."""
        try:
            month = int(date_str[5:7])
            year  = int(date_str[2:4])
            q     = (month - 1) // 3 + 1
            return f"Q{q} '{year:02d}"
        except (ValueError, IndexError):
            return date_str

    earn_labels = [_qtr_label(q) for q in earn_all_qtrs]

    CHART_COLORS = ["#4f8ef7","#00c896","#f0a500","#e05c5c","#9b59b6",
                    "#1abc9c","#e67e22","#3498db","#e91e63","#cddc39"]
    for i, t in enumerate(tickers):
        hist     = all_data[t].get("earnings", {}).get("history", [])
        hist_map = {r.get("quarter", ""): r.get("surprise_pct") for r in hist}
        vals     = [hist_map.get(q) for q in earn_all_qtrs]
        # colour per bar: green if beat, red if miss
        bar_colors = []
        for v in vals:
            if v is None:
                bar_colors.append("rgba(107,113,148,0.3)")
            elif v >= 0:
                bar_colors.append("rgba(0,200,150,0.75)")
            else:
                bar_colors.append("rgba(224,92,92,0.75)")
        earn_datasets.append({
            "label":           t,
            "data":            vals,
            "backgroundColor": bar_colors,
            "borderColor":     bar_colors,
            "borderWidth":     1,
            "borderRadius":    3,
        })

    js += f"""
var ctx_earn = document.getElementById('chart-earnings');
if(ctx_earn){{
  new Chart(ctx_earn, {{
    type: 'bar',
    data: {{
      labels: {json.dumps(earn_labels)},
      datasets: {json.dumps(earn_datasets)}
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{
          grid: {{display: false}},
          ticks: {{color: '#e8eaf2'}}
        }},
        y: {{
          grid: {{color: '#252a3a'}},
          ticks: {{
            color: '#6b7194',
            callback: function(v) {{ return (v > 0 ? '+' : '') + v + '%'; }}
          }},
          title: {{display: true, text: 'EPS Surprise %', color: '#6b7194'}}
        }}
      }},
      plugins: {{
        legend: {{
          position: 'bottom',
          labels: {{color: '#6b7194', boxWidth: 12, font: {{size: 11}}}}
        }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              var v = ctx.raw;
              if (v === null || v === undefined) return ctx.dataset.label + ': no data';
              var sign = v >= 0 ? '+' : '';
              var result = v >= 0 ? 'Beat' : 'Miss';
              return ctx.dataset.label + ': ' + result + ' ' + sign + v + '%';
            }}
          }}
        }}
      }}
    }}
  }});
}}
"""

    # ── Analyst stacked bar ────────────────────────────────────────────────────
    sb_vals = []; b_vals = []; h_vals = []; s_vals = []; ss_vals = []
    for t in tickers:
        rec = all_data[t].get("rec_summary", [])
        r   = rec[0] if rec else {}
        sb_vals.append(r.get("strongBuy", 0) or 0)
        b_vals.append(r.get("buy", 0) or 0)
        h_vals.append(r.get("hold", 0) or 0)
        s_vals.append(r.get("sell", 0) or 0)
        ss_vals.append(r.get("strongSell", 0) or 0)
    js += f"""
var ctx_an = document.getElementById('chart-analyst');
if(ctx_an){{
  new Chart(ctx_an,{{
    type:'bar',
    data:{{
      labels:{json.dumps(tickers)},
      datasets:[
        {{label:'Strong Buy', data:{json.dumps(sb_vals)}, backgroundColor:'#00c896bb',borderWidth:0,borderRadius:2}},
        {{label:'Buy',        data:{json.dumps(b_vals)},  backgroundColor:'#5cb85cbb',borderWidth:0}},
        {{label:'Hold',       data:{json.dumps(h_vals)},  backgroundColor:'#f0a500bb',borderWidth:0}},
        {{label:'Sell',       data:{json.dumps(s_vals)},  backgroundColor:'#e07b39bb',borderWidth:0}},
        {{label:'Strong Sell',data:{json.dumps(ss_vals)}, backgroundColor:'#e05c5cbb',borderWidth:0,borderRadius:2}}
      ]
    }},
    options:{{
      responsive:true,
      maintainAspectRatio:false,
      scales:{{
        x:{{stacked:true,grid:{{display:false}},ticks:{{color:'#e8eaf2',font:{{weight:'bold'}}}}}},
        y:{{stacked:true,grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194'}}}}
      }},
      plugins:{{
        legend:{{position:'bottom',labels:{{color:'#6b7194',boxWidth:12,font:{{size:11}}}}}},
        tooltip:{{mode:'index',intersect:false}}
      }}
    }}
  }});
}}
"""

    # ── Google Trends multi-line chart ─────────────────────────────────────────
    trends_datasets = []
    trends_has_data = False
    for i, t in enumerate(tickers):
        series = all_data[t].get("gt", {}).get("series", [])
        if len(series) >= 2:
            trends_has_data = True
            color = COLORS[i % len(COLORS)]
            trends_datasets.append({
                "label":       t,
                "data":        [float(v) for v in series[-12:]],
                "borderColor": color,
                "backgroundColor": color + "22",
                "borderWidth": 2,
                "pointRadius": 3,
                "tension":     0.3,
                "fill":        False,
            })
    if trends_has_data:
        week_labels = ["W" + str(i + 1) for i in range(12)]
        js += f"""
var ctx_tr = document.getElementById('chart-trends');
if(ctx_tr){{
  new Chart(ctx_tr, {{
    type: 'line',
    data: {{
      labels: {json.dumps(week_labels)},
      datasets: {json.dumps(trends_datasets)}
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{grid: {{color: '#252a3a'}}, ticks: {{color: '#6b7194'}}}},
        y: {{
          min: 0, max: 100,
          grid: {{color: '#252a3a'}},
          ticks: {{color: '#6b7194'}},
          title: {{display: true, text: 'Search Interest (0-100)', color: '#6b7194'}}
        }}
      }},
      plugins: {{
        legend: {{position: 'bottom', labels: {{color: '#6b7194', boxWidth: 12, font: {{size: 11}}}}}},
        tooltip: {{mode: 'index', intersect: false}}
      }}
    }}
  }});
}}
"""

    # ── Wikipedia views multi-line chart ───────────────────────────────────────
    wiki_datasets = []
    wiki_has_data = False
    for i, t in enumerate(tickers):
        series = all_data[t].get("wiki", {}).get("series", [])
        if len(series) >= 2:
            wiki_has_data = True
            color = COLORS[i % len(COLORS)]
            wiki_datasets.append({
                "label":       t,
                "data":        [int(v) for v in series],
                "borderColor": color,
                "backgroundColor": color + "22",
                "borderWidth": 2,
                "pointRadius": 2,
                "tension":     0.3,
                "fill":        False,
            })
    if wiki_has_data:
        # Use day indices as labels (we don't store the actual dates in series)
        wiki_max_len = max(len(all_data[t].get("wiki", {}).get("series", [])) for t in tickers)
        wiki_day_labels = ["D" + str(i + 1) for i in range(wiki_max_len)]
        js += f"""
var ctx_wk = document.getElementById('chart-wiki');
if(ctx_wk){{
  new Chart(ctx_wk, {{
    type: 'line',
    data: {{
      labels: {json.dumps(wiki_day_labels)},
      datasets: {json.dumps(wiki_datasets)}
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{grid: {{color: '#252a3a'}}, ticks: {{color: '#6b7194', maxTicksLimit: 15}}}},
        y: {{
          grid: {{color: '#252a3a'}},
          ticks: {{color: '#6b7194'}},
          title: {{display: true, text: 'Daily Page Views', color: '#6b7194'}}
        }}
      }},
      plugins: {{
        legend: {{position: 'bottom', labels: {{color: '#6b7194', boxWidth: 12, font: {{size: 11}}}}}},
        tooltip: {{mode: 'index', intersect: false}}
      }}
    }}
  }});
}}
"""

    js += "});\n"
    return js


# ── Full HTML assembly ─────────────────────────────────────────────────────────
def build_html(all_data: dict, days: int, vader: bool) -> str:
    tickers   = list(all_data.keys())
    date_str  = datetime.date.today().strftime("%B %d, %Y")
    vader_txt = ("VADER sentiment active" if vader
                 else "Using keyword sentiment (install vaderSentiment for better accuracy)")
    n_sources = 13 + (1 if PYTRENDS_AVAILABLE else 0)
    sources_txt = str(n_sources) + " data sources active"

    h  = '<!DOCTYPE html><html lang="en"><head>'
    h += '<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    h += f'<title>Sentiment Analysis — {date_str}</title>'
    h += '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'
    h += f'<style>{_CSS}</style>'
    h += '</head><body>'

    # Header
    h += '<div class="hdr">'
    h += '<div class="hdr-title">📡 Sentiment Analysis</div>'
    h += (f'<div class="hdr-sub">{len(tickers)} stock{"s" if len(tickers)!=1 else ""} · '
          f'{date_str} · {days}-day window</div>')
    h += f'<div class="hdr-badge">{vader_txt}</div>'
    h += f'<div class="hdr-badge">{sources_txt}</div>'
    h += '</div>'

    # Tab bar
    tabs = [
        ("overview", "Overview"),
        ("news",     "News & Social"),
        ("smart",    "Smart Money"),
        ("options",  "Options & Analyst"),
        ("sec",      "SEC Filings"),
    ]
    h += '<div class="tab-bar">'
    for i, (tid, tlbl) in enumerate(tabs):
        cls = " active" if i == 0 else ""
        h += (f'<button class="tab-btn{cls}" '
              f'onclick="switchTab(\'{tid}\',this)">{tlbl}</button>')
    h += '</div>'

    # Content
    h += '<div class="container">'
    h += f'<div id="tab-overview" class="tab-panel active">{_tab_overview(all_data)}</div>'
    h += f'<div id="tab-news"     class="tab-panel">{_tab_news_social(all_data)}</div>'
    h += f'<div id="tab-smart"    class="tab-panel">{_tab_smart_money(all_data)}</div>'
    h += f'<div id="tab-options"  class="tab-panel">{_tab_options_analyst(all_data)}</div>'
    h += f'<div id="tab-sec"      class="tab-panel">{_tab_sec_filings(all_data)}</div>'
    h += '</div>'

    # Help popover HTML element (shared, positioned on click)
    h += (
        '<div id="help-pop" role="tooltip">'
        '<button id="help-pop-close" onclick="closeHelp()" aria-label="Close">×</button>'
        '<div id="help-pop-title"></div>'
        '<div id="help-pop-body"></div>'
        '</div>'
    )

    # JS: tab switch + charts
    h += '<script>'
    # Build the help data JS object from the Python dict
    help_js_obj = json.dumps({k: v for k, v in _HELP_DATA.items()})
    h += f'var _HELP={help_js_obj};\n'
    h += """
function showHelp(key, evt) {
  evt.stopPropagation();
  var d = _HELP[key];
  if (!d) return;
  var pop = document.getElementById('help-pop');
  document.getElementById('help-pop-title').textContent = d.title;
  document.getElementById('help-pop-body').innerHTML = d.body;
  // Show offscreen first to measure height
  pop.style.visibility = 'hidden';
  pop.style.display    = 'block';
  var vw = window.innerWidth, vh = window.innerHeight;
  var pw = pop.offsetWidth, ph = pop.offsetHeight;
  var x  = evt.clientX + 14;
  var y  = evt.clientY + 14;
  // Horizontal boundary
  if (x + pw > vw - 8)  x = evt.clientX - pw - 14;
  if (x < 8)            x = 8;
  // Vertical boundary
  if (y + ph > vh - 8)  y = evt.clientY - ph - 14;
  if (y < 8)            y = 8;
  pop.style.left       = x + 'px';
  pop.style.top        = y + 'px';
  pop.style.visibility = '';
}
function closeHelp() {
  document.getElementById('help-pop').style.display = 'none';
}
document.addEventListener('click', function(e) {
  var pop = document.getElementById('help-pop');
  if (pop && pop.style.display !== 'none' && !pop.contains(e.target)) closeHelp();
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeHelp();
});
"""
    h += ('function switchTab(id,btn){'
          'document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));'
          'document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));'
          'document.getElementById("tab-"+id).classList.add("active");'
          'btn.classList.add("active");}\n')
    h += _build_charts_js(all_data)
    h += '</script>'
    h += '</body></html>'
    return h


# ── Portfolio loading ──────────────────────────────────────────────────────────
def load_portfolio(path: str) -> list:
    """Parse ticker,shares CSV. Also accepts comma-separated ticker string."""
    if not os.path.isfile(path):
        return []
    tickers = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = {k.strip().lower(): v.strip() for k, v in row.items()}
            t = (keys.get("ticker") or keys.get("symbol") or keys.get("stock") or "").upper().strip()
            if t and t != "CASH":
                tickers.append(t)
    return tickers


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-source sentiment analysis report",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "portfolio",
        help="Path to portfolio CSV (ticker,shares) or comma-separated tickers e.g. NVDA,AAPL",
    )
    parser.add_argument(
        "--days", type=int, default=30, metavar="DAYS",
        help="Lookback window in days for news, SEC filings, insider activity (default: 30)",
    )
    args = parser.parse_args()

    # Resolve tickers
    portfolio_arg = args.portfolio.strip()
    _looks_like_path = (
        portfolio_arg.lower().endswith(".csv")
        or os.sep in portfolio_arg
        or "/" in portfolio_arg
    )
    if os.path.isfile(portfolio_arg):
        tickers = load_portfolio(portfolio_arg)
        if not tickers:
            sys.exit(f"ERROR: No tickers found in {portfolio_arg}")
        print(f"\n  Portfolio: {portfolio_arg}")
    elif _looks_like_path:
        # Argument looks like a file path but wasn't found — abort instead of
        # passing the path to Yahoo Finance as a ticker symbol.
        abs_path = os.path.abspath(portfolio_arg)
        sys.exit(
            f"ERROR: Portfolio file not found: {portfolio_arg}\n"
            f"       Resolved to: {abs_path}\n"
            f"       Check the path and working directory."
        )
    else:
        tickers = [t.strip().upper() for t in portfolio_arg.split(",") if t.strip()]
        if not tickers:
            sys.exit("ERROR: No valid tickers provided.")

    print(f"  Tickers:   {', '.join(tickers)}")
    print(f"  Window:    {args.days} days")
    print(f"  VADER:     {'yes' if VADER_AVAILABLE else 'no (keyword fallback)'}\n")

    # Pre-fetch shared data (once for all tickers)
    print("[1/3] Loading congressional trading disclosures (STOCK Act)...")
    _load_congress_data()
    congress_loaded = bool(_CONGRESS_CACHE.get("house") or _CONGRESS_CACHE.get("senate"))
    print(f"       House: {len(_CONGRESS_CACHE.get('house', []))} records  "
          f"Senate: {len(_CONGRESS_CACHE.get('senate', []))} records")

    print("[2/3] Loading SEC EDGAR CIK mapping...")
    _load_cik_map()
    print(f"       {len(_CIK_MAP_CACHE)} companies indexed")

    # Resolve CIKs
    cik_map = {t: get_cik(t) for t in tickers}
    for t, cik in cik_map.items():
        if not cik:
            print(f"       WARNING: CIK not found for {t} — SEC filings will be skipped")

    # Parallel per-ticker analysis
    print(f"[3/3] Analysing {len(tickers)} ticker(s) in parallel...\n")
    all_data   = {}
    n_tickers  = len(tickers)
    completed  = [0]
    lock       = threading.Lock()

    def _run(ticker):
        cik    = cik_map.get(ticker)
        result = analyze_ticker(ticker, args.days, cik)
        with lock:
            completed[0] += 1
            comp    = result["composite"]
            label   = sentiment_label(comp)
            n_news  = len(result.get("news", []))
            n_cong  = len(result.get("congressional", []))
            print(f"  [{completed[0]}/{n_tickers}] {ticker:<6}  "
                  f"composite={comp:+.2f}  ({label})  "
                  f"news={n_news}  congress={n_cong}")
        return ticker, result

    with ThreadPoolExecutor(max_workers=min(n_tickers, 6)) as pool:
        futures = {pool.submit(_run, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                ticker, result = future.result()
                all_data[ticker] = result
            except Exception as exc:
                t = futures[future]
                print(f"  ERROR {t}: {exc}")

    # Sort alphabetically by ticker (affects all tables and charts)
    all_data = {t: all_data[t] for t in sorted(tickers) if t in all_data}

    if not all_data:
        sys.exit("ERROR: No data retrieved for any ticker.")

    # Build + save HTML
    print("\n  Generating report...")
    html = build_html(all_data, args.days, VADER_AVAILABLE)

    date_prefix = datetime.datetime.now().strftime("%Y_%m_%d")
    outfile     = os.path.join(OUTPUT_DIR, f"{date_prefix}_sentiment.html")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(outfile) / 1024
    print(f"\n  ✓  Report saved → {outfile}  ({size_kb:.0f} KB)")
    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        print(f"  Opening in browser...\n")
        webbrowser.open(f"file://{os.path.abspath(outfile)}")


if __name__ == "__main__":
    main()
