#!/usr/bin/env python3
"""
Topic Sentiment Analyzer
========================
Multi-source sentiment analysis for any keyword, phrase, or topic.

Data sources (no API keys required):
  · Google News  — RSS headlines + snippets (feedparser)
  · Reddit       — public search JSON API (r/all, last 30 days)
  · Hacker News  — Algolia search API (no auth)
  · Google Trends — pytrends 90-day relative search interest

Output:
  Dark-themed HTML report with:
    • Animated sentiment gauge (SVG)
    • Positive / Negative word clouds (embedded PNG)
    • Sentiment timeline chart (Chart.js)
    • Google Trends interest chart
    • Per-source breakdown bars
    • Full scrollable mentions table

Usage:
  python topicSentiment.py --query "Federal Reserve"
  python topicSentiment.py --query "NVDA earnings" --days 14
  python topicSentiment.py --query "AI regulation"
"""

import argparse, base64, datetime, html as _html, io, json, math, os, re, sys, webbrowser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus

import requests

# ── VADER ──────────────────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _sia     = SentimentIntensityAnalyzer()
    _VADER   = True
except ImportError:
    _sia = None; _VADER = False

def _score(text):
    if not _VADER or not text:
        return None
    return _sia.polarity_scores(str(text))


# ── FinBERT (optional) ─────────────────────────────────────────────────────────
# ProsusAI/finbert — BERT fine-tuned on ~4 500 financial news articles.
# Labels: positive / negative / neutral.
# First use auto-downloads ~440 MB to ~/.cache/huggingface; subsequent runs load
# from disk in a few seconds.
_FINBERT_PIPE   = None
_FINBERT_LOADED = False     # True once a load attempt has been made


def _ensure_finbert():
    global _FINBERT_PIPE, _FINBERT_LOADED
    if _FINBERT_LOADED:
        return _FINBERT_PIPE is not None
    _FINBERT_LOADED = True
    try:
        # Force PyTorch backend — avoids broken h5py / TensorFlow on some envs
        import os as _os
        _os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        _os.environ.setdefault("USE_TF", "0")
        from transformers import pipeline as _hf_pipeline
        print("  Loading FinBERT (ProsusAI/finbert)"
              " — first run downloads ~440 MB…", end=" ", flush=True)
        _FINBERT_PIPE = _hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,      # CPU; set device=0 for CUDA GPU
            top_k=None,     # return all 3 class probabilities, not just top label
        )
        print("ready ✓")
        return True
    except Exception as ex:
        print(f"  FinBERT unavailable ({type(ex).__name__}: {ex})")
        return False


def _score_finbert_batch(texts):
    """
    Batch-score *texts* with FinBERT.

    Returns a list of dicts  {"compound": float, "pos": float, "neg": float, "neu": float}
    (one per input text), or a list of None when the model is unavailable or fails.

    compound = positive_prob − negative_prob  → same −1 … +1 range as VADER,
    making the two scores directly comparable.
    """
    if not _ensure_finbert() or not texts:
        return [None] * len(texts)
    try:
        trunc   = [t[:512] for t in texts]      # hard-cap to avoid tokenizer OOM
        results = _FINBERT_PIPE(
            trunc, truncation=True, max_length=128,
            batch_size=16, top_k=None,
        )
        out = []
        for res in results:
            s   = {r["label"].lower(): r["score"] for r in res}
            pos = s.get("positive", 0.0)
            neg = s.get("negative", 0.0)
            neu = s.get("neutral",  0.0)
            out.append({"compound": round(pos - neg, 4),
                        "pos": round(pos, 4),
                        "neg": round(neg, 4),
                        "neu": round(neu, 4)})
        return out
    except Exception as ex:
        print(f"\n  FinBERT batch error: {ex}")
        return [None] * len(texts)

# ── wordcloud + matplotlib ─────────────────────────────────────────────────────
try:
    from wordcloud import WordCloud, STOPWORDS as _WC_STOPS
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _WC = True
except Exception:          # catches ImportError AND matplotlib backend errors
    _WC = False; _WC_STOPS = set()

# ── pytrends ──────────────────────────────────────────────────────────────────
try:
    from pytrends.request import TrendReq
    _TRENDS = True
except ImportError:
    _TRENDS = False

# ── Colour helpers ─────────────────────────────────────────────────────────────
def _col(c):
    if c is None:  return "#94a3b8"
    if c >  0.50:  return "#10b981"
    if c >  0.10:  return "#34d399"
    if c >= -0.10: return "#94a3b8"
    if c >= -0.50: return "#f87171"
    return "#ef4444"

def _lbl(c):
    if c is None:  return "Insufficient Data"
    if c >  0.50:  return "Very Positive"
    if c >  0.10:  return "Positive"
    if c >= -0.10: return "Neutral"
    if c >= -0.50: return "Negative"
    return "Very Negative"

# ── Ticker reference set ─────────────────────────────────────────────────────
# Base set of commonly discussed US equities + ETFs.
# _load_ticker_set() supplements this with the live S&P 500 list from Wikipedia.
_BASE_TICKERS = frozenset([
    # Mega-cap tech
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","ORCL",
    "ADBE","CRM","INTC","AMD","QCOM","TXN","ARM","SMCI","AMAT","LRCX","KLAC",
    "MU","MRVL","DELL","HPE","HPQ","CDNS","SNPS","ANSS","FTNT","PANW","CRWD",
    "NET","DDOG","MDB","SNOW","ZS","OKTA","TWLO","COIN","HOOD","PLTR","RBLX",
    "SNAP","PINS","SPOT","SHOP","SE","MELI","GRAB","U","DOCN","FSLY","NTNX",
    # Finance
    "JPM","BAC","WFC","GS","MS","C","USB","PNC","TFC","KEY","RF","FITB","ZION",
    "BLK","BX","KKR","APO","ARES","TPG","CG","BAM","IVZ","STT","NTRS","TROW",
    "V","MA","AXP","PYPL","SQ","AFRM","UPST","SOFI","NU","COIN","ALLY","COF",
    "ICE","CME","CBOE","NDAQ","MKTX","GPN","FIS","FISV","JKHY","WEX","FOUR",
    # Healthcare / biotech / pharma
    "JNJ","PFE","MRK","ABBV","BMY","LLY","AMGN","GILD","BIIB","REGN","VRTX",
    "MRNA","BNTX","NVAX","ILMN","IQV","DHR","TMO","A","HOLX","IDXX","ZBH",
    "HCA","UNH","CVS","CI","HUM","MOH","CNC","ELV","MCK","ABC","CAH","TDOC",
    # Consumer / retail
    "AMZN","WMT","TGT","COST","HD","LOW","TJX","ROST","DG","DLTR","FIVE",
    "MCD","SBUX","CMG","YUM","DPZ","QSR","JACK","WEN","DENN","DRI","EAT",
    "NKE","LULU","PVH","RL","UAA","UA","VF","HBI","GPS","M","KSS","JWN",
    "NFLX","DIS","PARA","WBD","CMCSA","FOXA","FOX","AMC","CNK","IMAX","LGF",
    "UBER","LYFT","ABNB","BKNG","EXPE","PCLN","TRIP","DKNG","PENN","MGM","LVS",
    # Industrials / defence / energy
    "BA","LMT","NOC","RTX","GD","HII","TDG","HEI","HEICO","AXON","CACI",
    "CAT","DE","CMI","PH","ITW","EMR","HON","GE","ETN","ROK","AME","FTV",
    "XOM","CVX","COP","SLB","HAL","BKR","MPC","PSX","VLO","HES","DVN","PXD",
    "NEE","DUK","SO","D","AEP","EXC","XEL","PCG","ED","PPL","AWK","CEG",
    # Materials / diversified
    "APD","LIN","DD","DOW","LYB","ALB","FCX","NEM","AEM","GOLD","WPM","AG",
    # Communications / media
    "T","VZ","TMUS","LUMN","DISH","CHTR","CABO","WOW","ATUS","LBRDA","SIRI",
    # Real estate
    "AMT","PLD","CCI","EQIX","SBAC","DLR","ARE","AVB","EQR","MAA","PSA","EXR",
    # Popular growth / retail-investor stocks
    "RIVN","LCID","NIO","LI","XPEV","NKLA","WKHS","GOEV","FFIE","MULN",
    "BABA","JD","PDD","BIDU","NTES","TME","IQ","VIPS","DIDI","KWEB",
    "MSTR","RIOT","MARA","HUT","BTBT","CLSK","BTCS","CIFR","WULF","IREN",
    "AMC","GME","BB","BBBYQ","CLOV","WISH","EXPR","KOSS","SPCE","NKLA",
    "IONQ","RGTI","QUBT","QBTS","ARQQ","SOUN","BBAI","GFAI","AIXI","AITX",
    "DJT","DWAC","PHUN","NKTR","CLNN","SAVA","ACAD","SAGE","SRPT","RARE",
    # Common ETFs frequently cited in financial news
    "SPY","QQQ","DIA","IWM","IWR","IWS","VTI","VOO","VGT","VUG","VTV",
    "GLD","SLV","GDX","GDXJ","IAU","USO","UNG","DBB","DBA","PDBC",
    "TLT","IEF","SHY","AGG","BND","HYG","LQD","JNK","EMB","VCIT",
    "XLK","XLF","XLV","XLE","XLI","XLY","XLP","XLU","XLB","XLRE","XLC",
    "ARKK","ARKG","ARKW","ARKF","ARKQ","ARKX","BOTZ","ROBO","IRBO",
    "SOXL","SOXS","TQQQ","SQQQ","SPXL","SPXS","UVXY","SVXY","VXX","VIXY",
])

# True tickers that are also common English words / abbreviations — skip bare-ALLCAPS matches.
# Explicit $TICKER notation always overrides this list.
_SKIP_TICKERS = frozenset([
    # Single letters
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    # Common words that appear all-caps in headlines
    "AM","AN","AS","AT","BE","BY","DO","GO","IF","IN","IS","IT",
    "ME","MY","NO","OF","ON","OR","SO","TO","UP","US","WE",
    "ARE","AND","BUT","FOR","NOT","OUT","THE","WAS","WHO",
    "ALL","CAN","DID","GET","GOT","HAS","HAD","HIM","HIS",
    "HOW","LET","NOW","OUR","OWN","PUT","SAY","SEE","SET",
    "SHE","THE","TRY","USE","WHO","WHY","YES","YET","HER",
    "NEW","OLD","BIG","TOP","DAY","WAY","END","WIN","LET",
    "CUT","HIT","RUN","BIT","ASK","ADD","PAY","BUY","OWN",
    # Finance abbreviations that are NOT equity tickers
    "IPO","CEO","CFO","CTO","COO","CIO","CDO","CAO","EVP","SVP","CMO",
    "GDP","CPI","PPI","PCE","NFP","PMI","ISM","ADP","FED","SEC","DOJ",
    "NYSE","CBOE","CFTC","IMF","WTO","WHO","NATO","ESG","SRI","DEI",
    "PE","EV","FCF","OCF","ICF","EPS","BPS","DPS","NII","NIM",
    "ROE","ROI","ROA","ROIC","ROCE","WACC","DCF","LBO","MBO","IRR","NPV",
    "EBIT","EBITA","EBITDA","NOPAT","NAV","AUM","NWC","CAPEX","OPEX",
    "YTD","YOY","QOQ","MOM","CAGR","TTM","LTM","FY","Q1","Q2","Q3","Q4",
    "UK","EU","EM","DM","VC","LP","LLC","LLP","LTD","INC","SPAC",
    "REIT","ADR","GDR","OTC","HFT","MMT","SPX","VIX","ATH","ATL",
    "MA","RSI","MACD","ATR","EMA","SMA","VWAP","OI","IV","BB","TD",
    "AI","ML","DL","NLP","LLM","AGI","API","SDK","GPU","CPU","RAM",
    "OK","HI","SO","GO","DO","NO","IT","US","BY","AT","BE","TO","VS",
    "KPI","OKR","CRM","ERP","SCM","WMS","TMS","LMS","HCM","SOP",
    "USD","EUR","GBP","JPY","CNY","CAD","AUD","CHF","INR","BRL","MXN",
    "BTC","ETH","XRP","SOL","ADA","DOT","NFT","DAO","WEB3","DEFI",
    "RE","PR","IR","HR","IT","IS","ID","IP","AR","AP","PO","SO","AC",
    "CX","FX","GX","HX","IX","JX","KX","LX","MX","NX","OX","PX","RX",
    "SAYS","SAID","ALSO","BEEN","EVEN","JUST","LIKE","WILL","OVER","THAN",
    "FROM","WITH","HAVE","THAT","THIS","THEY","WERE","WHEN","WHAT","SOME",
    "EACH","INTO","ONLY","VERY","MUCH","MANY","MOST","SUCH","BOTH","ZERO",
])

# Company name → ticker mapping (case-insensitive word match in headlines).
# Covers the companies most commonly referenced in financial/tech news.
_NAME_TICKER = {
    # Mega-cap tech
    "nvidia":["NVDA"], "microsoft":["MSFT"], "apple":["AAPL"],
    "alphabet":["GOOGL"], "google":["GOOGL"], "amazon":["AMZN"],
    "meta":["META"], "facebook":["META"], "tesla":["TSLA"],
    "broadcom":["AVGO"], "oracle":["ORCL"], "salesforce":["CRM"],
    "adobe":["ADBE"], "qualcomm":["QCOM"], "intel":["INTC"],
    "amd":["AMD"], "arm":["ARM"], "dell":["DELL"],
    "ibm":["IBM"], "cisco":["CSCO"], "palantir":["PLTR"],
    "snowflake":["SNOW"], "databricks":[], "openai":[], "anthropic":[],
    "cloudflare":["NET"], "datadog":["DDOG"], "mongodb":["MDB"],
    "crowdstrike":["CRWD"], "palo alto":["PANW"], "fortinet":["FTNT"],
    "zscaler":["ZS"], "okta":["OKTA"], "twilio":["TWLO"],
    "servicenow":["NOW"], "workday":["WDAY"], "intuit":["INTU"],
    "autodesk":["ADSK"], "cadence":["CDNS"], "synopsis":["SNPS"],
    "applied materials":["AMAT"], "lam research":["LRCX"],
    "kla":["KLAC"], "micron":["MU"], "marvell":["MRVL"],
    "samsung":[], "tsmc":["TSM"], "asml":["ASML"],
    "super micro":["SMCI"], "supermicro":["SMCI"],
    # Finance
    "jpmorgan":["JPM"], "goldman":["GS"], "morgan stanley":["MS"],
    "bank of america":["BAC"], "wells fargo":["WFC"], "citigroup":["C"],
    "blackrock":["BLK"], "blackstone":["BX"], "kkr":["KKR"],
    "visa":["V"], "mastercard":["MA"], "paypal":["PYPL"],
    "coinbase":["COIN"], "robinhood":["HOOD"],
    # Healthcare / pharma / biotech
    "johnson":["JNJ"], "pfizer":["PFE"], "merck":["MRK"],
    "abbvie":["ABBV"], "bristol myers":["BMY"], "eli lilly":["LLY"],
    "amgen":["AMGN"], "gilead":["GILD"], "moderna":["MRNA"],
    "biogen":["BIIB"], "regeneron":["REGN"], "vertex":["VRTX"],
    "unitedhealth":["UNH"], "cvs":["CVS"],
    # Consumer / retail / media
    "walmart":["WMT"], "target":["TGT"], "costco":["COST"],
    "home depot":["HD"], "netflix":["NFLX"], "disney":["DIS"],
    "spotify":["SPOT"], "uber":["UBER"], "lyft":["LYFT"],
    "airbnb":["ABNB"], "doordash":["DASH"], "instacart":["CART"],
    "shopify":["SHOP"], "mercadolibre":["MELI"],
    "nike":["NKE"], "lululemon":["LULU"],
    # Industrials / energy / defence
    "boeing":["BA"], "lockheed":["LMT"], "northrop":["NOC"],
    "raytheon":["RTX"], "general dynamics":["GD"],
    "caterpillar":["CAT"], "deere":["DE"], "honeywell":["HON"],
    "general electric":["GE"], "eaton":["ETN"],
    "exxon":["XOM"], "chevron":["CVX"], "conocophillips":["COP"],
    "schlumberger":["SLB"],
    # Telecom / comms
    "at&t":["T"], "verizon":["VZ"], "t-mobile":["TMUS"],
    "comcast":["CMCSA"],
    # EV / auto
    "rivian":["RIVN"], "lucid":["LCID"],
    "gm":["GM"], "ford":["F"],
    # Chinese tech
    "alibaba":["BABA"], "tencent":["TCEHY"],
    "baidu":["BIDU"], "jd.com":["JD"], "pinduoduo":["PDD"],
}

_TICKER_SET_CACHE = None


def _load_ticker_set():
    """Return known-ticker frozenset; try Wikipedia S&P 500 on first call."""
    global _TICKER_SET_CACHE
    if _TICKER_SET_CACHE is not None:
        return _TICKER_SET_CACHE
    extra = set()
    try:
        import pandas as pd
        tbl = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )[0]
        extra = set(tbl["Symbol"].str.upper().str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass
    _TICKER_SET_CACHE = _BASE_TICKERS | frozenset(extra)
    return _TICKER_SET_CACHE


# ── Extra stop words ──────────────────────────────────────────────────────────
_XSTOP = {
    # Common filler words
    "said","says","say","will","one","new","also","may","even","get","got",
    "like","just","year","years","time","way","now","day","back","first",
    "last","still","could","would","should","much","many","more","s","t",
    "re","ve","m","us","it","its","the","a","an","and","or","but","in",
    "on","at","to","for","of","with","from","this","that","is","was","are",
    "been","have","has","had","do","does","did","not","no","be","by","as",
    "up","out","so","if","he","she","they","we","i","you","can","about",
    "after","before","over","than","then","there","some","when","where","who",
    "which","what","how","all","any","other","into","than","their","them",
    "these","those","only","your","our","per","via","amid",
    "new","says","said","reports","according","week",
    "months","month","latest","report","story","article","read","more",
    # News wire / agency names
    "reuters","bloomberg","associated","press","ap",
    # News outlet component words — appear in titles/snippets as attribution
    # and dominate word clouds without adding analytical value.
    "motley","fool","foolish",                          # The Motley Fool
    "seeking","alpha","alphaai",                        # Seeking Alpha + join artifact
    "benzinga",                                         # Benzinga
    "investopedia",                                     # Investopedia
    "thestreet","kiplinger","barron","barrons",         # TheStreet, Kiplinger, Barron's
    "marketwatch",                                      # MarketWatch
    "zerohedge","hedge",                                # Zero Hedge
    "techcrunch","wired","verge",                       # Tech outlets
    "cnbc","msnbc","nbc",                               # Broadcast
    "nai",                                              # Fragment artifact from RSS join
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA FETCHING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Shared browser User-Agent — many RSS endpoints reject Python/feedparser UA
_NEWS_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36")

# Finance RSS feed config: (outlet, url_template, query_aware)
#   query_aware=True  → URL contains {query} placeholder; returns relevant results
#                       directly; relevance token-filter is skipped.
#   query_aware=False → static category feed; token-based relevance filter applied.
_FINANCE_FEEDS_CFG = [
    # CNBC's &keywords= parameter does NOT filter server-side — it returns the
    # same general business feed regardless of the query, so query_aware=False
    # ensures client-side word-boundary relevance filtering is applied.
    ("CNBC",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml"
     "?partnerId=wrss01&id=100003114",
     False),
    ("Yahoo Finance",
     "https://finance.yahoo.com/rss/",
     False),
    ("Benzinga",
     "https://www.benzinga.com/feed",
     False),
    ("Investing.com",
     "https://www.investing.com/rss/news.rss",
     False),
]


def fetch_google_news(query, max_items=60):
    items = []
    try:
        import feedparser
        url  = (f"https://news.google.com/rss/search"
                f"?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en")
        # Google News rejects bare feedparser requests without a browser UA
        feed = feedparser.parse(url, request_headers={
            "User-Agent": _NEWS_UA,
            "Accept": "application/rss+xml, application/xml, text/xml",
        })
        if not feed.entries:
            # Fallback: raw requests download then feedparser.parse from string
            r = requests.get(url, timeout=12, headers={"User-Agent": _NEWS_UA})
            feed = feedparser.parse(r.text)
        for e in feed.entries[:max_items]:
            pub = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime.datetime(*e.published_parsed[:6])
            snip = _html.unescape(re.sub(r"<[^>]+>", "", getattr(e, "summary", "") or ""))[:300]
            src  = "Google News"
            if hasattr(e, "source") and isinstance(e.source, dict):
                src = e.source.get("title", src)
            items.append({"source": "news", "outlet": src,
                          "title": e.get("title", ""), "snippet": snip,
                          "url": e.get("link", ""), "date": pub, "score": None})
    except Exception as ex:
        print(f"  Google News: {ex}")
    return items


def fetch_reddit(query, max_items=40):
    items = []
    try:
        url = (f"https://www.reddit.com/search.json"
               f"?q={quote_plus(query)}&sort=new&limit={max_items}&t=month")
        r = requests.get(url, headers={"User-Agent": "TopicSentiment/1.0"}, timeout=10)
        for post in r.json().get("data", {}).get("children", []):
            d = post.get("data", {})
            dt = datetime.datetime.utcfromtimestamp(d.get("created_utc", 0))
            items.append({
                "source": "reddit", "outlet": f"r/{d.get('subreddit','?')}",
                "title":   d.get("title", ""),
                "snippet": (d.get("selftext", "") or "")[:300],
                "url":  f"https://reddit.com{d.get('permalink','')}",
                "date": dt, "score": d.get("score", 0),
            })
    except Exception as ex:
        print(f"  Reddit: {ex}")
    return items


def fetch_hackernews(query, max_items=30, days=30):
    items = []
    try:
        import time as _time
        cutoff = int(_time.time()) - (days * 86400)
        url = (f"https://hn.algolia.com/api/v1/search"
               f"?query={quote_plus(query)}&tags=story&hitsPerPage={max_items}"
               f"&numericFilters=created_at_i>{cutoff}")
        r = requests.get(url, timeout=10)
        for h in r.json().get("hits", []):
            dt = None
            if h.get("created_at_i"):
                dt = datetime.datetime.utcfromtimestamp(h["created_at_i"])
            items.append({
                "source":  "hackernews", "outlet": "Hacker News",
                "title":   h.get("title", ""), "snippet": "",
                "url":     h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID','')}",
                "date":    dt, "score": h.get("points", 0),
            })
    except Exception as ex:
        print(f"  Hacker News: {ex}")
    return items


def fetch_bing_news(query, max_items=40, days=30):
    """
    Bing News RSS — query-aware feed, different source pool than Google News.
    Provides strong overlap with Reuters, AP, and regional outlets Google
    may under-represent.
    """
    items = []
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    try:
        import feedparser
        url  = f"https://www.bing.com/news/search?q={quote_plus(query)}&format=rss"
        r    = requests.get(url, timeout=12, headers={"User-Agent": _NEWS_UA})
        feed = feedparser.parse(r.text)
        for e in feed.entries[:max_items]:
            pub = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime.datetime(*e.published_parsed[:6])
                if pub < cutoff:
                    continue
            snip = _html.unescape(
                re.sub(r"<[^>]+>", "", getattr(e, "summary", "") or ""))[:300]
            # Bing wraps the real outlet in the title ("Headline - Outlet")
            title = e.get("title", "")
            outlet = "Bing News"
            if " - " in title:
                *head, tail = title.rsplit(" - ", 1)
                outlet = tail.strip()
                title  = head[0].strip()
            items.append({"source": "bing", "outlet": outlet,
                          "title": title, "snippet": snip,
                          "url": e.get("link", ""), "date": pub, "score": None})
    except Exception as ex:
        print(f"  Bing News: {ex}")
    return items


def _parse_rss_feed(outlet, url, query_tokens, cutoff,
                    max_items=20, query_aware=False):
    """
    Fetch one RSS feed and return items relevant to the query.

    query_aware=True  → URL already returns query-filtered results;
                        token relevance check is skipped (avoids false misses).
    query_aware=False → static category feed; at least one query token must
                        appear in title+snippet (case-insensitive).
    """
    items = []
    try:
        import feedparser
        r    = requests.get(url, timeout=12, headers={"User-Agent": _NEWS_UA})
        feed = feedparser.parse(r.text)
        for e in feed.entries:
            if len(items) >= max_items:
                break
            pub = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime.datetime(*e.published_parsed[:6])
                if cutoff and pub < cutoff:
                    continue
            title = e.get("title", "")
            snip  = _html.unescape(
                re.sub(r"<[^>]+>", "", getattr(e, "summary", "") or ""))[:300]
            # Relevance gate — only for static (non-query-aware) feeds.
            # Uses word-boundary regex so "gene" does NOT match "general" or
            # "generation", "edit" does not match "editor", etc.
            if not query_aware:
                combined = (title + " " + snip).lower()
                if not any(
                    re.search(r'\b' + re.escape(tok) + r'\b', combined)
                    for tok in query_tokens
                ):
                    continue
            items.append({"source": "finance", "outlet": outlet,
                          "title": title, "snippet": snip,
                          "url": e.get("link", ""), "date": pub, "score": None})
    except Exception as ex:
        print(f"  {outlet}: {ex}")
    return items


def fetch_finance_feeds(query, max_per_feed=20, days=30):
    """
    Fetch from finance RSS feeds in parallel.
    Query-aware feeds (CNBC keyword search) get their own targeted URL.
    Static category feeds (Yahoo Finance, Benzinga, Investing.com) are
    filtered by query relevance after fetching.
    Individual outlet names are preserved in each item's 'outlet' field.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)

    # Relevance tokens: lowercase words ≥4 chars (avoids short noise words)
    # plus explicit ticker tokens from the original query (e.g. "NVDA").
    # Word-boundary matching is applied in _parse_rss_feed so "gene" will NOT
    # match "general", "generation", "generate", etc.
    raw_tokens    = [t for t in re.sub(r"[^a-z0-9\s]", "",
                                       query.lower()).split() if len(t) >= 4]
    ticker_tokens = [t.lower() for t in re.findall(r'\b[A-Z]{1,5}\b', query)]
    query_tokens  = list(dict.fromkeys(raw_tokens + ticker_tokens))
    # For very short queries (e.g. "AI"), fall back to accepting all tokens
    if not query_tokens:
        query_tokens = [t for t in re.sub(r"[^a-z0-9\s]", "",
                                          query.lower()).split() if len(t) >= 2]

    items = []
    with ThreadPoolExecutor(max_workers=len(_FINANCE_FEEDS_CFG)) as pool:
        futures = {}
        for outlet, url_tpl, is_qa in _FINANCE_FEEDS_CFG:
            url = url_tpl.format(query=quote_plus(query)) if is_qa else url_tpl
            futures[pool.submit(
                _parse_rss_feed, outlet, url,
                query_tokens, cutoff, max_per_feed, is_qa
            )] = outlet
        for fut in as_completed(futures):
            try:
                items.extend(fut.result())
            except Exception as ex:
                print(f"  {futures[fut]}: {ex}")
    return items


def _dedup_urls(items):
    """
    Remove duplicate items by URL (keep first occurrence).
    Items without a URL are always kept.
    """
    seen = set()
    out  = []
    for it in items:
        url = (it.get("url") or "").strip()
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        out.append(it)
    return out


def fetch_trends(query):
    if not _TRENDS:
        return []
    try:
        pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        pt.build_payload([query[:100]], timeframe="today 3-m")
        df = pt.interest_over_time()
        if df.empty:
            return []
        col = df.columns[0]
        return [{"date": str(idx.date()), "value": int(row[col])}
                for idx, row in df.iterrows()
                if not row.get("isPartial", False)]
    except Exception as ex:
        print(f"  Google Trends: {ex}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENTIMENT SCORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_all(items, query):
    """
    Add 'sentiment' (VADER) and 'sentiment_fb' (FinBERT) dicts + '_text'
    to every item.  Returns items.
    """
    q_words = set(re.sub(r"[^a-z\s]", "", query.lower()).split())
    texts = []
    for it in items:
        text = (it.get("title", "") + " " + it.get("snippet", "")).strip()
        it["sentiment"] = _score(text)
        it["_text"]     = text
        it["_qstop"]    = q_words
        texts.append(text)

    # FinBERT batch — single forward pass over all items
    fb_scores = _score_finbert_batch(texts)
    for it, fb in zip(items, fb_scores):
        it["sentiment_fb"] = fb

    return items


def _avg(items, key="sentiment"):
    """Mean compound score for *key* ('sentiment' or 'sentiment_fb')."""
    vals = [i[key]["compound"] for i in items if i.get(key)]
    return round(sum(vals) / len(vals), 4) if vals else None


def _composite_compound(vader, fb):
    """
    Weighted composite of VADER and FinBERT compound scores.
    FinBERT carries 65 % weight (better on financial prose);
    VADER carries 35 % (broader vocabulary / social media coverage).
    Falls back gracefully when only one model scored the item.
    """
    if vader is None and fb is None:
        return None
    if vader is None:
        return fb
    if fb is None:
        return vader
    return round(0.65 * fb + 0.35 * vader, 4)


def _avg_composite(items):
    """Mean composite score across all items."""
    vals = [c for c in (
        _composite_compound(
            (it.get("sentiment")    or {}).get("compound"),
            (it.get("sentiment_fb") or {}).get("compound"),
        )
        for it in items
    ) if c is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WORD FREQUENCY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def word_freq(items, filt=None, top=100, query=""):
    q_words = set(re.sub(r"[^a-z\s]", "", query.lower()).split())
    # Dynamically derive outlet stop tokens from the item pool so that any
    # outlet name appearing in titles/snippets is excluded automatically.
    outlet_tokens: set[str] = set()
    for it in items:
        for tok in re.sub(r"[^a-z\s]", " ", (it.get("outlet") or "").lower()).split():
            if len(tok) >= 3:
                outlet_tokens.add(tok)
    stops = (_WC_STOPS | _XSTOP | q_words | outlet_tokens) if _WC else (_XSTOP | q_words | outlet_tokens)
    counts  = Counter()
    for it in items:
        if filt and not filt(it):
            continue
        txt = re.sub(r"https?\S+", "", it.get("_text", ""))
        txt = _html.unescape(txt)                          # &nbsp; → space, &amp; → &, etc.
        txt = re.sub(r"[^a-zA-Z\s]", " ", txt).lower()
        for w in txt.split():
            if len(w) >= 2 and w not in stops:   # allow 2-char terms (AI, ML, EV…)
                counts[w] += 1
    return counts.most_common(top)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WORD CLOUD PNG → base64
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _palette(kind):
    import random
    pal = {
        "positive": ["#10b981","#34d399","#6ee7b7","#a7f3d0","#059669","#047857"],
        "negative": ["#ef4444","#f87171","#fca5a5","#dc2626","#f97316","#b91c1c"],
        "neutral":  ["#60a5fa","#818cf8","#a78bfa","#38bdf8","#7dd3fc","#c4b5fd"],
    }
    chosen = pal.get(kind, pal["neutral"])
    def fn(word, font_size, position, orientation, random_state=None, **kwargs):
        return random.choice(chosen)
    return fn


def make_wordcloud(freq_list, kind="neutral", w=700, h=320):
    if not _WC or not freq_list:
        return None
    try:
        wc = WordCloud(
            width=w, height=h, background_color="#0f172a",
            max_words=80, collocations=True, color_func=_palette(kind),
            prefer_horizontal=0.7, min_font_size=10,
        ).generate_from_frequencies(dict(freq_list))
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        fig.patch.set_facecolor("#0f172a")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f172a", dpi=100)
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as ex:
        print(f"  Word cloud ({kind}): {ex}")
        return None


def build_css_wordcloud(freq_list, kind="neutral", max_words=80):
    """Pure HTML/CSS word cloud — no library dependencies, always works."""
    if not freq_list:
        return None
    import random
    palettes = {
        "positive": ["#10b981","#34d399","#6ee7b7","#059669","#a7f3d0","#047857"],
        "negative": ["#ef4444","#f87171","#fca5a5","#dc2626","#f97316","#b91c1c"],
        "neutral":  ["#60a5fa","#818cf8","#a78bfa","#38bdf8","#7dd3fc","#c4b5fd"],
    }
    colors = palettes.get(kind, palettes["neutral"])

    words = freq_list[:max_words]
    if not words:
        return None

    max_freq = words[0][1] if words else 1
    min_freq = words[-1][1] if len(words) > 1 else max_freq
    freq_range = max(max_freq - min_freq, 1)

    # Shuffle so large words don't all cluster at left
    shuffled = list(words)
    random.shuffle(shuffled)

    spans = []
    for word, freq in shuffled:
        # Map frequency → font size 13px–52px
        t = (freq - min_freq) / freq_range
        size = int(13 + t * 39)
        opacity = 0.65 + t * 0.35
        color = random.choice(colors)
        spans.append(
            f'<span style="font-size:{size}px;color:{color};opacity:{opacity:.2f};'
            f'padding:4px 6px;display:inline-block;cursor:default;'
            f'transition:transform 0.15s,opacity 0.15s;line-height:1.2;'
            f'font-weight:{400 + int(t * 300)};" '
            f'onmouseover="this.style.transform=\'scale(1.22)\';this.style.opacity=\'1\'" '
            f'onmouseout="this.style.transform=\'scale(1)\';this.style.opacity=\'{opacity:.2f}\'"'
            f'>{word}</span>'
        )

    return (
        '<div style="background:#0f172a;border-radius:8px;padding:18px 14px;'
        'min-height:140px;display:flex;flex-wrap:wrap;align-items:center;'
        'justify-content:center;gap:2px;overflow:hidden">'
        + "".join(spans)
        + "</div>"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TICKER CLOUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_ticker_mentions(items, query):
    """
    Scan titles + snippets for stock ticker mentions.

    Two passes:
      • Explicit  $AAPL  — always accepted (1–5 uppercase letters after $)
      • Bare       AAPL  — accepted only if in the known ticker set
                          AND not in the false-positive skip list

    Query tokens are excluded to avoid self-inflation (e.g. searching "NVDA"
    shouldn't make NVDA dominate the ticker cloud trivially).

    Returns: dict { ticker: {"count": N, "sentiments": [float, ...]} }
    """
    known = _load_ticker_set()

    # Uppercase tokens from the query itself → exclude from cloud
    q_skip = frozenset(re.findall(r"\b([A-Z]{1,5})\b", query.upper()))

    pat_dollar = re.compile(r"\$([A-Z]{1,5})\b")
    pat_bare   = re.compile(r"\b([A-Z]{1,5})\b")

    # Build a sorted list of company names (longest first) for efficient scanning
    name_list = sorted(_NAME_TICKER.keys(), key=len, reverse=True)

    # Lowercase query tokens to exclude from name matches
    q_lower_tokens = set(query.lower().split())

    mentions = {}
    for item in items:
        text = _html.unescape(" ".join(filter(None, [
            item.get("title", ""), item.get("snippet", ""),
        ])))
        compound = (item.get("sentiment") or {}).get("compound")

        found = set()

        # Pass 1: explicit $TICKER — unambiguous
        for m in pat_dollar.finditer(text):
            t = m.group(1)
            if t not in _SKIP_TICKERS and t not in q_skip:
                found.add(t)

        # Pass 2: bare ALL-CAPS — validate against known set
        for m in pat_bare.finditer(text):
            t = m.group(1)
            if (t in known
                    and t not in _SKIP_TICKERS
                    and t not in q_skip):
                found.add(t)

        # Pass 3: company name scan (catches "Nvidia", "Microsoft" etc. in news prose)
        text_lower = text.lower()
        for name in name_list:
            if name in q_lower_tokens:
                continue                    # skip if name is literally the search query
            # Word-boundary check: character before/after must not be alpha
            idx = text_lower.find(name)
            while idx != -1:
                before_ok = (idx == 0 or not text_lower[idx - 1].isalpha())
                after_ok  = (idx + len(name) >= len(text_lower)
                             or not text_lower[idx + len(name)].isalpha())
                if before_ok and after_ok:
                    for t in _NAME_TICKER[name]:
                        # Skip single-letter tickers even on company-name match —
                        # they're too ambiguous (V=Visa, F=Ford, C=Citigroup).
                        if t and len(t) > 1 and t not in q_skip:
                            found.add(t)
                    break
                idx = text_lower.find(name, idx + 1)

        for t in found:
            rec = mentions.setdefault(t, {"count": 0, "sentiments": []})
            rec["count"] += 1
            if compound is not None:
                rec["sentiments"].append(compound)

    return mentions


def build_css_ticker_cloud(ticker_mentions, max_tickers=50):
    """
    CSS ticker cloud.  Size = mention count.  Color = avg sentiment.
    Each ticker links to its Yahoo Finance quote page.
    Returns an HTML string, or None if nothing to show.
    """
    if not ticker_mentions:
        return None

    import random
    rng_state = random.Random(42)   # stable shuffle across re-runs

    sorted_t = sorted(ticker_mentions.items(),
                      key=lambda x: x[1]["count"], reverse=True)[:max_tickers]
    if not sorted_t:
        return None

    max_cnt = sorted_t[0][1]["count"]
    min_cnt = sorted_t[-1][1]["count"] if len(sorted_t) > 1 else max_cnt
    cnt_rng = max(max_cnt - min_cnt, 1)

    shuffled = list(sorted_t)
    rng_state.shuffle(shuffled)

    spans = []
    for ticker, data in shuffled:
        cnt   = data["count"]
        sents = data["sentiments"]
        avg_s = sum(sents) / len(sents) if sents else 0.0

        t       = (cnt - min_cnt) / cnt_rng
        size    = int(13 + t * 36)          # 13 – 49 px
        weight  = 500 + int(t * 300)        # 500 – 800
        opacity = 0.60 + t * 0.40           # 0.60 – 1.00
        col     = _col(avg_s)

        url   = f"https://finance.yahoo.com/quote/{ticker}"
        tip   = (f"{cnt} mention{'s' if cnt > 1 else ''} · "
                 f"avg sentiment {avg_s:+.3f} ({_lbl(avg_s)})")

        spans.append(
            f'<a href="{url}" target="_blank" title="{tip}" '
            f'style="font-size:{size}px;color:{col};opacity:{opacity:.2f};'
            f'padding:5px 8px;display:inline-block;text-decoration:none;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
            f'letter-spacing:0.04em;font-weight:{weight};'
            f'transition:transform 0.15s,opacity 0.15s;line-height:1.3;" '
            f'onmouseover="this.style.transform=\'scale(1.28)\';this.style.opacity=\'1\'" '
            f'onmouseout="this.style.transform=\'scale(1)\';this.style.opacity=\'{opacity:.2f}\'">'
            f'{ticker}</a>'
        )

    n   = len(sorted_t)
    leg = ('<span style="color:#10b981">■</span> positive &nbsp;'
           '<span style="color:#94a3b8">■</span> neutral &nbsp;'
           '<span style="color:#ef4444">■</span> negative')

    return (
        f'<div style="font-size:0.68em;color:#475569;margin-bottom:8px">'
        f'{n} ticker{"s" if n > 1 else ""} identified · '
        f'size&nbsp;=&nbsp;frequency · {leg} · click to open Yahoo Finance chart</div>'
        '<div style="background:#0f172a;border-radius:8px;padding:18px 14px;'
        'min-height:100px;display:flex;flex-wrap:wrap;align-items:center;'
        'justify-content:center;gap:2px;">'
        + "".join(spans)
        + "</div>"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SVG GAUGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gauge_svg(compound, size=220, model_label=""):
    R = 88

    def pt(c, r=R):
        θ = (1.0 - c) / 2.0 * math.pi
        return r * math.cos(θ), -r * math.sin(θ)

    def arc(c1, c2, colour):
        x1, y1 = pt(c1); x2, y2 = pt(c2)
        return (f'<path d="M{x1:.1f},{y1:.1f} A{R},{R} 0 0,1 {x2:.1f},{y2:.1f}" '
                f'fill="none" stroke="{colour}" stroke-width="16" stroke-linecap="butt"/>')

    zones = [(-1.0,-0.5,"#ef4444"),(-0.5,-0.1,"#f87171"),
             (-0.1, 0.1,"#64748b"),( 0.1, 0.5,"#34d399"),( 0.5, 1.0,"#10b981")]
    arcs  = "\n  ".join(arc(*z) for z in zones)

    c    = max(-1.0, min(1.0, compound if compound is not None else 0.0))
    nx, ny = pt(c, R - 10)
    col  = _col(compound)
    lbl  = _lbl(compound)
    score_txt = f"{c:+.3f}" if compound is not None else "—"

    # Optional model label tag above score
    label_el = (f'\n  <text x="0" y="-6" text-anchor="middle" fill="#475569"'
                f' font-family="sans-serif" font-size="8.5"'
                f' font-weight="600" letter-spacing="1">{model_label.upper()}</text>'
                if model_label else "")

    # viewBox: x=-110..110, y=-102..54  (height=156 covers arc up to y=-88 and
    # text at y=25 / y=43 with comfortable margin; ratio 156/220 ≈ 0.709)
    return (f'<svg viewBox="-110 -102 220 156" width="{size}" height="{int(size*0.709)}">'
            f'\n  <path d="M-88,0 A88,88 0 0,1 88,0" fill="none" stroke="#1e293b" stroke-width="18"/>'
            f'\n  {arcs}'
            f'\n  <line x1="0" y1="0" x2="{nx:.1f}" y2="{ny:.1f}"'
            f' stroke="{col}" stroke-width="3.5" stroke-linecap="round"/>'
            f'\n  <circle cx="0" cy="0" r="7" fill="{col}"/>'
            f'{label_el}'
            f'\n  <text x="0" y="25" text-anchor="middle" fill="#f1f5f9"'
            f' font-family="monospace" font-size="19" font-weight="bold">{score_txt}</text>'
            f'\n  <text x="0" y="43" text-anchor="middle" fill="{col}"'
            f' font-family="sans-serif" font-size="10">{lbl}</text>'
            f'\n</svg>')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIMELINE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_timeline(items, days=30):
    """
    Build daily timeline using composite (FinBERT+VADER) compound scores.
    Only includes items dated within the last *days* window to prevent old
    HN / Reddit stories from dominating the chart with deep historical data.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    by_date = {}
    for it in items:
        if not it.get("date"):
            continue
        if it["date"] < cutoff:
            continue
        c = _composite_compound(
            (it.get("sentiment")    or {}).get("compound"),
            (it.get("sentiment_fb") or {}).get("compound"),
        )
        if c is None:
            continue
        d = it["date"].strftime("%Y-%m-%d")
        by_date.setdefault(d, []).append(c)
    return sorted(
        [{"date": d, "avg": round(sum(v)/len(v), 4), "count": len(v)}
         for d, v in by_date.items()],
        key=lambda x: x["date"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_html(query, all_items, trends_data, ts, days=30):
    # Group items by source family
    _NEWS_SRC  = {"news", "bing", "finance"}
    news_items = [i for i in all_items if i["source"] in _NEWS_SRC]
    red_items  = [i for i in all_items if i["source"] == "reddit"]
    hn_items   = [i for i in all_items if i["source"] == "hackernews"]

    # Per-outlet counts for the source breakdown subtitle
    _outlet_counts = {}
    for it in news_items:
        _src = it["source"]
        _label = {"news": "Google", "bing": "Bing", "finance": "Finance"}.get(_src, _src)
        _outlet_counts[_label] = _outlet_counts.get(_label, 0) + 1
    _news_detail = " · ".join(f"{k} {v}" for k, v in _outlet_counts.items())

    # Per-model averages
    vader_overall = _avg(all_items)
    fb_overall    = _avg(all_items, key="sentiment_fb")
    overall       = _avg_composite(all_items)   # primary score used everywhere

    news_avg = _avg_composite(news_items)
    red_avg  = _avg_composite(red_items)
    hn_avg   = _avg_composite(hn_items)

    # pos/neg/neu counts based on composite score
    def _comp(it):
        return _composite_compound(
            (it.get("sentiment")    or {}).get("compound"),
            (it.get("sentiment_fb") or {}).get("compound"),
        )

    pos_n = sum(1 for i in all_items if (_comp(i) or 0.0) >  0.05)
    neg_n = sum(1 for i in all_items if (_comp(i) or 0.0) < -0.05)
    neu_n = len(all_items) - pos_n - neg_n
    total = len(all_items)

    has_fb = fb_overall is not None

    # ── Word clouds ─────────────────────────────────────────────────────────
    # Use a relaxed VADER threshold (±0.02) so that mildly toned technical
    # content (typical for topics like "AI", "Fed rates") still splits into
    # positive and negative clouds rather than collapsing entirely to neutral.
    pos_filt = lambda i: i.get("sentiment") and i["sentiment"]["compound"] >  0.02
    neg_filt = lambda i: i.get("sentiment") and i["sentiment"]["compound"] < -0.02

    def _make_cloud(freq_list, kind):
        """Try PNG first; fall back to CSS if PNG unavailable."""
        png = make_wordcloud(freq_list, kind)
        if png:
            return ("png", png)
        css = build_css_wordcloud(freq_list, kind)
        if css:
            return ("css", css)
        return (None, None)

    print("  Building positive word cloud…", end=" ", flush=True)
    wc_pos_type, wc_pos = _make_cloud(word_freq(all_items, pos_filt, 80, query), "positive")
    print("done")
    print("  Building negative word cloud…", end=" ", flush=True)
    wc_neg_type, wc_neg = _make_cloud(word_freq(all_items, neg_filt, 80, query), "negative")
    print("done")
    # Always build the combined cloud — shown as a third panel or sole panel
    # when there aren't enough pos/neg items for the split view.
    print("  Building combined word cloud…", end=" ", flush=True)
    wc_all_type, wc_all = _make_cloud(word_freq(all_items, None, 100, query), "neutral")
    print("done")

    # ── Ticker cloud ─────────────────────────────────────────────────────────
    print("  Building ticker cloud…", end=" ", flush=True)
    ticker_mentions = extract_ticker_mentions(all_items, query)
    ticker_html     = build_css_ticker_cloud(ticker_mentions)
    n_tickers = len(ticker_mentions)
    print(f"{n_tickers} ticker{'s' if n_tickers != 1 else ''} found")

    if ticker_html:
        ticker_section = f"""
<div class="card section">
  <h2>Associated Tickers</h2>
  {ticker_html}
</div>"""
    else:
        ticker_section = ""

    # ── Timeline + trends JSON ───────────────────────────────────────────────
    timeline_json = json.dumps(build_timeline(all_items, days=days))
    trends_json   = json.dumps(trends_data)

    # ── Gauge(s) + source bars ───────────────────────────────────────────────
    if has_fb:
        # Dual gauges: VADER  |  FinBERT
        # Labels are rendered as HTML above each SVG (not inside the viewBox) so
        # they stay readable; min-width forces the card wider than its 210px default.
        def _gauge_col(svg_html, label):
            return (
                '<div style="text-align:center">'
                f'<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
                f'color:#94a3b8;text-transform:uppercase;margin-bottom:5px">{label}</div>'
                f'{svg_html}'
                '</div>'
            )
        gauge_html = (
            '<div style="display:flex;gap:24px;justify-content:center;'
            'align-items:flex-start;min-width:350px">'
            + _gauge_col(gauge_svg(vader_overall, size=160), "VADER")
            + _gauge_col(gauge_svg(fb_overall,    size=160), "FinBERT")
            + '</div>'
        )
    else:
        gauge_html = gauge_svg(overall, size=200)

    # Per-source-family averages for the breakdown bars
    gnews_items   = [i for i in all_items if i["source"] == "news"]
    bing_items    = [i for i in all_items if i["source"] == "bing"]
    finance_items = [i for i in all_items if i["source"] == "finance"]

    sources = [
        ("Google News",   _avg_composite(gnews_items),   len(gnews_items)),
        ("Bing News",     _avg_composite(bing_items),    len(bing_items)),
        ("Finance Feeds", _avg_composite(finance_items), len(finance_items)),
        ("Reddit",        red_avg,                       len(red_items)),
        ("Hacker News",   hn_avg,                        len(hn_items)),
    ]
    # Drop sources with 0 items so the bar chart stays clean
    sources = [(n, a, c) for n, a, c in sources if c > 0]

    def src_bar(name, avg, count):
        if avg is not None:
            pct = int((avg + 1) / 2 * 100)
            col = _col(avg)
            val = f"{avg:+.3f}"
        else:
            pct, col, val = 50, "#475569", "—"
        return f"""
        <div style="margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;margin-bottom:5px">
            <span style="color:#cbd5e1;font-size:0.84em">{name}</span>
            <span style="color:{col};font-size:0.84em;font-family:monospace">{val}
              <span style="color:#475569;font-weight:400"> ({count})</span></span>
          </div>
          <div style="background:#0f172a;border-radius:4px;height:7px;overflow:hidden">
            <div style="width:{pct}%;background:{col};height:100%;border-radius:4px"></div>
          </div>
        </div>"""

    bars_html = "".join(src_bar(*s) for s in sources)

    # ── Mentions table ───────────────────────────────────────────────────────
    def _single_badge(s, model_tag=""):
        if not s:
            return ""
        c   = s["compound"]
        col = _col(c)
        tag = (f'<span style="font-size:0.62em;opacity:0.55;margin-left:3px">'
               f'{model_tag}</span>' if model_tag else "")
        return (f'<span style="display:inline-block;padding:2px 7px;border-radius:4px;'
                f'background:{col}22;color:{col};font-size:0.75em;font-weight:600;'
                f'font-family:monospace;white-space:nowrap">'
                f'{_lbl(c)}&nbsp;{c:+.3f}{tag}</span>')

    def badge(it):
        vader_s = it.get("sentiment")
        fb_s    = it.get("sentiment_fb")
        if has_fb and fb_s:
            v_part = _single_badge(vader_s, "V")
            f_part = _single_badge(fb_s,    "F")
            return (f'<div style="display:flex;flex-direction:column;gap:3px;'
                    f'align-items:center">{f_part}{v_part}</div>')
        return _single_badge(vader_s) or '<span style="color:#334155">—</span>'

    src_icon = {"news": "📰", "bing": "📰", "finance": "📰",
                "reddit": "💬", "hackernews": "🟠"}
    src_col  = {"news": "#3b82f6", "bing": "#60a5fa", "finance": "#818cf8",
                "reddit": "#f97316", "hackernews": "#e8630a"}

    rows = ""
    for it in sorted(all_items, key=lambda x: x.get("date") or datetime.datetime.min, reverse=True)[:120]:
        title = it.get("title","")
        t60   = title[:85] + ("…" if len(title) > 85 else "")
        url   = it.get("url","")
        link  = f'<a href="{url}" target="_blank" style="color:#e2e8f0;text-decoration:none;line-height:1.35">{t60}</a>' if url else t60
        dt    = it["date"].strftime("%b %d") if it.get("date") else "—"
        out   = it.get("outlet","")[:22]
        ic    = src_icon.get(it["source"],"")
        sc    = src_col.get(it["source"],"#64748b")
        rows += f"""
          <tr>
            <td style="padding:7px 10px;color:#64748b;white-space:nowrap;font-size:0.8em">{dt}</td>
            <td style="padding:7px 10px;font-size:0.78em">
              <span style="color:{sc}">{ic}</span>
              <span style="color:#64748b;margin-left:4px">{out}</span></td>
            <td style="padding:7px 10px;font-size:0.83em">{link}</td>
            <td style="padding:7px 10px;text-align:center">{badge(it)}</td>
          </tr>"""

    # ── Word cloud section HTML ──────────────────────────────────────────────
    def wc_block(wc_type, wc_data, title, sub):
        if not wc_data:
            return f'<div style="color:#334155;font-size:0.8em;text-align:center;padding:32px">No {title.lower()} data</div>'
        header = (f'<div style="font-size:0.7em;text-transform:uppercase;letter-spacing:0.09em;'
                  f'color:#475569;margin-bottom:4px">{title}</div>'
                  f'<div style="font-size:0.7em;color:#334155;margin-bottom:8px">{sub}</div>')
        if wc_type == "png":
            body = f'<img src="data:image/png;base64,{wc_data}" style="width:100%;border-radius:8px;display:block">'
        else:
            body = wc_data  # already HTML from build_css_wordcloud
        return f"<div>{header}{body}</div>"

    if wc_pos and wc_neg:
        # Full split view + combined below
        wc_html = f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:6px">
          {wc_block(wc_pos_type, wc_pos,"Positive Mentions","Words from items with positive tone")}
          {wc_block(wc_neg_type, wc_neg,"Negative Mentions","Words from items with negative tone")}
        </div>
        {"" if not wc_all else f'<div style="margin-top:14px">{wc_block(wc_all_type, wc_all,"All Mentions Combined","Every item — frequency-weighted")}</div>'}"""
    elif wc_all:
        # Only combined (topic scored mostly neutral by VADER — common for technical content)
        note = ('<div style="font-size:0.73em;color:#475569;margin-bottom:12px">'
                '⚠️ VADER scored most items as neutral — common for technical topics. '
                'Showing combined word cloud.</div>')
        wc_html = f'<div style="margin-top:6px">{note}{wc_block(wc_all_type, wc_all,"All Mentions","All items combined")}</div>'
    else:
        wc_html = ('<p style="color:#475569;text-align:center;padding:24px;font-size:0.85em">'
                   'Not enough text to build a word cloud for this query.</p>')

    # ── Trends section ───────────────────────────────────────────────────────
    trends_section = ""
    if trends_data:
        trends_section = """
<div class="card section">
  <h2>Google Trends — Search Interest (90 days)</h2>
  <canvas id="trendsChart" height="70"></canvas>
</div>"""

    trends_js = ""
    if trends_data:
        trends_js = f"""
const _tr = {trends_json};
if (_tr.length) {{
  new Chart(document.getElementById('trendsChart'), {{
    type:'line',
    data:{{
      labels: _tr.map(d => d.date),
      datasets:[{{
        label:'Search Interest', data:_tr.map(d => d.value),
        borderColor:'#f0b429', backgroundColor:'rgba(240,180,41,0.07)',
        borderWidth:2, tension:0.35, pointRadius:0, fill:true
      }}]
    }},
    options:{{
      responsive:true, maintainAspectRatio:true,
      plugins:{{ legend:{{ labels:{{ color:'#94a3b8',boxWidth:12 }} }} }},
      scales:{{
        x:{{ ticks:{{ color:'#475569',maxTicksLimit:12 }}, grid:{{ color:'#1e293b' }} }},
        y:{{ min:0, max:100, ticks:{{ color:'#475569' }}, grid:{{ color:'#1e293b' }} }}
      }}
    }}
  }});
}}"""

    now_str    = ts.strftime("%Y-%m-%d %H:%M")
    oc         = _col(overall)
    model_note = ("VADER + FinBERT (composite)" if has_fb
                  else "VADER sentiment · no API keys required")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentiment: {query}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0f172a; --card:#1a2540; --border:#1e293b;
    --text:#e2e8f0; --muted:#64748b;
  }}
  *{{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:var(--bg); color:var(--text);
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          padding:24px 28px; }}
  h1   {{ font-size:1.55rem; font-weight:700; color:#f1f5f9 }}
  h2   {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em;
          color:var(--muted); margin-bottom:14px }}
  .card  {{ background:var(--card); border:1px solid var(--border);
            border-radius:12px; padding:22px }}
  .section {{ margin-top:18px }}
  table  {{ width:100%; border-collapse:collapse }}
  thead th {{ font-size:0.7rem; color:var(--muted); text-transform:uppercase;
              letter-spacing:0.06em; padding:8px 10px;
              border-bottom:1px solid var(--border); text-align:left }}
  tbody tr {{ border-bottom:1px solid rgba(255,255,255,0.04) }}
  tbody tr:hover {{ background:rgba(255,255,255,0.025) }}
</style>
</head>
<body>

<!-- ── Header ─────────────────────────────────────────────────────────────── -->
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:22px">
  <div>
    <div style="font-size:0.72rem;color:#334155;text-transform:uppercase;
                letter-spacing:0.09em;margin-bottom:5px">Topic Sentiment Analysis</div>
    <h1 style="margin-bottom:5px">"{query}"</h1>
    <div style="color:#475569;font-size:0.8rem">{total} mentions · {_news_detail} · Reddit {len(red_items)} · HN {len(hn_items)}</div>
  </div>
  <div style="text-align:right;color:#475569;font-size:0.74rem;line-height:1.6">
    Generated {now_str}<br>{model_note}
  </div>
</div>

<!-- ── Top row: gauge / source bars / donut ──────────────────────────────── -->
<div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:16px;align-items:start">

  <!-- Gauge -->
  <div class="card" style="text-align:center;min-width:210px">
    <h2>Overall Sentiment</h2>
    {gauge_html}
    <div style="display:flex;justify-content:center;gap:22px;margin-top:10px">
      <div style="text-align:center">
        <div style="font-size:1.4rem;font-weight:700;color:#10b981">{pos_n}</div>
        <div style="font-size:0.7rem;color:#475569">Positive</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:1.4rem;font-weight:700;color:#64748b">{neu_n}</div>
        <div style="font-size:0.7rem;color:#475569">Neutral</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:1.4rem;font-weight:700;color:#ef4444">{neg_n}</div>
        <div style="font-size:0.7rem;color:#475569">Negative</div>
      </div>
    </div>
  </div>

  <!-- Source breakdown -->
  <div class="card">
    <h2>Source Breakdown</h2>
    {bars_html}
  </div>

  <!-- Sentiment distribution donut -->
  <div class="card">
    <h2>Distribution</h2>
    <div style="position:relative;height:175px">
      <canvas id="donutChart"></canvas>
    </div>
  </div>

</div>

<!-- ── Word clouds ──────────────────────────────────────────────────────────── -->
<div class="card section">
  <h2>Word Clouds</h2>
  {wc_html}
</div>

{ticker_section}

<!-- ── Sentiment timeline ───────────────────────────────────────────────────── -->
<div class="card section">
  <h2>Sentiment Over Time</h2>
  <canvas id="timelineChart" height="80"></canvas>
</div>

{trends_section}

<!-- ── Mentions table ───────────────────────────────────────────────────────── -->
<div class="card section">
  <h2>All Mentions ({total})</h2>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Source</th>
          <th>Headline</th>
          <th style="text-align:center">Sentiment</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>

<script>
// ── Donut ────────────────────────────────────────────────────────────────────
new Chart(document.getElementById('donutChart'), {{
  type:'doughnut',
  data:{{
    labels:['Positive','Neutral','Negative'],
    datasets:[{{
      data:[{pos_n},{neu_n},{neg_n}],
      backgroundColor:['#10b981','#334155','#ef4444'],
      borderWidth:0, hoverOffset:5
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{
      legend:{{ labels:{{ color:'#94a3b8',boxWidth:12,padding:14 }}, position:'right' }},
      tooltip:{{ callbacks:{{ label: ctx => ` ${{ctx.label}}: ${{ctx.raw}}` }} }}
    }},
    cutout:'64%'
  }}
}});

// ── Sentiment timeline ───────────────────────────────────────────────────────
const _tl = {timeline_json};
if (_tl.length > 1) {{
  new Chart(document.getElementById('timelineChart'), {{
    type:'line',
    data:{{
      labels: _tl.map(d => d.date),
      datasets:[
        {{
          label:'Avg Sentiment', data:_tl.map(d => d.avg),
          borderColor:'#6366f1', backgroundColor:'rgba(99,102,241,0.07)',
          borderWidth:2, tension:0.4, pointRadius:3,
          pointBackgroundColor: _tl.map(d =>
            d.avg > 0.05 ? '#10b981' : d.avg < -0.05 ? '#ef4444' : '#64748b'),
          fill:true, yAxisID:'y'
        }},
        {{
          label:'Mentions', data:_tl.map(d => d.count > 0 ? d.count : null),
          borderColor:'rgba(240,180,41,0.65)', backgroundColor:'rgba(240,180,41,0.04)',
          borderWidth:1.5, tension:0.3, pointRadius:2, fill:false, yAxisID:'y2',
          spanGaps:true
        }}
      ]
    }},
    options:{{
      responsive:true, maintainAspectRatio:true,
      plugins:{{ legend:{{ labels:{{ color:'#94a3b8',boxWidth:12 }} }} }},
      scales:{{
        x:{{ ticks:{{ color:'#475569',maxTicksLimit:14 }}, grid:{{ color:'#1e293b' }} }},
        y:{{
          min:-1, max:1,
          ticks:{{ color:'#475569' }}, grid:{{ color:'#1e293b' }},
          title:{{ display:true, text:'Sentiment', color:'#475569', font:{{ size:11 }} }}
        }},
        y2:{{
          type:'logarithmic',
          position:'right',
          min:0.9,
          ticks:{{
            color:'#475569',
            callback: function(v) {{ return Number.isInteger(Math.log10(v)) ? v : ''; }}
          }},
          grid:{{ display:false }},
          title:{{ display:true, text:'Mentions (log)', color:'#475569', font:{{ size:11 }} }}
        }}
      }}
    }}
  }});
}} else {{
  document.getElementById('timelineChart').closest('.card').querySelector('canvas').insertAdjacentHTML(
    'afterend','<p style="color:#334155;text-align:center;padding:20px;font-size:0.85em">Not enough dated items for a timeline</p>');
}}

{trends_js}
</script>
</body>
</html>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--query", "-q", default="")
    ap.add_argument("--days",  type=int, default=30)
    ap.add_argument("--help",  action="store_true")
    args = ap.parse_args()

    if args.help or not args.query.strip():
        print(__doc__); sys.exit(0)

    query = args.query.strip()

    print("=" * 60)
    print("  Topic Sentiment Analyzer")
    print("=" * 60)
    print(f'\n  Query: "{query}"')

    print("\n[1/4] Fetching news & social data…")
    # All sources fetched concurrently — no serial waiting
    with ThreadPoolExecutor(max_workers=7) as pool:
        f_gnews   = pool.submit(fetch_google_news,   query)
        f_bing    = pool.submit(fetch_bing_news,     query, 40, args.days)
        f_finance = pool.submit(fetch_finance_feeds, query, 15, args.days)
        f_red     = pool.submit(fetch_reddit,        query)
        f_hn      = pool.submit(fetch_hackernews,    query, days=args.days)
        f_trends  = pool.submit(fetch_trends,        query)

        gnews   = f_gnews.result()
        bing    = f_bing.result()
        finance = f_finance.result()
        red     = f_red.result()
        hn      = f_hn.result()
        trends  = f_trends.result()   # used below; skip separate [3/4] fetch

    # Merge and deduplicate news sources by URL
    all_news  = _dedup_urls(gnews + bing + finance)
    all_items = all_news + red + hn
    print(f"  Google News: {len(gnews)}  Bing: {len(bing)}  Finance feeds: {len(finance)}"
          f"  → {len(all_news)} news (after dedup)")
    print(f"  Reddit: {len(red)}  HN: {len(hn)}  → {len(all_items)} total")

    print("\n[2/4] Scoring sentiment…")
    score_all(all_items, query)
    scored_v  = sum(1 for i in all_items if i.get("sentiment"))
    scored_fb = sum(1 for i in all_items if i.get("sentiment_fb"))
    overall   = _avg_composite(all_items)
    fb_avail  = scored_fb > 0
    model_str = f"VADER={scored_v}"
    if fb_avail:
        model_str += f"  FinBERT={scored_fb}"
    if overall is not None:
        print(f"  {model_str}  Composite: {overall:+.3f} ({_lbl(overall)})")
    else:
        print(f"  {model_str}  No scored items")

    print(f"  Google Trends: {len(trends)} data points"
          f"  (fetched concurrently in step 1)")

    print("\n[3/4] Building report…")
    html    = build_html(query, all_items, trends, datetime.datetime.now(), days=args.days)
    slug    = re.sub(r"[^a-zA-Z0-9_-]", "_", query)[:40]
    outdir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topicData")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"topic_{slug}.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  ✓  Report saved → {outpath}")
    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        webbrowser.open(f"file://{outpath}")


if __name__ == "__main__":
    main()
