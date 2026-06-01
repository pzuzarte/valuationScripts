"""
International ETF Momentum Screener
=====================================
Ranks ~130 non-leveraged international ETFs by multi-timeframe momentum.
Covers single-country, regional, and thematic ETFs across developed and
emerging markets -- Europe, Asia-Pacific, Latin America, Middle East & Africa.
Leveraged and inverse ETFs are excluded automatically.

SCORING (0-100 pts)
--------------------
  Pillar 1  Price Momentum  55 pts  1M(10) + 3M(20) + 6M(15) + 12M(10)
  Pillar 2  Trend Strength  25 pts  vs 50-day MA(10) + vs 200-day MA(15)
  Pillar 3  Volume Trend    10 pts  20-day avg vol vs 60-day avg vol
  Pillar 4  Risk-adjusted   10 pts  3M return / annualised volatility

Grades:  A+(>=88) A(>=72) B+(>=56) B(>=40) C(<40)

Tiers:   Strong Momentum(>=70)  Building(>=50)  Neutral(>=30)
         Weakening(>=15)  Declining(<15)

COMMAND LINE
-------------
  --region  ALL|DEVELOPED|EMERGING|EUROPE|ASIA|LATAM|MEA|CHINA|INDIA
                                           (default: ALL)
  --top N   Show top N ETFs by score       (default: all)
  --min-aum Minimum AUM in $M              (default: 100)
  --csv     Export CSV alongside HTML

EXAMPLES
---------
  python intlMomentum.py
  python intlMomentum.py --region EUROPE
  python intlMomentum.py --region EMERGING --top 25
  python intlMomentum.py --min-aum 500 --csv
"""

import sys, os, math, datetime, argparse, csv, logging, time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed.  Run: pip install yfinance")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed.  Run: pip install numpy")
    sys.exit(1)

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────
# ETF Universe  (region codes: DEVELOPED EMERGING EUROPE JAPAN ASIA CHINA INDIA LATAM MEA FACTOR)
# ─────────────────────────────────────────────────────────────────────────────
ETF_UNIVERSE = {
    # ── Broad Developed ──────────────────────────────────────────────────────
    "VEA":  {"name": "Vanguard FTSE Developed Markets",         "region": "DEVELOPED", "category": "Broad"},
    "EFA":  {"name": "iShares MSCI EAFE",                       "region": "DEVELOPED", "category": "Broad"},
    "SCHF": {"name": "Schwab International Equity",             "region": "DEVELOPED", "category": "Broad"},
    "SPDW": {"name": "SPDR Portfolio Developed World ex-US",    "region": "DEVELOPED", "category": "Broad"},
    "VXUS": {"name": "Vanguard Total International Stock",      "region": "DEVELOPED", "category": "Broad"},
    "ACWX": {"name": "iShares MSCI ACWI ex-US",                "region": "DEVELOPED", "category": "Broad"},
    "IOO":  {"name": "iShares Global 100",                      "region": "DEVELOPED", "category": "Large Cap"},
    "SCZ":  {"name": "iShares MSCI EAFE Small-Cap",             "region": "DEVELOPED", "category": "Small Cap"},
    "VSS":  {"name": "Vanguard FTSE All-World ex-US Small-Cap", "region": "DEVELOPED", "category": "Small Cap"},
    "IDV":  {"name": "iShares International Select Dividend",   "region": "DEVELOPED", "category": "Dividend"},
    "VYMI": {"name": "Vanguard International High Dividend",    "region": "DEVELOPED", "category": "Dividend"},
    "DWX":  {"name": "SPDR S&P International Dividend",        "region": "DEVELOPED", "category": "Dividend"},
    "FNDF": {"name": "Schwab Fundamental Intl Large Co",        "region": "DEVELOPED", "category": "Value"},
    "HEFA": {"name": "iShares Currency Hedged MSCI EAFE",       "region": "DEVELOPED", "category": "Hedged"},
    "DBEF": {"name": "Xtrackers MSCI EAFE Hedged Equity",       "region": "DEVELOPED", "category": "Hedged"},
    "EWC":  {"name": "iShares MSCI Canada",                     "region": "DEVELOPED", "category": "Canada"},
    "EWA":  {"name": "iShares MSCI Australia",                  "region": "DEVELOPED", "category": "Australia"},
    # ── Broad Emerging ───────────────────────────────────────────────────────
    "VWO":  {"name": "Vanguard FTSE Emerging Markets",          "region": "EMERGING",  "category": "Broad"},
    "EEM":  {"name": "iShares MSCI Emerging Markets",           "region": "EMERGING",  "category": "Broad"},
    "IEMG": {"name": "iShares Core MSCI Emerging Markets",      "region": "EMERGING",  "category": "Broad"},
    "SCHE": {"name": "Schwab Emerging Markets Equity",          "region": "EMERGING",  "category": "Broad"},
    "SPEM": {"name": "SPDR Portfolio Emerging Markets",         "region": "EMERGING",  "category": "Broad"},
    "EMXC": {"name": "iShares MSCI EM ex-China",               "region": "EMERGING",  "category": "ex-China"},
    "FM":   {"name": "iShares MSCI Frontier & Select EM",      "region": "EMERGING",  "category": "Frontier"},
    "DEM":  {"name": "WisdomTree EM High Dividend",             "region": "EMERGING",  "category": "Dividend"},
    "EEMS": {"name": "iShares MSCI EM Small-Cap",              "region": "EMERGING",  "category": "Small Cap"},
    "EWX":  {"name": "SPDR S&P EM Small Cap",                  "region": "EMERGING",  "category": "Small Cap"},
    "FNDE": {"name": "Schwab Fundamental EM Large Co",          "region": "EMERGING",  "category": "Value"},
    "AAXJ": {"name": "iShares MSCI All Country Asia ex-JP",    "region": "EMERGING",  "category": "Asia Broad"},
    "GMF":  {"name": "SPDR S&P Emerging Asia Pacific",         "region": "EMERGING",  "category": "Asia Broad"},
    # ── Europe ───────────────────────────────────────────────────────────────
    "VGK":  {"name": "Vanguard FTSE Europe",                    "region": "EUROPE",    "category": "Broad"},
    "EZU":  {"name": "iShares MSCI Eurozone",                   "region": "EUROPE",    "category": "Eurozone"},
    "IEV":  {"name": "iShares S&P Europe 350",                  "region": "EUROPE",    "category": "Broad"},
    "IEUR": {"name": "iShares Core MSCI Europe",                "region": "EUROPE",    "category": "Broad"},
    "FEZ":  {"name": "SPDR EURO STOXX 50",                      "region": "EUROPE",    "category": "Large Cap"},
    "HEDJ": {"name": "WisdomTree Europe Hedged Equity",         "region": "EUROPE",    "category": "Hedged"},
    "DBEU": {"name": "Xtrackers MSCI Europe Hedged Equity",     "region": "EUROPE",    "category": "Hedged"},
    "EWG":  {"name": "iShares MSCI Germany",                    "region": "EUROPE",    "category": "Germany"},
    "EWU":  {"name": "iShares MSCI United Kingdom",             "region": "EUROPE",    "category": "UK"},
    "EWQ":  {"name": "iShares MSCI France",                     "region": "EUROPE",    "category": "France"},
    "EWI":  {"name": "iShares MSCI Italy",                      "region": "EUROPE",    "category": "Italy"},
    "EWP":  {"name": "iShares MSCI Spain",                      "region": "EUROPE",    "category": "Spain"},
    "EWD":  {"name": "iShares MSCI Sweden",                     "region": "EUROPE",    "category": "Sweden"},
    "EWL":  {"name": "iShares MSCI Switzerland",                "region": "EUROPE",    "category": "Switzerland"},
    "EWN":  {"name": "iShares MSCI Netherlands",                "region": "EUROPE",    "category": "Netherlands"},
    "EWK":  {"name": "iShares MSCI Belgium",                    "region": "EUROPE",    "category": "Belgium"},
    "EWO":  {"name": "iShares MSCI Austria",                    "region": "EUROPE",    "category": "Austria"},
    "ENOR": {"name": "iShares MSCI Norway",                     "region": "EUROPE",    "category": "Norway"},
    "EDEN": {"name": "iShares MSCI Denmark",                    "region": "EUROPE",    "category": "Denmark"},
    "EPOL": {"name": "iShares MSCI Poland",                     "region": "EUROPE",    "category": "Poland"},
    "TUR":  {"name": "iShares MSCI Turkey",                     "region": "EUROPE",    "category": "Turkey"},
    "HEWG": {"name": "iShares Currency Hedged MSCI Germany",    "region": "EUROPE",    "category": "Germany Hedged"},
    "FLGB": {"name": "Franklin FTSE United Kingdom",            "region": "EUROPE",    "category": "UK"},
    # ── Japan ────────────────────────────────────────────────────────────────
    "EWJ":  {"name": "iShares MSCI Japan",                      "region": "JAPAN",     "category": "Broad"},
    "HEWJ": {"name": "iShares Currency Hedged MSCI Japan",      "region": "JAPAN",     "category": "Hedged"},
    "DXJ":  {"name": "WisdomTree Japan Hedged Equity",          "region": "JAPAN",     "category": "Hedged"},
    "DBJP": {"name": "Xtrackers MSCI Japan Hedged Equity",      "region": "JAPAN",     "category": "Hedged"},
    "FLJP": {"name": "Franklin FTSE Japan",                     "region": "JAPAN",     "category": "Broad"},
    "DFJ":  {"name": "WisdomTree Japan Small Cap Dividend",     "region": "JAPAN",     "category": "Small Cap"},
    "VPL":  {"name": "Vanguard FTSE Pacific",                   "region": "JAPAN",     "category": "Asia-Pacific"},
    # ── Asia-Pacific (ex-Japan, ex-China, ex-India) ────────────────────────
    "EWY":  {"name": "iShares MSCI South Korea",                "region": "ASIA",      "category": "South Korea"},
    "EWT":  {"name": "iShares MSCI Taiwan",                     "region": "ASIA",      "category": "Taiwan"},
    "EWS":  {"name": "iShares MSCI Singapore",                  "region": "ASIA",      "category": "Singapore"},
    "EWH":  {"name": "iShares MSCI Hong Kong",                  "region": "ASIA",      "category": "Hong Kong"},
    "EWM":  {"name": "iShares MSCI Malaysia",                   "region": "ASIA",      "category": "Malaysia"},
    "THD":  {"name": "iShares MSCI Thailand",                   "region": "ASIA",      "category": "Thailand"},
    "EIDO": {"name": "iShares MSCI Indonesia",                  "region": "ASIA",      "category": "Indonesia"},
    "EPHE": {"name": "iShares MSCI Philippines",                "region": "ASIA",      "category": "Philippines"},
    "VNM":  {"name": "VanEck Vietnam",                          "region": "ASIA",      "category": "Vietnam"},
    "ENZL": {"name": "iShares MSCI New Zealand",                "region": "ASIA",      "category": "New Zealand"},
    "EPP":  {"name": "iShares MSCI Pacific ex-Japan",           "region": "ASIA",      "category": "Pacific ex-JP"},
    "FLKR": {"name": "Franklin FTSE South Korea",               "region": "ASIA",      "category": "South Korea"},
    # ── China ────────────────────────────────────────────────────────────────
    "FXI":  {"name": "iShares China Large-Cap",                 "region": "CHINA",     "category": "Large Cap"},
    "MCHI": {"name": "iShares MSCI China",                      "region": "CHINA",     "category": "Broad"},
    "GXC":  {"name": "SPDR S&P China",                         "region": "CHINA",     "category": "Broad"},
    "KWEB": {"name": "KraneShares CSI China Internet",          "region": "CHINA",     "category": "Internet"},
    "CQQQ": {"name": "Invesco China Technology",                "region": "CHINA",     "category": "Technology"},
    "CHIQ": {"name": "Global X MSCI China Consumer Disc.",      "region": "CHINA",     "category": "Consumer"},
    "ASHR": {"name": "Xtrackers Harvest CSI 300 China A",       "region": "CHINA",     "category": "A-Shares"},
    "CNYA": {"name": "iShares MSCI China A",                    "region": "CHINA",     "category": "A-Shares"},
    "CXSE": {"name": "WisdomTree China ex-State-Owned Ent.",    "region": "CHINA",     "category": "ex-SOE"},
    "FLCH": {"name": "Franklin FTSE China",                     "region": "CHINA",     "category": "Broad"},
    "KURE": {"name": "KraneShares MSCI All China Health",       "region": "CHINA",     "category": "Healthcare"},
    # ── India ────────────────────────────────────────────────────────────────
    "INDA": {"name": "iShares MSCI India",                      "region": "INDIA",     "category": "Broad"},
    "PIN":  {"name": "Invesco India",                           "region": "INDIA",     "category": "Broad"},
    "INDY": {"name": "iShares India 50",                        "region": "INDIA",     "category": "Large Cap"},
    "SMIN": {"name": "iShares MSCI India Small-Cap",            "region": "INDIA",     "category": "Small Cap"},
    "SCIF": {"name": "VanEck India Small-Cap",                  "region": "INDIA",     "category": "Small Cap"},
    "FLIN": {"name": "Franklin FTSE India",                     "region": "INDIA",     "category": "Broad"},
    # ── Latin America ────────────────────────────────────────────────────────
    "ILF":  {"name": "iShares Latin America 40",                "region": "LATAM",     "category": "Broad"},
    "EWZ":  {"name": "iShares MSCI Brazil",                     "region": "LATAM",     "category": "Brazil"},
    "EWW":  {"name": "iShares MSCI Mexico",                     "region": "LATAM",     "category": "Mexico"},
    "ECH":  {"name": "iShares MSCI Chile",                      "region": "LATAM",     "category": "Chile"},
    "GXG":  {"name": "Global X MSCI Colombia",                  "region": "LATAM",     "category": "Colombia"},
    "EPU":  {"name": "iShares MSCI Peru",                       "region": "LATAM",     "category": "Peru"},
    "ARGT": {"name": "Global X MSCI Argentina",                 "region": "LATAM",     "category": "Argentina"},
    "EWZS": {"name": "iShares MSCI Brazil Small-Cap",           "region": "LATAM",     "category": "Brazil Small"},
    "FLBR": {"name": "Franklin FTSE Brazil",                    "region": "LATAM",     "category": "Brazil"},
    # ── Middle East & Africa ─────────────────────────────────────────────────
    "KSA":  {"name": "iShares MSCI Saudi Arabia",               "region": "MEA",       "category": "Saudi Arabia"},
    "UAE":  {"name": "iShares MSCI UAE",                        "region": "MEA",       "category": "UAE"},
    "QAT":  {"name": "iShares MSCI Qatar",                      "region": "MEA",       "category": "Qatar"},
    "EZA":  {"name": "iShares MSCI South Africa",               "region": "MEA",       "category": "South Africa"},
    "EGPT": {"name": "VanEck Egypt",                            "region": "MEA",       "category": "Egypt"},
    "NGE":  {"name": "Global X Nigeria",                        "region": "MEA",       "category": "Nigeria"},
    "AFK":  {"name": "VanEck Africa",                           "region": "MEA",       "category": "Africa Broad"},
    "GULF": {"name": "WisdomTree Middle East Dividend",         "region": "MEA",       "category": "Gulf/ME"},
    # ── Factor / Smart-Beta ──────────────────────────────────────────────────
    # Momentum
    "IDMO": {"name": "iShares MSCI Intl Developed Momentum Factor","region": "FACTOR",   "category": "Momentum"},
    # Min-Volatility
    "EFAV": {"name": "iShares MSCI EAFE Min Volatility Factor",    "region": "FACTOR",   "category": "Min-Vol"},
    "EEMV": {"name": "iShares MSCI EM Min Volatility Factor",      "region": "FACTOR",   "category": "Min-Vol"},
    "ACWV": {"name": "iShares MSCI ACWI Min Volatility Factor",    "region": "FACTOR",   "category": "Min-Vol"},
    # Quality
    "IQLT": {"name": "iShares MSCI Intl Quality Factor",           "region": "FACTOR",   "category": "Quality"},
    "DGRE": {"name": "WisdomTree EM Quality Dividend Growth",      "region": "FACTOR",   "category": "Quality"},
    # Value
    "IVLU": {"name": "iShares MSCI Intl Value Factor",             "region": "FACTOR",   "category": "Value"},
    "FNDC": {"name": "Schwab Fundamental Intl Small Company",      "region": "FACTOR",   "category": "Value"},
    "FYLD": {"name": "Cambria Foreign Shareholder Yield",          "region": "FACTOR",   "category": "Yield"},
    "PID":  {"name": "Invesco Intl Dividend Achievers",            "region": "FACTOR",   "category": "Dividend"},
    "FIVA": {"name": "Fidelity Intl Value Factor",                 "region": "FACTOR",   "category": "Value"},
    "VIGI": {"name": "Vanguard Intl Dividend Appreciation",        "region": "FACTOR",   "category": "Dividend Growth"},
    "IGRO": {"name": "iShares Intl Dividend Growth",               "region": "FACTOR",   "category": "Dividend Growth"},
    # ── Additional Country ETFs ──────────────────────────────────────────────
    "GREK": {"name": "Global X MSCI Greece",                       "region": "EUROPE",   "category": "Greece"},
    "NORW": {"name": "Global X MSCI Norway",                       "region": "EUROPE",   "category": "Norway"},
    "HEWU": {"name": "iShares Currency Hedged MSCI United Kingdom","region": "EUROPE",   "category": "UK Hedged"},
    "FLCA": {"name": "Franklin FTSE Canada",                       "region": "DEVELOPED","category": "Canada"},
    "FLAU": {"name": "Franklin FTSE Australia",                    "region": "DEVELOPED","category": "Australia"},
    "FLMX": {"name": "Franklin FTSE Mexico",                       "region": "LATAM",    "category": "Mexico"},
    "JPXN": {"name": "iShares JPX-Nikkei 400",                     "region": "JAPAN",    "category": "Large Cap"},
    # ── Thematic Cross-Regional ──────────────────────────────────────────────
    "EMQQ": {"name": "Emerging Markets Internet & Ecommerce",      "region": "EMERGING", "category": "Internet"},
    "FMQQ": {"name": "EM ex-China Internet & Ecommerce",           "region": "EMERGING", "category": "Internet"},
    "KGRN": {"name": "KraneShares China Clean Technology",         "region": "CHINA",    "category": "Clean Energy"},
}

# UI filter groups: maps button label -> list of region codes it includes
FILTER_GROUPS = {
    "ALL":       None,
    "DEVELOPED": ["DEVELOPED"],
    "EMERGING":  ["EMERGING"],
    "EUROPE":    ["EUROPE"],
    "ASIA-PAC":  ["JAPAN", "ASIA", "CHINA", "INDIA"],
    "LATAM":     ["LATAM"],
    "MEA":       ["MEA"],
    "CHINA":     ["CHINA"],
    "INDIA":     ["INDIA"],
    "JAPAN":     ["JAPAN"],
    "FACTOR":    ["FACTOR"],
}

TIERS = [
    ("Strong Momentum", 70, 101, "#22c55e"),
    ("Building",        50,  70, "#14b8a6"),
    ("Neutral",         30,  50, "#eab308"),
    ("Weakening",       15,  30, "#f97316"),
    ("Declining",        0,  15, "#ef4444"),
]

GRADE_RANGES = [
    ("A+", 88), ("A", 72), ("B+", 56), ("B", 40), ("C", 0)
]

# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_one(ticker):
    """Fetch 1Y price+volume history and metadata for a single ETF via yfinance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y", auto_adjust=True)
        if hist is None or len(hist) < 30:
            return None

        closes = hist["Close"].dropna()
        vols   = hist["Volume"].dropna()
        if len(closes) < 30:
            return None

        price = float(closes.iloc[-1])
        n     = len(closes)

        def _ret(days):
            if n <= days:
                return None
            return float(closes.iloc[-1] / closes.iloc[-(days + 1)] - 1)

        ret_1m  = _ret(21)
        ret_3m  = _ret(63)
        ret_6m  = _ret(126)
        ret_12m = _ret(252) if n > 252 else None

        # Moving averages
        ma50  = float(closes.tail(50).mean())  if n >= 50  else None
        ma200 = float(closes.tail(200).mean()) if n >= 200 else None

        # Annualised volatility (from daily log returns, last 126 days)
        recent = closes.tail(126)
        log_rets = np.log(recent / recent.shift(1)).dropna()
        ann_vol  = float(log_rets.std() * np.sqrt(252)) if len(log_rets) >= 20 else None

        # Volume trend: 20-day avg vs 60-day avg
        vol_20 = float(vols.tail(20).mean())  if len(vols) >= 20 else None
        vol_60 = float(vols.tail(60).mean())  if len(vols) >= 60 else None
        vol_ratio = (vol_20 / vol_60) if (vol_20 and vol_60 and vol_60 > 0) else None

        # AUM & expense ratio from info (best-effort, non-blocking)
        aum  = None
        expr = None
        # Always call t.info — it carries expenseRatio for ETFs.
        # fast_info is faster but only exposes totalAssets; use it as fallback.
        try:
            full = t.info or {}
            aum_raw  = full.get("totalAssets")
            expr_raw = (full.get("annualReportExpenseRatio")
                        or full.get("expenseRatio")
                        or full.get("netExpenseRatio"))
            if aum_raw:
                aum = float(aum_raw) / 1e6   # -> $M
            if expr_raw:
                expr = float(expr_raw)
        except Exception:
            pass
        if aum is None:
            try:
                fi = t.fast_info
                aum_raw = getattr(fi, "total_assets", None)
                if aum_raw:
                    aum = float(aum_raw) / 1e6
            except Exception:
                pass

        return {
            "ticker":    ticker,
            "price":     price,
            "ret_1m":    ret_1m,
            "ret_3m":    ret_3m,
            "ret_6m":    ret_6m,
            "ret_12m":   ret_12m,
            "ma50":      ma50,
            "ma200":     ma200,
            "pct_vs_ma50":  (price / ma50  - 1) if ma50  else None,
            "pct_vs_ma200": (price / ma200 - 1) if ma200 else None,
            "ann_vol":   ann_vol,
            "vol_ratio": vol_ratio,
            "aum_m":     aum,
            "exp_ratio": expr,
        }
    except Exception:
        return None


def fetch_all(tickers, workers=20):
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            done += 1
            try:
                data = fut.result()
                if data:
                    results[t] = data
            except Exception:
                pass
            if done % 20 == 0 or done == len(tickers):
                print(f"  Fetched {done}/{len(tickers)} ETFs ({len(results)} OK)...")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
def _pts(val, brackets):
    """Award points based on (threshold, points) bracket list, highest first."""
    if val is None:
        return 0
    for threshold, pts in brackets:
        if val >= threshold:
            return pts
    return 0


def score_etf(d):
    s = 0
    # Pillar 1: Price Momentum (55 pts)
    s += _pts(d.get("ret_1m"),  [(0.08,10),(0.05, 7),(0.02, 4),(0.00, 2)])
    s += _pts(d.get("ret_3m"),  [(0.18,20),(0.10,15),(0.05, 9),(0.00, 4)])
    s += _pts(d.get("ret_6m"),  [(0.25,15),(0.15,11),(0.08, 7),(0.00, 3)])
    s += _pts(d.get("ret_12m"), [(0.35,10),(0.20, 7),(0.10, 4),(0.00, 2)])
    # Pillar 2: Trend Strength (25 pts)
    s += _pts(d.get("pct_vs_ma50"),  [(0.08,10),(0.04, 7),(0.01, 4),(0.00, 2)])
    s += _pts(d.get("pct_vs_ma200"), [(0.15,15),(0.08,11),(0.03, 7),(0.00, 4)])
    # Pillar 3: Volume Trend (10 pts)
    s += _pts(d.get("vol_ratio"), [(1.50,10),(1.20, 7),(1.00, 4),(0.80, 2)])
    # Pillar 4: Risk-adjusted 3M (10 pts)
    r3  = d.get("ret_3m")
    vol = d.get("ann_vol")
    if r3 is not None and vol and vol > 0:
        sharpe_3m = (r3 * 4) / vol   # annualise 3M return then divide by ann vol
        s += _pts(sharpe_3m, [(2.0,10),(1.0, 7),(0.5, 4),(0.0, 2)])
    return min(s, 100)


def get_tier(score):
    for name, lo, hi, color in TIERS:
        if lo <= score < hi:
            return name, color
    return "Declining", "#ef4444"


def get_grade(score):
    for label, threshold in GRADE_RANGES:
        if score >= threshold:
            return label
    return "C"


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fp(v, decimals=1):
    """Format a float as a percentage string, or '--' if None."""
    if v is None:
        return "--"
    return f"{v * 100:+.{decimals}f}%"


def _fa(v):
    """Format AUM in $M or $B."""
    if v is None:
        return "--"
    if v >= 1000:
        return f"${v/1000:.1f}B"
    return f"${v:.0f}M"


def _fe(v):
    """Format expense ratio."""
    if v is None:
        return "--"
    return f"{v * 100:.2f}%"


def _ret_color(v):
    """Return a CSS background color for a return value."""
    if v is None:
        return "transparent"
    if v >= 0.20:  return "rgba(34,197,94,0.35)"
    if v >= 0.10:  return "rgba(34,197,94,0.22)"
    if v >= 0.04:  return "rgba(34,197,94,0.12)"
    if v >= 0.00:  return "rgba(34,197,94,0.05)"
    if v >= -0.05: return "rgba(239,68,68,0.08)"
    if v >= -0.12: return "rgba(239,68,68,0.18)"
    return "rgba(239,68,68,0.30)"


def _ret_text_color(v):
    if v is None:
        return "#64748b"
    return "#4ade80" if v >= 0 else "#f87171"


# ─────────────────────────────────────────────────────────────────────────────
# Help Modal
# ─────────────────────────────────────────────────────────────────────────────
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

_HELP_MODAL = """
<div class="help-overlay" id="helpOverlay" onclick="if(event.target===this)closeHelp()">
 <div class="help-modal">
  <button class="help-close" onclick="closeHelp()">&#x2715;</button>
  <h2>International ETF Momentum Screener</h2>
  <p class="help-desc">
   Ranks ~130 non-leveraged international ETFs by multi-timeframe price momentum.
   Covers single-country, regional, and thematic ETFs across developed and emerging
   markets. Leveraged and inverse ETFs are excluded. All data is fetched from yfinance.
   Use the region buttons to focus on a specific area; sort any column by clicking its header.
  </p>
  <div class="help-sec">Scoring (0-100 pts)</div>
  <table class="help-tbl">
   <tr><td>Pillar 1 -- Price Momentum (55 pts)</td>
       <td>1M return (10) + 3M return (20) + 6M return (15) + 12M return (10).
           Higher and more consistent returns across all windows = higher score.</td></tr>
   <tr><td>Pillar 2 -- Trend Strength (25 pts)</td>
       <td>How far price is above the 50-day MA (10 pts) and 200-day MA (15 pts).
           Both above = confirmed uptrend; below either = score penalty.</td></tr>
   <tr><td>Pillar 3 -- Volume Trend (10 pts)</td>
       <td>Ratio of 20-day average volume to 60-day average volume.
           Rising volume confirms institutional buying; falling volume is a warning sign.</td></tr>
   <tr><td>Pillar 4 -- Risk-adjusted (10 pts)</td>
       <td>3M return annualised divided by annual volatility (Sharpe-like).
           Rewards ETFs gaining ground smoothly; penalises volatile surges.</td></tr>
  </table>
  <div class="help-sec">Momentum Tiers</div>
  <table class="help-tbl">
   <tr><td style="color:#4ade80">Strong Momentum (70+)</td><td>Broad uptrend, high across multiple windows. High-conviction momentum.</td></tr>
   <tr><td style="color:#2dd4bf">Building (50-69)</td><td>Positive trend establishing across most windows. Worth watching.</td></tr>
   <tr><td style="color:#fbbf24">Neutral (30-49)</td><td>Mixed signals. Some momentum but not confirmed across timeframes.</td></tr>
   <tr><td style="color:#fb923c">Weakening (15-29)</td><td>Momentum fading. Declining returns or trend breakdown.</td></tr>
   <tr><td style="color:#f87171">Declining (&lt;15)</td><td>Persistent downtrend across timeframes.</td></tr>
  </table>
  <div class="help-sec">Column Guide</div>
  <table class="help-tbl">
   <tr><td>Score / Grade</td><td>0-100 composite momentum score. Grade A+ through C.</td></tr>
   <tr><td>1M / 3M / 6M / 12M</td><td>Price return over the period. Cells are green/red heat-mapped by magnitude.</td></tr>
   <tr><td>vs SPY (3M)</td><td>3M return minus SPY 3M return. Positive = outperforming the US benchmark.</td></tr>
   <tr><td>vs MA50 / MA200</td><td>How far the current price is above (or below) the 50- and 200-day moving averages.</td></tr>
   <tr><td>Vol Ratio</td><td>20-day average volume / 60-day average volume. Values above 1.0 indicate increasing interest.</td></tr>
   <tr><td>AUM</td><td>Fund assets under management fetched from yfinance. Larger funds have tighter bid-ask spreads.</td></tr>
   <tr><td>Exp. Ratio</td><td>Annual management fee as a percentage of assets.</td></tr>
  </table>
 </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML Report
# ─────────────────────────────────────────────────────────────────────────────
def build_html(results, ts, total_fetched, spy_ret_3m, suite_port=5050):
    """Build a self-contained dark-themed HTML report."""

    # Sort by score descending
    results = sorted(results, key=lambda x: -x["score"])

    # Top 6 for cards
    top6 = results[:6]

    # ── CSS ──────────────────────────────────────────────────────────────────
    css = """
:root{--bg:#0f1117;--s1:#161a27;--s2:#1a1e2e;--bd:#252a3a;--text:#e2e8f0;--sub:#94a3b8;--acc:#6366f1;--acc2:#818cf8}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
a{color:inherit;text-decoration:none}

/* ── Nav ── */
.nav{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 32px;background:var(--s1);border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:200}
.nav-title{font-size:13px;font-weight:700;color:var(--text);margin-right:6px}

/* ── Hero ── */
.hero{padding:28px 32px 20px;background:linear-gradient(135deg,#0f1117 0%,#161a27 60%,#1a1e2e 100%);border-bottom:1px solid var(--bd)}
.hero h1{font-size:22px;font-weight:800;color:var(--text);margin-bottom:4px}
.hero .sub{font-size:12px;color:var(--sub)}
.stats{display:flex;gap:28px;margin-top:16px;flex-wrap:wrap}
.stat{display:flex;flex-direction:column;gap:2px}
.stat-val{font-size:20px;font-weight:700;color:var(--acc2)}
.stat-lbl{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:1px}

/* ── Top picks cards ── */
.top-section{padding:20px 32px;background:var(--s1);border-bottom:1px solid var(--bd)}
.top-section h2{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--acc);margin-bottom:14px;font-weight:700}
.cards{display:flex;gap:12px;flex-wrap:wrap}
.card{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:14px 16px;min-width:150px;flex:1;max-width:200px;cursor:default;transition:border-color .15s}
.card:hover{border-color:var(--acc)}
.card-tick{font-size:15px;font-weight:800;color:var(--text)}
.card-name{font-size:9px;color:var(--sub);margin:2px 0 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card-score{font-size:22px;font-weight:800}
.card-ret{font-size:10px;color:var(--sub);margin-top:4px}
.card-grade{font-size:10px;font-weight:700;margin-top:6px;padding:2px 7px;border-radius:4px;display:inline-block;background:rgba(99,102,241,.15);color:var(--acc2)}

/* ── Filters ── */
.filters{padding:14px 32px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--bd);background:var(--bg)}
.flt{padding:4px 13px;border-radius:16px;font-size:11px;font-weight:600;border:1px solid var(--bd);color:var(--sub);background:none;cursor:pointer;transition:all .15s}
.flt:hover{border-color:var(--acc);color:var(--acc2)}
.flt.active{background:rgba(99,102,241,.18);border-color:var(--acc);color:var(--acc2)}
.flt-label{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:1px;margin-right:4px}

/* ── Table ── */
.main{padding:20px 32px 40px}
.tbl-wrap{overflow-x:auto;border-radius:10px;border:1px solid var(--bd)}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--s2);color:var(--sub);font-size:10px;text-transform:uppercase;letter-spacing:1px;padding:10px 12px;text-align:right;white-space:nowrap;cursor:pointer;user-select:none;border-bottom:1px solid var(--bd)}
thead th:first-child,thead th:nth-child(2),thead th:nth-child(3),thead th:nth-child(4){text-align:left}
thead th:hover{color:var(--text)}
th.sort-asc::after{content:" ^"}
th.sort-desc::after{content:" v"}
tbody tr{border-bottom:1px solid var(--bd);transition:background .1s}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:rgba(99,102,241,.05)}
tbody tr.hidden{display:none}
td{padding:9px 12px;text-align:right;color:var(--sub);white-space:nowrap}
td:first-child{text-align:center;color:var(--sub);font-size:11px;width:36px}
td.tick{text-align:left;font-weight:700;color:var(--text);font-size:13px}
td.etf-name{text-align:left;color:var(--sub);font-size:11px;max-width:220px;overflow:hidden;text-overflow:ellipsis}
td.region-tag{text-align:left}
td.cat-tag{text-align:left}
.rtag{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;background:rgba(99,102,241,.12);color:var(--acc2)}
.ctag{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;color:var(--sub);background:rgba(255,255,255,.04)}
.score-cell{font-weight:700;font-size:13px}
.grade-badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:700;margin-left:4px}
.ret-cell{font-weight:600}

/* ── Footer ── */
.footer{padding:20px 32px;border-top:1px solid var(--bd);font-size:11px;color:var(--sub)}

@media(max-width:900px){.cards{gap:8px}.card{min-width:130px}.main,.hero,.top-section,.filters{padding-left:16px;padding-right:16px}}
"""
    # ── Grade badge colors ───────────────────────────────────────────────────
    grade_colors = {
        "A+": ("#4ade80","rgba(34,197,94,.15)"),
        "A":  ("#4ade80","rgba(34,197,94,.10)"),
        "B+": ("#a3e635","rgba(163,230,53,.10)"),
        "B":  ("#fbbf24","rgba(251,191,36,.10)"),
        "C":  ("#f87171","rgba(248,113,113,.10)"),
    }

    # ── JS for sorting + filtering ────────────────────────────────────────────
    js = """
var _dir = {};
function sortTable(col) {
  var tb = document.querySelector('#etfTable tbody');
  var rows = Array.from(tb.querySelectorAll('tr:not(.hidden)'));
  var dir = (_dir[col] = !_dir[col]);
  rows.sort(function(a,b){
    var av = a.cells[col] ? a.cells[col].dataset.v : '';
    var bv = b.cells[col] ? b.cells[col].dataset.v : '';
    var an = parseFloat(av), bn = parseFloat(bv);
    if(!isNaN(an)&&!isNaN(bn)) return dir ? an-bn : bn-an;
    return dir ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(function(r){tb.appendChild(r)});
  document.querySelectorAll('th').forEach(function(th,i){
    th.classList.remove('sort-asc','sort-desc');
    if(i===col) th.classList.add(dir?'sort-asc':'sort-desc');
  });
}
function filterRegion(group) {
  document.querySelectorAll('.flt').forEach(function(b){b.classList.remove('active')});
  event.target.classList.add('active');
  var rows = document.querySelectorAll('#etfTable tbody tr');
  rows.forEach(function(r){
    if(group==='ALL'){ r.classList.remove('hidden'); return; }
    r.classList.toggle('hidden', r.dataset.group !== group);
  });
}
"""

    # ── Top picks cards ──────────────────────────────────────────────────────
    cards_html = ""
    for r in top6:
        tier_name, tier_col = get_tier(r["score"])
        grade = r["grade"]
        gcol, gbg = grade_colors.get(grade, ("#94a3b8","rgba(148,163,184,.1)"))
        r3_s = _fp(r.get("ret_3m"))
        r1_s = _fp(r.get("ret_1m"))
        cards_html += f"""
<div class="card">
  <div class="card-tick">{r['ticker']}</div>
  <div class="card-name">{r['name'][:28]}</div>
  <div class="card-score" style="color:{tier_col}">{r['score']}</div>
  <div class="card-ret">3M: {r3_s} &nbsp; 1M: {r1_s}</div>
  <span class="card-grade" style="color:{gcol};background:{gbg}">{grade}</span>
</div>"""

    # ── Filter buttons ───────────────────────────────────────────────────────
    filter_btns = '<span class="flt-label">Region</span>'
    for label in FILTER_GROUPS:
        active = ' active' if label == "ALL" else ""
        filter_btns += (
            f'<button class="flt{active}" '
            f'onclick="filterRegion(\'{label}\')">{label}</button>'
        )

    # ── Table rows ───────────────────────────────────────────────────────────
    rows_html = ""
    for i, r in enumerate(results, 1):
        grade = r["grade"]
        gcol, gbg = grade_colors.get(grade, ("#94a3b8","rgba(148,163,184,.1)"))
        tier_name, tier_col = get_tier(r["score"])

        def _rc(v):
            if v is None:
                return f'<td class="ret-cell" data-v="" style="color:#4b5563">--</td>'
            bg   = _ret_color(v)
            tcol = _ret_text_color(v)
            pct  = _fp(v)
            return (f'<td class="ret-cell" data-v="{v:.4f}" '
                    f'style="background:{bg};color:{tcol}">{pct}</td>')

        rel3 = r.get("rel_spy_3m")
        rel3_s = _fp(rel3) if rel3 is not None else "--"
        rel3_col = _ret_text_color(rel3) if rel3 is not None else "#4b5563"

        ma50_s  = _fp(r.get("pct_vs_ma50"),  1)
        ma200_s = _fp(r.get("pct_vs_ma200"), 1)
        volr    = r.get("vol_ratio")
        volr_s  = f"{volr:.2f}x" if volr is not None else "--"
        aum_s   = _fa(r.get("aum_m"))
        expr_s  = _fe(r.get("exp_ratio"))

        rows_html += f"""
<tr data-group="{r['region']}">
  <td data-v="{i}">{i}</td>
  <td class="tick" data-v="{r['ticker']}">{r['ticker']}</td>
  <td class="etf-name" data-v="{r['name']}">{r['name']}</td>
  <td class="region-tag" data-v="{r['region']}"><span class="rtag">{r['region']}</span></td>
  <td class="cat-tag" data-v="{r['category']}"><span class="ctag">{r['category']}</span></td>
  <td class="score-cell" data-v="{r['score']}" style="color:{tier_col}">{r['score']}
      <span class="grade-badge" style="color:{gcol};background:{gbg}">{grade}</span></td>
  {_rc(r.get("ret_1m"))}
  {_rc(r.get("ret_3m"))}
  {_rc(r.get("ret_6m"))}
  {_rc(r.get("ret_12m"))}
  <td data-v="{rel3 if rel3 is not None else ''}" style="color:{rel3_col};font-weight:600">{rel3_s}</td>
  <td data-v="{r.get('pct_vs_ma50') or ''}">{ma50_s}</td>
  <td data-v="{r.get('pct_vs_ma200') or ''}">{ma200_s}</td>
  <td data-v="{volr or ''}">{volr_s}</td>
  <td data-v="{r.get('aum_m') or ''}">{aum_s}</td>
  <td data-v="{r.get('exp_ratio') or ''}">{expr_s}</td>
</tr>"""

    # ── Summary stats ─────────────────────────────────────────────────────────
    strong   = sum(1 for r in results if r["score"] >= 70)
    building = sum(1 for r in results if 50 <= r["score"] < 70)
    positive = sum(1 for r in results if (r.get("ret_3m") or 0) >= 0)
    spy_s    = _fp(spy_ret_3m) if spy_ret_3m is not None else "--"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>International ETF Momentum &mdash; {ts}</title>
<style>
{css}
{_HELP_CSS}
</style>
</head>
<body>

<nav class="nav">
  <span class="nav-title">International ETF Momentum</span>
  <button class="help-btn" onclick="openHelp()">&#x24D8; How it works</button>
</nav>

<div class="hero">
  <h1>International ETF Momentum</h1>
  <div class="sub">Generated {ts} &nbsp;&bull;&nbsp; {total_fetched} ETFs fetched &nbsp;&bull;&nbsp; {len(results)} scored</div>
  <div class="stats">
    <div class="stat"><span class="stat-val">{strong}</span><span class="stat-lbl">Strong Momentum</span></div>
    <div class="stat"><span class="stat-val">{building}</span><span class="stat-lbl">Building</span></div>
    <div class="stat"><span class="stat-val">{positive}</span><span class="stat-lbl">Positive 3M</span></div>
    <div class="stat"><span class="stat-val" style="color:#94a3b8">{spy_s}</span><span class="stat-lbl">SPY 3M (benchmark)</span></div>
  </div>
</div>

<div class="top-section">
  <h2>Top Momentum Leaders</h2>
  <div class="cards">{cards_html}</div>
</div>

<div class="filters">{filter_btns}</div>

<div class="main">
  <div class="tbl-wrap">
    <table id="etfTable">
      <thead>
        <tr>
          <th onclick="sortTable(0)">#</th>
          <th onclick="sortTable(1)">Ticker</th>
          <th onclick="sortTable(2)">Name</th>
          <th onclick="sortTable(3)">Region</th>
          <th onclick="sortTable(4)">Category</th>
          <th onclick="sortTable(5)">Score / Grade</th>
          <th onclick="sortTable(6)">1M</th>
          <th onclick="sortTable(7)">3M</th>
          <th onclick="sortTable(8)">6M</th>
          <th onclick="sortTable(9)">12M</th>
          <th onclick="sortTable(10)">vs SPY 3M</th>
          <th onclick="sortTable(11)">vs MA50</th>
          <th onclick="sortTable(12)">vs MA200</th>
          <th onclick="sortTable(13)">Vol Ratio</th>
          <th onclick="sortTable(14)">AUM</th>
          <th onclick="sortTable(15)">Exp. Ratio</th>
        </tr>
      </thead>
      <tbody>
{rows_html}
      </tbody>
    </table>
  </div>
</div>

<div class="footer">
  International ETF Momentum Screener &nbsp;&bull;&nbsp;
  Data: yfinance &nbsp;&bull;&nbsp;
  Leveraged &amp; inverse ETFs excluded &nbsp;&bull;&nbsp;
  Not investment advice.
</div>

{_HELP_MODAL}
<script>{js}{_HELP_JS}</script>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="International ETF Momentum Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--region",  default="ALL",
                        choices=list(FILTER_GROUPS.keys()) + [k.lower() for k in FILTER_GROUPS]
                              + ["ASIA-PAC", "asia-pac"],
                        help="Filter by region (default: ALL)")
    parser.add_argument("--top",     type=int,   default=None,
                        help="Show top N ETFs by momentum score")
    parser.add_argument("--min-aum", type=float, default=100.0, dest="min_aum",
                        help="Minimum AUM in $M (default: 100)")
    parser.add_argument("--csv",     action="store_true",
                        help="Export results to CSV")
    args = parser.parse_args()

    region_filter = args.region.upper()
    suite_port    = int(os.environ.get("VALUATION_SUITE_PORT", "5050"))

    print("\n" + "=" * 58)
    print("  INTERNATIONAL ETF MOMENTUM SCREENER")
    print("=" * 58)

    # Build ticker list for requested region
    region_codes = FILTER_GROUPS.get(region_filter)
    if region_codes:
        tickers_meta = {t: m for t, m in ETF_UNIVERSE.items()
                        if m["region"] in region_codes}
    else:
        tickers_meta = dict(ETF_UNIVERSE)

    print(f"\n  Universe: {len(tickers_meta)} ETFs "
          f"(region filter: {region_filter})")
    print(f"  Min AUM:  ${args.min_aum:.0f}M")

    # Fetch SPY as benchmark first
    print("\n[1/3] Fetching SPY benchmark data...")
    spy_data = _fetch_one("SPY")
    spy_ret_3m = spy_data.get("ret_3m") if spy_data else None
    if spy_ret_3m is not None:
        print(f"  SPY 3M return: {spy_ret_3m*100:+.1f}%")
    else:
        print("  SPY 3M return: unavailable")

    # Fetch all ETFs in parallel
    print(f"\n[2/3] Fetching ETF data ({len(tickers_meta)} ETFs, 20 workers)...")
    raw = fetch_all(list(tickers_meta.keys()), workers=20)
    print(f"  Fetched {len(raw)} of {len(tickers_meta)} ETFs successfully.")

    # Score
    print("\n[3/3] Scoring ETFs...")
    results = []
    skipped = 0
    for ticker, d in raw.items():
        meta = tickers_meta[ticker]

        # AUM filter: use fetched AUM if available, else skip AUM filter
        aum = d.get("aum_m")
        if aum is not None and aum < args.min_aum:
            skipped += 1
            continue

        score = score_etf(d)
        tier_name, tier_col = get_tier(score)
        grade = get_grade(score)

        # Relative to SPY
        rel_spy_3m = None
        if d.get("ret_3m") is not None and spy_ret_3m is not None:
            rel_spy_3m = d["ret_3m"] - spy_ret_3m

        results.append({
            "ticker":      ticker,
            "name":        meta["name"],
            "region":      meta["region"],
            "category":    meta["category"],
            "score":       score,
            "grade":       grade,
            "tier":        tier_name,
            "ret_1m":      d.get("ret_1m"),
            "ret_3m":      d.get("ret_3m"),
            "ret_6m":      d.get("ret_6m"),
            "ret_12m":     d.get("ret_12m"),
            "pct_vs_ma50":  d.get("pct_vs_ma50"),
            "pct_vs_ma200": d.get("pct_vs_ma200"),
            "vol_ratio":   d.get("vol_ratio"),
            "ann_vol":     d.get("ann_vol"),
            "aum_m":       d.get("aum_m"),
            "exp_ratio":   d.get("exp_ratio"),
            "rel_spy_3m":  rel_spy_3m,
        })

    results.sort(key=lambda x: -x["score"])
    print(f"  Scored: {len(results)} | Below min-AUM / no data: {skipped + (len(tickers_meta) - len(raw))}")

    if not results:
        print("ERROR: No ETFs scored. Check internet connection or try --region ALL.")
        sys.exit(1)

    # Apply top-N cap after scoring
    if args.top:
        results = results[:args.top]
        print(f"  Trimmed to top {args.top} by momentum score.")

    # Print summary to terminal
    print("\n  Tier Distribution:")
    for name, lo, hi, _ in TIERS:
        grp = [r for r in results if r["tier"] == name]
        if grp:
            bar = "|" * min(len(grp), 50)
            print(f"    {name:<22} {str(len(grp)):>3}  {bar}")

    print("\n  Top 10 by Momentum Score:")
    for i, r in enumerate(results[:10], 1):
        r3 = f"{r['ret_3m']*100:+.1f}%" if r.get("ret_3m") is not None else "--"
        r12 = f"{r['ret_12m']*100:+.1f}%" if r.get("ret_12m") is not None else "--"
        print(f"    {i:>2}. {r['ticker']:<6}  Score={r['score']:>3}  [{r['grade']}]  "
              f"3M={r3:<8}  12M={r12:<8}  {r['region']}/{r['category']}")

    # Build HTML
    ts        = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    out_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "momentumData")
    os.makedirs(out_dir, exist_ok=True)
    fname     = f"intl_momentum_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"
    fpath     = os.path.join(out_dir, fname)

    html = build_html(results, ts, len(raw), spy_ret_3m, suite_port=suite_port)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)

    kb = os.path.getsize(fpath) // 1024
    print(f"\n  HTML saved: {fpath}  ({kb} KB)")

    # CSV export
    if args.csv:
        csv_path = fpath.replace(".html", ".csv")
        csv_cols = ["ticker","name","region","category","score","grade","tier",
                    "ret_1m","ret_3m","ret_6m","ret_12m","rel_spy_3m",
                    "pct_vs_ma50","pct_vs_ma200","vol_ratio","ann_vol","aum_m","exp_ratio"]
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=csv_cols, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"  CSV saved: {csv_path}")

    # Open in browser
    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        import platform, subprocess as _sp
        if platform.system() == "Darwin":
            _sp.Popen(["open", fpath])
        else:
            import webbrowser
            webbrowser.open(f"file://{fpath}")


if __name__ == "__main__":
    main()
