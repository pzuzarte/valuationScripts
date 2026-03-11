"""
Growth & Momentum Screener
===========================
Identifies high-growth stocks with upside potential using a four-pillar scoring model:
  • Pillar 1 — Quality (0-35 pts)         ROIC, ROE, margins, FCF generation, liquidity
  • Pillar 2 — Growth Momentum (0-35 pts) Revenue/EPS/FCF/GP/NI/EBITDA growth + R&D ratio
  • Pillar 3 — Technical Signal (0-20 pts) TV rating, MoneyFlow, RSI, Stochastic, price perf
  • Pillar 4 — Valuation (0-10 pts)       PEG, P/FCF, EV/EBITDA, model upside

Price targets use three methods blended (60% PEG / 40% Revenue / 20% EV-EBITDA, renormalised):
  • Forward PEG Projection     — bear/base/bull targets (PEG 0.75 / 1.0 / 1.5)
  • Revenue Extrapolation      — forward revenue × net margin × sector P/E
  • EV/EBITDA Back-Into-Equity — (forward EBITDA × growth-adj. multiple − net debt) / shares

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install tradingview-screener

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMAND LINE ARGUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --index  INDEX    Which index to scan. Default: SPX
                    Options:
                      SPX   S&P 500      (~497 stocks, NYSE + NASDAQ large caps)
                      NDX   Nasdaq 100   (~100 stocks, NASDAQ large caps)
                      RUT   Russell 2000 (~2000 stocks, small/mid cap US)
                      TSX   TSX          (~130 stocks, Canadian equities)

                    Examples:
                      python sp500_growth_screener.py --index SPX
                      python sp500_growth_screener.py --index NDX
                      python sp500_growth_screener.py --index TSX

  --top    N        Limit fetch to top N stocks (by market cap). Overrides index default.
                    Useful for quick tests.
                    Example:
                      python sp500_growth_screener.py --top 50

  --csv             Also export results to sp500_growth_report.csv alongside the HTML.
                    Example:
                      python sp500_growth_screener.py --csv
                      python sp500_growth_screener.py --index NDX --csv

  --help            Print this help message and exit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python sp500_growth_screener.py                        # S&P 500, full scan
  python sp500_growth_screener.py --index NDX            # Nasdaq 100
  python sp500_growth_screener.py --index TSX --csv      # TSX + CSV export
  python sp500_growth_screener.py --index RUT --top 200  # Russell 2000, top 200
"""

import sys, math, datetime, webbrowser, os, csv, threading, logging, json, time as _time, argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, wait as _cf_wait, FIRST_COMPLETED as _CF_FIRST
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tradingview_screener import Query, col
except ImportError:
    print("ERROR: tradingview-screener not installed.")
    print("Run:  pip install tradingview-screener")
    sys.exit(1)

try:
    import yfinance as yf
    _YF_AVAIL = True
except ImportError:
    _YF_AVAIL = False

# ── yfinance 401 / Invalid Crumb recovery ─────────────────────────────────────
# Yahoo Finance rotates its crumb token.  yfinance sets hide_exceptions=True by
# default, meaning HTTPError is caught internally, printed via logger.error(),
# and the property returns None — the exception never reaches our code.
#
# Fix: install a logging handler on the 'yfinance' logger that sets a
# thread-local flag when it detects a 401 message.  _fetch_yf_data_gs checks
# the flag after every batch of property accesses and retries with a fresh
# session when needed.

_yf_refresh_lock = threading.Lock()
_yf_refresh_ts   = [0.0]   # last refresh timestamp — list for in-place mutation
_yf_401_tls      = threading.local()  # per-thread detected flag

def _is_yf_401_str(s: str) -> bool:
    """Return True if string *s* indicates a Yahoo Finance 401 / auth error."""
    return ("401" in s or "Invalid Crumb" in s
            or ("Unauthorized" in s and "yahoo" in s.lower())
            or "User is unable to access this feature" in s)

class _YF401LogHandler(logging.Handler):
    """Logging handler that sets a thread-local flag on 401 log messages.

    yfinance logs HTTP errors via utils.get_yf_logger().error() when
    hide_exceptions=True (the default).  This handler intercepts those messages
    so _fetch_yf_data_gs can detect and recover from 401s without needing the
    exception to propagate.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if _is_yf_401_str(record.getMessage()):
                _yf_401_tls.detected = True
        except Exception:
            pass

def _yf_refresh_session() -> bool:
    """Delete stale cookie cache and reset the YfData singleton.

    Thread-safe with 30-second debounce so parallel worker threads don't all
    hammer the refresh simultaneously — only the first thread does the work;
    subsequent threads within the debounce window return False immediately.
    Returns True if a refresh was performed, False if debounced.
    """
    import glob, time
    with _yf_refresh_lock:
        now = time.time()
        if now - _yf_refresh_ts[0] < 30:
            return False          # another thread just refreshed — skip
        _yf_refresh_ts[0] = now

        cache_dir = os.path.expanduser("~/Library/Caches/py-yfinance")
        for f in glob.glob(os.path.join(cache_dir, "cookies.db*")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            from yfinance.data import YfData
            yd = YfData()
            yd._crumb  = None
            yd._cookie = None
        except Exception:
            pass
        print("  [yfinance] session refreshed after 401 Invalid Crumb — retrying…")
        return True

# Install the handler once at module load (no-op if yfinance not available)
if _YF_AVAIL:
    logging.getLogger("yfinance").addHandler(_YF401LogHandler())

# ── yfinance disk cache (24-hour TTL) ─────────────────────────────────────────
# Saves per-ticker fundamentals to .yf_cache.json so repeat runs within a day
# skip all yfinance HTTP calls for already-fetched tickers.
_YF_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yf_cache.json")
_YF_CACHE_TTL  = 24 * 3600          # seconds — refresh after 24 hours
_yf_cache: dict = {}                 # populated by _load_yf_cache() at run time
_yf_cache_lock  = threading.Lock()  # guards writes from concurrent worker threads

def _load_yf_cache() -> None:
    """Load cache from disk, discarding entries older than TTL."""
    global _yf_cache
    try:
        if os.path.exists(_YF_CACHE_FILE):
            with open(_YF_CACHE_FILE) as _f:
                raw = json.load(_f)
            now = _time.time()
            _yf_cache = {k: v for k, v in raw.items()
                         if now - v.get("_ts", 0) < _YF_CACHE_TTL}
            if _yf_cache:
                print(f"  [cache] loaded {len(_yf_cache)} cached yfinance entries "
                      f"(TTL {_YF_CACHE_TTL//3600}h)")
    except Exception:
        _yf_cache = {}

def _save_yf_cache() -> None:
    """Persist the in-memory cache to disk (best-effort)."""
    try:
        with _yf_cache_lock:
            with open(_YF_CACHE_FILE, "w") as _f:
                json.dump(_yf_cache, _f)
    except Exception:
        pass

# ── Optional: import point-in-time backtest engine from valuationMaster ───────
# Used when --backtest flag is passed. Falls back to consensus ranking otherwise.
try:
    import urllib.request as _urllib_req
    _VM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "3_valuationTool")
    if _VM_DIR not in sys.path:
        sys.path.insert(0, _VM_DIR)
    from valuationMaster import (
        fetch_historical_prices       as _bt_fetch_prices,
        fetch_historical_fundamentals as _bt_fetch_annual,
        fetch_historical_fundamentals_ttm as _bt_fetch_quarterly,
        _pick_snapshot_for_date       as _bt_pick_snap,
        _rebuild_d_from_snapshot      as _bt_rebuild_d,
        _run_methods_for_snapshot     as _bt_run_methods,
    )
    _BT_ENGINE_AVAIL = True
except Exception:
    _BT_ENGINE_AVAIL = False

from valuation_models import (
    MARGIN_OF_SAFETY,
    get_growth_risk_flags,
    calc_peg_targets,
    calc_rev_target,
    calc_ev_target,
    calc_quality_score,
    calc_growth_momentum_score,
    calc_momentum_score,
    calc_valuation_score,
    derive_sentiment,
    derive_accumulation,
    run_scurve_tam,
    growth_adjusted_multiples,
    run_forward_peg,
    run_ev_ntm_revenue,
    run_pie,
    run_fcf_yield,
    run_graham_number,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# PEG targets for bear / base / bull price targets
PEG_BEAR   = 0.75
PEG_BASE   = 1.0
PEG_BULL   = 1.5

# Composite weight: PEG vs Revenue method
PEG_WEIGHT = 0.60
REV_WEIGHT = 0.40

# Growth caps — prevent absurd extrapolations
MAX_GROWTH_RATE   = 0.80   # cap at 80% annual growth for projections
MAX_FWD_PE        = 80.0   # cap applied P/E at 80x
MIN_GROWTH_THRESH = 0.05   # stocks below 5% revenue growth excluded from top tiers

# Upside tiers
TIERS = [
    ("EXCEPTIONAL GROWTH",   60, 999, "#00e5a0"),
    ("STRONG GROWTH",        30,  60, "#00c896"),
    ("MODERATE GROWTH",      10,  30, "#4fd4a0"),
    ("FAIRLY PRICED",        -5,  10, "#718096"),
    ("STRETCHED VALUATION", -999, -5, "#fc8181"),
]

# Fields to fetch
FIELDS = [
    "name", "close", "market_cap_basic", "sector",
    "total_shares_outstanding", "float_shares_outstanding",
    "earnings_per_share_diluted_ttm", "earnings_per_share_basic_ttm",
    "earnings_per_share_forecast_next_fq",   # analyst fwd EPS estimate (next quarter)
    "earnings_per_share_forecast_next_fy",   # analyst fwd EPS estimate (full next fiscal year) ← gold standard
    "revenue_forecast_next_fy",              # analyst fwd revenue estimate (full next fiscal year) ← gold standard
    "earnings_per_share_diluted_yoy_growth_ttm",
    "earnings_per_share_diluted_yoy_growth_fq",  # most recent qtr EPS growth YoY — fresher signal
    "earnings_per_share_diluted_yoy_growth_fy",  # full-year EPS growth — trend context
    "last_annual_eps",                            # last full fiscal year EPS
    "total_revenue_yoy_growth_fq",               # most recent qtr revenue growth YoY
    "total_revenue_yoy_growth_fy",               # full-year revenue growth — trend context
    "last_annual_revenue",                        # last full fiscal year revenue
    "ebitda_yoy_growth_ttm",                      # EBITDA growth — operational leverage signal
    "gross_profit_yoy_growth_ttm",               # gross profit growth — margin expansion signal
    "total_revenue", "total_revenue_yoy_growth_ttm",
    "gross_margin", "operating_margin", "after_tax_margin",
    "free_cash_flow", "free_cash_flow_margin_ttm",
    "return_on_equity", "price_earnings_ttm",
    "beta_1_year", "debt_to_equity",
    "price_52_week_high", "price_52_week_low",
    "Perf.1M", "Perf.3M", "Perf.6M",
    "Recommend.All",           # TradingView technical rating (-1 to 1)
    "RSI",                     # RSI(14) — overbought/oversold momentum
    "Mom",                     # Momentum indicator
    "MACD.hist",               # MACD histogram — accumulation pressure
    "MACD.macd",               # MACD line
    "MACD.signal",             # MACD signal line
    "relative_volume_10d_calc",# Relative volume vs 10d avg
    "average_volume_30d_calc", # 30-day avg volume
    "average_volume_90d_calc", # 90-day avg volume
    "ADX",                     # Average Directional Index — trend strength
    "ADX+DI",                  # +DI — buying pressure
    "ADX-DI",                  # -DI — selling pressure
    "return_on_invested_capital",       # ROIC — best single quality metric
    "return_on_assets",                  # ROA — capital efficiency
    "free_cash_flow_yoy_growth_ttm",     # FCF growth rate TTM
    "free_cash_flow_yoy_growth_fq",      # FCF growth rate (quarter)
    "research_and_dev_ratio_ttm",        # R&D as % of revenue
    "gross_profit_yoy_growth_fq",        # gross profit growth (quarter)
    "net_income_yoy_growth_ttm",         # net income growth TTM
    "net_income_yoy_growth_fq",          # net income growth (quarter)
    "current_ratio",                     # liquidity
    "quick_ratio",                       # liquidity (tighter)
    "earnings_release_next_date",        # upcoming earnings catalyst
    "price_free_cash_flow_ttm",          # P/FCF valuation
    "enterprise_value_ebitda_ttm",       # EV/EBITDA valuation
    "MoneyFlow",                         # Money Flow index — vol-weighted RSI
    "Stoch.K",                           # Stochastic %K
    "Stoch.D",                           # Stochastic %D
    "W.R",                               # Williams %R
    "Perf.YTD",                          # YTD performance
    "total_revenue_qoq_growth_fq",       # revenue QoQ sequential growth
    "earnings_per_share_diluted_qoq_growth_fq",  # EPS QoQ sequential growth
    "price_target_1y",         # analyst consensus 1-year price target
    "price_target_1y_delta",   # % upside to analyst consensus target
    "net_debt",           # net debt (MRQ) — EV bridge to equity value
    "net_income",
    "ebitda",
    "EMA13",                   # 13-day Exponential Moving Average
    "EMA50",                   # 50-day Exponential Moving Average
    "EMA200",                  # 200-day Exponential Moving Average
]

# Sector median P/E — used in revenue extrapolation method
SECTOR_PE = {
    "Technology":             28.0,
    "Communication Services": 20.0,
    "Consumer Discretionary": 22.0,
    "Consumer Staples":       18.0,
    "Health Care":            22.0,
    "Financials":             13.0,
    "Industrials":            20.0,
    "Materials":              16.0,
    "Energy":                 12.0,
    "Utilities":              16.0,
    "Real Estate":            30.0,
}
MARKET_PE = 22.0

# Sector median EV/EBITDA — used in EV-based target method
# Source: historical medians, biased toward growth-adjusted levels
SECTOR_EV_EBITDA = {
    "Technology":             22.0,
    "Communication Services": 14.0,
    "Consumer Discretionary": 14.0,
    "Consumer Staples":       12.0,
    "Health Care":            16.0,
    "Financials":              9.0,
    "Industrials":            14.0,
    "Materials":              10.0,
    "Energy":                  7.0,
    "Utilities":              11.0,
    "Real Estate":            20.0,
}
MARKET_EV_EBITDA = 14.0

# ── FETCH ─────────────────────────────────────────────────────────────────────
# ── INDEX CONSTITUENTS ───────────────────────────────────────────────────────
# Hardcoded ticker lists — used as primary filter via col("name").isin()
# Russell 2000 uses exchange fallback (too large / frequently changing)

# Current index constituents (as of early 2025)
# These are embedded directly — no network fetch needed at runtime

SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE",
    "ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG",
    "AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL",
    "ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BK","BBWI","BAX","BDX","BBY","TECH","BIO","BIIB","BLK","BX","BA","BKNG","BWA","BXP","BSX",
    "BMY","AVGO","BR","BRO","BF.B","BLDR","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL",
    "CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR",
    "CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL",
    "CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA",
    "CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DECK","DE","DELL","DAL","DVN","DXCM",
    "FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY",
    "ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS",
    "EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT",
    "FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN",
    "IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA",
    "DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM",
    "HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU",
    "ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE",
    "KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS",
    "LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC",
    "MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU",
    "MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ",
    "NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE",
    "NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW",
    "PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL",
    "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF",
    "RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
    "SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK",
    "SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP",
    "TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB",
    "TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR",
    "VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT",
    "DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WHR","WTW","WRK","WYNN","XEL","XYL",
    "YUM","ZBRA","ZBH","ZTS",
]

NASDAQ100 = [
    "ADBE","AMD","ABNB","GOOGL","GOOG","AMZN","AEP","AMGN","ADI","ANSS","AAPL","AMAT","APP",
    "ASML","AZN","TEAM","ADSK","ADP","AXON","BIIB","BKNG","AVGO","CDNS","CDW","CHTR","CTAS",
    "CSCO","CCEP","CTSH","CMCSA","CEG","CPRT","CSGP","COST","CRWD","CSX","DXCM","FANG","DDOG",
    "DLTR","EA","EXC","FAST","FTNT","GEHC","GILD","GFS","HON","IDXX","ILMN","INTC","INTU","ISRG",
    "KDP","KLAC","KHC","LRCX","LIN","LULU","MAR","MRVL","MELI","META","MCHP","MU","MSFT","MRNA",
    "MDLZ","MDB","MNST","NFLX","NVDA","NXPI","ORLY","ON","PCAR","PANW","PAYX","PYPL","PDD","QCOM",
    "REGN","ROP","ROST","SBUX","SNPS","TTWO","TMUS","TSLA","TXN","TTD","VRSK","VRTX","WBD","WBA",
    "WDAY","XEL","ZS","ZM",
]

TSX = [
    "SHOP","RY","TD","ENB","CP","CNR","BN","BAM","BCE","BMO","BNS","MFC","SLF","TRI","ABX",
    "CCO","IMO","CNQ","SU","CVE","PPL","TRP","K","POW","GWO","IAG","FFH","L","EMA","FTS",
    "H","CAR","REI","CHP","HR","SRU","AP","DIR","NWH","CSH","MRG","CRT","WPM","AEM","AGI",
    "KL","FNV","OR","EDV","MAG","ELD","SSL","IMG","PVG","OGC","TMX","X","ACO","MG","MDA",
    "OTEX","DSG","GIB","BB","CIGI","PHO","ATD","DOL","CTC","MRU","EMP","SAP","QSR","MTY",
    "TFII","TIH","WSP","STN","ATA","BYD","NFI","TDG","RBA","LIF","CCL","ITP","TCL","TXF",
    "WCN","BIN","GFL","SNC","STLC","HPS","FTT","IFP","PRE","FSV","TIXT","AC","CAE","CCA",
    "CHR","WJA","ABX","G","YRI","NGD","CG","LUG","BTG","SGY","TVE","BTE","ARX","BIR","CR",
    "ERF","GTE","MEG","NVA","OBE","PEY","PXT","RRX","SCL","SES","SPB","TOU","VET","VLE","WCP",
]

INDEX_CONFIG = {
    "SPX":    {"name": "S&P 500",                    "tickers": SP500},
    "NDX":    {"name": "Nasdaq 100",                 "tickers": NASDAQ100},
    "RUT":    {"name": "Russell 2000",               "tickers": None},
    "TSX":    {"name": "TSX",                        "tickers": TSX},
    "SPXRUT": {"name": "S&P 500 + Russell 2000",     "tickers": None},
}


def fetch_stocks(index_code, limit=None):
    # ── Combined index: fetch both sub-indices and merge ─────────────────────
    if index_code.upper() == "SPXRUT":
        print("  Fetching S&P 500 + Russell 2000 (two queries)...")
        spx_df = fetch_stocks("SPX", None)
        rut_df = fetch_stocks("RUT", None)
        combined = pd.concat([spx_df, rut_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["name"], keep="first")
        combined = combined.sort_values("market_cap_basic", ascending=False).reset_index(drop=True)
        combined["index_label"] = "S&P 500 + Russell 2000"
        print(f"  S&P 500 + Russell 2000: {len(combined)} stocks")
        return combined

    cfg  = INDEX_CONFIG.get(index_code.upper(), INDEX_CONFIG["SPX"])
    name = cfg["name"]
    print(f"  Querying TradingView for {name}...")

    tickers = cfg.get("tickers")

    # ── Primary: isin filter on known constituents ────────────────────────────
    if tickers:
        try:
            lim = len(tickers) + 20   # always fetch full constituent list
            _, df = (
                Query().select(*FIELDS)
                .where(col("name").isin(tickers), col("is_primary") == True)
                .order_by("market_cap_basic", ascending=False)
                .limit(lim).get_scanner_data()
            )
            df["index_label"] = name
            print(f"  {name}: {len(df)} stocks")
            return df
        except Exception as e:
            print(f"  isin filter failed ({str(e)[:60]}), trying exchange fallback...")

    # ── Fallback: exchange / country filter ───────────────────────────────────
    frames = []

    if index_code == "TSX":
        try:
            _, df = (
                Query().select(*FIELDS)
                .where(col("country") == "Canada", col("is_primary") == True,
                       col("typespecs").has_none_of("preferred"))
                .order_by("market_cap_basic", ascending=False)
                .limit(400).get_scanner_data()
            )
            df["index_label"] = name
            print(f"  {name}: {len(df)} stocks (country fallback)")
            return df
        except Exception as e:
            print(f"  TSX country filter failed: {str(e)[:60]}")

    else:
        # US indexes — NYSE + NASDAQ queried in parallel to halve fetch time
        cap = (503 if index_code == "SPX" else 110 if index_code == "NDX" else 2000)
        def _query_exchange(exch):
            _, df = (
                Query().select(*FIELDS)
                .where(col("exchange") == exch, col("is_primary") == True,
                       col("typespecs").has_none_of("preferred"))
                .order_by("market_cap_basic", ascending=False)
                .limit(1500).get_scanner_data()
            )
            return exch, df
        with ThreadPoolExecutor(max_workers=2) as _tv_pool:
            _tv_futs = {_tv_pool.submit(_query_exchange, exch): exch
                        for exch in ["NYSE", "NASDAQ"]}
            for _tv_fut in as_completed(_tv_futs):
                try:
                    exch, df = _tv_fut.result()
                    frames.append(df)
                    print(f"    {exch}: {len(df)} stocks")
                except Exception as e:
                    exch = _tv_futs[_tv_fut]
                    print(f"    {exch} failed: {str(e)[:60]}")

        if frames:
            combined = pd.concat([f.reset_index(drop=True) for f in frames], ignore_index=True)
            combined = combined.drop_duplicates(subset=["name"], keep="first")
            combined = combined.sort_values("market_cap_basic", ascending=False).reset_index(drop=True)
            if index_code == "RUT":
                combined = combined.iloc[503:2503].reset_index(drop=True)
            else:
                combined = combined.head(cap).reset_index(drop=True)
            combined["index_label"] = name
            print(f"  {name}: {len(combined)} stocks (exchange fallback)")
            return combined

    # ── Last resort ───────────────────────────────────────────────────────────
    print("  All filters failed — using large-cap global fallback")
    _, df = (
        Query().select(*FIELDS)
        .where(col("market_cap_basic") > 8e9, col("is_primary") == True,
               col("typespecs").has_none_of("preferred"), col("close") > 0)
        .order_by("market_cap_basic", ascending=False)
        .limit(503).get_scanner_data()
    )
    df["index_label"] = name
    return df


# ── PARSE ─────────────────────────────────────────────────────────────────────
def parse_row(row):
    def s(f, d=None):
        v = row.get(f, d)
        if v is None or (isinstance(v, float) and math.isnan(v)): return d
        try: return float(v)
        except Exception: return d

    price       = s("close", 0)
    mktcap      = s("market_cap_basic", 0)
    sector      = str(row.get("sector") or "Unknown")
    index_label = str(row.get("index_label") or "S&P 500")
    shares_total = s("total_shares_outstanding")
    shares_float = s("float_shares_outstanding")
    shares       = shares_total or shares_float

    # Closely-held % = non-float shares as % of total
    # Captures insider/strategic/restricted ownership — not freely tradeable
    if shares_total and shares_float and shares_total > 0:
        closely_held_pct = (shares_total - shares_float) / shares_total * 100
        closely_held_pct = max(0.0, min(100.0, closely_held_pct))  # clamp
    else:
        closely_held_pct = None

    eps_ttm    = s("earnings_per_share_diluted_ttm") or s("earnings_per_share_basic_ttm")
    eps_fy     = s("last_annual_eps")                       # last full fiscal year EPS
    eps_fwd_fq = s("earnings_per_share_forecast_next_fq")  # analyst est., next quarter
    eps_fwd_fy = s("earnings_per_share_forecast_next_fy")  # analyst est., full next FY ← preferred
    rev_fwd_fy = s("revenue_forecast_next_fy")             # analyst est., full next FY revenue ← preferred

    # Growth rates — we capture TTM, most-recent-quarter YoY, and full-year
    # to derive the best possible forward growth estimate
    eps_growth_ttm = s("earnings_per_share_diluted_yoy_growth_ttm")  # trailing 12m YoY %
    eps_growth_fq  = s("earnings_per_share_diluted_yoy_growth_fq")   # most recent qtr YoY %
    eps_growth_fy  = s("earnings_per_share_diluted_yoy_growth_fy")   # last full year YoY %

    rev      = s("total_revenue")
    rev_fy   = s("last_annual_revenue")                    # last full fiscal year revenue
    rev_growth_ttm = s("total_revenue_yoy_growth_ttm")    # trailing 12m YoY %
    rev_growth_fq  = s("total_revenue_yoy_growth_fq")     # most recent qtr YoY % — freshest
    rev_growth_fy  = s("total_revenue_yoy_growth_fy")     # last full year YoY %

    # ── Derive best forward growth estimates ──────────────────────────────────
    # Strategy: prefer the most recent quarterly signal, tempered by trend context.
    # If qtr growth is accelerating vs TTM → use weighted blend (more weight on qtr)
    # If qtr growth is decelerating vs TTM → use the lower figure (conservative)
    # This prevents both over-extrapolating a one-off quarter and ignoring acceleration.

    def best_growth(fq, ttm, fy):
        """Pick best growth rate from available signals with trend-awareness."""
        signals = [x for x in [fq, ttm, fy] if x is not None]
        if not signals: return None
        if len(signals) == 1: return signals[0]
        # Use quarterly as primary (freshest), TTM as anchor
        primary = fq if fq is not None else ttm
        anchor  = ttm if ttm is not None else fy
        if primary > anchor:
            # Accelerating — blend 70% primary + 30% anchor (reward but temper)
            return primary * 0.70 + anchor * 0.30
        else:
            # Decelerating or stable — blend 50/50 (be conservative)
            return primary * 0.50 + anchor * 0.50

    fwd_eps_growth = best_growth(eps_growth_fq, eps_growth_ttm, eps_growth_fy)
    fwd_rev_growth = best_growth(rev_growth_fq, rev_growth_ttm, rev_growth_fy)

    # Convenience aliases used throughout
    eps_growth = fwd_eps_growth   # forward-blended EPS growth %
    rev_growth = fwd_rev_growth   # forward-blended revenue growth %
    gross_margin  = s("gross_margin")                     # percent e.g. 74.5
    op_margin     = s("operating_margin")                 # percent e.g. 30.2
    net_margin    = s("after_tax_margin")                 # percent e.g. 22.1
    fcf           = s("free_cash_flow")
    fcf_margin    = s("free_cash_flow_margin_ttm")        # percent

    roe      = s("return_on_equity")                      # percent
    pe_ttm   = s("price_earnings_ttm")
    beta     = s("beta_1_year", 1.0) or 1.0
    de_ratio = s("debt_to_equity")
    net_debt_v = s("net_debt")          # net debt (MRQ)
    net_income = s("net_income")
    ebitda     = s("ebitda")

    hi52     = s("price_52_week_high")
    lo52     = s("price_52_week_low")
    perf_1m  = s("Perf.1M")
    perf_3m  = s("Perf.3M")
    perf_6m  = s("Perf.6M")
    tv_rating    = s("Recommend.All")
    rsi          = s("RSI")
    mom          = s("Mom")
    macd_hist    = s("MACD.hist")
    macd_line    = s("MACD.macd")
    macd_signal  = s("MACD.signal")
    rel_vol      = s("relative_volume_10d_calc")
    avg_vol_30d  = s("average_volume_30d_calc")
    avg_vol_90d  = s("average_volume_90d_calc")
    adx          = s("ADX")
    adx_plus     = s("ADX+DI")
    adx_minus    = s("ADX-DI")
    analyst_target = s("price_target_1y")
    analyst_upside = s("price_target_1y_delta")

    # ── New fields ───────────────────────────────────────────────────────
    roic           = s("return_on_invested_capital")    # % e.g. 45.2
    roa            = s("return_on_assets")              # % e.g. 18.3
    fcf_growth_ttm = s("free_cash_flow_yoy_growth_ttm")# % YoY
    fcf_growth_fq  = s("free_cash_flow_yoy_growth_fq") # % YoY (quarter)
    rd_ratio       = s("research_and_dev_ratio_ttm")    # R&D % of revenue
    gp_growth_ttm  = s("gross_profit_yoy_growth_ttm")  # gross profit growth %
    gp_growth_fq   = s("gross_profit_yoy_growth_fq")   # gross profit growth % (qtr)
    ni_growth_ttm  = s("net_income_yoy_growth_ttm")    # net income growth %
    ni_growth_fq   = s("net_income_yoy_growth_fq")     # net income growth % (qtr)
    ebitda_growth  = s("ebitda_yoy_growth_ttm")        # EBITDA growth %
    current_ratio  = s("current_ratio")                # liquidity ratio
    quick_ratio_v  = s("quick_ratio")                  # tighter liquidity
    earnings_next_ts = s("earnings_release_next_date") # Unix timestamp
    p_fcf          = s("price_free_cash_flow_ttm")     # P/FCF ratio
    ev_ebitda      = s("enterprise_value_ebitda_ttm")  # EV/EBITDA
    money_flow     = s("MoneyFlow")                    # Money Flow (14) — vol-weighted RSI
    stoch_k        = s("Stoch.K")                      # Stochastic %K
    stoch_d        = s("Stoch.D")                      # Stochastic %D
    williams_r     = s("W.R")                          # Williams %R (0=overbought, -100=oversold)
    perf_ytd       = s("Perf.YTD")                     # YTD return (decimal: -0.08 = -8%)
    rev_qoq        = s("total_revenue_qoq_growth_fq")  # sequential revenue growth %
    eps_qoq        = s("earnings_per_share_diluted_qoq_growth_fq")  # sequential EPS growth %

    # Convert earnings timestamp to days from now
    days_to_earnings = None
    if earnings_next_ts:
        try:
            earn_date = datetime.datetime.utcfromtimestamp(earnings_next_ts).date()
            today = datetime.date.today()
            diff = (earn_date - today).days
            if -5 <= diff <= 180:   # within reasonable window
                days_to_earnings = diff
        except Exception: pass

    # Blended FCF growth (same logic as EPS/rev)
    fcf_growth = best_growth(fcf_growth_fq, fcf_growth_ttm, None)
    gp_growth  = best_growth(gp_growth_fq, gp_growth_ttm, None)
    ni_growth  = best_growth(ni_growth_fq, ni_growth_ttm, None)

    # Perf.YTD comes as decimal (-0.0079), convert to %
    if perf_ytd is not None:
        perf_ytd = perf_ytd * 100

    # ── EMA values for below-EMA filter ───────────────────────────────────────
    ema13  = s("EMA13")
    ema50  = s("EMA50")
    ema200 = s("EMA200")
    below_ema13  = (price < ema13)  if (price and ema13)  else None
    below_ema50  = (price < ema50)  if (price and ema50)  else None
    below_ema200 = (price < ema200) if (price and ema200) else None

    pos52 = None
    if hi52 and lo52 and hi52 > lo52 and price:
        pos52 = (price - lo52) / (hi52 - lo52)

    # Annualised growth rates as decimal (TV gives percent)
    eps_growth_dec = (fwd_eps_growth / 100.0) if fwd_eps_growth is not None else None
    rev_growth_dec = (fwd_rev_growth / 100.0) if fwd_rev_growth is not None else None

    # ── Forward EPS (annualised) ─────────────────────────────────────────────
    # Priority:
    #   1. Analyst next-FY consensus (full fiscal year) — gold standard
    #   2. Annualise analyst next-quarter estimate (4× next_fq)
    #   3. Grow TTM EPS by blended forward growth rate
    #   4. Grow last annual EPS by blended forward growth rate

    if eps_fwd_fy and eps_fwd_fy > 0:
        # Best signal: full fiscal year analyst consensus EPS estimate
        # This is exactly what Wall St. uses for forward P/E and PEG calculations
        fwd_eps = eps_fwd_fy
        eps_source = "analyst_fy"
    elif eps_fwd_fq and eps_fwd_fq > 0:
        # Annualise next-quarter estimate × 4
        fwd_eps_annualised = eps_fwd_fq * 4
        # Sanity check: shouldn't be more than 3x TTM or less than 0.1x TTM
        if eps_ttm and eps_ttm > 0:
            fwd_eps_annualised = max(eps_ttm * 0.10, min(fwd_eps_annualised, eps_ttm * 3.0))
        fwd_eps = fwd_eps_annualised
        eps_source = "analyst_fq×4"
    elif eps_ttm and eps_ttm > 0 and eps_growth_dec is not None:
        fwd_eps = eps_ttm * (1 + min(eps_growth_dec, MAX_GROWTH_RATE))
        eps_source = "ttm×growth"
    elif eps_fy and eps_fy > 0 and eps_growth_dec is not None:
        fwd_eps = eps_fy * (1 + min(eps_growth_dec, MAX_GROWTH_RATE))
        eps_source = "fy×growth"
    else:
        fwd_eps = None
        eps_source = "none"

    # ── Implied forward EPS growth rate ──────────────────────────────────────
    # Derive from annualised forward EPS vs TTM — this is the number used
    # in the PEG calculation, and it's forward-looking not backward-looking
    fwd_eps_growth_implied = None
    if fwd_eps and fwd_eps > 0 and eps_ttm and eps_ttm > 0 and fwd_eps != eps_ttm:
        fwd_eps_growth_implied = (fwd_eps - eps_ttm) / abs(eps_ttm) * 100

    # Use implied growth if available, else fall back to blended estimate
    peg_growth_rate = fwd_eps_growth_implied if fwd_eps_growth_implied is not None else fwd_eps_growth

    # ── Forward revenue projection ────────────────────────────────────────────
    # Use blended forward revenue growth rate applied to TTM revenue (or FY revenue)
    if rev_fwd_fy and rev_fwd_fy > 0:
        # Best signal: full fiscal year analyst consensus revenue estimate
        fwd_rev = rev_fwd_fy
        rev_source = "analyst_fy"
    elif rev and rev_growth_dec is not None:
        fwd_rev = rev * (1 + min(rev_growth_dec, MAX_GROWTH_RATE))
        rev_source = "ttm×fwd_growth"
    elif rev_fy and rev_growth_dec is not None:
        fwd_rev = rev_fy * (1 + min(rev_growth_dec, MAX_GROWTH_RATE))
        rev_source = "fy×fwd_growth"
    elif rev:
        fwd_rev = rev
        rev_source = "flat"
    else:
        fwd_rev = None
        rev_source = "none"

    return dict(
        ticker=str(row.get("name", "?")), price=price, market_cap=mktcap,
        sector=sector, index_label=index_label, shares=shares,
        eps_ttm=eps_ttm, eps_fy=eps_fy, eps_fwd=fwd_eps,
        eps_fwd_fy=eps_fwd_fy, rev_fwd_fy=rev_fwd_fy,
        eps_source=eps_source,
        eps_growth=eps_growth,          # blended forward EPS growth %
        eps_growth_ttm=eps_growth_ttm,  # raw TTM for display
        eps_growth_fq=eps_growth_fq,    # raw quarterly for display
        eps_growth_dec=eps_growth_dec,
        peg_growth_rate=peg_growth_rate,  # best forward growth rate for PEG calc
        fwd_eps_growth_implied=fwd_eps_growth_implied,
        rev=rev, rev_fy=rev_fy, fwd_rev=fwd_rev,
        rev_growth=rev_growth,          # blended forward revenue growth %
        rev_growth_ttm=rev_growth_ttm,  # raw TTM for display
        rev_growth_fq=rev_growth_fq,    # raw quarterly for display
        rev_growth_dec=rev_growth_dec,
        rev_source=rev_source,
        gross_margin=gross_margin, op_margin=op_margin, net_margin=net_margin,
        fcf=fcf, fcf_margin=fcf_margin,
        roe=roe, pe_ttm=pe_ttm, beta=beta, de_ratio=de_ratio,
        net_debt_v=net_debt_v,
        net_income=net_income, ebitda=ebitda,
        hi52=hi52, lo52=lo52, pos52=pos52,
        perf_1m=perf_1m, perf_3m=perf_3m, perf_6m=perf_6m,
        tv_rating=tv_rating, rsi=rsi,
        closely_held_pct=closely_held_pct,
        mom=mom, macd_hist=macd_hist, macd_line=macd_line, macd_signal=macd_signal,
        rel_vol=rel_vol, avg_vol_30d=avg_vol_30d, avg_vol_90d=avg_vol_90d,
        adx=adx, adx_plus=adx_plus, adx_minus=adx_minus,
        analyst_target=analyst_target,
        analyst_upside=analyst_upside,
        # ── New fields ────────────────────────────────────────────────
        roic=roic, roa=roa,
        fcf_growth=fcf_growth, fcf_growth_ttm=fcf_growth_ttm, fcf_growth_fq=fcf_growth_fq,
        rd_ratio=rd_ratio,
        gp_growth=gp_growth, gp_growth_ttm=gp_growth_ttm,
        ni_growth=ni_growth, ni_growth_ttm=ni_growth_ttm,
        ebitda_growth=ebitda_growth,
        current_ratio=current_ratio, quick_ratio_v=quick_ratio_v,
        days_to_earnings=days_to_earnings,
        p_fcf=p_fcf, ev_ebitda=ev_ebitda,
        money_flow=money_flow, stoch_k=stoch_k, stoch_d=stoch_d,
        williams_r=williams_r, perf_ytd=perf_ytd,
        rev_qoq=rev_qoq, eps_qoq=eps_qoq,
        below_ema13=below_ema13, below_ema50=below_ema50, below_ema200=below_ema200,
    )


# ── METHOD 1: FORWARD PEG PROJECTION ─────────────────────────────────────────
# ── METHOD 2: REVENUE GROWTH EXTRAPOLATION ────────────────────────────────────
# ── METHOD 3: EV/EBITDA BACK-INTO-EQUITY VALUE ────────────────────────────────
# ── PILLAR 1: QUALITY (0–35 pts) ────────────────────────────────────────────
# ── PILLAR 2: GROWTH MOMENTUM (0–35 pts) ─────────────────────────────────────
# ── PILLAR 3: TECHNICAL / MARKET SIGNAL (0–20 pts) ────────────────────────────
# ── PILLAR 4: VALUATION (0–10 pts) ───────────────────────────────────────────
# ── COMPOSITE GROWTH SCORE (0–100) ────────────────────────────────────────────
# ── SCORE STOCK ───────────────────────────────────────────────────────────────
# ── ACCUMULATION SIGNAL ───────────────────────────────────────────────────────
# Proxy for institutional buying pressure using TradingView technical fields.
# Five sub-signals, each contributing points to a 0-100 score:
#   1. MACD — histogram above zero and rising = accumulation pressure
#   2. Momentum — sustained positive price pressure
#   3. Volume trend — current volume vs 30/90d avg (expanding = accumulation)
#   4. ADX + Directional — strong trend with +DI > -DI = buying dominance
#   5. CCI — deviation from mean price; sustained high = institutional demand

def _blend_weights(d, peg_ok, rev_ok, ev_ok, tam_ok):
    """
    Adaptive blending weights for price target methods.
    - PEG weight reduced for unprofitable companies (eps_ttm <= 0)
    - S-Curve TAM weight increased for hypergrowth (rev_growth > 30%)
    - EV weight zeroed if no EBITDA
    Returns (w_peg, w_rev, w_ev, w_tam) normalized to sum to 1, or all None if zero.
    """
    rev_growth_dec = ((d.get("rev_growth") or 0.0) / 100.0)
    eps_ttm = d.get("eps_ttm") or 0.0
    ebitda  = d.get("ebitda") or 0.0

    w_peg = 0.60 if eps_ttm > 0 else 0.20    # reduce PEG for unprofitable
    w_rev = 0.40
    w_ev  = 0.20 if ebitda > 0 else 0.0
    w_tam = 0.25 if rev_growth_dec > 0.30 else 0.10  # more weight for hypergrowth

    # Zero out missing methods
    if not peg_ok: w_peg = 0.0
    if not rev_ok: w_rev = 0.0
    if not ev_ok:  w_ev  = 0.0
    if not tam_ok: w_tam = 0.0

    total = w_peg + w_rev + w_ev + w_tam
    if total == 0:
        return None, None, None, None
    return w_peg / total, w_rev / total, w_ev / total, w_tam / total


def score_stock(d):
    price = d["price"]
    if not price or price <= 0: return None
    if not d["shares"] or d["shares"] <= 0: return None

    # Need at least some growth data
    if d["rev_growth"] is None and d["eps_growth"] is None:
        return None

    # ── Method 1: Forward PEG targets (bear / base / bull)
    peg_bear, peg_base, peg_bull, peg_growth_used, peg_eps_used = calc_peg_targets(d)

    # ── Method 2: Revenue extrapolation target
    rev_target = calc_rev_target(d)

    # ── Method 3: EV/EBITDA back-into-equity target
    ev_target = calc_ev_target(d)

    # ── Method 4: S-Curve TAM logistic growth target
    scurve_tam_target = None
    try:
        _rev_growth_dec = (d.get("rev_growth") or d.get("eps_growth") or 10.0) / 100.0
        _dsc = {
            "revenue":     d.get("rev"),
            "est_growth":  max(0.02, min(0.80, _rev_growth_dec)),
            "gross_margin": d.get("gross_margin") or 50.0,
            "net_income":  d.get("net_income"),
            "total_debt":  max(0.0, d.get("net_debt_v") or 0.0),
            "cash":        0.0,
            "shares":      d["shares"],
            "price":       d["price"],
            "market_cap":  d["market_cap"],
            "beta":        d.get("beta", 1.0),
            "wacc_override": None,
            "wacc_raw": {"beta_yf": None, "interest_expense": None,
                         "income_tax_expense": None, "pretax_income": None,
                         "total_debt_yf": None},
            "ev_approx":   d["market_cap"] + max(0.0, d.get("net_debt_v") or 0.0),
        }
        _r_sc = run_scurve_tam(_dsc)
        if _r_sc and _r_sc.get("fair_value") and _r_sc["fair_value"] > 0:
            scurve_tam_target = _r_sc["fair_value"]
    except Exception:
        scurve_tam_target = None

    # Need at least one method to produce a result
    if peg_base is None and rev_target is None and ev_target is None and scurve_tam_target is None:
        return None

    # ── Adaptive composite 12-month price target
    peg_ok = peg_base is not None
    rev_ok = rev_target is not None
    ev_ok  = ev_target is not None
    tam_ok = scurve_tam_target is not None
    w_peg, w_rev, w_ev, w_tam = _blend_weights(d, peg_ok, rev_ok, ev_ok, tam_ok)

    if w_peg is None:
        # No valid weights — fall back to simple average
        valid_parts = [v for v in [peg_base, rev_target, ev_target, scurve_tam_target] if v is not None]
        composite = sum(valid_parts) / len(valid_parts) if valid_parts else None
        if composite is None:
            return None
        w_peg_eff = 1.0 / len(valid_parts) if peg_ok else 0.0
        w_rev_eff = 1.0 / len(valid_parts) if rev_ok else 0.0
        w_ev_eff  = 1.0 / len(valid_parts) if ev_ok  else 0.0
        w_tam_eff = 1.0 / len(valid_parts) if tam_ok else 0.0
    else:
        parts_w = []
        if peg_ok:  parts_w.append((peg_base,           w_peg))
        if rev_ok:  parts_w.append((rev_target,          w_rev))
        if ev_ok:   parts_w.append((ev_target,           w_ev))
        if tam_ok:  parts_w.append((scurve_tam_target,   w_tam))
        composite = sum(p * w for p, w in parts_w)
        w_peg_eff, w_rev_eff, w_ev_eff, w_tam_eff = w_peg, w_rev, w_ev, w_tam

    # Bear / bull composite targets
    bear_parts, bull_parts = [], []
    if peg_ok and w_peg_eff > 0:
        bear_parts.append((peg_bear if peg_bear else peg_base * 0.80, w_peg_eff))
        bull_parts.append((peg_bull if peg_bull else peg_base * 1.20, w_peg_eff))
    if rev_ok and w_rev_eff > 0:
        bear_parts.append((rev_target * 0.80, w_rev_eff))
        bull_parts.append((rev_target * 1.20, w_rev_eff))
    if ev_ok and w_ev_eff > 0:
        bear_parts.append((ev_target * 0.80, w_ev_eff))
        bull_parts.append((ev_target * 1.20, w_ev_eff))
    if tam_ok and w_tam_eff > 0:
        bear_parts.append((scurve_tam_target * 0.80, w_tam_eff))
        bull_parts.append((scurve_tam_target * 1.20, w_tam_eff))

    bear_composite = sum(p * w for p, w in bear_parts) / sum(w for _, w in bear_parts) if bear_parts else composite * 0.80
    bull_composite = sum(p * w for p, w in bull_parts) / sum(w for _, w in bull_parts) if bull_parts else composite * 1.20

    # ── Sanity cap — prevent growth-multiple inflation or bad input data from
    # producing targets that are disconnected from current price.
    # A 12-month bull target above 3× current price has no historical precedent
    # except in binary biotech events; cap here keeps the column interpretable.
    _MAX_BASE_MULT = 2.5
    _MAX_BULL_MULT = 3.0
    _MAX_BEAR_MULT = 2.0
    if price > 0:
        composite      = min(composite,      price * _MAX_BASE_MULT)
        bear_composite = min(bear_composite, price * _MAX_BEAR_MULT)
        bull_composite = min(bull_composite, price * _MAX_BULL_MULT)

    upside_pct = (composite - price) / price * 100

    # ── Sentiment (derived from TV data already in d)
    sentiment    = derive_sentiment(d)
    accumulation = derive_accumulation(d)

    # ── Flags
    flags = get_growth_risk_flags(d)

    # ── PEG ratio (forward — uses forward growth rate, not trailing)
    # This is the number Wall Street uses: current P/E ÷ forward EPS growth rate
    peg_current = None
    growth_for_peg = d.get("peg_growth_rate") or d["eps_growth"]
    if d["pe_ttm"] and d["pe_ttm"] > 0 and growth_for_peg and growth_for_peg > 0:
        peg_current = round(d["pe_ttm"] / growth_for_peg, 2)

    # ── Tier
    tier_name = TIERS[-1][0]
    for name, lo, hi, _ in TIERS:
        if lo <= upside_pct < hi:
            tier_name = name; break

    # ── Growth Score — compute sub-scores once and reuse for display
    # (calc_growth_score internally calls these four functions, so calling both
    # would double-compute every pillar; inline the formula instead)
    _q  = calc_quality_score(d)
    _gm = calc_growth_momentum_score(d)
    _t  = calc_momentum_score(d)
    _v  = calc_valuation_score(d, composite)
    growth_score = max(0.0, min(100.0, round(_q + _gm + _t + _v - len(flags) * 4, 1)))

    # Momentum score for display (same as _t, no need to call again)
    momentum_score = round(_t, 1)

    return dict(
        ticker=d["ticker"], sector=d["sector"], price=price,
        # Targets
        target_bear=bear_composite, target_base=composite, target_bull=bull_composite,
        peg_target=peg_base, rev_target=rev_target, ev_target=ev_target,
        scurve_tam_target=scurve_tam_target,
        upside_pct=upside_pct,
        # PEG method details
        peg_bear=peg_bear, peg_base=peg_base, peg_bull=peg_bull,
        peg_current=peg_current,
        peg_growth_used=peg_growth_used,      # growth rate fed into PEG calc
        peg_eps_used=peg_eps_used,            # forward EPS used in PEG calc
        eps_source=d.get("eps_source"),       # "analyst_fq×4" / "ttm×growth" / etc.
        # Growth metrics — forward-blended
        rev_growth=d["rev_growth"], eps_growth=d["eps_growth"],
        rev_growth_ttm=d.get("rev_growth_ttm"), rev_growth_fq=d.get("rev_growth_fq"),
        eps_growth_ttm=d.get("eps_growth_ttm"), eps_growth_fq=d.get("eps_growth_fq"),
        fwd_eps_growth_implied=d.get("fwd_eps_growth_implied"),
        eps_fwd=d["eps_fwd"], eps_ttm=d["eps_ttm"], eps_fy=d.get("eps_fy"),
        rev=d["rev"], fwd_rev=d["fwd_rev"], rev_fy=d.get("rev_fy"),
        rev_fwd_fy=d.get("rev_fwd_fy"), eps_fwd_fy=d.get("eps_fwd_fy"),
        # Quality metrics
        gross_margin=d["gross_margin"], op_margin=d["op_margin"],
        net_margin=d["net_margin"], fcf_margin=d["fcf_margin"],
        roe=d["roe"], pe_ttm=d["pe_ttm"],
        # Momentum
        perf_1m=d["perf_1m"], perf_3m=d["perf_3m"],
        perf_6m=d["perf_6m"], pos52=d["pos52"],
        tv_rating=d["tv_rating"], rsi=d.get("rsi"), momentum_score=momentum_score,
        macd_hist=d.get("macd_hist"), rel_vol=d.get("rel_vol"), adx=d.get("adx"),
        closely_held_pct=d.get("closely_held_pct"),
        # New fields passed through for display
        roic=d.get("roic"), roa=d.get("roa"),
        fcf_growth=d.get("fcf_growth"), gp_growth=d.get("gp_growth"),
        ni_growth=d.get("ni_growth"), ebitda_growth=d.get("ebitda_growth"),
        rd_ratio=d.get("rd_ratio"),
        current_ratio=d.get("current_ratio"), quick_ratio_v=d.get("quick_ratio_v"),
        days_to_earnings=d.get("days_to_earnings"),
        p_fcf=d.get("p_fcf"), ev_ebitda=d.get("ev_ebitda"),
        money_flow=d.get("money_flow"), stoch_k=d.get("stoch_k"),
        williams_r=d.get("williams_r"), perf_ytd=d.get("perf_ytd"),
        rev_qoq=d.get("rev_qoq"), eps_qoq=d.get("eps_qoq"),
        # Meta
        market_cap=d["market_cap"], beta=d["beta"],
        shares=d["shares"], fcf=d.get("fcf"),
        net_debt_v=d.get("net_debt_v"), ebitda=d.get("ebitda"),
        tier=tier_name, growth_score=growth_score, flags=flags,
        score_quality=round(_q,1), score_growth=round(_gm,1),
        score_tech=round(_t,1), score_val=round(_v,1),
        # Analyst & sentiment
        analyst_target=d.get("analyst_target"),
        analyst_upside=d.get("analyst_upside"),
        sentiment=sentiment,
        accumulation=accumulation,
        below_ema13=d.get("below_ema13"), below_ema50=d.get("below_ema50"), below_ema200=d.get("below_ema200"),
    )


# ── CONSENSUS TOP-3 HELPERS ───────────────────────────────────────────────────

_GS_BM = {"pe": 22.0, "pfcf": 22.0, "ev_ebitda": 14.0}

def _build_vm_dict(r, yf_data=None):
    """
    Map score_stock result dict → valuation_models-compatible dict.
    yf_data (dict from _fetch_yf_data_gs) is preferred over TV values for
    FCF, EBITDA, revenue, total_debt, cash, and gross_margin when present,
    eliminating data-source divergence vs. valuationMaster.
    """
    yf = yf_data or {}

    # ── Revenue: yfinance annual > TV ────────────────────────────────────────
    rev = yf.get("revenue") or r.get("rev") or 0.0

    # ── FCF: yfinance annual > TV ────────────────────────────────────────────
    fcf = yf.get("fcf") or r.get("fcf")

    # ── Debt / cash: yfinance splits them; TV only provides net_debt ─────────
    yf_debt = yf.get("total_debt")
    yf_cash = yf.get("cash")
    if yf_debt is not None:
        total_debt = max(0.0, yf_debt or 0.0)
        cash       = max(0.0, yf_cash or 0.0)
    else:
        # Fall back to TV net_debt (cannot separate)
        total_debt = max(0.0, r.get("net_debt_v") or 0.0)
        cash       = 0.0
    net_debt = max(0.0, total_debt - cash)

    # ── EBITDA: yfinance > TV > derive from revenue + margins ────────────────
    op_margin = (r.get("op_margin") or 0.0) / 100.0
    ebitda = (yf.get("ebitda")
              or r.get("ebitda")
              or (rev * op_margin + rev * 0.05 if rev and op_margin else None))

    # ── Gross margin: yfinance (fraction→% already converted) > TV ───────────
    gross_margin = yf.get("gross_margin") or r.get("gross_margin")

    rev_growth = (r.get("rev_growth") or 0.0) / 100.0
    est_growth = max(0.02, min(0.80, rev_growth))
    mktcap     = r.get("market_cap") or 0.0

    shares = r["shares"]
    # Derived per-share / margin fields needed by _rebuild_d_from_snapshot (PIT backtest)
    fcf_per_share = (fcf / shares) if (fcf and shares and shares > 0) else None
    fcf_margin    = (fcf / rev)    if (fcf and rev  and rev   > 0) else None

    return {
        "price":         r["price"],
        "shares":        shares,
        "market_cap":    mktcap,
        "revenue":       rev if rev else None,
        "fcf":           fcf,
        "fcf_per_share": fcf_per_share,
        "fcf_margin":    fcf_margin,
        "eps":           r.get("eps_ttm"),
        "fwd_eps":       yf.get("fwd_eps") or r.get("eps_fwd"),
        "fwd_rev":       yf.get("fwd_rev") or r.get("fwd_rev"),
        "ebitda":        ebitda,
        "ebitda_method": ("yfinance/EDGAR" if yf.get("ebitda") else
                          "TV screener"    if r.get("ebitda") else
                          "op_margin proxy"),
        "net_income":    r.get("net_income"),
        "total_debt":    total_debt,
        "cash":          cash,
        "beta":          r.get("beta", 1.0),
        "op_margin":     r.get("op_margin"),
        "gross_margin":  gross_margin,
        "est_growth":    est_growth,
        "ev_approx":     mktcap + net_debt,
        "analyst_target": r.get("analyst_target"),
        "ticker":        r.get("ticker", ""),
        "wacc_raw":      {"beta_yf": None, "interest_expense": None,
                          "income_tax_expense": None, "pretax_income": None,
                          "total_debt_yf": None},
        "wacc_override": None,
        "ext":           {},   # empty — skips RIM/ROIC/DDM/NCAV in _run_methods_for_snapshot
    }


def _fetch_yf_data_gs(ticker):
    """
    Fetch key fundamentals from yfinance (no price history — not needed for
    consensus ranking).  Returns dict with keys: fcf, ebitda, revenue,
    total_debt, cash, gross_margin.  All values are None when unavailable.

    Retries once automatically if Yahoo Finance returns 401 Invalid Crumb.
    Detection uses the _YF401LogHandler installed on the 'yfinance' logger,
    because yfinance sets hide_exceptions=True by default — it catches the
    HTTPError internally, logs it, and returns None without re-raising.
    """
    _EMPTY = lambda: {"fcf": None, "ebitda": None,
                      "revenue": None, "fwd_rev": None, "total_debt": None,
                      "cash": None, "gross_margin": None, "fwd_eps": None}

    # ── Cache hit — skip all HTTP calls ──────────────────────────────────────
    # No lock needed: dict.get() is GIL-protected and safe from any thread
    _cached = _yf_cache.get(ticker)
    if _cached is not None:
        return {k: v for k, v in _cached.items() if k != "_ts"}

    for _attempt in range(2):
        out = _EMPTY()
        _yf_401_tls.detected = False   # reset per-thread flag before this attempt

        try:
            tk = yf.Ticker(ticker)

            # ── Gross margin + forward EPS from info (single HTTP call) ──────────
            try:
                info = tk.info or {}
                gm = info.get("grossMargins")
                if gm is not None:
                    out["gross_margin"] = round(gm * 100, 2)  # fraction → percent
                fwd_eps = info.get("forwardEps")
                if fwd_eps is not None and fwd_eps > 0:
                    out["fwd_eps"] = float(fwd_eps)
            except Exception:
                pass

            # ── Analyst NTM revenue estimate ──────────────────────────────────────
            # Gives EV/NTM Rev the same analyst-consensus forward revenue that
            # valuationMaster uses — critical for cyclical-recovery stocks (MU etc.)
            try:
                rev_est = tk.revenue_estimate
                if rev_est is not None and not rev_est.empty:
                    for row_label in ["+1y", "0y"]:
                        if row_label in rev_est.index:
                            avg = rev_est.loc[row_label, "avg"]
                            if avg is not None and float(avg) > 0:
                                out["fwd_rev"] = float(avg)
                                break
            except Exception:
                pass

            # ── FCF from annual cashflow statement ────────────────────────────────
            try:
                cf = tk.cashflow
                if cf is not None and not cf.empty:
                    for name in ["Free Cash Flow", "FreeCashFlow"]:
                        if name in cf.index:
                            v = cf.loc[name].dropna()
                            if len(v):
                                out["fcf"] = float(v.iloc[0])
                            break
                    if out["fcf"] is None:
                        # Derive: Operating CF + CapEx (CapEx stored as negative)
                        ocf = capex = None
                        for n in ["Operating Cash Flow",
                                  "Cash Flow From Continuing Operating Activities"]:
                            if n in cf.index:
                                v = cf.loc[n].dropna()
                                ocf = float(v.iloc[0]) if len(v) else None
                                break
                        for n in ["Capital Expenditure",
                                  "Purchase Of Property Plant And Equipment"]:
                            if n in cf.index:
                                v = cf.loc[n].dropna()
                                capex = float(v.iloc[0]) if len(v) else None
                                break
                        if ocf is not None and capex is not None:
                            out["fcf"] = ocf + capex   # capex is negative → FCF
            except Exception:
                pass

            # ── Revenue + EBITDA from annual income statement ─────────────────────
            try:
                inc = tk.income_stmt
                if inc is not None and not inc.empty:
                    for n in ["Total Revenue", "Revenue", "Net Revenue"]:
                        if n in inc.index:
                            v = inc.loc[n].dropna()
                            out["revenue"] = float(v.iloc[0]) if len(v) else None
                            break
                    for n in ["EBITDA", "Normalized EBITDA"]:
                        if n in inc.index:
                            v = inc.loc[n].dropna()
                            out["ebitda"] = float(v.iloc[0]) if len(v) else None
                            break
            except Exception:
                pass

            # ── Debt + cash from balance sheet ────────────────────────────────────
            try:
                bs = tk.balance_sheet
                if bs is not None and not bs.empty:
                    for n in ["Total Debt",
                              "Long Term Debt And Capital Lease Obligation",
                              "Long Term Debt"]:
                        if n in bs.index:
                            v = bs.loc[n].dropna()
                            out["total_debt"] = float(v.iloc[0]) if len(v) else None
                            break
                    for n in ["Cash And Cash Equivalents",
                              "Cash Cash Equivalents And Short Term Investments"]:
                        if n in bs.index:
                            v = bs.loc[n].dropna()
                            out["cash"] = float(v.iloc[0]) if len(v) else None
                            break
            except Exception:
                pass

        except Exception:
            pass

        # ── Retry once on 401 after refreshing the session ───────────────────
        # _yf_401_tls.detected is set by _YF401LogHandler when yfinance logs
        # an "HTTP Error 401 / Invalid Crumb" message (hide_exceptions=True path).
        if getattr(_yf_401_tls, "detected", False) and _attempt == 0:
            _yf_refresh_session()   # thread-safe, debounced — only one thread refreshes
            continue                # second attempt with fresh cookie
        break                       # success (or non-401 error) — exit loop

    # ── Write to cache if we got at least one useful value ────────────────────
    if any(out.get(k) is not None for k in ("fcf", "revenue", "gross_margin")):
        with _yf_cache_lock:
            _yf_cache[ticker] = {**out, "_ts": _time.time()}

    return out


def _run_gs_valuations(vm):
    """Run all computable valuation methods. Returns {method: fair_value}."""
    fvs = {}

    def _s(name, r):
        v = r.get("fair_value") if isinstance(r, dict) else r
        if v and v > 0:
            fvs[name] = v

    # ── DCF (simplified 2-stage, 5yr) ──────────────────────────────────────
    try:
        fcf    = vm.get("fcf")
        rev    = vm.get("revenue")
        shares = vm.get("shares")
        nd     = (vm.get("total_debt") or 0) - (vm.get("cash") or 0)
        g      = vm.get("est_growth") or 0.08
        beta   = vm.get("beta") or 1.0
        _RF, _ERP = 0.043, 0.055
        wacc   = max(0.07, min(0.18, _RF + beta * _ERP))
        cf     = fcf if (fcf and fcf > 0) else (rev * 0.10 if rev else None)
        if cf and cf > 0 and shares and shares > 0:
            pvs = [cf * (1 + g) ** yr / (1 + wacc) ** yr for yr in range(1, 6)]
            tv  = cf * (1 + g) ** 5 * 1.025 / (wacc - 0.025)
            ev  = sum(pvs) + tv / (1 + wacc) ** 5 - nd
            if ev > 0:
                _s("DCF", round(ev / shares * (1 - MARGIN_OF_SAFETY), 2))
    except Exception: pass

    # ── P/FCF ───────────────────────────────────────────────────────────────
    try:
        fcf    = vm.get("fcf")
        shares = vm.get("shares")
        g      = vm.get("est_growth") or 0.05
        if fcf and fcf > 0 and shares and shares > 0:
            fcf_ps = fcf / shares
            mults  = growth_adjusted_multiples(g)
            _s("P/FCF", round((fcf_ps * mults["target_pfcf"] +
                               fcf_ps * _GS_BM["pfcf"] +
                               fcf_ps * mults["conserv_pfcf"]) / 3, 2))
    except Exception: pass

    # ── EV/EBITDA ────────────────────────────────────────────────────────────
    try:
        ebitda = vm.get("ebitda")
        shares = vm.get("shares")
        nd     = (vm.get("total_debt") or 0) - (vm.get("cash") or 0)
        g      = vm.get("est_growth") or 0.05
        if ebitda and ebitda > 0 and shares and shares > 0:
            mults = growth_adjusted_multiples(g)
            def _ip(m): return (ebitda * m - nd) / shares
            _s("EV/EBITDA", round((_ip(mults["target_eveb"]) +
                                   _ip(_GS_BM["ev_ebitda"]) +
                                   _ip(mults["conserv_eveb"])) / 3, 2))
    except Exception: pass

    # ── NTM P/E ─────────────────────────────────────────────────────────────
    try:
        fwd_eps = vm.get("fwd_eps")
        eps     = vm.get("eps")
        g       = vm.get("est_growth") or 0.05
        ep      = fwd_eps if (fwd_eps and fwd_eps > 0) else eps
        if ep and ep > 0:
            mults = growth_adjusted_multiples(g)
            _s("NTM P/E", round((ep * mults["target_pe"] +
                                 ep * _GS_BM["pe"] +
                                 ep * mults["conserv_pe"]) / 3, 2))
    except Exception: pass

    try: _s("Fwd PEG",    run_forward_peg(vm))
    except Exception: pass
    try: _s("EV/NTM Rev", run_ev_ntm_revenue(vm))
    except Exception: pass
    try:
        r = run_pie(vm)
        if r and r.get("fair_value") and r["fair_value"] > 0:
            fvs["PIE"] = r["fair_value"]
    except Exception: pass
    try:
        r = run_fcf_yield(vm)
        if r and r.get("fair_value") and r["fair_value"] > 0:
            fvs["FCF Yield"] = r["fair_value"]
    except Exception: pass
    try: _s("Graham",     run_graham_number(vm))
    except Exception: pass

    # Analyst consensus target — professional anchor to prevent outlier inflation
    try:
        at = vm.get("analyst_target")
        if at and float(at) > 0:
            fvs["Analyst Target"] = float(at)
    except Exception: pass

    return fvs


def _consensus_top3_gs(fvs, price=None):
    """
    Consensus-based method selection: pick the 3 methods whose fair values
    are closest to the ensemble median.  Methods that agree with the majority
    are more likely to be accurate than methods that happen to sit near the
    current price (which is what static MAPE rewards).

    This aligns with valuationMaster's backtest winner pattern: EV/NTM Rev,
    RIM, Rule of 40 all cluster near the median; outlier methods (Fwd PEG,
    S-Curve TAM) are excluded unless the ensemble is pulled that way.

    price (optional): current stock price.  When supplied, fair values outside
    0.25×–4× current price are trimmed before computing the median, preventing
    growth-adjusted multiples from anchoring the consensus too high.

    Returns (mean_fv, [method_names]).
    """
    if not fvs:
        return None, []
    # Trim extreme outliers relative to current price
    if price and price > 0:
        fvs = {m: v for m, v in fvs.items()
               if 0.25 * price <= v <= 4.0 * price}
    if not fvs:
        return None, []
    vals = sorted(fvs.values())
    median_fv = vals[len(vals) // 2]
    ranked = sorted(fvs.items(), key=lambda x: abs(x[1] - median_fv))[:3]
    if not ranked:
        return None, []
    top3_fvs = [v for _, v in ranked]
    return sum(top3_fvs) / len(top3_fvs), [m for m, _ in ranked]


def _run_pit_backtest_gs(ticker, vm_dict, days=500):
    """
    Point-in-time backtest for a single ticker using the exact same engine as
    valuationMaster:
      1. Stooq price history (days trading days)
      2. yfinance quarterly TTM fundamentals
      3. SEC EDGAR annual XBRL fundamentals
      4. For each trading day: pick the right snapshot (75-day filing lag),
         rebuild d-dict, run all 20 valuation methods
      5. Rank methods by MAPE (same formula as valuationMaster)
      6. Return top-3 methods + their CURRENT fair values

    Returns (mean_current_fv, [method_names]) or (None, []) on failure.
    Requires _BT_ENGINE_AVAIL = True (valuationMaster importable).
    """
    # ── 1. Price history ─────────────────────────────────────────────────────
    # Primary: yfinance — covers all US-listed tickers including RUT small-caps.
    #   Stooq probes multiple symbol variants sequentially and each variant can
    #   block for 15-20 s when the ticker isn't indexed there, consuming the
    #   entire per-ticker budget before returning.  yfinance is faster and has
    #   better coverage for the indices growthScreener targets.
    # Fallback: Stooq — used only when yfinance returns insufficient history.
    prices = []
    if _YF_AVAIL:
        try:
            cal_days = math.ceil(days * 1.6) + 60
            hist = yf.Ticker(ticker).history(period=f"{cal_days}d")
            if hist is not None and not hist.empty:
                yf_prices = []
                for ts, row in hist.iterrows():
                    try:
                        cl = float(row["Close"])
                        dt = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                        if cl > 0 and dt:
                            yf_prices.append({"date": dt, "close": cl})
                    except (ValueError, TypeError):
                        continue
                yf_prices.sort(key=lambda x: x["date"])
                yf_prices = yf_prices[-days:]
                if len(yf_prices) >= 10:
                    prices = yf_prices
                    print(f"  [yfinance] Got {len(prices)} price points for '{ticker}'")
        except Exception as e:
            print(f"  [PIT:{ticker}] yfinance price error: {e}")

    if len(prices) < 10:
        try:
            prices = _bt_fetch_prices(ticker, days) or []
        except Exception as e:
            print(f"  [PIT:{ticker}] Stooq fallback error: {e}")

    if len(prices) < 10:
        print(f"  [PIT:{ticker}] insufficient price data ({len(prices)} points) — skipping")
        return None, [], []

    # ── 2. Fundamentals ──────────────────────────────────────────────────────
    try:
        q_snaps = _bt_fetch_quarterly(ticker)
    except Exception as e:
        print(f"  [PIT:{ticker}] yfinance quarterly error: {e}")
        q_snaps = []
    try:
        a_snaps = _bt_fetch_annual(ticker)
    except Exception as e:
        print(f"  [PIT:{ticker}] EDGAR annual error: {e}")
        a_snaps = []

    if not q_snaps and not a_snaps:
        print(f"  [PIT:{ticker}] no fundamental snapshots available — skipping")
        return None, [], []

    # Merge: quarterly preferred on same date (more accurate TTM)
    snap_dict = {}
    for s in a_snaps:
        snap_dict[s["date"]] = s
    for s in q_snaps:
        snap_dict[s["date"]] = s
    snapshots = sorted(snap_dict.values(), key=lambda x: x["date"])

    bm = {"pe": 22.0, "pfcf": 22.0, "ev_ebitda": 14.0}

    # ── 3. Build time series ─────────────────────────────────────────────────
    fv_cache    = {}
    time_series = []
    for day in prices:
        snap = _bt_pick_snap(snapshots, day["date"])
        if snap is None:
            continue
        sd = snap["date"]
        if sd not in fv_cache:
            try:
                d_h = _bt_rebuild_d(vm_dict, snap, day["close"], bm)
                fv_cache[sd] = _bt_run_methods(d_h, bm)
            except Exception as e:
                print(f"  [PIT:{ticker}] snapshot {sd} error: {e}")
                fv_cache[sd] = {}
        time_series.append({"date": day["date"], "price": day["close"],
                             "fv": fv_cache[sd]})

    if not time_series:
        print(f"  [PIT:{ticker}] no trading days overlapped with filed snapshots — skipping")
        return None, [], []

    print(f"  [PIT:{ticker}] {len(time_series)} days × {len(fv_cache)} snapshots "
          f"(q={len(q_snaps)}, a={len(a_snaps)})")

    # ── 4. MAPE per method ────────────────────────────────────────────────────
    all_methods = set(m for row in time_series for m in row["fv"])
    method_mapes = {}
    for name in all_methods:
        errors = [
            abs(row["fv"][name] - row["price"]) / row["price"] * 100
            for row in time_series
            if row["fv"].get(name) and row["fv"][name] > 0
        ]
        if len(errors) >= 5:
            method_mapes[name] = sum(errors) / len(errors)

    if not method_mapes:
        print(f"  [PIT:{ticker}] no methods produced enough valid days — skipping")
        return None, [], []

    # ── 5. Top-3 by lowest MAPE ───────────────────────────────────────────────
    top3 = sorted(method_mapes.items(), key=lambda x: x[1])[:3]
    top3_names = [m for m, _ in top3]

    # ── 6. Current fair values for top-3 methods ──────────────────────────────
    # Rebuild from the most recent snapshot so est_growth uses the FCF-margin
    # proxy (max 15%) — the same calculation used for every historical day in
    # the MAPE computation.  Without this, TV rev_growth (e.g. 120% for NVDA)
    # feeds est_growth = 0.80 into Three-Stage DCF / S-Curve TAM and inflates
    # the current FV to 3-5× the market price, distorting the column average.
    # After rebuilding we restore fwd_eps / fwd_rev from vm_dict so that
    # EV/NTM Rev and Fwd PEG still use today's analyst consensus estimates.
    try:
        latest_snap = snapshots[-1] if snapshots else None
        if latest_snap:
            d_current = _bt_rebuild_d(vm_dict, latest_snap, vm_dict["price"], bm)
            # Restore analyst forward estimates that _rebuild_d zeros out
            d_current["fwd_eps"] = vm_dict.get("fwd_eps")
            d_current["fwd_rev"] = vm_dict.get("fwd_rev")
        else:
            # Fallback: manually cap est_growth with FCF-margin proxy
            d_current = dict(vm_dict)
            fcm = vm_dict.get("fcf_margin")
            if fcm and fcm > 0.40:    d_current["est_growth"] = 0.15
            elif fcm and fcm > 0.20:  d_current["est_growth"] = 0.12
            elif fcm and fcm > 0.10:  d_current["est_growth"] = 0.09
            else:                     d_current["est_growth"] = 0.06
        current_fvs = _bt_run_methods(d_current, bm)
    except Exception as e:
        print(f"  [PIT:{ticker}] current FV computation error: {e}")
        return None, [], []

    top3_vals = [current_fvs[m] for m in top3_names
                 if current_fvs.get(m) and current_fvs[m] > 0]
    if not top3_vals:
        print(f"  [PIT:{ticker}] top-3 methods produced no current FV — skipping")
        return None, [], []

    g_used = d_current.get("est_growth", "?")
    indiv  = {m: round(current_fvs[m], 2) for m in top3_names if current_fvs.get(m)}
    print(f"  [PIT:{ticker}] top-3: {top3_names} g={g_used:.2%} "
          f"fvs={indiv} → ${sum(top3_vals)/len(top3_vals):,.2f}")
    # Return mean, method names, AND individual values so the caller can derive
    # bear/base/bull from min/mean/max of the historically-accurate methods.
    return sum(top3_vals) / len(top3_vals), top3_names[:len(top3_vals)], top3_vals


# ── FORMAT HELPERS ────────────────────────────────────────────────────────────
def fp(n):
    if n is None: return "—"
    return "$" + format(n, ",.2f")
def fb(n):
    if not n: return "—"
    if n >= 1e12: return "$" + format(n/1e12, ".2f") + "T"
    return "$" + format(n/1e9, ".1f") + "B"
def fpc(n, plus=True):
    if n is None: return "—"
    return ("+" if n >= 0 and plus else "") + format(n, ".1f") + "%"
def fpct(n):
    if n is None: return "—"
    return format(n, ".1f") + "%"
def fx(n):
    return format(n, ".2f") + "x" if n is not None else "—"
def fsc(n):
    if n is None: return "—"
    return format(n, ".0f")
def frating(n):
    if n is None: return "—"
    if n >= 0.5:  return "Strong Buy"
    if n >= 0.1:  return "Buy"
    if n >= -0.1: return "Neutral"
    if n >= -0.5: return "Sell"
    return "Strong Sell"


# ── CSV EXPORT ────────────────────────────────────────────────────────────────
def write_csv(results, filename):
    headers = [
        "Growth Score", "Ticker", "Sector", "Tier",
        "Price", "Target (Bear)", "Target (Base)", "Target (Bull)",
        "Upside %", "Fwd PEG", "PEG Target", "Rev Target", "S-Curve TAM Target",
        "Fwd EPS", "EPS Source", "Fwd EPS Growth %", "EPS Growth QoQ %",
        "Rev Growth % (Fwd)", "Rev Growth QoQ %",
        "P/E (TTM)", "Gross Margin %", "Op Margin %", "Net Margin %", "FCF Margin %",
        "ROE %", "Perf 1M %", "Perf 3M %", "Perf 6M %",
        "52wk Position %", "TV Rating", "Momentum Score",
        "Market Cap", "Beta", "Flags",
    ]
    def pn(v, dec=2):
        return round(v, dec) if v is not None else ""

    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in sorted(results, key=lambda x: -x["growth_score"]):
            w.writerow([
                pn(r["growth_score"], 1), r["ticker"], r["sector"], r["tier"],
                pn(r["price"]), pn(r["target_bear"]), pn(r["target_base"]), pn(r["target_bull"]),
                pn(r["upside_pct"], 1), pn(r["peg_current"]), pn(r["peg_target"]), pn(r["rev_target"]), pn(r.get("scurve_tam_target")),
                pn(r["peg_eps_used"]), r.get("eps_source", ""),
                pn(r.get("peg_growth_used"), 1), pn(r.get("eps_growth_fq"), 1),
                pn(r["rev_growth"], 1), pn(r.get("rev_growth_fq"), 1),
                pn(r["pe_ttm"], 1),
                pn(r["gross_margin"], 1), pn(r["op_margin"], 1),
                pn(r["net_margin"], 1), pn(r["fcf_margin"], 1),
                pn(r["roe"], 1),
                pn(r["perf_1m"], 1), pn(r["perf_3m"], 1), pn(r["perf_6m"], 1),
                pn(r["pos52"] * 100 if r["pos52"] is not None else None, 1),
                frating(r["tv_rating"]), pn(r["momentum_score"], 1),
                r["market_cap"], pn(r["beta"]),
                " | ".join(r["flags"]) if r["flags"] else "",
            ])
    print(f"  CSV saved: {filename}")


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');
:root {
  --bg:    #080b10;
  --s1:    #0c1018;
  --s2:    #111720;
  --bd:    #1c2333;
  --tx:    #c8d4e8;
  --mu:    #3d5070;
  --gr:    #00e5a0;
  --gr2:   #00c896;
  --yw:    #f5c842;
  --rd:    #ff4d6d;
  --bl:    #3b82f6;
  --pu:    #a78bfa;
  --or:    #fb923c;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body { background: var(--bg); color: var(--tx); font-family: 'DM Sans', sans-serif; font-size: 13px; }

/* ── HERO ── */
.hero {
  position: relative; overflow: hidden;
  padding: 80px 72px 64px;
  background: linear-gradient(135deg, #080b10 0%, #0a1020 60%, #060910 100%);
  border-bottom: 1px solid var(--bd);
}
.hero::before {
  content: ''; position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 60% 50% at 80% 20%, rgba(0,229,160,.07) 0%, transparent 70%),
    radial-gradient(ellipse 40% 60% at 10% 80%, rgba(59,130,246,.05) 0%, transparent 70%);
  pointer-events: none;
}
.hero-tag {
  font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 4px;
  text-transform: uppercase; color: var(--gr); margin-bottom: 18px;
  display: flex; align-items: center; gap: 10px;
}
.hero-tag::before {
  content: ''; display: inline-block; width: 24px; height: 1px; background: var(--gr);
}
.hero-title {
  font-family: 'Syne', sans-serif; font-size: clamp(52px, 8vw, 108px);
  font-weight: 800; line-height: .9; letter-spacing: -2px;
  color: #fff; margin-bottom: 28px;
}
.hero-title em { color: var(--gr); font-style: normal; }
.hero-sub {
  font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 2px;
  text-transform: uppercase; color: var(--mu); margin-bottom: 48px;
}
.stat-row { display: flex; gap: 40px; flex-wrap: wrap; }
.stat { display: flex; flex-direction: column; gap: 4px; }
.stat-l { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--mu); }
.stat-v { font-family: 'Syne', sans-serif; font-size: 44px; font-weight: 800; line-height: 1; }
.v-gr { color: var(--gr); } .v-yw { color: var(--yw); } .v-rd { color: var(--rd); } .v-mu { color: var(--mu); }

/* ── TOP PICKS BANNER ── */
.top-picks { padding: 20px 72px; background: var(--s1); border-bottom: 1px solid var(--bd); }
.tp-label { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--mu); margin-bottom: 12px; }
.tp-cards { display: flex; gap: 10px; flex-wrap: wrap; }
.tp-card {
  background: var(--s2); border: 1px solid var(--bd); border-radius: 8px;
  padding: 10px 16px; display: flex; flex-direction: column; gap: 4px;
  min-width: 130px; transition: border-color .2s;
}
.tp-card:hover { border-color: var(--gr); }
.tp-ticker { font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500; color: var(--bl); }
.tp-score { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; color: var(--gr); }
.tp-upside { font-family: 'DM Mono', monospace; font-size: 10px; color: var(--mu); }

/* ── METHODOLOGY BAR ── */
.method-bar {
  padding: 12px 72px; background: var(--s2); border-bottom: 1px solid var(--bd);
  display: flex; gap: 28px; flex-wrap: wrap; align-items: center;
  font-family: 'DM Mono', monospace; font-size: 10px; color: var(--tx);
}
.mb-label { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--mu); }
.mb-item b { color: var(--yw); }
.mb-note { color: var(--gr); }

/* ── LEGEND ── */
.legend {
  padding: 14px 72px; background: var(--bg); border-bottom: 1px solid var(--bd);
  display: flex; gap: 20px; flex-wrap: wrap; align-items: center;
  font-family: 'DM Mono', monospace; font-size: 10px;
}
.lg-label { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--mu); }
.lg-item { display: flex; align-items: center; gap: 6px; }
.lg-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

/* ── NAV ── */
.nav {
  position: sticky; top: 0; z-index: 200;
  background: rgba(8,11,16,.95); backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--bd);
  padding: 12px 72px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}
.nl { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--mu); margin-right: 4px; }
.npill {
  display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px;
  border-radius: 20px; text-decoration: none; font-size: 11px; font-weight: 600;
  font-family: 'DM Mono', monospace;
  color: var(--pillc); border: 1px solid var(--pillb); background: var(--pillbg);
  transition: opacity .15s, transform .1s; white-space: nowrap;
}
.npill:hover { opacity: .8; transform: translateY(-1px); }
.npill em { font-style: normal; font-size: 10px; background: rgba(255,255,255,.1); border-radius: 10px; padding: 1px 7px; }

/* ── FILTERS ── */
.secbar {
  padding: 12px 72px; border-bottom: 1px solid var(--bd);
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center; background: var(--s1);
}
.sbl { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--mu); margin-right: 4px; flex-shrink: 0; }
.sbtn {
  font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 500; letter-spacing: .5px;
  padding: 5px 13px; border-radius: 6px; border: 1px solid var(--bd);
  background: var(--s2); color: var(--mu); cursor: pointer; transition: all .15s; white-space: nowrap;
}
.sbtn:hover { border-color: var(--bl); color: var(--tx); }
.sbtn.active { background: var(--bl); border-color: var(--bl); color: #fff; }
.sbtn.all-btn.active { background: var(--mu); border-color: var(--mu); color: #fff; }
.filters {
  padding: 12px 72px; border-bottom: 1px solid var(--bd);
  display: flex; gap: 10px; align-items: center; flex-wrap: wrap; background: var(--bg);
}
.fl { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--mu); }
.filters input, .filters select {
  background: var(--s2); border: 1px solid var(--bd); border-radius: 6px;
  color: var(--tx); font-family: 'DM Mono', monospace; font-size: 11px;
  padding: 6px 12px; outline: none;
}
.filters input::placeholder { color: var(--mu); }
.filters input:focus, .filters select:focus { border-color: var(--bl); }

/* ── COLUMN TOGGLE PANEL ── */
.col-toggle {
  padding: 0 72px; background: var(--bg);
  border-bottom: 1px solid var(--bd); overflow: hidden;
  transition: max-height .3s ease, padding .3s ease;
}
.col-toggle.collapsed { max-height: 0; padding-top: 0; padding-bottom: 0; border-bottom: none; }
.col-toggle.expanded  { max-height: 500px; padding-top: 12px; padding-bottom: 14px; }
.ct-hd {
  display: flex; align-items: center; gap: 8px; cursor: pointer;
  padding: 10px 72px; background: var(--bg); border-bottom: 1px solid var(--bd);
  user-select: none;
}
.ct-hd:hover { background: var(--s2); }
.ct-label { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px;
  text-transform: uppercase; color: var(--mu); }
.ct-arrow { font-size: 10px; color: var(--mu); transition: transform .25s; margin-left: auto; }
.ct-arrow.open { transform: rotate(180deg); }
.ct-groups { display: flex; flex-wrap: wrap; gap: 20px; }
.ct-group { min-width: 160px; }
.ct-group-title { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 2px;
  text-transform: uppercase; color: var(--mu); margin-bottom: 8px; padding-bottom: 4px;
  border-bottom: 1px solid var(--bd); }
.ct-cb { display: flex; align-items: center; gap: 6px; cursor: pointer;
  font-family: 'DM Mono', monospace; font-size: 10px; color: var(--tx);
  padding: 2px 0; user-select: none; }
.ct-cb input[type=checkbox] { width: 12px; height: 12px; accent-color: var(--bl); cursor: pointer; }
.ct-cb:hover { color: var(--bl); }
.ct-actions { display: flex; gap: 8px; margin-bottom: 10px; }
.ct-btn { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 1px;
  text-transform: uppercase; color: var(--mu); background: var(--s2);
  border: 1px solid var(--bd); border-radius: 5px; padding: 4px 10px;
  cursor: pointer; transition: color .15s, border-color .15s; }
.ct-btn:hover { color: var(--tx); border-color: var(--bl); }

/* ── MAIN ── */
.main { padding: 0 72px 100px; }
.tier { margin-top: 56px; }
.tier-hd {
  display: flex; align-items: center; gap: 12px;
  padding-bottom: 14px; border-bottom: 1px solid var(--bd); margin-bottom: 16px;
}
.tdot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.tname { font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; }
.tcnt { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 10px; color: var(--mu); background: var(--s2); border: 1px solid var(--bd); border-radius: 4px; padding: 3px 10px; }

/* ── TABLE ── */
.tbl-wrap { overflow-x: auto; border-radius: 10px; border: 1px solid var(--bd); }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
thead tr { background: var(--s2); border-bottom: 1px solid var(--bd); }
th {
  padding: 10px 14px; font-family: 'DM Mono', monospace; font-size: 9px;
  letter-spacing: 1.5px; text-transform: uppercase; color: var(--mu);
  white-space: nowrap; text-align: left; cursor: pointer; user-select: none;
}
th:hover { color: var(--tx); }
th.sa::after { content: ' ↑'; color: var(--bl); }
th.sd::after { content: ' ↓'; color: var(--bl); }
tbody tr { border-bottom: 1px solid var(--bd); transition: background .1s; }
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: var(--s2); }
td { padding: 10px 14px; white-space: nowrap; }
td.mono { font-family: 'DM Mono', monospace; font-size: 11.5px; }
td.dim { color: var(--mu); }
td.sec { font-size: 11px; color: var(--mu); }

/* value colours */
.c-gr { color: var(--gr); font-weight: 600; }
.c-gr2{ color: var(--gr2); font-weight: 600; }
.c-yw { color: var(--yw); }
.c-rd { color: var(--rd); font-weight: 600; }
.c-bl { color: var(--bl); }
.c-or { color: var(--or); }

/* ticker badge */
.tk {
  display: inline-block; background: var(--s2); border: 1px solid var(--bd);
  border-radius: 4px; padding: 2px 8px;
  font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; color: var(--bl);
}

/* growth score badge */
.gs {
  display: inline-block; font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 800;
  min-width: 38px; text-align: center;
}
.gs-hi { color: var(--gr); }
.gs-md { color: var(--yw); }
.gs-lo { color: var(--mu); }

/* target cells */
.t-bear { color: var(--or); font-family: 'DM Mono', monospace; font-size: 11px; }
.t-base { color: var(--gr); font-family: 'DM Mono', monospace; font-size: 11.5px; font-weight: 500; }
.t-bull { color: var(--pu); font-family: 'DM Mono', monospace; font-size: 11px; }
.upside-pos { color: var(--gr); font-weight: 700; font-family: 'DM Mono', monospace; }
.upside-neg { color: var(--rd); font-weight: 700; font-family: 'DM Mono', monospace; }

/* flag badges */
.flag {
  display: inline-block; font-family: 'DM Mono', monospace; font-size: 9px;
  background: rgba(255,77,109,.12); border: 1px solid rgba(255,77,109,.35);
  border-radius: 3px; padding: 1px 5px; color: var(--rd); margin-right: 3px;
}

/* peg coloring */
.peg-gr { color: var(--gr); } .peg-yw { color: var(--yw); } .peg-rd { color: var(--rd); }

/* momentum bar */
.mom-bar {
  display: inline-block; height: 6px; border-radius: 3px;
  background: var(--gr); vertical-align: middle; margin-right: 4px;
}

/* tv rating */
.tv-sb { color: var(--gr); font-weight: 700; }
.tv-b  { color: var(--gr2); }
.tv-n  { color: var(--mu); }
.tv-s  { color: var(--rd); }

tr.hidden { display: none; }

footer {
  padding: 32px 72px; border-top: 1px solid var(--bd);
  font-family: 'DM Mono', monospace; font-size: 10px; color: var(--mu); line-height: 2;
}


/* ── ACCUMULATION ── */
.accum-sa  { color: #00e5a0; font-weight: 700; font-family: 'DM Mono', monospace; font-size: 11px; }
.accum-a   { color: #00c97a; font-weight: 600; font-family: 'DM Mono', monospace; font-size: 11px; }
.accum-n   { color: var(--mu); font-family: 'DM Mono', monospace; font-size: 11px; }
.accum-d   { color: #e07040; font-weight: 600; font-family: 'DM Mono', monospace; font-size: 11px; }
.accum-sd  { color: var(--rd); font-weight: 700; font-family: 'DM Mono', monospace; font-size: 11px; }

/* ── SENTIMENT ── */
.sent-bull  { color: var(--gr);  font-weight: 700; font-family: 'DM Mono', monospace; font-size: 11px; }
.sent-bull2 { color: var(--gr2); font-weight: 600; font-family: 'DM Mono', monospace; font-size: 11px; }
.sent-bear  { color: var(--rd);  font-weight: 700; font-family: 'DM Mono', monospace; font-size: 11px; }
.sent-bear2 { color: #e07040;    font-weight: 600; font-family: 'DM Mono', monospace; font-size: 11px; }
.sent-neu   { color: var(--mu);  font-family: 'DM Mono', monospace; font-size: 11px; }
.sent-na    { color: var(--mu);  font-family: 'DM Mono', monospace; font-size: 11px; opacity:.4; }
.hl-tip    { position: relative; cursor: help; }
.hl-tip:hover .hl-box { display: block; }
.hl-box {
  display: none; position: absolute; z-index: 999;
  bottom: calc(100% + 4px); left: 0;
  background: #1a2235; border: 1px solid var(--bd);
  border-radius: 6px; padding: 10px 14px;
  min-width: 300px; max-width: 400px;
  font-family: 'DM Sans', sans-serif; font-size: 11px;
  color: var(--tx); line-height: 1.7;
  box-shadow: 0 8px 32px rgba(0,0,0,.6);
  white-space: normal;
}
.hl-box li { list-style: none; padding: 3px 0; border-bottom: 1px solid var(--bd); }
.hl-box li:last-child { border-bottom: none; }
@media(max-width: 768px) {
  .hero, .main, .nav, .top-picks, .method-bar, .legend, .secbar, .filters {
    padding-left: 20px; padding-right: 20px;
  }
  .hero-title { font-size: 52px; }
  .gloss-panel { width: 95vw; }
}

/* ── GLOSSARY PANEL ── */
.gloss-overlay {
  display: none; position: fixed; inset: 0; z-index: 9998;
  background: rgba(0,0,0,.55); backdrop-filter: blur(4px);
}
.gloss-overlay.open { display: block; }
.gloss-panel {
  position: fixed; top: 0; right: -620px; z-index: 9999;
  width: 600px; max-width: 95vw; height: 100vh;
  background: var(--s1); border-left: 1px solid var(--bd);
  box-shadow: -8px 0 40px rgba(0,0,0,.5);
  transition: right .3s ease;
  display: flex; flex-direction: column;
}
.gloss-panel.open { right: 0; }
.gloss-hdr {
  padding: 24px 28px 18px; border-bottom: 1px solid var(--bd);
  display: flex; align-items: center; gap: 14px; flex-shrink: 0;
}
.gloss-hdr h2 {
  font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800;
  color: #fff; margin: 0; flex: 1;
}
.gloss-close {
  width: 32px; height: 32px; border-radius: 6px; border: 1px solid var(--bd);
  background: var(--s2); color: var(--mu); font-size: 18px;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: border-color .15s, color .15s;
}
.gloss-close:hover { border-color: var(--rd); color: var(--rd); }
.gloss-search {
  margin: 16px 28px 0; padding: 8px 14px;
  background: var(--s2); border: 1px solid var(--bd); border-radius: 6px;
  color: var(--tx); font-family: 'DM Mono', monospace; font-size: 12px;
  outline: none; flex-shrink: 0;
}
.gloss-search::placeholder { color: var(--mu); }
.gloss-search:focus { border-color: var(--bl); }
.gloss-body {
  flex: 1; overflow-y: auto; padding: 20px 28px 40px;
  scrollbar-width: thin; scrollbar-color: var(--bd) transparent;
}
.gloss-section {
  margin-bottom: 28px;
}
.gloss-section-title {
  font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px;
  text-transform: uppercase; color: var(--gr); margin-bottom: 12px;
  padding-bottom: 6px; border-bottom: 1px solid var(--bd);
}
.gloss-item {
  padding: 8px 0; border-bottom: 1px solid rgba(28,35,51,.5);
  display: flex; gap: 12px; align-items: baseline;
}
.gloss-item:last-child { border-bottom: none; }
.gloss-term {
  font-family: 'DM Mono', monospace; font-size: 11.5px; font-weight: 500;
  color: var(--bl); min-width: 130px; flex-shrink: 0;
}
.gloss-def {
  font-size: 12px; color: var(--tx); line-height: 1.6;
}
.gloss-item.hidden { display: none; }

/* Glossary trigger button in nav */
.gloss-btn {
  display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px;
  border-radius: 20px; font-size: 11px; font-weight: 600;
  font-family: 'DM Mono', monospace;
  color: var(--yw); border: 1px solid rgba(245,200,66,.35); background: rgba(245,200,66,.08);
  cursor: pointer; transition: opacity .15s, transform .1s; white-space: nowrap;
}
.gloss-btn:hover { opacity: .8; transform: translateY(-1px); }
"""


# ── JS ────────────────────────────────────────────────────────────────────────
JS = """
// Column sorting
document.querySelectorAll('th').forEach((th, i) => {
  th.addEventListener('click', () => {
    const tb = th.closest('table').querySelector('tbody');
    const asc = th.classList.toggle('sa');
    th.classList.toggle('sd', !asc);
    th.closest('thead').querySelectorAll('th').forEach(t => { if(t!==th){t.classList.remove('sa','sd');} });
    [...tb.querySelectorAll('tr:not(.hidden)')].sort((a, b) => {
      const ac = a.cells[i], bc = b.cells[i];
      const av = (ac?.dataset.sort !== undefined ? ac.dataset.sort : (ac?.textContent.trim().replace(/[$,%x+BTM↑↓—]/g,'') || ''));
      const bv = (bc?.dataset.sort !== undefined ? bc.dataset.sort : (bc?.textContent.trim().replace(/[$,%x+BTM↑↓—]/g,'') || ''));
      const an = parseFloat(av), bn = parseFloat(bv);
      return isNaN(an)||isNaN(bn) ? (asc?av.localeCompare(bv):bv.localeCompare(av)) : (asc?an-bn:bn-an);
    }).forEach(r => tb.appendChild(r));
  });
});

const activeSectors = new Set();
function setSector(sec, btn) {
  if (sec === '') {
    // "All Sectors" — clear selection
    activeSectors.clear();
  } else {
    if (activeSectors.has(sec)) {
      activeSectors.delete(sec);
    } else {
      activeSectors.add(sec);
    }
  }
  // Sync button states
  document.querySelectorAll('.sbtn:not(.all-btn)').forEach(b => {
    b.classList.toggle('active', activeSectors.has(b.textContent.trim()));
  });
  const allBtn = document.querySelector('.sbtn.all-btn');
  if (allBtn) allBtn.classList.toggle('active', activeSectors.size === 0);
  ft();
}


function ft() {
  const tk     = document.getElementById('fs').value.toUpperCase();
  const ff     = document.getElementById('ff').value;
  const ms     = parseFloat(document.getElementById('ms').value) || 0;
  const ema    = document.getElementById('ema').value;
  const rsiF   = document.getElementById('rsiF').value;
  const accumF = document.getElementById('accumF').value;
  const mrgRaw = document.getElementById('mrg').value;
  const mrg    = mrgRaw !== '' ? parseFloat(mrgRaw) : null;
  document.querySelectorAll('tr.sr').forEach(r => {
    const t   = r.querySelector('.tk')?.textContent || '';
    const s   = r.dataset.sector || '';
    const fl  = r.dataset.flags  || '';
    const gs  = parseFloat(r.dataset.score) || 0;
    const sectorMatch = activeSectors.size === 0 || activeSectors.has(s);
    const flagMatch = ff === '' || (ff === 'clean' && fl === '') || (ff === 'flagged' && fl !== '');
    let emaMatch = true;
    if (ema) {
      const parts = ema.split('+');
      emaMatch = parts.every(p => r.dataset['ema' + p] === '1');
    }
    let rsiMatch = true;
    if (rsiF) {
      const rv = parseFloat(r.dataset.rsi);
      if (isNaN(rv)) { rsiMatch = false; }
      else if (rsiF === 'os30')    { rsiMatch = rv < 30; }
      else if (rsiF === 'os40')    { rsiMatch = rv < 40; }
      else if (rsiF === 'neutral') { rsiMatch = rv >= 40 && rv <= 60; }
      else if (rsiF === 'ob60')    { rsiMatch = rv > 60; }
      else if (rsiF === 'ob70')    { rsiMatch = rv > 70; }
      else if (rsiF === 'ob80')    { rsiMatch = rv > 80; }
    }
    const accumMatch = !accumF || (r.dataset.accum === accumF);
    let revgMatch = true;
    if (mrg !== null && !isNaN(mrg)) {
      const rg = parseFloat(r.dataset.revgrowth);
      revgMatch = !isNaN(rg) && rg >= mrg;
    }
    r.classList.toggle('hidden',
      (tk && !t.includes(tk)) || !sectorMatch || !flagMatch || gs < ms ||
      !emaMatch || !rsiMatch || !accumMatch || !revgMatch
    );
  });
  document.querySelectorAll('.tier').forEach(tier => {
    const visible = [...tier.querySelectorAll('tr.sr')].some(r => !r.classList.contains('hidden'));
    tier.style.display = visible ? '' : 'none';
  });
}

// Animate tiers on scroll
const obs = new IntersectionObserver(es => es.forEach(e => {
  if(e.isIntersecting){ e.target.style.opacity='1'; e.target.style.transform='translateY(0)'; }
}), {threshold: 0.04});
document.querySelectorAll('.tier').forEach((el, i) => {
  el.style.opacity='0'; el.style.transform='translateY(20px)';
  el.style.transition=`opacity .45s ease ${i*.06}s, transform .45s ease ${i*.06}s`;
  obs.observe(el);
});

// ── Column visibility toggle ──────────────────────────────────────────────────
function toggleCol(n, checked) {
  document.querySelectorAll('.sr-table').forEach(tbl => {
    tbl.querySelectorAll('tr').forEach(row => {
      const cell = row.children[n - 1];
      if (cell) cell.style.display = checked ? '' : 'none';
    });
  });
}
function ctTogglePanel() {
  const panel = document.getElementById('ct-panel');
  const arrow = document.getElementById('ct-arrow');
  const open  = panel.classList.contains('expanded');
  panel.classList.toggle('expanded', !open);
  panel.classList.toggle('collapsed', open);
  arrow.classList.toggle('open', !open);
}
function ctShowAll() {
  document.querySelectorAll('.ct-cb input[type=checkbox]').forEach(cb => {
    cb.checked = true;
    toggleCol(parseInt(cb.dataset.col), true);
  });
}
function ctHideAll() {
  document.querySelectorAll('.ct-cb input[type=checkbox]').forEach(cb => {
    cb.checked = false;
    toggleCol(parseInt(cb.dataset.col), false);
  });
}

// Glossary panel
function openGlossary() {
  document.getElementById('glossOverlay').classList.add('open');
  document.getElementById('glossPanel').classList.add('open');
  document.body.style.overflow = 'hidden';
  setTimeout(() => document.getElementById('glossSearch').focus(), 350);
}
function closeGlossary() {
  document.getElementById('glossOverlay').classList.remove('open');
  document.getElementById('glossPanel').classList.remove('open');
  document.body.style.overflow = '';
}
function glossFilter() {
  const q = document.getElementById('glossSearch').value.toLowerCase();
  document.querySelectorAll('.gloss-item').forEach(el => {
    const text = el.textContent.toLowerCase();
    el.classList.toggle('hidden', q && !text.includes(q));
  });
  document.querySelectorAll('.gloss-section').forEach(sec => {
    const visible = [...sec.querySelectorAll('.gloss-item')].some(el => !el.classList.contains('hidden'));
    sec.style.display = visible ? '' : 'none';
  });
}
document.addEventListener('keydown', e => { if(e.key === 'Escape') closeGlossary(); });
"""


def _earnings_cell(r):
    """Render days-to-earnings cell with colour coding."""
    dte = r.get("days_to_earnings")
    if dte is None:
        return '<td style="color:var(--mu);font-size:11px;">—</td>'
    if dte < 0:
        return f'<td class="mono" style="color:var(--mu);">reported</td>'
    if dte <= 7:
        col = "var(--yw)"; label = f"{dte}d ⚡"
    elif dte <= 30:
        col = "var(--gr2)"; label = f"{dte}d"
    else:
        col = "var(--mu)"; label = f"{dte}d"
    return f'<td class="mono" style="color:{col};">{label}</td>'
def _closely_held_cell(r):
    """Render closely-held % cell, colour-coded by concentration level."""
    pct = r.get("closely_held_pct")
    if pct is None:
        return '<td style="color:var(--mu);font-size:11px;">—</td>'

    # Colour: high concentration (>40%) = green (insiders committed)
    #         moderate (20-40%) = yellow
    #         low (<20%) = muted (widely distributed float, less insider conviction)
    if pct >= 40:
        col = "var(--gr)"
    elif pct >= 20:
        col = "var(--yw)"
    else:
        col = "var(--mu)"

    return (
        f'<td class="mono" style="color:{col};" ' 
        f'title="Closely held: {pct:.1f}% of shares not in float">' 
        f'{pct:.1f}%</td>'
    )
def _accumulation_cells(r):
    """Render accumulation signal cells."""
    acc     = r.get("accumulation", {})
    label   = acc.get("label", "")
    score   = acc.get("score")
    signals = acc.get("signals", [])

    if not label or score is None:
        return '<td class="accum-n">—</td><td class="accum-n">—</td>'

    cls_map = {
        "Strong Accumulation": "accum-sa",
        "Accumulation":        "accum-a",
        "Neutral":             "accum-n",
        "Distribution":        "accum-d",
        "Strong Distribution": "accum-sd",
    }
    cls = cls_map.get(label, "accum-n")

    items = "".join(f"<li>{sig}</li>" for sig in signals)
    tip = (
        f'<div class="hl-box">'
        f'<b>Accumulation score: {score}/100</b><br>'
        f'<span style="font-size:10px;color:var(--mu);">MACD · Momentum · Volume · ADX · MoneyFlow</span><br><br>'
        f'<ul style="margin:0;padding:0;">{items}</ul>'
        f'</div>'
    ) if signals else ""

    label_cell = (
        f'<td class="hl-tip">'
        f'<span class="{cls}">{label}</span>'
        f'{tip}</td>'
    )
    col = "var(--gr)" if score >= 60 else "var(--rd)" if score < 40 else "var(--mu)"
    score_cell = (
        f'<td style="font-family:DM Mono,monospace;font-size:11px;color:{col};">'
        f'{score}/100</td>'
    )
    return label_cell + score_cell
def _sentiment_cells(s):
    """Render sentiment cells from TV-derived sentiment dict."""
    label   = s.get("label", "")
    score   = s.get("score")
    signals = s.get("signals", [])

    if not label or score is None:
        return '<td class="sent-na">—</td><td class="sent-na">—</td>'

    if   label == "Bullish":         cls = "sent-bull"
    elif label == "Leaning Bullish": cls = "sent-bull2"
    elif label == "Bearish":         cls = "sent-bear"
    elif label == "Leaning Bearish": cls = "sent-bear2"
    else:                             cls = "sent-neu"

    items = "".join(f"<li>{sig}</li>" for sig in signals)
    tip = (
        f'<div class="hl-box">'
        f'<b>Sentiment score: {score}/100</b><br><br>'
        f'<ul style="margin:0;padding:0;">{items}</ul>'
        f'</div>'
    ) if signals else ""

    label_cell = (
        f'<td class="hl-tip">'
        f'<span class="{cls}">{label}</span>'
        f'{tip}</td>'
    )
    col = "var(--gr)" if score >= 70 else "var(--rd)" if score < 40 else "var(--mu)"
    score_cell = (
        f'<td style="font-family:DM Mono,monospace;font-size:11px;color:{col};">'
        f'{score}/100</td>'
    )
    return label_cell + score_cell
def _analyst_target_cells(r):
    """Render analyst price target and upside cells, colour-coded."""
    target = r.get("analyst_target")
    upside = r.get("analyst_upside")
    price  = r.get("price")

    if target and target > 0 and price and price > 0:
        col = "var(--gr)" if target >= price else "var(--rd)"
        tgt_cell = f'<td class="mono" style="color:{col};">${target:,.2f}</td>'
    else:
        tgt_cell = '<td style="color:var(--mu);font-size:11px;">&#8212;</td>'

    if upside is not None:
        col = "var(--gr)" if upside >= 0 else "var(--rd)"
        upside_cell = f'<td class="mono" style="color:{col};">{upside:+.1f}%</td>'
    else:
        upside_cell = '<td style="color:var(--mu);font-size:11px;">&#8212;</td>'

    return tgt_cell + upside_cell


def _build_best3_cell(r):
    """Render the Backtest Top-3 table cell."""
    mean_fv = r.get("best3_mean")
    upside  = r.get("best3_upside")
    sort_val = f"{upside:.4f}" if upside is not None else "9999"
    if not mean_fv:
        return f'<td class="mono" data-sort="{sort_val}" style="color:#4a5568;text-align:center;">—</td>'
    methods = r.get("best3_methods") or []
    u_color = "#68d391" if (upside and upside >= 0) else "#fc8181"
    u_str   = fpc(upside) if upside is not None else "—"
    m_str   = " · ".join(methods)
    return (
        f'<td class="mono" data-sort="{sort_val}" style="font-size:11px;line-height:1.5;">'
        f'<span style="color:#e2e8f0;">{fp(mean_fv)}</span><br>'
        f'<span style="color:{u_color};font-size:10px;font-weight:600;">{u_str}</span><br>'
        f'<span style="color:#718096;font-size:9px;">{m_str}</span>'
        f'</td>'
    )


# ── Watchlist helpers ─────────────────────────────────────────────────────────
_WL_CSS = """
#wl-bar{position:fixed;bottom:0;left:0;right:0;background:#1a1f35;
  border-top:1px solid #252a3a;padding:10px 20px;display:flex;
  align-items:center;gap:12px;z-index:9999;transition:opacity .2s;}
#wl-count{color:#6b7194;font-size:12px;min-width:90px;}
#wl-add-btn{background:#4f8ef7;color:#fff;border:none;border-radius:5px;
  padding:6px 14px;font-size:12px;font-weight:600;cursor:pointer;}
#wl-add-btn:hover{background:#6ba3ff;}
#wl-clear-btn{background:transparent;color:#6b7194;border:1px solid #252a3a;
  border-radius:5px;padding:6px 12px;font-size:12px;cursor:pointer;}
#wl-clear-btn:hover{color:#e8eaf0;}
.cb-cell{width:28px;text-align:center;padding:4px 2px;}
.cb-th{width:28px;text-align:center;}
input.row-check{cursor:pointer;width:14px;height:14px;accent-color:#4f8ef7;}
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


def build_html(results, ts, total, index_name="S&P 500",
               best3_label="CONSENSUS TOP-3", best3_tooltip="",
               suite_port: int = 5050):
    tier_map = {t[0]: [] for t in TIERS}
    for r in results:
        if r["tier"] in tier_map:
            tier_map[r["tier"]].append(r)
    for t in tier_map:
        tier_map[t].sort(key=lambda x: -x["growth_score"])

    exceptional = len(tier_map["EXCEPTIONAL GROWTH"])
    strong      = len(tier_map["STRONG GROWTH"])
    moderate    = len(tier_map["MODERATE GROWTH"])
    stretched   = len(tier_map["STRETCHED VALUATION"])

    # Top 8 by growth score
    top8 = sorted(results, key=lambda x: -x["growth_score"])[:8]

    all_sectors = sorted(set(r["sector"] for r in results))

    # Nav pills
    nav_html = ""
    for name, lo, hi, color in TIERS:
        cnt = len(tier_map[name])
        if not cnt: continue
        sid = name.replace(" ", "_")
        h = color.lstrip("#")
        rv, gv, bv = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        bg  = f"rgba({rv},{gv},{bv},0.1)"
        brd = f"rgba({rv},{gv},{bv},0.35)"
        nav_html += (
            f'<a class="npill" href="#{sid}" '
            f'style="--pillc:{color};--pillb:{brd};--pillbg:{bg};">'
            f'{name} <em>{cnt}</em></a>'
        )

    # Sector filter buttons
    sector_btns = "".join(
        f'<button class="sbtn" onclick="setSector(\'{s}\', this)">{s}</button>'
        for s in all_sectors
    )

    # Top picks cards
    top_cards = ""
    for r in top8:
        gs_cls = "gs-hi" if r["growth_score"] >= 60 else "gs-md" if r["growth_score"] >= 35 else "gs-lo"
        up_str = fpc(r["upside_pct"]) if r["upside_pct"] is not None else "—"
        top_cards += (
            f'<div class="tp-card">'
            f'<span class="tp-ticker">{r["ticker"]}</span>'
            f'<span class="tp-score {gs_cls}">{fsc(r["growth_score"])}</span>'
            f'<span class="tp-upside">{up_str} upside</span>'
            f'</div>'
        )

    # Tier sections
    sections_html = ""
    for tier_name, lo, hi, color in TIERS:
        stocks = tier_map[tier_name]
        if not stocks: continue
        sid = tier_name.replace(" ", "_")

        rows = ""
        for r in stocks:
            gs = r["growth_score"]
            gs_cls = "gs-hi" if gs >= 60 else "gs-md" if gs >= 35 else "gs-lo"

            up = r["upside_pct"]
            up_cls = "upside-pos" if (up and up >= 0) else "upside-neg"

            # PEG coloring
            peg = r["peg_current"]
            peg_cls = ""
            if peg is not None:
                peg_cls = "peg-gr" if peg < 1.0 else "peg-yw" if peg < 1.5 else "peg-rd"

            # TV rating class
            tv = r["tv_rating"]
            tv_cls = "tv-sb" if (tv and tv >= 0.5) else "tv-b" if (tv and tv >= 0.1) else "tv-n" if (tv and tv >= -0.1) else "tv-s"
            tv_str = frating(tv)

            # Momentum mini-bar
            ms = r["momentum_score"] or 0
            bar_w = int(ms / 20 * 50)
            mom_html = f'<span class="mom-bar" style="width:{bar_w}px;opacity:.7;"></span>{fsc(ms)}/20'

            # Flag badges
            flag_html = "".join(f'<span class="flag">{f}</span>' for f in r["flags"])
            flag_data = "|".join(r["flags"])

            # Perf coloring
            def perf_cls(v):
                if v is None: return ""
                return "c-gr" if v >= 10 else "c-gr2" if v >= 0 else "c-rd"

            # EMA below flags for filtering
            ema13_v = "1" if r.get("below_ema13") else "0"
            ema50_v = "1" if r.get("below_ema50") else "0"
            ema200_v = "1" if r.get("below_ema200") else "0"

            # RSI value for filtering
            rsi_v = r.get("rsi")
            rsi_data   = f'{rsi_v:.1f}' if rsi_v is not None else ""
            accum_data = (r.get("accumulation") or {}).get("label", "")
            revg_v     = r.get("rev_growth")
            revg_data  = f'{revg_v:.4f}' if revg_v is not None else ""

            rows += (
                f'<tr class="sr" data-sector="{r["sector"]}" data-index="{r.get("index_label","")}" data-flags="{flag_data}" data-score="{gs}" data-ema13="{ema13_v}" data-ema50="{ema50_v}" data-ema200="{ema200_v}" data-rsi="{rsi_data}" data-accum="{accum_data}" data-revgrowth="{revg_data}">'
                f'<td class="cb-cell"><input type="checkbox" class="row-check" value="{r["ticker"]}"></td>'
                f'<td><span class="gs {gs_cls}">{fsc(gs)}</span></td>'
                f'<td><span class="tk">{r["ticker"]}</span></td>'
                f'<td class="sec">{r["sector"]}</td>'
                f'<td class="mono">{fp(r["price"])}</td>'
                f'<td class="t-bear">{fp(r["target_bear"])}</td>'
                f'<td class="t-base">{fp(r["target_base"])}</td>'
                f'<td class="t-bull">{fp(r["target_bull"])}</td>'
                + _build_best3_cell(r) +
                f'<td class="{up_cls}">{fpc(up)}</td>'
                f'<td class="mono {peg_cls}">{fx(peg)}</td>'
                f'<td class="mono c-yw">{fp(r["peg_eps_used"])}</td>'
                f'<td class="mono dim" style="font-size:10px;color:var(--mu)">{r.get("eps_source","—")}</td>'
                f'<td class="mono c-yw">{fpc(r.get("peg_growth_used"), False)}</td>'
                f'<td class="mono dim">{fpc(r.get("eps_growth_fq"), False)}</td>'
                f'<td class="mono c-gr">{fpc(r["rev_growth"], False)}</td>'
                f'<td class="mono dim">{fpc(r.get("rev_growth_fq"), False)}</td>'
                f'<td class="mono dim">{fpc(r.get("rev_qoq"), False)}</td>'
                f'<td class="mono dim">{fpc(r.get("eps_qoq"), False)}</td>'
                f'<td class="mono dim">{fpc(r.get("gp_growth"), False)}</td>'
                f'<td class="mono dim">{fpc(r.get("ni_growth"), False)}</td>'
                f'<td class="mono dim">{fpc(r.get("fcf_growth"), False)}</td>'
                f'<td class="mono dim">{fpc(r.get("ebitda_growth"), False)}</td>'
                f'<td class="mono dim">{fpct(r.get("rd_ratio"))}</td>'
                f'<td class="mono dim">{fpct(r["gross_margin"])}</td>'
                f'<td class="mono dim">{fpct(r["op_margin"])}</td>'
                f'<td class="mono dim">{fpct(r["roe"])}</td>'
                f'<td class="mono c-gr">{fpct(r.get("roic"))}</td>'
                f'<td class="mono dim">{fpct(r.get("roa"))}</td>'
                f'<td class="mono dim">{fx(r.get("current_ratio"))}</td>'
                f'<td class="mono dim">{fx(r.get("p_fcf"))}</td>'
                f'<td class="mono dim">{fx(r.get("ev_ebitda"))}</td>'
                + _earnings_cell(r)
                + f'<td class="mono {perf_cls(r["perf_1m"])}">{fpc(r["perf_1m"])}</td>'
                f'<td class="mono {perf_cls(r["perf_3m"])}">{fpc(r["perf_3m"])}</td>'
                f'<td class="mono {perf_cls(r["perf_6m"])}">{fpc(r["perf_6m"])}</td>'
                f'<td class="mono dim">{mom_html}</td>'
                f'<td class="{tv_cls}">{tv_str}</td>'
                + _sentiment_cells(r.get('sentiment', {}))
                + _accumulation_cells(r)
                + _closely_held_cell(r)
                + _analyst_target_cells(r)
                + f'<td class="mono dim">{fb(r["market_cap"])}</td>'
                  f'<td>{flag_html}</td>'
                  f'</tr>'
            )

        sections_html += (
            f'<section class="tier" id="{sid}">'
            f'<div class="tier-hd">'
            f'<div class="tdot" style="background:{color};box-shadow:0 0 8px {color};"></div>'
            f'<span class="tname" style="color:{color};">{tier_name}</span>'
            f'<span class="tcnt">{len(stocks)} stocks</span>'
            f'</div>'
            f'<div class="tbl-wrap"><table class="sr-table">'
            f'<thead><tr>'
            f'<th class="cb-th"><input type="checkbox" id="cb-all" title="Select all"></th>'
            f'<th>Score</th><th>Ticker</th><th>Sector</th><th>Price</th>'
            f'<th>Bear Target</th><th>Base Target</th><th>Bull Target</th>'
            f'<th title="{best3_tooltip}">{best3_label}</th>'
            f'<th>Upside</th><th>Fwd PEG</th>'
            f'<th>Fwd EPS</th><th>EPS Src</th>'
            f'<th title="Forward-blended growth rate used in PEG calculation">Fwd EPS Growth</th>'
            f'<th title="Most recent quarter YoY">EPS Gth QoQ</th>'
            f'<th title="Forward-blended revenue growth">Rev Growth</th>'
            f'<th title="Most recent quarter YoY">Rev Gth QoQ</th>'
            f'<th title="Sequential QoQ revenue growth">Rev QoQ</th>'
            f'<th title="Sequential QoQ EPS growth">EPS QoQ</th>'
            f'<th title="Gross profit growth TTM YoY">GP Growth</th>'
            f'<th title="Net income growth TTM YoY">NI Growth</th>'
            f'<th title="FCF growth TTM YoY">FCF Growth</th>'
            f'<th title="EBITDA growth TTM YoY">EBITDA Gth</th>'
            f'<th title="R&D as % of revenue">R&D %</th>'
            f'<th>Gross Margin</th><th>Op Margin</th><th>ROE</th>'
            f'<th title="Return on Invested Capital — best quality metric">ROIC</th>'
            f'<th title="Return on Assets">ROA</th>'
            f'<th title="Current Ratio — liquidity health">Curr Ratio</th>'
            f'<th title="Price / Free Cash Flow">P/FCF</th>'
            f'<th title="EV / EBITDA — capital-structure-neutral valuation">EV/EBITDA</th>'
            f'<th title="Days until next earnings release">Earnings In</th>'
            f'<th>Perf 1M</th><th>Perf 3M</th><th>Perf 6M</th>'
            f'<th>Momentum</th><th>TV Signal</th>'
            f'<th title="TV rating + RSI + momentum + EPS acceleration">Sentiment</th>'
            f'<th title="Sentiment score 0-100 (hover for breakdown)">Sent. Score</th>'
            f'<th title="Institutional accumulation proxy: MACD + Volume + ADX + CCI (hover for breakdown)">Accumulation</th>'
            f'<th title="Accumulation score 0-100">Accum. Score</th>'
            f'<th title="% of shares held by insiders/restricted holders (total - float) / total">Closely Held %</th>'
            f'<th title="TradingView analyst consensus 1-year price target">Analyst Target</th>'
            f'<th title="% upside/downside to analyst consensus target">Analyst Upside</th>'
            f'<th>Mkt Cap</th><th>Flags</th>'
            f'</tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table></div></section>'
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{index_name} Growth &amp; Momentum Screener</title>
<style>{CSS}__WL_CSS__</style>
</head>
<body>

<div class="hero">
  <div class="hero-tag">// Growth Intelligence</div>
  <h1 class="hero-title">Growth &amp;<br><em>Momentum</em></h1>
  <p class="hero-sub">GENERATED {ts} &nbsp;&middot;&nbsp; {total} STOCKS ANALYSED &nbsp;&middot;&nbsp; FORWARD PEG + REVENUE EXTRAPOLATION</p>
  <div class="stat-row">
    <div class="stat"><div class="stat-l">Exceptional</div><div class="stat-v v-gr">{exceptional}</div></div>
    <div class="stat"><div class="stat-l">Strong</div><div class="stat-v v-gr2">{strong}</div></div>
    <div class="stat"><div class="stat-l">Moderate</div><div class="stat-v v-yw">{moderate}</div></div>
    <div class="stat"><div class="stat-l">Stretched</div><div class="stat-v v-rd">{stretched}</div></div>
    <div class="stat"><div class="stat-l">Scored</div><div class="stat-v v-mu">{total}</div></div>
  </div>
</div>

<div class="top-picks">
  <div class="tp-label">Top 8 Growth Opportunities (by Score)</div>
  <div class="tp-cards">{top_cards}</div>
</div>

<div class="method-bar">
  <span class="mb-label">Methods</span>
  <span class="mb-item">Forward PEG <b>{int(PEG_WEIGHT*100)}%</b> &nbsp;&middot;&nbsp; Bear <b>{PEG_BEAR}x</b> / Base <b>{PEG_BASE}x</b> / Bull <b>{PEG_BULL}x</b></span>
  <span class="mb-item">Revenue Extrapolation <b>{int(REV_WEIGHT*100)}%</b> &nbsp;&middot;&nbsp; Fwd Rev &times; Margin &times; Sector P/E</span>
  <span class="mb-item">EV/EBITDA <b>20%</b> &nbsp;&middot;&nbsp; Fwd EBITDA &times; Sector Multiple &minus; Net Debt</span>
  <span class="mb-note">&#128202; EPS: analyst next-FY consensus &rarr; fq&times;4 &rarr; TTM&times;growth &nbsp;&middot;&nbsp; Rev: analyst next-FY consensus &rarr; TTM&times;fwd growth</span>
</div>

<div class="legend">
  <span class="lg-label">Score</span>
  <span class="lg-item"><span class="lg-dot" style="background:#00e5a0"></span>60–100 Exceptional</span>
  <span class="lg-item"><span class="lg-dot" style="background:#f5c842"></span>35–59 Strong</span>
  <span class="lg-item"><span class="lg-dot" style="background:#3d5070"></span>0–34 Moderate</span>
  &nbsp;&nbsp;
  <span class="lg-label">PEG</span>
  <span class="lg-item" style="color:#00e5a0">&#9632; &lt;1.0 Attractive</span>
  <span class="lg-item" style="color:#f5c842">&#9632; &lt;1.5 Reasonable</span>
  <span class="lg-item" style="color:#ff4d6d">&#9632; &gt;1.5 Stretched</span>
  &nbsp;&nbsp;
  <span class="lg-label">Targets</span>
  <span class="lg-item" style="color:#fb923c">&#9632; Bear (PEG {PEG_BEAR}x)</span>
  <span class="lg-item" style="color:#00e5a0">&#9632; Base (PEG {PEG_BASE}x)</span>
  <span class="lg-item" style="color:#a78bfa">&#9632; Bull (PEG {PEG_BULL}x)</span>
</div>

<nav class="nav"><span class="nl">Jump to</span>{nav_html}<button class="gloss-btn" onclick="openGlossary()">&#128214; Glossary</button></nav>

<div class="secbar">
  <span class="sbl">Sector</span>
  <button class="sbtn all-btn active" onclick="setSector('', this)">All Sectors</button>
  {sector_btns}
</div>

<div class="filters">
  <span class="fl">Filter</span>
  <input id="fs" type="text" placeholder="Search ticker…" oninput="ft()" style="width:130px;">
  <select id="ff" onchange="ft()">
    <option value="">All Stocks</option>
    <option value="clean">No Flags (Clean)</option>
    <option value="flagged">Flagged Only</option>
  </select>
  <label style="font-family:DM Mono,monospace;font-size:10px;color:var(--mu);display:flex;align-items:center;gap:6px;">
    Min Score
    <input id="ms" type="number" min="0" max="100" value="0" oninput="ft()" style="width:60px;">
  </label>
  <select id="ema" onchange="ft()" style="font-family:DM Mono,monospace;font-size:11px;">
    <option value="">All (No EMA Filter)</option>
    <option value="13">Below EMA 13</option>
    <option value="50">Below EMA 50</option>
    <option value="200">Below EMA 200</option>
    <option value="13+50">Below EMA 13 &amp; 50</option>
    <option value="13+50+200">Below EMA 13, 50 &amp; 200</option>
  </select>
  <select id="rsiF" onchange="ft()" style="font-family:DM Mono,monospace;font-size:11px;">
    <option value="">All (No RSI Filter)</option>
    <option value="os30">Oversold (RSI &lt; 30)</option>
    <option value="os40">Oversold–Weak (RSI &lt; 40)</option>
    <option value="neutral">Neutral (RSI 40–60)</option>
    <option value="ob60">Strong–Overbought (RSI &gt; 60)</option>
    <option value="ob70">Overbought (RSI &gt; 70)</option>
    <option value="ob80">Extreme Overbought (RSI &gt; 80)</option>
  </select>
  <select id="accumF" onchange="ft()" style="font-family:DM Mono,monospace;font-size:11px;">
    <option value="">All (No Accum. Filter)</option>
    <option value="Strong Accumulation">Strong Accumulation</option>
    <option value="Accumulation">Accumulation</option>
    <option value="Neutral">Neutral</option>
    <option value="Distribution">Distribution</option>
    <option value="Strong Distribution">Strong Distribution</option>
  </select>
  <label style="font-family:DM Mono,monospace;font-size:10px;color:var(--mu);display:flex;align-items:center;gap:6px;">
    Min Rev Growth
    <input id="mrg" type="number" placeholder="%" step="1" oninput="ft()" style="width:65px;">
  </label>
</div>

<div class="ct-hd" onclick="ctTogglePanel()">
  <span class="ct-label">Columns</span>
  <span style="font-family:DM Mono,monospace;font-size:10px;color:var(--mu);">Show / hide individual columns</span>
  <span class="ct-arrow" id="ct-arrow">▼</span>
</div>
<div class="col-toggle collapsed" id="ct-panel">
  <div class="ct-actions">
    <button class="ct-btn" onclick="ctShowAll()">Show All</button>
    <button class="ct-btn" onclick="ctHideAll()">Hide All</button>
  </div>
  <div class="ct-groups">
    <div class="ct-group">
      <div class="ct-group-title">Identifiers &amp; Price</div>
      <label class="ct-cb"><input type="checkbox" checked data-col="3" onchange="toggleCol(3,this.checked)"> Sector</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="4" onchange="toggleCol(4,this.checked)"> Price</label>
    </div>
    <div class="ct-group">
      <div class="ct-group-title">Targets &amp; Upside</div>
      <label class="ct-cb"><input type="checkbox" checked data-col="5" onchange="toggleCol(5,this.checked)"> Bear Target</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="6" onchange="toggleCol(6,this.checked)"> Base Target</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="7" onchange="toggleCol(7,this.checked)"> Bull Target</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="8" onchange="toggleCol(8,this.checked)"> Upside %</label>
    </div>
    <div class="ct-group">
      <div class="ct-group-title">Valuation</div>
      <label class="ct-cb"><input type="checkbox" checked data-col="9"  onchange="toggleCol(9,this.checked)"> Fwd PEG</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="10" onchange="toggleCol(10,this.checked)"> Fwd EPS</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="11" onchange="toggleCol(11,this.checked)"> EPS Src</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="29" onchange="toggleCol(29,this.checked)"> P/FCF</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="30" onchange="toggleCol(30,this.checked)"> EV/EBITDA</label>
    </div>
    <div class="ct-group">
      <div class="ct-group-title">Growth Rates</div>
      <label class="ct-cb"><input type="checkbox" checked data-col="12" onchange="toggleCol(12,this.checked)"> Fwd EPS Growth</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="13" onchange="toggleCol(13,this.checked)"> EPS Gth QoQ</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="14" onchange="toggleCol(14,this.checked)"> Rev Growth</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="15" onchange="toggleCol(15,this.checked)"> Rev Gth QoQ</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="16" onchange="toggleCol(16,this.checked)"> Rev QoQ</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="17" onchange="toggleCol(17,this.checked)"> EPS QoQ</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="18" onchange="toggleCol(18,this.checked)"> GP Growth</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="19" onchange="toggleCol(19,this.checked)"> NI Growth</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="20" onchange="toggleCol(20,this.checked)"> FCF Growth</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="21" onchange="toggleCol(21,this.checked)"> EBITDA Gth</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="22" onchange="toggleCol(22,this.checked)"> R&amp;D %</label>
    </div>
    <div class="ct-group">
      <div class="ct-group-title">Quality Metrics</div>
      <label class="ct-cb"><input type="checkbox" checked data-col="23" onchange="toggleCol(23,this.checked)"> Gross Margin</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="24" onchange="toggleCol(24,this.checked)"> Op Margin</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="25" onchange="toggleCol(25,this.checked)"> ROE</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="26" onchange="toggleCol(26,this.checked)"> ROIC</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="27" onchange="toggleCol(27,this.checked)"> ROA</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="28" onchange="toggleCol(28,this.checked)"> Curr Ratio</label>
    </div>
    <div class="ct-group">
      <div class="ct-group-title">Technical</div>
      <label class="ct-cb"><input type="checkbox" checked data-col="31" onchange="toggleCol(31,this.checked)"> Earnings In</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="32" onchange="toggleCol(32,this.checked)"> Perf 1M</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="33" onchange="toggleCol(33,this.checked)"> Perf 3M</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="34" onchange="toggleCol(34,this.checked)"> Perf 6M</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="35" onchange="toggleCol(35,this.checked)"> Momentum</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="36" onchange="toggleCol(36,this.checked)"> TV Signal</label>
    </div>
    <div class="ct-group">
      <div class="ct-group-title">Signal</div>
      <label class="ct-cb"><input type="checkbox" checked data-col="37" onchange="toggleCol(37,this.checked)"> Sentiment</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="38" onchange="toggleCol(38,this.checked)"> Sent. Score</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="39" onchange="toggleCol(39,this.checked)"> Accumulation</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="40" onchange="toggleCol(40,this.checked)"> Accum. Score</label>
    </div>
    <div class="ct-group">
      <div class="ct-group-title">Other</div>
      <label class="ct-cb"><input type="checkbox" checked data-col="41" onchange="toggleCol(41,this.checked)"> Closely Held %</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="42" onchange="toggleCol(42,this.checked)"> Analyst Target</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="43" onchange="toggleCol(43,this.checked)"> Analyst Upside</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="44" onchange="toggleCol(44,this.checked)"> Mkt Cap</label>
      <label class="ct-cb"><input type="checkbox" checked data-col="45" onchange="toggleCol(45,this.checked)"> Flags</label>
    </div>
  </div>
</div>

<main class="main">{sections_html}</main>

<footer>
  Generated {ts} &nbsp;&middot;&nbsp; Source: TradingView Screener<br>
  <b>Forward PEG:</b> Price Target = Forward EPS &times; min(Fwd Growth% &times; PEG target, {int(MAX_FWD_PE)}x) &nbsp;&middot;&nbsp; PEG Bear={PEG_BEAR}x / Base={PEG_BASE}x / Bull={PEG_BULL}x<br>
  <b>Forward EPS:</b> (1) Analyst next-FY consensus (preferred) &nbsp;&middot;&nbsp; (2) Analyst next-quarter &times;4 annualised &nbsp;&middot;&nbsp; (3) TTM EPS &times; blended growth<br>  <b>Forward Revenue:</b> (1) Analyst next-FY consensus (preferred) &nbsp;&middot;&nbsp; (2) TTM revenue &times; blended forward growth rate<br>
  <b>Forward Growth Rate:</b> Implied from fwd EPS vs TTM where available; else blend of quarterly YoY (70%) + TTM (30%) if accelerating, 50/50 if decelerating<br>
  <b>Revenue Extrapolation:</b> Fwd Revenue (TTM &times; fwd growth) &times; Net Margin &times; Growth-Adjusted Sector P/E &divide; Shares<br>
  <b>Composite Target:</b> {int(PEG_WEIGHT*100)}% Forward PEG + {int(REV_WEIGHT*100)}% Revenue Extrapolation<br>
  <b>Growth Score (0–100):</b> Quality 35pts (ROIC, ROE, Margins, FCF, Liquidity) + Growth Momentum 35pts (Rev/EPS/FCF/GP/NI/EBITDA growth, R&amp;D) + Technical 20pts (TV rating, MoneyFlow, RSI, Stoch, Price Perf) + Valuation 10pts (PEG, P/FCF, EV/EBITDA, Upside) &minus; 4pts per risk flag<br>
  <b>Flags:</b> UNPROFITABLE &middot; NEG FCF &middot; NEG MARGIN &middot; HIGH DEBT &middot; REV DECLINE<br>
  <b>Sentiment:</b> TV rating + RSI + 1M/3M momentum + 52wk position + EPS acceleration<br>
  <b>Closely Held %:</b> (Total shares &minus; Float) &divide; Total shares &mdash; captures insider/restricted/strategic ownership not freely tradeable<br>
  <b>Accumulation:</b> Institutional buying proxy &mdash; MACD histogram + Momentum + Relative volume trend + ADX/DI + MoneyFlow &mdash; hover any cell for signal breakdown<br>
  &#9888; Educational use only &mdash; not investment advice. Forward EPS estimates may be inaccurate. Always verify against SEC filings.
</footer>

<div id="glossOverlay" class="gloss-overlay" onclick="closeGlossary()"></div>
<div id="glossPanel" class="gloss-panel">
  <div class="gloss-hdr">
    <h2>&#128214; Glossary</h2>
    <button class="gloss-close" onclick="closeGlossary()">&times;</button>
  </div>
  <input id="glossSearch" class="gloss-search" type="text" placeholder="Search terms…" oninput="glossFilter()">
  <div class="gloss-body">

    <div class="gloss-section">
      <div class="gloss-section-title">Scoring Pillars</div>
      <div class="gloss-item"><span class="gloss-term">Growth Score</span><span class="gloss-def">Composite 0&ndash;100 rating combining four pillars: Quality (0&ndash;35), Growth Momentum (0&ndash;35), Technical Signal (0&ndash;20), and Valuation (0&ndash;10). Each risk flag deducts 4 points.</span></div>
      <div class="gloss-item"><span class="gloss-term">Quality Pillar</span><span class="gloss-def">0&ndash;35 pts. Measures business durability via ROIC (10), ROE (4), gross margin (8), operating margin (5), FCF generation (5), and liquidity/current ratio (3).</span></div>
      <div class="gloss-item"><span class="gloss-term">Growth Momentum</span><span class="gloss-def">0&ndash;35 pts. Multi-axis growth assessment: revenue growth + QoQ acceleration (10), EPS growth + acceleration (8), gross profit growth (4), FCF growth (5), net income growth (4), EBITDA growth (3), and R&amp;D investment (3).</span></div>
      <div class="gloss-item"><span class="gloss-term">Technical Signal</span><span class="gloss-def">0&ndash;20 pts. Market confirmation of fundamentals: TV composite rating (4), MoneyFlow (4), RSI (2), Stochastic (2), Williams %R (1), price performance across 1M/3M/6M/YTD (7), and 52-week position (1).</span></div>
      <div class="gloss-item"><span class="gloss-term">Valuation Pillar</span><span class="gloss-def">0&ndash;10 pts. Multi-lens value check: upside to model target (4), forward PEG (2), P/FCF (2), and EV/EBITDA (2).</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Price Target Methods</div>
      <div class="gloss-item"><span class="gloss-term">Forward PEG</span><span class="gloss-def">Price Target = Forward EPS &times; min(Forward Growth% &times; PEG target, {int(MAX_FWD_PE)}x). Three scenarios: Bear (PEG {PEG_BEAR}x), Base ({PEG_BASE}x), Bull ({PEG_BULL}x). Weight: {int(PEG_WEIGHT*100)}% of composite.</span></div>
      <div class="gloss-item"><span class="gloss-term">Revenue Extrap.</span><span class="gloss-def">Projects forward revenue &times; net margin &times; growth-adjusted sector P/E &divide; shares. Uses analyst consensus revenue when available. Weight: {int(REV_WEIGHT*100)}% of composite.</span></div>
      <div class="gloss-item"><span class="gloss-term">EV/EBITDA</span><span class="gloss-def">Fair value = (Forward EBITDA &times; growth-adjusted sector multiple) &minus; net debt, divided by shares. Captures capital-structure-neutral value. Weight: 20% of composite.</span></div>
      <div class="gloss-item"><span class="gloss-term">Composite Target</span><span class="gloss-def">Weighted average of all available methods, renormalised if a method has insufficient data. Bear/bull targets use &plusmn;20% on revenue and EV methods.</span></div>
      <div class="gloss-item"><span class="gloss-term">Forward EPS</span><span class="gloss-def">Priority: (1) Analyst next-FY consensus, (2) analyst next-quarter &times; 4 annualised, (3) TTM EPS &times; blended growth rate, (4) last fiscal year EPS &times; growth.</span></div>
      <div class="gloss-item"><span class="gloss-term">Forward Growth</span><span class="gloss-def">Implied from analyst forward EPS vs TTM when available. Otherwise blended: 70% quarterly + 30% TTM if accelerating, 50/50 if decelerating. Capped at {int(MAX_GROWTH_RATE*100)}%.</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Report Columns &mdash; Identifiers &amp; Targets</div>
      <div class="gloss-item"><span class="gloss-term">Score</span><span class="gloss-def">The composite Growth Score (0&ndash;100) for ranking stocks.</span></div>
      <div class="gloss-item"><span class="gloss-term">Ticker</span><span class="gloss-def">Stock symbol as listed on the exchange.</span></div>
      <div class="gloss-item"><span class="gloss-term">Sector</span><span class="gloss-def">GICS sector classification (Technology, Health Care, etc.).</span></div>
      <div class="gloss-item"><span class="gloss-term">Price</span><span class="gloss-def">Most recent closing price.</span></div>
      <div class="gloss-item"><span class="gloss-term">Bear / Base / Bull</span><span class="gloss-def">12-month composite price targets at pessimistic, expected, and optimistic scenarios.</span></div>
      <div class="gloss-item"><span class="gloss-term">Upside</span><span class="gloss-def">Percentage difference between the Base Target and current price. Positive = undervalued, negative = overvalued.</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Report Columns &mdash; Valuation &amp; Earnings</div>
      <div class="gloss-item"><span class="gloss-term">Fwd PEG</span><span class="gloss-def">Forward Price/Earnings-to-Growth ratio. Current P/E &divide; forward EPS growth rate. &lt;1.0 = attractive, &lt;1.5 = reasonable, &gt;1.5 = stretched.</span></div>
      <div class="gloss-item"><span class="gloss-term">Fwd EPS</span><span class="gloss-def">The annualised forward EPS estimate used in the PEG price target calculation.</span></div>
      <div class="gloss-item"><span class="gloss-term">EPS Src</span><span class="gloss-def">Source of the forward EPS estimate: analyst_fy (consensus full-year), analyst_fq&times;4 (quarterly annualised), ttm&times;growth, or fy&times;growth.</span></div>
      <div class="gloss-item"><span class="gloss-term">Fwd EPS Growth</span><span class="gloss-def">The forward-blended EPS growth rate (%) used in the PEG calculation. Derived from analyst estimates or trend-blended historical rates.</span></div>
      <div class="gloss-item"><span class="gloss-term">P/E (TTM)</span><span class="gloss-def">Price-to-Earnings ratio using trailing 12-month diluted EPS.</span></div>
      <div class="gloss-item"><span class="gloss-term">P/FCF</span><span class="gloss-def">Price-to-Free-Cash-Flow ratio. Lower = cheaper on a cash basis. Harder to manipulate than P/E.</span></div>
      <div class="gloss-item"><span class="gloss-term">EV/EBITDA</span><span class="gloss-def">Enterprise Value &divide; EBITDA. Capital-structure-neutral valuation metric. Allows comparison regardless of debt levels.</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Report Columns &mdash; Growth Rates</div>
      <div class="gloss-item"><span class="gloss-term">EPS Gth QoQ</span><span class="gloss-def">Most recent quarter EPS growth, year-over-year (%). Freshest earnings signal.</span></div>
      <div class="gloss-item"><span class="gloss-term">Rev Growth</span><span class="gloss-def">Forward-blended revenue growth rate (%). Combines quarterly and trailing data with trend awareness.</span></div>
      <div class="gloss-item"><span class="gloss-term">Rev Gth QoQ</span><span class="gloss-def">Most recent quarter revenue growth, year-over-year (%). Freshest top-line signal.</span></div>
      <div class="gloss-item"><span class="gloss-term">Rev QoQ</span><span class="gloss-def">Sequential quarter-over-quarter revenue growth (%). Shows whether revenue is expanding each quarter.</span></div>
      <div class="gloss-item"><span class="gloss-term">EPS QoQ</span><span class="gloss-def">Sequential quarter-over-quarter EPS growth (%). Shows whether earnings are expanding each quarter.</span></div>
      <div class="gloss-item"><span class="gloss-term">GP Growth</span><span class="gloss-def">Gross profit growth, TTM year-over-year (%). Indicates margin expansion potential.</span></div>
      <div class="gloss-item"><span class="gloss-term">NI Growth</span><span class="gloss-def">Net income growth, TTM year-over-year (%). Bottom-line profit trajectory.</span></div>
      <div class="gloss-item"><span class="gloss-term">FCF Growth</span><span class="gloss-def">Free cash flow growth, TTM year-over-year (%). Tracks cash generation improvement.</span></div>
      <div class="gloss-item"><span class="gloss-term">EBITDA Gth</span><span class="gloss-def">EBITDA growth, TTM year-over-year (%). Operational leverage indicator before interest, taxes, and amortisation.</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Report Columns &mdash; Quality &amp; Margins</div>
      <div class="gloss-item"><span class="gloss-term">R&amp;D %</span><span class="gloss-def">Research &amp; development spend as a percentage of revenue. Indicates investment in future growth.</span></div>
      <div class="gloss-item"><span class="gloss-term">Gross Margin</span><span class="gloss-def">(Revenue &minus; COGS) &divide; Revenue. Measures pricing power and production efficiency.</span></div>
      <div class="gloss-item"><span class="gloss-term">Op Margin</span><span class="gloss-def">Operating income &divide; Revenue. Measures core operational profitability before interest and taxes.</span></div>
      <div class="gloss-item"><span class="gloss-term">ROE</span><span class="gloss-def">Return on Equity. Net income &divide; shareholders&rsquo; equity. Measures profit generated per dollar of equity.</span></div>
      <div class="gloss-item"><span class="gloss-term">ROIC</span><span class="gloss-def">Return on Invested Capital. Measures efficiency of ALL capital (debt + equity). Best single quality metric &mdash; shows how well management allocates capital.</span></div>
      <div class="gloss-item"><span class="gloss-term">ROA</span><span class="gloss-def">Return on Assets. Net income &divide; total assets. How efficiently a company uses its asset base.</span></div>
      <div class="gloss-item"><span class="gloss-term">Curr Ratio</span><span class="gloss-def">Current assets &divide; current liabilities. Liquidity health: &gt;2.0 strong, &gt;1.0 adequate, &lt;1.0 risk.</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Report Columns &mdash; Momentum &amp; Technical</div>
      <div class="gloss-item"><span class="gloss-term">Earnings In</span><span class="gloss-def">Days until the next earnings release. &le;7 days highlighted as upcoming catalyst.</span></div>
      <div class="gloss-item"><span class="gloss-term">Perf 1M / 3M / 6M</span><span class="gloss-def">Stock price return over the past 1, 3, or 6 months (%). Green = positive, red = negative.</span></div>
      <div class="gloss-item"><span class="gloss-term">Momentum</span><span class="gloss-def">Technical Signal pillar score (0&ndash;20) shown as a mini-bar. Composite of TV rating, RSI, MoneyFlow, Stochastic, price performance, and 52-week position.</span></div>
      <div class="gloss-item"><span class="gloss-term">TV Signal</span><span class="gloss-def">TradingView composite technical rating. Aggregates 26 indicators into: Strong Buy, Buy, Neutral, Sell, Strong Sell.</span></div>
      <div class="gloss-item"><span class="gloss-term">RSI</span><span class="gloss-def">Relative Strength Index (14-period). &gt;70 = overbought, &lt;30 = oversold. 50&ndash;65 is the healthy momentum zone.</span></div>
      <div class="gloss-item"><span class="gloss-term">MoneyFlow</span><span class="gloss-def">Money Flow Index. Volume-weighted RSI (0&ndash;100). &gt;70 = strong institutional buying, &lt;30 = strong selling.</span></div>
      <div class="gloss-item"><span class="gloss-term">Stochastic %K</span><span class="gloss-def">Stochastic oscillator. Compares closing price to its range. Bullish when %K crosses above %D between 20&ndash;80.</span></div>
      <div class="gloss-item"><span class="gloss-term">Williams %R</span><span class="gloss-def">Williams Percent Range. 0 = overbought, &minus;100 = oversold. Healthy momentum sits between &minus;50 and &minus;20.</span></div>
      <div class="gloss-item"><span class="gloss-term">ADX</span><span class="gloss-def">Average Directional Index. Measures trend strength (not direction). &gt;25 = strong trend. Direction from +DI vs &minus;DI.</span></div>
      <div class="gloss-item"><span class="gloss-term">MACD</span><span class="gloss-def">Moving Average Convergence Divergence. Histogram &gt;0 with line above signal = bullish crossover (accumulation). Below = bearish (distribution).</span></div>
      <div class="gloss-item"><span class="gloss-term">EMA 13 / 50 / 200</span><span class="gloss-def">Exponential Moving Averages over 13, 50, and 200 days. Price below an EMA signals a pullback or downtrend at that timeframe. Use the EMA filter dropdown to find dip-buying opportunities in otherwise strong names.</span></div>
      <div class="gloss-item"><span class="gloss-term">Below EMA 13</span><span class="gloss-def">Short-term pullback &mdash; stock is dipping below its 13-day EMA. Often a brief retracement in an uptrend.</span></div>
      <div class="gloss-item"><span class="gloss-term">Below EMA 50</span><span class="gloss-def">Medium-term weakness &mdash; stock has broken below its 50-day EMA. May indicate a deeper correction or trend shift.</span></div>
      <div class="gloss-item"><span class="gloss-term">Below EMA 200</span><span class="gloss-def">Long-term downtrend territory &mdash; stock is trading below its 200-day EMA. Widely watched as a bull/bear dividing line.</span></div>
      <div class="gloss-item"><span class="gloss-term">Below EMA 13 &amp; 50</span><span class="gloss-def">Stock is below both short and medium-term EMAs. Confirms the pullback has more substance than a one-day dip.</span></div>
      <div class="gloss-item"><span class="gloss-term">Below EMA 13, 50 &amp; 200</span><span class="gloss-def">Deepest pullback &mdash; stock is below all three EMAs. Maximum discount but also maximum trend risk. Best combined with high Growth Score and clean flags.</span></div>
      <div class="gloss-item"><span class="gloss-term">RSI Filter</span><span class="gloss-def">Filters stocks by RSI(14) zone. Use the dropdown to isolate oversold dip-buy candidates or avoid overbought names.</span></div>
      <div class="gloss-item"><span class="gloss-term">Oversold (RSI &lt; 30)</span><span class="gloss-def">Classic oversold territory. Stock has seen heavy selling and may be due for a bounce. Strongest contrarian signal.</span></div>
      <div class="gloss-item"><span class="gloss-term">Oversold&ndash;Weak (RSI &lt; 40)</span><span class="gloss-def">Broader oversold zone including stocks with bearish momentum that haven&rsquo;t yet reached extreme levels. Wider net for dip candidates.</span></div>
      <div class="gloss-item"><span class="gloss-term">Neutral (RSI 40&ndash;60)</span><span class="gloss-def">Neither overbought nor oversold. Stock is in a consolidation or steady-trend phase with no extreme momentum signal.</span></div>
      <div class="gloss-item"><span class="gloss-term">Strong&ndash;Overbought (RSI &gt; 60)</span><span class="gloss-def">Stock has strong upward momentum. May continue trending higher but risk of a pullback increases as RSI rises.</span></div>
      <div class="gloss-item"><span class="gloss-term">Overbought (RSI &gt; 70)</span><span class="gloss-def">Classic overbought territory. Price may be extended and vulnerable to a correction or consolidation.</span></div>
      <div class="gloss-item"><span class="gloss-term">Extreme Overbought (RSI &gt; 80)</span><span class="gloss-def">Extremely overbought. Rare and usually unsustainable. High risk of a sharp pullback in the near term.</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Report Columns &mdash; Sentiment &amp; Accumulation</div>
      <div class="gloss-item"><span class="gloss-term">Sentiment</span><span class="gloss-def">Composite sentiment label derived from TV rating, RSI, 1M/3M price momentum, 52-week position, and EPS acceleration. Ranges from Bearish to Bullish. Hover for signal breakdown.</span></div>
      <div class="gloss-item"><span class="gloss-term">Sent. Score</span><span class="gloss-def">Sentiment score (0&ndash;100). Starts at 50 (neutral), adjusted by each signal. &ge;70 = Bullish, &le;30 = Bearish.</span></div>
      <div class="gloss-item"><span class="gloss-term">Accumulation</span><span class="gloss-def">Proxy for institutional buying pressure. Combines MACD crossovers, momentum, volume trends (relative + 30d vs 90d), ADX directional indicators, and MoneyFlow. Hover for full breakdown.</span></div>
      <div class="gloss-item"><span class="gloss-term">Accum. Score</span><span class="gloss-def">Accumulation score (0&ndash;100). &ge;75 = Strong Accumulation, &ge;60 = Accumulation, &le;30 = Strong Distribution.</span></div>
      <div class="gloss-item"><span class="gloss-term">Closely Held %</span><span class="gloss-def">(Total shares &minus; float shares) &divide; total shares. Captures insider, restricted, and strategic ownership not freely tradeable. High % = insider conviction.</span></div>
      <div class="gloss-item"><span class="gloss-term">Analyst Target</span><span class="gloss-def">Wall Street analyst consensus 12-month price target (from TradingView).</span></div>
      <div class="gloss-item"><span class="gloss-term">Analyst Upside</span><span class="gloss-def">Percentage difference between analyst consensus target and current price.</span></div>
      <div class="gloss-item"><span class="gloss-term">Mkt Cap</span><span class="gloss-def">Market capitalisation. Current share price &times; total shares outstanding. Displayed in billions (B) or trillions (T).</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Risk Flags</div>
      <div class="gloss-item"><span class="gloss-term">UNPROFITABLE</span><span class="gloss-def">TTM diluted EPS &le; 0. Company is not currently profitable on a per-share basis.</span></div>
      <div class="gloss-item"><span class="gloss-term">NEG FCF</span><span class="gloss-def">Free cash flow &lt; 0. Company is burning cash rather than generating it.</span></div>
      <div class="gloss-item"><span class="gloss-term">NEG MARGIN</span><span class="gloss-def">Operating margin &lt; 0. Core operations are losing money before interest and taxes.</span></div>
      <div class="gloss-item"><span class="gloss-term">HIGH DEBT</span><span class="gloss-def">Debt-to-equity ratio &gt; 3.0 (excludes Financials, Real Estate, Utilities where leverage is structural).</span></div>
      <div class="gloss-item"><span class="gloss-term">REV DECLINE</span><span class="gloss-def">Forward-blended revenue growth &lt; 0%. Top line is shrinking.</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Upside Tiers</div>
      <div class="gloss-item"><span class="gloss-term">Exceptional</span><span class="gloss-def">&ge;60% upside to base target. Highest-conviction growth opportunities.</span></div>
      <div class="gloss-item"><span class="gloss-term">Strong</span><span class="gloss-def">30&ndash;60% upside. Solid growth with meaningful room to run.</span></div>
      <div class="gloss-item"><span class="gloss-term">Moderate</span><span class="gloss-def">10&ndash;30% upside. Reasonable growth at fair-ish prices.</span></div>
      <div class="gloss-item"><span class="gloss-term">Fairly Priced</span><span class="gloss-def">&minus;5% to +10% upside. Market has largely priced in the growth.</span></div>
      <div class="gloss-item"><span class="gloss-term">Stretched</span><span class="gloss-def">&gt;5% downside. Trading above model fair value.</span></div>
    </div>

    <div class="gloss-section">
      <div class="gloss-section-title">Common Abbreviations</div>
      <div class="gloss-item"><span class="gloss-term">TTM</span><span class="gloss-def">Trailing Twelve Months. Sum of the last four reported quarters.</span></div>
      <div class="gloss-item"><span class="gloss-term">FY</span><span class="gloss-def">Fiscal Year. A company&rsquo;s full annual reporting period (may not align with calendar year).</span></div>
      <div class="gloss-item"><span class="gloss-term">FQ</span><span class="gloss-def">Fiscal Quarter. One quarter of the fiscal year.</span></div>
      <div class="gloss-item"><span class="gloss-term">YoY</span><span class="gloss-def">Year-over-Year. Compares a metric to the same period one year ago.</span></div>
      <div class="gloss-item"><span class="gloss-term">QoQ</span><span class="gloss-def">Quarter-over-Quarter. Compares a metric to the immediately prior quarter (sequential growth).</span></div>
      <div class="gloss-item"><span class="gloss-term">EPS</span><span class="gloss-def">Earnings Per Share. Net income &divide; diluted shares outstanding.</span></div>
      <div class="gloss-item"><span class="gloss-term">FCF</span><span class="gloss-def">Free Cash Flow. Operating cash flow &minus; capital expenditures. Cash available for dividends, buybacks, or reinvestment.</span></div>
      <div class="gloss-item"><span class="gloss-term">EBITDA</span><span class="gloss-def">Earnings Before Interest, Taxes, Depreciation &amp; Amortisation. Proxy for operating cash generation.</span></div>
      <div class="gloss-item"><span class="gloss-term">EV</span><span class="gloss-def">Enterprise Value. Market cap + total debt &minus; cash. Represents the total cost to acquire a business.</span></div>
      <div class="gloss-item"><span class="gloss-term">EMA</span><span class="gloss-def">Exponential Moving Average. A moving average that gives more weight to recent prices. Faster to react to price changes than a simple moving average (SMA).</span></div>
      <div class="gloss-item"><span class="gloss-term">P/E</span><span class="gloss-def">Price-to-Earnings ratio. Share price &divide; EPS. Higher = market expects more growth.</span></div>
      <div class="gloss-item"><span class="gloss-term">PEG</span><span class="gloss-def">Price/Earnings-to-Growth. P/E &divide; earnings growth rate. Normalises valuation for growth speed. &lt;1.0 considered attractive.</span></div>
      <div class="gloss-item"><span class="gloss-term">COGS</span><span class="gloss-def">Cost of Goods Sold. Direct costs attributable to producing goods/services sold.</span></div>
      <div class="gloss-item"><span class="gloss-term">GICS</span><span class="gloss-def">Global Industry Classification Standard. The sector/industry taxonomy used by S&amp;P and MSCI.</span></div>
      <div class="gloss-item"><span class="gloss-term">TV</span><span class="gloss-def">TradingView. The financial data platform used as the data source for this screener.</span></div>
      <div class="gloss-item"><span class="gloss-term">MRQ</span><span class="gloss-def">Most Recent Quarter. The latest reported fiscal quarter.</span></div>
      <div class="gloss-item"><span class="gloss-term">R&amp;D</span><span class="gloss-def">Research &amp; Development. Spending on innovation and future products/services.</span></div>
      <div class="gloss-item"><span class="gloss-term">D/E</span><span class="gloss-def">Debt-to-Equity ratio. Total debt &divide; shareholders&rsquo; equity. Measures financial leverage.</span></div>
    </div>

  </div>
</div>

__WL_BAR__<script>{JS}__WL_JS__</script>
</body>
</html>"""

    page = (page
            .replace("__WL_CSS__", _WL_CSS)
            .replace("__WL_BAR__", _WL_BAR)
            .replace("__WL_JS__",  _wl_js(suite_port)))
    return page


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Screen an index for high-growth stocks."
    )
    parser.add_argument("--index",        default="SPX",
                        choices=["SPX", "NDX", "RUT", "TSX", "SPXRUT"],
                        help="Index to screen (default: SPX)")
    parser.add_argument("--top",          type=int, default=None,
                        help="Limit to top N stocks by score")
    parser.add_argument("--csv",          action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--backtest",     action="store_true",
                        help="Run point-in-time backtest on top stocks")
    parser.add_argument("--backtest-days", dest="backtest_days",
                        type=int, default=None,
                        help="Number of days for backtest price history")
    args = parser.parse_args()
    index_code    = args.index
    limit         = args.top
    write_csv_flag = args.csv
    use_pit_backtest = args.backtest
    backtest_days = args.backtest_days if args.backtest_days is not None else 500

    index_name = INDEX_CONFIG.get(index_code, INDEX_CONFIG["SPX"])["name"]
    print("\n" + "="*60)
    print(f"  GROWTH & MOMENTUM SCREENER — {index_name}")
    print("="*60)

    print("\n[1/3] Fetching stocks...")
    df = fetch_stocks(index_code)

    print("\n[2/3] Scoring stocks...")
    results, skipped = [], 0
    for _, row in df.iterrows():
        try:
            d = parse_row(row)
            r = score_stock(d)
            if r: results.append(r)
            else: skipped += 1
        except Exception:
            skipped += 1

    print(f"  Scored: {len(results)} | Skipped: {skipped}")
    if not results:
        print("ERROR: No stocks scored."); sys.exit(1)

    n_flagged = sum(1 for r in results if r["flags"])
    print(f"  Flagged (unprofitable/risk): {n_flagged} | Clean: {len(results)-n_flagged}")

    print("\n  Tier Distribution:")
    for name, lo, hi, _ in TIERS:
        stocks = [r for r in results if r["tier"] == name]
        if stocks:
            print(f"    {name.ljust(25)} {str(len(stocks)).rjust(3)}  {'█'*min(len(stocks),50)}")

    results.sort(key=lambda x: -x["growth_score"])
    if limit:
        results = results[:limit]
        print(f"\n  Trimmed to top {limit} by growth score ({len(results)} stocks in report).")

    print("\n  Top 15 by Growth Score:")
    top15 = results[:15]
    for i, r in enumerate(top15, 1):
        flags = f" [{','.join(r['flags'])}]" if r["flags"] else ""
        eps_src = r.get('eps_source','?')[:8]
        fwd_g   = r.get('peg_growth_used')
        print(f"    {i:>2}. {r['ticker']:<6}  Score={fsc(r['growth_score']):>4}  "
              f"Upside={fpc(r['upside_pct']):>7}  "
              f"FwdG={fpc(fwd_g, False):>7}  "
              f"PEG={fx(r['peg_current']):>6}  "
              f"Src={eps_src}{flags}")

    # ── Top-3 column: PIT backtest or fast consensus ─────────────────────────
    # Step 1 (both modes): fetch yfinance fundamentals to improve method accuracy
    if _YF_AVAIL:
        _load_yf_cache()   # warm in-memory cache from disk before spawning workers
        print("\n  Enriching valuations with yfinance fundamentals (FCF, revenue, margins)...")
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_fetch_yf_data_gs, r["ticker"]): r["ticker"]
                       for r in results}
            bars_map = {}
            for fut in as_completed(futures):
                tk = futures[fut]
                try:    bars_map[tk] = fut.result()
                except Exception: bars_map[tk] = {"fcf": None, "ebitda": None, "fwd_rev": None,
                                         "revenue": None, "total_debt": None,
                                         "cash": None, "gross_margin": None,
                                         "fwd_eps": None}
        _save_yf_cache()   # persist any newly-fetched entries for next run
        n_fetched = sum(1 for v in bars_map.values()
                        if any(v.get(k) is not None
                               for k in ("fcf", "revenue", "gross_margin")))
        print(f"  Fundamentals: {n_fetched}/{len(results)} stocks fetched")
    else:
        bars_map = {}

    if use_pit_backtest and _BT_ENGINE_AVAIL:
        # ── Point-in-time MAPE backtest (same engine as valuationMaster) ─────
        # Only backtest the top N stocks by growth score — expensive per ticker.
        # limit controls both the initial fetch (--top) and the backtest scope.
        bt_results = results   # already sorted and trimmed to top N by growth score
        n = len(bt_results)
        est_min = round(n * 14 / 5 / 60, 1)   # ~14s/stock, 5 workers
        print(f"\n[3/4] Running point-in-time backtest on top {n} stocks "
              f"({backtest_days} days, est. ~{est_min} min with 5 workers)...")
        print("  Uses yfinance prices + SEC EDGAR + yfinance quarterly TTM")
        # 5 workers: EDGAR rate-limits at 10 req/s; 5 parallel is safe
        with ThreadPoolExecutor(max_workers=5) as pool:
            def _pit_task(r):
                yf_data = bars_map.get(r["ticker"], {})
                vm = _build_vm_dict(r, yf_data)
                return r["ticker"], _run_pit_backtest_gs(r["ticker"], vm, days=backtest_days)
            pit_futures = {pool.submit(_pit_task, r): r["ticker"] for r in bt_results}
            pit_map = {}
            done = 0
            # Per-ticker running timeout — each future gets _PIT_PER_TICKER_SEC
            # seconds from the moment its worker thread actually STARTS RUNNING.
            # We poll every 30 s so the UI stays responsive and stuck workers are
            # evicted quickly instead of blocking until a global deadline.
            _PIT_PER_TICKER_SEC = 90  # ~enough for Stooq + EDGAR + yfinance quarterly
            _pit_run_since: dict = {}  # fut → timestamp when first seen as running
            pending = set(pit_futures.keys())
            while pending:
                done_set, _ = _cf_wait(pending, timeout=30, return_when=_CF_FIRST)
                pending -= done_set
                for fut in done_set:
                    tk = pit_futures[fut]
                    done += 1
                    try:
                        _, res = fut.result()
                        pit_map[tk] = res
                    except Exception:
                        pit_map[tk] = (None, [], [])
                    if done % 5 == 0 or done == n:
                        print(f"  [{done}/{n}] done...", end="\r")
                # Evict futures that have been RUNNING (not just queued) too long
                now = _time.time()
                to_expire = []
                for fut in list(pending):
                    if fut.running():
                        if fut not in _pit_run_since:
                            _pit_run_since[fut] = now
                        elif now - _pit_run_since[fut] > _PIT_PER_TICKER_SEC:
                            to_expire.append(fut)
                for fut in to_expire:
                    pending.discard(fut)
                    tk = pit_futures[fut]
                    done += 1
                    print(f"\n  [{done}/{n}] {tk}: timed out — skipping")
                    pit_map[tk] = (None, [], [])
        print()  # newline after progress
        for r in results:
            mean_fv, methods, indiv_vals = pit_map.get(r["ticker"], (None, [], []))
            r["best3_mean"]    = mean_fv
            r["best3_methods"] = methods
            r["best3_upside"]  = ((mean_fv - r["price"]) / r["price"] * 100
                                  if mean_fv and r["price"] else None)
            # Replace bear/base/bull with backtest-derived values:
            #   BEAR = min of top-3 method values (most conservative)
            #   BASE = mean of top-3 method values
            #   BULL = max of top-3 method values (most optimistic)
            # The spread is the natural disagreement between historically-
            # accurate methods rather than a synthetic ±20% haircut.
            #
            # Price-relative cap (same limits as non-backtest path): prevents
            # methods like ERG/TAM from inflating bear/base/bull when the most
            # recent snapshot has an anomalous blowout quarter. The MAPE ranking
            # itself is unaffected — only the display values are bounded.
            if indiv_vals and r.get("price"):
                price = r["price"]
                capped = [min(v, price * 3.0) for v in indiv_vals]
                capped_mean = sum(capped) / len(capped)
                r["target_bear"]  = round(min(capped), 2)
                r["target_base"]  = round(capped_mean, 2)
                r["target_bull"]  = round(max(capped), 2)
                r["upside_pct"]   = round((capped_mean - price) / price * 100, 1)
                # Keep BACKTEST TOP-3 column consistent with base target
                r["best3_mean"]   = round(capped_mean, 2)
                r["best3_upside"] = r["upside_pct"]
        best3_label = "BACKTEST TOP-3"
        best3_tooltip = ("Top-3 valuation methods by point-in-time MAPE backtest — "
                         "same engine as valuationMaster (yfinance prices, SEC EDGAR + "
                         "yfinance quarterly fundamentals). Methods and fair values "
                         "match what valuationMaster would show for these stocks.")

    elif _YF_AVAIL:
        # ── Fast consensus ranking (default, ~30s) ────────────────────────────
        print("\n  Using consensus ranking (run with --backtest for full PIT engine)")
        for r in results:
            yf_data = bars_map.get(r["ticker"], {})
            vm  = _build_vm_dict(r, yf_data)
            fvs = _run_gs_valuations(vm)
            if fvs:
                mean_fv, methods = _consensus_top3_gs(fvs, price=r.get("price"))
                r["best3_mean"]    = mean_fv
                r["best3_methods"] = methods
                r["best3_upside"]  = ((mean_fv - r["price"]) / r["price"] * 100
                                      if mean_fv and r["price"] else None)
            else:
                r["best3_mean"]    = None
                r["best3_methods"] = []
                r["best3_upside"]  = None
        best3_label   = "CONSENSUS TOP-3"
        best3_tooltip = ("Mean of the 3 valuation methods closest to the ensemble "
                         "median. Run with --backtest for full point-in-time MAPE "
                         "ranking (same engine as valuationMaster).")

    else:
        for r in results:
            r["best3_mean"]    = None
            r["best3_methods"] = []
            r["best3_upside"]  = None
        best3_label   = "CONSENSUS TOP-3"
        best3_tooltip = "yfinance not available — column disabled."

    # ── Sentiment fetch ───────────────────────────────────────────────────────
    _report_step = "[4/4]" if (use_pit_backtest and _BT_ENGINE_AVAIL) else "[3/3]"
    print(f"\n{_report_step} Building report...")
    ts = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    if use_pit_backtest and _BT_ENGINE_AVAIL:
        ts += f" · Backtest {backtest_days}d"
    suite_port = int(os.environ.get("VALUATION_SUITE_PORT", "5050"))
    html = build_html(results, ts, len(results), index_name,
                      best3_label=best3_label, best3_tooltip=best3_tooltip,
                      suite_port=suite_port)

    index_slug = index_code.lower()  # e.g. "spx", "ndx", "tsx"
    date_str   = datetime.datetime.now().strftime("%Y_%m_%d")
    bt_suffix  = f"_bt{backtest_days}d" if (use_pit_backtest and _BT_ENGINE_AVAIL) else ""
    os.makedirs("growthData", exist_ok=True)
    out     = os.path.join("growthData", f"{date_str}_growth_report_{index_slug}{bt_suffix}.html")
    out_csv = os.path.join("growthData", f"{date_str}_growth_report_{index_slug}{bt_suffix}.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved: {out}")

    if write_csv_flag:
        write_csv(results, out_csv)
        print(f"  CSV saved:  {out_csv}")

    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        webbrowser.open("file://" + os.path.abspath(out))
    print(f"\nDone. ({len(results)} stocks scored)")
    print("="*60)


if __name__ == "__main__":
    main()
