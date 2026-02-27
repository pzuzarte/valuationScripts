"""
Valuation Screener — Powered by TradingView Screener
=====================================================
Identifies undervalued stocks using DCF, P/E, P/FCF, EV/EBITDA, and PEG
analysis. Benchmarks each stock against live sector medians fetched from
TradingView. Flags value traps and scores each stock 0-100.

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
                      python valueScreener.py --index SPX
                      python valueScreener.py --index NDX
                      python valueScreener.py --index TSX

  --top    N        Limit fetch to top N stocks by market cap.
                    Useful for quick tests.
                    Example:
                      python valueScreener.py --top 50

  --csv             Also export results to a CSV file alongside the HTML report.
                    Example:
                      python valueScreener.py --csv
                      python valueScreener.py --index NDX --csv

  --help            Print this help message and exit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python valueScreener.py                        # S&P 500, full scan
  python valueScreener.py --index NDX            # Nasdaq 100
  python valueScreener.py --index TSX --csv      # TSX + CSV export
  python valueScreener.py --index RUT --top 200  # Russell 2000, top 200
"""

import sys, math, datetime, webbrowser, os, csv, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tradingview_screener import Query, col
except ImportError:
    print("ERROR: tradingview-screener not installed.")
    print("Run:  pip install tradingview-screener")
    sys.exit(1)

from valuation_models import (
    MARGIN_OF_SAFETY,
    growth_adjusted_multiples,
    compute_peg,
    get_value_trap_flags,
    compute_value_rank_score,
    run_three_stage_dcf,
    run_fcf_yield,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
RISK_FREE_RATE       = 0.043
EQUITY_RISK_PREMIUM  = 0.055
TERMINAL_GROWTH      = 0.03
PROJECTION_YEARS     = 5
CONVERGENCE_THRESH   = 0.15

# Value-trap exclusion thresholds
MAX_DEBT_EQUITY_NONFIN = 3.0    # D/E above this → flag (non-financial sectors)
MIN_ROE_QUALITY        = 0.15   # ROE >= 15% → "quality" company
MIN_OP_MARGIN          = 0.0    # operating margin must be positive to score

# Income / dividend sectors where yield matters
INCOME_SECTORS = {"Utilities", "Real Estate", "Consumer Staples", "Energy"}

TIERS = [
    # discount = (fair_value - price) / price * 100
    ("SIGNIFICANTLY OVERVALUED",  -999, -30,  "#e53e3e"),
    ("OVERVALUED",                 -30, -15,  "#fc8181"),
    ("MODESTLY OVERVALUED",        -15,  -5,  "#f6ad55"),
    ("FAIRLY VALUED",               -5,   5,  "#718096"),
    ("MODESTLY UNDERVALUED",         5,  15,  "#4fd4a0"),
    ("UNDERVALUED",                 15,  30,  "#00c896"),
    ("DEEPLY UNDERVALUED",          30, 999,  "#00e5a0"),
]

FIELDS = [
    "name", "close", "market_cap_basic",
    "total_shares_outstanding", "float_shares_outstanding",
    "total_debt", "free_cash_flow", "beta_1_year",
    "earnings_per_share_diluted_ttm", "earnings_per_share_basic_ttm",
    "enterprise_value_ebitda_ttm", "gross_margin",
    "net_income", "total_revenue", "operating_margin",
    "price_earnings_ttm", "debt_to_equity", "sector",
    "total_revenue_yoy_growth_ttm",              # revenue growth YoY (TTM)
    "earnings_per_share_diluted_yoy_growth_ttm", # EPS diluted growth YoY (TTM)
    "return_on_equity",                          # ROE (TTM)
    "dividend_yield_recent",                     # dividend yield (forward)
    "price_52_week_high",                        # 52-wk high
    "price_52_week_low",                         # 52-wk low
    "Perf.1M",                                   # 1-month price performance
    "Perf.3M",                                   # 3-month price performance
    "enterprise_value_fq",                       # live enterprise value (MRQ)
]

# v3 fields that are fetched in a separate resilient pass.
# If any field name is wrong for the current TV API version, the main fetch
# still succeeds and these columns just show as missing (—) in the report.
FIELDS_V3 = [
    "return_on_invested_capital",          # ROIC (TTM)
    "cash_n_short_term_invest_fq",         # cash + ST investments (MRQ) — confirmed field name
    "price_book_fq",                       # price-to-book (MRQ)
    "earnings_per_share_forecast_next_fy", # analyst consensus EPS forecast — next fiscal year
    "revenue_forecast_next_fy",            # analyst consensus revenue forecast — next fiscal year
]

# Core fields (without v3 extras) — always used in the primary fetch
FIELDS_CORE = [f for f in FIELDS if f not in FIELDS_V3]


# ── SECTOR DEFAULTS ───────────────────────────────────────────────────────────
SECTOR_DEFAULTS = {
    "Technology":             {"pe": 28.0, "pfcf": 28.0, "ev_ebitda": 20.0, "growth_cap": 0.15},
    "Communication Services": {"pe": 20.0, "pfcf": 20.0, "ev_ebitda": 13.0, "growth_cap": 0.10},
    "Consumer Discretionary": {"pe": 22.0, "pfcf": 20.0, "ev_ebitda": 14.0, "growth_cap": 0.10},
    "Consumer Staples":       {"pe": 18.0, "pfcf": 18.0, "ev_ebitda": 13.0, "growth_cap": 0.07},
    "Health Care":            {"pe": 22.0, "pfcf": 20.0, "ev_ebitda": 14.0, "growth_cap": 0.10},
    "Financials":             {"pe": 13.0, "pfcf": 13.0, "ev_ebitda": 10.0, "growth_cap": 0.10},
    "Industrials":            {"pe": 20.0, "pfcf": 18.0, "ev_ebitda": 13.0, "growth_cap": 0.09},
    "Materials":              {"pe": 16.0, "pfcf": 15.0, "ev_ebitda": 10.0, "growth_cap": 0.08},
    "Energy":                 {"pe": 12.0, "pfcf": 10.0, "ev_ebitda":  8.0, "growth_cap": 0.07},
    "Utilities":              {"pe": 16.0, "pfcf": 15.0, "ev_ebitda": 10.0, "growth_cap": 0.05},
    "Real Estate":            {"pe": 30.0, "pfcf": 20.0, "ev_ebitda": 18.0, "growth_cap": 0.06},
}
MARKET_DEFAULTS = {"pe": 22.0, "pfcf": 22.0, "ev_ebitda": 14.0, "growth_cap": 0.10}


# ── INDEX CONSTITUENTS ────────────────────────────────────────────────────────
# Hardcoded ticker lists — used as primary filter via col("name").isin()
# Russell 2000 uses exchange fallback (too large / frequently changing)

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
    "SPX": {"name": "S&P 500",      "tickers": SP500},
    "NDX": {"name": "Nasdaq 100",   "tickers": NASDAQ100},
    "RUT": {"name": "Russell 2000", "tickers": None},
    "TSX": {"name": "TSX",          "tickers": TSX},
}


# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_stocks(index_code, limit=None):
    cfg  = INDEX_CONFIG.get(index_code.upper(), INDEX_CONFIG["SPX"])
    name = cfg["name"]
    print(f"  Querying TradingView for {name}...")

    tickers = cfg.get("tickers")

    import pandas as pd

    def _merge_v3(df):
        """
        Attempt a second pass to fetch v3 fields (ROIC, cash, P/B, buyback).
        Uses only the 'name' column to join back onto the main dataframe.
        If any field name is invalid the entire pass is skipped gracefully —
        the main fetch has already succeeded, so the run continues normally.
        """
        try:
            tickers_fetched = df["name"].tolist()
            _, v3df = (
                Query().select("name", *FIELDS_V3)
                .where(col("name").isin(tickers_fetched))
                .limit(len(tickers_fetched) + 20).get_scanner_data()
            )
            df = df.merge(v3df[["name"] + FIELDS_V3], on="name", how="left")
            print(f"  v3 fields (ROIC, cash, P/B, buyback) merged OK")
        except Exception as e:
            print(f"  v3 fields unavailable ({str(e)[:80]}), continuing without them")
        return df

    # ── Primary: isin filter on known constituents ────────────────────────────
    if tickers:
        try:
            lim = limit or len(tickers) + 20
            _, df = (
                Query().select(*FIELDS_CORE)
                .where(col("name").isin(tickers), col("is_primary") == True)
                .order_by("market_cap_basic", ascending=False)
                .limit(lim).get_scanner_data()
            )
            df = _merge_v3(df)
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
                Query().select(*FIELDS_CORE)
                .where(col("country") == "Canada", col("is_primary") == True,
                       col("typespecs").has_none_of("preferred"))
                .order_by("market_cap_basic", ascending=False)
                .limit(400).get_scanner_data()
            )
            df = _merge_v3(df)
            df["index_label"] = name
            print(f"  {name}: {len(df)} stocks (country fallback)")
            return df
        except Exception as e:
            print(f"  TSX country filter failed: {str(e)[:60]}")
    else:
        cap = limit or (503 if index_code == "SPX" else 110 if index_code == "NDX" else 2000)
        for exch in ["NYSE", "NASDAQ"]:
            try:
                _, df = (
                    Query().select(*FIELDS_CORE)
                    .where(col("exchange") == exch, col("is_primary") == True,
                           col("typespecs").has_none_of("preferred"))
                    .order_by("market_cap_basic", ascending=False)
                    .limit(1500).get_scanner_data()
                )
                frames.append(df)
                print(f"    {exch}: {len(df)} stocks")
            except Exception as e:
                print(f"    {exch} failed: {str(e)[:60]}")

        if frames:
            combined = pd.concat([f.reset_index(drop=True) for f in frames], ignore_index=True)
            combined = combined.drop_duplicates(subset=["name"], keep="first")
            combined = combined.sort_values("market_cap_basic", ascending=False).reset_index(drop=True)
            if index_code == "RUT":
                combined = combined.iloc[503:2503].reset_index(drop=True)
            else:
                combined = combined.head(cap).reset_index(drop=True)
            combined = _merge_v3(combined)
            combined["index_label"] = name
            print(f"  {name}: {len(combined)} stocks (exchange fallback)")
            return combined

    # ── Last resort ───────────────────────────────────────────────────────────
    print("  All filters failed — using large-cap global fallback")
    _, df = (
        Query().select(*FIELDS_CORE)
        .where(col("market_cap_basic") > 8e9, col("is_primary") == True,
               col("typespecs").has_none_of("preferred"), col("close") > 0)
        .order_by("market_cap_basic", ascending=False)
        .limit(503).get_scanner_data()
    )
    df = _merge_v3(df)
    df["index_label"] = name
    return df


def fetch_sp500(limit):
    """Legacy wrapper."""
    return fetch_stocks("SPX", limit)


def fetch_benchmarks():
    """
    Returns (bm_market, bm_sector).
    Per-sector medians for P/E, P/FCF, EV/EBITDA, and average growth cap.
    Falls back to SECTOR_DEFAULTS when live data is sparse.
    """
    bm_market = dict(MARKET_DEFAULTS)
    bm_sector = {}
    try:
        count, df = (
            Query().select("market_cap_basic", "free_cash_flow", "sector",
                           "price_earnings_ttm", "enterprise_value_ebitda_ttm",
                           "total_revenue_yoy_growth_ttm")
            .where(col("market_cap_basic") > 2e9, col("price_earnings_ttm") > 0)
            .order_by("market_cap_basic", ascending=False)
            .limit(800).get_scanner_data()
        )
        if len(df) >= 10:
            df["_pfcf"] = df["market_cap_basic"] / df["free_cash_flow"].replace(0, float("nan"))

            pe_all = df["price_earnings_ttm"].dropna()
            pe_all = pe_all[(pe_all > 5) & (pe_all < 100)]
            if len(pe_all) > 5: bm_market["pe"] = round(float(pe_all.median()), 1)
            pf_all = df["_pfcf"].dropna(); pf_all = pf_all[(pf_all > 5) & (pf_all < 100)]
            if len(pf_all) > 5: bm_market["pfcf"] = round(float(pf_all.median()), 1)
            ev_all = df["enterprise_value_ebitda_ttm"].dropna()
            ev_all = ev_all[(ev_all > 3) & (ev_all < 60)]
            if len(ev_all) > 5: bm_market["ev_ebitda"] = round(float(ev_all.median()), 1)

            for sector, grp in df.groupby("sector"):
                if not sector or len(grp) < 5: continue
                sec = str(sector)
                sb  = dict(SECTOR_DEFAULTS.get(sec, MARKET_DEFAULTS))

                pe_s = grp["price_earnings_ttm"].dropna()
                pe_s = pe_s[(pe_s > 5) & (pe_s < 100)]
                if len(pe_s) >= 4: sb["pe"] = round(float(pe_s.median()), 1)

                pf_s = grp["_pfcf"].dropna()
                pf_s = pf_s[(pf_s > 5) & (pf_s < 100)]
                if len(pf_s) >= 4: sb["pfcf"] = round(float(pf_s.median()), 1)

                ev_s = grp["enterprise_value_ebitda_ttm"].dropna()
                ev_s = ev_s[(ev_s > 3) & (ev_s < 60)]
                if len(ev_s) >= 4: sb["ev_ebitda"] = round(float(ev_s.median()), 1)

                # Sector average revenue growth → used to cap DCF growth assumption
                if "total_revenue_yoy_growth_ttm" in grp.columns:
                    g_s = grp["total_revenue_yoy_growth_ttm"].dropna()
                    g_s = g_s[(g_s > -0.5) & (g_s < 1.0)]
                    if len(g_s) >= 4:
                        avg_g = float(g_s.median())
                        # growth_cap = max of sector default and live median, bounded
                        sb["growth_cap"] = round(max(0.04, min(avg_g / 100.0, 0.25)), 3)

                bm_sector[sec] = sb

    except Exception as e:
        print(f"  Benchmark warning: {str(e)[:80]}")

    for sec, vals in SECTOR_DEFAULTS.items():
        if sec not in bm_sector:
            bm_sector[sec] = dict(vals)

    return bm_market, bm_sector


# ── PARSE ─────────────────────────────────────────────────────────────────────
def parse_row(row):
    def s(f, d=None):
        v = row.get(f, d)
        if v is None or (isinstance(v, float) and math.isnan(v)): return d
        try: return float(v)
        except: return d

    price   = s("close", 0)
    mktcap  = s("market_cap_basic", 0)
    debt    = s("total_debt", 0)
    fcf     = s("free_cash_flow")
    rev     = s("total_revenue")
    ni      = s("net_income")
    op_mar  = s("operating_margin")
    ev_eb   = s("enterprise_value_ebitda_ttm")
    eps     = s("earnings_per_share_diluted_ttm") or s("earnings_per_share_basic_ttm")
    shares  = s("total_shares_outstanding") or s("float_shares_outstanding")
    beta    = s("beta_1_year", 1.0) or 1.0
    sector  = str(row.get("sector") or "Unknown")

    rev_growth  = s("total_revenue_yoy_growth_ttm")   # percent e.g. 12.5 = 12.5% YoY
    eps_growth  = s("earnings_per_share_diluted_yoy_growth_ttm")
    roe         = s("return_on_equity")               # e.g. 22.0 = 22%
    div_yield   = s("dividend_yield_recent")          # e.g. 3.5 (percent)
    hi52        = s("price_52_week_high")
    lo52        = s("price_52_week_low")
    perf_1m     = s("Perf.1M")
    perf_3m     = s("Perf.3M")
    ev_live     = s("enterprise_value_fq")
    # v3 additions
    roic           = s("return_on_invested_capital")          # e.g. 18.5 = 18.5%
    cash_direct    = s("cash_n_short_term_invest_fq")          # cash + ST investments (MRQ)
    p_b            = s("price_book_fq")                        # price-to-book (MRQ)
    gross_margin_v = s("gross_margin")                         # gross margin % (TTM)
    eps_fwd_fy     = s("earnings_per_share_forecast_next_fy")  # forecast EPS next FY
    rev_fwd_fy     = s("revenue_forecast_next_fy")             # forecast revenue next FY

    # 52-week position: 0 = at 52wk low, 1 = at 52wk high
    pos52 = None
    if hi52 and lo52 and hi52 > lo52 and price:
        pos52 = (price - lo52) / (hi52 - lo52)

    debt_equity = s("debt_to_equity")
    fcf_margin  = (fcf / rev) if (fcf and rev and rev > 0) else None

    # ── Forward growth rate for DCF ──────────────────────────────────────────
    # Priority hierarchy (best to worst):
    #   1. Analyst forecast: implied growth from next-FY revenue vs TTM revenue
    #      blended 50/50 with implied EPS growth (forecast EPS vs TTM EPS)
    #   2. TTM actuals: 60/40 blend of reported revenue growth + EPS growth
    #   3. FCF-margin proxy (original fallback)
    def _growth_from_forecasts():
        """Derive implied growth from analyst consensus next-FY forecasts."""
        # Revenue-implied growth: (forecast_rev / ttm_rev) - 1
        rev_impl = None
        if rev_fwd_fy and rev and rev > 0:
            rev_impl = (rev_fwd_fy / rev) - 1.0
            rev_impl = max(-0.30, min(0.80, rev_impl))
        # EPS-implied growth: (forecast_eps / ttm_eps) - 1
        eps_impl = None
        if eps_fwd_fy and eps and eps > 0:
            eps_impl = (eps_fwd_fy / eps) - 1.0
            eps_impl = max(-0.30, min(0.80, eps_impl))
        if rev_impl is not None and eps_impl is not None:
            return max(0.02, rev_impl * 0.50 + eps_impl * 0.50)
        if rev_impl is not None: return max(0.02, rev_impl)
        if eps_impl is not None: return max(0.02, eps_impl)
        return None

    def _growth_from_actuals():
        """Derive growth from TTM reported revenue and EPS growth rates."""
        rg = rev_growth / 100.0 if rev_growth is not None else None
        eg = eps_growth / 100.0 if eps_growth is not None else None
        if rg is not None: rg = max(-0.30, min(0.80, rg))
        if eg is not None: eg = max(-0.30, min(0.80, eg))
        if rg is not None and eg is not None:
            return max(0.02, rg * 0.60 + eg * 0.40)
        if rg is not None: return max(0.02, rg)
        if eg is not None: return max(0.02, eg)
        return None

    fwd_growth    = _growth_from_forecasts()   # analyst forward view
    actual_growth = _growth_from_actuals()     # TTM reported actuals

    if fwd_growth is not None:
        fcf_growth = fwd_growth
        actual_growth_used = True
    elif actual_growth is not None:
        fcf_growth = actual_growth
        actual_growth_used = True
    else:
        # Fallback to FCF-margin proxy (original v1 logic)
        actual_growth_used = False
        if   fcf_margin and fcf_margin > 0.40: fcf_growth = 0.15
        elif fcf_margin and fcf_margin > 0.20: fcf_growth = 0.12
        elif fcf_margin and fcf_margin > 0.10: fcf_growth = 0.09
        else:                                  fcf_growth = 0.06

    # ── Cash: prefer direct fetch, fall back to back-calculation ─────────────
    ev_no_cash = mktcap + debt
    ebitda = None
    if ev_eb and ev_eb > 0 and ev_no_cash:
        ebitda = ev_no_cash / ev_eb
    elif rev and op_mar:
        ebitda = rev * (op_mar / 100) + rev * 0.05
    elif ni:
        ebitda = ni * 1.35

    if cash_direct is not None and cash_direct > 0:
        cash = cash_direct
    elif ebitda and ev_eb and ev_eb > 0:
        cash = max(0.0, mktcap + debt - ebitda * ev_eb)
    else:
        cash = 0.0

    # (buyback field not available in TradingView screener API)
    buyback_pct = None

    return dict(
        ticker=str(row.get("name", "?")), price=price, market_cap=mktcap,
        sector=sector, fcf=fcf, fcf_margin=fcf_margin,
        fcf_ps=(fcf / shares) if (fcf and shares and shares > 0) else None,
        cash=cash, debt=debt, net_debt=debt - cash, ebitda=ebitda,
        ev_ebitda=ev_eb, eps=eps, pe=s("price_earnings_ttm"),
        shares=shares, beta=beta, fcf_growth=fcf_growth, revenue=rev,
        net_income=ni, op_margin=op_mar, debt_equity=debt_equity,
        rev_growth=rev_growth, eps_growth=eps_growth,
        roe=roe, div_yield=div_yield,
        hi52=hi52, lo52=lo52, pos52=pos52,
        perf_1m=perf_1m, perf_3m=perf_3m,
        ev_live=ev_live,
        # v3
        roic=roic, p_b=p_b, buyback_pct=buyback_pct, gross_margin_v=gross_margin_v,
        eps_fwd_fy=eps_fwd_fy, rev_fwd_fy=rev_fwd_fy,
        actual_growth_used=actual_growth_used,
    )


# ── SCORE ─────────────────────────────────────────────────────────────────────
def score_stock(d, bm_market, bm_sector):
    """
    Values a stock using DCF + three relative methods.
    New in v2:
      - Growth is capped at sector average (avoids inflated DCF on mature cos)
      - Dividend-yield method added for income sectors
      - PEG ratio computed and stored
      - Value-trap flags identified
      - Conviction weighted by discount magnitude + quality (ROE)
      - Composite 0-100 rank score added
    """
    price = d["price"]
    if not price or price <= 0: return None

    # ── Resolve sector benchmark
    sec = d["sector"]
    sb  = bm_sector.get(sec, SECTOR_DEFAULTS.get(sec, bm_market))

    # ── Cap growth at sector average (prevents inflated DCF for slow sectors)
    growth_cap  = sb.get("growth_cap", MARKET_DEFAULTS["growth_cap"])
    growth      = min(d["fcf_growth"], growth_cap)

    m   = growth_adjusted_multiples(growth)
    fvs = {}

    # ── DCF (uses growth-capped rate)
    if d["fcf"] and d["fcf"] > 0 and d["shares"] and d["shares"] > 0:
        wacc = round(max(0.08, min(RISK_FREE_RATE + d["beta"]*EQUITY_RISK_PREMIUM, 0.15)), 4)
        cf, pvs = d["fcf"], []
        for yr in range(1, PROJECTION_YEARS+1):
            cf *= (1+growth); pvs.append(cf / (1+wacc)**yr)
        last  = d["fcf"] * (1+growth)**PROJECTION_YEARS
        pv_tv = (last*(1+TERMINAL_GROWTH)/(wacc-TERMINAL_GROWTH)) / (1+wacc)**PROJECTION_YEARS
        eq    = sum(pvs) + pv_tv - d["net_debt"]
        if eq > 0: fvs["DCF"] = eq / d["shares"]

    # ── Three-Stage DCF (library function)
    if d["fcf"] and d["fcf"] > 0 and d["shares"] and d["shares"] > 0:
        _d3 = {
            "fcf": d["fcf"], "shares": d["shares"],
            "est_growth": growth, "total_debt": d["debt"], "cash": d["cash"],
            "beta": d["beta"], "market_cap": d["market_cap"],
            "wacc_override": None,
            "wacc_raw": {"beta_yf": None, "interest_expense": None,
                         "income_tax_expense": None, "pretax_income": None,
                         "total_debt_yf": None},
        }
        _r3 = run_three_stage_dcf(_d3)
        if _r3 and _r3.get("fair_value") and _r3["fair_value"] > 0:
            fvs["Three-Stage DCF"] = _r3["fair_value"]

    # ── P/FCF (sector median)
    if d["fcf_ps"] and d["fcf_ps"] > 0:
        fvs["P/FCF"] = (
            d["fcf_ps"] * m["pf_t"]  +
            d["fcf_ps"] * sb["pfcf"] +
            d["fcf_ps"] * m["pf_c"]
        ) / 3

    # ── P/E TTM (sector median)
    if d["eps"] and d["eps"] > 0:
        fvs["P/E"] = (
            d["eps"] * m["pe_t"] +
            d["eps"] * sb["pe"] +
            d["eps"] * m["pe_c"]
        ) / 3

    # ── Forward P/E (analyst consensus next-FY EPS × sector median P/E)
    # This is the most widely-used professional valuation anchor.
    # A stock trading below (fwd_eps × sector_median_pe) is pricing in
    # pessimism that analysts don't share — a classic value signal.
    if d.get("eps_fwd_fy") and d["eps_fwd_fy"] > 0:
        fvs["Fwd P/E"] = (
            d["eps_fwd_fy"] * m["pe_t"]  +
            d["eps_fwd_fy"] * sb["pe"]   +
            d["eps_fwd_fy"] * m["pe_c"]
        ) / 3

    # ── EV/EBITDA (sector median)
    if d["ebitda"] and d["ebitda"] > 0 and d["shares"] and d["shares"] > 0:
        def ip(mult): return (d["ebitda"]*mult - d["debt"]) / d["shares"]
        mid = (
            ip(m["ev_t"])       +
            ip(sb["ev_ebitda"]) +
            ip(m["ev_c"])
        ) / 3
        if mid > 0: fvs["EV/EBITDA"] = mid

    # ── FCF Yield (floor method — 0.5× weight in composite so it doesn't dominate)
    _fv_fcf_yield = None
    if d.get("fcf_ps") and d["fcf_ps"] > 0:
        _target_yield = RISK_FREE_RATE + 0.04
        _fv_fcf_yield = d["fcf_ps"] / _target_yield
        if _fv_fcf_yield > 0:
            fvs["FCF Yield"] = _fv_fcf_yield

    # ── DDM H-Model (income sectors — replaces simple div yield method)
    # V = D0*(1+gL)/(r-gL) + D0*H*(gS-gL)/(r-gL), H=5, gL=0.03
    if (sec in INCOME_SECTORS and
            d["div_yield"] and d["div_yield"] > 0 and
            d["price"] and d["price"] > 0):
        D0   = d["price"] * (d["div_yield"] / 100.0)
        gS   = growth        # high-growth phase rate
        gL   = TERMINAL_GROWTH   # long-run terminal rate (0.03)
        H    = 5.0
        Ke   = max(0.06, RISK_FREE_RATE + d["beta"] * EQUITY_RISK_PREMIUM)
        if Ke > gL and D0 > 0:
            fv_ddm = (D0 * (1 + gL) / (Ke - gL)) + (D0 * H * (gS - gL) / (Ke - gL))
            if fv_ddm > 0:
                fvs["DDM H-Model"] = fv_ddm


    # ── P/B method (Financials and Real Estate only)
    # For banks, insurers, and REITs, book value is the primary valuation anchor.
    # A stock trading below its sector-median P/B is a meaningful signal that
    # earnings-based methods (P/E, EV/EBITDA) often miss for these sectors.
    PB_SECTOR_MEDIANS = {
        "Financials":   1.5,   # banks/insurers typically trade 1.0-2.0x book
        "Real Estate":  1.8,   # REITs often trade near or above book
    }
    if sec in PB_SECTOR_MEDIANS and d.get("p_b") and d["p_b"] > 0:
        sector_pb = PB_SECTOR_MEDIANS[sec]
        book_ps   = d["price"] / d["p_b"]          # book value per share
        pb_fv     = book_ps * sector_pb             # fair value at sector median P/B
        if pb_fv > 0:
            fvs["P/Book"] = pb_fv

    if not fvs: return None
    vals      = list(fvs.values())
    # FCF Yield is a floor method — apply 0.5× weight so it doesn't dominate
    _fvs_weights = {k: (0.5 if k == "FCF Yield" else 1.0) for k in fvs}
    _total_w = sum(_fvs_weights[k] for k in fvs)
    composite = sum(fvs[k] * _fvs_weights[k] for k in fvs) / _total_w
    if composite <= 0: return None

    discount = (composite - price) / price * 100

    # ── Conviction: method agreement + quality gates ─────────────────────────
    conv_count = sum(1 for v in vals if abs(v-composite)/composite <= CONVERGENCE_THRESH)
    raw_conv   = "HIGH" if conv_count==len(vals) else "MED" if conv_count>=len(vals)-1 else "LOW"

    # Downgrade if discount is small
    if discount < 5 and raw_conv == "HIGH": raw_conv = "MED"

    # Quality gates: ROE + ROIC both strong = highest-quality value candidate
    roic      = d.get("roic")
    roe_ok    = d["roe"] is not None and d["roe"] >= MIN_ROE_QUALITY * 100
    roic_ok   = roic is not None and roic >= 12.0   # ROIC >= 12% = above cost of capital
    gm_ok     = d.get("gross_margin_v") is not None and d["gross_margin_v"] >= 40.0

    quality_strong = roe_ok and roic_ok           # both profitability metrics confirm quality
    quality_decent = roe_ok or roic_ok            # at least one

    if quality_strong and conv_count == len(vals) and discount >= 15:
        conviction = "HIGH"
    elif quality_decent and conv_count >= len(vals)-1 and discount >= 10:
        conviction = raw_conv if raw_conv == "HIGH" else "MED"
    else:
        conviction = raw_conv

    # Gross margin bonus: wide moat + cheap = upgrade MED → HIGH when discount is meaningful
    if gm_ok and conviction == "MED" and discount >= 20 and quality_decent:
        conviction = "HIGH"

    # ── Value-trap flags
    trap_flags = get_value_trap_flags(d)

    # ── PEG ratio
    peg = compute_peg(d["pe"], growth)

    # ── Tier
    tier_name = TIERS[-1][0]
    for name, lo, hi, _ in TIERS:
        if lo <= discount < hi: tier_name = name; break

    # ── Composite 0-100 rank score
    rank_score = compute_value_rank_score(
        discount, conviction, d["roe"], d["pos52"],
        d["rev_growth"], d["eps_growth"],
        len(fvs), trap_flags, d["div_yield"], sec,
        roic=d.get("roic"),
        gross_margin_v=d.get("gross_margin_v"),
        buyback_pct=d.get("buyback_pct"),
    )

    return dict(
        ticker=d["ticker"], sector=d["sector"], price=price,
        composite=composite, fv_low=min(vals), fv_high=max(vals),
        discount=discount, mos_price=composite*(1-MARGIN_OF_SAFETY),
        tier=tier_name, conviction=conviction, n_methods=len(fvs),
        methods=list(fvs.keys()), market_cap=d["market_cap"],
        fv_three_stage_dcf=fvs.get("Three-Stage DCF"),
        fv_fcf_yield=fvs.get("FCF Yield"),
        fv_ddm=fvs.get("DDM H-Model"),
        growth=growth, pe=d["pe"], ev_ebitda=d["ev_ebitda"],
        fcf_margin=d["fcf_margin"], op_margin=d["op_margin"], beta=d["beta"],
        sec_pe=sb["pe"], sec_pfcf=sb["pfcf"], sec_eveb=sb["ev_ebitda"],
        peg=peg, roe=d["roe"], div_yield=d["div_yield"],
        rev_growth=d["rev_growth"], eps_growth=d["eps_growth"],
        pos52=d["pos52"], perf_1m=d["perf_1m"], perf_3m=d["perf_3m"],
        trap_flags=trap_flags, rank_score=rank_score,
        debt_equity=d["debt_equity"],
        # v3
        roic=d.get("roic"), gross_margin_v=d.get("gross_margin_v"),
        p_b=d.get("p_b"), buyback_pct=d.get("buyback_pct"),
        actual_growth_used=d.get("actual_growth_used", False),
        eps_fwd_fy=d.get("eps_fwd_fy"), rev_fwd_fy=d.get("rev_fwd_fy"),
    )


# ── FORMAT HELPERS ────────────────────────────────────────────────────────────
def fp(n):
    return ("$" + format(n, ",.2f")) if n is not None and n != 0 else "—"
def fb(n):
    if not n: return "—"
    if n >= 1e12: return "$" + format(n/1e12, ".2f") + "T"
    return "$" + format(n/1e9, ".1f") + "B"
def fx(n):
    return format(n, ".1f") + "x" if n is not None else "—"
def fpc(n, plus=True):
    if n is None: return "—"
    return ("+" if n >= 0 and plus else "") + format(n, ".1f") + "%"
def fpct(n):
    """Format a 0-1 decimal as percentage."""
    if n is None: return "—"
    return format(n * 100, ".1f") + "%"
def fpeg(n):
    if n is None: return "—"
    return format(n, ".2f")
def frank(n):
    if n is None: return "—"
    return format(n, ".0f")


# ── CSV EXPORT ────────────────────────────────────────────────────────────────
def write_csv(results, filename):
    headers = [
        "Rank Score", "Ticker", "Sector", "Tier", "Conviction",
        "Price", "Fair Value", "Discount %", "MoS Entry",
        "FV Low", "FV High", "Methods",
        "FV Three-Stage DCF", "FV FCF Yield", "FV DDM",
        "P/E", "PEG", "EV/EBITDA", "FCF Margin %", "Op Margin %",
        "ROIC %", "ROE %", "Gross Margin %",
        "Div Yield %", "Buyback % (12m)",
        "Rev Growth %", "EPS Growth %", "Growth Src",
        "Fwd EPS (next FY)", "Fwd Revenue (next FY)",
        "52wk Position %", "Perf 1M %", "Perf 3M %",
        "D/E", "Beta", "Market Cap", "Value Trap Flags",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in sorted(results, key=lambda x: -x["rank_score"]):
            def pn(v, mult=1, dec=2):
                return round(v*mult, dec) if v is not None else ""
            roic_v = r.get("roic")
            gm_v   = r.get("gross_margin_v")
            bb_v   = r.get("buyback_pct")
            w.writerow([
                pn(r["rank_score"], dec=1),
                r["ticker"], r["sector"], r["tier"], r["conviction"],
                pn(r["price"]), pn(r["composite"]),
                pn(r["discount"], dec=1), pn(r["mos_price"]),
                pn(r["fv_low"]), pn(r["fv_high"]),
                " | ".join(r["methods"]),
                pn(r.get("fv_three_stage_dcf")), pn(r.get("fv_fcf_yield")), pn(r.get("fv_ddm")),
                pn(r["pe"], dec=1), pn(r["peg"]),
                pn(r["ev_ebitda"], dec=1),
                pn(r["fcf_margin"], 100, 1) if r["fcf_margin"] else "",
                pn(r["op_margin"], 1, 1),
                pn(roic_v, 1, 1) if roic_v is not None else "",
                pn(r["roe"], 1, 1) if r["roe"] is not None else "",
                pn(gm_v, 1, 1) if gm_v is not None else "",
                pn(r["div_yield"], dec=2) if r["div_yield"] else "",
                pn(bb_v, 1, 1) if bb_v is not None else "",
                pn(r["rev_growth"], 1, 1) if r["rev_growth"] is not None else "",
                pn(r["eps_growth"], 1, 1) if r["eps_growth"] is not None else "",
                ("\u2713\u2713" if (r.get("eps_fwd_fy") or r.get("rev_fwd_fy")) else "\u2713") if r.get("actual_growth_used") else "\u007e",
                pn(r.get("eps_fwd_fy"), dec=2) if r.get("eps_fwd_fy") is not None else "",
                r.get("rev_fwd_fy", ""),
                pn(r["pos52"], 100, 1) if r["pos52"] is not None else "",
                pn(r["perf_1m"], dec=1) if r["perf_1m"] is not None else "",
                pn(r["perf_3m"], dec=1) if r["perf_3m"] is not None else "",
                pn(r["debt_equity"], dec=2) if r["debt_equity"] is not None else "",
                pn(r["beta"], dec=2),
                r["market_cap"],
                " | ".join(r["trap_flags"]) if r["trap_flags"] else "",
            ])
    print(f"  CSV saved: {filename}")


# ── HTML / CSS / JS ───────────────────────────────────────────────────────────
CSS = """
:root{--bg:#07080c;--s1:#0d0f16;--s2:#12151f;--bd:#1a1f2e;--tx:#cdd5e0;--mu:#4a5270;--gr:#00d68f;--rd:#ff4d6d;--yw:#ffd166;--bl:#4895ef;--pu:#a78bfa;}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--tx);font-family:'Inter',sans-serif;font-size:13px;min-height:100vh}
.hero{background:linear-gradient(150deg,#0a0d17 0%,#07080c 100%);border-bottom:1px solid var(--bd);padding:72px 64px 56px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-200px;right:-100px;width:600px;height:600px;border-radius:50%;background:radial-gradient(circle,rgba(0,214,143,.06) 0%,transparent 65%);pointer-events:none;}
.hero-eye{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:4px;text-transform:uppercase;color:var(--gr);margin-bottom:20px;}
.hero-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(64px,9vw,120px);line-height:.92;letter-spacing:2px;color:#fff;margin-bottom:24px;}
.hero-title span{color:var(--gr);}
.hero-sub{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--mu);margin-bottom:40px;}
.stat-row{display:flex;gap:32px;flex-wrap:wrap;}
.stat{display:flex;flex-direction:column;gap:4px;}
.stat-l{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--mu);}
.stat-v{font-family:'Bebas Neue',sans-serif;font-size:40px;line-height:1;}
.stat-v.g{color:var(--gr);}.stat-v.r{color:var(--rd);}.stat-v.m{color:var(--mu);}
.bmbar{display:flex;gap:32px;align-items:center;flex-wrap:wrap;padding:13px 64px;border-bottom:1px solid var(--bd);background:var(--s1);font-family:'IBM Plex Mono',monospace;font-size:11px;}
.bml{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--mu);margin-right:8px;}
.bmi{color:var(--tx);}.bmi b{color:var(--bl);}
.sectors{padding:24px 64px;border-bottom:1px solid var(--bd);background:var(--s2);}
.sec-l{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--mu);margin-bottom:12px;}
.chips{display:flex;gap:8px;flex-wrap:wrap;}
.chip{display:flex;align-items:center;gap:10px;background:var(--s1);border:1px solid var(--bd);border-radius:6px;padding:6px 14px;}
.chip span{font-size:12px;color:var(--tx);}.chip b{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--gr);font-weight:600;}
.nav{position:sticky;top:0;z-index:200;background:rgba(7,8,12,.93);backdrop-filter:blur(14px);border-bottom:1px solid var(--bd);padding:14px 64px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
.nl{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--mu);margin-right:4px;}
.npill{display:inline-flex;align-items:center;gap:8px;padding:5px 13px;border-radius:20px;text-decoration:none;font-size:11px;font-weight:600;color:var(--pillc);border:1px solid var(--pillb);background:var(--pillbg);transition:opacity .15s,transform .1s;white-space:nowrap;}
.npill:hover{opacity:.8;transform:translateY(-1px);}
.npill em{font-style:normal;font-family:'IBM Plex Mono',monospace;font-size:10px;background:rgba(255,255,255,.1);border-radius:10px;padding:1px 7px;}
.secbar{padding:12px 64px;border-bottom:1px solid var(--bd);display:flex;gap:8px;flex-wrap:wrap;align-items:center;background:var(--s1);}
.sbl{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--mu);margin-right:4px;flex-shrink:0;}
.sbtn{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.5px;padding:5px 13px;border-radius:6px;border:1px solid var(--bd);background:var(--s2);color:var(--mu);cursor:pointer;transition:all .15s;white-space:nowrap;}
.sbtn:hover{border-color:var(--bl);color:var(--tx);}.sbtn.active{background:var(--bl);border-color:var(--bl);color:#fff;}
.sbtn.all-btn{border-color:var(--mu);}.sbtn.all-btn.active{background:var(--mu);border-color:var(--mu);color:#fff;}
.filters{padding:14px 64px;border-bottom:1px solid var(--bd);display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:var(--bg);}
.fl{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--mu);}
.filters input,.filters select{background:var(--s2);border:1px solid var(--bd);border-radius:6px;color:var(--tx);font-family:'IBM Plex Mono',monospace;font-size:11px;padding:6px 12px;outline:none;}
.filters input::placeholder{color:var(--mu);}
.filters input:focus,.filters select:focus{border-color:var(--bl);}
.main{padding:0 64px 100px;}
.tier{margin-top:56px;}
.tier-hd{display:flex;align-items:center;gap:12px;padding-bottom:14px;border-bottom:1px solid var(--bd);margin-bottom:16px;}
.tdot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.tname{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;}
.tcnt{margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--mu);background:var(--s2);border:1px solid var(--bd);border-radius:4px;padding:3px 10px;}
.tbl-wrap{overflow-x:auto;border-radius:10px;border:1px solid var(--bd);}
table{width:100%;border-collapse:collapse;font-size:12px;}
thead tr{background:var(--s2);border-bottom:1px solid var(--bd);}
th{padding:10px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:var(--mu);white-space:nowrap;text-align:left;cursor:pointer;user-select:none;}
th:hover{color:var(--tx);}
th.sa::after{content:' \u2191';color:var(--bl);}
th.sd::after{content:' \u2193';color:var(--bl);}
tbody tr{border-bottom:1px solid var(--bd);transition:background .1s;}
tbody tr:last-child{border-bottom:none;}
tbody tr:hover{background:var(--s2);}
td{padding:10px 14px;white-space:nowrap;}
td.mono{font-family:'IBM Plex Mono',monospace;font-size:11.5px;}
td.dim{color:var(--mu);}td.sec{font-size:11px;color:var(--mu);}
td.hl{color:var(--bl);}td.pos{color:var(--gr);font-weight:600;}td.neg{color:var(--rd);font-weight:600;}
td.warn{color:var(--rd);font-size:10px;}
.tk{display:inline-block;background:var(--s2);border:1px solid var(--bd);border-radius:4px;padding:2px 8px;font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;color:var(--bl);}
.ch{color:var(--gr);font-weight:700;font-size:11px;}.cm{color:var(--yw);font-weight:700;font-size:11px;}.cl{color:var(--mu);font-weight:700;font-size:11px;}
.mtags i{display:inline-block;font-style:normal;font-family:'IBM Plex Mono',monospace;font-size:9px;background:var(--s2);border:1px solid var(--bd);border-radius:3px;padding:1px 5px;color:var(--mu);margin-right:3px;}
.trap{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:9px;background:rgba(255,77,109,.12);border:1px solid rgba(255,77,109,.35);border-radius:3px;padding:1px 5px;color:var(--rd);margin-right:3px;}
.rank-score{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;}
.rs-hi{color:var(--gr);}.rs-md{color:var(--yw);}.rs-lo{color:var(--mu);}
.peg-good{color:var(--gr);}.peg-ok{color:var(--yw);}.peg-bad{color:var(--rd);}
.perf-pos{color:var(--gr);}.perf-neg{color:var(--rd);}
tr.hidden{display:none;}
footer{padding:32px 64px;border-top:1px solid var(--bd);font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--mu);line-height:2;}
@media(max-width:768px){.hero,.main,.nav,.bmbar,.sectors,.filters,.secbar{padding-left:20px;padding-right:20px;}.hero-title{font-size:56px;}}
.legend{padding:16px 64px;background:var(--s2);border-bottom:1px solid var(--bd);display:flex;gap:24px;flex-wrap:wrap;align-items:center;}
.legend-l{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--mu);}
.legend-item{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--tx);}
"""

JS = """
document.querySelectorAll('th').forEach((th,i) => {
  th.addEventListener('click', () => {
    const tb = th.closest('table').querySelector('tbody');
    const asc = th.classList.toggle('sa');
    th.classList.toggle('sd', !asc);
    th.closest('thead').querySelectorAll('th').forEach(t => { if(t!==th){t.classList.remove('sa','sd');} });
    [...tb.querySelectorAll('tr:not(.hidden)')].sort((a,b) => {
      const av = a.cells[i]?.textContent.trim().replace(/[$,%x+BTM\u2014]/g,'') || '';
      const bv = b.cells[i]?.textContent.trim().replace(/[$,%x+BTM\u2014]/g,'') || '';
      const an = parseFloat(av), bn = parseFloat(bv);
      return isNaN(an)||isNaN(bn) ? (asc?av.localeCompare(bv):bv.localeCompare(av)) : (asc?an-bn:bn-an);
    }).forEach(r => tb.appendChild(r));
  });
});

let activeSector = '';

function setSector(sec, btn) {
  activeSector = sec;
  document.querySelectorAll('.sbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  ft();
  document.querySelectorAll('.tier').forEach(tier => {
    const visible = [...tier.querySelectorAll('tr.sr')].some(r => !r.classList.contains('hidden'));
    tier.style.display = visible ? '' : 'none';
  });
}

function ft() {
  const tk = document.getElementById('fs').value.toUpperCase();
  const sc = activeSector;
  const cv = document.getElementById('cs').value;
  const tf = document.getElementById('tf').value;
  document.querySelectorAll('tr.sr').forEach(r => {
    const t  = r.querySelector('.tk')?.textContent || '';
    const s  = r.dataset.sector || '';
    const c  = r.dataset.conv || '';
    const tr = r.dataset.trap || '';
    const trapMatch = tf === '' || (tf === 'clean' && tr === '') || (tf === 'flagged' && tr !== '');
    r.classList.toggle('hidden',
      (tk&&!t.includes(tk)) || (sc&&s!==sc) || (cv&&c!==cv) || !trapMatch
    );
  });
  document.querySelectorAll('.tier').forEach(tier => {
    const visible = [...tier.querySelectorAll('tr.sr')].some(r => !r.classList.contains('hidden'));
    tier.style.display = visible ? '' : 'none';
  });
}

const obs = new IntersectionObserver(es => es.forEach(e => {
  if(e.isIntersecting){ e.target.style.opacity='1'; e.target.style.transform='translateY(0)'; }
}), {threshold:0.04});
document.querySelectorAll('.tier').forEach((el,i) => {
  el.style.opacity='0';
  el.style.transform='translateY(20px)';
  el.style.transition='opacity .45s ease '+(i*.05)+'s,transform .45s ease '+(i*.05)+'s';
  obs.observe(el);
});
"""


def build_html(results, bm, ts, total):
    tier_map = {t[0]: [] for t in TIERS}
    for r in results:
        if r["tier"] in tier_map:
            tier_map[r["tier"]].append(r)
    # Sort each tier by rank_score descending (best opportunities first)
    for t in tier_map:
        tier_map[t].sort(key=lambda x: -x["rank_score"])

    uv  = sum(len(tier_map[t[0]]) for t in TIERS if t[1] >= 5)
    ov  = sum(len(tier_map[t[0]]) for t in TIERS if t[2] <= -5)
    fv  = len(tier_map["FAIRLY VALUED"])

    top_80 = sorted(results, key=lambda x: -x["rank_score"])[:5]

    sec_freq = {}
    for t in TIERS:
        if t[1] >= 5:
            for r in tier_map[t[0]]:
                sec_freq[r["sector"]] = sec_freq.get(r["sector"], 0) + 1
    top_sectors  = sorted(sec_freq.items(), key=lambda x: -x[1])[:8]
    all_sectors  = sorted(set(r["sector"] for r in results))

    # Nav pills
    nav_html = ""
    for name, lo, hi, color in TIERS:
        cnt = len(tier_map[name])
        if cnt == 0: continue
        sid = name.replace(" ", "_")
        h   = color.lstrip("#")
        r2, g2, b2 = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        bg  = "rgba(%d,%d,%d,0.1)" % (r2,g2,b2)
        brd = "rgba(%d,%d,%d,0.35)" % (r2,g2,b2)
        nav_html += (
            '<a class="npill" href="#' + sid + '" '
            'style="--pillc:' + color + ';--pillb:' + brd + ';--pillbg:' + bg + ';">'
            + name + ' <em>' + str(cnt) + '</em></a>'
        )

    chip_html = "".join(
        '<div class="chip"><span>' + s + '</span><b>' + str(c) + '</b></div>'
        for s, c in top_sectors
    )

    sector_btns = "".join(
        '<button class="sbtn" onclick="setSector(\'' + s.replace("'", "\\'") + '\', this)">' + s + '</button>'
        for s in all_sectors
    )

    # Top 5 best opportunities banner
    top5_html = ""
    for r in top_80:
        rs_cls = "rs-hi" if r["rank_score"] >= 60 else "rs-md" if r["rank_score"] >= 35 else "rs-lo"
        top5_html += (
            '<div class="chip">'
            '<span class="tk">' + r["ticker"] + '</span>'
            '<span class="rank-score ' + rs_cls + '">' + frank(r["rank_score"]) + '</span>'
            '</div>'
        )

    # Tier sections
    sections_html = ""
    for tier_name, lo, hi, color in TIERS:
        stocks = tier_map[tier_name]
        if not stocks: continue
        sid = tier_name.replace(" ", "_")

        rows = ""
        for r in stocks:
            disc_cls = "pos" if r["discount"] > 0 else "neg"
            conv_cls = "ch" if r["conviction"]=="HIGH" else "cm" if r["conviction"]=="MED" else "cl"
            mtags    = "".join("<i>" + m + "</i>" for m in r["methods"])
            trap_html= "".join('<span class="trap">' + f + '</span>' for f in r["trap_flags"])
            trap_data= "|".join(r["trap_flags"])

            rs_cls = "rs-hi" if r["rank_score"] >= 60 else "rs-md" if r["rank_score"] >= 35 else "rs-lo"

            # PEG colouring
            peg_cls = ""
            if r["peg"] is not None:
                peg_cls = "peg-good" if r["peg"] < 1.0 else "peg-ok" if r["peg"] < 1.5 else "peg-bad"

            # Perf colouring
            p1m = r["perf_1m"]
            p1m_cls = "perf-pos" if (p1m and p1m > 0) else "perf-neg" if (p1m and p1m < 0) else ""

            # 52-wk bar (simple text indicator)
            pos_txt = (format(r["pos52"]*100, ".0f") + "%") if r["pos52"] is not None else "—"

            # ROIC colouring
            roic_val = r.get("roic")
            roic_txt = (format(roic_val, ".1f") + "%") if roic_val is not None else "\u2014"
            roic_cls = "dim"
            if roic_val is not None:
                roic_cls = "perf-pos" if roic_val >= 15 else ("peg-ok" if roic_val >= 8 else "perf-neg")

            # Gross margin colouring
            gm_val = r.get("gross_margin_v")
            gm_txt = (format(gm_val, ".1f") + "%") if gm_val is not None else "\u2014"
            gm_cls = "perf-pos" if (gm_val and gm_val >= 50) else ("peg-ok" if (gm_val and gm_val >= 30) else "dim")

            # Buyback
            bb_val = r.get("buyback_pct")
            bb_txt = (format(bb_val, ".1f") + "%") if bb_val is not None else "\u2014"

            # Growth source indicator: ✓ = actual reported data, ~ = proxy
            # Growth source indicator: ✓✓ = analyst forecast used in DCF; ✓ = TTM actuals; ~ = proxy
            _fwd_used = r.get("eps_fwd_fy") is not None or r.get("rev_fwd_fy") is not None
            if r.get("actual_growth_used") and _fwd_used:
                growth_src = "\u2713\u2713"   # ✓✓ forecast
            elif r.get("actual_growth_used"):
                growth_src = "\u2713"          # ✓ TTM actuals
            else:
                growth_src = "\u007e"          # ~ proxy

            rows += (
                '<tr class="sr" data-sector="' + r["sector"] + '" data-conv="' + r["conviction"] + '" data-trap="' + trap_data + '">'
                '<td><span class="rank-score ' + rs_cls + '">' + frank(r["rank_score"]) + '</span></td>'
                '<td><span class="tk">' + r["ticker"] + '</span></td>'
                '<td class="sec">' + r["sector"] + '</td>'
                '<td class="mono">' + fp(r["price"]) + '</td>'
                '<td class="mono hl">' + fp(r["composite"]) + '</td>'
                '<td class="mono ' + disc_cls + '">' + fpc(r["discount"]) + '</td>'
                '<td class="mono">' + fp(r["mos_price"]) + '</td>'
                '<td class="mono dim">' + fp(r["fv_low"]) + " \u2013 " + fp(r["fv_high"]) + '</td>'
                '<td><span class="' + conv_cls + '">' + r["conviction"] + '</span></td>'
                '<td class="mtags">' + mtags + '</td>'
                '<td class="mono dim">' + fx(r["pe"]) + '</td>'
                '<td class="mono ' + peg_cls + '">' + fpeg(r["peg"]) + '</td>'
                '<td class="mono dim">' + fx(r["ev_ebitda"]) + '</td>'
                '<td class="mono dim">' + fpc(r["fcf_margin"]*100 if r["fcf_margin"] else None, False) + '</td>'
                '<td class="mono ' + roic_cls + '">' + roic_txt + '</td>'
                '<td class="mono dim">' + (format(r["roe"], ".1f") + "%" if r["roe"] is not None else "\u2014") + '</td>'
                '<td class="mono ' + gm_cls + '">' + gm_txt + '</td>'
                '<td class="mono dim">' + (fpc(r["div_yield"], False) if r["div_yield"] else "\u2014") + '</td>'
                '<td class="mono dim">' + bb_txt + '</td>'
                '<td class="mono dim">' + (fpc(r["rev_growth"]) if r["rev_growth"] is not None else "\u2014") + ' <span style="opacity:.45;font-size:9px;">' + growth_src + '</span></td>'
                '<td class="mono dim">' + (fp(r.get("eps_fwd_fy")) if r.get("eps_fwd_fy") is not None else "\u2014") + '</td>'
                '<td class="mono dim">' + (fb(r.get("rev_fwd_fy")) if r.get("rev_fwd_fy") is not None else "\u2014") + '</td>'
                '<td class="mono dim ' + p1m_cls + '">' + (fpc(r["perf_1m"]) if r["perf_1m"] is not None else "\u2014") + '</td>'
                '<td class="mono dim">' + pos_txt + '</td>'
                '<td class="mono dim">' + fb(r["market_cap"]) + '</td>'
                '<td class="warn">' + trap_html + '</td>'
                '</tr>'
            )

        sections_html += (
            '<section class="tier" id="' + sid + '">'
            '<div class="tier-hd" style="color:' + color + ';">'
            '<div class="tdot" style="background:' + color + ';box-shadow:0 0 8px ' + color + ';"></div>'
            '<span class="tname" style="color:' + color + ';">' + tier_name + '</span>'
            '<span class="tcnt">' + str(len(stocks)) + ' stocks</span>'
            '</div>'
            '<div class="tbl-wrap"><table>'
            '<thead><tr>'
            '<th>Score</th><th>Ticker</th><th>Sector</th><th>Price</th><th>Fair Value</th>'
            '<th>Discount</th><th>MoS Entry</th><th>FV Range</th>'
            '<th>Conv.</th><th>Methods</th>'
            '<th>P/E</th><th>PEG</th><th>EV/EBITDA</th>'
            '<th>FCF Margin</th>'
            '<th title="Return on Invested Capital — primary quality signal">ROIC</th>'
            '<th>ROE</th>'
            '<th title="Gross margin — pricing power proxy">Gross Margin</th>'
            '<th>Div Yield</th>'
            '<th title="Shares bought back as % of market cap (12m) — management conviction">Buyback %</th>'
            '<th title="Revenue growth YoY. ✓ = actual reported data used in DCF; ~ = FCF-margin proxy used">Rev Growth</th>'
            '<th>Perf 1M</th><th>52wk Pos</th>'
            '<th>Mkt Cap</th><th>Flags</th>'
            '</tr></thead>'
            '<tbody>' + rows + '</tbody>'
            '</table></div></section>'
        )

    page = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        '<title>Value Screener</title>'
        '<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue'
        '&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
        '<style>' + CSS + '</style></head><body>'

        '<div class="hero">'
        '<div class="hero-eye">// Valuation Intelligence v2</div>'
        '<h1 class="hero-title">Value<br><span>Screener</span></h1>'
        '<p class="hero-sub">GENERATED ' + ts + ' &nbsp;&middot;&nbsp; '
        + str(total) + ' STOCKS ANALYSED &nbsp;&middot;&nbsp; 6-METHOD COMPOSITE MODEL</p>'
        '<div class="stat-row">'
        '<div class="stat"><div class="stat-l">Undervalued</div><div class="stat-v g">' + str(uv) + '</div></div>'
        '<div class="stat"><div class="stat-l">Fairly Valued</div><div class="stat-v m">' + str(fv) + '</div></div>'
        '<div class="stat"><div class="stat-l">Overvalued</div><div class="stat-v r">' + str(ov) + '</div></div>'
        '<div class="stat"><div class="stat-l">Scored</div><div class="stat-v m">' + str(total) + '</div></div>'
        '</div></div>'

        '<div class="bmbar">'
        '<span class="bml">Benchmarks</span>'
        '<span class="bmi">Market P/E <b>' + str(bm["pe"]) + 'x</b></span>'
        '<span class="bmi">Market P/FCF <b>' + str(bm["pfcf"]) + 'x</b></span>'
        '<span class="bmi">Market EV/EBITDA <b>' + str(bm["ev_ebitda"]) + 'x</b></span>'
        '<span class="bmi" style="color:#4fd4a0;">&#10003; Sector-benchmarked</span>'
        '<span class="bmi">Risk-Free Rate <b>' + format(RISK_FREE_RATE*100, ".1f") + '%</b></span>'
        '<span class="bmi">MoS <b>' + format(MARGIN_OF_SAFETY*100, ".0f") + '%</b></span>'
        '</div>'

        '<div class="legend">'
        '<span class="legend-l">Score Guide</span>'
        '<span class="legend-item" style="color:#00d68f;">&#9632; 60–100 Strong Opportunity</span>'
        '<span class="legend-item" style="color:#ffd166;">&#9632; 35–59 Moderate</span>'
        '<span class="legend-item" style="color:#4a5270;">&#9632; 0–34 Weak / Watch</span>'
        '<span class="legend-item" style="color:#4895ef;">PEG &lt;1.0 Attractive &middot; PEG &lt;1.5 OK &middot; PEG &gt;1.5 Elevated</span>'
        '<span class="legend-item" style="color:#ff4d6d;">&#9888; Flags = potential value traps</span>'
        '</div>'

        '<div class="sectors">'
        '<div class="sec-l">Top 5 Opportunities (by Rank Score)</div>'
        '<div class="chips">' + top5_html + '</div>'
        '<div class="sec-l" style="margin-top:16px;">Top Undervalued Sectors</div>'
        '<div class="chips">' + chip_html + '</div>'
        '</div>'

        '<nav class="nav"><span class="nl">Jump to</span>' + nav_html + '</nav>'

        '<div class="secbar">'
        '<span class="sbl">Sector</span>'
        '<button class="sbtn all-btn active" onclick="setSector(\'\', this)">All Sectors</button>'
        + sector_btns +
        '</div>'

        '<div class="filters"><span class="fl">Filter</span>'
        '<input id="fs" type="text" placeholder="Search ticker\u2026" oninput="ft()" style="width:130px;">'
        '<select id="cs" onchange="ft()"><option value="">All Conviction</option>'
        '<option value="HIGH">High</option><option value="MED">Medium</option><option value="LOW">Low</option>'
        '</select>'
        '<select id="tf" onchange="ft()"><option value="">All Stocks</option>'
        '<option value="clean">No Flags (Clean)</option>'
        '<option value="flagged">Flagged Only</option>'
        '</select></div>'

        '<main class="main">' + sections_html + '</main>'

        '<footer>'
        'Generated ' + ts + ' &nbsp;&middot;&nbsp; Source: TradingView Screener<br>'
        'Methods: DCF &middot; P/FCF &middot; P/E &middot; EV/EBITDA &middot; EV&minus;Debt &middot; Dividend Yield (income sectors)<br>'
        'Score (0-100): Discount 40pts + Conviction 15pts + ROE 15pts + Momentum 10pts + Growth 10pts + Yield/Growth 10pts<br>'
        'Value-trap flags: NEG FCF &middot; NEG MARGIN &middot; HIGH DEBT &middot; REV DECLINE &middot; LOSING $<br>'
        'PEG Ratio (Lynch): &lt;1.0 attractive &middot; DCF growth capped at sector average to prevent overstatement<br>'
        'EV&minus;Debt: pure equity value = (Live Enterprise Value &minus; Total Debt) &divide; Shares<br>&#9888; Educational use only &mdash; not investment advice. Always verify against SEC filings (10-K / 10-Q).'
        '</footer>'

        '<script>' + JS + '</script>'
        '</body></html>'
    )
    return page


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    write_csv_flag = "--csv" in sys.argv

    # --index SPX|NDX|RUT|TSX  (default: SPX)
    if "--index" in sys.argv:
        try: index_code = sys.argv[sys.argv.index("--index")+1].upper()
        except: index_code = "SPX"
    else:
        index_code = "SPX"

    # --top N  overrides default limit for the chosen index
    if "--top" in sys.argv:
        try: limit = int(sys.argv[sys.argv.index("--top")+1])
        except: limit = None
    else:
        limit = None

    index_name = INDEX_CONFIG.get(index_code, INDEX_CONFIG["SPX"])["name"]
    print("\n" + "="*60)
    print(f"  VALUATION SCREENER — {index_name}")
    print("="*60)

    print("\n[1/4] Fetching stocks...")
    df = fetch_stocks(index_code, limit)

    print("\n[2/4] Fetching market benchmarks...")
    bm_market, bm_sector = fetch_benchmarks()
    bm = bm_market
    print(f"  Market — P/E={bm_market['pe']}x | P/FCF={bm_market['pfcf']}x | EV/EBITDA={bm_market['ev_ebitda']}x")
    print(f"  Sector benchmarks loaded for {len(bm_sector)} sectors")

    print("\n[3/4] Scoring stocks...")
    results, skipped = [], 0
    for _, row in df.iterrows():
        try:
            d = parse_row(row)
            r = score_stock(d, bm_market, bm_sector)
            if r: results.append(r)
            else: skipped += 1
        except Exception as e:
            skipped += 1

    print(f"  Scored: {len(results)} | Skipped: {skipped}")
    if not results:
        print("ERROR: No stocks scored."); sys.exit(1)

    # Count trap flags
    n_flagged = sum(1 for r in results if r["trap_flags"])
    n_clean   = len(results) - n_flagged

    print(f"\n  Value-trap flags: {n_flagged} flagged | {n_clean} clean")
    print("\n  Distribution:")
    tier_map = {t[0]: 0 for t in TIERS}
    for r in results:
        if r["tier"] in tier_map: tier_map[r["tier"]] += 1
    for name, lo, hi, color in TIERS:
        c = tier_map[name]
        if c: print("    " + name.ljust(30) + str(c).rjust(3) + "  " + "\u2588"*min(c,50))

    print("\n  Top 10 by Rank Score:")
    top10 = sorted(results, key=lambda x: -x["rank_score"])[:10]
    for i, r in enumerate(top10, 1):
        flags = " [" + ",".join(r["trap_flags"]) + "]" if r["trap_flags"] else ""
        print(f"    {i:>2}. {r['ticker']:<6}  Score={frank(r['rank_score']):>5}  Discount={fpc(r['discount']):>7}  ROE={fpct(r['roe']):>6}  PEG={fpeg(r['peg']):>5}{flags}")

    print("\n[4/4] Building HTML report...")
    ts         = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    html       = build_html(results, bm, ts, len(results))
    index_slug = index_code.lower()
    date_str   = datetime.datetime.now().strftime("%Y_%m_%d")
    os.makedirs("valueData", exist_ok=True)
    out     = os.path.join("valueData", f"{date_str}_valuation_report_{index_slug}.html")
    out_csv = os.path.join("valueData", f"{date_str}_valuation_report_{index_slug}.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved: {out}")

    if write_csv_flag:
        write_csv(results, out_csv)
        print(f"  CSV saved:  {out_csv}")

    webbrowser.open("file://" + os.path.abspath(out))
    print(f"\nDone. ({len(results)} stocks scored)")
    print("="*60)


if __name__ == "__main__":
    main()
