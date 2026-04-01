"""
Magic Formula Screener — Greenblatt's Magic Formula
=====================================================
Implements Joel Greenblatt's Magic Formula from "The Little Book That Beats the Market":

  1. Earnings Yield  = EBIT / Enterprise Value        (higher = better)
  2. Return on Capital = EBIT / (Net PP&E + Net Working Capital)  (higher = better; goodwill excluded)
  3. Rank all stocks on each metric independently (1 = best, N = worst)
  4. Sum the two ranks → Combined Rank (lower = better = buy signal)

Exclusions (per Greenblatt's original rules):
  • Financials and Utilities (fundamentally different capital structures)
  • Negative EV or EBIT (no meaningful earnings yield)
  • Non-primary share class listings
  • ADRs and depositary receipts (typespecs filter)
  • Stocks below $100M market cap

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMAND LINE ARGUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --index  INDEX    Which index to scan. Default: SPX
                    Options: SPX, NDX, RUT, TSX
  --top    N        Limit fetch to top N stocks by market cap.
  --csv             Export results to CSV alongside the HTML report.
  --help            Print this help message and exit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python magicFormula.py                  # S&P 500, full scan
  python magicFormula.py --index NDX      # Nasdaq 100
  python magicFormula.py --index RUT --top 500  # Russell 2000, top 500
  python magicFormula.py --csv            # S&P 500 + CSV export
"""

import sys, math, datetime, webbrowser, os, csv, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tradingview_screener import Query, col
except ImportError:
    print("ERROR: tradingview-screener not installed.")
    print("Run:  pip install tradingview-screener")
    sys.exit(1)

# ── FIELDS ────────────────────────────────────────────────────────────────────
FIELDS_CORE = [
    "name", "close", "market_cap_basic", "sector",
    "enterprise_value_fq",                      # Enterprise value (live MRQ)
    "operating_margin",                          # Operating margin % (TTM) → proxy for EBIT
    "total_revenue",                             # Revenue (TTM) → EBIT = op_margin * revenue
    "gross_margin",                              # Gross margin % (TTM)
    "price_earnings_ttm",                        # Trailing P/E
    "total_debt",                                # Total debt
    "free_cash_flow",                            # FCF (TTM)
    "net_income",                                # Net income (TTM)
    "return_on_equity",                          # ROE (TTM)
    "debt_to_equity",                            # D/E ratio
    "beta_1_year",                               # Beta
    "total_revenue_yoy_growth_ttm",              # Revenue growth YoY
    "earnings_per_share_diluted_yoy_growth_ttm", # EPS growth YoY
    "Perf.1M",                                   # 1-month price perf
    "Perf.3M",                                   # 3-month price perf
    "price_52_week_high",
    "price_52_week_low",
]

FIELDS_V3 = [
    "return_on_invested_capital",   # ROIC % — fallback ROC when balance sheet unavailable
    "cash_n_short_term_invest_fq",  # Cash + ST investments
    "price_book_fq",                # P/B ratio
    # Balance sheet fields for proper Greenblatt ROC = EBIT / (Net PP&E + Net WC)
    "total_assets_fq",              # Total assets
    "total_current_assets_fq",      # Current assets → Net Working Capital
    "total_current_liabilities_fq", # Current liabilities → Net Working Capital  (note: NOT "total_current_liabilities")
    "goodwill",                     # Strip goodwill from fixed assets per Greenblatt
]

# Per Greenblatt: exclude Financials and Utilities.
# Canada scanner uses "Finance" instead of "Financials" — include both.
EXCLUDE_SECTORS = {"Financials", "Finance", "Utilities"}

# Minimum market cap: original book used $50M (2005); $100M is more appropriate today
MIN_MARKET_CAP = 100e6   # $100 million

# ── INDEX CONFIG ──────────────────────────────────────────────────────────────
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
]

INDEX_CONFIG = {
    "SPX": {"name": "S&P 500",      "tickers": None,     "cap_limit": 503,  "is_rut": False, "is_tsx": False},
    "NDX": {"name": "Nasdaq 100",   "tickers": NASDAQ100,"cap_limit": 110,  "is_rut": False, "is_tsx": False},
    "RUT": {"name": "Russell 2000", "tickers": None,     "cap_limit": 2000, "is_rut": True,  "is_tsx": False},
    "TSX": {"name": "TSX",          "tickers": TSX,      "cap_limit": 200,  "is_rut": False, "is_tsx": True},
}


# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_stocks(index_code, limit=None):
    import pandas as pd

    cfg     = INDEX_CONFIG.get(index_code.upper(), INDEX_CONFIG["SPX"])
    name    = cfg["name"]
    tickers = cfg.get("tickers")
    cap     = limit or cfg["cap_limit"]
    # TSX stocks live on TradingView's "canada" scanner endpoint.
    # US indices use the default "america" endpoint.
    tv_market = "canada" if cfg["is_tsx"] else None

    print(f"  Querying TradingView for {name}...")

    def _q():
        """Return a Query() pre-set to the right market endpoint."""
        q = Query()
        if tv_market:
            q = q.set_markets(tv_market)
        return q

    def _merge_v3(df):
        """Fetch V3 balance-sheet fields and left-merge onto df."""
        try:
            tickers_fetched = df["name"].tolist()
            _, v3df = (
                _q().select("name", *FIELDS_V3)
                .where(col("name").isin(tickers_fetched))
                .limit(len(tickers_fetched) + 20).get_scanner_data()
            )
            df = df.merge(v3df[["name"] + FIELDS_V3], on="name", how="left")
            print(f"  v3 fields (ROIC, balance sheet, P/B) merged OK — {len(v3df)} rows")
        except Exception as e:
            print(f"  v3 fields unavailable ({str(e)[:80]}), continuing without")
        return df

    # ── Primary path: isin filter for curated constituent lists (NDX, TSX) ──
    if tickers:
        try:
            lim = limit or len(tickers) + 20
            _, df = (
                _q().select(*FIELDS_CORE)
                .where(col("name").isin(tickers), col("is_primary") == True)
                .order_by("market_cap_basic", ascending=False)
                .limit(lim).get_scanner_data()
            )
            df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
            df = _merge_v3(df)
            df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
            df["index_label"] = name
            print(f"  {name}: {len(df)} stocks fetched")
            return df
        except Exception as e:
            print(f"  isin filter failed ({str(e)[:60]}), trying fallback...")

    # ── TSX exchange fallback (canada market, top N by market cap) ──
    if cfg["is_tsx"]:
        try:
            _, df = (
                _q().select(*FIELDS_CORE)
                .where(col("is_primary") == True,
                       col("typespecs").has_none_of(["preferred"]),
                       col("market_cap_basic") > MIN_MARKET_CAP)
                .order_by("market_cap_basic", ascending=False)
                .limit(cap + 50).get_scanner_data()
            )
            df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
            df = _merge_v3(df)
            df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
            df["index_label"] = name
            print(f"  {name}: {len(df)} stocks (canada market fallback)")
            return df
        except Exception as e:
            print(f"  TSX fallback failed: {str(e)[:60]}")

    import pandas as pd
    frames = []
    for exch in ["NYSE", "NASDAQ"]:
        try:
            _, df = (
                Query().select(*FIELDS_CORE)
                # Greenblatt: US-listed, primary share class, no ADRs/preferreds, min market cap
                # No country filter — exchange + depositary exclusion already handles ADRs;
                # country == "US" would incorrectly drop companies incorporated abroad but
                # primarily listed on NYSE/NASDAQ as regular shares (not ADRs).
                .where(col("exchange") == exch, col("is_primary") == True,
                       col("typespecs").has_none_of(["preferred", "depositary"]),
                       col("market_cap_basic") > MIN_MARKET_CAP)
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
        if cfg["is_rut"]:
            combined = combined.iloc[503:503+cap].reset_index(drop=True)
        else:
            combined = combined.head(cap).reset_index(drop=True)
        combined = _merge_v3(combined)
        combined["index_label"] = name
        print(f"  {name}: {len(combined)} stocks (exchange fallback)")
        return combined

    print("  All filters failed — empty result"); sys.exit(1)


# ── PARSE ─────────────────────────────────────────────────────────────────────
def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))

def parse_row(row):
    def s(f, d=None):
        v = row.get(f, d)
        if _nan(v): return d
        try: return float(v)
        except Exception: return d

    price    = s("close", 0)
    mktcap   = s("market_cap_basic", 0)
    sector   = str(row.get("sector") or "Unknown")
    op_mar   = s("operating_margin")    # % form (e.g., 18.5 means 18.5%)
    rev      = s("total_revenue")
    ev       = s("enterprise_value_fq")
    roic     = s("return_on_invested_capital")  # ROIC % from TV (e.g., 22.1)
    roe      = s("return_on_equity")
    gross_m  = s("gross_margin")
    pe       = s("price_earnings_ttm")
    debt     = s("total_debt", 0)
    fcf      = s("free_cash_flow")
    ni       = s("net_income")
    beta     = s("beta_1_year", 1.0) or 1.0
    rev_growth = s("total_revenue_yoy_growth_ttm")
    eps_growth = s("earnings_per_share_diluted_yoy_growth_ttm")
    perf_1m  = s("Perf.1M")
    perf_3m  = s("Perf.3M")
    hi52     = s("price_52_week_high")
    lo52     = s("price_52_week_low")
    de           = s("debt_to_equity")
    cash         = s("cash_n_short_term_invest_fq", 0) or 0
    p_b          = s("price_book_fq")
    # Balance sheet fields for Greenblatt ROC = EBIT / (Net PP&E + Net WC)
    total_assets = s("total_assets_fq")
    curr_assets  = s("total_current_assets_fq")
    curr_liab    = s("total_current_liabilities_fq")
    goodwill_val = s("goodwill", 0) or 0

    # EBIT proxy: operating_margin (%) * total_revenue
    ebit = None
    if op_mar is not None and rev and rev > 0:
        ebit = (op_mar / 100.0) * rev

    # Earnings Yield: EBIT / EV  (Greenblatt component 1)
    ey = None
    if ebit is not None and ev and ev > 0 and ebit > 0:
        ey = ebit / ev  # raw ratio (e.g., 0.085 = 8.5%)

    # ── Greenblatt Return on Capital: EBIT / (Net PP&E + Net Working Capital) ──
    # Net Fixed Assets = total_assets − current_assets − goodwill
    #   Greenblatt explicitly excludes goodwill & intangibles so capital-light
    #   businesses (which earn high returns on tangible assets) score higher.
    #   We strip goodwill (available from TV). Other intangibles are not exposed
    #   by the TV screener API, so this is still a slight understatement for
    #   acquisition-heavy firms — conservative but accurate in direction.
    # Net Working Capital = max(0, current_assets − current_liabilities)
    #   Greenblatt uses max(0, …) so firms with negative WC get WC = 0.
    greenblatt_roc = None
    if (ebit is not None and ebit > 0 and
            total_assets is not None and
            curr_assets  is not None and
            curr_liab    is not None):
        net_fixed_assets    = max(0.0, total_assets - curr_assets - goodwill_val)
        net_working_capital = max(0.0, curr_assets  - curr_liab)
        invested_capital    = net_fixed_assets + net_working_capital
        if invested_capital > 0:
            greenblatt_roc = (ebit / invested_capital) * 100.0  # as %

    # FCF margin
    fcf_margin = (fcf / rev) if (fcf and rev and rev > 0) else None

    # 52-week position
    pos52 = None
    if hi52 and lo52 and hi52 > lo52 and price:
        pos52 = (price - lo52) / (hi52 - lo52)

    ticker = str(row.get("name") or "")

    return dict(
        ticker=ticker, sector=sector, price=price, mktcap=mktcap,
        ebit=ebit, ev=ev, ey=ey,
        greenblatt_roc=greenblatt_roc,  # proper Greenblatt ROC
        roic=roic,                      # TV's ROIC — fallback only
        roe=roe, gross_m=gross_m, op_mar=op_mar, pe=pe,
        debt=debt, cash=cash, fcf=fcf, ni=ni,
        fcf_margin=fcf_margin, beta=beta,
        rev_growth=rev_growth, eps_growth=eps_growth,
        perf_1m=perf_1m, perf_3m=perf_3m, pos52=pos52,
        de=de, p_b=p_b,
        # Stored so supplement_ppe() can recompute ROC with exact Net PP&E
        _curr_assets=curr_assets, _curr_liab=curr_liab,
    )


# ── SUPPLEMENT: exact Net PP&E from yfinance ──────────────────────────────────
def _fetch_ppe_yf(ticker):
    """Return (ticker, net_ppe_float) from the most recent yfinance annual balance sheet.
    Returns (ticker, None) on any failure."""
    try:
        import yfinance as yf
        bs = yf.Ticker(ticker).balance_sheet
        if bs is None or bs.empty or "Net PPE" not in bs.index:
            return ticker, None
        val = bs.loc["Net PPE"].iloc[0]   # most recent annual column
        v = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None
        return ticker, v
    except Exception:
        return ticker, None


def supplement_ppe(stocks, max_workers=12):
    """Fetch exact Net PP&E from yfinance in parallel and recompute Greenblatt ROC.

    Replaces the approximation (total_assets − current_assets − goodwill) with the
    real Net PP&E line directly from the balance sheet.  Fallback to the approximation
    is preserved for any ticker where yfinance cannot supply a value.
    """
    from concurrent.futures import ThreadPoolExecutor
    tickers = [s["ticker"] for s in stocks]
    print(f"  Fetching Net PP&E from yfinance for {len(tickers)} tickers", end="", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = dict(pool.map(_fetch_ppe_yf, tickers))

    exact = improved = 0
    for s in stocks:
        net_ppe = results.get(s["ticker"])
        if net_ppe is None or net_ppe < 0:
            continue
        ebit = s.get("ebit")
        if ebit is None or ebit <= 0:
            continue
        curr_assets = s.get("_curr_assets") or 0
        curr_liab   = s.get("_curr_liab")   or 0
        nwc = max(0.0, curr_assets - curr_liab)
        invested = net_ppe + nwc
        if invested > 0:
            s["greenblatt_roc"] = (ebit / invested) * 100.0
            s["_ppe_exact"] = True
            exact += 1
            improved += 1

    print(f" — {exact}/{len(tickers)} updated with exact Net PP&E")
    return stocks


# ── RANK ──────────────────────────────────────────────────────────────────────
def rank_and_score(parsed_rows):
    """
    Apply Greenblatt dual-ranking:
      EY rank  : rank by earnings_yield (EBIT/EV) descending (highest EY = rank 1)
      ROC rank : rank by Return on Capital descending (highest ROC = rank 1)
               → uses greenblatt_roc [EBIT/(Net PP&E + Net WC)] when available,
                 falls back to TV's ROIC when balance sheet data is missing
      Combined : EY_rank + ROC_rank → lower is better
    Returns list of dicts enriched with ey_rank, roc_rank, combined_rank, mf_rank.
    Stocks without both ey AND roc are placed at the end (unrankable).
    """
    # Annotate each row with the ROC value and method label to use for ranking
    for r in parsed_rows:
        if r.get("greenblatt_roc") is not None and r["greenblatt_roc"] > 0:
            r["_roc"]       = r["greenblatt_roc"]
            r["_roc_exact"] = True    # used the proper EBIT / (Net PP&E + Net WC)
        elif r.get("roic") is not None and r["roic"] > 0:
            r["_roc"]       = r["roic"]
            r["_roc_exact"] = False   # fell back to TV's ROIC
        else:
            r["_roc"]       = None
            r["_roc_exact"] = False

    # Separate into rankable vs unrankable
    rankable   = [r for r in parsed_rows if r.get("ey") is not None and r.get("ey") > 0
                                         and r.get("_roc") is not None]
    unrankable = [r for r in parsed_rows if r not in rankable]

    n = len(rankable)
    n_exact  = sum(1 for r in rankable if r["_roc_exact"])
    n_approx = n - n_exact
    if n_approx:
        print(f"  ROC method: {n_exact} exact (EBIT/invested capital), "
              f"{n_approx} fallback (TV ROIC — balance sheet data unavailable)")

    # EY rank (1 = highest EY)
    rankable.sort(key=lambda x: -(x["ey"] or 0))
    for i, r in enumerate(rankable, 1):
        r["ey_rank"] = i

    # ROC rank (1 = highest ROC)
    rankable.sort(key=lambda x: -(x["_roc"] or 0))
    for i, r in enumerate(rankable, 1):
        r["roc_rank"] = i

    # Combined rank
    for r in rankable:
        r["combined_rank"] = r["ey_rank"] + r["roc_rank"]

    # Sort by combined rank, break ties by EY rank
    rankable.sort(key=lambda x: (x["combined_rank"], x["ey_rank"]))

    # Final MF rank
    for i, r in enumerate(rankable, 1):
        r["mf_rank"] = i

    # Score: MF percentile 0-100 (rank 1 = score 100)
    for r in rankable:
        r["mf_score"] = round((1 - (r["mf_rank"] - 1) / max(n - 1, 1)) * 100, 1) if n > 1 else 100.0

    # Grade based on combined rank percentile
    for r in rankable:
        pct = r["mf_rank"] / max(n, 1)
        if   pct <= 0.10: r["grade"] = "A+"
        elif pct <= 0.20: r["grade"] = "A"
        elif pct <= 0.35: r["grade"] = "B+"
        elif pct <= 0.50: r["grade"] = "B"
        elif pct <= 0.65: r["grade"] = "C+"
        elif pct <= 0.80: r["grade"] = "C"
        else:             r["grade"] = "D"

    # Annotate unrankable stocks
    for r in unrankable:
        r["ey_rank"] = r["roc_rank"] = r["combined_rank"] = r["mf_rank"] = None
        r["mf_score"] = 0.0
        r["grade"] = "N/A"

    return rankable + unrankable


# ── FORMAT HELPERS ────────────────────────────────────────────────────────────
def _fp(n):
    return "$" + format(n, ",.2f") if n is not None and n != 0 else "—"
def _fb(n):
    if not n: return "—"
    if n >= 1e12: return "$" + format(n / 1e12, ".2f") + "T"
    if n >= 1e9:  return "$" + format(n / 1e9,  ".1f") + "B"
    if n >= 1e6:  return "$" + format(n / 1e6,  ".0f") + "M"
    return "$" + format(n, ",.0f")
def _fpc(n, plus=False):
    if n is None: return "—"
    return ("+" if n >= 0 and plus else "") + format(n, ".1f") + "%"
def _fpct(n):      # 0-1 ratio → percent
    return format(n * 100, ".1f") + "%" if n is not None else "—"
def _fx(n):
    return format(n, ".1f") + "x" if n is not None else "—"
def _fi(n):
    return str(int(n)) if n is not None else "—"


# ── CSV EXPORT ────────────────────────────────────────────────────────────────
def write_csv(results, filename):
    headers = [
        "MF Rank", "EY Rank", "ROC Rank", "Combined Rank", "MF Score", "Grade",
        "Ticker", "Sector", "Price", "Market Cap",
        "EBIT", "EV", "Earnings Yield %", "ROC % (Greenblatt)",
        "ROE %", "Gross Margin %", "Op Margin %", "FCF Margin %",
        "P/E", "D/E", "Beta", "Rev Growth %", "EPS Growth %",
        "Perf 1M %", "Perf 3M %",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in results:
            def pn(v, mult=1, dec=2):
                return round(v * mult, dec) if v is not None else ""
            w.writerow([
                pn(r.get("mf_rank"), dec=0),
                pn(r.get("ey_rank"),  dec=0),
                pn(r.get("roc_rank"), dec=0),
                pn(r.get("combined_rank"), dec=0),
                pn(r.get("mf_score"), dec=1),
                r.get("grade", ""),
                r["ticker"], r["sector"],
                pn(r["price"]), pn(r["mktcap"]),
                pn(r.get("ebit")), pn(r.get("ev")),
                pn(r.get("ey"), 100, 2) if r.get("ey") else "",
                pn(r.get("_roc"), 1, 1) if r.get("_roc") is not None else "",
                pn(r.get("roe"),  1, 1) if r.get("roe")  is not None else "",
                pn(r.get("gross_m"), 1, 1) if r.get("gross_m") is not None else "",
                pn(r.get("op_mar"),  1, 1) if r.get("op_mar")  is not None else "",
                pn(r.get("fcf_margin"), 100, 1) if r.get("fcf_margin") is not None else "",
                pn(r.get("pe"), dec=1) if r.get("pe") else "",
                pn(r.get("de"), dec=2) if r.get("de") is not None else "",
                pn(r.get("beta"), dec=2) if r.get("beta") else "",
                pn(r.get("rev_growth"), 1, 1) if r.get("rev_growth") is not None else "",
                pn(r.get("eps_growth"), 1, 1) if r.get("eps_growth") is not None else "",
                pn(r.get("perf_1m"), 1, 1) if r.get("perf_1m") is not None else "",
                pn(r.get("perf_3m"), 1, 1) if r.get("perf_3m") is not None else "",
            ])


# ── DEEP DIVE / WATCHLIST BAR (shared with value + growth screeners) ──────────
_WL_CSS = """
#wl-bar{display:flex;align-items:center;gap:10px;padding:7px 16px;
  background:#1a1e2e;border-top:1px solid #252a3a;
  position:sticky;bottom:0;z-index:100;}
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


# ── HTML ──────────────────────────────────────────────────────────────────────
_GRADE_COLORS = {
    "A+": "#00e5a0", "A": "#00c896", "B+": "#4fd4a0",
    "B": "#68d391",  "C+": "#f0b429","C": "#fc8181",
    "D": "#e05c5c",  "N/A": "#4a4f6a",
}

def _rank_color(rank, total):
    if rank is None or total == 0: return "#4a4f6a"
    pct = rank / total
    if   pct <= 0.10: return "#00e5a0"
    elif pct <= 0.25: return "#00c896"
    elif pct <= 0.40: return "#4fd4a0"
    elif pct <= 0.55: return "#a0aec0"
    elif pct <= 0.70: return "#f0b429"
    elif pct <= 0.85: return "#fc8181"
    else:             return "#e05c5c"

def _pct_color(val, lo=0, hi=None, reverse=False):
    """Return color for a percentage value. reverse=True means lower is better."""
    if val is None: return "#6b7194"
    if hi is None: hi = lo + 20
    ratio = (val - lo) / max(hi - lo, 0.001)
    ratio = max(0.0, min(1.0, ratio))
    if reverse: ratio = 1.0 - ratio
    if   ratio >= 0.75: return "#00c896"
    elif ratio >= 0.50: return "#4fd4a0"
    elif ratio >= 0.25: return "#f0b429"
    else:               return "#fc8181"


# ── Help modal ────────────────────────────────────────────────────────────────
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

_HELP_MODAL_MF = """
<div id="helpOverlay" class="help-overlay" onclick="if(event.target===this)closeHelp()"><div class="help-modal">
<button class="help-close" onclick="closeHelp()">&#x2715;</button>
<h2>&#x24D8; Magic Formula &mdash; How It Works</h2>
<p class="help-desc">Implements Joel Greenblatt&rsquo;s Magic Formula: simultaneously rank every stock by <b>Earnings Yield</b> (EBIT &divide; Enterprise Value &mdash; higher = cheaper) and <b>Return on Capital</b> (EBIT &divide; (Net PP&amp;E + Net Working Capital) &mdash; higher = better quality). The combined rank is the sum of both ranks. <b>Rank 1 = simultaneously cheapest AND highest quality.</b> Financials and Utilities are excluded as their capital structures make EY/ROIC non-comparable.</p>
<div class="help-sec">How Ranking Works</div>
<table class="help-tbl">
<tr><td>EY Rank</td><td>Rank by Earnings Yield descending. Rank 1 = highest EBIT/EV (cheapest stock).</td></tr>
<tr><td>ROIC Rank</td><td>Rank by Return on Capital descending. Rank 1 = highest ROIC (best quality).</td></tr>
<tr><td>Combined Rank</td><td>EY Rank + ROIC Rank. Lower is better. Rank 1 = best blend of cheap + quality.</td></tr>
<tr><td>MF Rank</td><td>Final position after sorting by Combined Rank. This is the main sorting column.</td></tr>
</table>
<div class="help-sec">Column Guide</div>
<table class="help-tbl">
<tr><td>EY %</td><td>Earnings Yield = EBIT &divide; Enterprise Value. Higher = stock is cheaper vs. operating earnings.</td></tr>
<tr><td>ROIC %</td><td>Return on Invested Capital = EBIT &divide; (Net PP&amp;E + Net Working Capital). Higher = more efficient capital use.</td></tr>
<tr><td>EV/EBITDA</td><td>Enterprise Value &divide; EBITDA. Shown as a cross-check valuation metric.</td></tr>
<tr><td>Grade</td><td>Letter grade based on combined rank percentile across the screened universe.</td></tr>
</table>
<div class="help-sec">Color Coding</div>
<table class="help-tbl">
<tr><td style="color:#00d68f">A+ / A (green)</td><td>Top-quartile combined rank &mdash; best value + quality stocks</td></tr>
<tr><td style="color:#4895ef">B (blue)</td><td>Above-median rank</td></tr>
<tr><td style="color:#ffd166">C (yellow)</td><td>Below-median rank</td></tr>
<tr><td style="color:#ff4d6d">D (red)</td><td>Bottom-quartile &mdash; expensive or low-quality relative to universe</td></tr>
</table>
</div></div>
"""


def build_html(results, ts, total_in, index_label, suite_port=5050):
    ranked = [r for r in results if r.get("mf_rank") is not None]
    unranked = [r for r in results if r.get("mf_rank") is None]
    n = len(ranked)

    all_sectors = sorted({r.get("sector", "") for r in results if r.get("sector")})
    sector_btns = "".join(
        f'<button class="sbtn" onclick="setSector({repr(s)}, this)">{s}</button>'
        for s in all_sectors
    )

    rows_html = []
    for r in results:
        is_ranked  = r.get("mf_rank") is not None
        mf_rank    = r.get("mf_rank")
        ey_rank    = r.get("ey_rank")
        roc_rank   = r.get("roc_rank")
        comb       = r.get("combined_rank")
        grade      = r.get("grade", "N/A")
        gc         = _GRADE_COLORS.get(grade, "#4a4f6a")
        rc         = _rank_color(mf_rank, n)
        ey_pct     = (r.get("ey") or 0) * 100
        ey_c       = _pct_color(ey_pct, lo=0, hi=15)
        # ROC display: prefer Greenblatt ROC; fall back to TV ROIC; mark fallback with ~
        roc_v      = r.get("_roc")          # greenblatt_roc if available, else TV ROIC
        roc_exact  = r.get("_roc_exact", False)
        roc_c      = _pct_color(roc_v, lo=0, hi=25) if roc_v is not None else "#6b7194"
        roc_title  = ("EBIT \u00f7 (Net PP&amp;E + Net WC) \u2014 Greenblatt exact"
                      if roc_exact else
                      "TV ROIC proxy \u2014 balance sheet data unavailable")
        roc_suffix = "" if roc_exact else '<sup style="color:#f0b429;font-size:9px">~</sup>'
        roc_display= (f"{_fpc(roc_v)}{roc_suffix}" if roc_v is not None else "\u2014")
        roe_v      = r.get("roe")
        gm_v       = r.get("gross_m")
        op_v       = r.get("op_mar")
        p1c        = ("#00c896" if (r.get("perf_1m") or 0) >= 0 else "#e05c5c")
        p3c        = ("#00c896" if (r.get("perf_3m") or 0) >= 0 else "#e05c5c")

        mf_rank_td = (f'<td class="mono" style="color:{rc};font-weight:700">{mf_rank}</td>'
                      if is_ranked else '<td style="color:#4a4f6a">—</td>')
        ey_rank_td = (f'<td class="mono" style="color:{_rank_color(ey_rank,n)}">{ey_rank}</td>'
                      if is_ranked else '<td style="color:#4a4f6a">—</td>')
        roc_rank_td= (f'<td class="mono" style="color:{_rank_color(roc_rank,n)}">{roc_rank}</td>'
                      if is_ranked else '<td style="color:#4a4f6a">—</td>')
        comb_td    = (f'<td class="mono" style="color:{rc}">{comb}</td>'
                      if is_ranked else '<td style="color:#4a4f6a">—</td>')
        grade_td   = f'<td><span style="background:{gc}20;color:{gc};padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600">{grade}</span></td>'

        sec_safe = r.get("sector", "").replace("'", "")
        rows_html.append(f"""
        <tr data-sector="{sec_safe}">
          <td class="cb-cell"><input type="checkbox" class="row-check" value="{r['ticker']}"></td>
          {mf_rank_td}
          <td class="mono" style="color:#e8eaf0;font-weight:600">{r["ticker"]}</td>
          <td style="color:#9ca3c8;font-size:11px">{r["sector"]}</td>
          <td class="mono">{_fp(r["price"])}</td>
          <td class="mono">{_fb(r["mktcap"])}</td>
          <td class="mono" style="color:{ey_c}">{_fpc(ey_pct) if r.get("ey") else "—"}</td>
          {ey_rank_td}
          <td class="mono" style="color:{roc_c}" title="{roc_title}">{roc_display}</td>
          {roc_rank_td}
          {comb_td}
          {grade_td}
          <td class="mono" style="color:{_pct_color(gm_v, 20, 65)}">{_fpc(gm_v) if gm_v is not None else "—"}</td>
          <td class="mono" style="color:{_pct_color(op_v, 0, 25)}">{_fpc(op_v) if op_v is not None else "—"}</td>
          <td class="mono" style="color:{_pct_color(roe_v, 5, 30)}">{_fpc(roe_v) if roe_v is not None else "—"}</td>
          <td class="mono">{_fx(r.get("pe")) if r.get("pe") else "—"}</td>
          <td class="mono">{_fpc(r.get("rev_growth")) if r.get("rev_growth") is not None else "—"}</td>
          <td class="mono" style="color:{p1c}">{_fpc(r.get("perf_1m"), plus=True) if r.get("perf_1m") is not None else "—"}</td>
          <td class="mono" style="color:{p3c}">{_fpc(r.get("perf_3m"), plus=True) if r.get("perf_3m") is not None else "—"}</td>
        </tr>""")

    rows_str = "\n".join(rows_html)
    top5_html = ""
    if ranked:
        top5 = ranked[:5]
        top5_html = " ".join([
            f'<span style="display:inline-flex;align-items:center;gap:6px;background:#1a2035;'
            f'border:1px solid #252a3a;border-radius:6px;padding:6px 12px;margin:3px;">'
            f'<span style="color:#00c896;font-weight:700;font-size:13px">#{r["mf_rank"]}</span>'
            f'<span style="color:#e8eaf0;font-weight:600">{r["ticker"]}</span>'
            f'<span style="color:#6b7194;font-size:11px">EY:{_fpc(r.get("ey",0)*100)} ROC:{_fpc(r.get("_roc"))}</span>'
            f'</span>' for r in top5
        ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Magic Formula Screener — {index_label}</title>
<style>
  :root {{
    --bg:#0f111a; --bg2:#161927; --bg3:#1a1e2e; --card:#1e2538;
    --border:#252a3a; --text:#e8eaf0; --mu:#6b7194;
    --gr:#00c896; --rd:#e05c5c; --yw:#f0b429; --bl:#4f8ef7; --pu:#a78bfa;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:13px;line-height:1.5}}
  a{{color:var(--bl);text-decoration:none}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
  .header{{background:linear-gradient(135deg,#0f1824 0%,#0f111a 100%);border-bottom:1px solid var(--border);padding:20px 24px}}
  .header h1{{font-size:22px;font-weight:700;color:#fff;letter-spacing:-0.3px}}
  .header .sub{{color:var(--mu);font-size:12px;margin-top:4px}}
  .meta-grid{{display:flex;flex-wrap:wrap;gap:12px;margin:14px 16px 0}}
  .meta-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 18px;min-width:140px}}
  .meta-card .val{{font-size:20px;font-weight:700;color:#e8eaf0}}
  .meta-card .lbl{{font-size:11px;color:var(--mu);margin-top:2px}}
  .top5{{margin:10px 16px;padding:10px 14px;background:#161927;border:1px solid var(--border);border-radius:8px}}
  .top5 .t5-label{{color:var(--mu);font-size:11px;margin-bottom:6px}}
  .tbl-wrap{{margin:12px 16px;overflow-x:auto;border-radius:8px;border:1px solid var(--border)}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  thead tr{{background:#1a1e2e}}
  th{{padding:8px 10px;text-align:left;color:var(--mu);font-weight:600;font-size:11px;
      white-space:nowrap;cursor:pointer;user-select:none;border-bottom:1px solid var(--border)}}
  th:hover{{color:var(--text)}}
  td{{padding:7px 10px;border-bottom:1px solid #1a1e2e;white-space:nowrap;vertical-align:middle}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#1a1e2e}}
  .mono{{font-family:'JetBrains Mono','Fira Code',monospace}}
  .controls{{margin:10px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
  .search-box{{background:#1e2538;border:1px solid var(--border);border-radius:6px;
               padding:6px 12px;color:var(--text);font-size:12px;width:200px;outline:none}}
  .search-box:focus{{border-color:var(--bl)}}
  .filter-btn{{background:#1e2538;border:1px solid var(--border);border-radius:6px;
               padding:5px 12px;color:var(--mu);font-size:11px;cursor:pointer}}
  .filter-btn.active{{background:#4f8ef720;border-color:var(--bl);color:var(--bl)}}
  .secbar{{margin:0 16px 10px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
  .secbar .lbl{{color:var(--mu);font-size:11px;white-space:nowrap;margin-right:4px}}
  .sbtn{{background:#1e2538;border:1px solid var(--border);border-radius:5px;
         padding:4px 10px;color:var(--mu);font-size:11px;cursor:pointer}}
  .sbtn:hover{{color:var(--text)}}
  .sbtn.active{{background:#4f8ef720;border-color:var(--bl);color:var(--bl)}}
  .legend{{margin:14px 16px;padding:12px 16px;background:var(--card);border:1px solid var(--border);
           border-radius:8px;font-size:11px;color:var(--mu)}}
  .legend b{{color:var(--text)}}
  .methodology{{margin:12px 16px;padding:14px 18px;background:#161927;border:1px solid var(--border);
                border-left:3px solid var(--bl);border-radius:8px;font-size:12px;color:var(--mu)}}
  .methodology b{{color:var(--text)}}
  .sort-asc::after{{content:" ▲"}}
  .sort-desc::after{{content:" ▼"}}
  #excluded-note{{margin:8px 16px;font-size:11px;color:var(--mu)}}
{_WL_CSS}
{_HELP_CSS}
</style>
</head>
<body>

<div class="header">
  <h1>📐 Magic Formula Screener — {index_label}</h1>
  <div class="sub">Greenblatt dual-ranking: Earnings Yield + Return on Capital · Generated {ts}</div>
</div>

<div class="meta-grid">
  <div class="meta-card"><div class="val">{total_in}</div><div class="lbl">Stocks Scanned</div></div>
  <div class="meta-card"><div class="val" style="color:var(--gr)">{n}</div><div class="lbl">Ranked</div></div>
  <div class="meta-card"><div class="val" style="color:var(--yw)">{total_in - n}</div><div class="lbl">Excluded / No Data</div></div>
  <div class="meta-card"><div class="val" style="color:var(--bl)">{sum(1 for r in ranked if r.get("grade") in ("A+","A"))}</div><div class="lbl">A / A+ Grade</div></div>
  <div class="meta-card"><div class="val">{round((ranked[0].get("ey",0) or 0)*100,1) if ranked else "—"}%</div><div class="lbl">Best Earn. Yield</div></div>
  <div class="meta-card"><div class="val">{round((ranked[0].get("_roc") or 0),1) if ranked else "—"}%</div><div class="lbl">Best ROC</div></div>
</div>

<div class="methodology">
  <b>Methodology (Greenblatt 2005):</b> Stocks are ranked on two dimensions:
  <b>Earnings Yield</b> = EBIT ÷ Enterprise Value (cheap → high EY = rank 1) and
  <b>Return on Capital</b> = EBIT ÷ (Net Fixed Assets + Net Working Capital) — tangible capital only,
  explicitly excluding goodwill &amp; intangibles so capital-light businesses score higher.
  Each stock receives a 1–N rank on both metrics; ranks are summed. <b>Lower combined rank = better blend of cheap + quality.</b>
  <b>Data note:</b> Net Fixed Assets uses exact Net PP&amp;E from yfinance where available (most stocks);
  falls back to total non-current assets minus goodwill when yfinance data is missing.
  ROC falls back to TV's ROIC as a proxy (marked <span style="color:#f0b429">~</span> in the table) when EBIT or balance sheet data is unavailable.
  <b>Excluded:</b> Financials, Utilities, ADRs/depositary receipts, stocks with market cap &lt; $100M, negative EBIT or EV.
</div>

<div class="top5">
  <div class="t5-label">TOP 5 MAGIC FORMULA PICKS</div>
  {top5_html}
</div>

<div class="controls">
  <input class="search-box" id="search" placeholder="Filter ticker or sector…" oninput="filterTable()">
  <button class="filter-btn" id="btn-all" onclick="setGradeFilter('all')" style="border-color:var(--bl);color:var(--bl)">All Grades</button>
  <button class="help-btn" onclick="openHelp()">&#x24D8; How it works</button>
  <button class="filter-btn" id="btn-a" onclick="setGradeFilter('A')">A / A+ Only</button>
  <button class="filter-btn" id="btn-b" onclick="setGradeFilter('B')">B+ / B Only</button>
  <span style="margin-left:auto;color:var(--mu);font-size:11px" id="row-count">{n + len(unranked)} stocks</span>
</div>
<div class="secbar">
  <span class="lbl">Sector</span>
  <button class="sbtn all-btn active" onclick="setSector('', this)">All</button>
  {sector_btns}
</div>

<div class="tbl-wrap">
<table id="mf-table">
<thead>
<tr>
  <th class="cb-th"><input type="checkbox" id="cb-all" title="Select all visible"></th>
  <th onclick="sortTable(1)" title="Magic Formula combined rank (lower=better)">MF Rank</th>
  <th onclick="sortTable(2)">Ticker</th>
  <th onclick="sortTable(3)">Sector</th>
  <th onclick="sortTable(4)">Price</th>
  <th onclick="sortTable(5)">Mkt Cap</th>
  <th onclick="sortTable(6)" title="EBIT / Enterprise Value — cheapness metric">Earn. Yield</th>
  <th onclick="sortTable(7)" title="Earnings Yield rank (1=cheapest)">EY Rank</th>
  <th onclick="sortTable(8)" title="Return on Capital: EBIT ÷ (Net PP&amp;E + Net WC) per Greenblatt. ~ = TV ROIC proxy used">ROC <span style="color:#f0b429;font-size:9px">~=proxy</span></th>
  <th onclick="sortTable(9)" title="Return on Capital rank (1=highest quality)">ROC Rank</th>
  <th onclick="sortTable(10)" title="EY Rank + ROC Rank — lower is better">Combined Rank</th>
  <th onclick="sortTable(11)" title="Magic Formula grade based on combined rank percentile">Grade</th>
  <th onclick="sortTable(12)">Gross Margin</th>
  <th onclick="sortTable(13)">Op Margin</th>
  <th onclick="sortTable(14)">ROE</th>
  <th onclick="sortTable(15)">P/E</th>
  <th onclick="sortTable(16)">Rev Growth</th>
  <th onclick="sortTable(17)">Perf 1M</th>
  <th onclick="sortTable(18)">Perf 3M</th>
</tr>
</thead>
<tbody id="mf-tbody">
{rows_str}
</tbody>
</table>
</div>

<div id="excluded-note">
  {len(unranked)} stocks excluded (Financials/Utilities or missing EBIT/EV/ROC data) — not shown in ranking.
</div>

<div class="legend">
  <b>Grade scale</b> (by combined rank percentile):
  <span style="color:#00e5a0">A+</span> top 10% ·
  <span style="color:#00c896">A</span> top 20% ·
  <span style="color:#4fd4a0">B+</span> top 35% ·
  <span style="color:#68d391">B</span> top 50% ·
  <span style="color:#f0b429">C+</span> top 65% ·
  <span style="color:#fc8181">C</span> top 80% ·
  <span style="color:#e05c5c">D</span> bottom 20% ·
  <span style="color:#4a4f6a">N/A</span> unranked<br>
  <b>Earnings Yield</b> = EBIT ÷ EV (higher = cheaper) ·
  <b>ROC</b> = EBIT ÷ (Net PP&amp;E + Net WC) per Greenblatt — tangible capital return (goodwill excluded) ·
  <span style="color:#f0b429">~</span> suffix = TV ROIC proxy used (balance sheet fields unavailable) ·
  <b>EV</b> = enterprise value (market cap + net debt)
</div>

<script>
let _sortCol = -1, _sortAsc = true, _gradeFilter = 'all';
const activeSectors = new Set();
function setSector(sec, btn) {{
  if (sec === '') {{
    activeSectors.clear();
  }} else {{
    if (activeSectors.has(sec)) activeSectors.delete(sec);
    else activeSectors.add(sec);
  }}
  document.querySelectorAll('.sbtn:not(.all-btn)').forEach(b =>
    b.classList.toggle('active', activeSectors.has(b.textContent.trim()))
  );
  const allBtn = document.querySelector('.sbtn.all-btn');
  if (allBtn) allBtn.classList.toggle('active', activeSectors.size === 0);
  filterTable();
}}

function sortTable(col) {{
  const tbody = document.getElementById('mf-tbody');
  const rows  = Array.from(tbody.rows);
  if (_sortCol === col) {{ _sortAsc = !_sortAsc; }}
  else {{ _sortCol = col; _sortAsc = true; }}
  document.querySelectorAll('th').forEach((th,i) => {{
    th.classList.remove('sort-asc','sort-desc');
    if (i === col) th.classList.add(_sortAsc ? 'sort-asc' : 'sort-desc');
  }});
  rows.sort((a,b) => {{
    let av = a.cells[col]?.textContent?.trim() ?? '';
    let bv = b.cells[col]?.textContent?.trim() ?? '';
    const an = parseFloat(av.replace(/[^0-9.-]/g,''));
    const bn = parseFloat(bv.replace(/[^0-9.-]/g,''));
    if (!isNaN(an) && !isNaN(bn)) return _sortAsc ? an - bn : bn - an;
    return _sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

function setGradeFilter(g) {{
  _gradeFilter = g;
  document.querySelectorAll('.filter-btn').forEach(b => b.style.borderColor = '');
  const btn = document.getElementById('btn-' + g);
  if (btn) {{ btn.style.borderColor = 'var(--bl)'; btn.style.color = 'var(--bl)'; }}
  filterTable();
}}

function filterTable() {{
  const q   = document.getElementById('search').value.toLowerCase();
  const rows = document.querySelectorAll('#mf-tbody tr');
  let shown = 0;
  rows.forEach(row => {{
    const ticker = row.cells[2]?.textContent?.toLowerCase() ?? '';
    const sector = row.cells[3]?.textContent?.toLowerCase() ?? '';
    const grade  = row.cells[11]?.textContent?.trim() ?? '';
    const matchQ = !q || ticker.includes(q) || sector.includes(q);
    const matchS = activeSectors.size === 0 || activeSectors.has(row.dataset.sector ?? '');
    const matchG = _gradeFilter === 'all'
      || (_gradeFilter === 'A' && (grade==='A+' || grade==='A'))
      || (_gradeFilter === 'B' && (grade==='B+' || grade==='B'));
    row.style.display = (matchQ && matchS && matchG) ? '' : 'none';
    if (matchQ && matchS && matchG) shown++;
  }});
  document.getElementById('row-count').textContent = shown + ' stocks';
}}
</script>
<script>{_wl_js(suite_port)}</script>
<script>{_HELP_JS}</script>
{_WL_BAR}
{_HELP_MODAL_MF}
</body>
</html>"""


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--index", default="SPX")
    ap.add_argument("--top",   type=int, default=None)
    ap.add_argument("--csv",   action="store_true")
    ap.add_argument("--help",  action="store_true")
    args = ap.parse_args()

    if args.help:
        print(__doc__); sys.exit(0)

    index_code = args.index.upper()
    if index_code not in INDEX_CONFIG:
        print(f"ERROR: Unknown index '{index_code}'. Valid: {', '.join(INDEX_CONFIG)}")
        sys.exit(1)

    print("=" * 60)
    print("  Magic Formula Screener")
    print("=" * 60)

    # ── Fetch
    print("\n[1/4] Fetching stocks from TradingView...")
    df = fetch_stocks(index_code)
    total_in = len(df)
    print(f"  Total fetched: {total_in}")

    # ── Parse
    print("\n[2/4] Parsing rows...")
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            d = parse_row(row.to_dict())
            if d["ticker"]:
                rows.append(d)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1

    # Apply Greenblatt exclusions
    excluded_sectors = [r for r in rows if r["sector"] in EXCLUDE_SECTORS]
    included = [r for r in rows if r["sector"] not in EXCLUDE_SECTORS]
    print(f"  Parsed: {len(rows)} | Sector-excluded: {len(excluded_sectors)} (Fin/Util) | Remaining: {len(included)}")

    # ── Supplement: replace approximate Net Fixed Assets with exact Net PP&E
    print("\n[2.5/4] Supplementing ROC with exact Net PP&E from yfinance...")
    supplement_ppe(included)

    # ── Rank
    print("\n[3/4] Ranking by Earnings Yield + Return on Capital...")
    ranked_all = rank_and_score(included)
    ranked     = [r for r in ranked_all if r.get("mf_rank") is not None]
    unranked   = [r for r in ranked_all if r.get("mf_rank") is None]
    if args.top:
        ranked = ranked[:args.top]
        print(f"\n  Trimmed to top {args.top} by Magic Formula rank ({len(ranked)} stocks in report).")
    print(f"  Ranked: {len(ranked)} | Missing data: {len(unranked)}")

    # Print top 10
    print("\n  Top 10 Magic Formula picks:")
    for r in ranked[:10]:
        ey_s    = f"{r.get('ey',0)*100:.1f}%" if r.get('ey') else "—"
        roic_s  = f"{r.get('roic',0):.1f}%" if r.get('roic') is not None else "—"
        print(f"    #{r['mf_rank']:>3}  {r['ticker']:<6}  EY={ey_s:<7} ROIC={roic_s:<7}"
              f"  Combined={r.get('combined_rank','?'):>4}  [{r.get('grade','')}]")

    # ── Build HTML
    print("\n[4/4] Building HTML report...")
    ts         = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    cfg        = INDEX_CONFIG[index_code]
    suite_port = int(os.environ.get("VALUATION_SUITE_PORT", "5050"))
    # Combine ranked (sorted by mf_rank) + excluded-sectors (at end, not ranked)
    all_results = ranked_all + excluded_sectors
    html = build_html(all_results, ts, total_in, cfg["name"], suite_port=suite_port)

    index_slug = index_code.lower()
    date_str   = datetime.datetime.now().strftime("%Y_%m_%d")
    out_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "formulaData")
    os.makedirs(out_dir, exist_ok=True)
    out     = os.path.join(out_dir, f"{date_str}_magic_formula_{index_slug}.html")
    out_csv = os.path.join(out_dir, f"{date_str}_magic_formula_{index_slug}.csv")

    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved: {out}")

    if args.csv:
        write_csv(ranked_all, out_csv)
        print(f"  CSV  saved: {out_csv}")

    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        webbrowser.open("file://" + os.path.abspath(out))
    print(f"\nDone. ({len(ranked)} stocks ranked)")
    print("=" * 60)


if __name__ == "__main__":
    main()
