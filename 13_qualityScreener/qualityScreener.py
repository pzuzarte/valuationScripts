"""
Quality / Compounder Screener
===============================
Identifies high-quality "compounder" businesses with durable competitive advantages.
Uses a 4-pillar scoring model (0-100 total):

  Pillar 1 — Return on Capital   (0-35 pts)
      ROIC, ROE, capital consistency bonus
      Best compounders sustain ROIC > 15% through full business cycles.

  Pillar 2 — Profitability & Margins  (0-25 pts)
      Gross margin, operating margin, FCF margin
      High gross margins signal pricing power; high FCF margin signals cash conversion.

  Pillar 3 — Balance Sheet Health  (0-25 pts)
      Net debt / EBITDA, Debt-to-Equity
      Capital-light businesses with low leverage survive downturns and self-fund growth.

  Pillar 4 — Capital Efficiency  (0-15 pts)
      FCF conversion (FCF/NI) and revenue growth
      Strong FCF conversion + consistent growth = efficient capital allocation.

Minimum Quality Gate:
  • ROIC > 0% (positive returns on capital)
  • Operating margin > 0% (profitable core operations)

Grade scale:
  A+ ≥ 90  ·  A ≥ 80  ·  B+ ≥ 70  ·  B ≥ 60  ·  C+ ≥ 50  ·  C ≥ 40  ·  D < 40

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMAND LINE ARGUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --index  INDEX    Which index to scan. Default: SPX
                    Options: SPX, NDX, RUT, TSX
  --top    N        Limit fetch to top N stocks by market cap.
  --csv             Export results to CSV alongside the HTML report.
  --min-score N     Only include stocks with quality score ≥ N (default: 0).
  --help            Print this help message and exit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python qualityScreener.py                         # S&P 500, all stocks
  python qualityScreener.py --index NDX             # Nasdaq 100
  python qualityScreener.py --min-score 70          # B+ and above only
  python qualityScreener.py --index RUT --top 500   # Russell 2000, top 500
  python qualityScreener.py --csv                   # S&P 500 + CSV export
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
    "operating_margin",                          # Op margin % (TTM) — Pillar 1 + 2
    "gross_margin",                              # Gross margin % (TTM) — Pillar 2
    "return_on_equity",                          # ROE % (TTM) — Pillar 1
    "total_revenue",                             # Revenue (TTM)
    "net_income",                                # Net income (TTM) — FCF conversion
    "free_cash_flow",                            # FCF (TTM) — Pillar 2 + 4
    "total_debt",                                # Total debt — Pillar 3
    "debt_to_equity",                            # D/E ratio — Pillar 3
    "enterprise_value_ebitda_ttm",               # EV/EBITDA (for EBITDA proxy) — Pillar 3
    "enterprise_value_fq",                       # Enterprise value (live) — Pillar 3
    "price_earnings_ttm",                        # Trailing P/E — display
    "beta_1_year",                               # Beta — display
    "dividend_yield_recent",                     # Dividend yield — display
    "total_revenue_yoy_growth_ttm",              # Revenue growth % — Pillar 4
    "earnings_per_share_diluted_yoy_growth_ttm", # EPS growth %
    "Perf.1M",                                   # 1-month perf
    "Perf.3M",                                   # 3-month perf
    "price_52_week_high",
    "price_52_week_low",
]

FIELDS_V3 = [
    "return_on_invested_capital",   # ROIC % — core quality metric (Pillar 1)
    "cash_n_short_term_invest_fq",  # Cash + ST invest — for net debt calc
    "price_book_fq",                # P/B ratio
    "earnings_per_share_forecast_next_fy",  # Fwd EPS
    "revenue_forecast_next_fy",             # Fwd revenue
]

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

    print(f"  Querying TradingView for {name}...")

    def _merge_v3(df):
        try:
            tickers_fetched = df["name"].tolist()
            _, v3df = (
                Query().select("name", *FIELDS_V3)
                .where(col("name").isin(tickers_fetched))
                .limit(len(tickers_fetched) + 20).get_scanner_data()
            )
            df = df.merge(v3df[["name"] + FIELDS_V3], on="name", how="left")
            print(f"  v3 fields (ROIC, cash, P/B, fwd estimates) merged OK")
        except Exception as e:
            print(f"  v3 fields unavailable ({str(e)[:80]}), continuing without")
        return df

    # Primary: isin filter for known constituent lists
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
            print(f"  {name}: {len(df)} stocks fetched")
            return df
        except Exception as e:
            print(f"  isin filter failed ({str(e)[:60]}), trying exchange fallback...")

    # Exchange fallback
    if cfg["is_tsx"]:
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
            print(f"  TSX fallback failed: {str(e)[:60]}")

    import pandas as pd
    frames = []
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
        if cfg["is_rut"]:
            combined = combined.iloc[503:503 + cap].reset_index(drop=True)
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
    op_mar   = s("operating_margin")      # % (e.g., 18.5)
    gross_m  = s("gross_margin")          # % (e.g., 52.0)
    roe      = s("return_on_equity")      # % (e.g., 22.0)
    roic     = s("return_on_invested_capital")  # % (e.g., 18.5)
    rev      = s("total_revenue")
    ni       = s("net_income")
    fcf      = s("free_cash_flow")
    debt     = s("total_debt", 0) or 0
    de       = s("debt_to_equity")
    ev_eb    = s("enterprise_value_ebitda_ttm")
    ev_live  = s("enterprise_value_fq")
    pe       = s("price_earnings_ttm")
    beta     = s("beta_1_year", 1.0) or 1.0
    div_yld  = s("dividend_yield_recent")
    rev_g    = s("total_revenue_yoy_growth_ttm")
    eps_g    = s("earnings_per_share_diluted_yoy_growth_ttm")
    perf_1m  = s("Perf.1M")
    perf_3m  = s("Perf.3M")
    hi52     = s("price_52_week_high")
    lo52     = s("price_52_week_low")
    cash     = s("cash_n_short_term_invest_fq", 0) or 0
    p_b      = s("price_book_fq")
    eps_fwd  = s("earnings_per_share_forecast_next_fy")
    rev_fwd  = s("revenue_forecast_next_fy")

    # FCF margin
    fcf_margin = (fcf / rev) if (fcf is not None and rev and rev > 0) else None

    # Net income margin
    ni_margin  = (ni / rev)  if (ni  is not None and rev and rev > 0) else None

    # FCF / NI conversion ratio
    fcf_ni = None
    if fcf is not None and ni is not None and ni > 0:
        fcf_ni = fcf / ni

    # EBITDA estimate (for net debt/EBITDA)
    ebitda = None
    if ev_eb and ev_eb > 0 and mktcap and debt is not None:
        ev_no_cash = mktcap + debt - cash
        if ev_no_cash > 0:
            ebitda = ev_no_cash / ev_eb
    elif rev and op_mar:
        ebitda = rev * (op_mar / 100.0) * 1.2  # rough: add back ~D&A = 20% of EBIT

    # Net debt / EBITDA
    net_debt = max(0.0, debt - cash)
    net_debt_ebitda = (net_debt / ebitda) if (ebitda and ebitda > 0 and net_debt > 0) else None
    if net_debt <= 0:
        net_debt_ebitda = -1.0  # net cash position (flag as best case)

    # 52-week position
    pos52 = None
    if hi52 and lo52 and hi52 > lo52 and price:
        pos52 = (price - lo52) / (hi52 - lo52)

    # Implied fwd growth
    fwd_growth = None
    if rev_fwd and rev and rev > 0 and eps_fwd:
        rev_impl = (rev_fwd / rev) - 1.0
        eps_impl = None
        if eps_fwd and pe and price:
            fwd_eps_growth = None
            if rev and rev > 0:
                fwd_eps_growth = (rev_fwd / rev) - 1.0
            eps_impl = fwd_eps_growth
        if rev_impl is not None:
            fwd_growth = max(-0.30, min(0.80, rev_impl))

    ticker = str(row.get("name") or "")

    return dict(
        ticker=ticker, sector=sector, price=price, mktcap=mktcap,
        roic=roic, roe=roe, op_mar=op_mar, gross_m=gross_m,
        fcf_margin=fcf_margin, ni_margin=ni_margin, fcf_ni=fcf_ni,
        net_debt_ebitda=net_debt_ebitda, de=de, ebitda=ebitda,
        rev=rev, ni=ni, fcf=fcf, debt=debt, cash=cash,
        pe=pe, p_b=p_b, beta=beta, div_yld=div_yld,
        rev_g=rev_g, eps_g=eps_g, fwd_growth=fwd_growth,
        perf_1m=perf_1m, perf_3m=perf_3m, pos52=pos52,
        ev_live=ev_live, ev_eb=ev_eb,
        eps_fwd=eps_fwd, rev_fwd=rev_fwd,
    )


# ── SCORING ───────────────────────────────────────────────────────────────────

def _score_return_on_capital(d):
    """Pillar 1: Return on Capital — 0 to 35 pts"""
    pts = 0.0

    # ROIC component (0-20): primary quality signal
    roic = d.get("roic")
    if roic is not None:
        if   roic >= 30: pts += 20
        elif roic >= 25: pts += 17
        elif roic >= 20: pts += 14
        elif roic >= 15: pts += 11
        elif roic >= 10: pts += 8
        elif roic >= 5:  pts += 5
        elif roic > 0:   pts += 2
        # roic <= 0 → 0 pts

    # ROE component (0-10): secondary quality signal
    roe = d.get("roe")
    if roe is not None:
        if   roe >= 25: pts += 10
        elif roe >= 20: pts += 8
        elif roe >= 15: pts += 6
        elif roe >= 10: pts += 4
        elif roe >= 5:  pts += 2
        # roe < 5 → 0 pts

    # Consistency bonus (0-5): both metrics strong signals a true compounder
    if roic is not None and roe is not None and roic >= 12 and roe >= 12:
        pts += 5

    return round(min(pts, 35), 1)


def _score_profitability(d):
    """Pillar 2: Profitability & Margins — 0 to 25 pts"""
    pts = 0.0

    # Gross margin (0-10): pricing power signal
    gm = d.get("gross_m")
    if gm is not None:
        if   gm >= 65: pts += 10
        elif gm >= 55: pts += 8
        elif gm >= 45: pts += 6
        elif gm >= 35: pts += 4
        elif gm >= 25: pts += 2
        elif gm >= 15: pts += 1
        # < 15% → 0 pts

    # Operating margin (0-10): operating leverage + efficiency
    om = d.get("op_mar")
    if om is not None:
        if   om >= 25: pts += 10
        elif om >= 20: pts += 8
        elif om >= 15: pts += 6
        elif om >= 10: pts += 4
        elif om >= 5:  pts += 2
        elif om > 0:   pts += 1
        # < 0 → 0 pts

    # FCF margin (0-5): cash conversion quality
    fcfm = d.get("fcf_margin")
    if fcfm is not None:
        fcfm_pct = fcfm * 100.0
        if   fcfm_pct >= 20: pts += 5
        elif fcfm_pct >= 15: pts += 4
        elif fcfm_pct >= 10: pts += 3
        elif fcfm_pct >= 5:  pts += 2
        elif fcfm_pct > 0:   pts += 1

    return round(min(pts, 25), 1)


def _score_balance_sheet(d):
    """Pillar 3: Balance Sheet Health — 0 to 25 pts"""
    pts = 0.0

    # Net debt / EBITDA (0-15): leverage quality
    nde = d.get("net_debt_ebitda")
    if nde is not None:
        if   nde < 0:   pts += 15   # net cash
        elif nde <= 0.5: pts += 13
        elif nde <= 1.0: pts += 11
        elif nde <= 1.5: pts += 9
        elif nde <= 2.0: pts += 7
        elif nde <= 2.5: pts += 5
        elif nde <= 3.0: pts += 3
        elif nde <= 4.0: pts += 1
        # > 4x → 0 pts
    else:
        # fallback: use D/E if EBITDA not available
        de = d.get("de")
        if de is not None:
            if   de <= 0:   pts += 12   # net cash (negative D/E from net cash)
            elif de <= 0.3: pts += 10
            elif de <= 0.6: pts += 7
            elif de <= 1.0: pts += 5
            elif de <= 1.5: pts += 3
            elif de <= 2.5: pts += 1

    # D/E ratio (0-10): complementary leverage measure
    de = d.get("de")
    if de is not None:
        if   de <= 0:   pts += 10   # net cash
        elif de <= 0.2: pts += 9
        elif de <= 0.5: pts += 7
        elif de <= 0.8: pts += 5
        elif de <= 1.2: pts += 3
        elif de <= 2.0: pts += 1
        # > 2.0 → 0 pts
    else:
        pts += 4  # neutral assumption if missing

    return round(min(pts, 25), 1)


def _score_capital_efficiency(d):
    """Pillar 4: Capital Efficiency — 0 to 15 pts"""
    pts = 0.0

    # FCF conversion: FCF / NI (0-8) — cash earnings quality
    fcf_ni = d.get("fcf_ni")
    if fcf_ni is not None:
        if   fcf_ni >= 1.2: pts += 8
        elif fcf_ni >= 1.0: pts += 6
        elif fcf_ni >= 0.8: pts += 4
        elif fcf_ni >= 0.6: pts += 2
        elif fcf_ni >= 0.4: pts += 1
        # < 0.4 → 0 pts (accrual-heavy earnings)
    else:
        pts += 3  # neutral assumption

    # Revenue growth (0-7) — efficient capital deployed → growth
    rg = d.get("rev_g")
    if rg is not None:
        if   rg >= 25: pts += 7
        elif rg >= 20: pts += 6
        elif rg >= 15: pts += 5
        elif rg >= 10: pts += 4
        elif rg >= 5:  pts += 3
        elif rg >= 0:  pts += 2
        else:          pts += 0  # shrinking revenue

    return round(min(pts, 15), 1)


def score_stock(d):
    """
    Apply 4-pillar quality scoring.
    Returns None if stock fails minimum quality gate.
    Quality gate: ROIC > 0 AND operating_margin > 0.
    """
    # Minimum quality gate
    roic  = d.get("roic")
    op_mar= d.get("op_mar")
    if roic is not None and roic <= 0:
        return None    # negative return on capital
    if op_mar is not None and op_mar <= 0:
        return None    # operating losses

    p1 = _score_return_on_capital(d)
    p2 = _score_profitability(d)
    p3 = _score_balance_sheet(d)
    p4 = _score_capital_efficiency(d)
    total = p1 + p2 + p3 + p4

    if   total >= 90: grade = "A+"
    elif total >= 80: grade = "A"
    elif total >= 70: grade = "B+"
    elif total >= 60: grade = "B"
    elif total >= 50: grade = "C+"
    elif total >= 40: grade = "C"
    else:             grade = "D"

    return dict(
        ticker=d["ticker"], sector=d["sector"], price=d["price"],
        mktcap=d["mktcap"],
        quality_score=round(total, 1),
        grade=grade,
        score_cap=p1,   # return on capital
        score_prof=p2,  # profitability
        score_bs=p3,    # balance sheet
        score_eff=p4,   # capital efficiency
        roic=d.get("roic"), roe=d.get("roe"),
        op_mar=d.get("op_mar"), gross_m=d.get("gross_m"),
        fcf_margin=d.get("fcf_margin"), fcf_ni=d.get("fcf_ni"),
        net_debt_ebitda=d.get("net_debt_ebitda"),
        de=d.get("de"), pe=d.get("pe"), p_b=d.get("p_b"),
        beta=d.get("beta"), div_yld=d.get("div_yld"),
        rev_g=d.get("rev_g"), eps_g=d.get("eps_g"),
        perf_1m=d.get("perf_1m"), perf_3m=d.get("perf_3m"),
        pos52=d.get("pos52"),
        ev_eb=d.get("ev_eb"), ev_live=d.get("ev_live"),
        eps_fwd=d.get("eps_fwd"), rev_fwd=d.get("rev_fwd"),
    )


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
def _fpct(n):
    return format(n * 100, ".1f") + "%" if n is not None else "—"
def _fx(n, dec=1):
    return format(n, f".{dec}f") + "x" if n is not None else "—"


def _pct_color(val, lo=0, hi=None, reverse=False):
    if val is None: return "#6b7194"
    if hi is None: hi = lo + 20
    ratio = (val - lo) / max(hi - lo, 0.001)
    ratio = max(0.0, min(1.0, ratio))
    if reverse: ratio = 1.0 - ratio
    if   ratio >= 0.75: return "#00c896"
    elif ratio >= 0.50: return "#4fd4a0"
    elif ratio >= 0.25: return "#f0b429"
    else:               return "#fc8181"


_GRADE_COLORS = {
    "A+": "#00e5a0", "A": "#00c896", "B+": "#4fd4a0",
    "B": "#68d391",  "C+": "#f0b429","C": "#fc8181",
    "D": "#e05c5c",
}

_PILLAR_COLORS = {
    "cap":  "#4f8ef7",  # blue
    "prof": "#00c896",  # green
    "bs":   "#a78bfa",  # purple
    "eff":  "#f0b429",  # yellow
}


def _score_bar(pts, max_pts, color):
    pct = (pts / max_pts) * 100 if max_pts > 0 else 0
    return (f'<div style="display:flex;align-items:center;gap:5px;margin:1px 0">'
            f'<div style="width:60px;height:5px;background:#252a3a;border-radius:3px;overflow:hidden">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:3px"></div>'
            f'</div>'
            f'<span style="color:{color};font-size:10px;font-family:monospace">{pts:.0f}/{max_pts}</span>'
            f'</div>')


# ── CSV EXPORT ────────────────────────────────────────────────────────────────
def write_csv(results, filename):
    headers = [
        "Quality Score", "Grade", "Rank",
        "Ticker", "Sector", "Price", "Market Cap",
        "P1 Return on Capital", "P2 Profitability", "P3 Balance Sheet", "P4 Capital Efficiency",
        "ROIC %", "ROE %", "Gross Margin %", "Op Margin %", "FCF Margin %",
        "FCF/NI", "Net Debt/EBITDA", "D/E",
        "P/E", "P/B", "EV/EBITDA",
        "Rev Growth %", "EPS Growth %",
        "Perf 1M %", "Perf 3M %",
        "Beta", "Div Yield %",
    ]
    sorted_r = sorted(results, key=lambda x: -x.get("quality_score", 0))
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for rank, r in enumerate(sorted_r, 1):
            def pn(v, mult=1, dec=2):
                return round(v * mult, dec) if v is not None else ""
            nde = r.get("net_debt_ebitda")
            nde_str = "Net Cash" if nde is not None and nde < 0 else pn(nde, dec=1)
            w.writerow([
                pn(r.get("quality_score"), dec=1), r.get("grade",""), rank,
                r["ticker"], r["sector"],
                pn(r["price"]), pn(r["mktcap"]),
                pn(r.get("score_cap"), dec=1), pn(r.get("score_prof"), dec=1),
                pn(r.get("score_bs"), dec=1),  pn(r.get("score_eff"), dec=1),
                pn(r.get("roic"),    dec=1) if r.get("roic")    is not None else "",
                pn(r.get("roe"),     dec=1) if r.get("roe")     is not None else "",
                pn(r.get("gross_m"), dec=1) if r.get("gross_m") is not None else "",
                pn(r.get("op_mar"),  dec=1) if r.get("op_mar")  is not None else "",
                pn(r.get("fcf_margin"), 100, 1) if r.get("fcf_margin") is not None else "",
                pn(r.get("fcf_ni"),  dec=2) if r.get("fcf_ni")  is not None else "",
                nde_str,
                pn(r.get("de"),   dec=2) if r.get("de")   is not None else "",
                pn(r.get("pe"),   dec=1) if r.get("pe")   else "",
                pn(r.get("p_b"),  dec=2) if r.get("p_b")  is not None else "",
                pn(r.get("ev_eb"),dec=1) if r.get("ev_eb") is not None else "",
                pn(r.get("rev_g"),dec=1) if r.get("rev_g") is not None else "",
                pn(r.get("eps_g"),dec=1) if r.get("eps_g") is not None else "",
                pn(r.get("perf_1m"),dec=1) if r.get("perf_1m") is not None else "",
                pn(r.get("perf_3m"),dec=1) if r.get("perf_3m") is not None else "",
                pn(r.get("beta"), dec=2),
                pn(r.get("div_yld"), dec=2) if r.get("div_yld") else "",
            ])


# ── DEEP DIVE / WATCHLIST BAR ─────────────────────────────────────────────────
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
def build_html(results, ts, total_in, index_label, failed_gate=0, suite_port=5050):
    n = len(results)
    a_plus = sum(1 for r in results if r.get("grade") == "A+")
    a_gr   = sum(1 for r in results if r.get("grade") in ("A+","A"))
    b_plus = sum(1 for r in results if r.get("grade") in ("A+","A","B+","B"))
    top5   = results[:5]

    top5_html = " ".join([
        f'<span style="display:inline-flex;align-items:center;gap:6px;background:#1a2035;'
        f'border:1px solid #252a3a;border-radius:6px;padding:6px 12px;margin:3px;">'
        f'<span style="color:#00e5a0;font-weight:700;font-size:13px">#{i+1}</span>'
        f'<span style="color:#e8eaf0;font-weight:600">{r["ticker"]}</span>'
        f'<span style="background:{_GRADE_COLORS.get(r["grade"],"#6b7194")}20;'
        f'color:{_GRADE_COLORS.get(r["grade"],"#6b7194")};padding:1px 5px;border-radius:3px;font-size:11px">'
        f'{r["grade"]}</span>'
        f'<span style="color:#6b7194;font-size:11px">{r.get("quality_score",0):.0f}/100</span>'
        f'</span>'
        for i, r in enumerate(top5)
    ])

    rows_html = []
    for rank, r in enumerate(results, 1):
        grade  = r.get("grade", "D")
        gc     = _GRADE_COLORS.get(grade, "#e05c5c")
        qs     = r.get("quality_score", 0)
        sc     = "#00c896" if qs >= 70 else "#f0b429" if qs >= 50 else "#e05c5c"
        roic_c = _pct_color(r.get("roic"),  lo=0, hi=25)
        gm_c   = _pct_color(r.get("gross_m"), lo=15, hi=65)
        om_c   = _pct_color(r.get("op_mar"),  lo=0,  hi=25)
        fcfm_v = (r.get("fcf_margin") or 0) * 100
        fcfm_c = _pct_color(fcfm_v if r.get("fcf_margin") else None, lo=0, hi=20)
        p1c    = "#00c896" if (r.get("perf_1m") or 0) >= 0 else "#e05c5c"
        p3c    = "#00c896" if (r.get("perf_3m") or 0) >= 0 else "#e05c5c"

        nde    = r.get("net_debt_ebitda")
        if nde is not None and nde < 0:
            nde_str = '<span style="color:#00c896;font-size:10px">Net Cash</span>'
        elif nde is not None:
            nde_color = _pct_color(nde, lo=0, hi=4, reverse=True)
            nde_str = f'<span style="color:{nde_color}">{_fx(nde)}</span>'
        else:
            nde_str = "—"

        # Pillar bars
        bar_cap  = _score_bar(r.get("score_cap",0),  35, "#4f8ef7")
        bar_prof = _score_bar(r.get("score_prof",0), 25, "#00c896")
        bar_bs   = _score_bar(r.get("score_bs",0),   25, "#a78bfa")
        bar_eff  = _score_bar(r.get("score_eff",0),  15, "#f0b429")

        # Score ring visual (CSS-only, 0-100)
        pct = int(qs)
        rows_html.append(f"""
        <tr data-grade="{grade}" data-score="{qs}">
          <td class="cb-cell"><input type="checkbox" class="row-check" value="{r['ticker']}"></td>
          <td class="mono" style="color:#6b7194;font-size:11px">{rank}</td>
          <td class="mono" style="color:#e8eaf0;font-weight:600">{r["ticker"]}</td>
          <td style="color:#9ca3c8;font-size:11px">{r["sector"]}</td>
          <td class="mono">{_fp(r["price"])}</td>
          <td class="mono">{_fb(r["mktcap"])}</td>
          <td>
            <div style="display:flex;align-items:center;gap:7px">
              <span class="mono" style="color:{sc};font-weight:700;font-size:13px">{qs:.0f}</span>
              <span style="background:{gc}20;color:{gc};padding:1px 6px;border-radius:4px;font-size:11px;font-weight:600">{grade}</span>
            </div>
          </td>
          <td>
            <div title="P1 Return on Capital: {r.get('score_cap',0):.0f}/35">{bar_cap}</div>
            <div title="P2 Profitability: {r.get('score_prof',0):.0f}/25">{bar_prof}</div>
            <div title="P3 Balance Sheet: {r.get('score_bs',0):.0f}/25">{bar_bs}</div>
            <div title="P4 Capital Efficiency: {r.get('score_eff',0):.0f}/15">{bar_eff}</div>
          </td>
          <td class="mono" style="color:{roic_c}">{_fpc(r.get("roic")) if r.get("roic") is not None else "—"}</td>
          <td class="mono" style="color:{_pct_color(r.get('roe'), lo=5, hi=30)}">{_fpc(r.get("roe")) if r.get("roe") is not None else "—"}</td>
          <td class="mono" style="color:{gm_c}">{_fpc(r.get("gross_m")) if r.get("gross_m") is not None else "—"}</td>
          <td class="mono" style="color:{om_c}">{_fpc(r.get("op_mar")) if r.get("op_mar") is not None else "—"}</td>
          <td class="mono" style="color:{fcfm_c}">{_fpct(r.get("fcf_margin")) if r.get("fcf_margin") is not None else "—"}</td>
          <td class="mono">{nde_str}</td>
          <td class="mono" style="color:{_pct_color(r.get('de'), lo=0, hi=2, reverse=True)}">{_fx(r.get("de")) if r.get("de") is not None else "—"}</td>
          <td class="mono">{_fx(r.get("pe")) if r.get("pe") else "—"}</td>
          <td class="mono">{_fpc(r.get("rev_g"), plus=True) if r.get("rev_g") is not None else "—"}</td>
          <td class="mono" style="color:{p1c}">{_fpc(r.get("perf_1m"), plus=True) if r.get("perf_1m") is not None else "—"}</td>
          <td class="mono" style="color:{p3c}">{_fpc(r.get("perf_3m"), plus=True) if r.get("perf_3m") is not None else "—"}</td>
        </tr>""")

    rows_str = "\n".join(rows_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quality / Compounder Screener — {index_label}</title>
<style>
  :root {{
    --bg:#0f111a; --bg2:#161927; --bg3:#1a1e2e; --card:#1e2538;
    --border:#252a3a; --text:#e8eaf0; --mu:#6b7194;
    --gr:#00c896; --rd:#e05c5c; --yw:#f0b429; --bl:#4f8ef7; --pu:#a78bfa;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:13px;line-height:1.5}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
  .header{{background:linear-gradient(135deg,#0f1824 0%,#0f111a 100%);border-bottom:1px solid var(--border);padding:20px 24px}}
  .header h1{{font-size:22px;font-weight:700;color:#fff;letter-spacing:-0.3px}}
  .header .sub{{color:var(--mu);font-size:12px;margin-top:4px}}
  .meta-grid{{display:flex;flex-wrap:wrap;gap:12px;margin:14px 16px 0}}
  .meta-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 18px;min-width:130px}}
  .meta-card .val{{font-size:20px;font-weight:700;color:#e8eaf0}}
  .meta-card .lbl{{font-size:11px;color:var(--mu);margin-top:2px}}
  .top5{{margin:10px 16px;padding:10px 14px;background:#161927;border:1px solid var(--border);border-radius:8px}}
  .top5 .t5-label{{color:var(--mu);font-size:11px;margin-bottom:6px}}
  .pillars{{margin:10px 16px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
  .pillar-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px}}
  .pillar-card .p-title{{font-size:11px;color:var(--mu);margin-bottom:4px}}
  .pillar-card .p-pts{{font-size:18px;font-weight:700}}
  .pillar-card .p-desc{{font-size:10px;color:var(--mu);margin-top:2px}}
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
  .methodology{{margin:12px 16px;padding:14px 18px;background:#161927;border:1px solid var(--border);
                border-left:3px solid var(--pu);border-radius:8px;font-size:12px;color:var(--mu)}}
  .methodology b{{color:var(--text)}}
  .legend{{margin:14px 16px;padding:12px 16px;background:var(--card);border:1px solid var(--border);
           border-radius:8px;font-size:11px;color:var(--mu)}}
  .sort-asc::after{{content:" ▲"}}
  .sort-desc::after{{content:" ▼"}}
  @media (max-width:900px){{.pillars{{grid-template-columns:repeat(2,1fr)}}}}
{_WL_CSS}
</style>
</head>
<body>

<div class="header">
  <h1>🏆 Quality / Compounder Screener — {index_label}</h1>
  <div class="sub">4-Pillar quality scoring: Return on Capital + Margins + Balance Sheet + Capital Efficiency · Generated {ts}</div>
</div>

<div class="meta-grid">
  <div class="meta-card"><div class="val">{total_in}</div><div class="lbl">Stocks Scanned</div></div>
  <div class="meta-card"><div class="val" style="color:var(--gr)">{n}</div><div class="lbl">Passed Quality Gate</div></div>
  <div class="meta-card"><div class="val" style="color:var(--yw)">{failed_gate}</div><div class="lbl">Failed Gate (neg ROIC/margin)</div></div>
  <div class="meta-card"><div class="val" style="color:#00e5a0">{a_plus}</div><div class="lbl">A+ Grade</div></div>
  <div class="meta-card"><div class="val" style="color:#00c896">{a_gr}</div><div class="lbl">A / A+ Grade</div></div>
  <div class="meta-card"><div class="val" style="color:#4fd4a0">{b_plus}</div><div class="lbl">B or Better</div></div>
</div>

<div class="pillars">
  <div class="pillar-card">
    <div class="p-title">PILLAR 1 · RETURN ON CAPITAL</div>
    <div class="p-pts" style="color:#4f8ef7">0 – 35 pts</div>
    <div class="p-desc">ROIC (primary, 0-20pts) + ROE (0-10pts) + consistency bonus (0-5pts)</div>
  </div>
  <div class="pillar-card">
    <div class="p-title">PILLAR 2 · PROFITABILITY</div>
    <div class="p-pts" style="color:#00c896">0 – 25 pts</div>
    <div class="p-desc">Gross margin (0-10pts) + operating margin (0-10pts) + FCF margin (0-5pts)</div>
  </div>
  <div class="pillar-card">
    <div class="p-title">PILLAR 3 · BALANCE SHEET</div>
    <div class="p-pts" style="color:#a78bfa">0 – 25 pts</div>
    <div class="p-desc">Net Debt/EBITDA (0-15pts) + D/E ratio (0-10pts)</div>
  </div>
  <div class="pillar-card">
    <div class="p-title">PILLAR 4 · CAPITAL EFFICIENCY</div>
    <div class="p-pts" style="color:#f0b429">0 – 15 pts</div>
    <div class="p-desc">FCF/NI conversion (0-8pts) + revenue growth (0-7pts)</div>
  </div>
</div>

<div class="methodology">
  <b>Methodology:</b> Each stock is scored on 4 pillars totaling 0–100 points.
  <b>Quality Gate:</b> stocks with negative ROIC or negative operating margin are excluded — they fail the minimum
  compounders definition.
  <b>Pillar bars</b> (shown in the Score column) show each pillar's contribution relative to its max.
  Blue = Return on Capital · Green = Profitability · Purple = Balance Sheet · Yellow = Capital Efficiency.
</div>

<div class="top5">
  <div class="t5-label">TOP 5 QUALITY COMPOUNDERS</div>
  {top5_html}
</div>

<div class="controls">
  <input class="search-box" id="search" placeholder="Filter ticker or sector…" oninput="filterTable()">
  <button class="filter-btn" id="btn-all"  onclick="setFilter('all')"  style="border-color:var(--bl);color:var(--bl)">All</button>
  <button class="filter-btn" id="btn-aplus" onclick="setFilter('aplus')">A+ Only</button>
  <button class="filter-btn" id="btn-a"    onclick="setFilter('a')">A / A+</button>
  <button class="filter-btn" id="btn-b"    onclick="setFilter('b')">B+ and Above</button>
  <span style="margin-left:auto;color:var(--mu);font-size:11px" id="row-count">{n} stocks</span>
</div>

<div class="tbl-wrap">
<table id="qs-table">
<thead>
<tr>
  <th class="cb-th"><input type="checkbox" id="cb-all" title="Select all visible"></th>
  <th onclick="sortTable(1)">#</th>
  <th onclick="sortTable(2)">Ticker</th>
  <th onclick="sortTable(3)">Sector</th>
  <th onclick="sortTable(4)">Price</th>
  <th onclick="sortTable(5)">Mkt Cap</th>
  <th onclick="sortTable(6)" title="Total quality score 0-100 + letter grade">Score / Grade</th>
  <th title="Pillar breakdown bars (hover for pts)">Pillars</th>
  <th onclick="sortTable(8)" title="Return on Invested Capital (TTM)">ROIC</th>
  <th onclick="sortTable(9)" title="Return on Equity (TTM)">ROE</th>
  <th onclick="sortTable(10)" title="Gross Margin (TTM) — pricing power">Gross Mgn</th>
  <th onclick="sortTable(11)" title="Operating Margin (TTM)">Op Mgn</th>
  <th onclick="sortTable(12)" title="FCF Margin (FCF / Revenue)">FCF Mgn</th>
  <th onclick="sortTable(13)" title="Net Debt / EBITDA — leverage quality">ND/EBITDA</th>
  <th onclick="sortTable(14)" title="Debt-to-Equity ratio">D/E</th>
  <th onclick="sortTable(15)">P/E</th>
  <th onclick="sortTable(16)" title="Revenue Growth YoY %">Rev Growth</th>
  <th onclick="sortTable(17)">Perf 1M</th>
  <th onclick="sortTable(18)">Perf 3M</th>
</tr>
</thead>
<tbody id="qs-tbody">
{rows_str}
</tbody>
</table>
</div>

<div class="legend">
  <b>Grade scale:</b>
  <span style="color:#00e5a0">A+</span> ≥90 ·
  <span style="color:#00c896">A</span> ≥80 ·
  <span style="color:#4fd4a0">B+</span> ≥70 ·
  <span style="color:#68d391">B</span> ≥60 ·
  <span style="color:#f0b429">C+</span> ≥50 ·
  <span style="color:#fc8181">C</span> ≥40 ·
  <span style="color:#e05c5c">D</span> &lt;40 ·
  <b>Quality Gate:</b> stocks with ROIC ≤ 0% or Op Margin ≤ 0% are excluded from results. ·
  <b>ND/EBITDA:</b> "Net Cash" = company holds more cash than debt (best possible outcome).
</div>

<script>
let _sortCol = -1, _sortAsc = true, _filt = 'all';

function sortTable(col) {{
  const tbody = document.getElementById('qs-tbody');
  const rows  = Array.from(tbody.rows);
  if (_sortCol === col) {{ _sortAsc = !_sortAsc; }}
  else {{ _sortCol = col; _sortAsc = (col !== 0 && col !== 5); }}
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

function setFilter(f) {{
  _filt = f;
  document.querySelectorAll('.filter-btn').forEach(b => {{
    b.style.borderColor=''; b.style.color='';
  }});
  const btn = document.getElementById('btn-' + f);
  if (btn) {{ btn.style.borderColor='var(--bl)'; btn.style.color='var(--bl)'; }}
  filterTable();
}}

function filterTable() {{
  const q    = document.getElementById('search').value.toLowerCase();
  const rows = document.querySelectorAll('#qs-tbody tr');
  let shown  = 0;
  rows.forEach(row => {{
    const ticker = row.cells[2]?.textContent?.toLowerCase() ?? '';
    const sector = row.cells[3]?.textContent?.toLowerCase() ?? '';
    const grade  = row.getAttribute('data-grade') ?? '';
    const score  = parseFloat(row.getAttribute('data-score') ?? '0');
    const matchQ = !q || ticker.includes(q) || sector.includes(q);
    const matchF = _filt === 'all'
      || (_filt === 'aplus' && grade === 'A+')
      || (_filt === 'a'     && (grade === 'A+' || grade === 'A'))
      || (_filt === 'b'     && (grade === 'A+' || grade === 'A' || grade === 'B+' || grade === 'B'));
    row.style.display = (matchQ && matchF) ? '' : 'none';
    if (matchQ && matchF) shown++;
  }});
  document.getElementById('row-count').textContent = shown + ' stocks';
}}
</script>
<script>{_wl_js(suite_port)}</script>
{_WL_BAR}
</body>
</html>"""


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--index",     default="SPX")
    ap.add_argument("--top",       type=int, default=None)
    ap.add_argument("--csv",       action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--help",      action="store_true")
    args = ap.parse_args()

    if args.help:
        print(__doc__); sys.exit(0)

    index_code = args.index.upper()
    if index_code not in INDEX_CONFIG:
        print(f"ERROR: Unknown index '{index_code}'. Valid: {', '.join(INDEX_CONFIG)}")
        sys.exit(1)

    print("=" * 60)
    print("  Quality / Compounder Screener")
    print("=" * 60)

    # ── Fetch
    print("\n[1/4] Fetching stocks from TradingView...")
    df = fetch_stocks(index_code, limit=args.top)
    total_in = len(df)
    print(f"  Total fetched: {total_in}")

    # ── Parse
    print("\n[2/4] Parsing rows...")
    parsed = []
    for _, row in df.iterrows():
        try:
            d = parse_row(row.to_dict())
            if d["ticker"]:
                parsed.append(d)
        except Exception:
            pass

    # ── Score
    print("\n[3/4] Scoring quality pillars...")
    results    = []
    failed_gate= 0
    for d in parsed:
        r = score_stock(d)
        if r is None:
            failed_gate += 1
        else:
            results.append(r)

    # Apply min-score filter
    if args.min_score > 0:
        before = len(results)
        results = [r for r in results if r["quality_score"] >= args.min_score]
        print(f"  Min-score filter ({args.min_score}): {before} → {len(results)} stocks")

    # Sort by quality score descending
    results.sort(key=lambda x: -x["quality_score"])
    print(f"  Scored: {len(results)} | Failed quality gate: {failed_gate}")

    # Print top 10
    print("\n  Top 10 Quality Compounders:")
    for i, r in enumerate(results[:10], 1):
        print(f"    {i:>2}. {r['ticker']:<6}  Score={r['quality_score']:>5.1f}  Grade={r['grade']}"
              f"  ROIC={r.get('roic') or 0:>5.1f}%  GM={r.get('gross_m') or 0:>5.1f}%  [{r['sector'][:18]}]")

    # Distribution
    dist = {"A+":0,"A":0,"B+":0,"B":0,"C+":0,"C":0,"D":0}
    for r in results:
        dist[r["grade"]] = dist.get(r["grade"], 0) + 1
    print("\n  Grade Distribution:")
    for g, cnt in dist.items():
        if cnt: print(f"    {g:<4} {cnt:>4}  {'█' * min(cnt, 40)}")

    # ── Build HTML
    print("\n[4/4] Building HTML report...")
    ts         = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    cfg        = INDEX_CONFIG[index_code]
    suite_port = int(os.environ.get("VALUATION_SUITE_PORT", "5050"))
    html = build_html(results, ts, total_in, cfg["name"],
                      failed_gate=failed_gate, suite_port=suite_port)

    index_slug = index_code.lower()
    date_str   = datetime.datetime.now().strftime("%Y_%m_%d")
    out_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qualityData")
    os.makedirs(out_dir, exist_ok=True)
    out     = os.path.join(out_dir, f"{date_str}_quality_screener_{index_slug}.html")
    out_csv = os.path.join(out_dir, f"{date_str}_quality_screener_{index_slug}.csv")

    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved: {out}")

    if args.csv:
        write_csv(results, out_csv)
        print(f"  CSV  saved: {out_csv}")

    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        webbrowser.open("file://" + os.path.abspath(out))
    print(f"\nDone. ({len(results)} stocks scored)")
    print("=" * 60)


if __name__ == "__main__":
    main()
